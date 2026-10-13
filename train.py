#!/usr/bin/env python3
"""Minimal full-field training. The smoke model is only a pipeline check."""

import argparse
import json
from pathlib import Path
import time

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
import yaml

from emulator.data.dataset import FullFieldDataset
from emulator.data.loaders import build_loaders, loader_options
from emulator.data.mesh import load_mesh
from emulator.data.normalization import compute_normalization, prepare_normalization
from emulator.data.splits import split_years
from emulator.models.smoke import SmokeModel
from emulator.training.checkpoints import (
    load_resume, load_selected_checkpoint, prepare_output_directory,
    restore_training_state, save_checkpoint, verify_reload_metrics,
)
from emulator.training.config import load_config
from emulator.training.engine import build_optimizer_scheduler, evaluate, train_one_epoch
from emulator.training.losses import load_spatial_weights
from emulator.training.metrics import training_metrics
from emulator.training.runtime import (
    cleanup_runtime, finish_runtime, initialize_runtime, start_memory_monitor,
)
from emulator.training.smoke import benchmark_loading


def write_run_metadata(output_dir, config, directory, available, splits, datasets, sample_counts):
    """Record the resolved configuration and simulation-year/sample identity."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    manifest = {"source": config["data"]["source"], "available_years": available,
                "split_years": splits, "sample_counts": sample_counts,
                "monthly_sample_counts": {
                    name: {str(path.relative_to(directory)): sum(m == i for m, _ in ds.index)
                           for i, path in enumerate(ds.files)} for name, ds in datasets.items()}}
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))


def run(config, smoke_test):
    """Coordinate one configured experiment using the focused package modules."""
    train, evaluation = config["training"], config["evaluation"]
    seed = config["experiment"]["seed"]
    device, world, rank, use_amp = initialize_runtime(config)
    memory_stop, memory_thread, memory_report = start_memory_monitor()
    start = time.perf_counter()
    directory = Path(config["data"]["root"]) / config["data"]["source"]
    available = sorted(int(p.name) for p in directory.iterdir() if p.is_dir() and p.name.isdigit())
    splits = split_years(available, config["split"], seed)
    if not splits["val"]:
        raise ValueError("No real VAL year exists; checkpoint selection requires a validation split")
    mesh_cpu = load_mesh(directory / "mesh.pt")
    node_id = mesh_cpu["node_id"]
    datasets = {}
    for name, years in splits.items():
        if years:
            print(f"Indexing {name}: {len(years)} simulation years", flush=True)
            datasets[name] = FullFieldDataset(directory, years, mesh_cpu, config["data"]["history_hours"])
            print(f"Indexed {name}: {len(datasets[name])} samples", flush=True)
    runtime = {"device": str(device), "cuda_available": torch.cuda.is_available(), "world_size": world,
               "dataset_initialization_seconds": time.perf_counter() - start,
               "available_years": available, "year_counts": {k: len(v) for k, v in splits.items()},
               "sample_counts": {k: len(datasets[k]) if k in datasets else 0 for k in splits},
               "mesh_nodes": len(node_id), "mesh_directed_edges": mesh_cpu["edge_index"].shape[1]}
    print(json.dumps(runtime), flush=True)
    resume = load_resume(config, node_id, splits, smoke_test)
    output_dir = prepare_output_directory(config, resume, smoke_test, world, rank)
    if rank == 0:
        write_run_metadata(output_dir, config, directory, available, splits, datasets, runtime["sample_counts"])

    stats_start = time.perf_counter()
    normalization = resume["normalization"] if resume else (compute_normalization(datasets["train"], config["dataloader"]["num_workers"]) if rank == 0 else None)
    if world > 1:
        payload = [normalization]
        dist.broadcast_object_list(payload, src=0)
        normalization = payload[0]
    runtime["normalization_seconds"] = time.perf_counter() - stats_start
    if rank == 0:
        torch.save(normalization, output_dir / "normalization.pt")
    norm_device = prepare_normalization(normalization, device)
    weights = load_spatial_weights(mesh_cpu, config["spatial_weighting"]).to(device)
    mesh = {k: v.to(device) for k, v in mesh_cpu.items()}
    options = loader_options(config)
    start = time.perf_counter()
    train_loader, eval_loaders, sampler, generator = build_loaders(
        datasets, train, options, seed, world, rank, smoke_test)
    runtime["dataloader_initialization_seconds"] = time.perf_counter() - start
    if smoke_test:
        benchmark_loading(datasets["train"], options, seed, runtime)
    raw_model = SmokeModel().to(device)
    optimizer, scheduler = build_optimizer_scheduler(raw_model, train)
    start_epoch, best_rmse, best_epoch = restore_training_state(
        resume, raw_model, optimizer, scheduler, output_dir)
    model = DistributedDataParallel(raw_model, device_ids=[device.index] if device.type == "cuda" else None) if world > 1 else raw_model
    last_epoch = start_epoch + 1 if smoke_test else train["epochs"]
    runtime.update(forward_seconds=[], backward_seconds=[], optimizer_seconds=[], batch_loading_seconds=[], training_batch_sizes=[])
    for epoch in range(start_epoch, last_epoch):
        total, lr_used = train_one_epoch(
            model, raw_model, train_loader, sampler, generator, optimizer, scheduler,
            mesh, norm_device, weights, device, use_amp, train, seed, epoch, world,
            smoke_test, runtime)
        if smoke_test:
            # Release the persistent training workers before held-out evaluation.
            del train_loader
        val_metrics = None
        improved = False
        if (epoch + 1) % evaluation["val_every_epochs"] == 0 or epoch + 1 == last_epoch:
            val_metrics, val_runtime = evaluate(raw_model, eval_loaders["val"], mesh, norm_device, weights, device, use_amp)
            runtime["validation"] = val_runtime
            improved = val_metrics["all_rmse"] < best_rmse
            if improved:
                best_rmse, best_epoch = val_metrics["all_rmse"], epoch
        if rank == 0:
            save_checkpoint(output_dir, raw_model, optimizer, scheduler, epoch=epoch,
                            config=config, normalization=normalization, splits=splits,
                            node_id=node_id, val_metrics=val_metrics, best_rmse=best_rmse,
                            best_epoch=best_epoch, smoke_test=smoke_test, improved=improved)
            row = training_metrics(epoch, lr_used, total, val_metrics, smoke_test)
            with (output_dir / "training_metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
        if world > 1:
            dist.barrier()

    selected = load_selected_checkpoint(output_dir, raw_model)
    reloaded_val, _ = evaluate(raw_model, eval_loaders["val"], mesh, norm_device, weights, device, use_amp)
    verify_reload_metrics(reloaded_val, selected)
    runtime["checkpoint_reload_succeeded"] = True
    test_metrics = None
    if evaluation["test_at_end"] and "test" in eval_loaders:
        test_metrics, runtime["test"] = evaluate(raw_model, eval_loaders["test"], mesh, norm_device, weights, device, use_amp)
    finish_runtime(runtime, device, memory_stop, memory_thread, memory_report)
    summary = {"smoke_test": smoke_test,
               "evaluation_scope": "seeded subset of at most two batches per held-out split" if smoke_test else "complete split",
               "implemented_metrics": ["all_rmse", "all_mae"], "selected_checkpoint": "best_overall.pt",
               "selected_epoch": selected["epoch"], "validation": reloaded_val, "test": test_metrics,
               "test_status": "evaluated" if test_metrics else ("no real TEST split" if "test" not in datasets else "disabled"),
               "resume_succeeded": resume is not None, "runtime": runtime}
    if rank == 0:
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
        print(f"Results: {output_dir}", flush=True)
        print(json.dumps({k: v for k, v in summary.items() if k != "runtime"}), flush=True)
    cleanup_runtime()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ncep.yaml")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Full TRAIN statistics, four real training batches, limited VAL/TEST")
    args = parser.parse_args()
    run(load_config(args.config), args.smoke_test)


if __name__ == "__main__":
    main()
