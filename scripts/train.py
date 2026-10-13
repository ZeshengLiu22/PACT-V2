#!/usr/bin/env python3
"""Minimal full-field training. The smoke model is only a pipeline check."""

import argparse
import json
import os
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
    load_resume,
    load_selected_checkpoint,
    prepare_output_directory,
    restore_training_state,
    save_checkpoint,
    verify_reload_metrics,
)
from emulator.training.config import load_config
from emulator.training.engine import build_optimizer_scheduler, evaluate, train_one_epoch
from emulator.training.logging import (
    close_logging,
    configure_logging,
    log_epoch,
    log_final,
    log_initialization,
    log_model_parameters,
    log_resume,
    logger,
)
from emulator.training.losses import load_spatial_weights
from emulator.training.metrics import training_metrics
from emulator.training.runtime import (
    cleanup_runtime,
    finish_runtime,
    initialize_runtime,
    start_memory_monitor,
)
from emulator.training.smoke import benchmark_loading


def write_run_metadata(
    output_dir, config, directory, available, splits, datasets, sample_counts
):
    """Save configuration and split membership, including counts per monthly file.

    Monthly paths are relative to the source directory so the manifest exposes
    simulation-year membership even when a month's calendar year differs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    manifest = {
        "source": config["data"]["source"],
        "available_years": available,
        "split_years": splits,
        "sample_counts": sample_counts,
        "monthly_sample_counts": {
            name: {
                str(path.relative_to(directory)): sum(m == i for m, _ in ds.index)
                for i, path in enumerate(ds.files)
            }
            for name, ds in datasets.items()
        },
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))


def run(config, smoke_test):
    """Run a validated configuration and write its checkpoints and final metrics.

    Smoke mode keeps full TRAIN normalization but limits training and held-out
    evaluation. Only rank zero writes artifacts; all ranks participate in DDP.
    """
    wall_start = time.perf_counter()
    logger.info(
        "Starting experiment: %s | Source: %s | Model: %s",
        config["experiment"]["name"], config["data"]["source"], config["model"]["name"],
    )

    # Initialize the runtime before indexing so memory measurements include data setup.
    train, evaluation = config["training"], config["evaluation"]
    seed = config["experiment"]["seed"]
    device, world, rank, use_amp = initialize_runtime(config)
    memory_stop, memory_thread, memory_report = start_memory_monitor()

    # Directory years define independent simulation groups, including cross-year months.
    start = time.perf_counter()
    directory = Path(config["data"]["root"]) / config["data"]["source"]
    available = sorted(
        int(p.name) for p in directory.iterdir() if p.is_dir() and p.name.isdigit()
    )
    splits = split_years(available, config["split"], seed)
    if not splits["val"]:
        raise ValueError(
            "No real VAL year exists; checkpoint selection requires a validation split"
        )

    mesh_cpu = load_mesh(directory / "mesh.pt")
    node_id = mesh_cpu["node_id"]
    datasets = {}
    for name, years in splits.items():
        if years:
            logger.info("Indexing %s: %d simulation years", name.upper(), len(years))
            indexing_start = time.perf_counter()
            datasets[name] = FullFieldDataset(
                directory, years, mesh_cpu, config["data"]["history_hours"]
            )
            logger.info(
                "Indexed %s: %d samples | Time: %.2fs",
                name.upper(), len(datasets[name]), time.perf_counter() - indexing_start,
            )

    runtime = {
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "world_size": world,
        "dataset_initialization_seconds": time.perf_counter() - start,
        "available_years": available,
        "year_counts": {k: len(v) for k, v in splits.items()},
        "sample_counts": {k: len(datasets[k]) if k in datasets else 0 for k in splits},
        "mesh_nodes": len(node_id),
        "mesh_directed_edges": mesh_cpu["edge_index"].shape[1],
    }
    logger.info(
        "Dataset initialization complete: %.2fs | Mesh: %d nodes, %d directed edges",
        runtime["dataset_initialization_seconds"], runtime["mesh_nodes"],
        runtime["mesh_directed_edges"],
    )

    # Resume identity checks precede writes to an existing experiment directory.
    resume = load_resume(config, node_id, splits, smoke_test)
    output_dir = prepare_output_directory(config, resume, smoke_test, world, rank)
    if rank == 0:
        write_run_metadata(
            output_dir, config, directory, available, splits, datasets,
            runtime["sample_counts"],
        )

    # File logging starts only after the same resume/path checks that guard artifacts.
    configure_logging(rank=rank, output_dir=output_dir, resume=resume is not None)
    log_initialization(
        config, runtime, splits, output_dir, use_amp,
        datasets["train"][0]["surge"].shape[-1], smoke_test,
    )

    # Fit statistics only on TRAIN, once on rank zero, or reuse the saved statistics.
    if resume:
        logger.info("TRAIN normalization: reusing saved statistics from %s", train["resume_from"])
    else:
        logger.info(
            "TRAIN normalization started: %d months, %d samples (full TRAIN population)",
            len(datasets["train"].files), len(datasets["train"]),
        )
    stats_start = time.perf_counter()
    normalization = (
        resume["normalization"]
        if resume
        else (
            compute_normalization(datasets["train"], config["dataloader"]["num_workers"])
            if rank == 0
            else None
        )
    )
    if world > 1:
        payload = [normalization]
        dist.broadcast_object_list(payload, src=0)
        normalization = payload[0]
    runtime["normalization_seconds"] = time.perf_counter() - stats_start
    logger.info(
        "TRAIN normalization %s | Time: %.2fs | Forcing observations/channel: %d | WW targets: %d",
        "reused" if resume else "complete", runtime["normalization_seconds"],
        normalization["forcing_count_per_channel"], normalization["surge_count"],
    )
    if rank == 0:
        torch.save(normalization, output_dir / "normalization.pt")

    norm_device = prepare_normalization(normalization, device)
    weights = load_spatial_weights(mesh_cpu, config["spatial_weighting"]).to(device)
    # One shared mesh stays on the device; dynamic batches carry no mesh copies.
    mesh = {k: v.to(device) for k, v in mesh_cpu.items()}

    # Keep loader setup and its standalone benchmark separate from training timings.
    logger.info("DataLoader initialization started")
    options = loader_options(config)
    start = time.perf_counter()
    train_loader, eval_loaders, sampler, generator = build_loaders(
        datasets, train, options, seed, world, rank, smoke_test
    )
    runtime["dataloader_initialization_seconds"] = time.perf_counter() - start
    logger.info(
        "DataLoader initialization complete | Time: %.3fs | Workers: %d | Batch size: %d",
        runtime["dataloader_initialization_seconds"], config["dataloader"]["num_workers"],
        train["batch_size"],
    )
    if smoke_test:
        benchmark_loading(datasets["train"], options, seed, runtime)

    # Construct and restore the raw model before wrapping it for distributed training.
    logger.info("Training initialization started")
    raw_model = SmokeModel().to(device)
    log_model_parameters(raw_model)
    optimizer, scheduler = build_optimizer_scheduler(raw_model, train)
    start_epoch, best_rmse, best_epoch = restore_training_state(
        resume, raw_model, optimizer, scheduler, output_dir
    )
    if resume:
        log_resume(resume, train["resume_from"])
    model = (
        DistributedDataParallel(
            raw_model, device_ids=[device.index] if device.type == "cuda" else None
        )
        if world > 1
        else raw_model
    )
    last_epoch = start_epoch + 1 if smoke_test else train["epochs"]
    runtime.update(
        forward_seconds=[],
        backward_seconds=[],
        optimizer_seconds=[],
        batch_loading_seconds=[],
        training_batch_sizes=[],
    )

    logger.info(
        "Training initialization complete | Starting epoch: %03d | Epochs this invocation: %d",
        start_epoch + 1, max(0, last_epoch - start_epoch),
    )
    training_start = time.perf_counter()

    # Select checkpoints exclusively by VAL AllRMSE; TEST never affects training.
    for epoch in range(start_epoch, last_epoch):
        epoch_start = time.perf_counter()
        total, lr_used = train_one_epoch(
            model, raw_model, train_loader, sampler, generator, optimizer, scheduler,
            mesh, norm_device, weights, device, use_amp, train, seed, epoch, world,
            smoke_test, runtime,
        )
        if smoke_test:
            # Release the persistent training workers before held-out evaluation.
            del train_loader

        val_metrics = None
        improved = False
        if (epoch + 1) % evaluation["val_every_epochs"] == 0 or epoch + 1 == last_epoch:
            logger.info("Epoch %03d/%03d | VAL evaluation started", epoch + 1, train["epochs"])
            # Uneven evaluation partitions use the raw model to avoid DDP collectives.
            val_metrics, val_runtime = evaluate(
                raw_model, eval_loaders["val"], mesh, norm_device, weights, device,
                use_amp,
            )
            runtime["validation"] = val_runtime
            improved = val_metrics["all_rmse"] < best_rmse
            if improved:
                best_rmse, best_epoch = val_metrics["all_rmse"], epoch

        if rank == 0:
            save_checkpoint(
                output_dir, raw_model, optimizer, scheduler,
                epoch=epoch,
                config=config,
                normalization=normalization,
                splits=splits,
                node_id=node_id,
                val_metrics=val_metrics,
                best_rmse=best_rmse,
                best_epoch=best_epoch,
                smoke_test=smoke_test,
                improved=improved,
            )
            row = training_metrics(epoch, lr_used, total, val_metrics, smoke_test)
            with (output_dir / "training_metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            log_epoch(row, train["epochs"], time.perf_counter() - epoch_start)
        if world > 1:
            dist.barrier()

    training_seconds = time.perf_counter() - training_start

    # Reload the selected checkpoint and verify VAL before using it for final TEST.
    logger.info("Reloading selected checkpoint: %s", output_dir / "best_overall.pt")
    selected = load_selected_checkpoint(output_dir, raw_model)
    reloaded_val, _ = evaluate(
        raw_model, eval_loaders["val"], mesh, norm_device, weights, device, use_amp
    )
    verify_reload_metrics(reloaded_val, selected)
    runtime["checkpoint_reload_succeeded"] = True
    logger.info("Selected checkpoint VAL metrics verified after reload")

    test_metrics = None
    if evaluation["test_at_end"] and "test" in eval_loaders:
        logger.info("Final TEST evaluation started")
        test_metrics, runtime["test"] = evaluate(
            raw_model, eval_loaders["test"], mesh, norm_device, weights, device, use_amp
        )
    finish_runtime(runtime, device, memory_stop, memory_thread, memory_report)

    # Report the actual evaluation scope, including an absent or disabled TEST split.
    summary = {
        "smoke_test": smoke_test,
        "evaluation_scope": (
            "seeded subset of at most two batches per held-out split"
            if smoke_test else "complete split"
        ),
        "implemented_metrics": ["all_rmse", "all_mae"],
        "selected_checkpoint": "best_overall.pt",
        "selected_epoch": selected["epoch"],
        "validation": reloaded_val,
        "test": test_metrics,
        "test_status": (
            "evaluated"
            if test_metrics
            else ("no real TEST split" if "test" not in datasets else "disabled")
        ),
        "resume_succeeded": resume is not None,
        "runtime": runtime,
    }
    if rank == 0:
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False)
        )
        log_final(summary, output_dir, time.perf_counter() - wall_start, training_seconds)
    cleanup_runtime()


def main():
    """Parse the public CLI, validate its YAML configuration, and run the experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ncep.yaml")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Full TRAIN statistics, four real training batches, limited VAL/TEST",
    )
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", 0))
    configure_logging(rank=rank)
    try:
        run(load_config(args.config), args.smoke_test)
    except Exception:
        # Keep the traceback in the active log and preserve the failing exit status.
        logger.exception("Training failed on rank %d", rank)
        raise
    finally:
        close_logging()


if __name__ == "__main__":
    main()
