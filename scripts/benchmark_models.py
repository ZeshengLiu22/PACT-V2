#!/usr/bin/env python3
"""One-batch real NCEP diagnostics through the existing trainer; no experiments.

Reuse saved full-TRAIN normalization. Shorter histories reuse those statistics
for shape/runtime diagnostics only, never for scientific training or metrics.
Run from the repository root with ``python -m scripts.benchmark_models``.
"""

import argparse
from copy import deepcopy
import gc
import json
from pathlib import Path
import resource
import time

import torch
from torch.utils.data import default_collate
import yaml

from emulator.data.dataset import FullFieldDataset
from emulator.data.mesh import load_mesh
from emulator.data.normalization import prepare_normalization
from emulator.models import BaseModel, IGARSS27Model
from emulator.training.engine import build_optimizer_scheduler, train_one_epoch
from emulator.training.losses import load_spatial_weights


def benchmark_case(model, batch, mesh, normalization, weights, device, train, use_amp, profile_stages=False):
    """Time a single real trainer update, including finite-gradient validation."""
    optimizer, scheduler = build_optimizer_scheduler(model, train)
    runtime = {name: [] for name in (
        "forward_seconds", "backward_seconds", "optimizer_seconds",
        "batch_loading_seconds", "training_batch_sizes",
    )}
    result = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameter_tensors": sum(p.requires_grad for p in model.parameters()),
    }
    ticks = {}

    def begin_forward(module, inputs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ticks["forward"] = time.perf_counter()

    def end_forward(module, inputs, outputs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result["model_forward_seconds"] = time.perf_counter() - ticks["forward"]
        output = outputs["surge"]
        result["output_shape"] = list(output.shape)
        result["finite_output"] = bool(torch.isfinite(output).all())
        expected = (len(batch["forcing"]), len(mesh["node_id"]), 6)
        if tuple(output.shape) != expected or not result["finite_output"]:
            raise AssertionError(f"Invalid surge output: {tuple(output.shape)}, expected {expected}")

    def begin_step(active_optimizer, args, kwargs):
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        result["parameters_with_gradients"] = len(gradients)
        result["finite_gradients"] = len(gradients) == result["trainable_parameter_tensors"] and all(
            bool(torch.isfinite(gradient).all()) for gradient in gradients
        )
        if not result["finite_gradients"]:
            raise AssertionError("Missing or nonfinite gradients")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        ticks["optimizer"] = time.perf_counter()

    def end_step(active_optimizer, args, kwargs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        result["adam_step_seconds"] = time.perf_counter() - ticks["optimizer"]

    handles = [
        model.register_forward_pre_hook(begin_forward),
        model.register_forward_hook(end_forward),
        optimizer.register_step_pre_hook(begin_step),
        optimizer.register_step_post_hook(end_step),
    ]
    if profile_stages:
        stages = {
            getattr(model, name): name for name in (
                "temporal", "temporal_encoder", "spatial_encoder", "projector",
                "mesh_encoder", "scale_score", "gnn", "decoder", "head", "surge_head",
            ) if hasattr(model, name)
        }
        result["stage_forward_seconds"] = {}

        def begin_stage(module, inputs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            ticks[module] = time.perf_counter()

        def end_stage(module, inputs, outputs):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - ticks[module]
            name = stages[module]
            result["stage_forward_seconds"][name] = result["stage_forward_seconds"].get(name, 0.0) + elapsed

        for module in stages:
            handles.extend((module.register_forward_pre_hook(begin_stage), module.register_forward_hook(end_stage)))
    try:
        totals, _ = train_one_epoch(
            model, model, [batch], None, torch.Generator(), optimizer, scheduler,
            mesh, normalization, weights, device, use_amp, train,
            seed=42, epoch=0, world=1, smoke_test=False, runtime=runtime,
        )
    finally:
        for handle in handles:
            handle.remove()
    result.update(
        physical_mse=float(totals[0] / totals[1]),
        finite_loss=bool(torch.isfinite(totals).all()),
        eligible_node_hours=int(totals[4]),
        backward_seconds=runtime["backward_seconds"][0],
        trainer_forward_and_loss_seconds=runtime["forward_seconds"][0],
        trainer_optimizer_and_gradient_checks_seconds=runtime["optimizer_seconds"][0],
    )
    if not result["finite_loss"] or not torch.isfinite(totals[0] / totals[1]):
        raise AssertionError("Nonfinite loss")
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="All selected histories run at smallest B; larger B runs B1-B3/I1-I4 at H48 only.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--config", type=Path, default=Path("configs/ncep.yaml"))
    parser.add_argument("--data", type=Path, default=Path(
        "/glade/u/home/zel/scratch/PACT-V2/dataset/NCEP"
    ))
    parser.add_argument("--normalization", type=Path, default=Path(
        "/glade/u/home/zel/scratch/PACT-V2/results/"
        "NCEP_Smoke_smoke_20261013T142506Z/normalization.pt"
    ))
    parser.add_argument("--year", type=int, help="A year in the saved normalization TRAIN split")
    parser.add_argument("--batch-sizes", type=int, nargs="+", help="Default: CPU 1; CUDA 1 64")
    parser.add_argument("--histories", type=int, nargs="+", default=[48, 42, 36, 24, 0])
    parser.add_argument("--cases", nargs="+", default=[
        "B1", "B2", "B3", "I1", "I2", "I3", "I4", "lstm", "temporal_cnn", "tcn",
    ])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true", help="Disable configured CUDA bf16 AMP")
    parser.add_argument("--profile-stages", action="store_true", help="Instrument component forward times; extra CUDA synchronization affects total timing")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu for CPU diagnostics")
    if any(hours not in range(0, 49, 6) for hours in args.histories):
        parser.error("--histories must be drawn from 0, 6, ..., 48")
    batch_sizes = args.batch_sizes or ([1, 64] if device.type == "cuda" else [1])
    if min(batch_sizes) < 1 or args.threads < 1:
        parser.error("Batch sizes and threads must be positive")

    base = dict(name="base", hidden_dim=64, gnn_layers=2, temporal_hidden_dim=32)
    igarss = dict(
        name="igarss27", temporal_encoder="gru", temporal_hidden_dim=32,
        spatial_encoder="unet", unet_channels=[32, 64, 128], scales=[0, 1, 2],
        projector="bilinear", projected_dim=32, mesh_encoder="gnn",
        mesh_hidden_dim=32, mesh_encoder_layers=2, fusion="concat",
        decoder="mlp", decoder_hidden_dim=64, decoder_gnn_layers=2,
    )
    cases = {
        "B1": dict(base, variant="mlp"),
        "B2": dict(base, variant="gnn"),
        "B3": dict(base, variant="stgnn"),
        "I1": dict(igarss),
        "I2": dict(igarss, fusion="adaptive"),
        "I3": dict(igarss, fusion="adaptive", decoder="gnn"),
        "I4": dict(igarss, scales=[0]),
        **{name: dict(igarss, temporal_encoder=name) for name in ("lstm", "temporal_cnn", "tcn")},
    }
    if set(args.cases) - cases.keys():
        parser.error(f"Unknown cases; choose from {list(cases)}")

    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    config = yaml.safe_load(args.config.read_text())
    normalization = torch.load(args.normalization, map_location="cpu", weights_only=True)
    saved_config = yaml.safe_load((args.normalization.parent / "resolved_config.yaml").read_text())
    manifest = json.loads((args.normalization.parent / "split_manifest.json").read_text())
    saved_data = saved_config["data"]
    saved_directory = Path(saved_data["root"]) / saved_data["source"]
    if saved_directory.resolve() != args.data.resolve():
        raise ValueError("Saved normalization belongs to a different data directory")
    train_years = manifest["split_years"]["train"]
    year = args.year if args.year is not None else train_years[0]
    if year not in train_years:
        raise ValueError("Diagnostic year must belong to the saved normalization TRAIN split")

    use_amp = config["training"]["amp"] and device.type == "cuda" and not args.no_amp
    if use_amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Configured bf16 AMP is unsupported on this CUDA device")
    torch.backends.cuda.matmul.allow_tf32 = config["training"]["tf32"]
    torch.backends.cudnn.allow_tf32 = config["training"]["tf32"]
    mesh_cpu = load_mesh(args.data / "mesh.pt")
    dataset = FullFieldDataset(args.data, [year], mesh_cpu, history_hours=48)
    if max(batch_sizes) > len(dataset):
        raise ValueError("Requested batch exceeds the diagnostic year's distinct samples")
    forcing_grid = dataset[0]["forcing"][-1, 3:5].clone()
    expected_count = (
        manifest["sample_counts"]["train"] * (saved_data["history_hours"] // 6 + 1)
        * forcing_grid.shape[-2] * forcing_grid.shape[-1]
    )
    if normalization["forcing_count_per_channel"] != expected_count:
        raise ValueError("Saved normalization count does not match the recorded full TRAIN population")
    mesh = {key: value.to(device) for key, value in mesh_cpu.items()}
    norm_device = prepare_normalization(normalization, device)
    weights = load_spatial_weights(mesh_cpu, config["spatial_weighting"]).to(device)
    report = {
        "device": str(device), "torch_version": str(torch.__version__),
        "gpu_model": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_tested": device.type == "cuda", "amp_bf16": use_amp,
        "profile_stages": args.profile_stages,
        "threads": args.threads, "data": str(args.data), "year": year,
        "mesh_nodes": len(mesh_cpu["node_id"]),
        "mesh_directed_edges": mesh_cpu["edge_index"].shape[1],
        "forcing_grid_shape": list(forcing_grid.shape),
        "normalization_file": str(args.normalization),
        "normalization_history_hours": saved_data["history_hours"],
        "normalization_train_years": train_years,
        "normalization_counts": {key: normalization[key] for key in (
            "forcing_count_per_channel", "surge_count",
        )},
        "measurement_scope": "Single first update including lazy initialization; no warmup or experiment",
        "batch_scope": "All selected histories at smallest B; larger B at H48 for B1-B3/I1-I4 only",
        "memory_scope": "CUDA peaks reset per case; CPU RSS high-water mark is cumulative for process",
        "short_history_note": f"Saved H{saved_data['history_hours']} statistics reused for shape/runtime diagnostics only; no scientific metrics",
        "results": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for hours in args.histories:
        for batch_size in batch_sizes:
            if batch_size != min(batch_sizes) and hours != 48:
                continue
            # FullFieldDataset owns history slicing and continues to return only dynamic tensors.
            dataset.history_steps = hours // 6 + 1
            for case in args.cases:
                if batch_size != min(batch_sizes) and case not in ("B1", "B2", "B3", "I1", "I2", "I3", "I4"):
                    continue
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(device)
                row = dict(case=case, history_hours=hours, batch_size=batch_size, model_config=cases[case])
                row["normalization_matches_history"] = hours == saved_data["history_hours"]
                model = batch = None
                print(f"Running {case}: H={hours}, B={batch_size}, {device}", flush=True)
                try:
                    batch = default_collate([dataset[index] for index in range(batch_size)])
                    torch.manual_seed(42)
                    if cases[case]["name"] == "base":
                        model = BaseModel(cases[case], mesh_cpu, forcing_grid, history_hours=hours)
                    else:
                        model = IGARSS27Model(cases[case], mesh_cpu, forcing_grid)
                    model = model.to(device)
                    train = deepcopy(config["training"])
                    train.update(batch_size=batch_size, grad_accum_steps=1)
                    row.update(benchmark_case(
                        model, batch, mesh, norm_device, weights, device, train, use_amp,
                        profile_stages=args.profile_stages,
                    ))
                    row["status"] = "passed"
                except (RuntimeError, AssertionError, ValueError) as error:
                    row.update(status="oom" if isinstance(error, torch.OutOfMemoryError) else "failed", error=str(error))
                row.update(
                    peak_cpu_main_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
                    peak_gpu_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
                    peak_gpu_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else None,
                )
                report["results"].append(row)
                args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
                print(json.dumps(row, allow_nan=False), flush=True)
                del model, batch
    if any(row["status"] != "passed" for row in report["results"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
