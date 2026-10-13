#!/usr/bin/env python3
"""Minimal full-field training. The smoke model is only a pipeline check."""
import argparse
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import random
import resource
import time
import threading

import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
import yaml

from dataset import FullFieldDataset


class SmokeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 6)

    def forward(self, batch, mesh):
        values = self.linear(batch["forcing"][:, -1].mean(dim=(-2, -1)))
        return {"surge": values[:, None, :].expand(-1, len(mesh["node_id"]), -1)}


def split_years(years, config, seed=42):
    """StormSurge graph_store.py semantics, using simulation-directory years."""
    train_ratio, val_ratio = config["train_ratio"], config["val_ratio"]
    ratio_sum = train_ratio + val_ratio
    no_test = math.isclose(ratio_sum, 1.0)
    if (not math.isfinite(ratio_sum) or train_ratio <= 0 or val_ratio <= 0
            or (ratio_sum > 1.0 and not no_test)):
        raise ValueError("Require positive train/val ratios with sum <= 1")
    years = sorted(y for y in years if not config["future_only"]
                   or y > config["future_year_threshold"])
    if not years:
        raise ValueError("No simulation years remain after filtering")
    if config["shuffle_years"]:
        random.Random(seed).shuffle(years)
    count = len(years)
    if no_test:
        if count < 2:
            raise ValueError("A train/val-only split requires at least two years")
        ntrain = min(count - 1, max(1, round(train_ratio * count)))
        nval = count - ntrain
    else:
        ntrain = max(1, round(train_ratio * count))
        nval = max(1, round(val_ratio * count))
        if count <= 2:
            ntrain, nval = 1, count - 1
        elif ntrain + nval >= count:
            ntrain, nval = count - 2, 1
    return {"train": years[:ntrain], "val": years[ntrain:ntrain + nval],
            "test": years[ntrain + nval:]}


def month_moments(path, history_steps):
    """Reduce one eager-loaded month; return only small FP64 moments/counts."""
    samples = torch.load(path, map_location="cpu", weights_only=True)
    forcing_sum = torch.zeros(5, dtype=torch.float64)
    forcing_sq = torch.zeros_like(forcing_sum)
    surge_sum = torch.tensor(0., dtype=torch.float64)
    surge_sq = torch.zeros_like(surge_sum)
    forcing_count = surge_count = 0
    for sample in samples:
        forcing = sample["forcing"][-history_steps:].double()
        state, target = sample["state"], sample["surge"]
        valid = state == 0
        values = target[valid].double()
        if (not torch.isfinite(forcing).all() or not torch.isfinite(values).all()
                or torch.any((state < 0) | (state > 3))
                or not torch.isnan(target[~valid]).all()):
            raise ValueError(f"Invalid forcing/WW target/state in {path}")
        forcing_sum += forcing.sum(dim=(0, 2, 3))
        forcing_sq += forcing.square().sum(dim=(0, 2, 3))
        forcing_count += forcing.numel() // 5
        surge_sum += values.sum()
        surge_sq += values.square().sum()
        surge_count += values.numel()
    return forcing_sum, forcing_sq, forcing_count, surge_sum, surge_sq, surge_count


def compute_normalization(dataset, num_workers=1):
    """All TRAIN observations once, at most one resident month per worker."""
    forcing_sum = torch.zeros(5, dtype=torch.float64)
    forcing_sq = torch.zeros_like(forcing_sum)
    surge_sum = torch.tensor(0., dtype=torch.float64)
    surge_sq = torch.zeros_like(surge_sum)
    forcing_count = surge_count = 0
    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        # Results arrive in file order, so FP64 accumulation is reproducible.
        moments = pool.map(month_moments, dataset.files, [dataset.history_steps] * len(dataset.files))
        for month, (fs, fq, fc, ss, sq, sc) in enumerate(moments, 1):
            forcing_sum += fs
            forcing_sq += fq
            forcing_count += fc
            surge_sum += ss
            surge_sq += sq
            surge_count += sc
            if month % 10 == 0 or month == len(dataset.files):
                print(f"Normalization: {month}/{len(dataset.files)} TRAIN months", flush=True)
    if not forcing_count or not surge_count:
        raise ValueError("TRAIN normalization has no valid observations")
    forcing_mean = forcing_sum / forcing_count
    surge_mean = surge_sum / surge_count
    result = {"forcing_mean": forcing_mean,
              "forcing_std": (forcing_sq / forcing_count - forcing_mean.square()).clamp_min(0).sqrt(),
              "surge_mean": surge_mean,
              "surge_std": (surge_sq / surge_count - surge_mean.square()).clamp_min(0).sqrt(),
              "forcing_count_per_channel": forcing_count, "surge_count": surge_count}
    if any(not torch.isfinite(result[k]).all() or torch.any(result[k] <= 0)
           for k in ("forcing_std", "surge_std")):
        raise ValueError("TRAIN standard deviations must be finite and positive")
    return result


def load_spatial_weights(mesh, config):
    mode, path = config["mode"], config["weights_file"]
    if mode not in ("uniform", "weighted"):
        raise ValueError("Spatial weighting mode must be uniform or weighted")
    if path is None:
        if mode == "weighted":
            raise ValueError("weighted mode requires weights_file")
        return torch.ones(len(mesh["node_id"]), dtype=torch.float32)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    ids, weights = saved["node_id"], saved["weights"]
    if ids.dtype != torch.int64 or not torch.equal(ids, mesh["node_id"]):
        raise ValueError("Weight node_id must match canonical mesh order exactly")
    if (weights.shape != mesh["node_id"].shape or not weights.is_floating_point()
            or not torch.isfinite(weights).all() or torch.any(weights < 0)):
        raise ValueError("Weights must be finite, nonnegative floating values of shape [N]")
    weights = (weights > 0).float() if mode == "uniform" else weights
    if not torch.any(weights > 0):
        raise ValueError("Weights need at least one positive value")
    return weights


def error_totals(prediction, target, state, weights):
    """Physical-meter sums: weighted SSE/count, unweighted SSE/SAE/count."""
    valid = (state == 0) & (weights[None, :, None] > 0)
    safe_target = torch.where(valid, target, 0.)
    error = torch.where(valid, prediction - safe_target, 0.)
    if not torch.isfinite(prediction).all() or not torch.isfinite(error).all():
        raise FloatingPointError("Nonfinite prediction or eligible WW target")
    squared = error.square()
    totals = torch.stack(((squared * weights[None, :, None]).sum(dtype=torch.float64),
                          (valid * weights[None, :, None]).sum(dtype=torch.float64),
                          squared.sum(dtype=torch.float64),
                          error.abs().sum(dtype=torch.float64),
                          valid.sum(dtype=torch.float64)))
    if not torch.isfinite(totals).all():
        raise FloatingPointError("Nonfinite loss/metric accumulation")
    return totals


def evaluate(model, loader, mesh, normalization, weights, device, use_amp):
    model.eval()
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    samples = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            inputs = {k: batch[k].to(device, non_blocking=True)
                      for k in ("forcing", "tide_boundary", "tide_potential", "tide_time")}
            inputs["forcing"] = ((inputs["forcing"] - normalization["forcing_mean"])
                                 / normalization["forcing_std"])
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                output = model(inputs, mesh)["surge"]
            prediction = output.float() * normalization["surge_std"] + normalization["surge_mean"]
            totals += error_totals(prediction, batch["surge"].to(device),
                                   batch["state"].to(device), weights)
            samples += len(batch["surge"])
    if dist.is_initialized():
        dist.all_reduce(totals)
        count = torch.tensor(samples, device=device, dtype=torch.int64)
        dist.all_reduce(count)
        samples = count.item()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    if totals[4] == 0:
        raise ValueError("Evaluation has no eligible WW node-hours")
    metrics = {"all_rmse": math.sqrt((totals[2] / totals[4]).item()),
               "all_mae": (totals[3] / totals[4]).item()}
    return metrics, {"seconds": seconds, "samples": samples,
                     "samples_per_second": samples / seconds,
                     "eligible_node_hours": int(totals[4].item())}


def monitor_memory(stop, report):
    """Sample process-tree PSS: shared file/IPC pages are apportioned once."""
    while not stop.is_set():
        pids = [os.getpid()]
        pids += [int(p) for p in Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children").read_text().split()]
        pss_kib = 0
        for pid in pids:
            try:
                lines = Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines()
            except (FileNotFoundError, ProcessLookupError):
                continue  # A DataLoader worker exited between enumeration and measurement.
            pss_kib += sum(int(line.split()[1]) for line in lines if line.startswith("Pss:"))
        report["peak_cpu_process_tree_pss_gib"] = max(report.get("peak_cpu_process_tree_pss_gib", 0), pss_kib / 2**20)
        stop.wait(0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/ncep.yaml")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Full TRAIN statistics, four real training batches, limited VAL/TEST")
    args = parser.parse_args()
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    train = config["training"]
    if (config["normalization"] != {"forcing": "mean_std", "target": "mean_std"}
            or not config["loss"]["global_mse"]):
        raise ValueError("Only mean/std normalization and Global MSE are implemented")
    if config["loss"]["tail"]["weight"] != 0 or config["loss"]["episode_peak"]["weight"] != 0:
        raise NotImplementedError("Tail and Episode Peak losses await the Episode Audit")
    if config["model"]["name"] != "smoke":
        if config["model"]["name"] in ("base", "igarss27", "full_v1", "full_v2"):
            raise NotImplementedError(f"{config['model']['name']} architecture has not been implemented")
        raise ValueError("Unknown model identifier")
    evaluation = config["evaluation"]
    if evaluation["checkpoint_selection"] != "overall" or not evaluation["save_overall_checkpoint"]:
        raise NotImplementedError("This initial pipeline requires best_overall checkpoint selection")
    if evaluation["save_predictions"]:
        raise NotImplementedError("Full-field prediction export is not implemented")
    if train["optimizer"] != "adam" or train["scheduler"] != "cosine" or train["amp_dtype"] != "bf16":
        raise ValueError("Use Adam, cosine scheduling, and bf16 AMP")
    if (min(train["epochs"], train["batch_size"], train["grad_accum_steps"], evaluation["val_every_epochs"]) < 1
            or train["warmup_epochs"] < 0 or train["warmup_epochs"] >= train["epochs"]
            or not 0 < train["warmup_start_factor"] <= 1
            or not 0 <= train["min_lr"] <= train["lr"] or train["lr"] <= 0
            or train["weight_decay"] < 0 or train["max_grad_norm"] < 0):
        raise ValueError("Invalid training/scheduler configuration")
    seed = config["experiment"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(train["deterministic"])
    torch.backends.cuda.matmul.allow_tf32 = train["tf32"]
    torch.backends.cudnn.allow_tf32 = train["tf32"]
    torch.backends.cudnn.benchmark = not train["deterministic"]
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if train["distributed"] not in ("auto", True, False):
        raise ValueError("distributed must be auto or a YAML boolean")
    if world > 1 and train["distributed"] is False:
        raise ValueError("torchrun WORLD_SIZE > 1 but distributed is false")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if train["device"] == "auto" else train["device"])
    if device.type == "cuda":
        if world > 1:
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        elif device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    if world > 1:
        # The initial full TRAIN statistics pass can exceed the NCCL default timeout.
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(hours=1))
    use_amp = train["amp"] and device.type == "cuda"
    if use_amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requested bf16 AMP is unsupported by this CUDA device")
    print(f"rank={rank}/{world} device={device} torch={torch.__version__} AMP={use_amp}", flush=True)
    memory_stop = threading.Event()
    memory_report = {}
    memory_thread = threading.Thread(target=monitor_memory, args=(memory_stop, memory_report), daemon=True)
    memory_thread.start()
    start = time.perf_counter()
    directory = Path(config["data"]["root"]) / config["data"]["source"]
    available = sorted(int(p.name) for p in directory.iterdir() if p.is_dir() and p.name.isdigit())
    splits = split_years(available, config["split"], seed)
    if not splits["val"]:
        raise ValueError("No real VAL year exists; checkpoint selection requires a validation split")
    mesh_cpu = torch.load(directory / "mesh.pt", map_location="cpu", weights_only=True)
    node_id = mesh_cpu["node_id"]
    n = len(node_id)
    expected = {"node_id": (torch.int64, (n,)), "pos": (torch.float64, (n, 2)),
                "x": (torch.float32, (n, 5)), "edge_index": (torch.int64, (2, None)),
                "edge_attr": (torch.float64, (None, 3)), "face": (torch.int64, (3, None)),
                "boundary_node_index": (torch.int64, (None,))}
    for key, (dtype, shape) in expected.items():
        value = mesh_cpu[key]
        if (value.dtype != dtype or value.ndim != len(shape)
                or any(a is not None and a != b for a, b in zip(shape, value.shape))
                or (value.is_floating_point() and not torch.isfinite(value).all())):
            raise ValueError(f"Invalid shared mesh field {key}")
    if (not n or torch.any(node_id <= 0) or torch.any(node_id[1:] <= node_id[:-1])
            or len(mesh_cpu["edge_attr"]) != mesh_cpu["edge_index"].shape[1]):
        raise ValueError("Invalid canonical mesh identity or edges")
    for key in ("edge_index", "face", "boundary_node_index"):
        value = mesh_cpu[key]
        if value.numel() and (value.min() < 0 or value.max() >= n):
            raise ValueError(f"Invalid shared mesh indices: {key}")
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
               "mesh_nodes": n, "mesh_directed_edges": mesh_cpu["edge_index"].shape[1]}
    print(json.dumps(runtime), flush=True)
    resume = torch.load(train["resume_from"], map_location="cpu", weights_only=True) if train["resume_from"] else None
    if resume:
        if (resume["model_identifier"] != config["model"]["name"]
                or resume["config"]["experiment"]["seed"] != seed
                or not torch.equal(resume["node_id"], node_id) or resume["split_years"] != splits):
            raise ValueError("Resume model, seed, canonical node identity, or split mismatch")
        for section in ("data", "split", "normalization", "spatial_weighting", "model", "loss"):
            if resume["config"][section] != config[section]:
                raise ValueError(f"Resume configuration mismatch: {section}")
        for key in train:
            if key not in ("resume_from", "device", "distributed") and train[key] != resume["config"]["training"][key]:
                raise ValueError(f"Resume training configuration mismatch: {key}")
        if resume["smoke_test"] != args.smoke_test:
            raise ValueError("Cannot mix smoke and full-training checkpoints")
        output_dir = Path(train["resume_from"]).resolve().parent
    else:
        name = config["experiment"]["name"]
        if args.smoke_test:
            name += "_smoke_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = Path(config["experiment"]["output_dir"]) / name
        if world > 1:
            shared_path = [str(output_dir) if rank == 0 else None]
            dist.broadcast_object_list(shared_path, src=0)
            output_dir = Path(shared_path[0])
        if (output_dir / "last.pt").exists():
            raise FileExistsError(f"Run exists: set training.resume_from to {output_dir / 'last.pt'} in YAML")
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        manifest = {"source": config["data"]["source"], "available_years": available,
                    "split_years": splits, "sample_counts": runtime["sample_counts"],
                    "monthly_sample_counts": {
                        name: {str(path.relative_to(directory)): sum(m == i for m, _ in ds.index)
                               for i, path in enumerate(ds.files)} for name, ds in datasets.items()}}
        (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    stats_start = time.perf_counter()
    normalization = resume["normalization"] if resume else (compute_normalization(datasets["train"], config["dataloader"]["num_workers"]) if rank == 0 else None)
    if world > 1:
        payload = [normalization]
        dist.broadcast_object_list(payload, src=0)
        normalization = payload[0]
    runtime["normalization_seconds"] = time.perf_counter() - stats_start
    if rank == 0:
        torch.save(normalization, output_dir / "normalization.pt")
    norm_device = {k: normalization[k].to(device=device, dtype=torch.float32)
                   for k in ("forcing_mean", "forcing_std", "surge_mean", "surge_std")}
    norm_device["forcing_mean"] = norm_device["forcing_mean"][None, None, :, None, None]
    norm_device["forcing_std"] = norm_device["forcing_std"][None, None, :, None, None]
    weights = load_spatial_weights(mesh_cpu, config["spatial_weighting"]).to(device)
    mesh = {k: v.to(device) for k, v in mesh_cpu.items()}
    workers = config["dataloader"]["num_workers"]
    loader_options = dict(batch_size=train["batch_size"], num_workers=workers,
                          pin_memory=config["dataloader"]["pin_memory"])
    if workers:
        loader_options.update(persistent_workers=config["dataloader"]["persistent_workers"],
                              prefetch_factor=config["dataloader"]["prefetch_factor"])
    start = time.perf_counter()
    sampler = DistributedSampler(datasets["train"], num_replicas=world, rank=rank, shuffle=True, seed=seed) if world > 1 else None
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(datasets["train"], shuffle=sampler is None, sampler=sampler,
                              generator=generator, **loader_options)
    eval_loaders = {}
    for name in ("val", "test"):
        if name not in datasets:
            continue
        # Rank-strided evaluation never pads or duplicates held-out observations.
        indices = list(range(len(datasets[name])))
        if args.smoke_test:
            indices = sorted(torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed)).tolist()[:2 * train["batch_size"]])
        eval_loaders[name] = DataLoader(datasets[name], sampler=indices[rank::world], **loader_options)
    runtime["dataloader_initialization_seconds"] = time.perf_counter() - start
    if args.smoke_test:
        benchmark_loader = DataLoader(datasets["train"], shuffle=True,
                                      generator=torch.Generator().manual_seed(seed), **loader_options)
        tick = time.perf_counter()
        benchmark_iterator = iter(benchmark_loader)
        loading_seconds, batch_sizes = [], []
        for _ in range(min(6, len(benchmark_loader))):
            loaded = next(benchmark_iterator)
            loading_seconds.append(time.perf_counter() - tick)
            batch_sizes.append(len(loaded["forcing"]))
            del loaded
            tick = time.perf_counter()
        runtime["random_access_benchmark"] = {
            "first_batch_seconds": loading_seconds[0], "repeated_batch_seconds": loading_seconds[1:],
            "repeated_samples_per_second": sum(batch_sizes[1:]) / sum(loading_seconds[1:]) if len(loading_seconds) > 1 else None,
            "total_samples_per_second": sum(batch_sizes) / sum(loading_seconds),
            "batch_sizes": batch_sizes}
        del benchmark_iterator, benchmark_loader
        print("Random-access DataLoader benchmark: " + json.dumps(runtime["random_access_benchmark"]), flush=True)
    raw_model = SmokeModel().to(device)
    optimizer = torch.optim.Adam(raw_model.parameters(), lr=train["lr"], weight_decay=train["weight_decay"])
    warmup = train["warmup_epochs"]
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train["epochs"] - warmup, eta_min=train["min_lr"])
    if warmup:
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=train["warmup_start_factor"], total_iters=warmup), cosine],
            milestones=[warmup])
    else:
        scheduler = cosine
    start_epoch, best_rmse, best_epoch = 0, math.inf, None
    if resume:
        raw_model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        start_epoch, best_rmse, best_epoch = resume["epoch"] + 1, resume["best_val_all_rmse"], resume["best_epoch"]
        if best_epoch is not None and not (output_dir / "best_overall.pt").is_file():
            raise FileNotFoundError("Resume requires the run's validation-selected best_overall.pt")
        print(f"Resumed last.pt at epoch {start_epoch}; normalization reused", flush=True)
    model = DistributedDataParallel(raw_model, device_ids=[device.index] if device.type == "cuda" else None) if world > 1 else raw_model
    last_epoch = start_epoch + 1 if args.smoke_test else train["epochs"]
    runtime.update(forward_seconds=[], backward_seconds=[], optimizer_seconds=[], batch_loading_seconds=[], training_batch_sizes=[])
    for epoch in range(start_epoch, last_epoch):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        else:
            generator.manual_seed(seed + epoch)
        # Reconstruct exactly the global permutation, without advancing the loader's RNG.
        if args.smoke_test:
            if sampler is not None:
                shuffled = list(sampler)
            else:
                probe = torch.Generator().manual_seed(seed + epoch)
                torch.empty((), dtype=torch.int64).random_(generator=probe)  # DataLoader base seed
                shuffled = torch.randperm(len(datasets["train"]), generator=probe).tolist()
            first_indices = shuffled[:4 * train["batch_size"]]
            identities = [{"sample_index": i, "simulation_year": int(datasets["train"].files[datasets["train"].index[i][0]].parent.name),
                           "monthly_file": datasets["train"].files[datasets["train"].index[i][0]].name,
                           "local_sample_index": datasets["train"].index[i][1]} for i in first_indices]
            runtime["shuffled_batches"] = [identities[i:i + train["batch_size"]] for i in range(0, len(identities), train["batch_size"])]
            if len({p["monthly_file"] for p in identities[:train["batch_size"]]}) < 2:
                raise RuntimeError("Smoke shuffle did not mix monthly files")
            if len(splits["train"]) > 1 and len({p["simulation_year"] for p in identities[:train["batch_size"]]}) < 2:
                raise RuntimeError("Smoke shuffle did not mix simulation years")
        total = torch.zeros(5, device=device, dtype=torch.float64)
        denominator = torch.zeros((), device=device, dtype=torch.float64)
        optimizer.zero_grad(set_to_none=True)
        steps = min(4, len(train_loader)) if args.smoke_test else len(train_loader)
        tick = time.perf_counter()
        iterator = iter(train_loader)
        for step in range(steps):
            batch = next(iterator)
            runtime["batch_loading_seconds"].append(time.perf_counter() - tick)
            runtime["training_batch_sizes"].append(len(batch["forcing"]))
            if args.smoke_test:
                # Check actual returned data against the sampled indices (no metadata in batches).
                offset = step * train["batch_size"]
                for j, index in enumerate(first_indices[offset:offset + len(batch["forcing"])]):
                    if not torch.equal(batch["forcing"][j], datasets["train"][index]["forcing"]):
                        raise RuntimeError("Observed DataLoader order differs from global sample permutation")
            inputs = {k: batch[k].to(device, non_blocking=True)
                      for k in ("forcing", "tide_boundary", "tide_potential", "tide_time")}
            inputs["forcing"] = (inputs["forcing"] - norm_device["forcing_mean"]) / norm_device["forcing_std"]
            target, state = batch["surge"].to(device), batch["state"].to(device)
            update = (step + 1) % train["grad_accum_steps"] == 0 or step + 1 == steps
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tick = time.perf_counter()
            with model.no_sync() if world > 1 and not update else nullcontext():
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                    output = model(inputs, mesh)["surge"]
                prediction = output.float() * norm_device["surge_std"] + norm_device["surge_mean"]
                batch_totals = error_totals(prediction, target, state, weights)
                # Tail/GT-aligned Episode Peak losses join Global MSE here after the Episode Audit.
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                runtime["forward_seconds"].append(time.perf_counter() - tick)
                tick = time.perf_counter()
                batch_totals[0].backward()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                runtime["backward_seconds"].append(time.perf_counter() - tick)
            total += batch_totals.detach()
            denominator += batch_totals[1].detach()
            if update:
                tick = time.perf_counter()
                if world > 1:
                    dist.all_reduce(denominator)
                if denominator <= 0:
                    raise ValueError("Accumulation window has no positive-weight WW supervision")
                # DDP averages gradients; divide once by the complete window's global weight count.
                for parameter in raw_model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.mul_(world / denominator)
                        if not torch.isfinite(parameter.grad).all():
                            raise FloatingPointError("Nonfinite training gradients")
                if train["max_grad_norm"] > 0:
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), train["max_grad_norm"], error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                denominator.zero_()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                runtime["optimizer_seconds"].append(time.perf_counter() - tick)
            if args.smoke_test:
                print(f"epoch={epoch} batch={step + 1}/{steps} physical_mse={(batch_totals[0] / batch_totals[1]).item():.6f}", flush=True)
            tick = time.perf_counter()
        if world > 1:
            dist.all_reduce(total)
        if args.smoke_test:
            runtime["global_shuffle_verified"] = True
            del iterator, train_loader
        lr_used = optimizer.param_groups[0]["lr"]
        scheduler.step()
        val_metrics = None
        improved = False
        if (epoch + 1) % evaluation["val_every_epochs"] == 0 or epoch + 1 == last_epoch:
            val_metrics, val_runtime = evaluate(raw_model, eval_loaders["val"], mesh, norm_device, weights, device, use_amp)
            runtime["validation"] = val_runtime
            improved = val_metrics["all_rmse"] < best_rmse
            if improved:
                best_rmse, best_epoch = val_metrics["all_rmse"], epoch
        if rank == 0:
            checkpoint = {"model": raw_model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scheduler": scheduler.state_dict(), "amp_scaler": None, "epoch": epoch,
                          "config": config, "normalization": normalization, "split_years": splits,
                          "model_identifier": config["model"]["name"], "node_id": node_id,
                          "validation_metrics": val_metrics, "best_val_all_rmse": best_rmse,
                          "best_epoch": best_epoch, "smoke_test": args.smoke_test}
            torch.save(checkpoint, output_dir / "last.pt")
            if improved:
                torch.save(checkpoint, output_dir / "best_overall.pt")
            row = {"epoch": epoch, "lr": lr_used, "train_global_mse": (total[0] / total[1]).item(),
                   "validation": val_metrics, "smoke_test": args.smoke_test}
            with (output_dir / "training_metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row), flush=True)
        if world > 1:
            dist.barrier()
    selected = torch.load(output_dir / "best_overall.pt", map_location="cpu", weights_only=True)
    raw_model.load_state_dict(selected["model"])
    reloaded_val, _ = evaluate(raw_model, eval_loaders["val"], mesh, norm_device, weights, device, use_amp)
    if any(not math.isclose(reloaded_val[k], selected["validation_metrics"][k], rel_tol=1e-6, abs_tol=1e-7) for k in reloaded_val):
        raise RuntimeError("Checkpoint reload validation metrics changed")
    runtime["checkpoint_reload_succeeded"] = True
    test_metrics = None
    if evaluation["test_at_end"] and "test" in eval_loaders:
        test_metrics, runtime["test"] = evaluate(raw_model, eval_loaders["test"], mesh, norm_device, weights, device, use_amp)
    memory_stop.set()
    memory_thread.join()
    runtime.update(memory_report)
    runtime["peak_cpu_main_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    runtime["peak_gpu_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    loading = runtime["batch_loading_seconds"]
    runtime["initial_batch_loading_seconds"] = loading[0] if loading else None
    runtime["repeated_shuffled_batch_loading_seconds"] = loading[1:]
    runtime["prefetched_batch_availability_samples_per_second"] = (sum(runtime["training_batch_sizes"][1:]) / sum(loading[1:])) if len(loading) > 1 else None
    summary = {"smoke_test": args.smoke_test,
               "evaluation_scope": "seeded subset of at most two batches per held-out split" if args.smoke_test else "complete split",
               "implemented_metrics": ["all_rmse", "all_mae"], "selected_checkpoint": "best_overall.pt",
               "selected_epoch": selected["epoch"], "validation": reloaded_val, "test": test_metrics,
               "test_status": "evaluated" if test_metrics else ("no real TEST split" if "test" not in datasets else "disabled"),
               "resume_succeeded": resume is not None, "runtime": runtime}
    if rank == 0:
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
        print(f"Results: {output_dir}", flush=True)
        print(json.dumps({k: v for k, v in summary.items() if k != "runtime"}), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
