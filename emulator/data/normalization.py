"""TRAIN-only FP64 statistics and model-input normalization tensors."""

from concurrent.futures import ThreadPoolExecutor

import torch


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


def prepare_normalization(normalization, device):
    """Move saved statistics to the model device and broadcast forcing channels."""
    norm_device = {k: normalization[k].to(device=device, dtype=torch.float32)
                   for k in ("forcing_mean", "forcing_std", "surge_mean", "surge_std")}
    norm_device["forcing_mean"] = norm_device["forcing_mean"][None, None, :, None, None]
    norm_device["forcing_std"] = norm_device["forcing_std"][None, None, :, None, None]
    return norm_device
