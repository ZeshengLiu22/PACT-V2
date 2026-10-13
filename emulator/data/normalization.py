"""TRAIN-only FP64 statistics and model-input normalization tensors."""

from concurrent.futures import ThreadPoolExecutor

import torch


def month_moments(path, history_steps):
    """Return forcing [C] and scalar WW surge sums, squared sums, and counts.

    Forcing has shape [T, C, Y, X]; only the requested history contributes.
    Surge/state have shape [N, H], and state 0 identifies eligible WW targets.
    """
    # Each worker eagerly reads one month; only small moments leave this call.
    samples = torch.load(path, map_location="cpu", weights_only=True)
    forcing_sum = torch.zeros(5, dtype=torch.float64)
    forcing_sq = torch.zeros_like(forcing_sum)
    surge_sum = torch.tensor(0., dtype=torch.float64)
    surge_sq = torch.zeros_like(surge_sum)
    forcing_count = surge_count = 0

    for sample in samples:
        # FP64 sums and squared sums reduce precision loss over the full TRAIN set.
        forcing = sample["forcing"][-history_steps:].double()
        state, target = sample["state"], sample["surge"]
        valid = state == 0
        # Select WW entries before reducing: the excluded targets contain NaNs.
        values = target[valid].double()
        if (
            not torch.isfinite(forcing).all()
            or not torch.isfinite(values).all()
            or torch.any((state < 0) | (state > 3))
            or not torch.isnan(target[~valid]).all()
        ):
            raise ValueError(f"Invalid forcing/WW target/state in {path}")

        forcing_sum += forcing.sum(dim=(0, 2, 3))
        forcing_sq += forcing.square().sum(dim=(0, 2, 3))
        forcing_count += forcing.numel() // 5
        surge_sum += values.sum()
        surge_sq += values.square().sum()
        surge_count += values.numel()

    return forcing_sum, forcing_sq, forcing_count, surge_sum, surge_sq, surge_count


def compute_normalization(dataset, num_workers=1):
    """Compute population mean/std from every sample in the supplied TRAIN set.

    Callers must pass only TRAIN data to avoid held-out information leakage.
    Return FP64 forcing [C] and scalar surge statistics plus observation counts;
    at most one eagerly loaded month is resident per worker.
    """
    forcing_sum = torch.zeros(5, dtype=torch.float64)
    forcing_sq = torch.zeros_like(forcing_sum)
    surge_sum = torch.tensor(0., dtype=torch.float64)
    surge_sq = torch.zeros_like(surge_sum)
    forcing_count = surge_count = 0

    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as pool:
        # Results arrive in file order, so FP64 accumulation is reproducible.
        moments = pool.map(
            month_moments,
            dataset.files,
            [dataset.history_steps] * len(dataset.files),
        )
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

    # Population moments weight every observation equally, regardless of month.
    forcing_mean = forcing_sum / forcing_count
    surge_mean = surge_sum / surge_count
    result = {
        "forcing_mean": forcing_mean,
        "forcing_std": (
            (forcing_sq / forcing_count - forcing_mean.square()).clamp_min(0).sqrt()
        ),
        "surge_mean": surge_mean,
        "surge_std": (surge_sq / surge_count - surge_mean.square()).clamp_min(0).sqrt(),
        "forcing_count_per_channel": forcing_count,
        "surge_count": surge_count,
    }
    if any(
        not torch.isfinite(result[k]).all() or torch.any(result[k] <= 0)
        for k in ("forcing_std", "surge_std")
    ):
        raise ValueError("TRAIN standard deviations must be finite and positive")
    return result


def prepare_normalization(normalization, device):
    """Return FP32 device statistics for model inputs and physical surge outputs.

    Forcing statistics broadcast as [1, 1, C, 1, 1] over [B, T, C, Y, X];
    surge statistics remain scalar. Saved FP64 statistics are left unchanged.
    """
    norm_device = {
        k: normalization[k].to(device=device, dtype=torch.float32)
        for k in ("forcing_mean", "forcing_std", "surge_mean", "surge_std")
    }
    norm_device["forcing_mean"] = norm_device["forcing_mean"][None, None, :, None, None]
    norm_device["forcing_std"] = norm_device["forcing_std"][None, None, :, None, None]
    return norm_device
