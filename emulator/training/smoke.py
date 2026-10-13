"""Smoke-only random-access profiling and observed shuffle verification."""

import time

import torch
from torch.utils.data import DataLoader

from emulator.training.logging import logger, log_loading_benchmark


def benchmark_loading(dataset, loader_options, seed, runtime):
    """Record at most six shuffled batch-load timings without model computation."""
    logger.info("Smoke random-access DataLoader benchmark started: up to six shuffled batches")
    # A separate generator keeps profiling from advancing training sample order.
    benchmark_loader = DataLoader(
        dataset, shuffle=True,
        generator=torch.Generator().manual_seed(seed), **loader_options,
    )
    tick = time.perf_counter()
    benchmark_iterator = iter(benchmark_loader)
    loading_seconds, batch_sizes = [], []

    # The first timing includes iterator/worker startup; later ones include any
    # prefetch overlap. Count actual samples, including a possible partial batch.
    for _ in range(min(6, len(benchmark_loader))):
        loaded = next(benchmark_iterator)
        loading_seconds.append(time.perf_counter() - tick)
        batch_sizes.append(len(loaded["forcing"]))
        del loaded
        tick = time.perf_counter()

    runtime["random_access_benchmark"] = {
        "first_batch_seconds": loading_seconds[0],
        "repeated_batch_seconds": loading_seconds[1:],
        "repeated_samples_per_second": (
            sum(batch_sizes[1:]) / sum(loading_seconds[1:])
            if len(loading_seconds) > 1 else None
        ),
        "total_samples_per_second": sum(batch_sizes) / sum(loading_seconds),
        "batch_sizes": batch_sizes,
    }
    # Drop the benchmark's worker/loader references before training begins.
    del benchmark_iterator, benchmark_loader
    log_loading_benchmark(runtime["random_access_benchmark"])


def record_shuffle(dataset, sampler, seed, epoch, batch_size, runtime):
    """Reconstruct the permutation without advancing the loader's RNG."""
    if sampler is not None:
        shuffled = list(sampler)
    else:
        probe = torch.Generator().manual_seed(seed + epoch)
        # DataLoader draws its worker base seed before generating the permutation.
        # Mirror that draw in the probe so the expected sample order is exact.
        torch.empty((), dtype=torch.int64).random_(generator=probe)
        shuffled = torch.randperm(len(dataset), generator=probe).tolist()

    first_indices = shuffled[:4 * batch_size]
    identities = [
        {
            "sample_index": i,
            "simulation_year": int(dataset.files[dataset.index[i][0]].parent.name),
            "monthly_file": dataset.files[dataset.index[i][0]].name,
            "local_sample_index": dataset.index[i][1],
        }
        for i in first_indices
    ]
    runtime["shuffled_batches"] = [
        identities[i:i + batch_size] for i in range(0, len(identities), batch_size)
    ]

    # Smoke diagnostics require evidence that shuffling spans source partitions.
    if len({p["monthly_file"] for p in identities[:batch_size]}) < 2:
        raise RuntimeError("Smoke shuffle did not mix monthly files")
    if (
        len(dataset.years) > 1
        and len({p["simulation_year"] for p in identities[:batch_size]}) < 2
    ):
        raise RuntimeError("Smoke shuffle did not mix simulation years")
    return first_indices


def verify_batch_order(batch, dataset, first_indices, step, batch_size):
    """Check that returned forcing [B, T, C, Y, X] follows the expected shuffle."""
    # Check actual returned data against the sampled indices (no metadata in batches).
    offset = step * batch_size
    for j, index in enumerate(first_indices[offset:offset + len(batch["forcing"])]):
        if not torch.equal(batch["forcing"][j], dataset[index]["forcing"]):
            raise RuntimeError("Observed DataLoader order differs from global sample permutation")
