"""Smoke-only random-access profiling and observed shuffle verification."""

import json
import time

import torch
from torch.utils.data import DataLoader


def benchmark_loading(dataset, loader_options, seed, runtime):
    benchmark_loader = DataLoader(dataset, shuffle=True,
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


def record_shuffle(dataset, sampler, seed, epoch, batch_size, runtime):
    """Reconstruct the permutation without advancing the loader's RNG."""
    if sampler is not None:
        shuffled = list(sampler)
    else:
        probe = torch.Generator().manual_seed(seed + epoch)
        torch.empty((), dtype=torch.int64).random_(generator=probe)  # DataLoader base seed
        shuffled = torch.randperm(len(dataset), generator=probe).tolist()
    first_indices = shuffled[:4 * batch_size]
    identities = [{"sample_index": i, "simulation_year": int(dataset.files[dataset.index[i][0]].parent.name),
                   "monthly_file": dataset.files[dataset.index[i][0]].name,
                   "local_sample_index": dataset.index[i][1]} for i in first_indices]
    runtime["shuffled_batches"] = [identities[i:i + batch_size] for i in range(0, len(identities), batch_size)]
    if len({p["monthly_file"] for p in identities[:batch_size]}) < 2:
        raise RuntimeError("Smoke shuffle did not mix monthly files")
    if len(dataset.years) > 1 and len({p["simulation_year"] for p in identities[:batch_size]}) < 2:
        raise RuntimeError("Smoke shuffle did not mix simulation years")
    return first_indices


def verify_batch_order(batch, dataset, first_indices, step, batch_size):
    # Check actual returned data against the sampled indices (no metadata in batches).
    offset = step * batch_size
    for j, index in enumerate(first_indices[offset:offset + len(batch["forcing"])]):
        if not torch.equal(batch["forcing"][j], dataset[index]["forcing"]):
            raise RuntimeError("Observed DataLoader order differs from global sample permutation")
