"""Globally shuffled training and unpadded rank-strided evaluation loaders."""

import torch
from torch.utils.data import DataLoader, DistributedSampler


def loader_options(config):
    """Keep worker-only options absent when loading on the main process."""
    workers = config["dataloader"]["num_workers"]
    options = dict(
        batch_size=config["training"]["batch_size"],
        num_workers=workers,
        pin_memory=config["dataloader"]["pin_memory"],
    )
    if workers:
        options.update(
            persistent_workers=config["dataloader"]["persistent_workers"],
            prefetch_factor=config["dataloader"]["prefetch_factor"],
        )
    return options


def build_loaders(datasets, train_config, options, seed, world, rank, smoke_test):
    """Return train/evaluation loaders and the sampler/generator for epoch seeding.

    Training shuffles the dataset's global sample index. Evaluation partitions
    held-out indices across ranks without padding; smoke runs first choose the
    same seeded subset on every rank, then partition it.
    """
    sampler = (
        DistributedSampler(
            datasets["train"], num_replicas=world, rank=rank, shuffle=True, seed=seed
        )
        if world > 1 else None
    )
    # Keep loader randomness independent of model initialization and updates.
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        datasets["train"],
        shuffle=sampler is None,
        sampler=sampler,
        generator=generator,
        **options,
    )

    eval_loaders = {}
    for name in ("val", "test"):
        if name not in datasets:
            continue
        # Rank-strided evaluation never pads or duplicates held-out observations.
        indices = list(range(len(datasets[name])))
        if smoke_test:
            indices = sorted(
                torch.randperm(
                    len(indices), generator=torch.Generator().manual_seed(seed)
                ).tolist()[:2 * train_config["batch_size"]]
            )
        eval_loaders[name] = DataLoader(
            datasets[name], sampler=indices[rank::world], **options
        )

    return train_loader, eval_loaders, sampler, generator
