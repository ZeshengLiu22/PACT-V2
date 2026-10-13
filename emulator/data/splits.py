"""Simulation-year partitioning with deterministic optional shuffling."""

import math
import random


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

