"""Unweighted overall metrics from complete eligible WW populations."""

import math


def overall_metrics(totals):
    """Convert globally summed error totals [5] to unweighted RMSE and MAE.

    Positive weights determine eligibility, but their magnitudes do not weight
    overall metrics. Summing before division avoids averaging batch RMSEs.
    """
    if totals[4] == 0:
        raise ValueError("Evaluation has no eligible WW node-hours")
    metrics = {
        "all_rmse": math.sqrt((totals[2] / totals[4]).item()),
        "all_mae": (totals[3] / totals[4]).item(),
    }
    return metrics


def training_metrics(epoch, lr_used, total, val_metrics, smoke_test):
    """Report the weighted training objective separately from overall metrics."""
    return {
        "epoch": epoch,
        "lr": lr_used,
        "train_global_mse": (total[0] / total[1]).item(),
        "validation": val_metrics,
        "smoke_test": smoke_test,
    }
