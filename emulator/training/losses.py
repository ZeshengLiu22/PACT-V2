"""Canonical spatial weights and NaN-safe physical-meter error sums."""

import torch


def load_spatial_weights(mesh, config):
    """Return nonnegative weights [N] in the mesh's canonical node order.

    A supplied zero weight excludes a node in both modes. Uniform mode treats
    positive entries as eligibility only; weighted mode retains their values.
    """
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
    if (
        weights.shape != mesh["node_id"].shape
        or not weights.is_floating_point()
        or not torch.isfinite(weights).all()
        or torch.any(weights < 0)
    ):
        raise ValueError("Weights must be finite, nonnegative floating values of shape [N]")

    weights = (weights > 0).float() if mode == "uniform" else weights
    if not torch.any(weights > 0):
        raise ValueError("Weights need at least one positive value")
    return weights


def error_totals(prediction, target, state, weights):
    """Return differentiable physical-meter error sums for [B, N, H] fields.

    B is batch size, N is mesh nodes, and H is forecast horizons.
    Weights [N] broadcast across batches and horizons. The five FP64 totals are
    weighted SSE/count followed by unweighted SSE/SAE/count, restricted to WW
    (state zero) node-hours with positive weights.
    """
    valid = (state == 0) & (weights[None, :, None] > 0)
    # Non-WW targets contain NaNs. Replace them before subtraction, then mask
    # the error so excluded entries contribute neither loss nor gradients.
    safe_target = torch.where(valid, target, 0.)
    error = torch.where(valid, prediction - safe_target, 0.)
    if not torch.isfinite(prediction).all() or not torch.isfinite(error).all():
        raise FloatingPointError("Nonfinite prediction or eligible WW target")

    squared = error.square()
    # Keep weighted training sums separate from unweighted population metrics.
    totals = torch.stack(
        (
            (squared * weights[None, :, None]).sum(dtype=torch.float64),
            (valid * weights[None, :, None]).sum(dtype=torch.float64),
            squared.sum(dtype=torch.float64),
            error.abs().sum(dtype=torch.float64),
            valid.sum(dtype=torch.float64),
        )
    )
    if not torch.isfinite(totals).all():
        raise FloatingPointError("Nonfinite loss/metric accumulation")
    return totals

