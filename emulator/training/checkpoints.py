"""Epoch checkpoints, resume identity guards, and VAL-selected reloads."""

from datetime import datetime, timezone
import math
from pathlib import Path

import torch
from torch import distributed as dist

from emulator.training.logging import logger, log_best_checkpoint


def load_resume(config, node_id, splits, smoke_test):
    """Load a CPU checkpoint only if experiment identity and settings match.

    Canonical node IDs and simulation-year splits must agree exactly. Device
    placement may change, but settings defining the learned experiment may not.
    """
    train = config["training"]
    seed = config["experiment"]["seed"]
    resume = (
        torch.load(train["resume_from"], map_location="cpu", weights_only=True)
        if train["resume_from"] else None
    )
    if resume:
        if (
            resume["model_identifier"] != config["model"]["name"]
            or resume["config"]["experiment"]["seed"] != seed
            or not torch.equal(resume["node_id"], node_id)
            or resume["split_years"] != splits
        ):
            raise ValueError("Resume model, seed, canonical node identity, or split mismatch")

        for section in ("data", "split", "normalization", "spatial_weighting", "model", "loss"):
            if resume["config"][section] != config[section]:
                raise ValueError(f"Resume configuration mismatch: {section}")
        for key in train:
            if (
                key not in ("resume_from", "device", "distributed")
                and train[key] != resume["config"]["training"][key]
            ):
                raise ValueError(f"Resume training configuration mismatch: {key}")
        if resume["smoke_test"] != smoke_test:
            raise ValueError("Cannot mix smoke and full-training checkpoints")
    return resume


def prepare_output_directory(config, resume, smoke_test, world, rank):
    """Choose one shared run directory, retaining the checkpoint directory on resume."""
    train = config["training"]
    if resume:
        output_dir = Path(train["resume_from"]).resolve().parent
    else:
        name = config["experiment"]["name"]
        if smoke_test:
            name += "_smoke_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = Path(config["experiment"]["output_dir"]) / name
        if world > 1:
            # Rank-zero naming prevents smoke timestamps from separating ranks.
            shared_path = [str(output_dir) if rank == 0 else None]
            dist.broadcast_object_list(shared_path, src=0)
            output_dir = Path(shared_path[0])
        if (output_dir / "last.pt").exists():
            raise FileExistsError(
                f"Run exists: set training.resume_from to {output_dir / 'last.pt'} in YAML"
            )
    return output_dir


def restore_training_state(resume, raw_model, optimizer, scheduler, output_dir):
    """Restore weights and update state; return the next epoch and saved VAL best."""
    start_epoch, best_rmse, best_epoch = 0, math.inf, None
    if resume:
        raw_model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        start_epoch, best_rmse, best_epoch = (
            resume["epoch"] + 1, resume["best_val_all_rmse"], resume["best_epoch"]
        )
        # The last epoch need not be the best epoch; final evaluation must still
        # have access to the checkpoint previously selected on validation RMSE.
        if best_epoch is not None and not (output_dir / "best_overall.pt").is_file():
            raise FileNotFoundError("Resume requires the run's validation-selected best_overall.pt")
    return start_epoch, best_rmse, best_epoch


def save_checkpoint(
    output_dir, raw_model, optimizer, scheduler, *, epoch,
    config, normalization, splits, node_id, val_metrics,
    best_rmse, best_epoch, smoke_test, improved,
):
    """Save complete epoch state and update the best checkpoint on VAL improvement."""
    # Persist the unwrapped model so CPU, single-device, and DDP runs share keys.
    # Keep the existing checkpoint schema, including the unused bf16 scaler slot.
    checkpoint = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "amp_scaler": None,
        "epoch": epoch,
        "config": config,
        "normalization": normalization,
        "split_years": splits,
        "model_identifier": config["model"]["name"],
        "node_id": node_id,
        "validation_metrics": val_metrics,
        "best_val_all_rmse": best_rmse,
        "best_epoch": best_epoch,
        "smoke_test": smoke_test,
    }
    torch.save(checkpoint, output_dir / "last.pt")
    logger.info("Saved checkpoint: %s", output_dir / "last.pt")
    if improved:
        torch.save(checkpoint, output_dir / "best_overall.pt")
        log_best_checkpoint(epoch, val_metrics, output_dir / "best_overall.pt")


def load_selected_checkpoint(output_dir, raw_model):
    """Restore the validation-selected model for final VAL/TEST evaluation."""
    selected = torch.load(
        output_dir / "best_overall.pt", map_location="cpu", weights_only=True
    )
    raw_model.load_state_dict(selected["model"])
    return selected


def verify_reload_metrics(reloaded_val, selected):
    """Check that reloading preserves the selected checkpoint's validation scores."""
    if any(
        not math.isclose(
            reloaded_val[k], selected["validation_metrics"][k], rel_tol=1e-6, abs_tol=1e-7
        )
        for k in reloaded_val
    ):
        raise RuntimeError("Checkpoint reload validation metrics changed")
