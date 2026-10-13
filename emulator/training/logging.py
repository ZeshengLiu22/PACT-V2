"""Console and persistent training messages, separate from numerical reporting."""

import logging
from pathlib import Path
import sys


logger = logging.getLogger("pact.training")
_handlers = []


def close_logging():
    """Flush and close this module's handlers, leaving unrelated loggers alone."""
    for handler in _handlers:
        logger.removeHandler(handler)
        handler.flush()
        handler.close()
    _handlers.clear()


def configure_logging(rank=0, output_dir=None, resume=False):
    """Configure console output and, after run validation, rank-zero train.log.

    Reconfiguration replaces owned handlers, so adding the file after indexing
    does not duplicate console messages. The caller resolves and validates the
    run directory before requesting a file handler; nonzero ranks never open it.
    """
    close_logging()
    level = logging.INFO if rank == 0 else logging.ERROR
    logger.setLevel(level)
    logger.propagate = False
    rank_context = "" if rank == 0 else f"[rank {rank}] "
    formatter = logging.Formatter(
        "[%(asctime)s] " + rank_context + "%(message)s",
        datefmt="%Y-%m-%d|%H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout if rank == 0 else sys.stderr)
    console.setFormatter(formatter)
    console.setLevel(level)
    logger.addHandler(console)
    _handlers.append(console)

    if rank == 0 and output_dir is not None:
        persistent = logging.FileHandler(
            Path(output_dir) / "train.log",
            mode="a" if resume else "w",
            encoding="utf-8",
        )
        persistent.setFormatter(formatter)
        persistent.setLevel(logging.INFO)
        logger.addHandler(persistent)
        _handlers.append(persistent)
        if resume:
            logger.info("================ RESUMING TRAINING ================")
    return logger


def log_initialization(
    config, runtime, splits, output_dir, use_amp, forecast_horizon, smoke_test
):
    """Summarize already indexed data and configured execution without rescanning."""
    experiment, data, train = (
        config["experiment"], config["data"], config["training"]
    )
    logger.info("Starting experiment: %s", experiment["name"])
    logger.info(
        "Source: %s | Model: %s | Seed: %s",
        data["source"], config["model"]["name"], experiment["seed"],
    )
    logger.info(
        "Device: %s | Distributed processes: %d | AMP: %s | Configured AMP dtype: %s",
        runtime["device"], runtime["world_size"],
        "enabled" if use_amp else "disabled", train["amp_dtype"],
    )
    logger.info(
        "History: %s h | Forecast horizon: %s steps | Mesh: %s nodes, %s directed edges",
        data["history_hours"], forecast_horizon,
        f'{runtime["mesh_nodes"]:,}', f'{runtime["mesh_directed_edges"]:,}',
    )
    for name in ("train", "val", "test"):
        years = ", ".join(str(year) for year in splits[name]) or "none"
        logger.info(
            "%s years: %s | Samples: %s",
            name.upper(), years, f'{runtime["sample_counts"][name]:,}',
        )
    logger.info(
        "Dataset indexing completed in %.2fs", runtime["dataset_initialization_seconds"]
    )
    logger.info(
        "Batch size: %d | Gradient accumulation: %d | Configured LR: %.2e | Epochs: %d",
        train["batch_size"], train["grad_accum_steps"], train["lr"], train["epochs"],
    )
    logger.info(
        "Optimizer: %s | Scheduler: %s | Output directory: %s",
        train["optimizer"], train["scheduler"], output_dir,
    )
    logger.info(
        "TRAIN objective: spatially weighted GlobalMSE (m²) on eligible WW node-hours; "
        "VAL/TEST: unweighted AllRMSE and AllMAE (m) on eligible WW node-hours"
    )
    if smoke_test:
        logger.info(
            "Smoke test: full TRAIN normalization; up to four training batches; "
            "seeded subset of at most two batches per held-out split"
        )
    else:
        logger.info("Evaluation scope: complete held-out splits")


def log_model_parameters(model):
    """Report counts from the instantiated model, including frozen parameters."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    logger.info("Total parameters: %s | Trainable parameters: %s", f"{total:,}", f"{trainable:,}")


def log_epoch(row, epochs, elapsed):
    """Render the unchanged JSONL row with a one-based epoch and physical units."""
    validation = row["validation"]
    val_message = (
        f'Val AllRMSE: {validation["all_rmse"]:.6f} m | '
        f'Val AllMAE: {validation["all_mae"]:.6f} m'
        if validation is not None else "Val: not scheduled"
    )
    logger.info(
        "Epoch %03d/%d | Train GlobalMSE: %.6f m² | %s | LR: %.2e | Time: %.1fs",
        row["epoch"] + 1, epochs, row["train_global_mse"],
        val_message, row["lr"], elapsed,
    )


def log_batch_progress(
    epoch, epochs, batch, batches, global_mse, elapsed, smoke_test=False, *, world=1
):
    """Format infrequent progress without transferring or reducing tensors here.

    Under DDP the caller's running totals describe rank zero only. Full epoch
    summaries use the existing globally reduced totals instead.
    """
    label = "Batch GlobalMSE" if smoke_test else "Cumulative GlobalMSE"
    if world > 1:
        label = "Rank 0 local " + label
    metric = "unavailable (no eligible WW node-hours)" if global_mse is None else f"{global_mse:.6f} m²"
    logger.info(
        "%sEpoch %03d/%d | Batch %03d/%d | %s: %s | Elapsed: %.1fs",
        "Smoke | " if smoke_test else "", epoch + 1, epochs,
        batch, batches, label, metric, elapsed,
    )


def log_best_checkpoint(epoch, val_metrics, path):
    """Announce an already saved validation-selected checkpoint."""
    logger.info(
        "[Best overall] Epoch %03d | Val AllRMSE: %.6f m | Checkpoint: %s",
        epoch + 1, val_metrics["all_rmse"], path,
    )


def log_resume(resume, path):
    """Report restored checkpoint state using human-readable epoch numbers."""
    previous_best = resume["best_epoch"]
    logger.info("Checkpoint: %s", path)
    logger.info(
        "Restored epoch: %03d | Starting epoch: %03d",
        resume["epoch"] + 1, resume["epoch"] + 2,
    )
    if previous_best is None:
        logger.info("Previous best epoch: none | Previous best VAL AllRMSE: unavailable")
    else:
        logger.info(
            "Previous best epoch: %03d | Previous best VAL AllRMSE: %.6f m",
            previous_best + 1, resume["best_val_all_rmse"],
        )
    logger.info("Normalization: reused")


def log_loading_benchmark(report):
    """Keep detailed smoke loading diagnostics readable and persistent."""
    repeated = ", ".join(f"{seconds:.6f}s" for seconds in report["repeated_batch_seconds"])
    logger.info(
        "Smoke random-access DataLoader benchmark | Batch sizes: %s | First batch: %.6fs "
        "| Repeated batches: %s | Total throughput: %.3f samples/s",
        report["batch_sizes"], report["first_batch_seconds"],
        repeated or "none", report["total_samples_per_second"],
    )
    throughput = report["repeated_samples_per_second"]
    if throughput is not None:
        logger.info("Smoke repeated-loading throughput: %.3f samples/s", throughput)
    else:
        logger.info("Smoke repeated-loading throughput: unavailable (one batch)")


def log_final(summary, output_dir, wall_seconds, training_seconds):
    """Display final metrics and existing measurements without changing summary.json."""
    logger.info("FINAL EVALUATION")
    logger.info(
        "Selected best checkpoint: %s | Selected best epoch: %03d",
        Path(output_dir) / summary["selected_checkpoint"], summary["selected_epoch"] + 1,
    )
    logger.info(
        "Evaluation scope: %s%s",
        "smoke-test " if summary["smoke_test"] else "", summary["evaluation_scope"],
    )
    for split in ("validation", "test"):
        metrics = summary[split]
        name = "VAL" if split == "validation" else "TEST"
        if metrics is not None:
            logger.info(
                "Final %s AllRMSE: %.6f m | AllMAE: %.6f m",
                name, metrics["all_rmse"], metrics["all_mae"],
            )
    logger.info("TEST status: %s", summary["test_status"])
    logger.info("Training time: %.2fs | Total wall time: %.2fs", training_seconds, wall_seconds)
    runtime = summary["runtime"]
    for key, label in (
        ("dataset_initialization_seconds", "Dataset indexing"),
        ("normalization_seconds", "Normalization preparation"),
        ("dataloader_initialization_seconds", "DataLoader initialization"),
        ("initial_batch_loading_seconds", "Initial training batch loading"),
    ):
        value = runtime.get(key)
        if value is not None:
            logger.info("%s: %.3fs", label, value)
    for key, label in (
        ("forward_seconds", "Forward and loss"),
        ("backward_seconds", "Backward"),
        ("optimizer_seconds", "Optimizer update"),
    ):
        timings = runtime.get(key, [])
        if timings:
            logger.info("Mean %s time: %.6fs", label.lower(), sum(timings) / len(timings))
    for split, label in (("validation", "VAL"), ("test", "TEST")):
        report = runtime.get(split)
        if report is not None:
            logger.info(
                "Measured %s evaluation: %s samples in %.3fs | %.3f samples/s",
                label, report["samples"], report["seconds"], report["samples_per_second"],
            )
    for key, label in (
        ("peak_cpu_process_tree_pss_gib", "Peak CPU process-tree PSS (sampled)"),
        ("peak_cpu_main_rss_gib", "Peak CPU main-process RSS"),
        ("peak_gpu_allocated_gib", "Peak GPU allocated memory"),
    ):
        value = runtime.get(key)
        if value is not None:
            logger.info("%s: %.3f GiB", label, value)
    if runtime.get("checkpoint_reload_succeeded"):
        logger.info("Selected checkpoint reload: verified")
    logger.info("Results: %s | Structured summary: %s", output_dir, Path(output_dir) / "summary.json")
