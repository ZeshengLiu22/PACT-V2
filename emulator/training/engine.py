"""Training and evaluation with complete weighted accumulation windows."""

from contextlib import nullcontext
import time

import torch
from torch import distributed as dist

from emulator.training.losses import error_totals
from emulator.training.metrics import overall_metrics
from emulator.training.smoke import record_shuffle, verify_batch_order


def model_inputs(batch, normalization, device):
    """Move only predictor fields; labels never enter the model."""
    inputs = {k: batch[k].to(device, non_blocking=True)
              for k in ("forcing", "tide_boundary", "tide_potential", "tide_time")}
    inputs["forcing"] = ((inputs["forcing"] - normalization["forcing_mean"])
                         / normalization["forcing_std"])
    return inputs


def build_optimizer_scheduler(model, train):
    """Preserve Adam and optional linear-warmup/cosine scheduling defaults."""
    optimizer = torch.optim.Adam(model.parameters(), lr=train["lr"], weight_decay=train["weight_decay"])
    warmup = train["warmup_epochs"]
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train["epochs"] - warmup, eta_min=train["min_lr"])
    if warmup:
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=train["warmup_start_factor"], total_iters=warmup), cosine],
            milestones=[warmup])
    else:
        scheduler = cosine
    return optimizer, scheduler


def evaluate(model, loader, mesh, normalization, weights, device, use_amp):
    model.eval()
    totals = torch.zeros(5, device=device, dtype=torch.float64)
    samples = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            inputs = model_inputs(batch, normalization, device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                output = model(inputs, mesh)["surge"]
            prediction = output.float() * normalization["surge_std"] + normalization["surge_mean"]
            totals += error_totals(prediction, batch["surge"].to(device),
                                   batch["state"].to(device), weights)
            samples += len(batch["surge"])
    if dist.is_initialized():
        dist.all_reduce(totals)
        count = torch.tensor(samples, device=device, dtype=torch.int64)
        dist.all_reduce(count)
        samples = count.item()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    metrics = overall_metrics(totals)
    return metrics, {"seconds": seconds, "samples": samples,
                     "samples_per_second": samples / seconds,
                     "eligible_node_hours": int(totals[4].item())}


def train_one_epoch(model, raw_model, train_loader, sampler, generator,
                    optimizer, scheduler, mesh, normalization, weights,
                    device, use_amp, train, seed, epoch, world, smoke_test, runtime):
    """Sum gradients, then scale once per complete global supervision window."""
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)
    else:
        generator.manual_seed(seed + epoch)
    if smoke_test:
        first_indices = record_shuffle(train_loader.dataset, sampler, seed, epoch,
                                       train["batch_size"], runtime)
    total = torch.zeros(5, device=device, dtype=torch.float64)
    denominator = torch.zeros((), device=device, dtype=torch.float64)
    optimizer.zero_grad(set_to_none=True)
    steps = min(4, len(train_loader)) if smoke_test else len(train_loader)
    tick = time.perf_counter()
    iterator = iter(train_loader)
    for step in range(steps):
        batch = next(iterator)
        runtime["batch_loading_seconds"].append(time.perf_counter() - tick)
        runtime["training_batch_sizes"].append(len(batch["forcing"]))
        if smoke_test:
            verify_batch_order(batch, train_loader.dataset, first_indices, step, train["batch_size"])
        inputs = model_inputs(batch, normalization, device)
        target, state = batch["surge"].to(device), batch["state"].to(device)
        update = (step + 1) % train["grad_accum_steps"] == 0 or step + 1 == steps
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        with model.no_sync() if world > 1 and not update else nullcontext():
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                output = model(inputs, mesh)["surge"]
            prediction = output.float() * normalization["surge_std"] + normalization["surge_mean"]
            batch_totals = error_totals(prediction, target, state, weights)
            # Tail/GT-aligned Episode Peak losses join Global MSE here after the Episode Audit.
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            runtime["forward_seconds"].append(time.perf_counter() - tick)
            tick = time.perf_counter()
            batch_totals[0].backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            runtime["backward_seconds"].append(time.perf_counter() - tick)
        total += batch_totals.detach()
        denominator += batch_totals[1].detach()
        if update:
            tick = time.perf_counter()
            if world > 1:
                dist.all_reduce(denominator)
            if denominator <= 0:
                raise ValueError("Accumulation window has no positive-weight WW supervision")
            # DDP averages gradients; divide once by the complete window's global weight count.
            for parameter in raw_model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(world / denominator)
                    if not torch.isfinite(parameter.grad).all():
                        raise FloatingPointError("Nonfinite training gradients")
            if train["max_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), train["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            denominator.zero_()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            runtime["optimizer_seconds"].append(time.perf_counter() - tick)
        if smoke_test:
            print(f"epoch={epoch} batch={step + 1}/{steps} physical_mse={(batch_totals[0] / batch_totals[1]).item():.6f}", flush=True)
        tick = time.perf_counter()
    if world > 1:
        dist.all_reduce(total)
    if smoke_test:
        runtime["global_shuffle_verified"] = True
        del iterator, train_loader
    lr_used = optimizer.param_groups[0]["lr"]
    scheduler.step()
    return total, lr_used
