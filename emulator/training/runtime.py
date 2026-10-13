"""Device/DDP setup, reproducibility settings, and memory measurements."""

from datetime import timedelta
import os
from pathlib import Path
import random
import resource
import threading

import torch
from torch import distributed as dist


def initialize_runtime(config):
    train = config["training"]
    seed = config["experiment"]["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(train["deterministic"])
    torch.backends.cuda.matmul.allow_tf32 = train["tf32"]
    torch.backends.cudnn.allow_tf32 = train["tf32"]
    torch.backends.cudnn.benchmark = not train["deterministic"]
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if train["distributed"] not in ("auto", True, False):
        raise ValueError("distributed must be auto or a YAML boolean")
    if world > 1 and train["distributed"] is False:
        raise ValueError("torchrun WORLD_SIZE > 1 but distributed is false")
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if train["device"] == "auto" else train["device"])
    if device.type == "cuda":
        if world > 1:
            device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        elif device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    if world > 1:
        # The initial full TRAIN statistics pass can exceed the NCCL default timeout.
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(hours=1))
    use_amp = train["amp"] and device.type == "cuda"
    if use_amp and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requested bf16 AMP is unsupported by this CUDA device")
    print(f"rank={rank}/{world} device={device} torch={torch.__version__} AMP={use_amp}", flush=True)
    return device, world, rank, use_amp


def monitor_memory(stop, report):
    """Sample process-tree PSS: shared file/IPC pages are apportioned once."""
    while not stop.is_set():
        pids = [os.getpid()]
        pids += [int(p) for p in Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children").read_text().split()]
        pss_kib = 0
        for pid in pids:
            try:
                lines = Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines()
            except (FileNotFoundError, ProcessLookupError):
                continue  # A DataLoader worker exited between enumeration and measurement.
            pss_kib += sum(int(line.split()[1]) for line in lines if line.startswith("Pss:"))
        report["peak_cpu_process_tree_pss_gib"] = max(report.get("peak_cpu_process_tree_pss_gib", 0), pss_kib / 2**20)
        stop.wait(0.5)


def start_memory_monitor():
    memory_stop = threading.Event()
    memory_report = {}
    memory_thread = threading.Thread(target=monitor_memory, args=(memory_stop, memory_report), daemon=True)
    memory_thread.start()
    return memory_stop, memory_thread, memory_report


def finish_runtime(runtime, device, memory_stop, memory_thread, memory_report):
    memory_stop.set()
    memory_thread.join()
    runtime.update(memory_report)
    runtime["peak_cpu_main_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    runtime["peak_gpu_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
    loading = runtime["batch_loading_seconds"]
    runtime["initial_batch_loading_seconds"] = loading[0] if loading else None
    runtime["repeated_shuffled_batch_loading_seconds"] = loading[1:]
    runtime["prefetched_batch_availability_samples_per_second"] = (sum(runtime["training_batch_sizes"][1:]) / sum(loading[1:])) if len(loading) > 1 else None


def cleanup_runtime():
    if dist.is_initialized():
        dist.destroy_process_group()
