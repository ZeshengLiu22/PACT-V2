"""Read the existing monthly full-field tensors without copying the shared mesh."""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torch.utils.serialization import config as serialization_config

# Avoid per-storage random archive reads on the shared filesystem (PyTorch >= 2.7).
serialization_config.load.calculate_storage_offsets = True


class FullFieldDataset(Dataset):
    """One global sample index over memory-mapped monthly PyTorch files.

    Each file is opened once at initialization. Linux DataLoader workers inherit
    these mappings, and the operating system pages tensor data in on demand.
    """

    def __init__(self, directory, years, mesh, history_hours=48):
        if type(history_hours) is not int or history_hours not in range(0, 49, 6):
            raise ValueError("history_hours must be one of 0, 6, ..., 48")
        directory = Path(directory)
        self.source = directory.name
        self.years = list(years)
        self.history_steps = history_hours // 6 + 1
        self.files = []
        self.index = []
        self.issue_times = []
        self._months = []
        expected_shapes = None
        dtypes = {
            "forcing": torch.float32,
            "tide_boundary": torch.float64,
            "tide_potential": torch.float64,
            "tide_time": torch.float64,
            "surge": torch.float32,
            "state": torch.int64,
        }
        node_ids = mesh["node_id"]
        if node_ids.dtype != torch.int64 or node_ids.ndim != 1:
            raise ValueError("mesh node_id must be int64 [N]")
        if len(set(self.years)) != len(self.years):
            raise ValueError("Dataset simulation years must be unique")

        for year in self.years:
            year_dir = directory / str(year)
            files = sorted(year_dir.glob("samples_*.pt"))
            if not files:
                raise FileNotFoundError(f"No monthly sample files in {year_dir}")
            expected_meta = {p.name.replace("samples_", "meta_", 1) for p in files}
            actual_meta = {p.name for p in year_dir.glob("meta_*.pt")}
            if expected_meta != actual_meta:
                raise ValueError(
                    f"Unpaired sample/metadata files in {year_dir}: "
                    f"{sorted(expected_meta ^ actual_meta)}"
                )
            for path in files:
                samples = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                metadata = torch.load(
                    path.with_name(path.name.replace("samples_", "meta_", 1)),
                    map_location="cpu", weights_only=True, mmap=True,
                )
                if type(samples) is not list or not samples:
                    raise ValueError(f"{path}: expected a nonempty sample list")
                if type(metadata) is not list or len(metadata) != len(samples):
                    raise ValueError(f"{path}: metadata must be a parallel list")

                checked_node_storages = set()
                month_index = len(self.files)
                for local_index, (sample, meta) in enumerate(zip(samples, metadata)):
                    label = f"{path}, sample {local_index}"
                    if not isinstance(sample, dict) or set(sample) != set(dtypes):
                        raise ValueError(f"{label}: expected the six dynamic tensor keys")
                    for key, dtype in dtypes.items():
                        if not isinstance(sample[key], torch.Tensor) or sample[key].dtype != dtype:
                            raise ValueError(f"{label}: {key} must have dtype {dtype}")
                    shapes = {key: tuple(value.shape) for key, value in sample.items()}
                    if expected_shapes is None:
                        forcing, boundary, potential = (
                            shapes["forcing"], shapes["tide_boundary"], shapes["tide_potential"]
                        )
                        if not (
                            len(forcing) == 4 and forcing[:2] == (9, 5) and min(forcing) > 0
                            and len(boundary) == 3 and boundary[-1] == 5 and min(boundary) > 0
                            and boundary[0] == len(mesh["boundary_node_index"])
                            and len(potential) == 2 and potential[-1] == 5 and min(potential) > 0
                            and shapes["tide_time"] == (6,)
                            and shapes["surge"] == shapes["state"] == (len(node_ids), 6)
                        ):
                            raise ValueError(f"{label}: invalid sample shapes {shapes}")
                        expected_shapes = shapes
                    elif shapes != expected_shapes:
                        raise ValueError(f"{label}: inconsistent sample shapes {shapes}")

                    if not isinstance(meta, dict) or meta.get("source") != self.source or meta.get("simulation_year") != year:
                        raise ValueError(f"{label}: metadata source/simulation_year mismatch")
                    ids = meta.get("node_id")
                    if not isinstance(ids, torch.Tensor) or ids.dtype != torch.int64 or ids.shape != node_ids.shape:
                        raise ValueError(f"{label}: metadata node_id must be int64 [N]")
                    storage = (ids.untyped_storage().data_ptr(), ids.storage_offset(), ids.stride())
                    if storage not in checked_node_storages:
                        if not torch.equal(ids, node_ids):
                            raise ValueError(f"{label}: node order differs from shared mesh")
                        checked_node_storages.add(storage)
                    issue_time = meta.get("issue_time_unix_ns")
                    if type(issue_time) is not int:
                        raise ValueError(f"{label}: issue_time_unix_ns must be an integer")
                    self.index.append((month_index, local_index))
                    self.issue_times.append(issue_time)
                self.files.append(path)
                self._months.append(samples)
                del metadata, meta, ids

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        month_index, local_index = self.index[index]
        sample = self._months[month_index][local_index]
        return {key: value[-self.history_steps:] if key == "forcing" else value
                for key, value in sample.items()}
