"""Small monthly datasets shared by the numerical and integration unit tests."""

from pathlib import Path
import tempfile
import unittest

import torch


class SyntheticDatasetTestCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "NCEP"
        self.mesh = {"node_id": torch.tensor([11, 22, 33]),
                     "pos": torch.zeros(3, 2), "x": torch.zeros(3, 3),
                     "edge_index": torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                     "edge_attr": torch.zeros(4, 3), "face": torch.tensor([[0], [1], [2]]),
                     "boundary_node_index": torch.tensor([0, 2])}
        self.split = {"train_ratio": 0.6, "val_ratio": 0.2, "shuffle_years": False,
                      "future_only": False, "future_year_threshold": 2030}
        for year in (2000, 2001, 2002):
            year_dir = self.directory / str(year)
            year_dir.mkdir(parents=True)
            for month_index, month in enumerate((f"{year}11", f"{year + 1}01")):
                samples, metadata = [], []
                for local_index in range(3):
                    tag = (year - 2000) * 1000 + month_index * 100 + local_index * 10
                    forcing = (torch.arange(9, dtype=torch.float32)[:, None, None, None]
                               + torch.arange(5, dtype=torch.float32)[None, :, None, None] * 100
                               + torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2) / 4 + tag)
                    state = torch.zeros(3, 6, dtype=torch.int64)
                    state[0, 3:] = torch.tensor([1, 2, 3])
                    surge = torch.arange(18, dtype=torch.float32).reshape(3, 6) + tag
                    surge[state != 0] = float("nan")
                    samples.append({"forcing": forcing, "surge": surge, "state": state,
                                    "tide_boundary": torch.full((2, 2, 5), 0.123456789 + tag, dtype=torch.float64),
                                    "tide_potential": torch.full((2, 5), 0.987654321 + tag, dtype=torch.float64),
                                    "tide_time": torch.arange(6, dtype=torch.float64) / 24})
                    metadata.append({"source": "NCEP", "simulation_year": year,
                                     "node_id": self.mesh["node_id"], "issue_time_unix_ns": tag * 10**9})
                torch.save(samples, year_dir / f"samples_{month}.pt")
                torch.save(metadata, year_dir / f"meta_{month}.pt")
        torch.save(self.mesh, self.directory / "mesh.pt")

