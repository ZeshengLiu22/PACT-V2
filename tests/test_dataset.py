"""Focused dataset regression tests for the full-field pipeline."""

import random
import unittest

import torch
from torch.utils.data import DataLoader

from emulator.data.dataset import FullFieldDataset
from emulator.data.splits import split_years
from tests.fixtures import SyntheticDatasetTestCase


class DatasetTests(SyntheticDatasetTestCase):
    """Protect sample identity, simulation-year boundaries, and input schemas."""

    def test_year_splits_and_no_leakage(self):
        """Keep years disjoint and preserve seeded and future-only selection."""
        for count, sizes in (
            (1, [1, 0, 0]),
            (2, [1, 1, 0]),
            (3, [1, 1, 1]),
            (36, [22, 7, 7]),
        ):
            years = list(range(1979, 1979 + count))
            result = split_years(list(reversed(years)), self.split)
            flattened = sum((result[key] for key in ("train", "val", "test")), [])

            self.assertEqual(sorted(flattened), years)
            self.assertEqual(len(set(flattened)), len(flattened))
            self.assertEqual(
                [len(result[key]) for key in ("train", "val", "test")], sizes
            )

        result = split_years(list(range(1979, 2015)), self.split)
        self.assertEqual(
            [len(result[key]) for key in ("train", "val", "test")], [22, 7, 7]
        )
        self.assertEqual(result["train"], list(range(1979, 2001)))

        years = list(range(1979, 2015))
        random.Random(17).shuffle(years)
        result = split_years(
            list(range(1979, 2015)), {**self.split, "shuffle_years": True}, seed=17
        )
        self.assertEqual(
            result, {"train": years[:22], "val": years[22:29], "test": years[29:]}
        )

        future = split_years(list(range(2028, 2036)), {**self.split, "future_only": True})
        self.assertEqual(sorted(sum(future.values(), [])), list(range(2031, 2036)))
        with self.assertRaises(ValueError):
            split_years(
                [2000, 2001, 2002],
                {**self.split, "train_ratio": 0.9, "val_ratio": 0.5},
            )

    def test_global_shuffle_and_simulation_year(self):
        """Shuffle across files while assigning January to its simulation year."""
        dataset = FullFieldDataset(self.directory, [2000, 2001, 2002], self.mesh)
        self.assertEqual(len(dataset), 18)
        lookup = {float(dataset[i]["forcing"][-1, 0, 0, 0]): i for i in range(len(dataset))}
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            generator=torch.Generator().manual_seed(17),
        )

        order, batches = [], []
        for batch in loader:
            indices = [lookup[float(value)] for value in batch["forcing"][:, -1, 0, 0, 0]]
            order.extend(indices)
            batches.append([dataset.files[dataset.index[i][0]] for i in indices])

        self.assertEqual(sorted(order), list(range(len(dataset))))
        self.assertNotEqual(order, list(range(len(dataset))))
        self.assertTrue(any(len(set(paths)) > 1 for paths in batches))
        self.assertTrue(any(len({path.parent.name for path in paths}) > 1 for paths in batches))

        single_year = FullFieldDataset(self.directory, [2000], self.mesh)
        self.assertEqual(len(single_year), 6)
        self.assertEqual({path.parent.name for path in single_year.files}, {"2000"})
        self.assertIn("samples_200101.pt", {path.name for path in single_year.files})

    def test_history_slices_and_shared_mesh(self):
        """Keep the newest forcing steps and preserve tides, NaNs, and sample keys."""
        original = FullFieldDataset(self.directory, [2000], self.mesh)[0]
        # Geometry stays shared: it must not become a per-sample batch field.
        expected_keys = {
            "forcing", "tide_boundary", "tide_potential", "tide_time", "surge", "state"
        }

        for hours, steps in ((48, 9), (42, 8), (36, 7), (24, 5), (0, 1)):
            dataset = FullFieldDataset(self.directory, [2000], self.mesh, history_hours=hours)
            self.assertEqual(dataset.history_steps, steps)
            torch.testing.assert_close(dataset[0]["forcing"], original["forcing"][-steps:])

            batch = next(iter(DataLoader(dataset, batch_size=2)))
            self.assertEqual(set(batch), expected_keys)
            self.assertEqual(batch["forcing"].shape, (2, steps, 5, 2, 2))
            for name in ("tide_boundary", "tide_potential", "tide_time"):
                self.assertEqual(batch[name].dtype, torch.float64)
                torch.testing.assert_close(batch[name][0], original[name], rtol=0, atol=0)
            self.assertTrue(torch.isnan(batch["surge"][batch["state"] != 0]).all())

        with self.assertRaises(ValueError):
            FullFieldDataset(self.directory, [2000], self.mesh, history_hours=25)


if __name__ == "__main__":
    unittest.main()
