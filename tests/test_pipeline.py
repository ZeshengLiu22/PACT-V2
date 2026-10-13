"""Small numerical tests; real-data runtime verification is train.py --smoke-test."""

import math
from pathlib import Path
import random
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader

from dataset import FullFieldDataset
from train import SmokeModel, compute_normalization, error_totals, load_spatial_weights, split_years


class PipelineTests(unittest.TestCase):
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

    def test_year_splits_and_no_leakage(self):
        for count, sizes in ((1, [1, 0, 0]), (2, [1, 1, 0]), (3, [1, 1, 1]), (36, [22, 7, 7])):
            years = list(range(1979, 1979 + count))
            result = split_years(list(reversed(years)), self.split)
            flattened = sum((result[key] for key in ("train", "val", "test")), [])
            self.assertEqual(sorted(flattened), years)
            self.assertEqual(len(set(flattened)), len(flattened))
            self.assertEqual([len(result[key]) for key in ("train", "val", "test")], sizes)
        result = split_years(list(range(1979, 2015)), self.split)
        self.assertEqual([len(result[key]) for key in ("train", "val", "test")], [22, 7, 7])
        self.assertEqual(result["train"], list(range(1979, 2001)))
        years = list(range(1979, 2015))
        random.Random(17).shuffle(years)
        result = split_years(list(range(1979, 2015)), {**self.split, "shuffle_years": True}, seed=17)
        self.assertEqual(result, {"train": years[:22], "val": years[22:29], "test": years[29:]})
        future = split_years(list(range(2028, 2036)), {**self.split, "future_only": True})
        self.assertEqual(sorted(sum(future.values(), [])), list(range(2031, 2036)))
        with self.assertRaises(ValueError):
            split_years([2000, 2001, 2002], {**self.split, "train_ratio": 0.9, "val_ratio": 0.5})

    def test_global_shuffle_and_simulation_year(self):
        dataset = FullFieldDataset(self.directory, [2000, 2001, 2002], self.mesh)
        self.assertEqual(len(dataset), 18)
        lookup = {float(dataset[i]["forcing"][-1, 0, 0, 0]): i for i in range(len(dataset))}
        loader = DataLoader(dataset, batch_size=4, shuffle=True,
                            generator=torch.Generator().manual_seed(17))
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
        original = FullFieldDataset(self.directory, [2000], self.mesh)[0]
        expected_keys = {"forcing", "tide_boundary", "tide_potential", "tide_time", "surge", "state"}
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

    def test_train_ww_only_normalization_and_selected_history(self):
        train = FullFieldDataset(self.directory, [2000], self.mesh, history_hours=24)
        validation = FullFieldDataset(self.directory, [2001], self.mesh, history_hours=24)
        stats = compute_normalization(train)
        parallel_stats = compute_normalization(train, num_workers=2)
        for key in ("forcing_mean", "forcing_std", "surge_mean", "surge_std"):
            torch.testing.assert_close(stats[key], parallel_stats[key], rtol=0, atol=0)
        forcing = torch.cat([sample["forcing"].double().movedim(1, 0).reshape(5, -1)
                             for sample in train], dim=1)
        surge = torch.cat([sample["surge"][sample["state"] == 0].double() for sample in train])
        torch.testing.assert_close(torch.as_tensor(stats["forcing_mean"]).double(), forcing.mean(1), rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(torch.as_tensor(stats["forcing_std"]).double(), forcing.std(1, unbiased=False), rtol=1e-6, atol=1e-6)
        self.assertAlmostEqual(float(stats["surge_mean"]), float(surge.mean()), places=5)
        self.assertAlmostEqual(float(stats["surge_std"]), float(surge.std(unbiased=False)), places=5)
        self.assertLess(float(stats["forcing_mean"][0]), float(validation[0]["forcing"][:, 0].mean()))
        self.assertTrue(all(torch.isfinite(torch.as_tensor(stats[key])).all()
                            for key in ("forcing_mean", "forcing_std", "surge_mean", "surge_std")))

    def test_spatial_weights_and_invalid_node_order(self):
        uniform = load_spatial_weights(self.mesh, {"mode": "uniform", "weights_file": None})
        torch.testing.assert_close(uniform, torch.ones(3))
        path = self.root / "weights.pt"
        payload = {"node_id": self.mesh["node_id"], "weights": torch.tensor([0.0, 2.0, 4.0])}
        torch.save(payload, path)
        torch.testing.assert_close(load_spatial_weights(self.mesh, {"mode": "uniform", "weights_file": str(path)}),
                                   torch.tensor([0.0, 1.0, 1.0]))
        torch.testing.assert_close(load_spatial_weights(self.mesh, {"mode": "weighted", "weights_file": str(path)}),
                                   payload["weights"])
        torch.save({**payload, "weights": payload["weights"].double()}, path)
        self.assertEqual(load_spatial_weights(self.mesh, {"mode": "weighted", "weights_file": str(path)}).dtype, torch.float64)
        for invalid in ({**payload, "node_id": self.mesh["node_id"].flip(0)},
                        {**payload, "weights": torch.tensor([0.0, -1.0, 2.0])},
                        {**payload, "weights": torch.tensor([0.0, float("nan"), 2.0])}):
            torch.save(invalid, path)
            with self.assertRaises(ValueError):
                load_spatial_weights(self.mesh, {"mode": "weighted", "weights_file": str(path)})
        meta_path = self.directory / "2000" / "meta_200011.pt"
        metadata = torch.load(meta_path, weights_only=True)
        metadata[0]["node_id"] = self.mesh["node_id"].flip(0)
        torch.save(metadata, meta_path)
        with self.assertRaisesRegex(ValueError, "node order"):
            FullFieldDataset(self.directory, [2000], self.mesh)

    def test_nan_safe_weighted_physical_mse_and_zero_exclusion(self):
        target = torch.full((1, 3, 6), float("nan"))
        state = torch.ones((1, 3, 6), dtype=torch.int64)
        target[0, 0, :2], target[0, 1, :2], target[0, 2, :2] = torch.tensor([1., 2.]), torch.tensor([100., 200.]), torch.tensor([2., 0.])
        state[:, :, :2] = 0
        prediction = torch.zeros_like(target, requires_grad=True)
        totals = error_totals(prediction, target, state, torch.tensor([1., 0., 3.]))
        torch.testing.assert_close(totals, torch.tensor([17., 8., 9., 5., 4.], dtype=torch.float64))
        loss = totals[0] / totals[1]
        self.assertEqual(float(loss), 17 / 8)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(float(prediction.grad[:, 1].abs().sum()), 0.0)
        self.assertEqual(float(prediction.grad[state != 0].abs().sum()), 0.0)
        torch.testing.assert_close(prediction.grad[0, 0, :2], torch.tensor([-0.25, -0.5]))
        torch.testing.assert_close(prediction.grad[0, 2, :2], torch.tensor([-1.5, 0.0]))

    def test_population_metric_reduction(self):
        target = torch.zeros(2, 1, 6)
        prediction = torch.ones_like(target)
        prediction[0] = 3.0
        state = torch.zeros_like(target, dtype=torch.int64)
        state[0, :, 1:] = 3
        target[state != 0] = float("nan")
        weights = torch.ones(1)
        first = error_totals(prediction[:1], target[:1], state[:1], weights)
        second = error_totals(prediction[1:], target[1:], state[1:], weights)
        totals = first + second
        torch.testing.assert_close(totals, error_totals(prediction, target, state, weights))
        torch.testing.assert_close(totals[2:], torch.tensor([15., 9., 7.], dtype=torch.float64))
        self.assertAlmostEqual(float((totals[2] / totals[4]).sqrt()), math.sqrt(15 / 7))
        self.assertAlmostEqual(float(totals[3] / totals[4]), 9 / 7)
        batch_average_rmse = ((first[2] / first[4]).sqrt() + (second[2] / second[4]).sqrt()) / 2
        self.assertNotAlmostEqual(float((totals[2] / totals[4]).sqrt()), float(batch_average_rmse))

    def test_gradient_accumulation_matches_whole_windows(self):
        generator = torch.Generator().manual_seed(19)
        forcing = torch.randn(5, 1, 5, 2, 2, generator=generator)
        target = torch.randn(5, 3, 6, generator=generator)
        state = torch.ones(5, 3, 6, dtype=torch.int64)
        for sample, valid_count in enumerate((1, 5, 8, 13, 17)):
            state[sample].view(-1)[:valid_count] = 0
        target[state != 0] = float("nan")
        weights = torch.tensor([0.5, 2.0, 5.0])
        accumulated, whole = SmokeModel(), SmokeModel()
        whole.load_state_dict(accumulated.state_dict())
        optimizers = [torch.optim.Adam(model.parameters(), lr=0.001) for model in (accumulated, whole)]
        # Three microbatches of sizes 2, 2, 1 with accumulation=2: final window is partial.
        for start, end in ((0, 4), (4, 5)):
            for optimizer in optimizers:
                optimizer.zero_grad()
            denominator = torch.tensor(0.0, dtype=torch.float64)
            for offset in range(start, end, 2):
                stop = min(offset + 2, end)
                prediction = accumulated({"forcing": forcing[offset:stop]}, self.mesh)["surge"]
                totals = error_totals(prediction, target[offset:stop], state[offset:stop], weights)
                totals[0].backward()
                denominator += totals[1].detach()
            for parameter in accumulated.parameters():
                parameter.grad.div_(denominator)
            prediction = whole({"forcing": forcing[start:end]}, self.mesh)["surge"]
            totals = error_totals(prediction, target[start:end], state[start:end], weights)
            (totals[0] / totals[1]).backward()
            for actual, expected in zip(accumulated.parameters(), whole.parameters()):
                torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-7)
            for optimizer in optimizers:
                optimizer.step()
            for actual, expected in zip(accumulated.parameters(), whole.parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)

    def test_model_forward_backward_and_checkpoint_optimizer_resume(self):
        dataset = FullFieldDataset(self.directory, [2000], self.mesh)
        batch = next(iter(DataLoader(dataset, batch_size=2)))
        inputs = {key: value for key, value in batch.items() if key not in ("surge", "state")}
        model = SmokeModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        output = model(inputs, self.mesh)["surge"]
        self.assertEqual(output.shape, (2, 3, 6))
        totals = error_totals(output, batch["surge"], batch["state"], torch.ones(3))
        (totals[0] / totals[1]).backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()))
        optimizer.step()
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, model.parameters())))
        path = self.root / "checkpoint.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        restored = SmokeModel()
        restored.load_state_dict(checkpoint["model"])
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.001)
        restored_optimizer.load_state_dict(checkpoint["optimizer"])
        torch.testing.assert_close(restored(inputs, self.mesh)["surge"], model(inputs, self.mesh)["surge"], rtol=0, atol=0)
        for network, active_optimizer in ((model, optimizer), (restored, restored_optimizer)):
            active_optimizer.zero_grad()
            totals = error_totals(network(inputs, self.mesh)["surge"], batch["surge"], batch["state"], torch.ones(3))
            (totals[0] / totals[1]).backward()
            active_optimizer.step()
        torch.testing.assert_close(restored(inputs, self.mesh)["surge"], model(inputs, self.mesh)["surge"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
