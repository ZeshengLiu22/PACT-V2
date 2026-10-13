"""Focused numerics regression tests for the full-field pipeline."""

import math
import unittest

import torch

from emulator.data.dataset import FullFieldDataset
from emulator.data.normalization import compute_normalization
from emulator.models.smoke import SmokeModel
from emulator.training.losses import error_totals, load_spatial_weights
from emulator.training.metrics import overall_metrics
from tests.fixtures import SyntheticDatasetTestCase


class NumericsTests(SyntheticDatasetTestCase):
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
        metrics = overall_metrics(totals)
        self.assertEqual(set(metrics), {"all_rmse", "all_mae"})
        self.assertAlmostEqual(metrics["all_rmse"], math.sqrt(15 / 7))
        self.assertAlmostEqual(metrics["all_mae"], 9 / 7)
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


if __name__ == "__main__":
    unittest.main()
