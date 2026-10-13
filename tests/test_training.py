"""Focused training regression tests for the full-field pipeline."""

from copy import deepcopy
import unittest

import torch
from torch.utils.data import DataLoader

from emulator.data.dataset import FullFieldDataset
from emulator.models.smoke import SmokeModel
from emulator.training.checkpoints import (
    load_resume,
    load_selected_checkpoint,
    restore_training_state,
    save_checkpoint,
    verify_reload_metrics,
)
from emulator.training.engine import build_optimizer_scheduler
from emulator.training.losses import error_totals
from tests.fixtures import SyntheticDatasetTestCase


class TrainingTests(SyntheticDatasetTestCase):
    """Protect the model interface and complete checkpoint continuation state."""

    def test_model_forward_backward_and_checkpoint_optimizer_resume(self):
        """Restore predictions and the next Adam update exactly after a checkpoint."""
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
        self.assertTrue(all(
            torch.isfinite(parameter.grad).all() for parameter in model.parameters()
        ))
        optimizer.step()
        self.assertTrue(any(
            not torch.equal(old, new) for old, new in zip(before, model.parameters())
        ))

        path = self.root / "checkpoint.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        restored = SmokeModel()
        restored.load_state_dict(checkpoint["model"])
        restored_optimizer = torch.optim.Adam(restored.parameters(), lr=0.001)
        restored_optimizer.load_state_dict(checkpoint["optimizer"])
        torch.testing.assert_close(
            restored(inputs, self.mesh)["surge"],
            model(inputs, self.mesh)["surge"],
            rtol=0,
            atol=0,
        )

        # Matching the next update checks Adam moments as well as model weights.
        for network, active_optimizer in ((model, optimizer), (restored, restored_optimizer)):
            active_optimizer.zero_grad()
            totals = error_totals(
                network(inputs, self.mesh)["surge"],
                batch["surge"],
                batch["state"],
                torch.ones(3),
            )
            (totals[0] / totals[1]).backward()
            active_optimizer.step()
        torch.testing.assert_close(
            restored(inputs, self.mesh)["surge"],
            model(inputs, self.mesh)["surge"],
            rtol=0,
            atol=0,
        )

    def test_checkpoint_schema_state_restore_and_resume_guards(self):
        """Preserve checkpoint fields, continuation state, and experiment identity."""
        config = {
            "experiment": {"seed": 42},
            "data": {"root": str(self.root), "source": "NCEP", "history_hours": 48},
            "split": self.split.copy(),
            "normalization": {"forcing": "mean_std", "target": "mean_std"},
            "spatial_weighting": {"mode": "uniform", "weights_file": None},
            "model": {"name": "smoke"},
            "loss": {"global_mse": True},
            "training": {
                "lr": 0.001,
                "weight_decay": 1e-5,
                "epochs": 4,
                "warmup_epochs": 2,
                "warmup_start_factor": 0.1,
                "min_lr": 1e-6,
                "resume_from": None,
                "device": "cpu",
                "distributed": False,
            },
        }
        splits = {"train": [2000], "val": [2001], "test": [2002]}
        normalization = {
            "forcing_mean": torch.zeros(5, dtype=torch.float64),
            "forcing_std": torch.ones(5, dtype=torch.float64),
            "surge_mean": torch.tensor(0., dtype=torch.float64),
            "surge_std": torch.tensor(1., dtype=torch.float64),
            "forcing_count_per_channel": 108,
            "surge_count": 90,
        }
        metrics = {"all_rmse": 0.5, "all_mae": 0.25}
        batch = next(iter(DataLoader(
            FullFieldDataset(self.directory, [2000], self.mesh), batch_size=2
        )))
        inputs = {key: value for key, value in batch.items() if key not in ("surge", "state")}
        model = SmokeModel()
        optimizer, scheduler = build_optimizer_scheduler(model, config["training"])

        # Save after an update so both Adam and the warmup scheduler have state.
        totals = error_totals(
            model(inputs, self.mesh)["surge"], batch["surge"], batch["state"], torch.ones(3)
        )
        (totals[0] / totals[1]).backward()
        optimizer.step()
        scheduler.step()
        save_checkpoint(
            self.root,
            model,
            optimizer,
            scheduler,
            epoch=0,
            config=config,
            normalization=normalization,
            splits=splits,
            node_id=self.mesh["node_id"],
            val_metrics=metrics,
            best_rmse=0.5,
            best_epoch=0,
            smoke_test=True,
            improved=True,
        )

        saved = torch.load(self.root / "last.pt", map_location="cpu", weights_only=True)
        self.assertEqual(set(saved), {
            "model", "optimizer", "scheduler", "amp_scaler", "epoch",
            "config", "normalization", "split_years", "model_identifier",
            "node_id", "validation_metrics", "best_val_all_rmse",
            "best_epoch", "smoke_test",
        })
        self.assertIsNone(saved["amp_scaler"])
        self.assertEqual(saved["split_years"], splits)
        self.assertEqual(saved["validation_metrics"], metrics)
        torch.testing.assert_close(saved["node_id"], self.mesh["node_id"], rtol=0, atol=0)
        for key in normalization:
            torch.testing.assert_close(
                saved["normalization"][key], normalization[key], rtol=0, atol=0
            )

        # Runtime placement may change on resume; scientific settings may not.
        resume_config = deepcopy(config)
        resume_config["training"].update(
            resume_from=str(self.root / "last.pt"), device="auto", distributed="auto"
        )
        resume = load_resume(resume_config, self.mesh["node_id"], splits, True)
        restored = SmokeModel()
        restored_optimizer, restored_scheduler = build_optimizer_scheduler(
            restored, resume_config["training"]
        )
        self.assertEqual(
            restore_training_state(
                resume, restored, restored_optimizer, restored_scheduler, self.root
            ),
            (1, 0.5, 0),
        )
        self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())
        torch.testing.assert_close(
            restored(inputs, self.mesh)["surge"],
            model(inputs, self.mesh)["surge"],
            rtol=0,
            atol=0,
        )

        # Compare a continued step to catch missing optimizer or scheduler state.
        for network, active_optimizer, active_scheduler in (
            (model, optimizer, scheduler),
            (restored, restored_optimizer, restored_scheduler),
        ):
            active_optimizer.zero_grad()
            totals = error_totals(
                network(inputs, self.mesh)["surge"],
                batch["surge"],
                batch["state"],
                torch.ones(3),
            )
            (totals[0] / totals[1]).backward()
            active_optimizer.step()
            active_scheduler.step()
        torch.testing.assert_close(
            restored(inputs, self.mesh)["surge"],
            model(inputs, self.mesh)["surge"],
            rtol=0,
            atol=0,
        )
        self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())
        for actual, expected in zip(restored_optimizer.state.values(), optimizer.state.values()):
            self.assertEqual(actual.keys(), expected.keys())
            for key in actual:
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)

        selected = load_selected_checkpoint(self.root, restored)
        verify_reload_metrics(metrics, selected)
        with self.assertRaisesRegex(RuntimeError, "validation metrics changed"):
            verify_reload_metrics({"all_rmse": 0.6, "all_mae": 0.25}, selected)

        changes = (
            ("experiment", "seed", 43),
            ("model", "name", "base"),
            ("data", "history_hours", 24),
            ("split", "shuffle_years", True),
            ("normalization", "forcing", "other"),
            ("spatial_weighting", "mode", "weighted"),
            ("loss", "global_mse", False),
            ("training", "lr", 0.002),
        )
        for section, key, value in changes:
            with self.subTest(section=section, key=key):
                changed = deepcopy(resume_config)
                changed[section][key] = value
                with self.assertRaises(ValueError):
                    load_resume(changed, self.mesh["node_id"], splits, True)

        for identity, node_id, split_identity, smoke in (
            ("node_order", self.mesh["node_id"].flip(0), splits, True),
            ("split", self.mesh["node_id"], {**splits, "train": [2001]}, True),
            ("smoke_mode", self.mesh["node_id"], splits, False),
        ):
            with self.subTest(identity=identity):
                with self.assertRaises(ValueError):
                    load_resume(resume_config, node_id, split_identity, smoke)

        # Resume must retain the selected model even when last.pt is available.
        (self.root / "best_overall.pt").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "best_overall"):
            restore_training_state(
                resume, restored, restored_optimizer, restored_scheduler, self.root
            )


if __name__ == "__main__":
    unittest.main()
