"""Existing-trainer and checkpoint regressions on small model inputs.

Real NCEP diagnostics are run separately with scripts.benchmark_models so unit
tests neither require the real data nor recompute full TRAIN normalization.
"""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch
import yaml

from emulator.data.normalization import prepare_normalization
from emulator.models import BaseModel, IGARSS27Model, FullA0Model, FullA1Model
from emulator.training.checkpoints import (
    load_resume, load_selected_checkpoint, restore_training_state,
    save_checkpoint, verify_reload_metrics,
)
from emulator.training.engine import (
    build_optimizer_scheduler, evaluate, model_inputs, train_one_epoch,
)
from emulator.training.config import validate_config


class ModelIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(41)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous_threads)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / "configs/ncep.yaml").read_text()
        )
        self.config["training"].update(
            epochs=3, warmup_epochs=0, batch_size=2, device="cpu",
            amp=False, distributed=False, resume_from=None,
        )
        lat, lon = torch.meshgrid(torch.linspace(3, -3, 5), torch.linspace(-2, 2, 7), indexing="ij")
        self.grid = torch.stack((lon, lat))
        pos = torch.tensor([[-1.5, 2.0], [0.5, -1.0], [1.25, 1.0], [0.0, 0.0]])
        self.mesh = {
            "node_id": torch.tensor([11, 22, 33, 44]),
            "pos": pos.double(),
            "x": torch.cat((pos, torch.eye(4, 3)), dim=1),
            "edge_index": torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        }
        self.normalization = {
            "forcing_mean": torch.tensor([1., 2., 3., -1., 2.], dtype=torch.float64),
            "forcing_std": torch.tensor([2., 3., 4., 2., 3.], dtype=torch.float64),
            "surge_mean": torch.tensor(0.1, dtype=torch.float64),
            "surge_std": torch.tensor(0.2, dtype=torch.float64),
        }

    def test_trainer_and_saved_configuration_reload(self):
        variants = [dict(name="base", variant=variant, hidden_dim=8, temporal_hidden_dim=4)
                    for variant in ("mlp", "gnn", "stgnn")]
        igarss = dict(
            name="igarss27", temporal_encoder="gru", temporal_hidden_dim=4,
            unet_channels=[4, 6, 8], scales=[0, 1, 2], projected_dim=4,
            mesh_hidden_dim=4, mesh_encoder_layers=2, fusion="concat",
            decoder="mlp", decoder_hidden_dim=8, decoder_gnn_layers=2,
        )
        variants.extend([
            igarss, dict(igarss, fusion="adaptive"),
            dict(igarss, fusion="adaptive", decoder="gnn"),
            dict(igarss, scales=[0]),
            *[dict(igarss, temporal_encoder=name) for name in ("lstm", "temporal_cnn", "tcn")],
        ])
        device = torch.device("cpu")
        norm = prepare_normalization(self.normalization, device)
        splits = {"train": [2000], "val": [2001], "test": [2002]}
        for hours in (48, 0):
            for variant in variants:
                with self.subTest(hours=hours, model=variant):
                    config = deepcopy(self.config)
                    config["model"] = variant
                    config["data"]["history_hours"] = hours
                    validate_config(config)
                    forcing = torch.randn(2, hours // 6 + 1, 5, 5, 7)
                    forcing[:, :, 3:5] = self.grid
                    batch = dict(
                        forcing=forcing, tide_boundary=torch.zeros(2, 2, 1, 5),
                        tide_potential=torch.zeros(2, 1, 5), tide_time=torch.zeros(2, 6),
                        surge=torch.randn(2, 4, 6), state=torch.zeros(2, 4, 6, dtype=torch.int64),
                    )
                    batch["state"][0, 0, 0] = 1
                    batch["surge"][0, 0, 0] = float("nan")
                    if variant["name"] == "base":
                        model = BaseModel(variant, self.mesh, self.grid, history_hours=hours)
                    else:
                        model = IGARSS27Model(variant, self.mesh, self.grid)
                    optimizer, scheduler = build_optimizer_scheduler(model, config["training"])
                    before = [parameter.detach().clone() for parameter in model.parameters()]
                    runtime = {key: [] for key in (
                        "forward_seconds", "backward_seconds", "optimizer_seconds",
                        "batch_loading_seconds", "training_batch_sizes",
                    )}
                    totals, _ = train_one_epoch(
                        model, model, [batch], None, torch.Generator(), optimizer, scheduler,
                        self.mesh, norm, torch.ones(4), device, False, config["training"],
                        seed=42, epoch=0, world=1, smoke_test=False, runtime=runtime,
                    )
                    self.assertTrue(torch.isfinite(totals).all())
                    self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, model.parameters())))
                    self.assertEqual(runtime["training_batch_sizes"], [2])
                    inputs = model_inputs(batch, norm, device)
                    self.assertNotIn("surge", inputs)
                    self.assertNotIn("state", inputs)
                    expected = model(inputs, self.mesh)["surge"].detach()
                    self.assertEqual(tuple(expected.shape), (2, 4, 6))
                    metrics, _ = evaluate(model, [batch], self.mesh, norm, torch.ones(4), device, False)
                    save_checkpoint(
                        self.root, model, optimizer, scheduler, epoch=0, config=config,
                        normalization=self.normalization, splits=splits, node_id=self.mesh["node_id"],
                        val_metrics=metrics, best_rmse=metrics["all_rmse"], best_epoch=0,
                        smoke_test=True, improved=True,
                    )
                    resume_config = deepcopy(config)
                    resume_config["training"]["resume_from"] = str(self.root / "last.pt")
                    saved = load_resume(resume_config, self.mesh["node_id"], splits, True)
                    # Reconstruct architecture and history dimensions from checkpoint configuration.
                    saved_model = saved["config"]["model"]
                    saved_history = saved["config"]["data"]["history_hours"]
                    if saved_model["name"] == "base":
                        restored = BaseModel(saved_model, self.mesh, self.grid, history_hours=saved_history)
                    else:
                        restored = IGARSS27Model(saved_model, self.mesh, self.grid)
                    resumed_optimizer, resumed_scheduler = build_optimizer_scheduler(
                        restored, saved["config"]["training"],
                    )
                    self.assertEqual(restore_training_state(
                        saved, restored, resumed_optimizer, resumed_scheduler, self.root,
                    ), (1, metrics["all_rmse"], 0))
                    torch.testing.assert_close(restored(inputs, self.mesh)["surge"], expected, rtol=0, atol=0)
                    self.assertEqual(resumed_scheduler.state_dict(), scheduler.state_dict())
                    for actual, original in zip(resumed_optimizer.state.values(), optimizer.state.values()):
                        for key in original:
                            torch.testing.assert_close(actual[key], original[key], rtol=0, atol=0)
                    selected = load_selected_checkpoint(self.root, restored)
                    reloaded_metrics, _ = evaluate(
                        restored, [batch], self.mesh, norm, torch.ones(4), device, False,
                    )
                    verify_reload_metrics(reloaded_metrics, selected)
                    wrong_history = deepcopy(resume_config)
                    wrong_history["data"]["history_hours"] = 24
                    with self.assertRaisesRegex(ValueError, "configuration mismatch: data"):
                        load_resume(wrong_history, self.mesh["node_id"], splits, True)

    def test_full_interfaces_fail_explicitly(self):
        for model_type, name in ((FullA0Model, "full_v1"), (FullA1Model, "full_v2")):
            with self.subTest(model=model_type.__name__):
                with self.assertRaises(NotImplementedError):
                    model_type()
                config = deepcopy(self.config)
                config["model"] = {"name": name}
                with self.assertRaises(NotImplementedError):
                    validate_config(config)


if __name__ == "__main__":
    unittest.main()
