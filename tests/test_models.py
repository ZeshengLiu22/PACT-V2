"""Small model regressions, including every supported six-hour history."""

from io import BytesIO
from itertools import combinations, product
import unittest

import torch
from torch import nn

from emulator.models.base import BaseModel
from emulator.models.common import BilinearProjector, SpatialGNN, mean_adjacency
from emulator.models.full_a0 import FullA0Model
from emulator.models.full_a1 import FullA1Model
from emulator.models.igarss27 import IGARSS27Model, SpatialUNet, TemporalEncoder


HISTORIES = (48, 42, 36, 24, 0, 6, 12, 18, 30)
SCALE_SETS = tuple(
    subset for count in (1, 2, 3) for subset in combinations((0, 1, 2), count)
)
IGARSS_CONFIG = {
    "name": "igarss27",
    "temporal_hidden_dim": 4,
    "unet_channels": [4, 6, 8],
    "projected_dim": 3,
    "mesh_hidden_dim": 4,
    "mesh_encoder_layers": 2,
    "decoder_hidden_dim": 6,
    "decoder_gnn_layers": 2,
}


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous_threads)

    def setUp(self):
        torch.manual_seed(17)
        lat, lon = torch.meshgrid(
            torch.linspace(5, -1, 9), torch.linspace(2, 8, 11), indexing="ij"
        )
        self.grid = torch.stack((lon, lat))
        self.mesh = {
            "node_id": torch.tensor([19, 2, 44, 7, 35]),
            "pos": torch.tensor([[7., 0.], [2., 5.], [4., 2.], [8., -1.], [3., 4.]]),
            "edge_index": torch.tensor([
                [0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
                [1, 0, 2, 1, 3, 2, 4, 3, 0, 4],
            ]),
        }
        self.mesh["x"] = torch.cat((self.mesh["pos"], torch.rand(5, 3)), dim=1)
        self.forcing = torch.randn(2, 9, 5, 9, 11)
        self.forcing[:, :, 3:] = self.grid

    def assert_finite_backward(self, model, history_hours):
        forcing = self.forcing[:, -(history_hours // 6 + 1):].clone().requires_grad_()
        model.zero_grad(set_to_none=True)
        output = model({"forcing": forcing}, self.mesh)["surge"]
        self.assertEqual(output.shape, (2, 5, 6))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        gradients = []
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                # Coarse-only projection keeps the same U-Net architecture;
                # finer decoder stages cannot affect the selected features.
                self.assertIsInstance(model, IGARSS27Model)
                unused = tuple(f"spatial_encoder.decoders.{scale}." for scale in range(min(model.scales)))
                self.assertTrue(name.startswith(unused), f"Unexpected unused parameter: {name}")
            else:
                gradients.append(parameter.grad)
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertTrue(any(torch.count_nonzero(g) for g in gradients))
        self.assertIsNotNone(forcing.grad)
        self.assertTrue(torch.isfinite(forcing.grad).all())

    def test_base_all_histories_forward_backward_and_eager_dimensions(self):
        for variant, history in product(("mlp", "gnn", "stgnn"), HISTORIES):
            with self.subTest(variant=variant, history_hours=history):
                model = BaseModel(
                    {"variant": variant, "hidden_dim": 6, "temporal_hidden_dim": 4},
                    self.mesh, self.grid, history_hours=history,
                )
                self.assertFalse(any(
                    isinstance(p, nn.parameter.UninitializedParameter)
                    for p in model.parameters()
                ))
                if variant in ("mlp", "gnn"):
                    first_linear = next(m for m in model.modules() if isinstance(m, nn.Linear))
                    self.assertEqual(first_linear.in_features, 3 * (history // 6 + 1) + 5)
                self.assert_finite_backward(model, history)

    def test_igarss_all_variants_and_histories_forward_backward(self):
        variants = product(
            ("gru", "lstm", "temporal_cnn", "tcn"),
            ("concat", "adaptive"), ("mlp", "gnn"), SCALE_SETS,
        )
        for temporal, fusion, decoder, scales in variants:
            config = dict(
                IGARSS_CONFIG, temporal_encoder=temporal, fusion=fusion,
                decoder=decoder, scales=list(scales),
            )
            model = IGARSS27Model(config, self.mesh, self.grid)
            for history in HISTORIES:
                with self.subTest(
                    temporal=temporal, fusion=fusion, decoder=decoder,
                    scales=scales, history_hours=history,
                ):
                    self.assert_finite_backward(model, history)

    def test_stgnn_accepts_shorter_sequences_without_reinitialization(self):
        model = BaseModel(
            {"variant": "stgnn", "hidden_dim": 6, "temporal_hidden_dim": 4},
            self.mesh, self.grid, history_hours=48,
        )
        for history in HISTORIES:
            with self.subTest(history_hours=history):
                self.assert_finite_backward(model, history)

    def test_temporal_encoders_preserve_cells_and_batch_items(self):
        for kind in ("gru", "lstm", "temporal_cnn", "tcn"):
            encoder = TemporalEncoder(kind, 4)
            self.assertFalse(any(isinstance(m, (nn.Conv2d, nn.Conv3d)) for m in encoder.modules()))
            for history in HISTORIES:
                with self.subTest(kind=kind, history_hours=history):
                    values = torch.randn(2, history // 6 + 1, 3, 3, 5, requires_grad=True)
                    encoded = encoder(values)
                    self.assertEqual(encoded.shape, (2, 4, 3, 5))
                    encoded[0, :, 1, 2].square().sum().backward()
                    elsewhere = values.grad.clone()
                    elsewhere[0, :, :, 1, 2] = 0
                    self.assertEqual(torch.count_nonzero(elsewhere).item(), 0)
                    changed = values.detach().clone()
                    changed[0, :, :, 1, 2] += 3
                    difference = encoder(changed) - encoded.detach()
                    difference[0, :, 1, 2] = 0
                    torch.testing.assert_close(difference, torch.zeros_like(difference))

    def test_tcn_convolutions_are_causal(self):
        encoder = TemporalEncoder("tcn", 4)
        convolutions = [m for m in encoder.modules() if isinstance(m, nn.Conv1d)]
        self.assertTrue(convolutions)
        self.assertTrue(any(m.dilation[0] > 1 for m in convolutions))
        observed = []
        hooks = [m.register_forward_hook(
            lambda module, inputs, output: observed.append(output.detach().clone())
        ) for m in convolutions]
        try:
            values = torch.randn(1, 9, 3, 3, 5)
            encoder(values)
            full = observed[:]
            observed.clear()
            encoder(values[:, :4])
            self.assertEqual(len(full), len(observed))
            for full_sequence, prefix in zip(full, observed):
                torch.testing.assert_close(full_sequence[..., :4], prefix[..., :4])
        finally:
            for hook in hooks:
                hook.remove()
        # A causal stack must still reach the oldest frame of the 48 h window.
        # Positive weights keep ReLU from obscuring its actual receptive field.
        with torch.no_grad():
            for convolution in convolutions:
                convolution.weight.fill_(0.1)
                convolution.bias.fill_(0.1)
        for steps in (9, 8):
            values = torch.ones(1, steps, 3, 3, 5, requires_grad=True)
            encoder(values).sum().backward()
            self.assertTrue((values.grad[:, 0] > 0).all())

    def test_unet_handles_odd_sizes_at_all_scales(self):
        unet = SpatialUNet(4, (4, 6, 8))
        for height, width in ((9, 11), (7, 13), (13, 9)):
            with self.subTest(height=height, width=width):
                features = unet(torch.randn(2, 4, height, width))
                for scale, channels in enumerate((4, 6, 8)):
                    self.assertEqual(
                        features[scale].shape,
                        (2, channels, max(2, height // 2**scale), max(2, width // 2**scale)),
                    )

    def test_projection_selection_preserves_unet_architecture_and_features(self):
        full = SpatialUNet(4, (4, 6, 8))
        weights = full.state_dict()
        shapes = {name: value.shape for name, value in weights.items()}
        values = torch.randn(2, 4, 9, 11)
        all_features = full(values)
        for scales in SCALE_SETS:
            with self.subTest(scales=scales):
                selected = SpatialUNet(4, (4, 6, 8), scales=scales)
                self.assertEqual({name: value.shape for name, value in selected.state_dict().items()}, shapes)
                selected.load_state_dict(weights)
                features = selected(values)
                self.assertEqual(set(features), set(scales))
                for scale in scales:
                    torch.testing.assert_close(features[scale], all_features[scale])

    def test_bilinear_projection_orientation_order_and_shared_scale_geometry(self):
        pos = self.mesh["pos"]
        expected = 2 * pos[:, 0] - 3 * pos[:, 1] + 0.4 * pos[:, 0] * pos[:, 1]
        for flip_rows, flip_columns in product((False, True), repeat=2):
            with self.subTest(flip_rows=flip_rows, flip_columns=flip_columns):
                grid = self.grid
                if flip_rows:
                    grid = grid.flip(1)
                if flip_columns:
                    grid = grid.flip(2)
                projector = BilinearProjector(grid, pos)
                for height, width in ((9, 11), (5, 6), (3, 3)):
                    lat, lon = torch.meshgrid(
                        torch.linspace(grid[1, 0, 0], grid[1, -1, 0], height),
                        torch.linspace(grid[0, 0, 0], grid[0, 0, -1], width),
                        indexing="ij",
                    )
                    field = 2 * lon - 3 * lat + 0.4 * lon * lat
                    projected = projector(torch.stack((field, field + 7))[None])
                    self.assertEqual(projected.shape, (1, 5, 2))
                    torch.testing.assert_close(projected[0, :, 0], expected, atol=3e-6, rtol=1e-6)
                    torch.testing.assert_close(projected[0, :, 1], expected + 7, atol=3e-6, rtol=1e-6)

    def test_projector_handles_nonuniform_rectilinear_axes(self):
        pos = self.mesh["pos"]
        expected = 2 * pos[:, 0] - 3 * pos[:, 1] + 0.4 * pos[:, 0] * pos[:, 1]
        lat, lon = torch.meshgrid(
            torch.tensor([5., 4.9, 2., -1.]),
            torch.tensor([2., 2.3, 3.2, 5.4, 8.]), indexing="ij",
        )
        for flip_rows, flip_columns in product((False, True), repeat=2):
            with self.subTest(flip_rows=flip_rows, flip_columns=flip_columns):
                grid = torch.stack((lon, lat))
                if flip_rows:
                    grid = grid.flip(1)
                if flip_columns:
                    grid = grid.flip(2)
                field = 2 * grid[0] - 3 * grid[1] + 0.4 * grid[0] * grid[1]
                actual = BilinearProjector(grid, pos)(field[None, None])
                torch.testing.assert_close(actual[0, :, 0], expected, atol=5e-6, rtol=1e-6)

    def test_projector_rejects_uncovered_nodes_and_nonrectilinear_grid(self):
        positions = self.mesh["pos"].clone()
        positions[0, 0] = 9
        with self.assertRaises(ValueError):
            BilinearProjector(self.grid, positions)
        grid = self.grid.clone()
        grid[0, 3, 4] += 0.1
        with self.assertRaises(ValueError):
            BilinearProjector(grid, self.mesh["pos"])

    def test_mean_adjacency_uses_incoming_neighbors_and_handles_isolated_nodes(self):
        edges = torch.tensor([[0, 2, 0], [1, 1, 2]])
        adjacency = mean_adjacency(edges, 4)
        self.assertEqual(adjacency.shape, (4, 4))
        self.assertNotEqual(adjacency.layout, torch.strided)
        aggregated = torch.sparse.mm(adjacency, torch.tensor([[1.], [3.], [7.], [11.]]))
        torch.testing.assert_close(aggregated, torch.tensor([[0.], [4.], [1.], [0.]]))

    def test_graph_variants_use_topology_and_keep_batch_items_independent(self):
        other_mesh = dict(self.mesh, edge_index=torch.arange(5).repeat(2, 1))
        constructors = [
            (variant, lambda mesh, variant=variant: BaseModel(
                {"variant": variant, "hidden_dim": 8, "temporal_hidden_dim": 4},
                mesh, self.grid,
            )) for variant in ("mlp", "gnn", "stgnn")
        ]
        constructors.append(("igarss27", lambda mesh: IGARSS27Model(
            dict(IGARSS_CONFIG, temporal_encoder="gru", scales=[0], fusion="concat", decoder="mlp"),
            mesh, self.grid,
        )))
        for name, constructor in constructors:
            with self.subTest(model=name):
                torch.manual_seed(8)
                model = constructor(self.mesh)
                torch.manual_seed(8)
                changed = constructor(other_mesh)
                together = model({"forcing": self.forcing}, self.mesh)["surge"]
                apart = torch.cat([
                    model({"forcing": sample[None]}, self.mesh)["surge"]
                    for sample in self.forcing
                ])
                torch.testing.assert_close(together, apart)
                alternate = changed({"forcing": self.forcing}, other_mesh)["surge"]
                if name == "mlp":
                    torch.testing.assert_close(together, alternate)
                else:
                    self.assertFalse(torch.allclose(together, alternate))
                    sparse_buffers = [b for b in model.buffers() if b.layout != torch.strided]
                    self.assertEqual(len(sparse_buffers), 1)
                    self.assertEqual(sparse_buffers[0].shape, (5, 5))

    def test_fusion_dimensions_softmax_and_gating_gradients(self):
        for fusion, scales in product(("concat", "adaptive"), SCALE_SETS):
            with self.subTest(fusion=fusion, scales=scales):
                model = IGARSS27Model(
                    dict(IGARSS_CONFIG, temporal_encoder="gru", scales=list(scales), fusion=fusion, decoder="mlp"),
                    self.mesh, self.grid,
                )
                projected = [torch.randn(2, 5, 3, requires_grad=True) for _ in scales]
                mesh_features = torch.randn(5, 4, requires_grad=True)
                fused, weights = model.fuse_scales(projected, mesh_features)
                self.assertEqual(fused.shape, (2, 5, len(scales) * 3 + 4))
                torch.testing.assert_close(fused[..., -4:], mesh_features.expand(2, -1, -1))
                if fusion == "concat":
                    self.assertIsNone(weights)
                    torch.testing.assert_close(fused[..., :-4], torch.cat(projected, dim=-1))
                else:
                    self.assertEqual(weights.shape, (2, 5, len(scales)))
                    self.assertTrue((weights >= 0).all())
                    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 5))
                    torch.testing.assert_close(fused[..., :-4], torch.cat([
                        values * weights[..., index:index + 1]
                        for index, values in enumerate(projected)
                    ], dim=-1))
                    fused.square().mean().backward()
                    gradients = [p.grad for p in model.scale_score.parameters()]
                    self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                    if len(scales) > 1:
                        self.assertTrue(any(torch.count_nonzero(g) for g in gradients))
                    else:
                        torch.testing.assert_close(weights, torch.ones_like(weights))

    def test_static_and_dynamic_gnns_are_separate_and_share_only_topology(self):
        model = IGARSS27Model(
            dict(IGARSS_CONFIG, temporal_encoder="gru", scales=[0, 1, 2], fusion="adaptive", decoder="gnn"),
            self.mesh, self.grid,
        )
        self.assertIsInstance(model.mesh_encoder, SpatialGNN)
        self.assertIsInstance(model.decoder, SpatialGNN)
        static_ids = {id(p) for p in model.mesh_encoder.parameters()}
        decoder_ids = {id(p) for p in model.decoder.parameters()}
        self.assertFalse(static_ids & decoder_ids)
        static_calls, dynamic_calls = [], []
        hooks = [
            model.mesh_encoder.register_forward_pre_hook(lambda module, args: static_calls.append(args)),
            model.decoder.register_forward_pre_hook(lambda module, args: dynamic_calls.append(args)),
        ]
        try:
            self.assert_finite_backward(model, 48)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(len(static_calls), 1)
        self.assertEqual(len(dynamic_calls), 1)
        torch.testing.assert_close(static_calls[0][0].reshape(5, 5), self.mesh["x"])
        self.assertEqual(dynamic_calls[0][0].shape, (2, 5, 13))
        self.assertIs(static_calls[0][1], dynamic_calls[0][1])
        self.assertEqual(static_calls[0][1].shape, (5, 5))
        for module in (model.mesh_encoder, model.decoder):
            gradients = [p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertTrue(any(torch.count_nonzero(g) for g in gradients))

    def test_coordinate_channels_and_tides_do_not_enter_weather_encoder(self):
        models = [BaseModel({"variant": v}, self.mesh, self.grid) for v in ("mlp", "gnn", "stgnn")]
        models.append(IGARSS27Model(
            dict(IGARSS_CONFIG, temporal_encoder="gru", scales=[0], fusion="concat", decoder="mlp"),
            self.mesh, self.grid,
        ))
        modified = self.forcing.clone()
        modified[:, :, 3:] = float("nan")
        batch = {"forcing": modified, "tide_boundary": torch.tensor(float("nan")),
                 "tide_potential": torch.tensor(float("nan")), "tide_time": torch.tensor(float("nan"))}
        for model in models:
            with self.subTest(model=type(model).__name__):
                torch.testing.assert_close(
                    model(batch, self.mesh)["surge"], model({"forcing": self.forcing}, self.mesh)["surge"]
                )

    def test_configuration_and_weights_reload_for_each_model_variant(self):
        configs = [{"name": "base", "variant": variant, "hidden_dim": 6, "temporal_hidden_dim": 4}
                   for variant in ("mlp", "gnn", "stgnn")]
        configs += [dict(IGARSS_CONFIG, temporal_encoder=temporal, fusion=fusion, decoder=decoder, scales=[0, 2])
                    for temporal, fusion, decoder in product(
                        ("gru", "lstm", "temporal_cnn", "tcn"), ("concat", "adaptive"), ("mlp", "gnn")
                    )]
        for config, history in product(configs, (48, 0)):
            with self.subTest(config=config, history_hours=history):
                if config["name"] == "base":
                    model = BaseModel(config, self.mesh, self.grid, history_hours=history)
                else:
                    model = IGARSS27Model(config, self.mesh, self.grid)
                data = BytesIO()
                torch.save({"config": {"model": config, "data": {"history_hours": history}},
                            "model": model.state_dict()}, data)
                data.seek(0)
                checkpoint = torch.load(data, weights_only=True)
                saved = checkpoint["config"]
                if saved["model"]["name"] == "base":
                    restored = BaseModel(saved["model"], self.mesh, self.grid, history_hours=saved["data"]["history_hours"])
                else:
                    restored = IGARSS27Model(saved["model"], self.mesh, self.grid)
                restored.load_state_dict(checkpoint["model"])
                batch = {"forcing": self.forcing[:, -(history // 6 + 1):]}
                torch.testing.assert_close(
                    restored(batch, self.mesh)["surge"], model(batch, self.mesh)["surge"], rtol=0, atol=0
                )

    def test_full_architectures_fail_explicitly(self):
        for model_class in (FullA0Model, FullA1Model):
            with self.subTest(model=model_class.__name__), self.assertRaises(NotImplementedError):
                model_class({}, self.mesh, self.grid)


if __name__ == "__main__":
    unittest.main()
