"""Temporal encoding, spatial U-Net, and bilinear grid-to-mesh IGARSS models."""

import torch
from torch import nn
from torch.nn import functional as F

from .common import BilinearProjector, SpatialGNN, mean_adjacency


class TemporalEncoder(nn.Module):
    """Encode independent grid-cell histories, including a single issue time."""

    def __init__(self, kind, hidden_dim):
        super().__init__()
        self.kind = kind
        if kind == "gru":
            self.network = nn.GRU(3, hidden_dim, batch_first=True)
        elif kind == "lstm":
            self.network = nn.LSTM(3, hidden_dim, batch_first=True)
        elif kind in ("temporal_cnn", "tcn"):
            # Padding preserves every sequence length; no temporal flattening.
            # TCN's 15-step receptive field covers the longest nine-step history.
            dilations = (1, 2, 4) if kind == "tcn" else (1, 1)
            self.network = nn.ModuleList([
                nn.Conv1d(3 if index == 0 else hidden_dim, hidden_dim, 3, dilation=dilation)
                for index, dilation in enumerate(dilations)
            ])
        else:
            raise ValueError(f"Unknown IGARSS temporal_encoder: {kind}")

    def forward(self, forcing):
        """Map [B,T,3,H,W] to [B,D,H,W] without mixing grid cells."""
        batch_size, steps, channels, height, width = forcing.shape
        sequence = forcing.permute(0, 3, 4, 1, 2).reshape(-1, steps, channels)
        if self.kind in ("gru", "lstm"):
            _, hidden = self.network(sequence)
            if self.kind == "lstm":
                hidden = hidden[0]
            embedding = hidden[-1]
        else:
            sequence = sequence.transpose(1, 2)
            for layer in self.network:
                padding = (layer.kernel_size[0] - 1) * layer.dilation[0]
                # TCN is causal: each output uses only its present and past.
                sides = (padding, 0) if self.kind == "tcn" else (padding // 2,) * 2
                sequence = F.relu(layer(F.pad(sequence, sides)))
            embedding = sequence[:, :, -1] if self.kind == "tcn" else sequence.mean(-1)
        return embedding.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2)


class SpatialUNet(nn.Module):
    """Three spatial scales with exact-size skip connections for odd grids.

    F0 and F1 are decoded features; F2 is the bottleneck. The architecture
    stays identical for every projection-scale selection.
    """

    def __init__(self, in_dim, channels=(32, 64, 128), scales=(0, 1, 2)):
        super().__init__()
        if len(channels) != 3:
            raise ValueError("IGARSS unet_channels must contain three widths")
        self.scales = tuple(scales)
        if not self.scales or len(set(self.scales)) != len(self.scales) or any(
            scale not in (0, 1, 2) for scale in self.scales
        ):
            raise ValueError("IGARSS scales must be a nonempty subset of [0,1,2]")
        self.encoders = nn.ModuleList()
        for out_dim in channels:
            self.encoders.append(nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_dim, out_dim, 3, padding=1),
                nn.ReLU(),
            ))
            in_dim = out_dim
        self.decoders = nn.ModuleDict()
        for scale in (1, 0):
            self.decoders[str(scale)] = nn.Sequential(
                nn.Conv2d(channels[scale + 1] + channels[scale], channels[scale], 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(channels[scale], channels[scale], 3, padding=1),
                nn.ReLU(),
            )

    def forward(self, values):
        """Return selected spatial features as a dictionary indexed by scale."""
        encoded = []
        for scale, encoder in enumerate(self.encoders):
            if scale:
                # Endpoint-preserving resizing keeps one geographic sampling
                # map valid at every scale (including odd atmospheric grids).
                size = tuple(max(2, dimension // 2) for dimension in values.shape[-2:])
                values = F.interpolate(values, size=size, mode="bilinear", align_corners=True)
            values = encoder(values)
            encoded.append(values)
        features = {2: values}
        for scale in (1, 0):
            values = F.interpolate(
                values, size=encoded[scale].shape[-2:], mode="bilinear", align_corners=True,
            )
            values = self.decoders[str(scale)](torch.cat((encoded[scale], values), dim=1))
            features[scale] = values
        return {scale: features[scale] for scale in self.scales}


class IGARSS27Model(nn.Module):
    """Independent temporal/spatial encoders, static mesh GNN, and mesh decoder."""

    def __init__(self, config, mesh, forcing_grid):
        super().__init__()
        for key, expected in (
            ("spatial_encoder", "unet"), ("projector", "bilinear"), ("mesh_encoder", "gnn"),
        ):
            if config.get(key, expected) != expected:
                raise ValueError(f"IGARSS {key} must be {expected!r}")
        self.scales = tuple(config.get("scales", [0, 1, 2]))
        self.fusion = config.get("fusion", "concat")
        self.decoder_kind = config.get("decoder", "mlp")
        if self.fusion not in ("concat", "adaptive"):
            raise ValueError(f"Unknown IGARSS fusion: {self.fusion}")
        if self.decoder_kind not in ("mlp", "gnn"):
            raise ValueError(f"Unknown IGARSS decoder: {self.decoder_kind}")
        temporal_dim = config.get("temporal_hidden_dim", 32)
        channels = config.get("unet_channels", [32, 64, 128])
        projected_dim = config.get("projected_dim", 32)
        mesh_dim = config.get("mesh_hidden_dim", 32)
        decoder_dim = config.get("decoder_hidden_dim", 64)

        self.temporal_encoder = TemporalEncoder(config.get("temporal_encoder", "gru"), temporal_dim)
        self.spatial_encoder = SpatialUNet(temporal_dim, channels, self.scales)
        # Reduce channels before projecting large full-field mesh tensors.
        self.channel_reducers = nn.ModuleDict({
            str(scale): nn.Conv2d(channels[scale], projected_dim, 1) for scale in self.scales
        })
        self.projector = BilinearProjector(forcing_grid, mesh["pos"])
        self.register_buffer(
            "adjacency", mean_adjacency(mesh["edge_index"], len(mesh["pos"])), persistent=False,
        )
        self.mesh_encoder = SpatialGNN(5, mesh_dim, config.get("mesh_encoder_layers", 2))
        if self.fusion == "adaptive":
            # One shared score per scale/node; mesh features condition the score.
            self.scale_score = nn.Sequential(
                nn.Linear(projected_dim + mesh_dim, projected_dim),
                nn.Tanh(),
                nn.Linear(projected_dim, 1),
            )

        fused_dim = len(self.scales) * projected_dim + mesh_dim
        if self.decoder_kind == "gnn":
            self.decoder = SpatialGNN(fused_dim, decoder_dim, config.get("decoder_gnn_layers", 2))
        else:
            self.decoder = nn.Sequential(
                nn.Linear(fused_dim, decoder_dim), nn.ReLU(),
                nn.Linear(decoder_dim, decoder_dim), nn.ReLU(),
            )
        self.surge_head = nn.Linear(decoder_dim, 6)

    def fuse_scales(self, projected, mesh_features):
        """Return weighted concatenation and optional [B,N,S] scale weights."""
        static = mesh_features.unsqueeze(0).expand(projected[0].shape[0], -1, -1)
        weights = None
        if self.fusion == "adaptive":
            scores = torch.cat([
                self.scale_score(torch.cat((values, static), dim=-1)) for values in projected
            ], dim=-1)
            weights = scores.softmax(dim=-1)
            projected = [values * weights[..., index:index + 1] for index, values in enumerate(projected)]
        return torch.cat((*projected, static), dim=-1), weights

    def forward(self, batch, mesh):
        """Return normalized surge [B,N,6] for any nonempty forcing history."""
        temporal = self.temporal_encoder(batch["forcing"][:, :, :3])
        spatial = self.spatial_encoder(temporal)
        projected = [
            self.projector(self.channel_reducers[str(scale)](spatial[scale])) for scale in self.scales
        ]
        # Shared static features are encoded once per forward and stay trainable.
        mesh_features = self.mesh_encoder(mesh["x"], self.adjacency)
        fused, _ = self.fuse_scales(projected, mesh_features)
        decoded = self.decoder(fused, self.adjacency) if self.decoder_kind == "gnn" else self.decoder(fused)
        return {"surge": self.surge_head(decoded)}
