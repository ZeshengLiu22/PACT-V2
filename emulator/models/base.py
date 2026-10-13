"""Raw weather projection followed by a node MLP, GNN, or temporal-first GNN."""

import torch
from torch import nn

from emulator.models.common import BilinearProjector, SpatialGNN, mean_adjacency


class BaseModel(nn.Module):
    """Predict six normalized surge hours without an atmospheric spatial encoder."""

    def __init__(self, config, mesh, forcing_grid, history_hours=48):
        super().__init__()
        if type(history_hours) is not int or history_hours not in range(0, 49, 6):
            raise ValueError("history_hours must be one of 0, 6, ..., 48")
        self.history_steps = history_hours // 6 + 1
        self.variant = config.get("variant", "mlp")
        if self.variant not in ("mlp", "gnn", "stgnn"):
            raise ValueError(f"Unknown Base variant: {self.variant}")
        self.projector = BilinearProjector(forcing_grid, mesh["pos"])
        hidden = config.get("hidden_dim", 64)
        in_dim = 3 * self.history_steps + 5
        if self.variant == "stgnn":
            temporal_hidden = config.get("temporal_hidden_dim", 32)
            self.temporal = nn.GRU(3, temporal_hidden, batch_first=True)
            in_dim = temporal_hidden + 5
        if self.variant == "mlp":
            self.head = nn.Sequential(
                nn.Linear(in_dim, hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 6),
            )
        else:
            self.register_buffer(
                "adjacency", mean_adjacency(mesh["edge_index"], len(mesh["node_id"])),
                persistent=False,
            )
            self.gnn = SpatialGNN(in_dim, hidden, config.get("gnn_layers", 2))
            self.head = nn.Linear(hidden, 6)

    def forward(self, batch, mesh):
        weather = batch["forcing"][:, :, :3]
        b, t, _, height, width = weather.shape
        if self.variant == "stgnn":
            # Project each time independently; recurrence never flattens time
            # into feature channels or mixes node sequences.
            projected = self.projector(weather.reshape(b * t, 3, height, width))
            n = projected.shape[1]
            sequences = projected.reshape(b, t, n, 3).permute(0, 2, 1, 3).reshape(b * n, t, 3)
            _, hidden = self.temporal(sequences)
            dynamic = hidden[-1].reshape(b, n, -1)
        else:
            if t != self.history_steps:
                raise ValueError(f"Base {self.variant} expects {self.history_steps} history steps, got {t}")
            dynamic = self.projector(weather.reshape(b, 3 * t, height, width))
        static = mesh["x"].to(dynamic.dtype).unsqueeze(0).expand(b, -1, -1)
        features = torch.cat((dynamic, static), dim=-1)
        if self.variant != "mlp":
            features = self.gnn(features, self.adjacency)
        return {"surge": self.head(features)}
