"""Linear integration-test model, not a scientific full-field emulator."""

import torch


class SmokeModel(torch.nn.Module):
    """Exercise the full-field interface with one linear prediction per horizon.

    Latest-time forcing [B, C, Y, X] has batch size B and C channels and is
    averaged over the spatial grid (Y, X). The
    resulting [B, H] values are shared across all N mesh nodes; this is only a
    pipeline check, not a model of spatial surge variation.
    """

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 6)

    def forward(self, batch, mesh):
        """Return normalized ``surge`` [B, N, H] using forcing and mesh identity."""
        values = self.linear(batch["forcing"][:, -1].mean(dim=(-2, -1)))
        return {"surge": values[:, None, :].expand(-1, len(mesh["node_id"]), -1)}

