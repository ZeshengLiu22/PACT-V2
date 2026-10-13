"""Linear integration-test model, not a scientific full-field emulator."""

import torch


class SmokeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(5, 6)

    def forward(self, batch, mesh):
        values = self.linear(batch["forcing"][:, -1].mean(dim=(-2, -1)))
        return {"surge": values[:, None, :].expand(-1, len(mesh["node_id"]), -1)}

