"""Reserved Full A0 interface; its architecture has not been agreed yet."""

from torch import nn


class FullA0Model(nn.Module):
    def __init__(self, config=None, mesh=None, forcing_grid=None):
        super().__init__()
        raise NotImplementedError("Full A0 (full_v1) architecture is deferred")

    def forward(self, batch, mesh):
        raise NotImplementedError("Full A0 (full_v1) architecture is deferred")
