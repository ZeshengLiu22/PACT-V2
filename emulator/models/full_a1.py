"""Reserved Full A1 interface; its architecture has not been agreed yet."""

from torch import nn


class FullA1Model(nn.Module):
    def __init__(self, config=None, mesh=None, forcing_grid=None):
        super().__init__()
        raise NotImplementedError("Full A1 (full_v2) architecture is deferred")

    def forward(self, batch, mesh):
        raise NotImplementedError("Full A1 (full_v2) architecture is deferred")
