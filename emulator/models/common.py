"""Physical grid-to-mesh interpolation and shared-topology mean aggregation."""

import torch
from torch import nn
from torch.nn import functional as F


class BilinearProjector(nn.Module):
    """Precompute mesh positions in a physical rectilinear forcing grid.

    ``forcing_grid`` is the ORIGINAL longitude/latitude [2,H,W], before
    normalization. Axes may increase or decrease and need not be uniform.
    All feature scales use the same endpoint-aligned source-index coordinates.
    """

    def __init__(self, forcing_grid, pos):
        super().__init__()
        grid, pos = forcing_grid.detach().double(), pos.detach().double()
        if grid.ndim != 3 or grid.shape[0] != 2 or min(grid.shape[1:]) < 2:
            raise ValueError("Physical forcing_grid must be [2,H,W] with H,W >= 2")
        if not torch.isfinite(grid).all() or not torch.isfinite(pos).all():
            raise ValueError("Projection coordinates must be finite")
        lon, lat = grid[0, 0], grid[1, :, 0]
        if not (
            torch.allclose(grid[0], lon[None, :].expand_as(grid[0]), rtol=0, atol=1e-6)
            and torch.allclose(grid[1], lat[:, None].expand_as(grid[1]), rtol=0, atol=1e-6)
        ):
            raise ValueError("Bilinear projection requires a rectilinear physical grid")
        axes = (lon, lat)
        lower = torch.stack([a.min() for a in axes])
        upper = torch.stack([a.max() for a in axes])
        outside = ((pos < lower) | (pos > upper)).any(dim=1)
        if outside.any():
            indices = outside.nonzero().flatten()
            raise ValueError(
                f"{len(indices)} mesh nodes outside forcing domain; "
                f"first node indices {indices[:10].tolist()}, "
                f"lon/lat {pos[indices[:10]].tolist()}"
            )
        sampling = []
        for coordinate, axis in zip(pos.T, axes):
            delta = axis[1:] - axis[:-1]
            if not ((delta > 0).all() or (delta < 0).all()):
                raise ValueError("Physical grid axes must be strictly monotonic")
            # Negating a decreasing axis permits search without reordering data.
            direction = 1 if axis[-1] > axis[0] else -1
            ordered = (axis * direction).contiguous()
            values = (coordinate * direction).contiguous()
            left = torch.searchsorted(ordered, values, right=True) - 1
            left = left.clamp(0, len(axis) - 2)  # endpoint interval, not node clipping
            fraction = (values - ordered[left]) / (ordered[left + 1] - ordered[left])
            sampling.append(2 * (left + fraction) / (len(axis) - 1) - 1)
        self.register_buffer("sampling_grid", torch.stack(sampling, dim=-1).float()[None, :, None])

    def forward(self, grid):
        # FP32 interpolation also works under BF16 AMP, without changing geometry.
        with torch.autocast(grid.device.type, enabled=False):
            values = F.grid_sample(
                grid.float(), self.sampling_grid.expand(grid.shape[0], -1, -1, -1),
                mode="bilinear", padding_mode="zeros", align_corners=True,
            )
        return values.squeeze(-1).transpose(1, 2).to(grid.dtype)


def mean_adjacency(edge_index, num_nodes):
    """One sparse [N,N] incoming-neighbor mean, shared across all samples."""
    source, target = edge_index
    degree = torch.bincount(target, minlength=num_nodes).clamp_min(1)
    values = degree[target].float().reciprocal()
    return torch.sparse_coo_tensor(
        torch.stack((target, source)), values, (num_nodes, num_nodes)
    ).coalesce()


class SpatialGNN(nn.Module):
    """GraphSAGE-style self plus mean-neighbor layers with independent weights."""

    def __init__(self, in_dim, hidden_dim, layers=2):
        super().__init__()
        if layers < 1:
            raise ValueError("SpatialGNN needs at least one layer")
        widths = [in_dim] + [hidden_dim] * layers
        self.self_linears = nn.ModuleList(
            nn.Linear(a, b) for a, b in zip(widths[:-1], widths[1:])
        )
        self.neighbor_linears = nn.ModuleList(
            nn.Linear(a, b, bias=False) for a, b in zip(widths[:-1], widths[1:])
        )

    def forward(self, x, adjacency):
        static = x.ndim == 2
        if static:
            x = x.unsqueeze(0)
        for own, neighbor in zip(self.self_linears, self.neighbor_linears):
            b, n, c = x.shape
            # Sparse FP32 avoids unsupported sparse BF16 kernels under AMP.
            with torch.autocast(x.device.type, enabled=False):
                mean = torch.sparse.mm(adjacency, x.transpose(0, 1).reshape(n, b * c).float())
            mean = mean.reshape(n, b, c).transpose(0, 1).to(x.dtype)
            x = F.gelu(own(x) + neighbor(mean))
        return x.squeeze(0) if static else x
