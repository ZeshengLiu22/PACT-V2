"""Load and validate the canonical shared full-field mesh."""

import torch


def load_mesh(path):
    """Read one CPU mesh and enforce the existing tensor and node-order contract."""
    mesh_cpu = torch.load(path, map_location="cpu", weights_only=True)
    node_id = mesh_cpu["node_id"]
    n = len(node_id)
    expected = {"node_id": (torch.int64, (n,)), "pos": (torch.float64, (n, 2)),
                "x": (torch.float32, (n, 5)), "edge_index": (torch.int64, (2, None)),
                "edge_attr": (torch.float64, (None, 3)), "face": (torch.int64, (3, None)),
                "boundary_node_index": (torch.int64, (None,))}
    for key, (dtype, shape) in expected.items():
        value = mesh_cpu[key]
        if (value.dtype != dtype or value.ndim != len(shape)
                or any(a is not None and a != b for a, b in zip(shape, value.shape))
                or (value.is_floating_point() and not torch.isfinite(value).all())):
            raise ValueError(f"Invalid shared mesh field {key}")
    if (not n or torch.any(node_id <= 0) or torch.any(node_id[1:] <= node_id[:-1])
            or len(mesh_cpu["edge_attr"]) != mesh_cpu["edge_index"].shape[1]):
        raise ValueError("Invalid canonical mesh identity or edges")
    for key in ("edge_index", "face", "boundary_node_index"):
        value = mesh_cpu[key]
        if value.numel() and (value.min() < 0 or value.max() >= n):
            raise ValueError(f"Invalid shared mesh indices: {key}")
    return mesh_cpu
