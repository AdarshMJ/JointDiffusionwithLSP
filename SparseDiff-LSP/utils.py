"""
Utility functions for SparseDiffLSP.
Sparse graph representations, conversions, and data containers.
"""

import torch
import torch.nn.functional as F
from torch_geometric.utils import (
    to_dense_adj,
    to_dense_batch,
    dense_to_sparse,
    coalesce,
    remove_self_loops,
)
import torch_geometric.nn.pool as pool
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
#  Data containers
# ---------------------------------------------------------------------------
@dataclass
class PlaceHolder:
    """Dense graph data container."""
    X: torch.Tensor
    E: torch.Tensor
    y: torch.Tensor

    def mask(self, node_mask):
        x_mask = node_mask.unsqueeze(-1)
        e_mask1 = x_mask.unsqueeze(2)
        e_mask2 = x_mask.unsqueeze(1)
        self.X = self.X * x_mask
        self.E = self.E * e_mask1 * e_mask2
        assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self


@dataclass
class SparsePlaceHolder:
    """Sparse graph prediction container."""
    node: torch.Tensor          # (N, dx)
    edge_attr: torch.Tensor     # (M, de)
    edge_index: torch.Tensor    # (2, M)
    y: torch.Tensor             # (bs, dy)
    batch: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
#  Dense / Sparse conversions
# ---------------------------------------------------------------------------
def to_dense(x, edge_index, edge_attr, batch):
    """Convert PyG batch to dense (X, E, node_mask)."""
    batch = batch.to(torch.int64)
    X, node_mask = to_dense_batch(x=x, batch=batch)
    max_num_nodes = X.size(1)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    E = to_dense_adj(
        edge_index=edge_index,
        batch=batch,
        edge_attr=edge_attr,
        max_num_nodes=max_num_nodes,
    )
    E = encode_no_edge(E)
    return PlaceHolder(X=X, E=E, y=None), node_mask


def encode_no_edge(E):
    """Add a no-edge class as channel 0."""
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = (
        torch.eye(E.shape[1], dtype=torch.bool)
        .unsqueeze(0)
        .expand(E.shape[0], -1, -1)
    )
    E[diag] = 0
    return E


def to_undirected(edge_index, edge_attr=None):
    """Make edge_index undirected by adding reverse edges."""
    rev = edge_index[[1, 0]]
    new_index = torch.cat([edge_index, rev], dim=1)
    if edge_attr is not None:
        new_attr = torch.cat([edge_attr, edge_attr], dim=0)
        new_index, new_attr = coalesce(new_index, new_attr, reduce="max")
        return new_index, new_attr
    new_index = coalesce(new_index, reduce="max")
    return new_index


def undirected_to_directed(edge_index, edge_attr):
    """Keep only upper-triangle edges (i < j)."""
    mask = edge_index[0] < edge_index[1]
    return edge_index[:, mask], edge_attr[mask]
