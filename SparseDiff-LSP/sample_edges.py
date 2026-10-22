"""
Sparse edge sampling utilities for SparseDiffLSP.
Adapted from SparseDiff (Qin et al., 2023).

Provides:
    - Condensed <-> matrix index conversion for upper-triangle edge indexing
    - sample_query_edges: sample a fraction of all possible edges
    - get_computational_graph: merge query edges with existing noisy edges
"""

import torch
import torch.nn.functional as F
from torch_geometric.utils import coalesce
import utils as U


# ---------------------------------------------------------------------------
#  Condensed <-> Matrix index conversion
# ---------------------------------------------------------------------------
def condensed_to_matrix_index(condensed_index, num_nodes):
    """Convert condensed upper-triangle index to (i, j) pairs.
    condensed_index: (E,)
    num_nodes: scalar
    """
    b = 1 - (2 * num_nodes)
    i = torch.div(
        (-b - torch.sqrt(b ** 2 - 8 * condensed_index)), 2, rounding_mode="floor"
    )
    j = condensed_index + torch.div(i * (b + i + 2), 2, rounding_mode="floor") + 1
    return torch.vstack((i.long(), j.long()))


def condensed_to_matrix_index_batch(condensed_index, num_nodes, edge_batch, ptr):
    """Batched version: handles multiple graphs with node offsets.
    condensed_index: (E,) local indices per graph
    num_nodes: (bs,)
    edge_batch: (E,) graph assignment
    ptr: (bs+1,) node offset
    """
    bb = -2 * num_nodes[edge_batch] + 1
    ii = torch.div(
        (-bb - torch.sqrt(bb ** 2 - 8 * condensed_index)), 2, rounding_mode="floor"
    )
    jj = (
        condensed_index
        + torch.div(ii * (bb + ii + 2), 2, rounding_mode="floor")
        + 1
    )
    return torch.vstack((ii.long(), jj.long())) + ptr[edge_batch]


def matrix_to_condensed_index_batch(matrix_index, num_nodes, edge_batch):
    """Convert (i,j) pairs to condensed index per graph."""
    n = num_nodes[edge_batch]
    i = matrix_index[0]
    j = matrix_index[1]
    return n * (n - 1) / 2 - (n - i) * (n - i - 1) / 2 + j - i - 1


# ---------------------------------------------------------------------------
#  Uniform condensed sampling
# ---------------------------------------------------------------------------
def _sampled_condensed_indices_uniformly(max_condensed_value, num_edges_to_sample):
    """Sample condensed indices uniformly without replacement.
    max_condensed_value: (bs,) long — n*(n-1)/2 per graph
    num_edges_to_sample: (bs,) long
    Returns: condensed_index (flat), edge_batch (flat)
    """
    batch_size = max_condensed_value.shape[0]
    device = max_condensed_value.device

    # Optimised path: all graphs same size
    if (len(torch.unique(max_condensed_value)) == 1
            and len(torch.unique(num_edges_to_sample)) == 1):
        max_val = max_condensed_value[0]
        to_sample = num_edges_to_sample[0]
        sampled = torch.multinomial(
            torch.ones(max_val, device=device),
            num_samples=to_sample,
            replacement=False,
        )
        if batch_size == 1:
            edge_batch = torch.zeros(to_sample, device=device, dtype=torch.long)
            return sampled, edge_batch

        sampled_rep = sampled.unsqueeze(0).expand(batch_size, -1).flatten()
        edge_batch = torch.arange(batch_size, device=device).repeat_interleave(to_sample)
        return sampled_rep, edge_batch

    # General case: varying sizes
    max_size = torch.max(max_condensed_value)
    randperm = torch.randperm(max_size, device=device)
    randperm_exp = randperm.unsqueeze(0).expand(batch_size, -1)

    mask1 = randperm_exp < max_condensed_value.unsqueeze(1)
    mask2 = torch.cumsum(mask1, dim=1) <= num_edges_to_sample.unsqueeze(1)
    complete_mask = mask1 * mask2

    condensed_index = randperm_exp[complete_mask]
    edge_batch = (
        torch.arange(batch_size, device=device)
        .unsqueeze(1)
        .expand(-1, max_size)[complete_mask]
    )
    return condensed_index, edge_batch


# ---------------------------------------------------------------------------
#  Public API: sample_query_edges
# ---------------------------------------------------------------------------
def sample_query_edges(num_nodes_per_graph, edge_proportion):
    """Sample edge_proportion fraction of all possible edges per graph.

    Args:
        num_nodes_per_graph: (bs,) long tensor
        edge_proportion: float in (0, 1]

    Returns:
        triu_edge_index: (2, E_query) upper-triangle edges with global node indices
        batch: (N_total,) node-to-graph assignment
    """
    assert num_nodes_per_graph.dtype == torch.long
    assert 0 < edge_proportion <= 1.0

    batch_size = len(num_nodes_per_graph)
    device = num_nodes_per_graph.device
    n = num_nodes_per_graph

    max_condensed = (n * (n - 1) / 2).long()
    num_to_sample = torch.ceil(edge_proportion * max_condensed.float()).long()
    num_to_sample = torch.clamp(num_to_sample, min=1)  # at least 1 edge

    condensed_index, edge_batch = _sampled_condensed_indices_uniformly(
        max_condensed, num_to_sample
    )

    offset = torch.cumsum(n, dim=0)
    offset = torch.cat([torch.zeros(1, device=device, dtype=torch.long), offset[:-1]])

    edge_index = condensed_to_matrix_index_batch(
        condensed_index, num_nodes=n, edge_batch=edge_batch, ptr=offset
    )

    batch = torch.arange(batch_size, device=device).repeat_interleave(n)
    return edge_index, batch


# ---------------------------------------------------------------------------
#  Computational graph: merge query + existing noisy edges
# ---------------------------------------------------------------------------
def get_computational_graph(triu_query_edge_index, clean_edge_index, clean_edge_attr):
    """Build computational graph = union(query_edges, noisy_edges).

    Query edges that don't exist in the noisy graph get default attr [1, 0] (no-edge).
    Returns a mask indicating which edges in the merged graph are query edges.

    Args:
        triu_query_edge_index: (2, Q) upper-triangle query edges
        clean_edge_index:      (2, M) existing noisy edges (undirected)
        clean_edge_attr:       (M, 2) one-hot edge attributes

    Returns:
        query_mask:     (M', ) bool — True for query-position edges
        comp_edge_index: (2, M') merged edge index
        comp_edge_attr:  (M', 2) merged edge attributes
    """
    de = clean_edge_attr.shape[-1]
    device = triu_query_edge_index.device

    # Default query attr: no-edge
    default_attr = torch.zeros(triu_query_edge_index.shape[1], de, device=device)
    default_attr[:, 0] = 1

    # Make query edges undirected
    query_edge_index, default_attr = U.to_undirected(triu_query_edge_index, default_attr)

    # Concatenate
    comp_edge_index = torch.hstack([clean_edge_index, query_edge_index])
    default_comp_attr = torch.argmax(
        torch.vstack([clean_edge_attr, default_attr]), -1
    )

    # Coalesce: resolve duplicates
    _, min_attr = coalesce(comp_edge_index, default_comp_attr, reduce="min")
    max_edge_index, max_attr = coalesce(comp_edge_index, default_comp_attr, reduce="max")

    # Query mask: edges whose min-class is 0 (no-edge) — i.e., at least one
    # version was a query-default
    query_mask = min_attr == 0
    comp_edge_attr = F.one_hot(max_attr.long(), num_classes=de).float()

    return query_mask, max_edge_index, comp_edge_attr
