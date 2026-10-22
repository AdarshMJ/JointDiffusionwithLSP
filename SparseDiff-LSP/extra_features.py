"""
Extra features for SparseDiffLSP.

Supports: 'none' (default), 'cycles', 'eigen', 'all'.
Features are computed from the dense noisy adjacency matrix.
"""

import torch
import torch.nn.functional as F
import utils as U


class DummyExtraFeatures:
    """No extra features — returns empty tensors."""

    def __call__(self, noisy_data):
        X_t = noisy_data["X_t"]
        E_t = noisy_data["E_t"]
        y_t = noisy_data["y_t"]
        node_mask = noisy_data["node_mask"]

        empty_x = X_t.new_zeros((*X_t.shape[:-1], 0))
        empty_e = E_t.new_zeros((*E_t.shape[:-1], 0))
        empty_y = y_t.new_zeros((y_t.shape[0], 0))
        return U.PlaceHolder(X=empty_x, E=empty_e, y=empty_y)


class CycleFeatures:
    """Cycle counts (k=3,4,5) as node features, (k=3,4,5,6) as graph features."""

    def __call__(self, noisy_data):
        E_t = noisy_data["E_t"]          # (bs, n, n, 2)
        node_mask = noisy_data["node_mask"]
        y_t = noisy_data["y_t"]

        adj = E_t[..., 1:].sum(dim=-1).float()  # (bs, n, n)
        adj = adj * node_mask.unsqueeze(1).float() * node_mask.unsqueeze(2).float()

        # Powers of A
        A2 = adj @ adj
        A3 = A2 @ adj
        A4 = A3 @ adj
        A5 = A4 @ adj
        A6 = A5 @ adj

        diag2 = torch.diagonal(A2, dim1=-2, dim2=-1)
        diag3 = torch.diagonal(A3, dim1=-2, dim2=-1)
        diag4 = torch.diagonal(A4, dim1=-2, dim2=-1)
        diag5 = torch.diagonal(A5, dim1=-2, dim2=-1)
        diag6 = torch.diagonal(A6, dim1=-2, dim2=-1)

        # Node features: triangles, 4-cycles, 5-cycles per node
        k3_node = diag3 / 2.0
        k4_node = (diag4 - diag2 * (diag2 - 1)) / 2.0
        k5_node = (diag5 - 2 * diag3 * diag2) / 2.0

        x_feat = torch.stack([k3_node, k4_node, k5_node], dim=-1)
        x_feat = torch.clamp(x_feat, 0, 5) / 5.0 * node_mask.unsqueeze(-1).float()

        # Graph features: traces
        bs = adj.shape[0]
        k3_y = diag3.sum(-1) / 6.0
        k4_y = (diag4.sum(-1) - diag2.sum(-1) ** 2 + diag2.sum(-1)) / 8.0
        k5_y = (diag5.sum(-1)) / 10.0
        k6_y = (diag6.sum(-1)) / 12.0
        y_feat = torch.stack([k3_y, k4_y, k5_y, k6_y], dim=-1)
        y_feat = torch.clamp(y_feat, 0, 5) / 5.0

        empty_e = E_t.new_zeros((*E_t.shape[:-1], 0))
        return U.PlaceHolder(X=x_feat, E=empty_e, y=y_feat)


class EigenFeatures:
    """Laplacian eigenvector (node) and eigenvalue (graph) features."""

    def __init__(self, num_eigenvectors=8, num_eigenvalues=5):
        self.num_eigenvectors = num_eigenvectors
        self.num_eigenvalues = num_eigenvalues

    def __call__(self, noisy_data):
        E_t = noisy_data["E_t"]
        node_mask = noisy_data["node_mask"]
        y_t = noisy_data["y_t"]

        adj = E_t[..., 1:].sum(dim=-1).float()
        adj = adj * node_mask.unsqueeze(1).float() * node_mask.unsqueeze(2).float()

        # Combinatorial Laplacian
        deg = adj.sum(dim=-1)
        D = torch.diag_embed(deg)
        L = D - adj
        L = (L + L.transpose(-1, -2)) / 2.0

        # Mask invalid nodes
        bs, n = node_mask.shape
        mask_diag = 2 * n * torch.eye(n, device=L.device).unsqueeze(0)
        mask_diag = mask_diag * (~node_mask.unsqueeze(1)).float() * (~node_mask.unsqueeze(2)).float()
        L = L * node_mask.unsqueeze(1).float() * node_mask.unsqueeze(2).float() + mask_diag

        try:
            eigvals, eigvecs = torch.linalg.eigh(L)
        except Exception:
            eigvals, eigvecs = torch.linalg.eigh(L.cpu())
            eigvals = eigvals.to(L.device)
            eigvecs = eigvecs.to(L.device)

        n_valid = node_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        eigvals = eigvals / n_valid

        # Number of connected components
        n_cc = (eigvals < 1e-4).sum(dim=-1)  # (bs,)

        # First k eigenvalues after zero eigenvalues
        k = self.num_eigenvalues
        to_extend = max(0, int(n_cc.max().item()) + k - n)
        if to_extend > 0:
            eigvals = torch.cat([eigvals, 2 * torch.ones(bs, to_extend, device=eigvals.device)], dim=1)
        indices = torch.arange(k, device=eigvals.device).unsqueeze(0) + n_cc.unsqueeze(1)
        batch_eigenvals = torch.gather(eigvals, 1, indices.long())

        # First k eigenvectors after zero eigenvalues
        kv = self.num_eigenvectors
        to_extend_v = max(0, int(n_cc.max().item()) + kv - n)
        if to_extend_v > 0:
            eigvecs = torch.cat([eigvecs, torch.zeros(bs, n, to_extend_v, device=eigvecs.device)], dim=2)
        indices_v = torch.arange(kv, device=eigvecs.device).unsqueeze(0).unsqueeze(0) + n_cc.unsqueeze(1).unsqueeze(2)
        indices_v = indices_v.expand(-1, n, -1).long()
        first_k_evec = torch.gather(eigvecs, 2, indices_v)
        first_k_evec = first_k_evec * node_mask.unsqueeze(2).float()

        # Combine
        x_feat = first_k_evec  # (bs, n, kv)
        y_feat = torch.cat([n_cc.unsqueeze(-1).float(), batch_eigenvals], dim=-1)

        empty_e = E_t.new_zeros((*E_t.shape[:-1], 0))
        return U.PlaceHolder(X=x_feat, E=empty_e, y=y_feat)


class AllExtraFeatures:
    """Combines cycle + eigen features."""

    def __init__(self, num_eigenvectors=8, num_eigenvalues=5):
        self.cycles = CycleFeatures()
        self.eigen = EigenFeatures(num_eigenvectors, num_eigenvalues)

    def __call__(self, noisy_data):
        cf = self.cycles(noisy_data)
        ef = self.eigen(noisy_data)
        return U.PlaceHolder(
            X=torch.cat([cf.X, ef.X], dim=-1),
            E=cf.E,  # both empty
            y=torch.cat([cf.y, ef.y], dim=-1),
        )


def get_extra_features(name, num_eigenvectors=8, num_eigenvalues=5):
    """Factory function: create extra features by name.

    Args:
        name: 'none', 'cycles', 'eigen', 'all'
    """
    if name is None or name == "none":
        return DummyExtraFeatures()
    elif name == "cycles":
        return CycleFeatures()
    elif name == "eigen":
        return EigenFeatures(num_eigenvectors, num_eigenvalues)
    elif name == "all":
        return AllExtraFeatures(num_eigenvectors, num_eigenvalues)
    else:
        raise ValueError(f"Unknown extra_features: {name}. Use 'none', 'cycles', 'eigen', or 'all'.")
