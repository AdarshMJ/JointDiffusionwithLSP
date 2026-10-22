"""
Extra features for diffusion model input.
"""
import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import PlaceHolder


class DummyExtraFeatures:
    """Returns empty tensors (no extra features)."""
    
    def __init__(self):
        pass

    def __call__(self, noisy_data):
        X = noisy_data['X_t']
        E = noisy_data['E_t']
        y = noisy_data['y_t']
        empty_x = X.new_zeros((*X.shape[:-1], 0))
        empty_e = E.new_zeros((*E.shape[:-1], 0))
        empty_y = y.new_zeros((y.shape[0], 0))
        return PlaceHolder(X=empty_x, E=empty_e, y=empty_y)


class CRMDegreeResidualFeatures:
    """Degree residual: log(d_i + δ) − u_i − log(c · Σ_j w_j).

    Measures how much each node's observed degree in the noisy graph A_t
    deviates from what the rank-1 lognormal prior predicts given its
    current latent sociability u_t.

    Under the rank-1 prior:
        E[d_i | w] ≈ c · w_i · Σ_j w_j    (sparse regime)

    So the residual:
        r_i = log(d_i + δ) − log(c · w_i · Σ_j w_j)
            = log(d_i + δ) − log(c) − u_i − log(Σ_j w_j)

    is zero in expectation under the prior and non-zero whenever the
    node's local connectivity pattern deviates from rank-1 structure
    (assortativity, community membership, etc.).

    Key property: informative at ALL noise levels because w_i = exp(u_i)
    carries signal even when A_t is near-random at high t.

    Returns a single node feature (bs, n, 1). Requires 'u_t' in
    noisy_data; when absent (dimension-probe call) returns zeros of the
    correct shape.

    Args:
        c:         Edge rate constant matching the model (default 4.0).
        delta:     Laplace smoothing for log-degree (default 0.5).
        u_clamp:   Clamp applied to u_t before exp() (default 15.0).
    """

    def __init__(self, c: float = 4.0, delta: float = 0.5, u_clamp: float = 15.0):
        self.c = c
        self.delta = delta
        self.u_clamp = u_clamp

    def __call__(self, noisy_data):
        E_t = noisy_data['E_t']           # (bs, n, n, 2) one-hot
        node_mask = noisy_data['node_mask']  # (bs, n) bool
        bs, n = node_mask.shape

        # Observed degree from noisy adjacency (sum of edge-present channel)
        A = E_t[..., 1:].sum(dim=-1).float()   # (bs, n, n)
        d = (A * node_mask.float().unsqueeze(1)).sum(dim=2) * node_mask.float()  # (bs, n)

        u_t = noisy_data.get('u_t', None)

        if u_t is None:
            # Dimension probe — return correctly-shaped zero tensor
            r = E_t.new_zeros(bs, n, 1)
        else:
            w = torch.exp(u_t.clamp(-self.u_clamp, self.u_clamp)) * node_mask.float()  # (bs, n)

            # Σ_j w_j (sum over valid nodes per graph)
            sum_w = (w * node_mask.float()).sum(dim=1, keepdim=True).clamp(min=1e-8)  # (bs, 1)

            # log of rank-1 predicted degree: log(c · w_i · Σ_j w_j)
            # = log(c) + u_i + log(Σ_j w_j)
            log_pred = (
                torch.log(torch.tensor(self.c, device=E_t.device))
                + u_t.clamp(-self.u_clamp, self.u_clamp)
                + torch.log(sum_w)
            )  # (bs, n)

            log_obs = torch.log(d + self.delta)   # (bs, n)

            r = (log_obs - log_pred) * node_mask.float()   # (bs, n)
            r = r.unsqueeze(-1)                             # (bs, n, 1)

        empty_e = E_t.new_zeros(bs, n, n, 0)
        empty_y = E_t.new_zeros(bs, 0)
        return PlaceHolder(X=r, E=empty_e, y=empty_y)


class CRMDensityResidualFeatures:
    """Graph-level density residual: log(|E_t| / E_prior[edges | u_t]).

    Measures how much the current noisy graph's edge count deviates from
    what the rank-1 lognormal prior predicts given the current latents u_t.

    Under the rank-1 prior (sparse regime):
        E_prior[edges | w] ≈ (c/2) * [ (Σ_i w_i)^2 - Σ_i w_i^2 ]
                            = (c/2) * sum_w^2 * (1 - Σ w_i^2 / sum_w^2)

    This is O(n) to compute (no pairwise products needed).

    The residual:
        ρ_res = log(|E_t| + δ) − log(E_prior[edges | u_t] + δ)

    Properties:
    - Zero in expectation when the denoiser is on-distribution
    - Positive when the graph is denser than the prior predicts → "too many edges"
    - Negative when the graph is sparser than the prior predicts → "too few edges"
    - Informative at ALL noise levels because it uses u_t even when A_t is noisy
    - Acts as a density error signal the model cannot route around (unlike per-node
      degree residuals which can be satisfied by rewiring without changing density)

    Returns a single y (graph-level) feature (bs, 1). Requires 'u_t' in
    noisy_data; when absent (dimension-probe call) returns zeros.

    Args:
        c:        Edge rate constant matching the model (default 4.0).
        delta:    Laplace smoothing for log of edge counts (default 0.5).
        u_clamp:  Clamp applied to u_t before exp() (default 15.0).
    """

    def __init__(self, c: float = 4.0, delta: float = 0.5, u_clamp: float = 15.0):
        self.c = c
        self.delta = delta
        self.u_clamp = u_clamp

    def __call__(self, noisy_data):
        E_t = noisy_data['E_t']           # (bs, n, n, 2) one-hot
        node_mask = noisy_data['node_mask']  # (bs, n) bool
        bs, n = node_mask.shape

        # Observed edge count from noisy adjacency (upper triangle only, undirected)
        A = E_t[..., 1:].sum(dim=-1).float()   # (bs, n, n)
        # Count each undirected edge once: sum all, divide by 2
        mask2d = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)
        edge_count = (A * mask2d).sum(dim=[1, 2]) / 2.0   # (bs,)

        u_t = noisy_data.get('u_t', None)

        if u_t is None:
            # Dimension probe — return correctly-shaped zero scalar
            rho = E_t.new_zeros(bs, 1)
        else:
            w = torch.exp(u_t.clamp(-self.u_clamp, self.u_clamp)) * node_mask.float()  # (bs, n)

            # O(n) approximation for E_prior[edges | w]:
            #   ≈ (c/2) * [ (Σ w_i)^2 - Σ w_i^2 ]
            sum_w = w.sum(dim=1)           # (bs,)
            sum_w2 = (w ** 2).sum(dim=1)   # (bs,)
            expected_edges = (self.c / 2.0) * (sum_w ** 2 - sum_w2).clamp(min=0.0)  # (bs,)

            log_obs = torch.log(edge_count + self.delta)
            log_pred = torch.log(expected_edges + self.delta)

            rho = (log_obs - log_pred).unsqueeze(-1)   # (bs, 1)

        empty_x = E_t.new_zeros(bs, n, 0)
        empty_e = E_t.new_zeros(bs, n, n, 0)
        return PlaceHolder(X=empty_x, E=empty_e, y=rho)


class CRMPriorEdgeProbFeatures:
    """Prior edge probability p_ij as an edge feature.

    Computes the rank-1 lognormal prior's belief for each edge:

        p_ij = 1 - exp(-c * w_i * w_j)
             = 1 - exp(-c * exp(u_i) * exp(u_j))

    where u_i = u_t[i] is the current noisy sociability.

    This is the pre-computed bilinear interaction that the transformer
    would otherwise have to reconstruct across multiple layers from
    separate node features u_i, u_j.  Giving it explicitly means:

    - Every layer/head immediately knows the rank-1 prior's prediction
      for each edge, for free.
    - The model only needs to learn deviations from rank-1 (community
      structure, assortativity, structural holes) from the training data.
    - No covariate shift: p_ij depends only on u_t (not A_t), so it is
      as well-calibrated at t=T as at t=0, unlike graph-statistic
      features which collapse to near-constant at high noise.

    Returns a (bs, n, n, 1) edge feature in [0, 1].  Padding positions
    (where node_mask is False) are zeroed out.

    When u_t is absent (dimension-probe call), returns zeros of the
    correct shape.

    Args:
        c:        Edge rate constant (default 4.0, matches model).
        u_clamp:  Clamp applied to u_t before exp() (default 15.0).
    """

    def __init__(self, c: float = 4.0, u_clamp: float = 15.0):
        self.c = c
        self.u_clamp = u_clamp

    def __call__(self, noisy_data):
        E_t = noisy_data['E_t']           # (bs, n, n, 2)
        node_mask = noisy_data['node_mask']  # (bs, n) bool
        bs, n = node_mask.shape

        u_t = noisy_data.get('u_t', None)

        if u_t is None:
            # Dimension probe — correct shape, all zeros
            p = E_t.new_zeros(bs, n, n, 1)
        else:
            w = torch.exp(u_t.clamp(-self.u_clamp, self.u_clamp))  # (bs, n)

            # Outer product w_i * w_j  →  (bs, n, n)
            ww = w.unsqueeze(2) * w.unsqueeze(1)   # (bs, n, n)

            # p_ij = 1 - exp(-c * w_i * w_j)
            p = 1.0 - torch.exp(-self.c * ww)      # (bs, n, n), values in [0, 1]

            # Zero out padding rows/cols
            mask2d = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)   # (bs, n, n)
            p = p * mask2d.float()

            p = p.unsqueeze(-1)   # (bs, n, n, 1)

        empty_x = E_t.new_zeros(bs, n, 0)
        empty_y = E_t.new_zeros(bs, 0)
        return PlaceHolder(X=empty_x, E=p, y=empty_y)


class CRMResidualSpectralFeatures:
    """Bootstrapped residual spectral node features.

    Computes the top-K eigenvectors of the degree-normalised residual:

        R_tilde = D_Pi^{-1/2} (A_source - Pi) D_Pi^{-1/2}

    where Pi[i,j] = 1 - exp(-c * exp(u_i) * exp(u_j)) is the rank-1 prior
    prediction and A_source depends on context:

        Training:  A_0  (clean ground-truth graph, available from noisy_data)
        Inference: pred_A0_soft  (bootstrapped from model's previous-step
                   prediction of A_0, injected by sample_batch)

    Why use A_0 during training (not A_t):
    - A_t at high t is near-random → R̃ eigenvalues = random matrix noise ~1.15
      → the model learns to ignore the features entirely
    - A_0 is the true graph → R̃ captures genuine community/assortativity signal
      that is orthogonal to the rank-1 prior at every training timestep
    - The model learns "when eigenvectors have community structure, use them"
    - At inference, pred_A0 starts noisy (high t) and converges to A_0 (low t),
      giving a natural smooth handoff consistent with what the model was trained on.

    Sign convention: flip each eigenvector so the w_i-weighted sum is positive.
    This consistently assigns '+' to the community containing the highest hub.

    Adds K node dims.

    Args:
        c:         Edge rate constant (default 4.0, matches model).
        u_clamp:   Clamp for u_t before exp() (default 15.0).
        K:         Number of top eigenvectors to return (default 2).
        log_every: Print diagnostics every N __call__ invocations. 0 = disabled.
    """

    def __init__(self, c: float = 4.0, u_clamp: float = 15.0,
                 K: int = 2, log_every: int = 200):
        self.c = c
        self.u_clamp = u_clamp
        self.K = K
        self.log_every = log_every
        self._call_count = 0

    def __call__(self, noisy_data):
        E_t = noisy_data['E_t']           # (bs, n, n, 2)
        node_mask = noisy_data['node_mask']  # (bs, n) bool
        bs, n = node_mask.shape
        K = self.K

        u_t = noisy_data.get('u_t', None)

        if u_t is None:
            # Dimension probe — return correct shape, all zeros
            z = E_t.new_zeros(bs, n, K)
            empty_e = E_t.new_zeros(bs, n, n, 0)
            empty_y = E_t.new_zeros(bs, 0)
            return PlaceHolder(X=z, E=empty_e, y=empty_y)

        mask_float = node_mask.float()
        mask2d = mask_float.unsqueeze(2) * mask_float.unsqueeze(1)  # (bs, n, n)

        # A_t from one-hot (used as fallback and for density logging)
        A_t = E_t[..., 1:].sum(dim=-1).float() * mask2d   # (bs, n, n)

        # --- Select A_source for residual computation ---
        # Priority 1 — inference: bootstrapped pred_A0 from previous reverse step
        # Priority 2 — training: clean A_0 (available from apply_noise output)
        # Priority 3 — fallback (dim-probe already handled above via u_t=None)
        pred_A0 = noisy_data.get('pred_A0', None)
        A0_onehot = noisy_data.get('A_0_onehot', None)

        if pred_A0 is not None:
            # Inference: soft A_0 prediction — edge-present channel
            A_for_R = pred_A0[..., 1:].sum(dim=-1).float() * mask2d
            mode_str = 'infer(bootstrap)'
        elif A0_onehot is not None:
            # Training: use clean ground-truth A_0, not noisy A_t
            A_for_R = A0_onehot[..., 1:].sum(dim=-1).float() * mask2d
            mode_str = 'train(A_0)'
        else:
            # Should not reach here during normal training/inference
            A_for_R = A_t
            mode_str = 'fallback(A_t)'

        # Rank-1 prior prediction Pi(u_t)
        w = torch.exp(u_t.clamp(-self.u_clamp, self.u_clamp)) * mask_float  # (bs, n)
        ww = w.unsqueeze(2) * w.unsqueeze(1)                                 # (bs, n, n)
        Pi = (1.0 - torch.exp(-self.c * ww)) * mask2d                       # (bs, n, n)

        # Prior expected degree D_Pi[i] = sum_j Pi[i,j]
        d_pi = Pi.sum(dim=-1) * mask_float                  # (bs, n)
        d_inv_sqrt = (1.0 / d_pi.clamp(min=1e-4).sqrt()) * mask_float  # (bs, n)

        # Residual R = A_for_R - Pi, then mean-centre to remove global density bias.
        # Without this, a bootstrapped pred_A0 that overshoots density uniformly
        # produces a positive global offset in R → its eigenvectors carry a
        # "add more edges" bias → runaway density during inference.
        # Mean-centering ensures eigenvectors capture only *relative* community
        # structure (which node-pairs exceed Pi) not absolute density drift.
        R = (A_for_R - Pi) * mask2d
        n_pairs = mask2d.sum(dim=[1, 2]).clamp(min=1.0)                           # (bs,)
        R_mean = R.sum(dim=[1, 2]) / n_pairs                                      # (bs,)
        R = R - R_mean.view(-1, 1, 1) * mask2d                                   # remove global offset

        R_tilde = d_inv_sqrt.unsqueeze(2) * R * d_inv_sqrt.unsqueeze(1)  # (bs, n, n)
        R_tilde = (R_tilde + R_tilde.transpose(1, 2)) / 2.0              # enforce symmetry
        R_tilde = R_tilde * mask2d                                        # reapply after symmetrise

        # Push padded nodes to large negative eigenvalues so they don't
        # appear in the top-K (eigh returns ascending; we take last K cols)
        pad_neg = -2.0 * float(n)
        R_tilde = R_tilde + torch.diag_embed((~node_mask).float() * pad_neg)

        # Eigendecomposition
        try:
            eigvals, eigvecs = torch.linalg.eigh(R_tilde)  # ascending: (bs,n), (bs,n,n)
        except Exception:
            z = E_t.new_zeros(bs, n, K)
            empty_e = E_t.new_zeros(bs, n, n, 0)
            empty_y = E_t.new_zeros(bs, 0)
            return PlaceHolder(X=z, E=empty_e, y=empty_y)

        # Top K eigenvectors = last K columns; apply node mask
        top_vecs = eigvecs[:, :, -K:].clone() * mask_float.unsqueeze(-1)   # (bs, n, K)
        top_vals = eigvals[:, -K:].clone()                                  # (bs, K)

        # Sign normalisation: flip so that w-weighted sum per eigenvector > 0
        # (dominant community = community containing highest-sociability node gets '+')
        w_dot = (w.unsqueeze(-1) * top_vecs).sum(dim=1)   # (bs, K)
        signs = w_dot.sign()
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        top_vecs = top_vecs * signs.unsqueeze(1)           # (bs, n, K)

        node_features = top_vecs  # (bs, n, K)

        # --- Diagnostic logging ---
        self._call_count += 1
        if self.log_every > 0 and self._call_count % self.log_every == 1:
            t_val = noisy_data.get('t', None)
            t_str = f"t={t_val[0, 0].item():.3f}" if t_val is not None else "t=?"
            with torch.no_grad():
                top_ev_mean = top_vals[:, -1].mean().item()
                stds = []
                for b in range(min(bs, 4)):
                    n_b = int(node_mask[b].sum().item())
                    for k in range(K):
                        v = top_vecs[b, :n_b, k]
                        if n_b > 1:
                            stds.append(v.std().item())
                avg_std = float(sum(stds) / len(stds)) if stds else 0.0
                # density of A_for_R vs A_t (shows how much pred_A0 has refined)
                a_for_r_density = A_for_R[node_mask.unsqueeze(2) * node_mask.unsqueeze(1)].mean().item()
                a_t_density = A_t[mask2d.bool()].mean().item()
                r_mean_val = R_mean.mean().item()
                print(f"  [ResidualSpectral {t_str} mode={mode_str}] "
                      f"top_eigval={top_ev_mean:+.4f}  "
                      f"eigvec_std={avg_std:.4f}  "
                      f"A_for_R_density={a_for_r_density:.4f}  "
                      f"A_t_density={a_t_density:.4f}  "
                      f"R_mean={r_mean_val:+.4f}  "
                      f"K={K}  call#{self._call_count}")

        empty_e = E_t.new_zeros(bs, n, n, 0)
        empty_y = E_t.new_zeros(bs, 0)
        return PlaceHolder(X=node_features, E=empty_e, y=empty_y)


class ExtraFeatures:
    """Compute extra features for the diffusion model.

    Supported feature types:
      'cycles'      — raw cycle counts (k3–k6), divided by 10 and clamped to 1.
      'norm_cycles' — normalized cycles: k3 replaced by local clustering coefficient
                      (bounded [0,1]); k4/k5/k6 still divided by 10 and clamped.
      'eigenvalues' — raw cycles + 5 smallest combinatorial Laplacian eigenvalues.
      'all'         — raw cycles + eigenvalues + eigenvectors (LCC indicator + k=2 vecs).
      'norm_all'    — norm_cycles + eigenvalues + eigenvectors (same spectral as 'all').
    """

    def __init__(self, extra_features_type, dataset_info):
        self.max_n_nodes = dataset_info.max_n_nodes
        normalize = extra_features_type in ['norm_cycles', 'norm_all']
        self.ncycles = NodeCycleFeatures(normalize=normalize)
        self.features_type = extra_features_type
        # Map norm_all -> 'all' for EigenFeatures mode
        eigen_mode = 'all' if extra_features_type == 'norm_all' else extra_features_type
        if extra_features_type in ['eigenvalues', 'all', 'norm_all']:
            self.eigenfeatures = EigenFeatures(mode=eigen_mode)

    def __call__(self, noisy_data):
        n = noisy_data['node_mask'].sum(dim=1).unsqueeze(1) / self.max_n_nodes
        x_cycles, y_cycles = self.ncycles(noisy_data)

        if self.features_type in ('cycles', 'norm_cycles'):
            E = noisy_data['E_t']
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            return PlaceHolder(X=x_cycles, E=extra_edge_attr, y=torch.hstack((n, y_cycles)))

        elif self.features_type == 'eigenvalues':
            eigenfeatures = self.eigenfeatures(noisy_data)
            E = noisy_data['E_t']
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            n_components, batched_eigenvalues = eigenfeatures
            return PlaceHolder(X=x_cycles, E=extra_edge_attr, y=torch.hstack((n, y_cycles, n_components,
                                                                                batched_eigenvalues)))
        elif self.features_type in ('all', 'norm_all'):
            eigenfeatures = self.eigenfeatures(noisy_data)
            E = noisy_data['E_t']
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            n_components, batched_eigenvalues, nonlcc_indicator, k_lowest_eigvec = eigenfeatures

            return PlaceHolder(X=torch.cat((x_cycles, nonlcc_indicator, k_lowest_eigvec), dim=-1),
                               E=extra_edge_attr,
                               y=torch.hstack((n, y_cycles, n_components, batched_eigenvalues)))
        else:
            raise ValueError(f"Features type {self.features_type} not implemented")


class NodeCycleFeatures:
    """Compute cycle counts for nodes.

    When normalize=True (used by 'norm_cycles' / 'norm_all'):
      - k3x (per-node) is replaced by the local clustering coefficient
            c_i = 2 * triangles_i / (d_i * (d_i - 1))  in [0, 1]
      - k3y (graph-level) is replaced by the mean local clustering coefficient
      - k4x, k5x, k4y, k5y, k6y are still divided by 10 and clamped to [0, 1]
    """

    def __init__(self, normalize=False):
        self.kcycles = KNodeCycles()
        self.normalize = normalize

    def __call__(self, noisy_data):
        adj_matrix = noisy_data['E_t'][..., 1:].sum(dim=-1).float()
        node_mask = noisy_data['node_mask']

        x_cycles, y_cycles = self.kcycles.k_cycles(adj_matrix=adj_matrix)
        x_cycles = x_cycles.type_as(adj_matrix) * node_mask.unsqueeze(-1)

        if self.normalize:
            d = adj_matrix.sum(dim=-1)  # (bs, n)  — zero for padded nodes

            # Local clustering coefficient: c_i = 2*k3x_i / (d_i*(d_i-1)), bounded in [0,1]
            denom_3 = (d * (d - 1)).clamp(min=1.0).unsqueeze(-1)  # (bs, n, 1)
            k3x_norm = 2.0 * x_cycles[..., 0:1] / denom_3         # (bs, n, 1)

            k4x = (x_cycles[..., 1:2] / 10).clamp(max=1.0)
            k5x = (x_cycles[..., 2:3] / 10).clamp(max=1.0)
            x_cycles = torch.cat([k3x_norm, k4x, k5x], dim=-1)

            # Mean local clustering coefficient over valid nodes (graph-level)
            n_valid = node_mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            avg_clust = (k3x_norm.squeeze(-1) * node_mask.float()).sum(dim=1, keepdim=True) / n_valid
            k4y = (y_cycles[..., 1:2] / 10).clamp(max=1.0)
            k5y = (y_cycles[..., 2:3] / 10).clamp(max=1.0)
            k6y = (y_cycles[..., 3:4] / 10).clamp(max=1.0)
            y_cycles = torch.cat([avg_clust, k4y, k5y, k6y], dim=-1)
        else:
            # Avoid large values when the graph is dense
            x_cycles = x_cycles / 10
            y_cycles = y_cycles / 10
            x_cycles[x_cycles > 1] = 1
            y_cycles[y_cycles > 1] = 1

        return x_cycles, y_cycles


class EigenFeatures:
    """Compute eigenvalue-based features."""
    
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, noisy_data):
        E_t = noisy_data['E_t']
        mask = noisy_data['node_mask']
        A = E_t[..., 1:].sum(dim=-1).float() * mask.unsqueeze(1) * mask.unsqueeze(2)
        L = compute_laplacian(A, normalize=False)
        mask_diag = 2 * L.shape[-1] * torch.eye(A.shape[-1]).type_as(L).unsqueeze(0)
        mask_diag = mask_diag * (~mask.unsqueeze(1)) * (~mask.unsqueeze(2))
        L = L * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag

        if self.mode == 'eigenvalues':
            eigvals = torch.linalg.eigvalsh(L)
            eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)

            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigenvalues=eigvals)
            return n_connected_comp.type_as(A), batch_eigenvalues.type_as(A)

        elif self.mode == 'all':
            eigvals, eigvectors = torch.linalg.eigh(L)
            eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)
            eigvectors = eigvectors * mask.unsqueeze(2) * mask.unsqueeze(1)
            
            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigenvalues=eigvals)
            nonlcc_indicator, k_lowest_eigenvector = get_eigenvectors_features(
                vectors=eigvectors, node_mask=noisy_data['node_mask'], n_connected=n_connected_comp)
            return n_connected_comp, batch_eigenvalues, nonlcc_indicator, k_lowest_eigenvector
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented")


def compute_laplacian(adjacency, normalize: bool):
    """Compute graph Laplacian."""
    diag = torch.sum(adjacency, dim=-1)
    n = diag.shape[-1]
    D = torch.diag_embed(diag)
    combinatorial = D - adjacency

    if not normalize:
        return (combinatorial + combinatorial.transpose(1, 2)) / 2

    diag0 = diag.clone()
    diag[diag == 0] = 1e-12

    diag_norm = 1 / torch.sqrt(diag)
    D_norm = torch.diag_embed(diag_norm)
    L = torch.eye(n).unsqueeze(0) - D_norm @ adjacency @ D_norm
    L[diag0 == 0] = 0
    return (L + L.transpose(1, 2)) / 2


def get_eigenvalues_features(eigenvalues, k=5):
    """Get eigenvalue-based features."""
    ev = eigenvalues
    bs, n = ev.shape
    n_connected_components = (ev < 1e-5).sum(dim=-1)
    
    # Ensure at least one connected component (handle edge cases)
    n_connected_components = torch.clamp(n_connected_components, min=1)

    to_extend = max(n_connected_components) + k - n
    if to_extend > 0:
        eigenvalues = torch.hstack((eigenvalues, 2 * torch.ones(bs, to_extend).type_as(eigenvalues)))
    indices = torch.arange(k).type_as(eigenvalues).long().unsqueeze(0) + n_connected_components.unsqueeze(1)
    first_k_ev = torch.gather(eigenvalues, dim=1, index=indices)
    return n_connected_components.unsqueeze(-1), first_k_ev


def get_eigenvectors_features(vectors, node_mask, n_connected, k=2):
    """Get eigenvector-based features."""
    bs, n = vectors.size(0), vectors.size(1)

    first_ev = torch.round(vectors[:, :, 0], decimals=3) * node_mask
    random = torch.randn(bs, n, device=node_mask.device) * (~node_mask)
    first_ev = first_ev + random
    most_common = torch.mode(first_ev, dim=1).values
    mask = ~ (first_ev == most_common.unsqueeze(1))
    not_lcc_indicator = (mask * node_mask).unsqueeze(-1).float()

    to_extend = max(n_connected) + k - n
    if to_extend > 0:
        vectors = torch.cat((vectors, torch.zeros(bs, n, to_extend).type_as(vectors)), dim=2)
    indices = torch.arange(k).type_as(vectors).long().unsqueeze(0).unsqueeze(0) + n_connected.unsqueeze(2)
    indices = indices.expand(-1, n, -1)
    first_k_ev = torch.gather(vectors, dim=2, index=indices)
    first_k_ev = first_k_ev * node_mask.unsqueeze(2)

    return not_lcc_indicator, first_k_ev


def batch_trace(X):
    """Compute trace of batched matrices."""
    diag = torch.diagonal(X, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1)
    return trace


def batch_diagonal(X):
    """Extract diagonal from batched matrices."""
    return torch.diagonal(X, dim1=-2, dim2=-1)


class KNodeCycles:
    """Builds cycle counts for each node in a graph."""

    def __init__(self):
        super().__init__()

    def calculate_kpowers(self):
        self.k1_matrix = self.adj_matrix.float()
        self.d = self.adj_matrix.sum(dim=-1)
        self.k2_matrix = self.k1_matrix @ self.adj_matrix.float()
        self.k3_matrix = self.k2_matrix @ self.adj_matrix.float()
        self.k4_matrix = self.k3_matrix @ self.adj_matrix.float()
        self.k5_matrix = self.k4_matrix @ self.adj_matrix.float()
        self.k6_matrix = self.k5_matrix @ self.adj_matrix.float()

    def k3_cycle(self):
        """tr(A ** 3)."""
        c3 = batch_diagonal(self.k3_matrix)
        return (c3 / 2).unsqueeze(-1).float(), (torch.sum(c3, dim=-1) / 6).unsqueeze(-1).float()

    def k4_cycle(self):
        diag_a4 = batch_diagonal(self.k4_matrix)
        c4 = diag_a4 - self.d * (self.d - 1) - (self.adj_matrix @ self.d.unsqueeze(-1)).sum(dim=-1)
        return (c4 / 2).unsqueeze(-1).float(), (torch.sum(c4, dim=-1) / 8).unsqueeze(-1).float()

    def k5_cycle(self):
        diag_a5 = batch_diagonal(self.k5_matrix)
        triangles = batch_diagonal(self.k3_matrix)
        c5 = diag_a5 - 2 * triangles * self.d - (self.adj_matrix @ triangles.unsqueeze(-1)).sum(dim=-1) + triangles
        return (c5 / 2).unsqueeze(-1).float(), (c5.sum(dim=-1) / 10).unsqueeze(-1).float()

    def k6_cycle(self):
        term_1_t = batch_trace(self.k6_matrix)
        term_2_t = batch_trace(self.k3_matrix ** 2)
        term3_t = torch.sum(self.adj_matrix * self.k2_matrix.pow(2), dim=[-2, -1])
        d_t4 = batch_diagonal(self.k2_matrix)
        a_4_t = batch_diagonal(self.k4_matrix)
        term_4_t = (d_t4 * a_4_t).sum(dim=-1)
        term_5_t = batch_trace(self.k4_matrix)
        term_6_t = batch_trace(self.k3_matrix)
        term_7_t = batch_diagonal(self.k2_matrix).pow(3).sum(-1)
        term8_t = torch.sum(self.k3_matrix, dim=[-2, -1])
        term9_t = batch_diagonal(self.k2_matrix).pow(2).sum(-1)
        term10_t = batch_trace(self.k2_matrix)

        c6_t = (term_1_t - 3 * term_2_t + 9 * term3_t - 6 * term_4_t + 6 * term_5_t - 4 * term_6_t + 4 * term_7_t +
                3 * term8_t - 12 * term9_t + 4 * term10_t)
        return None, (c6_t / 12).unsqueeze(-1).float()

    def k_cycles(self, adj_matrix, verbose=False):
        self.adj_matrix = adj_matrix
        self.calculate_kpowers()

        k3x, k3y = self.k3_cycle()
        k4x, k4y = self.k4_cycle()
        k5x, k5y = self.k5_cycle()
        _, k6y = self.k6_cycle()
        
        # Clamp to zero to handle numerical precision issues
        # (cycle counts should theoretically be non-negative)
        k3x = torch.clamp(k3x, min=0.0)
        k3y = torch.clamp(k3y, min=0.0)
        k4x = torch.clamp(k4x, min=0.0)
        k4y = torch.clamp(k4y, min=0.0)
        k5x = torch.clamp(k5x, min=0.0)
        k5y = torch.clamp(k5y, min=0.0)
        k6y = torch.clamp(k6y, min=0.0)

        kcyclesx = torch.cat([k3x, k4x, k5x], dim=-1)
        kcyclesy = torch.cat([k3y, k4y, k5y, k6y], dim=-1)
        return kcyclesx, kcyclesy
