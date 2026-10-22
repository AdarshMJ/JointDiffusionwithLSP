"""
SparseDiffLSP: Sparse Diffusion with LogNormal Sociability Prior.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import utils as U
from noise_schedule import NoiseSchedule
from network import GraphTransformerConv
from lognormal_prior import (
    calibrate_lognormal_prior,
    sample_lognormal_terminal_batch,
    sample_lognormal_terminal_sparse,
)
from sample_edges import sample_query_edges, get_computational_graph
from extra_features import get_extra_features


class SparseDiffLSP(nn.Module):
    """Sparse Diffusion Model with LogNormal Sociability Prior.

    Key design:
      - Per-edge transition matrices Qt_bar from sociability weights w
      - Sparse training via query-edge subsampling
      - Joint continuous (u) + discrete (A) diffusion
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Diffusion params
        self.T = cfg.model.diffusion_steps
        self.edge_fraction = cfg.model.edge_fraction

        # LSP toggle
        self.use_lsp = cfg.model.get("use_lsp", True)

        # LSP params
        self.c = 4.0
        self.gamma = cfg.model.gamma              # tempering
        self.lambda_w = cfg.model.lambda_w         # latent loss weight
        self.u_clamp = cfg.model.u_clamp
        self.pred_param = cfg.model.pred_param     # 'x0' or 'eps'

        # Prior (calibrated later — or marginal q for non-LSP)
        self.prior_mu = None
        self.prior_sigma_scale = None
        self._prior_calibrated = False

        # Marginal-uniform terminal (non-LSP mode)
        self.marginal_q = None  # scalar: P(edge=1) from training data

        # Noise schedule
        self.noise_schedule = NoiseSchedule(self.T)

        # Extra features
        self.extra_features = get_extra_features(
            cfg.model.extra_features,
            num_eigenvectors=cfg.model.get("num_eigenvectors", 8),
            num_eigenvalues=cfg.model.get("num_eigenvalues", 5),
        )

        # Node count distribution (set after data loading)
        self.node_dist = None

        # Network will be built after we know input dims
        self._network_built = False
        self.model = None

    # ------------------------------------------------------------------
    #  Build network (called after calibration & data inspection)
    # ------------------------------------------------------------------
    def build_network(self, input_dims, output_dims):
        """Build the sparse Graph Transformer.

        Args:
            input_dims: dict {'X': dx_in, 'E': de_in, 'y': dy_in}
            output_dims: dict {'X': 1, 'E': 2}
        """
        cfg = self.cfg
        hidden_dims = {
            "dx": cfg.model.hidden_dims.dx,
            "de": cfg.model.hidden_dims.de,
            "dy": cfg.model.hidden_dims.dy,
            "n_head": cfg.model.hidden_dims.n_head,
            "dim_ffX": cfg.model.hidden_dims.dim_ffX,
            "dim_ffE": cfg.model.hidden_dims.dim_ffE,
            "dim_ffy": cfg.model.hidden_dims.get("dim_ffy", 128),
        }
        self.model = GraphTransformerConv(
            n_layers=cfg.model.n_layers,
            input_dims=input_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
            dropout=cfg.model.dropout,
        )
        self._network_built = True
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"  [SparseDiffLSP] Network built: {n_params:,} parameters")
        print(f"    input_dims  = {input_dims}")
        print(f"    output_dims = {output_dims}")

    # ------------------------------------------------------------------
    #  Prior calibration
    # ------------------------------------------------------------------
    def calibrate_prior(self, rho_data, n_ref, degree_sequences):
        """Calibrate mu_pi, sigma_pi from training data (LSP mode),
        or store marginal q (non-LSP mode)."""
        if self.use_lsp:
            mu, sigma = calibrate_lognormal_prior(
                rho_data=rho_data,
                degree_sequences=degree_sequences,
                c=self.c,
                verbose=True,
            )
            self.prior_mu = mu
            self.prior_sigma_scale = sigma
        else:
            # Marginal-uniform: q = average edge density
            self.marginal_q = rho_data
            print(f"  [SparseDiff] Marginal-uniform mode: q={self.marginal_q:.4f}")
        self._prior_calibrated = True

    def _ensure_calibrated(self):
        if not self._prior_calibrated:
            raise RuntimeError("Prior not calibrated. Call calibrate_prior() first.")

    # ------------------------------------------------------------------
    #  Latent <-> Weight conversions
    # ------------------------------------------------------------------
    @property
    def device(self):
        if self.model is not None:
            return next(self.model.parameters()).device
        return torch.device("cpu")

    def _u_to_w(self, u):
        return torch.exp(torch.clamp(u, -self.u_clamp, self.u_clamp))

    def _edge_prob_from_w(self, w, node_mask):
        """P(A_ij=1 | w) = 1 - exp(-c * w_i * w_j). Dense: (bs, n, n)."""
        ww = w.unsqueeze(2) * w.unsqueeze(1)
        prob1 = 1.0 - torch.exp(-self.c * ww)
        pair_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)
        return prob1 * pair_mask.float()

    # ------------------------------------------------------------------
    #  Per-edge transition matrices (dense)
    # ------------------------------------------------------------------
    def _build_Qt_bar(self, alpha_bar, w, node_mask):
        """Cumulative per-edge transition: Qt_bar = ab*I + (1-ab)*1*p_ij^T.
        Returns: (bs, n, n, 2, 2)
        """
        bs, n = w.shape
        prob1 = self._edge_prob_from_w(w, node_mask)
        prob0 = 1.0 - prob1
        ab = alpha_bar.view(bs, 1, 1)

        q00 = ab + (1.0 - ab) * prob0
        q01 = (1.0 - ab) * prob1
        q10 = (1.0 - ab) * prob0
        q11 = ab + (1.0 - ab) * prob1

        row0 = torch.stack([q00, q01], dim=-1)
        row1 = torch.stack([q10, q11], dim=-1)
        return torch.stack([row0, row1], dim=-2)

    def _build_Qt(self, beta_t, w, node_mask):
        """Single-step: Qt = (1-bt)*I + bt*1*p_ij^T. Returns: (bs,n,n,2,2)"""
        bs, n = w.shape
        prob1 = self._edge_prob_from_w(w, node_mask)
        prob0 = 1.0 - prob1
        bt = beta_t.view(bs, 1, 1)

        q00 = (1.0 - bt) + bt * prob0
        q01 = bt * prob1
        q10 = bt * prob0
        q11 = (1.0 - bt) + bt * prob1

        row0 = torch.stack([q00, q01], dim=-1)
        row1 = torch.stack([q10, q11], dim=-1)
        return torch.stack([row0, row1], dim=-2)

    # --- Marginal-uniform versions (non-LSP) ---
    def _build_Qt_bar_marginal(self, alpha_bar, node_mask):
        """Global marginal Qt_bar = ab*I + (1-ab)*1*q^T. Returns (bs,n,n,2,2)."""
        bs, n = node_mask.shape
        q1 = self.marginal_q
        q0 = 1.0 - q1
        ab = alpha_bar.view(bs, 1, 1)

        q00 = ab + (1.0 - ab) * q0
        q01 = (1.0 - ab) * q1
        q10 = (1.0 - ab) * q0
        q11 = ab + (1.0 - ab) * q1

        row0 = torch.stack([q00.expand(bs, n, n), q01.expand(bs, n, n)], dim=-1)
        row1 = torch.stack([q10.expand(bs, n, n), q11.expand(bs, n, n)], dim=-1)
        return torch.stack([row0, row1], dim=-2)

    def _build_Qt_marginal(self, beta_t, node_mask):
        """Single-step marginal Qt. Returns (bs,n,n,2,2)."""
        bs, n = node_mask.shape
        q1 = self.marginal_q
        q0 = 1.0 - q1
        bt = beta_t.view(bs, 1, 1)

        q00 = (1.0 - bt) + bt * q0
        q01 = bt * q1
        q10 = bt * q0
        q11 = (1.0 - bt) + bt * q1

        row0 = torch.stack([q00.expand(bs, n, n), q01.expand(bs, n, n)], dim=-1)
        row1 = torch.stack([q10.expand(bs, n, n), q11.expand(bs, n, n)], dim=-1)
        return torch.stack([row0, row1], dim=-2)

    # ------------------------------------------------------------------
    #  Derive u_0 from data
    # ------------------------------------------------------------------
    def _derive_u0_from_data(self, A_onehot, node_mask):
        """u_0 = mu_pi + (log(d_i + delta) - mean(log(d_k + delta))).
        A_onehot: (bs, n, n, 2), node_mask: (bs, n)
        Returns: (bs, n)
        """
        mu_pi = self.prior_mu
        A_class = A_onehot.argmax(dim=-1).float()
        A_class = A_class * node_mask.unsqueeze(1).float()
        degrees = A_class.sum(dim=2) * node_mask.float()

        delta = 0.05
        log_d = torch.log((degrees + delta).clamp(min=1e-8))
        n_valid = node_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        mean_log_d = (log_d * node_mask.float()).sum(dim=1, keepdim=True) / n_valid
        tilde_u = (log_d - mean_log_d) * node_mask.float()

        return (mu_pi + tilde_u) * node_mask.float()

    # ------------------------------------------------------------------
    #  Forward diffusion (dense, for noise application)
    # ------------------------------------------------------------------
    def apply_noise_dense(self, A_onehot, y, node_mask):
        """Apply forward noise to produce (u_t, A_t) — dense tensors.

        Args:
            A_onehot: (bs, n, n, 2)
            y: (bs, dy)
            node_mask: (bs, n)

        Returns:
            noisy_data dict with dense u_t, E_t, etc.
        """
        self._ensure_calibrated()
        bs, n = node_mask.shape

        # Sample timestep
        t_int = torch.randint(1, self.T + 1, (bs, 1), device=self.device).float()
        t_float = t_int / self.T

        beta_t = self.noise_schedule(t_int)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int)

        if self.use_lsp:
            # --- LSP mode: continuous u_t + per-edge Qt_bar ---
            mu_pi = self.prior_mu
            sigma_pi = self.prior_sigma_scale

            u_0 = self._derive_u0_from_data(A_onehot, node_mask)
            eps = torch.randn(bs, n, device=self.device)

            sqrt_ab = torch.sqrt(alpha_t_bar).view(bs, 1)
            sqrt_1m_ab = torch.sqrt((1.0 - alpha_t_bar).clamp(min=1e-8)).view(bs, 1)
            u_t = mu_pi + sqrt_ab * (u_0 - mu_pi) + sigma_pi * sqrt_1m_ab * eps
            u_t = u_t * node_mask.float()

            w_0 = self._u_to_w(u_0)
            Qtb = self._build_Qt_bar(alpha_t_bar, w_0, node_mask)
        else:
            # --- Marginal-uniform mode: no u, global Qt_bar ---
            u_0 = torch.zeros(bs, n, device=self.device)
            u_t = torch.zeros(bs, n, device=self.device)
            eps = torch.zeros(bs, n, device=self.device)
            Qtb = self._build_Qt_bar_marginal(alpha_t_bar, node_mask)

        # Discrete forward
        prob_At = torch.einsum("bijk,bijkl->bijl", A_onehot.float(), Qtb)
        prob_At = torch.clamp(prob_At, min=1e-8)
        prob_At = prob_At / prob_At.sum(dim=-1, keepdim=True)

        A_t = self._sample_edges_dense(prob_At, node_mask)
        A_t_onehot = F.one_hot(A_t, num_classes=2).float()
        pair_mask = (node_mask.unsqueeze(2) * node_mask.unsqueeze(1)).unsqueeze(-1)
        A_t_onehot = A_t_onehot * pair_mask.float()

        return {
            "t_int": t_int,
            "t": t_float,
            "beta_t": beta_t,
            "alpha_t_bar": alpha_t_bar,
            "u_0": u_0,
            "u_t": u_t,
            "eps": eps,
            "E_t": A_t_onehot,
            "A_0_onehot": A_onehot,
            "X_t": node_mask.unsqueeze(-1).float(),
            "y_t": y,
            "node_mask": node_mask,
        }

    def _sample_edges_dense(self, prob_E, node_mask):
        """Sample edges from (bs, n, n, 2) probabilities. Returns (bs, n, n) long."""
        bs, n, _, _ = prob_E.shape
        inv_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
        diag = torch.eye(n, device=prob_E.device).bool().unsqueeze(0).expand(bs, -1, -1)
        prob_E[inv_mask] = torch.tensor([1.0, 0.0], device=prob_E.device)
        prob_E[diag] = torch.tensor([1.0, 0.0], device=prob_E.device)

        flat = prob_E.reshape(bs * n * n, 2)
        sampled = flat.multinomial(1).reshape(bs, n, n)
        sampled = torch.triu(sampled, diagonal=1)
        sampled = sampled + sampled.transpose(1, 2)
        return sampled

    # ------------------------------------------------------------------
    #  Dense -> Sparse conversion for noisy data
    # ------------------------------------------------------------------
    def _densify_to_sparse(self, noisy_data):
        """Convert dense noisy data to sparse PyG format for the network.
        Returns dict with sparse node_t, edge_index_t, edge_attr_t, batch, ptr.
        """
        E_t = noisy_data["E_t"]           # (bs, n, n, 2)
        node_mask = noisy_data["node_mask"]  # (bs, n)
        u_t = noisy_data["u_t"]            # (bs, n)

        bs, n = node_mask.shape
        device = E_t.device

        all_node = []
        all_u = []
        all_edge_index = []
        all_edge_attr = []
        all_batch = []
        offset = 0

        for i in range(bs):
            n_i = int(node_mask[i].sum().item())
            # Node features (constant 1)
            all_node.append(torch.ones(n_i, 1, device=device))
            all_u.append(u_t[i, :n_i])
            all_batch.append(torch.full((n_i,), i, device=device, dtype=torch.long))

            # Extract edges (only class 1)
            Ei = E_t[i, :n_i, :n_i].argmax(dim=-1)
            src, dst = Ei.nonzero(as_tuple=True)
            if src.shape[0] > 0:
                all_edge_index.append(torch.stack([src + offset, dst + offset]))
                edge_attr_i = torch.zeros(src.shape[0], 2, device=device)
                edge_attr_i[:, 1] = 1.0
                all_edge_attr.append(edge_attr_i)

            offset += n_i

        node = torch.cat(all_node)
        u_sparse = torch.cat(all_u)
        batch = torch.cat(all_batch)

        if len(all_edge_index) > 0:
            edge_index = torch.cat(all_edge_index, dim=1)
            edge_attr = torch.cat(all_edge_attr)
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            edge_attr = torch.zeros(0, 2, device=device)

        ptr = torch.zeros(bs + 1, dtype=torch.long, device=device)
        for i in range(bs):
            ptr[i + 1] = ptr[i] + int(node_mask[i].sum().item())

        return {
            "node_t": node,
            "u_t": u_sparse,
            "edge_index_t": edge_index,
            "edge_attr_t": edge_attr,
            "batch": batch,
            "ptr": ptr,
        }

    # ------------------------------------------------------------------
    #  Compute extra features (dense, then extract for nodes)
    # ------------------------------------------------------------------
    def compute_extra_data(self, noisy_data):
        """Compute extra features from dense noisy data.
        Returns PlaceHolder with dense X, E, y features.
        """
        extra = self.extra_features(noisy_data)
        t = noisy_data["t"]
        extra_y = torch.cat([extra.y, t], dim=1)
        return U.PlaceHolder(X=extra.X, E=extra.E, y=extra_y)

    # ------------------------------------------------------------------
    #  Training step
    # ------------------------------------------------------------------
    def training_step(self, data, step=0, log_every=50):
        """Single training step with sparse edge subsampling.

        Returns: loss tensor
        """
        if not self._network_built:
            raise RuntimeError("Call build_network() before training.")

        # Convert to dense
        dense_data, node_mask = U.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        dense_data = dense_data.mask(node_mask)
        E = dense_data.E
        y = data.y if hasattr(data, "y") and data.y is not None else \
            torch.zeros(node_mask.size(0), 0, device=self.device)

        # Forward noise (dense)
        noisy_data = self.apply_noise_dense(E, y, node_mask)

        # Extra features (dense)
        extra_data = self.compute_extra_data(noisy_data)

        # Convert to sparse
        sparse_nd = self._densify_to_sparse(noisy_data)

        # Sample query edges
        n_per_graph = noisy_data["node_mask"].sum(dim=1).long()
        triu_query, _ = sample_query_edges(n_per_graph, self.edge_fraction)
        triu_query = triu_query.to(self.device)

        # Build computational graph
        query_mask, comp_edge_index, comp_edge_attr = get_computational_graph(
            triu_query, sparse_nd["edge_index_t"], sparse_nd["edge_attr_t"]
        )

        # Build node input
        bs, n_max = node_mask.shape
        N_total = sparse_nd["u_t"].shape[0]

        # Extract dense extra features for valid nodes
        extra_X_flat = []
        for i in range(bs):
            n_i = int(node_mask[i].sum().item())
            extra_X_flat.append(extra_data.X[i, :n_i])
        if bs > 0 and extra_data.X.shape[-1] > 0:
            extra_X_sparse = torch.cat(extra_X_flat, dim=0)  # (N, dx_extra)
        else:
            extra_X_sparse = sparse_nd["u_t"].new_zeros(N_total, 0)

        if self.use_lsp:
            u_sparse = sparse_nd["u_t"].unsqueeze(-1)  # (N, 1)
            X_input = torch.cat([u_sparse, extra_X_sparse], dim=-1)
        else:
            X_input = extra_X_sparse if extra_X_sparse.shape[-1] > 0 else \
                torch.ones(N_total, 1, device=self.device)

        # Build y input: [extra_y]  (per graph)
        y_input = extra_data.y.float()

        # Forward through network
        pred = self.model(
            X=X_input,
            edge_attr=comp_edge_attr,
            edge_index=comp_edge_index,
            y=y_input,
            batch=sparse_nd["batch"],
        )

        # --- Loss L_w: latent MSE (LSP only) ---
        if self.use_lsp:
            raw_pred_u = pred.node.squeeze(-1)  # (N,)
            if self.pred_param == "eps":
                target_w = self._extract_sparse_target(noisy_data["eps"], node_mask)
            else:
                target_w = self._extract_sparse_target(noisy_data["u_0"], node_mask)
            loss_w = ((raw_pred_u - target_w) ** 2).mean()
        else:
            loss_w = torch.tensor(0.0, device=self.device)

        # --- Loss L_A: edge cross-entropy on QUERY edges only ---
        query_pred_E = pred.edge_attr[query_mask]    # (Q, 2) logits
        query_edge_index = comp_edge_index[:, query_mask]

        # Get ground truth for query edges from clean A_0
        A_0_onehot = noisy_data["A_0_onehot"]  # (bs, n, n, 2)
        true_classes = self._get_true_edge_classes(
            query_edge_index, A_0_onehot, node_mask, sparse_nd["ptr"]
        )

        if query_pred_E.shape[0] > 0:
            loss_A = F.cross_entropy(query_pred_E, true_classes)
        else:
            loss_A = torch.tensor(0.0, device=self.device)

        # Scale edge loss by 1/lambda (unbiased estimator)
        loss = (1.0 / self.edge_fraction) * loss_A + self.lambda_w * loss_w

        if step % log_every == 0:
            print(f"  [step {step}] L_A={loss_A.item():.4f} "
                  f"L_w={loss_w.item():.4f} total={loss.item():.4f}")

        return loss

    def _extract_sparse_target(self, dense_target, node_mask):
        """Extract valid-node values from (bs, n) dense tensor."""
        bs, n = node_mask.shape
        vals = []
        for i in range(bs):
            n_i = int(node_mask[i].sum().item())
            vals.append(dense_target[i, :n_i])
        return torch.cat(vals)

    def _get_true_edge_classes(self, query_edge_index, A_0_onehot, node_mask, ptr):
        """Look up ground-truth edge class for each query edge.

        query_edge_index: (2, Q) global node indices
        A_0_onehot: (bs, n, n, 2) dense clean edges
        Returns: (Q,) long class labels
        """
        Q = query_edge_index.shape[1]
        src = query_edge_index[0]
        dst = query_edge_index[1]
        bs = ptr.shape[0] - 1

        # Determine which graph each edge belongs to
        edge_graph = torch.zeros(Q, dtype=torch.long, device=src.device)
        for g in range(bs):
            mask = (src >= ptr[g]) & (src < ptr[g + 1])
            edge_graph[mask] = g

        # Local indices
        local_src = src - ptr[edge_graph]
        local_dst = dst - ptr[edge_graph]

        # Lookup
        classes = A_0_onehot[edge_graph, local_src, local_dst].argmax(dim=-1)
        return classes

    # ------------------------------------------------------------------
    #  Validation step
    # ------------------------------------------------------------------
    @torch.no_grad()
    def validation_step(self, data):
        """Compute validation loss (same as training, no grad)."""
        return self.training_step(data, step=-1, log_every=999999).item()

    # ------------------------------------------------------------------
    #  Generation (reverse process)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_batch(self, batch_size, num_nodes=None):
        """Generate a batch of graphs.

        Args:
            batch_size: int
            num_nodes: int or (bs,) tensor, or None (sample from training dist)

        Returns:
            List of (node_types, adj_matrix) tuples
        """
        self._ensure_calibrated()

        # Number of nodes
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif isinstance(num_nodes, int):
            n_nodes = torch.full((batch_size,), num_nodes, device=self.device, dtype=torch.long)
        else:
            n_nodes = num_nodes.to(self.device)

        n_max = int(n_nodes.max().item())
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)

        # --- Terminal sampling ---
        if self.use_lsp:
            mu_pi = self.prior_mu
            sigma_pi = self.prior_sigma_scale
            A_init, w_T, u_T = sample_lognormal_terminal_batch(
                batch_size=batch_size,
                n_nodes=n_nodes,
                mu_pi=mu_pi,
                sigma_pi=sigma_pi,
                c=self.c,
                device=self.device,
            )
            u_T = torch.clamp(u_T, -self.u_clamp, self.u_clamp) * node_mask.float()
            u_t = u_T
        else:
            # Marginal-uniform terminal: each edge ~ Bernoulli(q)
            q = self.marginal_q
            pair_mask = node_mask.unsqueeze(2).float() * node_mask.unsqueeze(1).float()
            prob = torch.full((batch_size, n_max, n_max), q, device=self.device) * pair_mask
            prob.diagonal(dim1=1, dim2=2).zero_()
            # Sample symmetric
            upper = torch.bernoulli(torch.triu(prob, diagonal=1))
            A_init = (upper + upper.transpose(1, 2)).long()
            u_t = torch.zeros(batch_size, n_max, device=self.device)

        A_t_onehot = F.one_hot(A_init, num_classes=2).float()

        # Dummy y
        y = torch.zeros(batch_size, 0, device=self.device)

        # --- Reverse loop ---
        for s_int in tqdm(reversed(range(0, self.T)), total=self.T, desc="Sampling"):
            s_array = torch.full((batch_size, 1), s_int, device=self.device, dtype=torch.float)
            t_array = s_array + 1
            t_float = t_array / self.T

            beta_t = self.noise_schedule(t_array)
            alpha_s_bar = self.noise_schedule.get_alpha_bar(s_array)
            alpha_t_bar = self.noise_schedule.get_alpha_bar(t_array)

            # Build noisy_data dict (dense)
            noisy_data = {
                "u_t": u_t,
                "E_t": A_t_onehot,
                "X_t": node_mask.unsqueeze(-1).float(),
                "y_t": y,
                "t": t_float,
                "t_int": t_array,
                "node_mask": node_mask,
            }

            # Extra features
            extra_data = self.compute_extra_data(noisy_data)

            # Multi-pass: process edges in ceil(1/lambda) passes
            n_passes = math.ceil(1.0 / self.edge_fraction)

            # We do a single pass through the network for simplicity
            # (full computational graph with all current edges + query)
            sparse_nd = self._densify_to_sparse(noisy_data)

            # Sample ALL edges as queries (full pass for generation)
            n_per_graph = node_mask.sum(dim=1).long()
            triu_query, _ = sample_query_edges(n_per_graph, 1.0)
            triu_query = triu_query.to(self.device)

            query_mask, comp_edge_index, comp_edge_attr = get_computational_graph(
                triu_query, sparse_nd["edge_index_t"], sparse_nd["edge_attr_t"]
            )

            # Node input
            total_nodes = sparse_nd["batch"].shape[0]
            bs_n = node_mask.shape[0]
            extra_X_flat = []
            for i in range(bs_n):
                n_i = int(node_mask[i].sum().item())
                extra_X_flat.append(extra_data.X[i, :n_i])
            if extra_data.X.shape[-1] > 0:
                extra_X_sparse = torch.cat(extra_X_flat, dim=0)
            else:
                extra_X_sparse = torch.zeros(total_nodes, 0, device=self.device)

            if self.use_lsp:
                u_sparse = sparse_nd["u_t"].unsqueeze(-1)
                X_input = torch.cat([u_sparse, extra_X_sparse], dim=-1)
            else:
                if extra_X_sparse.shape[-1] > 0:
                    X_input = extra_X_sparse
                else:
                    X_input = torch.ones(total_nodes, 1, device=self.device)
            y_input = extra_data.y.float()

            pred = self.model(
                X=X_input,
                edge_attr=comp_edge_attr,
                edge_index=comp_edge_index,
                y=y_input,
                batch=sparse_nd["batch"],
            )

            # --- A) Reverse continuous step ---
            if self.use_lsp:
                raw_hat = pred.node.squeeze(-1)  # (N,) — one value per node

                # Scatter back to dense (bs, n)
                u_0_hat_dense = torch.zeros_like(u_t)
                offset = 0
                for i in range(batch_size):
                    n_i = int(node_mask[i].sum().item())
                    u_0_hat_dense[i, :n_i] = raw_hat[offset: offset + n_i]
                    offset += n_i

                if self.pred_param == "eps":
                    mu_pi = self.prior_mu
                    sigma_pi = self.prior_sigma_scale
                    sqrt_ab_t = torch.sqrt(alpha_t_bar.clamp(min=1e-8)).view(batch_size, 1)
                    sqrt_1m_ab_t = torch.sqrt((1 - alpha_t_bar).clamp(min=1e-8)).view(batch_size, 1)
                    eps_hat = torch.clamp(u_0_hat_dense, -5.0, 5.0) * node_mask.float()
                    u_0_hat_dense = mu_pi + (u_t - mu_pi - sigma_pi * sqrt_1m_ab_t * eps_hat) / sqrt_ab_t
                u_0_hat_dense = torch.clamp(u_0_hat_dense, -self.u_clamp, self.u_clamp) * node_mask.float()

                # DDPM posterior
                mu_pi = self.prior_mu
                sigma_pi = self.prior_sigma_scale
                ab_t = alpha_t_bar.view(batch_size, 1)
                ab_s = alpha_s_bar.view(batch_size, 1)
                bt = beta_t.view(batch_size, 1)
                alpha_t = (1.0 - bt)
                sqrt_alpha_t = torch.sqrt(alpha_t.clamp(min=1e-8))
                sqrt_ab_s = torch.sqrt(ab_s.clamp(min=0.0))
                denom = (1.0 - ab_t).clamp(min=1e-8)

                coef_x0 = sqrt_ab_s * bt / denom
                coef_xt = sqrt_alpha_t * (1.0 - ab_s) / denom
                mu = mu_pi + coef_x0 * (u_0_hat_dense - mu_pi) + coef_xt * (u_t - mu_pi)

                if s_int > 0:
                    sigma2 = bt * (1.0 - ab_s) / denom
                    sigma = sigma_pi * torch.sqrt(sigma2.clamp(min=1e-8, max=1.0))
                    u_next = mu + sigma * torch.randn_like(mu)
                else:
                    u_next = mu

                u_next = torch.clamp(u_next, -self.u_clamp, self.u_clamp) * node_mask.float()
            else:
                # Non-LSP: no continuous reverse, u_t stays zero
                u_0_hat_dense = torch.zeros_like(u_t)
                u_next = u_t

            # --- B) Reverse discrete step (tempered) ---
            # Scatter edge predictions back to dense
            pred_E_dense = self._scatter_edge_pred_to_dense(
                pred.edge_attr, comp_edge_index, batch_size, n_max,
                node_mask, sparse_nd["ptr"]
            )
            pred_p0 = F.softmax(pred_E_dense, dim=-1)  # (bs, n, n, 2)

            # Build transition matrices
            if self.use_lsp:
                w_0_hat = self._u_to_w(u_0_hat_dense)
                Qt = self._build_Qt(beta_t, w_0_hat, node_mask)
                Qtb = self._build_Qt_bar(alpha_t_bar, w_0_hat, node_mask)
                Qsb = self._build_Qt_bar(alpha_s_bar, w_0_hat, node_mask)
            else:
                Qt = self._build_Qt_marginal(beta_t, node_mask)
                Qtb = self._build_Qt_bar_marginal(alpha_t_bar, node_mask)
                Qsb = self._build_Qt_bar_marginal(alpha_s_bar, node_mask)

            A_t_onehot = self._reverse_discrete_step(
                A_t_onehot, pred_p0, Qt, Qsb, Qtb, node_mask
            )

            u_t = u_next

        # Collapse to output
        A_final = torch.argmax(A_t_onehot, dim=-1)
        results = []
        for i in range(batch_size):
            n = int(n_nodes[i].item())
            atom_types = torch.ones(n, 1)
            edge_types = A_final[i, :n, :n].cpu()
            results.append([atom_types, edge_types])

        return results

    def _scatter_edge_pred_to_dense(self, edge_attr, edge_index, bs, n_max,
                                     node_mask, ptr):
        """Convert sparse edge predictions to dense (bs, n, n, 2) tensor.
        Edges not in computational graph get uniform logits [0, 0].
        """
        dense = torch.zeros(bs, n_max, n_max, 2, device=edge_attr.device)

        src = edge_index[0]
        dst = edge_index[1]

        # Find graph assignment
        for g in range(bs):
            mask = (src >= ptr[g]) & (src < ptr[g + 1])
            local_src = src[mask] - ptr[g]
            local_dst = dst[mask] - ptr[g]
            dense[g, local_src, local_dst] = edge_attr[mask]

        # Symmetrise
        dense = (dense + dense.transpose(1, 2)) / 2.0
        return dense

    def _reverse_discrete_step(self, A_t_oh, pred_p0, Qt, Qsb, Qtb, node_mask):
        """Tempered reverse discrete posterior.

        log p~(x_s) = gamma * [log Qt(A_t|x_s) + log sum_x0 Qsb(x_s|x0)*p0(x0)
                                                - log sum_x0 Qtb(A_t|x0)*p0(x0)]
        """
        bs, n = node_mask.shape
        A_t_class = torch.argmax(A_t_oh, dim=-1)

        # Term 1: log Qt(A_t | x_s) for each candidate x_s in {0,1}
        A_t_idx = A_t_class.unsqueeze(-1).unsqueeze(-2).expand(-1, -1, -1, 2, -1)
        log_Qt_At = torch.log(
            torch.gather(Qt, dim=-1, index=A_t_idx).squeeze(-1) + 1e-30
        )

        # Term 2: log sum_x0 Qsb(x_s | x0) * p0(x0)
        p0_exp = pred_p0.unsqueeze(-2)
        Qsb_weighted = torch.matmul(p0_exp, Qsb).squeeze(-2)
        log_Qsb_term = torch.log(Qsb_weighted + 1e-30)

        # Term 3: log sum_x0 Qtb(A_t | x0) * p0(x0) (normalizer)
        Qtb_At = torch.gather(Qtb, dim=-1, index=A_t_idx).squeeze(-1)
        denom = (pred_p0 * Qtb_At).sum(dim=-1, keepdim=True)
        log_denom = torch.log(denom + 1e-30)

        # Combine with tempering
        log_unnorm = self.gamma * (log_Qt_At + log_Qsb_term - log_denom)
        log_prob = log_unnorm - torch.logsumexp(log_unnorm, dim=-1, keepdim=True)
        prob = torch.exp(log_prob)
        prob = torch.clamp(prob, min=1e-8)
        prob = prob / prob.sum(dim=-1, keepdim=True)

        A_s = self._sample_edges_dense(prob, node_mask)
        A_s_oh = F.one_hot(A_s, num_classes=2).float()
        pair_mask = (node_mask.unsqueeze(2) * node_mask.unsqueeze(1)).unsqueeze(-1)
        return A_s_oh * pair_mask.float()
