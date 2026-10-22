"""
Joint Continuous-Discrete Diffusion Model for Graph Generation.
Replaces the standard Erdős–Rényi terminal distribution with a Norros-Reittu Model.
Jointly diffuses:
  1. Continuous latent node variables u_t in R^n (log-space sociabilities)
  2. Discrete adjacency matrix A_t in {0,1}^{n x n}

Edge probabilities conditioned on latents:
  P(A_ij=1 | w) = 1 - exp(-c * w_i * w_j),  where w = exp(u)
"""
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer_model import GraphTransformer
from diffusion.noise_schedule import PredefinedNoiseScheduleDiscrete
from diffusion import diffusion_utils
from diffusion.extra_features import DummyExtraFeatures
from crm_prior.terminal import sample_terminal_batch, estimate_alpha_for_n
from crm_prior.calibrate import calibrate_prior_stats, solve_tau
from crm_prior.lognormal_prior import (
    calibrate_lognormal_prior, sample_lognormal_terminal_batch
)
import utils


class CRMDiffusion(nn.Module):
    """
    Combines continuous DDPM diffusion on latent sociabilities with
    discrete edge diffusion using per-edge transition matrices conditioned
    on the latent weights.
    """

    def __init__(self, cfg, dataset_infos, sampling_metrics,
                 visualization_tools, extra_features, domain_features):
        super().__init__()

        self.cfg = cfg
        self.name = cfg.general.name
        self.T = cfg.model.diffusion_steps

        # CRM hyper-parameters
        crm = cfg.model.crm
        self.c = 4.0                    # fixed at 4 (spec v2)
        self.gamma = crm.gamma          # tempering parameter
        self.lambda_w = crm.lambda_w    # weight for latent MSE loss
        self.u_clamp = crm.u_clamp      # clamping range for u_t

        # CRM prior parameters
        prior = getattr(crm, 'prior', None)
        if prior is not None:
            self.prior_family = getattr(prior, 'family', 'lognormal')
            # GGP-specific params (kept for backward compat / checkpoint loading)
            self.prior_alpha = getattr(prior, 'alpha', 1.0)
            self._config_alpha = self.prior_alpha
            self.prior_sigma = getattr(prior, 'sigma', 0.4)
            self.prior_tau = None
            self.prior_eps = getattr(prior, 'eps', 1e-4)
            self.prior_max_tries = getattr(prior, 'max_tries', 100)
            self.prior_alpha_growth = getattr(prior, 'alpha_growth', 1.2)
            self.use_crm_prior = True

            # Shifted-scaled DDPM: prior statistics
            # Calibrated from training data (LogNormal: direct; GGP: via solve_tau)
            self.prior_mu = None
            self.prior_sigma_scale = None
            self._prior_calibrated = False
        else:
            # Legacy: fall back to i.i.d. lognormal (old behavior)
            self.use_crm_prior = False
            self.prior_mu = 0.0
            self.prior_sigma_scale = 1.0
            self.prior_tau = 1.0
            self._prior_calibrated = True

        # Dataset info
        self.dataset_info = dataset_infos
        self.node_dist = dataset_infos.nodes_dist

        # Metrics & visualisation
        self.sampling_metrics = sampling_metrics
        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        # ------------------------------------------------------------------
        # Network: Input / Output dimensions
        # ------------------------------------------------------------------
        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims

        # Store for convenience
        self.Xdim_input = input_dims['X']
        self.Edim_input = input_dims['E']
        self.ydim_input = input_dims['y']
        self.Edim_output = output_dims['E']   # 2 (binary edges)
        self.Xdim_output = output_dims['X']   # 1 (epsilon prediction dim)

        # Build transformer
        skip_x = getattr(crm, 'skip_x', False)
        self.pred_param = getattr(crm, 'pred_param', 'x0')  # 'x0' or 'eps'
        assert self.pred_param in ('x0', 'eps'), \
            f"pred_param must be 'x0' or 'eps', got '{self.pred_param}'"
        self.model = GraphTransformer(
            n_layers=cfg.model.n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
            skip_x=skip_x
        )
        print(f"  [CRM] pred_param={self.pred_param}  skip_x={skip_x}")

        # ------------------------------------------------------------------
        # Noise schedule (shared cosine schedule for alpha_bar values)
        # ------------------------------------------------------------------
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            cfg.model.diffusion_noise_schedule,
            timesteps=cfg.model.diffusion_steps
        )

        # Book-keeping
        self.number_chain_steps = cfg.general.number_chain_steps
        self.current_epoch = 0
        self.best_val_nll = 1e8

    # ------------------------------------------------------------------#
    #             PRIOR CALIBRATION  (auto tau + mu_pi + sigma_pi)      #
    # ------------------------------------------------------------------#
    def calibrate_prior(self, rho_data: float, n_ref: int,
                        degree_sequences=None, num_samples: int = 500):
        """Full auto-calibration pipeline.

        If prior_family == 'lognormal' (default, new):
            Calibrates mu_pi and sigma_pi directly from training degree
            sequences and target density using Gauss-Hermite quadrature.
            No GGP sampling required.

        If prior_family == 'ggp' (legacy):
            Falls back to the old GGP pipeline (solve_tau + terminal MC).

        This must be called once before training or sampling.

        Args:
            rho_data:           Mean graph density of the training set.
            n_ref:              Mean node count of the training set (rounded int).
            degree_sequences:   List of numpy arrays, one per training graph,
                                each containing node degrees.  Required for
                                lognormal calibration.
            num_samples:        Terminal samples for GGP calibration (legacy only).
        """
        if not self.use_crm_prior:
            self.prior_mu = 0.0
            self.prior_sigma_scale = 1.0
            self.prior_tau = 1.0
            self._prior_calibrated = True
            return

        if self.prior_family == 'lognormal':
            # ----- LogNormal prior: direct calibration -----
            if degree_sequences is None or len(degree_sequences) == 0:
                raise ValueError(
                    "LogNormal prior calibration requires degree_sequences. "
                    "Pass them via model.calibrate_prior(..., degree_sequences=[...])."
                )
            mu, sigma_scale = calibrate_lognormal_prior(
                rho_data=rho_data,
                degree_sequences=degree_sequences,
                c=self.c,
                verbose=True,
            )
            self.prior_mu = mu
            self.prior_sigma_scale = sigma_scale
            self.prior_tau = None   # not applicable
            self._prior_calibrated = True

            # Sanity checks
            if sigma_scale < 0.05:
                print(f"  [WARN] sigma_pi={sigma_scale:.4f} < 0.05: prior may "
                      f"have collapsed.")
            if sigma_scale > 3.5:
                print(f"  [WARN] sigma_pi={sigma_scale:.4f} > 3.5: dynamic "
                      f"range of exp(u_t) will be extreme. Consider relaxing "
                      f"u_clamp (currently {self.u_clamp}).")

        else:
            # ----- Legacy GGP prior -----
            # Step 1: solve tau
            tau = solve_tau(
                rho_data=rho_data,
                n_ref=n_ref,
                sigma=self.prior_sigma,
                c=self.c,
                eps=self.prior_eps,
                alpha=self.prior_alpha,
                tolerance=0.02,
                max_tries=self.prior_max_tries,
                alpha_growth=self.prior_alpha_growth,
                verbose=True,
            )
            self.prior_tau = tau

            # Step 2: compute alpha_init
            print(f"\n  Estimating alpha_init for n_ref={n_ref} "
                  f"(tau={tau:.6f}, sigma={self.prior_sigma}, c={self.c}) ...")
            alpha_init = estimate_alpha_for_n(
                n=n_ref,
                sigma=self.prior_sigma,
                tau=tau,
                eps=self.prior_eps,
                c=self.c,
                n_probe=50,
                safety=1.5,
                verbose=True,
            )
            self.prior_alpha = alpha_init
            print(f"    prior_alpha = {alpha_init:.4f} "
                  f"(was {getattr(self, '_config_alpha', 1.0):.4f} from config)")

            # Step 3: calibrate mu_pi, sigma_pi from terminal samples
            print(f"\n  Calibrating mu_pi, sigma_pi with solved "
                  f"tau={tau:.6f}, c={self.c} ...")
            mu, sigma_scale = calibrate_prior_stats(
                n=n_ref, sigma=self.prior_sigma, tau=tau,
                alpha=self.prior_alpha, eps=self.prior_eps, c=self.c,
                max_tries=self.prior_max_tries,
                alpha_growth=self.prior_alpha_growth,
                num_samples=num_samples, verbose=True,
            )
            self.prior_mu = mu
            self.prior_sigma_scale = sigma_scale
            self._prior_calibrated = True

            # Sanity checks
            if tau < 0.01 or tau > 20.0:
                print(f"  [WARN] solved tau={tau:.6f} outside [0.01, 20.0]. "
                      f"sigma may need adjustment.")
            if sigma_scale < 0.3:
                print(f"  [WARN] sigma_pi={sigma_scale:.4f} < 0.3: prior may "
                      f"have collapsed.")
            if sigma_scale > 3.5:
                print(f"  [WARN] sigma_pi={sigma_scale:.4f} > 3.5: dynamic "
                      f"range of exp(u_t) will be extreme. Consider relaxing "
                      f"u_clamp (currently {self.u_clamp}).")
            if not (-10.0 <= mu <= 2.0):
                print(f"  [WARN] mu_pi={mu:.4f} outside [-10, 2].")

            print(f"\n  [AutoCalib] COMPLETE (GGP)")
            print(f"    rho_data   = {rho_data:.4f}")
            print(f"    n_ref      = {n_ref}")
            print(f"    c          = {self.c} (fixed)")
            print(f"    tau        = {tau:.6f} (solved)")
            print(f"    alpha_init = {self.prior_alpha:.4f} (auto)")
            print(f"    mu_pi      = {mu:.4f}")
            print(f"    sigma_pi   = {sigma_scale:.4f}")
            print(f"    u_T ~ N({mu:.4f}, {sigma_scale:.4f}^2)")
            print(f"    w_T median ~ exp(mu_pi) = {float(torch.tensor(mu).exp()):.6f}")
            print()

    def _ensure_calibrated(self, n: int = 10):
        """Assert that prior statistics have been calibrated."""
        if not self._prior_calibrated:
            raise RuntimeError(
                "CRM prior not calibrated. Call "
                "model.calibrate_prior(rho_data, n_ref) before "
                "training or sampling."
            )

    # ------------------------------------------------------------------#
    #                       PROPERTIES                                  #
    # ------------------------------------------------------------------#
    @property
    def device(self):
        return next(self.parameters()).device

    # ------------------------------------------------------------------#
    #                LATENT  <-->  WEIGHT  CONVERSIONS                   #
    # ------------------------------------------------------------------#
    def _u_to_w(self, u):
        """Convert log-space latents to positive sociability weights.
        Clamps u for numerical safety before exponentiation."""
        u_clamped = torch.clamp(u, -self.u_clamp, self.u_clamp)
        return torch.exp(u_clamped)

    def _edge_prob_from_w(self, w, node_mask):
        """Compute P(A_ij=1 | w) = 1 - exp(-c * w_i * w_j).

        Args:
            w: (bs, n) positive weights
            node_mask: (bs, n) bool

        Returns:
            prob1: (bs, n, n)  probability of edge=1
        """
        # outer product  (bs, n, 1) * (bs, 1, n)
        ww = w.unsqueeze(2) * w.unsqueeze(1)          # (bs, n, n)
        prob1 = 1.0 - torch.exp(-self.c * ww)
        # mask invalid node pairs
        pair_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)   # (bs, n, n)
        prob1 = prob1 * pair_mask.float()
        return prob1

    # ------------------------------------------------------------------#
    #              PER-EDGE TRANSITION MATRICES                          #
    # ------------------------------------------------------------------#
    def _build_Qt_bar(self, alpha_bar, w, node_mask):
        """Build cumulative per-edge transition matrix bar{Q}_t.

        bar{Q}_t^{ij} = alpha_bar * I + (1-alpha_bar) * 1 * p_ij^T

        where p_ij = [exp(-c w_i w_j),  1 - exp(-c w_i w_j)]

        Args:
            alpha_bar: (bs, 1) cumulative signal retention
            w: (bs, n) positive sociability weights
            node_mask: (bs, n)

        Returns:
            Qt_bar: (bs, n, n, 2, 2)  per-edge transition matrices
        """
        bs, n = w.shape
        prob1 = self._edge_prob_from_w(w, node_mask)   # (bs, n, n)
        prob0 = 1.0 - prob1                              # (bs, n, n)

        ab = alpha_bar.view(bs, 1, 1)  # (bs, 1, 1) for broadcasting

        # Q[..., from_state, to_state]
        # from=0  -> to=0: ab + (1-ab)*prob0,   to=1: (1-ab)*prob1
        # from=1  -> to=0: (1-ab)*prob0,         to=1: ab + (1-ab)*prob1
        q00 = ab + (1.0 - ab) * prob0
        q01 = (1.0 - ab) * prob1
        q10 = (1.0 - ab) * prob0
        q11 = ab + (1.0 - ab) * prob1

        # Stack into (bs, n, n, 2, 2)
        row0 = torch.stack([q00, q01], dim=-1)  # (bs,n,n,2)
        row1 = torch.stack([q10, q11], dim=-1)
        Qt_bar = torch.stack([row0, row1], dim=-2)  # (bs,n,n,2,2)
        return Qt_bar

    def _build_Qt(self, beta_t, w, node_mask):
        """Build single-step per-edge transition matrix Q_t.

        Q_t = (1-beta_t)*I + beta_t * 1 * p_ij^T

        Args:
            beta_t: (bs, 1)
            w: (bs, n)
            node_mask: (bs, n)

        Returns:
            Qt: (bs, n, n, 2, 2)
        """
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
        Qt = torch.stack([row0, row1], dim=-2)
        return Qt

    # ------------------------------------------------------------------#
    #                  FORWARD  DIFFUSION                                #
    # ------------------------------------------------------------------#
    def _derive_u0_from_data(self, A_onehot, node_mask):
        """Derive data-based log-sociabilities u_0 from the clean graph.

        Constructs mean-centered heterogeneity signals from the degree
        sequence and shifts them to the globally calibrated prior mean:

            tilde_u_i = log(d_i + 0.5) - mean_k[log(d_k + 0.5)]
            u_0_i     = mu_pi + tilde_u_i

        This ensures u_0 has the same mean as the prior (mu_pi) but
        carries per-node structural information through heterogeneity.
        The forward process interpolates from u_0 toward N(mu_pi, sigma_pi^2),
        giving a genuine SNR gradient (u_0 is NOT already equal to u_T's
        distribution because the per-node pattern tilde_u is data-specific).

        Args:
            A_onehot: (bs, n, n, 2) clean edge one-hot  (channel 1 = edge)
            node_mask: (bs, n) bool

        Returns:
            u_0: (bs, n) data-derived log-weights
        """
        mu_pi = self.prior_mu

        # Extract degrees from clean graph
        A_class = A_onehot.argmax(dim=-1).float()           # (bs, n, n)
        A_class = A_class * node_mask.unsqueeze(1).float()   # mask columns
        degrees = A_class.sum(dim=2) * node_mask.float()     # (bs, n)

        # Heterogeneity signal: tilde_u_i = log(d_i + delta) - mean(log(d_k + delta))
        delta = 0.05
        log_d = torch.log((degrees + delta).clamp(min=1e-8))                # (bs, n)
        # Mean only over valid nodes
        n_valid = node_mask.float().sum(dim=1, keepdim=True).clamp(min=1)   # (bs, 1)
        mean_log_d = (log_d * node_mask.float()).sum(dim=1, keepdim=True) / n_valid  # (bs, 1)
        tilde_u = (log_d - mean_log_d) * node_mask.float()                 # (bs, n)

        # u_0 = global anchor + per-node heterogeneity
        u_0 = (mu_pi + tilde_u) * node_mask.float()

        return u_0

    def apply_noise(self, A_onehot, y, node_mask):
        """Apply joint forward noise to latents and edges.

        Option B (shifted-scaled DDPM):
        1) Derive u_0 from data graph degrees (not random)
        2) Forward process interpolates toward N(mu_pi, sigma_pi^2):
           u_t = mu_pi + sqrt(alpha_bar_t) * (u_0 - mu_pi)
                       + sigma_pi * sqrt(1 - alpha_bar_t) * eps
        3) Compute w_t = exp(u_t)
        4) Build per-edge Qt_bar(w_t, alpha_bar_t)
        5) Noise A_0 -> A_t via edge-specific Qt_bar

        At t=0: u_0 (data-derived weights)
        At t=T: ~ N(mu_pi, sigma_pi^2) matching GGP prior

        Args:
            A_onehot: (bs, n, n, 2)  clean edge one-hot [no-edge, edge]
            y: (bs, dy)
            node_mask: (bs, n) bool

        Returns:
            noisy_data dict
        """
        bs, n = node_mask.shape
        lowest_t = 0 if self.training else 1

        # Ensure prior is calibrated
        self._ensure_calibrated(n=n)
        mu_pi = self.prior_mu
        sigma_pi = self.prior_sigma_scale

        # --------------- time sampling ---------------
        t_int = torch.randint(lowest_t, self.T + 1, (bs, 1), device=self.device).float()
        s_int = t_int - 1
        t_float = t_int / self.T
        s_float = s_int / self.T

        # --------------- schedule values ---------------
        beta_t = self.noise_schedule(t_normalized=t_float)           # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)

        # --------------- continuous forward: data u_0 -> noisy u_t -----
        # Derive u_0 from data graph degrees
        u_0 = self._derive_u0_from_data(A_onehot, node_mask)         # (bs, n)
        eps = torch.randn(bs, n, device=self.device)                 # eps ~ N(0,I)

        sqrt_ab = torch.sqrt(alpha_t_bar).view(bs, 1)               # (bs, 1)
        sqrt_1m_ab = torch.sqrt((1.0 - alpha_t_bar).clamp(min=1e-8)).view(bs, 1)

        # Shifted-scaled forward process:
        # u_t = mu_pi + sqrt(ab) * (u_0 - mu_pi) + sigma_pi * sqrt(1-ab) * eps
        u_t = mu_pi + sqrt_ab * (u_0 - mu_pi) + sigma_pi * sqrt_1m_ab * eps
        u_t = u_t * node_mask.float()

        w_t = self._u_to_w(u_t)                                     # (bs, n)
        w_0 = self._u_to_w(u_0)                                     # (bs, n)

        # --------------- discrete forward: noise edges via Qt_bar(w_0) ----
        # The stationary column of Qt_bar is p_ij = 1-exp(-c*w_i*w_j).
        # This must use the CLEAN w_0 (data scale) so that the forward marginal
        # at t=T matches the data-implied density (~rho_data), not the prior lognormal.
        # Using noisy w_t here biases the corruption toward prior scale (too dense).
        Qtb = self._build_Qt_bar(alpha_t_bar, w_0, node_mask)       # (bs,n,n,2,2)

        # A_onehot: (bs, n, n, 2) – multiply: prob(A_t | A_0) = A_0 @ Qtb
        # A_onehot[..., k] is 1 for class k. Matrix-vector: out_j = sum_k A_k * Qtb[k,j]
        # Using einsum: (bs,n,n,2) @ (bs,n,n,2,2) -> (bs,n,n,2)
        prob_At = torch.einsum('bijk,bijkl->bijl', A_onehot.float(), Qtb)  # (bs,n,n,2)

        # Clamp small negatives from numerical noise
        prob_At = torch.clamp(prob_At, min=1e-8)
        prob_At = prob_At / prob_At.sum(dim=-1, keepdim=True)

        # Sample A_t from the probabilities
        A_t = self._sample_edges(prob_At, node_mask)                 # (bs, n, n) long
        A_t_onehot = F.one_hot(A_t, num_classes=2).float()          # (bs, n, n, 2)

        # mask & symmetrise
        pair_mask = (node_mask.unsqueeze(2) * node_mask.unsqueeze(1)).unsqueeze(-1)
        A_t_onehot = A_t_onehot * pair_mask.float()

        noisy_data = {
            't_int': t_int,
            't': t_float,
            'beta_t': beta_t,
            'alpha_s_bar': alpha_s_bar,
            'alpha_t_bar': alpha_t_bar,
            # continuous
            'u_0': u_0,
            'u_t': u_t,
            'eps': eps,
            'w_t': w_t,
            # discrete
            'E_t': A_t_onehot,
            'A_0_onehot': A_onehot,
            # kept for interface compat
            'X_t': node_mask.unsqueeze(-1).float(),   # constant 1-feature for masked nodes
            'y_t': y,
            'node_mask': node_mask,
        }
        return noisy_data

    def _sample_edges(self, prob_E, node_mask):
        """Sample binary edges from probability tensor.

        Args:
            prob_E: (bs, n, n, 2)  probabilities [P(no-edge), P(edge)]
            node_mask: (bs, n)

        Returns:
            E: (bs, n, n) long, symmetric upper-triangular-sampled
        """
        bs, n, _, _ = prob_E.shape

        # Mask invalid entries
        inv_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
        diag = torch.eye(n, device=prob_E.device).bool().unsqueeze(0).expand(bs, -1, -1)
        prob_E[inv_mask] = torch.tensor([1.0, 0.0], device=prob_E.device)
        prob_E[diag] = torch.tensor([1.0, 0.0], device=prob_E.device)

        flat = prob_E.reshape(bs * n * n, 2)
        sampled = flat.multinomial(1).reshape(bs, n, n)

        # Symmetrise: keep upper triangle, mirror
        sampled = torch.triu(sampled, diagonal=1)
        sampled = sampled + sampled.transpose(1, 2)
        return sampled

    # ------------------------------------------------------------------#
    #              NETWORK  FORWARD  PASS                                #
    # ------------------------------------------------------------------#
    def forward(self, noisy_data, extra_data, node_mask):
        """Run denoiser network.

        Node features X = [u_t] + extra_features  (dim = 1 + extras)
        Edge features E = A_t one-hot              (dim = 2)

        Returns PlaceHolder(X=u0_pred, E=edge_logits, y=y_pred)
        Note: X0-parameterisation — network predicts clean u_0 directly, not eps.
        """
        # Build node input: u_t as primary feature
        u_t = noisy_data['u_t'].unsqueeze(-1)                       # (bs, n, 1)
        X = torch.cat([u_t, extra_data.X], dim=-1).float()          # (bs, n, 1+extra)
        E = torch.cat([noisy_data['E_t'], extra_data.E], dim=-1).float()
        y = torch.hstack([noisy_data['y_t'], extra_data.y]).float()

        out = self.model(X, E, y, node_mask)
        return out   # out.X = u0_pred (bs,n,1), out.E = edge logits (bs,n,n,2)

    def compute_extra_data(self, noisy_data):
        """Compute extra features (cycle counts, eigenvalues, etc.)."""
        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    # ------------------------------------------------------------------#
    #                    TRAINING  STEP                                   #
    # ------------------------------------------------------------------#
    def training_step(self, data, log_every_steps=50, step=0):
        """Single training step.

        Loss = L_A (edge cross-entropy) + lambda_w * L_w (latent noise MSE)
        """
        if data.edge_index.numel() == 0:
            print("Found a batch with no edges. Skipping.")
            return None

        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        E = dense_data.E   # (bs, n, n, 2) one-hot
        y = data.y if data.y is not None else torch.zeros(node_mask.size(0), 0, device=self.device)

        # Forward noise
        noisy_data = self.apply_noise(E, y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # --- Loss L_w: latent prediction MSE ---
        mask_float = node_mask.float()
        n_nodes = mask_float.sum(dim=1, keepdim=True).clamp(min=1)
        raw_pred = pred.X.squeeze(-1)                                 # (bs, n)
        if self.pred_param == 'eps':
            # eps-parameterisation: supervise against the true noise stored in noisy_data
            target_w = noisy_data['eps']                              # (bs, n)
        else:
            # x0-parameterisation: supervise against clean u_0
            target_w = noisy_data['u_0']                              # (bs, n)
        loss_w = ((raw_pred - target_w) ** 2 * mask_float).sum(dim=1, keepdim=True) / n_nodes
        loss_w = loss_w.mean()

        # --- Loss L_A: edge cross-entropy ---
        true_E = noisy_data['A_0_onehot']                            # (bs, n, n, 2)
        pred_E = pred.E                                              # (bs, n, n, 2) logits

        # Flatten and mask
        pair_mask = (node_mask.unsqueeze(1) * node_mask.unsqueeze(2))  # (bs, n, n)
        diag = torch.eye(node_mask.size(1), device=self.device).bool().unsqueeze(0)
        valid_mask = pair_mask & (~diag)                              # exclude diagonal

        flat_pred = pred_E[valid_mask]                               # (N_valid, 2)
        flat_true = true_E[valid_mask]                               # (N_valid, 2)

        if flat_pred.numel() > 0:
            target_classes = torch.argmax(flat_true, dim=-1)
            loss_A = F.cross_entropy(flat_pred, target_classes)
        else:
            loss_A = torch.tensor(0.0, device=self.device)

        loss = loss_A + self.lambda_w * loss_w

        if step % log_every_steps == 0:
            print(f"  [step {step}] L_A={loss_A.item():.4f}  "
                  f"L_w({self.pred_param})={loss_w.item():.4f}  "
                  f"total={loss.item():.4f}")

        return loss

    # ------------------------------------------------------------------#
    #                 REVERSE  PROCESS  (GENERATION)                     #
    # ------------------------------------------------------------------#
    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int,
                     number_chain_steps: int, save_final: int, num_nodes=None):
        """Sample a batch of graphs via the joint reverse process.

        If use_crm_prior=True (faithful Caron-Fox):
          1. Sample CRM atoms (w_T) from GGP Poisson process, condition to n nodes
          2. A_T from Caron-Fox link: Bern(1 - exp(-c w_i w_j))
          3. u_T = log(w_T)
        Else (legacy lognormal):
          1. u_T ~ N(0, I), w_T = exp(u_T)
          2. A_T ~ Bern(1 - exp(-c w_T,i w_T,j))

        Then for t = T..1:
          a) reverse continuous  -> u_{t-1}
          b) reverse discrete (tempered)  -> A_{t-1}
        """
        # --- number of nodes ---
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif isinstance(num_nodes, int):
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            n_nodes = num_nodes.to(self.device)

        n_max = torch.max(n_nodes).item()
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)    # (bs, n)

        # --- initialise terminal distribution ---
        if self.use_crm_prior:
            if self.prior_family == 'lognormal':
                # LogNormal sociability prior: direct sampling
                A_init, w_T, u_T = sample_lognormal_terminal_batch(
                    batch_size=batch_size,
                    n_nodes=n_nodes,
                    mu_pi=self.prior_mu,
                    sigma_pi=self.prior_sigma_scale,
                    c=self.c,
                    device=self.device,
                )
                alpha_finals = None
            else:
                # Legacy GGP: sample CRM atoms, graph, condition to n
                A_init, w_T, u_T, alpha_finals = sample_terminal_batch(
                    batch_size=batch_size,
                    n_nodes=n_nodes,
                    alpha=self.prior_alpha,
                    sigma=self.prior_sigma,
                    tau=self.prior_tau,
                    eps=self.prior_eps,
                    c=self.c,
                    seed=None,
                    max_tries=self.prior_max_tries,
                    alpha_growth=self.prior_alpha_growth,
                    device=self.device,
                    verbose=False,
                )
            # Clamp u_T for numerical stability
            u_T = torch.clamp(u_T, -self.u_clamp, self.u_clamp)
            u_T = u_T * node_mask.float()
            A_t_onehot = F.one_hot(A_init, num_classes=2).float()


        else:
            # Legacy: i.i.d. lognormal latents
            u_T = torch.randn(batch_size, n_max, device=self.device) * node_mask.float()
            w_T = self._u_to_w(u_T)
            prob1 = self._edge_prob_from_w(w_T, node_mask)    # (bs, n, n)
            A_init = torch.bernoulli(prob1).long()
            A_init = torch.triu(A_init, diagonal=1)
            A_init = A_init + A_init.transpose(1, 2)
            A_t_onehot = F.one_hot(A_init, num_classes=2).float()



        u_t = u_T

        # Dummy y
        y = torch.zeros(batch_size, 0, device=self.device)

        # --- chain storage ---
        chain_E_size = torch.Size((number_chain_steps, keep_chain, n_max, n_max))
        chain_E = torch.zeros(chain_E_size)

        # --- u_std arc tracking ---
        # Record u_std at ~21 evenly-spaced steps T..0 for post-run verdict.
        _arc_stride   = max(1, self.T // 20)
        _arc_record_t = set(range(0, self.T + 1, _arc_stride)) | {0, self.T}
        _arc_t   = []   # timestep index (t value, high→low)
        _arc_std = []   # u_std at that timestep (masked nodes only)
        # Record starting point t=T
        _u_std_T = float(u_T[node_mask].std().item())
        _arc_t.append(self.T)
        _arc_std.append(_u_std_T)

        # --- Bootstrapped A0 prediction for residual spectral features ---
        # Initialised to A_T (same graph the model sees at the first step).
        # After each forward pass pred_A0_soft is updated to softmax(pred.E),
        # so at step t the spectral features are computed from the model's
        # *previous-step* prediction of the clean graph rather than A_t.
        # This gives near-zero residuals at high t (pred ≈ prior ≈ A_t in
        # expectation) and accurate community signal at low t.
        pred_A0_soft = A_t_onehot.float().detach()   # (bs, n, n, 2)

        # --- reverse loop ---
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones(batch_size, 1, device=self.device)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            beta_t = self.noise_schedule(t_normalized=t_norm)
            alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_norm)
            alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_norm)

            # ---- A) reverse continuous step: u_t -> u_{t-1} ----
            noisy_data = {
                'u_t': u_t, 'E_t': A_t_onehot,
                'X_t': node_mask.unsqueeze(-1).float(),
                'y_t': y, 't': t_norm, 'node_mask': node_mask,
                'pred_A0': pred_A0_soft,   # bootstrapped prev-step prediction
            }
            extra_data = self.compute_extra_data(noisy_data)
            pred = self.forward(noisy_data, extra_data, node_mask)

            # Update bootstrapped prediction for next reverse step
            pred_A0_soft = F.softmax(pred.E, dim=-1).detach()   # (bs, n, n, 2)

            mu_pi = self.prior_mu
            sigma_pi = self.prior_sigma_scale

            alpha_t = 1.0 - beta_t.view(-1, 1)                       # (bs, 1)
            sqrt_alpha_t = torch.sqrt(alpha_t.clamp(min=1e-8))
            ab_t = alpha_t_bar.view(-1, 1)                            # (bs, 1)
            ab_s = alpha_s_bar.view(-1, 1)                            # (bs, 1)
            sqrt_ab_s = torch.sqrt(ab_s.clamp(min=0.0))

            # Derive u_0_hat from network output based on parameterisation
            raw_hat = pred.X.squeeze(-1)                              # (bs, n)
            if self.pred_param == 'eps':
                # eps-parameterisation: reconstruct u_0 from predicted noise
                # forward: u_t = mu_pi + sqrt(ab_t)*(u_0-mu_pi) + sigma_pi*sqrt(1-ab_t)*eps
                # => u_0  = mu_pi + (u_t - mu_pi - sigma_pi*sqrt(1-ab_t)*eps_hat) / sqrt(ab_t)
                sqrt_ab_t = torch.sqrt(ab_t.clamp(min=1e-8))
                sqrt_1m_ab_t = torch.sqrt((1.0 - ab_t).clamp(min=1e-8))
                eps_hat = torch.clamp(raw_hat, -5.0, 5.0) * node_mask.float()
                u_0_hat = mu_pi + (u_t - mu_pi - sigma_pi * sqrt_1m_ab_t * eps_hat) / sqrt_ab_t
            else:
                # x0-parameterisation: network directly predicts clean u_0
                u_0_hat = raw_hat
            u_0_hat = torch.clamp(u_0_hat, -self.u_clamp, self.u_clamp) * node_mask.float()

            # DDPM posterior mean (x0 form, valid for both parameterisations):
            # mu(u_t, u_0_hat) = mu_pi
            #   + sqrt(ab_s) * beta_t / (1 - ab_t) * (u_0_hat - mu_pi)
            #   + sqrt(alpha_t) * (1 - ab_s) / (1 - ab_t) * (u_t - mu_pi)
            denom = (1.0 - ab_t).clamp(min=1e-8)
            coef_x0 = sqrt_ab_s * beta_t.view(-1, 1) / denom
            coef_xt = sqrt_alpha_t * (1.0 - ab_s) / denom

            mu = mu_pi + coef_x0 * (u_0_hat - mu_pi) + coef_xt * (u_t - mu_pi)

            if s_int > 0:
                # posterior variance: sigma_pi^2 * beta_t * (1 - ab_s) / (1 - ab_t)
                sigma2 = beta_t.view(-1, 1) * (1.0 - ab_s) / denom
                sigma = sigma_pi * torch.sqrt(sigma2.clamp(min=1e-8, max=1.0))
                u_t_minus1 = mu + sigma * torch.randn_like(mu)
            else:
                u_t_minus1 = mu

            # Clamp latents to prevent NaN from exponential blowup
            u_t_minus1 = torch.clamp(u_t_minus1, -self.u_clamp, self.u_clamp)
            u_t_minus1 = u_t_minus1 * node_mask.float()

            # ---- B) reverse discrete step: A_t -> A_{t-1} (tempered) ----
            pred_E_prob = F.softmax(pred.E, dim=-1)   # (bs,n,n,2)  predicted p(A_0)

            # Build transition matrices anchored at predicted CLEAN weights w_0_hat.
            # Qt_bar encodes the stationary data distribution p_ij = 1-exp(-c*w_0_i*w_0_j);
            # using noisy w_t here inflates p_ij toward the prior (high-variance lognormal)
            # and systematically over-generates edges.  Use w_0_hat for Qtb and Qsb;
            # Qt (single-step) also uses w_0_hat for consistency.
            w_0_hat = self._u_to_w(u_0_hat)
            Qt  = self._build_Qt(beta_t, w_0_hat, node_mask)              # (bs,n,n,2,2)
            Qtb = self._build_Qt_bar(alpha_t_bar, w_0_hat, node_mask)     # (bs,n,n,2,2)
            Qsb = self._build_Qt_bar(alpha_s_bar, w_0_hat, node_mask)     # (bs,n,n,2,2)

            A_t_onehot = self._reverse_discrete_step(
                A_t_onehot, pred_E_prob, Qt, Qsb, Qtb, node_mask)

            u_t = u_t_minus1

            # --- arc tracking: record u_std at selected steps ---
            if s_int in _arc_record_t:
                _std_s = float(u_t[node_mask].std().item())
                _arc_t.append(s_int)
                _arc_std.append(_std_s)

            # store chain
            write_index = (s_int * number_chain_steps) // self.T
            if write_index < number_chain_steps and keep_chain > 0:
                chain_E[write_index] = A_t_onehot[:keep_chain, :, :, 1].cpu()

        # --- Final output ---
        # Collapse to class labels
        A_final = torch.argmax(A_t_onehot, dim=-1)  # (bs, n, n)

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i].item()
            atom_types = torch.ones(n, 1)           # constant node type
            edge_types = A_final[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        # Visualise if requested
        if self.visualization_tools is not None and keep_chain > 0:
            self._visualize_results(molecule_list, chain_E, batch_id,
                                    batch_size, keep_chain, save_final, n_max)

        return molecule_list

    def _reverse_discrete_step(self, A_t_oh, pred_p0, Qt, Qsb, Qtb, node_mask):
        """Tempered reverse discrete posterior in log-space.

        For each edge (i,j) and candidate state x in {0,1}:
          log p~(x) = gamma * [log Qt(A_t|x) + log sum_{x0} Qsb(x|x0)*p0(x0)
                                            - log sum_{x0} Qtb(A_t|x0)*p0(x0)]

        Args:
            A_t_oh:  (bs, n, n, 2)  current edge one-hot
            pred_p0: (bs, n, n, 2)  predicted clean-state probs
            Qt:      (bs, n, n, 2, 2) single-step transition
            Qsb:     (bs, n, n, 2, 2) cumulative transition to s
            Qtb:     (bs, n, n, 2, 2) cumulative transition to t
            node_mask: (bs, n)

        Returns:
            A_s_oh: (bs, n, n, 2) sampled edge one-hot for step s
        """
        bs, n = node_mask.shape

        # A_t class index  (bs, n, n)
        A_t_class = torch.argmax(A_t_oh, dim=-1)

        # --- Term 1: log Qt(A_t | x)  for x in {0,1} ---
        # Qt[..., x, A_t]  ->  gather along last dim
        A_t_idx = A_t_class.unsqueeze(-1).unsqueeze(-2).expand(-1, -1, -1, 2, -1)  # (bs,n,n,2,1)
        log_Qt_At_given_x = torch.log(
            torch.gather(Qt, dim=-1, index=A_t_idx).squeeze(-1) + 1e-30
        )   # (bs, n, n, 2)   [for x=0 and x=1]

        # --- Term 2: log sum_{x0} Qsb(x|x0) * p0(x0) ---
        # Qsb: (bs,n,n,2,2)  ->  Qsb[..., x0, x]
        # p0:  (bs,n,n,2)  ->  p0[..., x0]
        # matmul:  sum_{x0} p0(x0) * Qsb(x0, x)  =  p0 @ Qsb  -> (bs,n,n,2)
        pred_p0_expanded = pred_p0.unsqueeze(-2)   # (bs,n,n,1,2)
        Qsb_weighted = torch.matmul(pred_p0_expanded, Qsb).squeeze(-2)  # (bs,n,n,2)
        log_Qsb_term = torch.log(Qsb_weighted + 1e-30)

        # --- Term 3: log sum_{x0} Qtb(A_t|x0) * p0(x0) ---
        # Qtb[..., x0, A_t]  -> gather
        Qtb_At = torch.gather(Qtb, dim=-1, index=A_t_idx).squeeze(-1)  # (bs,n,n,2)
        # sum_{x0} p0(x0) * Qtb(x0, A_t)
        denom = (pred_p0 * Qtb_At).sum(dim=-1, keepdim=True)  # (bs,n,n,1)
        log_denom = torch.log(denom + 1e-30)

        # --- Combine with tempering ---
        log_unnorm = self.gamma * (log_Qt_At_given_x + log_Qsb_term - log_denom)

        # --- Normalise in log-space ---
        log_prob = log_unnorm - torch.logsumexp(log_unnorm, dim=-1, keepdim=True)
        prob = torch.exp(log_prob)

        # Clamp & renormalise for safety
        prob = torch.clamp(prob, min=1e-8)
        prob = prob / prob.sum(dim=-1, keepdim=True)

        # --- Sample ---
        A_s = self._sample_edges(prob, node_mask)
        A_s_oh = F.one_hot(A_s, num_classes=2).float()

        # Mask
        pair_mask = (node_mask.unsqueeze(2) * node_mask.unsqueeze(1)).unsqueeze(-1)
        A_s_oh = A_s_oh * pair_mask.float()

        return A_s_oh

    # ------------------------------------------------------------------#
    #                  VALIDATION / TEST                                  #
    # ------------------------------------------------------------------#
    def validation_step(self, data):
        """Compute validation loss (same as training loss, no grad)."""
        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        E = dense_data.E
        y = data.y if data.y is not None else torch.zeros(node_mask.size(0), 0, device=self.device)

        noisy_data = self.apply_noise(E, y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # L_w
        mask_float = node_mask.float()
        n_nodes = mask_float.sum(dim=1, keepdim=True).clamp(min=1)
        raw_pred = pred.X.squeeze(-1)
        target_w = noisy_data['eps'] if self.pred_param == 'eps' else noisy_data['u_0']
        loss_w = ((raw_pred - target_w) ** 2 * mask_float).sum(dim=1, keepdim=True) / n_nodes
        loss_w = loss_w.mean()

        # L_A
        true_E = noisy_data['A_0_onehot']
        pred_E = pred.E
        pair_mask = (node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
        diag = torch.eye(node_mask.size(1), device=self.device).bool().unsqueeze(0)
        valid_mask = pair_mask & (~diag)
        flat_pred = pred_E[valid_mask]
        flat_true = true_E[valid_mask]
        if flat_pred.numel() > 0:
            target_classes = torch.argmax(flat_true, dim=-1)
            loss_A = F.cross_entropy(flat_pred, target_classes)
        else:
            loss_A = torch.tensor(0.0, device=self.device)

        loss = loss_A + self.lambda_w * loss_w
        return loss.item()

    def test_step(self, data):
        """Same as validation_step."""
        return self.validation_step(data)

    def reset_metrics(self, test=False):
        """Reset sampling metrics."""
        if self.sampling_metrics is not None:
            self.sampling_metrics.reset()

    # ------------------------------------------------------------------#
    #                    VISUALISATION HELPERS                            #
    # ------------------------------------------------------------------#
    def _visualize_results(self, molecule_list, chain_E, batch_id,
                           batch_size, keep_chain, save_final, n_max):
        """Visualise generated graphs and chains."""
        if self.visualization_tools is None:
            return

        current_path = os.getcwd()
        if hasattr(self.cfg.general, 'experiment_dir'):
            result_path = os.path.join(
                current_path,
                f'{self.cfg.general.experiment_dir}/graphs/epoch{self.current_epoch}_b{batch_id}/')
        else:
            result_path = os.path.join(
                current_path,
                f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
