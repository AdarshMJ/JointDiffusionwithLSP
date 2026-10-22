"""
Empirical LogNormal Sociability Prior.
LogNormal calibration.  The model is:

    u_i ~ N(mu_pi, sigma_pi^2)
    w_i = exp(u_i)
    A_ij ~ Bernoulli(1 - exp(-c * w_i * w_j))

Calibration is from training data (degree sequences + density).

    sigma_pi = std of empirical log-degree heterogeneity signals
    mu_pi    = solved via 1D root-finding so that
               E[1 - exp(-c * w_i * w_j)] = rho_data

The expectation is evaluated by Gauss-Hermite quadrature (exact for
the LogNormal model, no sparse-regime approximation).

Usage:
    from crm_prior.lognormal_prior import calibrate_lognormal_prior
    from crm_prior.lognormal_prior import sample_lognormal_terminal_batch

    mu_pi, sigma_pi = calibrate_lognormal_prior(
        rho_data=0.134, degree_sequences=[...], c=4.0)

    A_batch, w_batch, u_batch = sample_lognormal_terminal_batch(
        batch_size=8, n_nodes=50, mu_pi=mu_pi, sigma_pi=sigma_pi, c=4.0)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, Union, List, Optional


# ------------------------------------------------------------------
#  Gauss-Hermite density evaluation
# ------------------------------------------------------------------
def _density_lognormal_gh(mu: float, sigma: float, c: float,
                          n_quad: int = 64) -> float:
    """Compute expected edge density under the LogNormal sociability model.

    rho(mu, sigma) = E[1 - exp(-c * exp(u_i) * exp(u_j))]

    where u_i, u_j ~ iid N(mu, sigma^2).  Then u_i + u_j ~ N(2*mu, 2*sigma^2),
    so the expectation reduces to a 1D integral:

        rho = 1 - E_{Z ~ N(0,1)}[ exp(-c * exp(2*mu + sqrt(2)*sigma*Z)) ]

    Evaluated via Gauss-Hermite quadrature (the physicist convention
    uses weight function exp(-x^2), matching the substitution Z = x*sqrt(2)).

    Args:
        mu:      Mean of log-sociability.
        sigma:   Std of log-sociability.
        c:       Edge rate constant (typically 4.0).
        n_quad:  Number of Gauss-Hermite quadrature nodes.

    Returns:
        rho:  Expected edge density in [0, 1].
    """
    # Gauss-Hermite nodes and weights (physicist convention: w(x) = exp(-x^2))
    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)

    # Substitution: Z = x * sqrt(2) maps N(0,1) density to Hermite weight
    # E_Z[f(Z)] = (1/sqrt(pi)) * sum_i w_i * f(x_i * sqrt(2))
    z_vals = nodes * np.sqrt(2.0)                   # standard normal points
    exponent = 2.0 * mu + np.sqrt(2.0) * sigma * z_vals  # = 2*mu + sqrt(2)*sigma*Z
    # exp(-c * exp(exponent))  — clamp exponent to avoid overflow
    exponent = np.clip(exponent, -50.0, 50.0)
    inner = np.exp(-c * np.exp(exponent))

    # Weighted sum with normalization 1/sqrt(pi) for Hermite convention
    rho = 1.0 - np.sum(weights * inner) / np.sqrt(np.pi)
    return float(np.clip(rho, 0.0, 1.0))


# ------------------------------------------------------------------
#  Calibration
# ------------------------------------------------------------------
def calibrate_lognormal_prior(
    rho_data: float,
    degree_sequences: List[np.ndarray],
    c: float = 2.0,
    delta: float = 0.5,
    n_quad: int = 64,
    tol: float = 1e-6,
    max_iter: int = 60,
    verbose: bool = True
) -> Tuple[float, float]:
    """Calibrate (mu_pi, sigma_pi) from training data.

    Step 1: Compute heterogeneity signals tilde_u_i for each graph, pool,
            and set sigma_pi = std(tilde_u).
    Step 2: Solve for mu_pi via bisection so that the Gauss-Hermite density
            matches rho_data.

    Args:
        rho_data:          Mean edge density of the training set.
        degree_sequences:  List of 1D arrays, one per training graph, each
                           containing node degrees.
        c:                 Edge rate constant (fixed, typically 4.0).
        delta:             Laplace smoothing for log-degree (default 0.5).
        n_quad:            Gauss-Hermite quadrature nodes (default 64).
        tol:               Root-finding tolerance on density (default 1e-6).
        max_iter:          Maximum bisection iterations (default 60).
        verbose:           Print progress.

    Returns:
        mu_pi:      Calibrated mean of log-sociability.
        sigma_pi:   Calibrated std of log-sociability (>= 0.05 floor).
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"  LOGNORMAL PRIOR CALIBRATION")
        print(f"  rho_data={rho_data:.6f}, c={c}, n_graphs={len(degree_sequences)}")
        print(f"{'='*70}")

    # ---- Step 1: sigma_pi from empirical heterogeneity ----
    all_tilde_u = []
    for deg in degree_sequences:
        deg = np.asarray(deg, dtype=np.float64)
        if len(deg) < 2:
            continue
        log_d = np.log(deg + delta)
        mean_log_d = log_d.mean()
        tilde_u = log_d - mean_log_d       # mean-centered heterogeneity
        all_tilde_u.extend(tilde_u.tolist())

    all_tilde_u = np.array(all_tilde_u)
    sigma_pi = float(all_tilde_u.std())
    sigma_pi = max(sigma_pi, 0.05)         # floor to prevent collapse

    if verbose:
        print(f"  Step 1: sigma_pi = {sigma_pi:.6f}  "
              f"(from {len(all_tilde_u)} node signals across "
              f"{len(degree_sequences)} graphs)")

    # ---- Step 2: mu_pi via bisection ----
    # Sparse-regime initial guess (only used as a bracket hint)
    mu_init = 0.5 * (np.log(max(rho_data, 1e-10) / c) - sigma_pi ** 2)

    # Bracket: find mu_lo where rho(mu_lo) < rho_data
    #          and   mu_hi where rho(mu_hi) > rho_data
    # rho is monotone increasing in mu (larger mu → larger weights → more edges)
    mu_lo = mu_init - 10.0
    mu_hi = mu_init + 10.0

    rho_lo = _density_lognormal_gh(mu_lo, sigma_pi, c, n_quad)
    rho_hi = _density_lognormal_gh(mu_hi, sigma_pi, c, n_quad)

    if verbose:
        print(f"  Step 2: solving mu_pi")
        print(f"    Initial guess (sparse approx): mu_init = {mu_init:.6f}")
        print(f"    Bracket: mu_lo={mu_lo:.4f} -> rho={rho_lo:.6f}  |  "
              f"mu_hi={mu_hi:.4f} -> rho={rho_hi:.6f}")

    if rho_lo > rho_data:
        # Even very negative mu gives too-high density — shouldn't happen
        # but handle gracefully
        print(f"  [WARNING] rho at mu_lo={mu_lo:.4f} is {rho_lo:.6f} > "
              f"rho_data={rho_data:.6f}. Extending bracket.")
        mu_lo -= 20.0
        rho_lo = _density_lognormal_gh(mu_lo, sigma_pi, c, n_quad)

    if rho_hi < rho_data:
        # Very positive mu still too sparse — extend bracket
        print(f"  [WARNING] rho at mu_hi={mu_hi:.4f} is {rho_hi:.6f} < "
              f"rho_data={rho_data:.6f}. Extending bracket.")
        mu_hi += 20.0
        rho_hi = _density_lognormal_gh(mu_hi, sigma_pi, c, n_quad)

    # Bisection
    for iteration in range(max_iter):
        mu_mid = (mu_lo + mu_hi) / 2.0
        rho_mid = _density_lognormal_gh(mu_mid, sigma_pi, c, n_quad)
        err = abs(rho_mid - rho_data)

        if err < tol:
            if verbose:
                print(f"    Converged at iteration {iteration}: "
                      f"mu_pi={mu_mid:.6f}, rho={rho_mid:.6f}, err={err:.2e}")
            break

        if rho_mid < rho_data:
            mu_lo = mu_mid
        else:
            mu_hi = mu_mid

    mu_pi = (mu_lo + mu_hi) / 2.0
    rho_final = _density_lognormal_gh(mu_pi, sigma_pi, c, n_quad)

    if verbose:
        print(f"\n  CALIBRATION RESULT:")
        print(f"    mu_pi      = {mu_pi:.6f}")
        print(f"    sigma_pi   = {sigma_pi:.6f}")
        print(f"    rho_check  = {rho_final:.6f}  (target: {rho_data:.6f})")
        print(f"    w_median   = exp(mu_pi) = {np.exp(mu_pi):.6f}")
        print(f"    E[w]       = exp(mu + sigma^2/2) = "
              f"{np.exp(mu_pi + sigma_pi**2 / 2):.6f}")
        print(f"    u_T ~ N({mu_pi:.4f}, {sigma_pi:.4f}^2)")
        print(f"{'='*70}\n")

    return mu_pi, sigma_pi


# ------------------------------------------------------------------
#  Terminal Sampling
# ------------------------------------------------------------------
def sample_lognormal_terminal_batch(
    batch_size: int,
    n_nodes: Union[int, torch.Tensor, np.ndarray],
    mu_pi: float,
    sigma_pi: float,
    c: float = 4.0,
    device: torch.device = torch.device('cpu'),
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample a batch of terminal graphs from the LogNormal sociability prior.

    For each graph:
        u_i ~ N(mu_pi, sigma_pi^2)
        w_i = exp(u_i)
        A_ij ~ Bernoulli(1 - exp(-c * w_i * w_j))

    Args:
        batch_size: Number of graphs.
        n_nodes:    Int or (batch_size,) tensor of per-graph node counts.
        mu_pi:      Prior mean of log-sociability.
        sigma_pi:   Prior std of log-sociability.
        c:          Edge rate constant.
        device:     Torch device.
        seed:       Optional random seed.

    Returns:
        A_batch:  (batch_size, n_max, n_max) long — binary adjacency matrices.
        w_batch:  (batch_size, n_max)         float — sociability weights.
        u_batch:  (batch_size, n_max)         float — log-sociabilities.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Parse n_nodes
    if isinstance(n_nodes, int):
        node_counts = [n_nodes] * batch_size
    elif isinstance(n_nodes, torch.Tensor):
        node_counts = n_nodes.cpu().tolist()
    else:
        node_counts = list(n_nodes)

    n_max = max(node_counts)

    A_batch = torch.zeros(batch_size, n_max, n_max, dtype=torch.long, device=device)
    w_batch = torch.zeros(batch_size, n_max, dtype=torch.float32, device=device)
    u_batch = torch.zeros(batch_size, n_max, dtype=torch.float32, device=device)

    for i in range(batch_size):
        ni = int(node_counts[i])
        if ni == 0:
            continue

        # Sample latents
        u_i = torch.randn(ni, device=device) * sigma_pi + mu_pi
        w_i = torch.exp(u_i)

        # Pairwise edge probabilities
        ww = w_i.unsqueeze(1) * w_i.unsqueeze(0)     # (ni, ni)
        prob = 1.0 - torch.exp(-c * ww)               # (ni, ni)
        prob.fill_diagonal_(0.0)

        # Sample edges (upper triangle, then symmetrise)
        rand = torch.rand(ni, ni, device=device)
        A_upper = (rand < prob).long()
        A_upper = torch.triu(A_upper, diagonal=1)
        A_i = A_upper + A_upper.T

        # Store
        A_batch[i, :ni, :ni] = A_i
        w_batch[i, :ni] = w_i
        u_batch[i, :ni] = u_i

    return A_batch, w_batch, u_batch
