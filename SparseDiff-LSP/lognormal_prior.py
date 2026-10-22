"""
LogNormal Sociability Prior: calibration and terminal sampling.

Model:
    u_i ~ N(mu_pi, sigma_pi^2)
    w_i = exp(u_i)
    A_ij ~ Bernoulli(1 - exp(-c * w_i * w_j))

Calibration:
    sigma_pi = std of empirical log-degree heterogeneity signals
    mu_pi    = bisection so E[1-exp(-c*w_i*w_j)] = rho_data
"""

import numpy as np
import torch
from typing import Tuple, List, Optional, Union


def _density_lognormal_gh(mu: float, sigma: float, c: float,
                          n_quad: int = 64) -> float:
    """Expected edge density via Gauss-Hermite quadrature.

    rho = 1 - (1/sqrt(pi)) sum_k w_k exp(-c exp(2mu + sqrt(2)*sigma*x_k))
    """
    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    z_vals = nodes * np.sqrt(2.0)
    exponent = 2.0 * mu + np.sqrt(2.0) * sigma * z_vals
    exponent = np.clip(exponent, -50.0, 50.0)
    inner = np.exp(-c * np.exp(exponent))
    rho = 1.0 - np.sum(weights * inner) / np.sqrt(np.pi)
    return float(np.clip(rho, 0.0, 1.0))


def calibrate_lognormal_prior(
    rho_data: float,
    degree_sequences: List[np.ndarray],
    c: float = 4.0,
    delta: float = 0.5,
    n_quad: int = 64,
    tol: float = 1e-6,
    max_iter: int = 60,
    verbose: bool = True,
) -> Tuple[float, float]:
    """Calibrate (mu_pi, sigma_pi) from training data.

    Step 1: sigma_pi from pooled log-degree heterogeneity.
    Step 2: mu_pi via bisection matching rho_data.
    """
    if verbose:
        print(f"\n{'=' * 70}")
        print(f"  LOGNORMAL PRIOR CALIBRATION")
        print(f"  rho_data={rho_data:.6f}, c={c}, n_graphs={len(degree_sequences)}")
        print(f"{'=' * 70}")

    # Step 1: sigma_pi
    all_tilde_u = []
    for deg in degree_sequences:
        deg = np.asarray(deg, dtype=np.float64)
        if len(deg) < 2:
            continue
        log_d = np.log(deg + delta)
        mean_log_d = log_d.mean()
        tilde_u = log_d - mean_log_d
        all_tilde_u.extend(tilde_u.tolist())

    all_tilde_u = np.array(all_tilde_u)
    sigma_pi = float(all_tilde_u.std()) if len(all_tilde_u) > 0 else 0.5
    sigma_pi = max(sigma_pi, 0.05)

    if verbose:
        print(f"  Step 1: sigma_pi = {sigma_pi:.6f}  "
              f"(from {len(all_tilde_u)} node signals)")

    # Step 2: mu_pi via bisection
    mu_init = 0.5 * (np.log(max(rho_data, 1e-10) / c) - sigma_pi ** 2)
    mu_lo = mu_init - 10.0
    mu_hi = mu_init + 10.0

    rho_lo = _density_lognormal_gh(mu_lo, sigma_pi, c, n_quad)
    rho_hi = _density_lognormal_gh(mu_hi, sigma_pi, c, n_quad)

    if rho_lo > rho_data:
        mu_lo -= 20.0
    if rho_hi < rho_data:
        mu_hi += 20.0

    for iteration in range(max_iter):
        mu_mid = (mu_lo + mu_hi) / 2.0
        rho_mid = _density_lognormal_gh(mu_mid, sigma_pi, c, n_quad)
        if abs(rho_mid - rho_data) < tol:
            if verbose:
                print(f"    Converged at iter {iteration}: mu_pi={mu_mid:.6f}")
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
        print(f"    u_T ~ N({mu_pi:.4f}, {sigma_pi:.4f}^2)")
        print(f"{'=' * 70}\n")

    return mu_pi, sigma_pi


def sample_lognormal_terminal_batch(
    batch_size: int,
    n_nodes: Union[int, torch.Tensor],
    mu_pi: float,
    sigma_pi: float,
    c: float = 4.0,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample terminal graphs from LogNormal prior.

    Returns:
        A_batch: (batch_size, n_max, n_max) long
        w_batch: (batch_size, n_max) float
        u_batch: (batch_size, n_max) float
    """
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

        u_i = torch.randn(ni, device=device) * sigma_pi + mu_pi
        w_i = torch.exp(u_i)

        ww = w_i.unsqueeze(1) * w_i.unsqueeze(0)
        prob = 1.0 - torch.exp(-c * ww)
        prob.fill_diagonal_(0.0)

        rand = torch.rand(ni, ni, device=device)
        A_upper = (rand < prob).long()
        A_upper = torch.triu(A_upper, diagonal=1)
        A_i = A_upper + A_upper.T

        A_batch[i, :ni, :ni] = A_i
        w_batch[i, :ni] = w_i
        u_batch[i, :ni] = u_i

    return A_batch, w_batch, u_batch


def sample_lognormal_terminal_sparse(
    n_nodes: torch.Tensor,
    mu_pi: float,
    sigma_pi: float,
    c: float = 4.0,
    device: torch.device = torch.device("cpu"),
):
    """Sample terminal graphs in sparse (PyG) format.

    Args:
        n_nodes: (bs,) node counts per graph

    Returns:
        u_all:      (N_total,) log-sociabilities
        edge_index: (2, E_total) undirected edges
        edge_attr:  (E_total, 2) one-hot [no-edge, edge] — all are edge-class 1
        batch:      (N_total,)
        ptr:        (bs+1,)
    """
    bs = n_nodes.shape[0]
    node_counts = n_nodes.cpu().tolist()
    N_total = sum(node_counts)

    u_all = torch.zeros(N_total, device=device)
    all_edges_src = []
    all_edges_dst = []
    batch = torch.zeros(N_total, dtype=torch.long, device=device)
    ptr = [0]
    offset = 0

    for i in range(bs):
        ni = int(node_counts[i])
        if ni == 0:
            ptr.append(offset)
            continue

        u_i = torch.randn(ni, device=device) * sigma_pi + mu_pi
        u_all[offset: offset + ni] = u_i
        batch[offset: offset + ni] = i

        w_i = torch.exp(torch.clamp(u_i, -15.0, 15.0))
        ww = w_i.unsqueeze(1) * w_i.unsqueeze(0)
        prob = 1.0 - torch.exp(-c * ww)
        prob.fill_diagonal_(0.0)

        # Sample upper triangle
        rand = torch.rand(ni, ni, device=device)
        A_upper = torch.triu((rand < prob), diagonal=1)
        src, dst = A_upper.nonzero(as_tuple=True)

        # Make undirected
        src_all = torch.cat([src, dst]) + offset
        dst_all = torch.cat([dst, src]) + offset
        all_edges_src.append(src_all)
        all_edges_dst.append(dst_all)

        offset += ni
        ptr.append(offset)

    if len(all_edges_src) > 0:
        edge_index = torch.stack([torch.cat(all_edges_src), torch.cat(all_edges_dst)])
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)

    n_edges = edge_index.shape[1]
    edge_attr = torch.zeros(n_edges, 2, device=device)
    edge_attr[:, 1] = 1.0  # all edges are class 1

    ptr = torch.tensor(ptr, dtype=torch.long, device=device)

    return u_all, edge_index, edge_attr, batch, ptr
