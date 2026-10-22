"""
Evaluation metrics and visualization for SparseDiffLSP.

MMD metrics: degree, spectral, clustering — with ratio r = MMD²(gen,test)/MMD²(train,test).
Visualization: GT graphs, generated graphs, and side-by-side comparisons.
"""

import os
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import concurrent.futures
from scipy.linalg import eigvalsh


# ======================================================================
#  MMD kernel & distance
# ======================================================================

def gaussian_tv(x, y, sigma=1.0):
    """Gaussian kernel with Total Variation distance."""
    support_size = max(len(x), len(y))
    x = x.astype(float)
    y = y.astype(float)
    if len(x) < support_size:
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    if len(y) < support_size:
        y = np.hstack((y, [0.0] * (support_size - len(y))))
    dist = np.abs(x - y).sum() / 2.0
    return np.exp(-dist * dist / (2 * sigma * sigma))


def _disc(samples1, samples2, kernel):
    """Pairwise kernel discrepancy (parallelised)."""
    def _worker(s1):
        return sum(kernel(s1, s2) for s2 in samples2)

    d = 0.0
    with concurrent.futures.ThreadPoolExecutor() as pool:
        for val in pool.map(_worker, samples1):
            d += val
    n = len(samples1) * len(samples2)
    return d / n if n > 0 else 1e6


def compute_mmd(samples1, samples2, kernel=gaussian_tv, is_hist=True):
    """Unbiased MMD² between two sets of histogram samples."""
    if is_hist:
        samples1 = [s / (np.sum(s) + 1e-6) for s in samples1]
        samples2 = [s / (np.sum(s) + 1e-6) for s in samples2]
    return (_disc(samples1, samples1, kernel)
            + _disc(samples2, samples2, kernel)
            - 2.0 * _disc(samples1, samples2, kernel))


# ======================================================================
#  Per-metric MMD functions
# ======================================================================

def _degree_hist(G):
    return np.array(nx.degree_histogram(G))


def degree_stats(graphs_ref, graphs_pred):
    """MMD between degree distributions."""
    ref = [_degree_hist(G) for G in graphs_ref if G.number_of_nodes() > 0]
    pred = [_degree_hist(G) for G in graphs_pred if G.number_of_nodes() > 0]
    if not ref or not pred:
        return 0.0
    return compute_mmd(ref, pred, kernel=gaussian_tv)


def _spectral_hist(G, n_bins=200):
    try:
        eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    except Exception:
        eigs = np.zeros(G.number_of_nodes())
    hist, _ = np.histogram(eigs, bins=n_bins, range=(-1e-5, 2), density=False)
    return hist.astype(float)


def spectral_stats(graphs_ref, graphs_pred):
    """MMD between spectral (eigenvalue) distributions."""
    ref = [_spectral_hist(G) for G in graphs_ref if G.number_of_nodes() > 0]
    pred = [_spectral_hist(G) for G in graphs_pred if G.number_of_nodes() > 0]
    if not ref or not pred:
        return 0.0
    return compute_mmd(ref, pred, kernel=gaussian_tv)


def _clustering_hist(G, bins=100):
    cc = list(nx.clustering(G).values())
    hist, _ = np.histogram(cc, bins=bins, range=(0.0, 1.0), density=False)
    return hist.astype(float)


def clustering_stats(graphs_ref, graphs_pred, bins=100):
    """MMD between clustering coefficient distributions."""
    ref = [_clustering_hist(G, bins) for G in graphs_ref if G.number_of_nodes() > 0]
    pred = [_clustering_hist(G, bins) for G in graphs_pred if G.number_of_nodes() > 0]
    if not ref or not pred:
        return 0.0
    return compute_mmd(ref, pred, kernel=lambda x, y: gaussian_tv(x, y, sigma=1.0/10))


# --- Assortativity ---

def _assortativity_scalar(G):
    """Degree assortativity as a 1-element array for MMD kernel."""
    if G.number_of_edges() == 0:
        return np.array([0.0])
    try:
        r = nx.degree_assortativity_coefficient(G)
        if np.isnan(r):
            r = 0.0
    except Exception:
        r = 0.0
    return np.array([r])


def assortativity_stats(graphs_ref, graphs_pred):
    ref = [_assortativity_scalar(G) for G in graphs_ref if G.number_of_nodes() > 0]
    pred = [_assortativity_scalar(G) for G in graphs_pred if G.number_of_nodes() > 0]
    if not ref or not pred:
        return 0.0
    return compute_mmd(ref, pred, kernel=gaussian_tv, is_hist=False)


# --- Characteristic Path Length (CPL) ---

def _cpl_scalar(G):
    """Average shortest-path length over largest connected component."""
    if G.number_of_nodes() < 2:
        return np.array([0.0])
    try:
        if nx.is_connected(G):
            cpl = nx.average_shortest_path_length(G)
        else:
            gcc = max(nx.connected_components(G), key=len)
            sg = G.subgraph(gcc).copy()
            cpl = nx.average_shortest_path_length(sg)
    except Exception:
        cpl = 0.0
    return np.array([cpl])


def cpl_stats(graphs_ref, graphs_pred):
    ref = [_cpl_scalar(G) for G in graphs_ref if G.number_of_nodes() > 0]
    pred = [_cpl_scalar(G) for G in graphs_pred if G.number_of_nodes() > 0]
    if not ref or not pred:
        return 0.0
    return compute_mmd(ref, pred, kernel=gaussian_tv, is_hist=False)


# --- Power-Law Exponent (PLE) ---

def _ple_scalar(G):
    """Estimate power-law exponent via MLE on degree sequence (Newman 2005)."""
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    degrees = degrees[degrees >= 1]  # filter zero-degree
    if len(degrees) < 2:
        return np.array([0.0])
    x_min = 1.0
    n = len(degrees)
    alpha = 1.0 + n / np.sum(np.log(degrees / x_min))
    return np.array([alpha])


def ple_stats(graphs_ref, graphs_pred):
    ref = [_ple_scalar(G) for G in graphs_ref if G.number_of_nodes() > 0]
    pred = [_ple_scalar(G) for G in graphs_pred if G.number_of_nodes() > 0]
    if not ref or not pred:
        return 0.0
    return compute_mmd(ref, pred, kernel=gaussian_tv, is_hist=False)


# --- Triangle count ---

def _triangle_hist(G, n_bins=50):
    """Per-node triangle counts as histogram."""
    tri = list(nx.triangles(G).values())
    if not tri or max(tri) == 0:
        return np.array([float(len(tri))] + [0.0] * (n_bins - 1))
    hist, _ = np.histogram(tri, bins=n_bins, range=(0, max(tri) + 1), density=False)
    return hist.astype(float)


def triangle_stats(graphs_ref, graphs_pred):
    ref = [_triangle_hist(G) for G in graphs_ref if G.number_of_nodes() > 0]
    pred = [_triangle_hist(G) for G in graphs_pred if G.number_of_nodes() > 0]
    if not ref or not pred:
        return 0.0
    return compute_mmd(ref, pred, kernel=gaussian_tv)


# ======================================================================
#  MMD ratio:  r = MMD²(gen, test) / MMD²(train, test)
# ======================================================================

def compute_all_mmd(graphs_a, graphs_b):
    """Compute all MMD² metrics between two graph sets.
    Returns dict {metric_name: mmd_value}.
    """
    return {
        "degree": degree_stats(graphs_a, graphs_b),
        "clustering": clustering_stats(graphs_a, graphs_b),
        "spectral": spectral_stats(graphs_a, graphs_b),
        "assortativity": assortativity_stats(graphs_a, graphs_b),
        "cpl": cpl_stats(graphs_a, graphs_b),
        "ple": ple_stats(graphs_a, graphs_b),
        "triangle": triangle_stats(graphs_a, graphs_b),
    }


def compute_mmd_ratios(gen_graphs, train_graphs, test_graphs):
    """Compute MMD ratios r = MMD²(gen, test) / MMD²(train, test).

    Args:
        gen_graphs: list of NetworkX graphs (generated)
        train_graphs: list of NetworkX graphs (training set)
        test_graphs: list of NetworkX graphs (test set)

    Returns:
        dict with keys 'mmd_gen_test', 'mmd_train_test', 'ratio' — each a dict
        of {degree, spectral, clustering}.
    """
    print("\n  Computing MMD²(gen, test)...")
    mmd_gen_test = compute_all_mmd(gen_graphs, test_graphs)

    print("  Computing MMD²(train, test)...")
    mmd_train_test = compute_all_mmd(train_graphs, test_graphs)

    ratios = {}
    for k in mmd_gen_test:
        denom = mmd_train_test[k]
        if denom > 1e-12:
            ratios[k] = mmd_gen_test[k] / denom
        else:
            ratios[k] = float("inf") if mmd_gen_test[k] > 1e-12 else 1.0

    return {
        "mmd_gen_test": mmd_gen_test,
        "mmd_train_test": mmd_train_test,
        "ratio": ratios,
    }


def print_mmd_report(results):
    """Print a formatted MMD ratio report."""
    W = 76
    print(f"\n{'=' * W}")
    print(f"  MMD² Evaluation Report")
    print(f"{'=' * W}")
    print(f"  {'Metric':<15}  {'MMD²(gen,test)':<18}  {'MMD²(train,test)':<18}  {'Ratio r':<10}")
    print(f"  {'-' * 15}  {'-' * 18}  {'-' * 18}  {'-' * 10}")

    metric_order = ["degree", "clustering", "spectral", "assortativity",
                     "cpl", "ple", "triangle"]
    for k in metric_order:
        if k not in results["mmd_gen_test"]:
            continue
        gt = results["mmd_gen_test"][k]
        tt = results["mmd_train_test"][k]
        r = results["ratio"][k]
        r_str = f"{r:.4f}" if r < 1e6 else "inf"
        print(f"  {k:<15}  {gt:<18.6f}  {tt:<18.6f}  {r_str:<10}")

    # Named ratios summary
    print(f"\n  MMD² ratios (r = MMD²(gen,test) / MMD²(train,test)):")
    for k in metric_order:
        if k not in results["ratio"]:
            continue
        r = results["ratio"][k]
        r_str = f"{r:.4f}" if r < 1e6 else "inf"
        print(f"    r_{k}: {r_str}")

    # Average ratio
    finite_ratios = [v for v in results["ratio"].values() if v < 1e6]
    if finite_ratios:
        avg_r = np.mean(finite_ratios)
        print(f"\n  Average ratio r = {avg_r:.4f}  (ideal: ~1.0)")
    print(f"{'=' * W}\n")


# ======================================================================
#  Graph visualization
# ======================================================================

def _community_layout(G, seed=42):
    """Community-aware spring layout. Returns (pos, node_colors)."""
    palette = list(plt.cm.Set1.colors) + list(plt.cm.Set2.colors)

    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(G))
        node_to_comm = {}
        for cid, comm in enumerate(comms):
            for v in comm:
                node_to_comm[v] = cid
        community_ids = [node_to_comm.get(v, 0) for v in G.nodes()]
    except Exception:
        community_ids = None

    if community_ids is not None and len(set(community_ids)) > 1:
        blocks = sorted(set(community_ids))
        n_blocks = len(blocks)
        block_index = {b: i for i, b in enumerate(blocks)}
        rng = np.random.default_rng(seed)
        init_pos = {}
        for v, b in zip(G.nodes(), community_ids):
            angle = 2 * np.pi * block_index[b] / n_blocks
            cx, cy = 2.0 * np.cos(angle), 2.0 * np.sin(angle)
            init_pos[v] = np.array([cx + rng.normal(0, 0.3),
                                    cy + rng.normal(0, 0.3)])
        pos = nx.spring_layout(G, pos=init_pos, seed=seed, k=0.5)
        node_colors = [palette[block_index[b] % len(palette)]
                       for b in community_ids]
    else:
        pos = nx.spring_layout(G, seed=seed)
        node_colors = "lightblue"

    return pos, node_colors


def visualize_graphs(save_dir, graphs, prefix="graph", num_graphs=8,
                     node_size=50):
    """Save individual graph PNGs. No titles per user preference."""
    os.makedirs(save_dir, exist_ok=True)

    for i in range(min(num_graphs, len(graphs))):
        G = graphs[i]
        if G.number_of_nodes() == 0:
            continue

        pos, colors = _community_layout(G, seed=42)
        fig, ax = plt.subplots(figsize=(6, 6))
        nx.draw(G, pos, ax=ax, node_size=node_size, node_color=colors,
                edge_color="gray", with_labels=False)
        ax.set_axis_off()
        plt.savefig(os.path.join(save_dir, f"{prefix}_{i}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved {min(num_graphs, len(graphs))} {prefix} plots to {save_dir}")


def visualize_comparison(save_dir, gt_graphs, gen_graphs, num_pairs=8,
                         node_size=50):
    """Side-by-side GT vs Generated comparison. No titles per user preference."""
    os.makedirs(save_dir, exist_ok=True)

    n = min(num_pairs, len(gt_graphs), len(gen_graphs))
    for i in range(n):
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # GT
        G_gt = gt_graphs[i]
        if G_gt.number_of_nodes() > 0:
            pos_gt, col_gt = _community_layout(G_gt, seed=42)
            nx.draw(G_gt, pos_gt, ax=axes[0], node_size=node_size,
                    node_color=col_gt, edge_color="gray", with_labels=False)
        axes[0].set_xlabel("Ground Truth", fontsize=25)
        axes[0].set_axis_off()

        # Generated
        G_gen = gen_graphs[i]
        if G_gen.number_of_nodes() > 0:
            pos_gen, col_gen = _community_layout(G_gen, seed=42)
            nx.draw(G_gen, pos_gen, ax=axes[1], node_size=node_size,
                    node_color=col_gen, edge_color="gray", with_labels=False)
        axes[1].set_xlabel("Generated", fontsize=25)
        axes[1].set_axis_off()

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"comparison_{i}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved {n} comparison plots to {save_dir}")
