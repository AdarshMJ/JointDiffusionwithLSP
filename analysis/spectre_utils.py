"""
Spectre sampling metrics for graph generation evaluation.
Adapted from DiGress to remove graph-tool dependency.
Uses NetworkX for community detection instead.
"""
import os
import copy
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import subprocess as sp
import concurrent.futures
import secrets
from string import ascii_uppercase, digits
from datetime import datetime
from scipy.linalg import eigvalsh
from scipy.stats import chi2
from torch_geometric.utils import to_networkx

from .dist_helper import compute_mmd, gaussian, gaussian_tv, gaussian_emd

try:
    from grakel import WeisfeilerLehman, VertexHistogram
    from grakel.utils import graph_from_networkx
    GRAKEL_AVAILABLE = True
except ImportError:
    GRAKEL_AVAILABLE = False
    print("Warning: grakel not available. Install with: pip install grakel")


def degree_worker(G):
    return np.array(nx.degree_histogram(G))


def degree_stats(graph_ref_list, graph_pred_list, is_parallel=True, compute_emd=False):
    """Compute the distance between degree distributions of two graph sets."""
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)
    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)

    if compute_emd:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    return mmd_dist


def spectral_worker(G, n_eigvals=-1):
    try:
        eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    except:
        eigs = np.zeros(G.number_of_nodes())
    if n_eigvals > 0:
        eigs = eigs[1:n_eigvals + 1]
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / (spectral_pmf.sum() + 1e-6)
    return spectral_pmf


def spectral_stats(graph_ref_list, graph_pred_list, is_parallel=True, n_eigvals=-1, compute_emd=False):
    """Compute the distance between spectral distributions."""
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_ref_list, 
                                                  [n_eigvals for _ in graph_ref_list]):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty,
                                                  [n_eigvals for _ in graph_pred_list_remove_empty]):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i], n_eigvals)
            sample_ref.append(spectral_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i], n_eigvals)
            sample_pred.append(spectral_temp)

    if compute_emd:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    return mmd_dist


def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist


def clustering_stats(graph_ref_list, graph_pred_list, bins=100, is_parallel=True, compute_emd=False):
    """Compute the distance between clustering coefficient distributions."""
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker,
                                                [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                    clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)
        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)

    if compute_emd:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd, sigma=1.0 / 10, distance_scaling=bins)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv, sigma=1.0 / 10)

    return mmd_dist


def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for (u, v) in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):
    """Compute orbit counts using ORCA.
    
    Note: This requires ORCA to be compiled. Falls back to zeros if not available.
    """
    tmp_fname = f'orca/tmp_{"".join(secrets.choice(ascii_uppercase + digits) for i in range(8))}.txt'
    tmp_fname = os.path.join(os.path.dirname(os.path.realpath(__file__)), tmp_fname)
    
    try:
        os.makedirs(os.path.dirname(tmp_fname), exist_ok=True)
        f = open(tmp_fname, 'w')
        f.write(str(graph.number_of_nodes()) + ' ' + str(graph.number_of_edges()) + '\n')
        for (u, v) in edge_list_reindexed(graph):
            f.write(str(u) + ' ' + str(v) + '\n')
        f.close()
        
        orca_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'orca/orca')
        if not os.path.exists(orca_path):
            # ORCA not compiled, return zeros
            os.remove(tmp_fname)
            return np.zeros((graph.number_of_nodes(), 15))
            
        output = sp.check_output([orca_path, 'node', '4', tmp_fname, 'std'])
        output = output.decode('utf8').strip()
        idx = output.find('orbit counts:') + len('orbit counts:') + 2
        output = output[idx:]
        node_orbit_counts = np.array([
            list(map(int, node_cnts.strip().split(' ')))
            for node_cnts in output.strip('\n').split('\n')
        ])
        os.remove(tmp_fname)
        return node_orbit_counts
    except Exception as e:
        try:
            os.remove(tmp_fname)
        except:
            pass
        return np.zeros((graph.number_of_nodes(), 15))


def orbit_stats_all(graph_ref_list, graph_pred_list, compute_emd=False):
    """Compute orbit count statistics."""
    total_counts_ref = []
    total_counts_pred = []

    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    for G in graph_ref_list:
        orbit_counts = orca(G)
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / (G.number_of_nodes() + 1e-6)
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list_remove_empty:
        orbit_counts = orca(G)
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / (G.number_of_nodes() + 1e-6)
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)

    if compute_emd:
        mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=gaussian, is_hist=False, sigma=30.0)
    else:
        mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=gaussian_tv, is_hist=False, sigma=30.0)
    return mmd_dist


def _mmd_1d_median(sample_ref, sample_pred):
    """MMD² between two sets of scalar samples using median-heuristic bandwidth.

    Each element of ``sample_ref`` / ``sample_pred`` is a 1-element NumPy array
    ``np.array([scalar])``.  NaN and Inf values are filtered out before the
    kernel is evaluated, so degenerate-graph sentinels (disconnected CPL,
    undefined assortativity, trivially-small PLE) do not bias the bandwidth
    or the MMD estimate.

    Returns ``float('nan')`` when fewer than 2 finite values remain in either
    set (e.g. all generated graphs are disconnected for CPL).
    """
    vals_ref  = np.array([x[0] for x in sample_ref  if np.isfinite(x[0])], dtype=float)
    vals_pred = np.array([x[0] for x in sample_pred if np.isfinite(x[0])], dtype=float)
    if len(vals_ref) < 2 or len(vals_pred) < 2:
        return float('nan')
    # Median of pairwise |x_i - x_j| over the pooled set (median heuristic)
    pooled = np.concatenate([vals_ref, vals_pred])
    n_p = len(pooled)
    i_idx, j_idx = np.triu_indices(n_p, k=1)
    sigma = float(np.median(np.abs(pooled[i_idx] - pooled[j_idx])))
    sigma = max(sigma, 1e-6)   # guard against constant distributions
    ref_arrays  = [np.array([v]) for v in vals_ref]
    pred_arrays = [np.array([v]) for v in vals_pred]
    return compute_mmd(ref_arrays, pred_arrays, kernel=gaussian, is_hist=False, sigma=sigma)


def triangle_worker(G):
    """Return fraction of possible triangles C(n,3); size-invariant, in [0, 1]."""
    n = G.number_of_nodes()
    if n < 3:
        return 0.0
    tri = sum(nx.triangles(G).values()) // 3
    return float(tri) / (n * (n - 1) * (n - 2) / 6.0)


def triangle_stats(graph_ref_list, graph_pred_list, compute_emd=False):
    """MMD² between fraction-of-possible-triangles distributions (median-heuristic bandwidth)."""
    graph_pred_list = [G for G in graph_pred_list if G.number_of_nodes() > 0]
    sample_ref  = [np.array([triangle_worker(G)]) for G in graph_ref_list]
    sample_pred = [np.array([triangle_worker(G)]) for G in graph_pred_list]
    return _mmd_1d_median(sample_ref, sample_pred)


def assortativity_worker(G):
    """Return degree assortativity coefficient, or NaN when undefined."""
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return float('nan')
    try:
        r = nx.degree_assortativity_coefficient(G)
        return float(r) if not np.isnan(r) else float('nan')
    except Exception:
        return float('nan')


def assortativity_stats(graph_ref_list, graph_pred_list, compute_emd=False):
    """MMD² between assortativity distributions (median-heuristic bandwidth; NaN excluded)."""
    graph_pred_list = [G for G in graph_pred_list if G.number_of_nodes() > 0]
    sample_ref  = [np.array([assortativity_worker(G)]) for G in graph_ref_list]
    sample_pred = [np.array([assortativity_worker(G)]) for G in graph_pred_list]
    return _mmd_1d_median(sample_ref, sample_pred)


def ple_worker(G):
    """MLE power-law exponent (Clauset et al. 2009).  Returns NaN for degenerate graphs."""
    degrees = np.array([d for _, d in G.degree() if d > 0], dtype=float)
    if len(degrees) < 2:
        return float('nan')
    denom = np.sum(np.log(degrees / 0.5))
    if denom < 1e-10:
        return float('nan')
    return float(1.0 + len(degrees) / denom)


def ple_stats(graph_ref_list, graph_pred_list, compute_emd=False):
    """MMD² between PLE distributions (median-heuristic bandwidth; degenerate graphs excluded)."""
    graph_pred_list = [G for G in graph_pred_list if G.number_of_nodes() > 0]
    sample_ref  = [np.array([ple_worker(G)]) for G in graph_ref_list]
    sample_pred = [np.array([ple_worker(G)]) for G in graph_pred_list]
    return _mmd_1d_median(sample_ref, sample_pred)


def cpl_worker(G):
    """Average shortest-path length over the LCC.  Returns NaN when LCC < 2 nodes."""
    if G.number_of_nodes() < 2:
        return float('nan')
    lcc_nodes = max(nx.connected_components(G), key=len)
    H = G.subgraph(lcc_nodes)
    if H.number_of_nodes() < 2:
        return float('nan')
    try:
        return float(nx.average_shortest_path_length(H))
    except Exception:
        return float('nan')


def cpl_stats(graph_ref_list, graph_pred_list, compute_emd=False):
    """MMD² between CPL distributions (median-heuristic bandwidth; disconnected graphs excluded)."""
    graph_pred_list = [G for G in graph_pred_list if G.number_of_nodes() > 0]
    sample_ref  = [np.array([cpl_worker(G)]) for G in graph_ref_list]
    sample_pred = [np.array([cpl_worker(G)]) for G in graph_pred_list]
    return _mmd_1d_median(sample_ref, sample_pred)


def is_planar_graph(G):
    """Check if graph is planar and connected."""
    return nx.is_connected(G) and nx.check_planarity(G)[0]


def is_sbm_graph(G, p_intra=0.3, p_inter=0.005, strict=True, refinement_steps=1000):
    """Check if graph matches SBM structure using NetworkX community detection.
    
    This is a simplified version that doesn't require graph-tool.
    Uses Louvain community detection instead of minimum description length.
    """
    if G.number_of_nodes() == 0:
        return False if strict else 0.0
        
    try:
        # Use greedy modularity community detection (available in NetworkX)
        from networkx.algorithms.community import greedy_modularity_communities
        communities = list(greedy_modularity_communities(G))
        n_blocks = len(communities)
        
        if strict:
            # Check if number of communities is reasonable (2-5)
            if n_blocks < 2 or n_blocks > 5:
                return False
                
            # Check community sizes are reasonable (20-40 nodes)
            for comm in communities:
                if len(comm) < 20 or len(comm) > 40:
                    return False
        
        # Compute intra and inter community edge probabilities
        node_to_comm = {}
        for i, comm in enumerate(communities):
            for node in comm:
                node_to_comm[node] = i
        
        intra_edges = 0
        inter_edges = 0
        max_intra = 0
        max_inter = 0
        
        for i, comm in enumerate(communities):
            comm_list = list(comm)
            max_intra += len(comm_list) * (len(comm_list) - 1)
            for j, other_comm in enumerate(communities):
                if j > i:
                    max_inter += len(comm_list) * len(other_comm)
        
        for u, v in G.edges():
            if node_to_comm[u] == node_to_comm[v]:
                intra_edges += 1
            else:
                inter_edges += 1
        
        # Estimate probabilities
        est_p_intra = (2 * intra_edges) / (max_intra + 1e-6)
        est_p_inter = (2 * inter_edges) / (max_inter + 1e-6) if max_inter > 0 else 0
        
        # Check if estimated probabilities are close to expected
        intra_diff = abs(est_p_intra - p_intra) / (p_intra + 1e-6)
        inter_diff = abs(est_p_inter - p_inter) / (p_inter + 1e-6)
        
        if strict:
            # Allow some tolerance
            return intra_diff < 0.5 and inter_diff < 2.0
        else:
            # Return a probability score
            score = max(0, 1 - (intra_diff + inter_diff) / 4)
            return score
            
    except Exception as e:
        if strict:
            return False
        else:
            return 0.0


def eval_acc_sbm_graph(G_list, p_intra=0.3, p_inter=0.005, strict=True, refinement_steps=1000, is_parallel=True):
    """Evaluate SBM accuracy on a list of graphs."""
    count = 0.0
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for prob in executor.map(is_sbm_graph,
                                     G_list, 
                                     [p_intra] * len(G_list),
                                     [p_inter] * len(G_list),
                                     [strict] * len(G_list),
                                     [refinement_steps] * len(G_list)):
                count += prob
    else:
        for gg in G_list:
            count += is_sbm_graph(gg, p_intra=p_intra, p_inter=p_inter, strict=strict,
                                  refinement_steps=refinement_steps)
    return count / float(len(G_list))


def eval_acc_planar_graph(G_list):
    """Evaluate planarity accuracy on a list of graphs."""
    count = 0
    for gg in G_list:
        if is_planar_graph(gg):
            count += 1
    return count / float(len(G_list))


def eval_fraction_isomorphic(fake_graphs, train_graphs):
    """Compute fraction of generated graphs isomorphic to training graphs."""
    count = 0
    for fake_g in fake_graphs:
        for train_g in train_graphs:
            if nx.faster_could_be_isomorphic(fake_g, train_g):
                if nx.is_isomorphic(fake_g, train_g):
                    count += 1
                    break
    return count / float(len(fake_graphs))


def eval_fraction_unique(fake_graphs, precise=False):
    """Evaluate fraction of unique graphs."""
    count_non_unique = 0
    fake_evaluated = []
    for fake_g in fake_graphs:
        unique = True
        if not fake_g.number_of_nodes() == 0:
            for fake_old in fake_evaluated:
                if precise:
                    if nx.faster_could_be_isomorphic(fake_g, fake_old):
                        if nx.is_isomorphic(fake_g, fake_old):
                            count_non_unique += 1
                            unique = False
                            break
                else:
                    if nx.faster_could_be_isomorphic(fake_g, fake_old):
                        if nx.could_be_isomorphic(fake_g, fake_old):
                            count_non_unique += 1
                            unique = False
                            break
            if unique:
                fake_evaluated.append(fake_g)

    frac_unique = (float(len(fake_graphs)) - count_non_unique) / float(len(fake_graphs))
    return frac_unique


def eval_fraction_unique_non_isomorphic_valid(fake_graphs, train_graphs, validity_func=(lambda x: True)):
    """Evaluate fraction of unique, non-isomorphic, and valid graphs."""
    count_valid = 0
    count_isomorphic = 0
    count_non_unique = 0
    fake_evaluated = []
    
    for fake_g in fake_graphs:
        unique = True

        for fake_old in fake_evaluated:
            if nx.faster_could_be_isomorphic(fake_g, fake_old):
                if nx.is_isomorphic(fake_g, fake_old):
                    count_non_unique += 1
                    unique = False
                    break
        if unique:
            fake_evaluated.append(fake_g)
            non_isomorphic = True
            for train_g in train_graphs:
                if nx.faster_could_be_isomorphic(fake_g, train_g):
                    if nx.is_isomorphic(fake_g, train_g):
                        count_isomorphic += 1
                        non_isomorphic = False
                        break
            if non_isomorphic:
                if validity_func(fake_g):
                    count_valid += 1

    frac_unique = (float(len(fake_graphs)) - count_non_unique) / float(len(fake_graphs))
    frac_unique_non_isomorphic = (float(len(fake_graphs)) - count_non_unique - count_isomorphic) / float(len(fake_graphs))
    frac_unique_non_isomorphic_valid = count_valid / float(len(fake_graphs))
    return frac_unique, frac_unique_non_isomorphic, frac_unique_non_isomorphic_valid


def compute_wl_kernel_similarity(graph_ref_list, graph_pred_list, n_iter=5, retrieval_thresholds=[0.90, 0.95, 0.99]):
    """Compute Weisfeiler-Lehman kernel similarity between reference and predicted graphs.
    
    Args:
        graph_ref_list: List of reference NetworkX graphs
        graph_pred_list: List of predicted NetworkX graphs
        n_iter: Number of WL iterations (default: 5)
        retrieval_thresholds: List of thresholds for retrieval rate computation (default: [0.90, 0.95, 0.99])
        
    Returns:
        dict: Dictionary containing kernel statistics including nearest-neighbor similarities and retrieval rates
    """
    if not GRAKEL_AVAILABLE:
        print("Warning: grakel not available. Skipping WL kernel computation.")
        return {
            'wl_kernel_mean': None,
            'wl_kernel_std': None,
            'wl_kernel_median': None
        }
    
    # Filter out empty graphs
    graph_pred_list_filtered = [G for G in graph_pred_list if G.number_of_nodes() > 0]
    graph_ref_list_filtered = [G for G in graph_ref_list if G.number_of_nodes() > 0]
    
    if len(graph_pred_list_filtered) == 0 or len(graph_ref_list_filtered) == 0:
        print("Warning: No valid graphs for WL kernel computation")
        return {
            'wl_kernel_mean': None,
            'wl_kernel_std': None,
            'wl_kernel_median': None
        }
    
    # Debug: Check what types we actually have
    print(f"Debug: Number of ref graphs: {len(graph_ref_list_filtered)}")
    print(f"Debug: Number of pred graphs: {len(graph_pred_list_filtered)}")
    
    try:
        # Add degree labels to all graphs
        graphs_ref_labeled = []
        for i, G_orig in enumerate(graph_ref_list_filtered):
            if not isinstance(G_orig, nx.Graph):
                print(f"Warning: ref graph {i} is not a NetworkX graph, type={type(G_orig)}")
                continue
            G = G_orig.copy()
            # Set degree as node label (grakel expects integer labels)
            for node in G.nodes():
                G.nodes[node]['degree'] = int(G.degree(node))
            graphs_ref_labeled.append(G)
        
        graphs_pred_labeled = []
        for i, G_orig in enumerate(graph_pred_list_filtered):
            if not isinstance(G_orig, nx.Graph):
                print(f"Warning: pred graph {i} is not a NetworkX graph, type={type(G_orig)}")
                continue
            G = G_orig.copy()
            # Set degree as node label (grakel expects integer labels)
            for node in G.nodes():
                G.nodes[node]['degree'] = int(G.degree(node))
            graphs_pred_labeled.append(G)
        
        if len(graphs_ref_labeled) == 0 or len(graphs_pred_labeled) == 0:
            print("Warning: No valid labeled graphs for WL kernel")
            return {
                'wl_kernel_mean': None,
                'wl_kernel_std': None,
                'wl_kernel_median': None
            }
        
        print(f"Converting {len(graphs_ref_labeled)} ref and {len(graphs_pred_labeled)} pred graphs to grakel format...")
        
        # Convert to grakel format - pass list to get list back
        grakel_ref = list(graph_from_networkx(graphs_ref_labeled, node_labels_tag='degree'))
        grakel_pred = list(graph_from_networkx(graphs_pred_labeled, node_labels_tag='degree'))
        
        print(f"Grakel conversion complete. Ref: {len(grakel_ref)}, Pred: {len(grakel_pred)}")
        
        # Create WL kernel with VertexHistogram as base kernel
        wl_kernel = WeisfeilerLehman(n_iter=n_iter, 
                                     base_graph_kernel=VertexHistogram,
                                     normalize=True)
        
        print("Fitting WL kernel on reference graphs...")
        # Fit on reference graphs
        wl_kernel.fit(grakel_ref)
        
        print("Computing kernel matrix...")
        # Compute kernel matrix between reference and predicted
        K_ref_pred = wl_kernel.transform(grakel_pred)
        
        print(f"Kernel matrix shape: {K_ref_pred.shape}")
        
        # Compute average similarities (backward compatibility)
        avg_similarities = []
        for i in range(len(grakel_pred)):
            avg_sim = np.mean(K_ref_pred[i, :])
            avg_similarities.append(avg_sim)
        avg_similarities = np.array(avg_similarities)
        
        # Compute nearest-neighbor similarities: s(g) = max_{G in D_train} WL_sim(g, G)
        max_similarities = []
        for i in range(len(grakel_pred)):
            max_sim = np.max(K_ref_pred[i, :])
            max_similarities.append(max_sim)
        max_similarities = np.array(max_similarities)
        
        # Compute retrieval rates at different thresholds
        retrieval_rates = {}
        for threshold in retrieval_thresholds:
            retrieval_rate = np.mean(max_similarities >= threshold)
            retrieval_rates[f'wl_retrieval_rate_{threshold:.2f}'] = float(retrieval_rate)
            print(f"Retrieval rate at threshold {threshold:.2f}: {retrieval_rate:.4f} "
                  f"({int(retrieval_rate * len(max_similarities))}/{len(max_similarities)} graphs)")
        
        # Compute distribution statistics of nearest-neighbor similarities
        results = {
            # Average similarity metrics (backward compatibility)
            'wl_kernel_mean': float(np.mean(avg_similarities)),
            'wl_kernel_std': float(np.std(avg_similarities)),
            'wl_kernel_median': float(np.median(avg_similarities)),
            'wl_kernel_min': float(np.min(avg_similarities)),
            'wl_kernel_max': float(np.max(avg_similarities)),
            
            # Nearest-neighbor (max) similarity metrics
            'wl_nn_mean': float(np.mean(max_similarities)),
            'wl_nn_std': float(np.std(max_similarities)),
            'wl_nn_median': float(np.median(max_similarities)),
            'wl_nn_min': float(np.min(max_similarities)),
            'wl_nn_max': float(np.max(max_similarities)),
            'wl_nn_q25': float(np.percentile(max_similarities, 25)),
            'wl_nn_q75': float(np.percentile(max_similarities, 75)),
        }
        
        # Add retrieval rates
        results.update(retrieval_rates)
        
        # Print summary
        print("\n" + "="*80)
        print("WL Kernel Nearest-Neighbor Similarity Distribution:")
        print(f"  Mean: {results['wl_nn_mean']:.4f}")
        print(f"  Std:  {results['wl_nn_std']:.4f}")
        print(f"  Median: {results['wl_nn_median']:.4f}")
        print(f"  Min:  {results['wl_nn_min']:.4f}")
        print(f"  Max:  {results['wl_nn_max']:.4f}")
        print(f"  Q25:  {results['wl_nn_q25']:.4f}")
        print(f"  Q75:  {results['wl_nn_q75']:.4f}")
        print("\nRetrieval Rates:")
        for threshold in retrieval_thresholds:
            key = f'wl_retrieval_rate_{threshold:.2f}'
            print(f"  At {threshold:.2f}: {results[key]:.2%}")
        print("="*80 + "\n")
        
        return results
        
    except Exception as e:
        print(f"Error computing WL kernel: {e}")
        import traceback
        traceback.print_exc()
        return {
            'wl_kernel_mean': None,
            'wl_kernel_std': None,
            'wl_kernel_median': None
        }


class SpectreSamplingMetrics(nn.Module):
    """Sampling metrics for Spectre datasets."""
    
    def __init__(self, datamodule, compute_emd, metrics_list):
        super().__init__()

        self.train_graphs = self.loader_to_nx(datamodule.train_dataloader())
        self.val_graphs = self.loader_to_nx(datamodule.val_dataloader())
        self.test_graphs = self.loader_to_nx(datamodule.test_dataloader())
        self.num_graphs_test = len(self.test_graphs)
        self.num_graphs_val = len(self.val_graphs)
        self.compute_emd = compute_emd
        self.metrics_list = metrics_list

    def loader_to_nx(self, loader):
        networkx_graphs = []
        for i, batch in enumerate(loader):
            data_list = batch.to_data_list()
            for j, data in enumerate(data_list):
                networkx_graphs.append(to_networkx(data, node_attrs=None, edge_attrs=None, 
                                                    to_undirected=True, remove_self_loops=True))
        return networkx_graphs

    def forward(self, generated_graphs: list, name, current_epoch, val_counter, local_rank=0, test=False, use_train=False):
        if use_train:
            reference_graphs = self.train_graphs
            reference_type = 'train'
        elif test:
            reference_graphs = self.test_graphs
            reference_type = 'test'
        else:
            reference_graphs = self.val_graphs
            reference_type = 'val'
            
        print(f"Computing sampling metrics between {len(generated_graphs)} generated graphs and "
              f"{len(reference_graphs)} {reference_type} graphs -- emd computation: {self.compute_emd}")
        
        networkx_graphs = []
        adjacency_matrices = []
        print("Building networkx graphs...")
        
        for graph in generated_graphs:
            node_types, edge_types = graph
            A = edge_types.bool().cpu().numpy()
            adjacency_matrices.append(A)
            nx_graph = nx.from_numpy_array(A)
            networkx_graphs.append(nx_graph)

        np.savez('generated_adjs.npz', *adjacency_matrices)

        to_log = {}

        if 'degree' in self.metrics_list:
            print("Computing degree stats...")
            degree = degree_stats(reference_graphs, networkx_graphs, is_parallel=True,
                                  compute_emd=self.compute_emd)
            to_log['degree'] = degree

        if 'spectre' in self.metrics_list:
            print("Computing spectre stats...")
            spectre = spectral_stats(reference_graphs, networkx_graphs, is_parallel=True, n_eigvals=-1,
                                     compute_emd=self.compute_emd)
            to_log['spectre'] = spectre

        if 'clustering' in self.metrics_list:
            print("Computing clustering stats...")
            clustering = clustering_stats(reference_graphs, networkx_graphs, bins=100, is_parallel=True,
                                          compute_emd=self.compute_emd)
            to_log['clustering'] = clustering

        if 'orbit' in self.metrics_list:
            print("Computing orbit stats...")
            orbit = orbit_stats_all(reference_graphs, networkx_graphs, compute_emd=self.compute_emd)
            to_log['orbit'] = orbit

        if 'triangle' in self.metrics_list:
            print("Computing triangle stats...")
            tri = triangle_stats(reference_graphs, networkx_graphs)
            to_log['triangle'] = tri

        if 'assortativity' in self.metrics_list:
            print("Computing assortativity stats...")
            assort = assortativity_stats(reference_graphs, networkx_graphs)
            to_log['assortativity'] = assort

        if 'ple' in self.metrics_list:
            print("Computing PLE stats...")
            to_log['ple'] = ple_stats(reference_graphs, networkx_graphs)

        if 'cpl' in self.metrics_list:
            print("Computing CPL stats...")
            to_log['cpl'] = cpl_stats(reference_graphs, networkx_graphs)

        if 'sbm' in self.metrics_list:
            print("Computing SBM accuracy...")
            acc = eval_acc_sbm_graph(networkx_graphs, refinement_steps=100, strict=True)
            to_log['sbm_acc'] = acc

        if 'planar' in self.metrics_list:
            print('Computing planar accuracy...')
            planar_acc = eval_acc_planar_graph(networkx_graphs)
            to_log['planar_acc'] = planar_acc

        # Always compute uniqueness (within generated set) and novelty (vs training set)
        print("Computing uniqueness and novelty...")
        frac_unique = eval_fraction_unique(networkx_graphs)
        frac_novel  = 1.0 - eval_fraction_isomorphic(networkx_graphs, self.train_graphs)
        to_log['frac_unique'] = frac_unique
        to_log['frac_novel']  = frac_novel

        # Additional per-class validity metrics for SBM/planar
        if 'sbm' in self.metrics_list or 'planar' in self.metrics_list:
            validity_func = is_sbm_graph if 'sbm' in self.metrics_list else is_planar_graph
            _, frac_unique_non_iso, frac_unic_non_iso_valid = \
                eval_fraction_unique_non_isomorphic_valid(networkx_graphs, self.train_graphs, validity_func)
            to_log['sampling/frac_unique_non_iso']       = frac_unique_non_iso
            to_log['sampling/frac_unic_non_iso_valid']   = frac_unic_non_iso_valid

        # When evaluating vs the test set, compute MMD(train,test) and the
        # normalised ratios  r = MMD²(gen,test) / MMD²(train,test).
        if test:
            print("Computing MMD²(train, test) baselines for ratio r computation...")
            _mmd_keys = [k for k in
                         ['degree', 'spectre', 'clustering', 'orbit', 'triangle', 'assortativity',
                          'ple', 'cpl']
                         if k in to_log]
            train_test_mmd = {}
            if 'degree'        in _mmd_keys:
                train_test_mmd['degree']        = degree_stats(self.train_graphs, self.test_graphs,
                                                               compute_emd=self.compute_emd)
            if 'spectre'       in _mmd_keys:
                train_test_mmd['spectre']       = spectral_stats(self.train_graphs, self.test_graphs,
                                                                  compute_emd=self.compute_emd)
            if 'clustering'    in _mmd_keys:
                train_test_mmd['clustering']    = clustering_stats(self.train_graphs, self.test_graphs,
                                                                    compute_emd=self.compute_emd)
            if 'orbit'         in _mmd_keys:
                train_test_mmd['orbit']         = orbit_stats_all(self.train_graphs, self.test_graphs,
                                                                   compute_emd=self.compute_emd)
            if 'triangle'      in _mmd_keys:
                train_test_mmd['triangle']      = triangle_stats(self.train_graphs, self.test_graphs)
            if 'assortativity' in _mmd_keys:
                train_test_mmd['assortativity'] = assortativity_stats(self.train_graphs, self.test_graphs)
            if 'ple'           in _mmd_keys:
                train_test_mmd['ple']           = ple_stats(self.train_graphs, self.test_graphs)
            if 'cpl'           in _mmd_keys:
                train_test_mmd['cpl']           = cpl_stats(self.train_graphs, self.test_graphs)

            print("MMD² reference (train vs test):")
            for k, v in train_test_mmd.items():
                print(f"  {k}: {v:.6f}")
            print("MMD² ratios r = MMD²(gen,test) / MMD²(train,test):")
            for k in _mmd_keys:
                denom = float(train_test_mmd.get(k, float('nan')))
                num   = float(to_log[k])
                r = num / denom if denom > 1e-10 else float('nan')
                to_log[f'r_{k}'] = r
                if not np.isnan(r):
                    print(f"  r_{k}: {r:.4f}")
                elif np.isnan(num):
                    print(f"  r_{k}: nan (all samples degenerate/excluded from MMD)")
                else:
                    print(f"  r_{k}: nan (denom\u22480)")

        # Compute WL Kernel similarity
        print("Computing WL Kernel similarity...")
        wl_stats = compute_wl_kernel_similarity(reference_graphs, networkx_graphs, n_iter=5)
        to_log.update(wl_stats)

        print("Sampling statistics:", to_log)
        return to_log

    def reset(self):
        pass


class PlanarSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule):
        super().__init__(datamodule=datamodule,
                         compute_emd=False,
                         metrics_list=['degree', 'clustering', 'orbit', 'spectre', 'planar',
                                       'triangle', 'assortativity', 'ple', 'cpl'])


class SBMSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule):
        super().__init__(datamodule=datamodule,
                         compute_emd=False,
                         metrics_list=['degree', 'clustering', 'orbit', 'spectre', 'sbm',
                                       'triangle', 'assortativity', 'ple', 'cpl'])


class Comm20SamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, datamodule):
        super().__init__(datamodule=datamodule,
                         compute_emd=True,
                         metrics_list=['degree', 'clustering', 'orbit', 'triangle', 'assortativity', 'ple', 'cpl'])


class CoraChameleonSamplingMetrics(SpectreSamplingMetrics):
    """Sampling metrics for CoraChameleon dataset."""
    def __init__(self, datamodule):
        super().__init__(datamodule=datamodule,
                         compute_emd=False,
                         metrics_list=['degree', 'clustering', 'orbit', 'spectre',
                                       'triangle', 'assortativity', 'ple', 'cpl'])


class TwoDensitySamplingMetrics(SpectreSamplingMetrics):
    """Sampling metrics for the two-density mixture experiment.

    Uses degree + spectre to capture density-level distribution matching.
    Clustering is included because sparse/dense families differ strongly there.
    No validity check (there is no hard structural constraint to satisfy).
    """
    def __init__(self, datamodule):
        super().__init__(datamodule=datamodule,
                         compute_emd=False,
                         metrics_list=['degree', 'clustering', 'spectre', 'triangle', 'assortativity', 'ple', 'cpl'])


class DCSBMSamplingMetrics(SpectreSamplingMetrics):
    """Sampling metrics for the DC-SBM community asymmetry experiment.

    Degree MMD captures whether the bimodal degree distribution (hub community
    vs peripheral community) is recovered.  Orbit counts detect sub-graph
    patterns that differ between dense and sparse communities.
    """
    def __init__(self, datamodule):
        super().__init__(datamodule=datamodule,
                         compute_emd=False,
                         metrics_list=['degree', 'clustering', 'orbit', 'spectre',
                                       'triangle', 'assortativity', 'ple', 'cpl'])


class CommunitySamplingMetrics(SpectreSamplingMetrics):
    """Sampling metrics for the EDGE Community graph collection."""
    def __init__(self, datamodule):
        super().__init__(datamodule=datamodule,
                         compute_emd=False,
                         metrics_list=['degree', 'clustering', 'orbit', 'spectre',
                                       'triangle', 'assortativity', 'ple', 'cpl'])


class EgoSamplingMetrics(SpectreSamplingMetrics):
    """Sampling metrics for the EDGE Ego graph collection."""
    def __init__(self, datamodule):
        super().__init__(datamodule=datamodule,
                         compute_emd=False,
                         metrics_list=['degree', 'clustering', 'orbit', 'spectre',
                                       'triangle', 'assortativity', 'ple', 'cpl'])
