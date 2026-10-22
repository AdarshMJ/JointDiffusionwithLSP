#!/usr/bin/env python3
"""Fast seeded synthetic graph generator."""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sparse
import scipy as sp
import torch
import networkx as nx
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.utils import remove_self_loops


def compute_adaptive_degree_constraints(node_size: int) -> Tuple[float, float]:
    """Compute adaptive min/max average degree constraints based on node size.
    
    For small graphs (n <= 10), we need much lower min_avg_degree to allow
    diverse structures like trees, paths, cycles, etc. For example, with n=5:
    - Complete graph K5 has avg_degree = 4.0 (max possible)
    - A tree has avg_degree ≈ 1.6 (n-1 edges = 4 edges, avg = 8/5 = 1.6)
    - A cycle has avg_degree = 2.0
    - We want diversity from trees to dense graphs
    
    Args:
        node_size: Number of nodes in the graph
        
    Returns:
        (min_avg_degree, max_avg_degree) tuple
    """
    # Maximum possible average degree for n nodes (complete graph)
    max_possible_degree = node_size - 1
    
    # For very small graphs (n <= 10), allow very sparse to dense range
    # This enables trees, paths, cycles, sparse graphs, and dense graphs
    if node_size <= 10:
        # Minimum: allow trees (n-1 edges), which have avg_degree = 2*(n-1)/n ≈ 2 - 2/n
        # For n=5: tree avg_degree = 1.6, cycle = 2.0
        # We use a very low floor to allow sparse graphs
        min_avg_degree = 2.0 * (node_size - 1) / node_size - 0.2  # Slightly below tree
        min_avg_degree = max(1.0, min_avg_degree)  # Floor at 1.0 (very sparse)
        
        # Maximum: allow up to 90% of complete graph for variety
        max_avg_degree = 0.90 * max_possible_degree
    elif node_size <= 20:
        # For medium-small graphs: still allow sparse to fairly dense
        min_fraction = 0.25  # Allow sparser graphs
        max_fraction = 0.85
        min_avg_degree = max(1.5, min_fraction * max_possible_degree)
        max_avg_degree = max_fraction * max_possible_degree
    else:
        # For large graphs, use fixed reasonable constraints
        min_avg_degree = 4.0
        max_avg_degree = 12.0
    
    # Safety check: ensure min < max with reasonable gap
    if min_avg_degree >= max_avg_degree - 0.5:
        max_avg_degree = min_avg_degree + 1.0
    
    return min_avg_degree, max_avg_degree


def gen_stats(G: nx.Graph) -> np.ndarray:
    """Compute basic graph statistics.
    
    Returns array of 15 statistics:
    [num_nodes, num_edges, density, avg_degree, std_degree, max_degree, min_degree,
     avg_clustering, transitivity, assortativity, num_components, largest_cc_ratio,
     avg_shortest_path, diameter, num_triangles]
    """
    n = G.number_of_nodes()
    m = G.number_of_edges()
    
    if n == 0:
        return np.zeros(15)
    
    # Basic stats
    density = nx.density(G) if n > 1 else 0.0
    degrees = [d for _, d in G.degree()]
    avg_degree = np.mean(degrees) if degrees else 0.0
    std_degree = np.std(degrees) if degrees else 0.0
    max_degree = max(degrees) if degrees else 0
    min_degree = min(degrees) if degrees else 0
    
    # Clustering
    try:
        avg_clustering = nx.average_clustering(G)
    except:
        avg_clustering = 0.0
    
    try:
        transitivity = nx.transitivity(G)
    except:
        transitivity = 0.0
    
    # Assortativity
    try:
        assortativity = nx.degree_assortativity_coefficient(G) if m > 0 else 0.0
    except:
        assortativity = 0.0
    
    # Components
    num_components = nx.number_connected_components(G)
    if num_components > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        largest_cc_ratio = len(largest_cc) / n
    else:
        largest_cc_ratio = 0.0
    
    # Path metrics (only for largest connected component)
    try:
        if len(largest_cc) > 1:
            subG = G.subgraph(largest_cc)
            avg_shortest_path = nx.average_shortest_path_length(subG)
            diameter = nx.diameter(subG)
        else:
            avg_shortest_path = 0.0
            diameter = 0
    except:
        avg_shortest_path = 0.0
        diameter = 0
    
    # Triangles
    try:
        num_triangles = sum(nx.triangles(G).values()) // 3
    except:
        num_triangles = 0
    
    return np.array([
        n, m, density, avg_degree, std_degree, max_degree, min_degree,
        avg_clustering, transitivity, assortativity, num_components, largest_cc_ratio,
        avg_shortest_path, diameter, num_triangles
    ], dtype=float)


@dataclass
class SeedStats:
    name: str
    class_probs: np.ndarray
    pair_probs: np.ndarray
    avg_degree: float
    num_classes: int


class NodeFeaturePool:
    """Pool of node features from real-world graphs, reduced to fixed dimension via PCA.
    
    Features are sampled (with replacement) when generating synthetic graphs.
    """
    
    def __init__(
        self,
        dataset_names: List[str],
        root: str = './planetoid',
        target_dim: int = 100,
        seed: int = 42
    ):
        """Initialize feature pool from specified datasets.
        
        Args:
            dataset_names: List of dataset names (e.g., ['Cora', 'Chameleon'])
            root: Root directory for datasets
            target_dim: Target feature dimension after PCA
            seed: Random seed for PCA
        """
        self.dataset_names = dataset_names
        self.target_dim = target_dim
        self.seed = seed
        
        # Load and pool features from all datasets
        all_features = []
        max_dim = 0
        
        # First pass: load and find max dimension
        loaded_features = []
        for name in dataset_names:
            features = self._load_features(name, root)
            loaded_features.append(features)
            max_dim = max(max_dim, features.shape[1])
            print(f"  Loaded {features.shape[0]} nodes from {name} ({features.shape[1]}-dim)")
        
        # Second pass: pad all to same dimension
        for i, features in enumerate(loaded_features):
            if features.shape[1] < max_dim:
                padding = np.zeros((features.shape[0], max_dim - features.shape[1]))
                features = np.hstack([features, padding])
            all_features.append(features)
        
        # Concatenate all features
        pooled = np.vstack(all_features)
        print(f"  Total pooled: {pooled.shape[0]} nodes, padded to {max_dim}-dim")
        
        # Apply PCA to reduce dimension
        np.random.seed(seed)
        actual_dim = min(target_dim, pooled.shape[1], pooled.shape[0])
        self.pca = PCA(n_components=actual_dim, random_state=seed)
        self.features = self.pca.fit_transform(pooled)
        
        # Normalize to [0, 1] range
        self.scaler = MinMaxScaler()
        self.features = self.scaler.fit_transform(self.features)
        
        # Pad to target_dim if needed
        if self.features.shape[1] < target_dim:
            padding = np.zeros((self.features.shape[0], target_dim - self.features.shape[1]))
            self.features = np.hstack([self.features, padding])
        
        self.features = self.features.astype(np.float32)
        print(f"  Final feature pool: {self.features.shape} (PCA variance explained: {self.pca.explained_variance_ratio_.sum():.3f})")
    
    def _load_features(self, name: str, root: str) -> np.ndarray:
        """Load node features from a dataset."""
        name_lower = name.lower()
        if name_lower in ['chameleon', 'squirrel', 'crocodile']:
            dataset = WikipediaNetwork(root=root, name=name_lower)
        else:
            dataset = Planetoid(root=root, name=name)
        
        data = dataset[0]
        return data.x.cpu().numpy()
    
    def sample(self, n: int, seed: int = None) -> torch.Tensor:
        """Sample n node features from the pool (with replacement).
        
        Args:
            n: Number of features to sample
            seed: Optional random seed for reproducibility
            
        Returns:
            Tensor of shape [n, target_dim]
        """
        rng = np.random.RandomState(seed) if seed is not None else np.random
        indices = rng.choice(len(self.features), size=n, replace=True)
        sampled = self.features[indices]
        return torch.from_numpy(sampled).float()
    
    def __len__(self) -> int:
        return len(self.features)


def create_feature_pools(root: str = './planetoid', target_dim: int = 100, seed: int = 42) -> Tuple['NodeFeaturePool', 'NodeFeaturePool']:
    """Create feature pools for S1 and S2.
    
    S1: Cora + Chameleon
    S2: Citeseer + Squirrel
    
    Returns:
        (pool_s1, pool_s2): Feature pools for each split
    """
    print("Creating feature pool for S1 (Cora + Chameleon)...")
    pool_s1 = NodeFeaturePool(
        dataset_names=['Cora', 'Chameleon'],
        root=root,
        target_dim=target_dim,
        seed=seed
    )
    
    print("Creating feature pool for S2 (Citeseer + Squirrel)...")
    pool_s2 = NodeFeaturePool(
        dataset_names=['Citeseer', 'Squirrel'],
        root=root,
        target_dim=target_dim,
        seed=seed + 1  # Different seed for variety
    )
    
    return pool_s1, pool_s2


def _coalesce_edges(edge_index: torch.Tensor) -> np.ndarray:
    edge_index, _ = remove_self_loops(edge_index)
    edges = set()
    for u, v in edge_index.t().tolist():
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        edges.add((a, b))
    return np.array(list(edges), dtype=np.int64)


def load_seed_stats(name: str, root: str) -> SeedStats:
    dataset = Planetoid(root=root, name=name)
    data = dataset[0]
    edges = _coalesce_edges(data.edge_index)
    labels = data.y.cpu().numpy()
    num_nodes = int(data.num_nodes)
    num_classes = int(labels.max()) + 1

    class_counts = np.bincount(labels, minlength=num_classes).astype(float)
    class_probs = class_counts / class_counts.sum()

    pair_counts = np.zeros((num_classes, num_classes), dtype=np.float64)
    for u, v in edges:
        cu, cv = labels[u], labels[v]
        pair_counts[cu, cv] += 1.0
        if cu != cv:
            pair_counts[cv, cu] += 1.0

    if pair_counts.sum() == 0:
        pair_probs = np.full((num_classes, num_classes), 1.0 / (num_classes**2))
    else:
        pair_probs = pair_counts / pair_counts.sum()

    avg_degree = 2.0 * len(edges) / num_nodes if num_nodes else 0.0

    return SeedStats(
        name=name,
        class_probs=class_probs,
        pair_probs=pair_probs,
        avg_degree=avg_degree,
        num_classes=num_classes,
    )


def load_full_graph(name: str, root: str) -> Tuple[nx.Graph, np.ndarray, Dict]:
    """Load a Planetoid or WikipediaNetwork dataset as a NetworkX graph with labels.
    
    Returns:
        G: NetworkX graph
        labels: Node labels array
        metadata: Dict with dataset info
    """
    # Check if it's a WikipediaNetwork dataset
    if name.lower() in ['chameleon', 'squirrel', 'crocodile']:
        dataset = WikipediaNetwork(root=root, name=name.lower())
    else:
        # Planetoid datasets (Cora, CiteSeer, PubMed)
        dataset = Planetoid(root=root, name=name)
    
    data = dataset[0]
    
    # Build NetworkX graph
    G = nx.Graph()
    num_nodes = int(data.num_nodes)
    G.add_nodes_from(range(num_nodes))
    
    # Add edges
    edges = _coalesce_edges(data.edge_index)
    G.add_edges_from([(int(u), int(v)) for u, v in edges])
    
    # Get labels
    labels = data.y.cpu().numpy()
    num_classes = int(labels.max()) + 1
    
    # Metadata
    metadata = {
        'name': name,
        'num_nodes': num_nodes,
        'num_edges': len(edges),
        'num_classes': num_classes,
        'avg_degree': 2.0 * len(edges) / num_nodes if num_nodes > 0 else 0.0
    }
    
    return G, labels, metadata


def sample_subgraph_random_walk(G: nx.Graph, target_size: int, max_attempts: int = 100) -> List[int]:
    """Sample a connected subgraph using random walk.
    
    Args:
        G: NetworkX graph
        target_size: Desired number of nodes
        max_attempts: Maximum number of attempts
        
    Returns:
        List of node indices in the subgraph
    """
    nodes = list(G.nodes())
    
    # Pre-compute largest connected component to avoid isolated nodes
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        nodes = list(largest_cc)
    
    for attempt in range(max_attempts):
        # Start from random node with preference for higher degree nodes (better connectivity)
        if attempt < max_attempts // 2:
            start = random.choice(nodes)
        else:
            # After half the attempts, prefer high-degree nodes
            degrees = [(n, G.degree(n)) for n in nodes]
            degrees.sort(key=lambda x: x[1], reverse=True)
            start = degrees[random.randint(0, min(20, len(degrees)-1))][0]
        
        visited = {start}
        current = start
        
        # Random walk until we have enough nodes
        stuck_counter = 0
        while len(visited) < target_size * 1.5:  # Collect extra to ensure connectivity
            neighbors = list(G.neighbors(current))
            if not neighbors:
                # Dead end - jump to random visited node
                if len(visited) > 1:
                    current = random.choice(list(visited))
                else:
                    break
                stuck_counter += 1
                if stuck_counter > 10:
                    break
                continue
            
            # Choose next node: higher probability for neighbors
            if random.random() < 0.8 and neighbors:
                current = random.choice(neighbors)
            else:
                # Jump to explore different regions (but within visited + their neighbors)
                candidate_nodes = list(visited)
                for v in list(visited)[:10]:  # Check neighbors of some visited nodes
                    candidate_nodes.extend(G.neighbors(v))
                current = random.choice(candidate_nodes) if candidate_nodes else random.choice(nodes)
            
            visited.add(current)
            stuck_counter = 0
        
        # Check if subgraph is connected and take largest component if needed
        if len(visited) >= target_size:
            subG = G.subgraph(list(visited))
            if nx.is_connected(subG):
                return list(visited)[:target_size]
            else:
                # Take largest connected component
                largest_cc = max(nx.connected_components(subG), key=len)
                if len(largest_cc) >= target_size:
                    return list(largest_cc)[:target_size]
    
    # Final fallback: use BFS from high-degree node
    degrees = [(n, G.degree(n)) for n in nodes]
    degrees.sort(key=lambda x: x[1], reverse=True)
    start = degrees[0][0]
    visited = list(nx.bfs_tree(G, start).nodes())[:target_size * 2]
    subG = G.subgraph(visited)
    if nx.is_connected(subG) and len(visited) >= target_size:
        return visited[:target_size]
    
    raise RuntimeError(f"Failed to sample connected subgraph of size {target_size} after {max_attempts} attempts")


def sample_subgraph_bfs(G: nx.Graph, target_size: int, max_attempts: int = 100) -> List[int]:
    """Sample a connected subgraph using BFS from a random seed.
    
    Args:
        G: NetworkX graph
        target_size: Desired number of nodes
        max_attempts: Maximum number of attempts
        
    Returns:
        List of node indices in the subgraph
    """
    nodes = list(G.nodes())
    
    # Pre-compute largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        nodes = list(largest_cc)
    
    for attempt in range(max_attempts):
        # Start from random node, prefer high-degree after some attempts
        if attempt < max_attempts // 2:
            start = random.choice(nodes)
        else:
            degrees = [(n, G.degree(n)) for n in nodes]
            degrees.sort(key=lambda x: x[1], reverse=True)
            start = degrees[random.randint(0, min(20, len(degrees)-1))][0]
        
        visited = []
        queue = [start]
        seen = {start}
        
        while queue and len(visited) < target_size:
            current = queue.pop(0)
            visited.append(current)
            
            # Add neighbors to queue
            neighbors = list(G.neighbors(current))
            random.shuffle(neighbors)
            
            for neighbor in neighbors:
                if neighbor not in seen and len(visited) + len(queue) < target_size * 1.2:
                    seen.add(neighbor)
                    queue.append(neighbor)
        
        # BFS naturally produces connected components
        if len(visited) >= target_size:
            return visited[:target_size]
    
    raise RuntimeError(f"Failed to sample connected subgraph of size {target_size} after {max_attempts} attempts")


def sample_subgraph_forest_fire(G: nx.Graph, target_size: int, p_forward: float = 0.7, 
                                 p_backward: float = 0.3, max_attempts: int = 100) -> List[int]:
    """Sample a connected subgraph using forest fire algorithm.
    
    Args:
        G: NetworkX graph
        target_size: Desired number of nodes
        p_forward: Probability of burning forward edges
        p_backward: Probability of burning backward edges
        max_attempts: Maximum number of attempts
        
    Returns:
        List of node indices in the subgraph
    """
    nodes = list(G.nodes())
    
    # Pre-compute largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        nodes = list(largest_cc)
    
    for attempt in range(max_attempts):
        # Start from random node, prefer high-degree nodes for better spread
        if attempt < max_attempts // 2:
            start = random.choice(nodes)
        else:
            degrees = [(n, G.degree(n)) for n in nodes]
            degrees.sort(key=lambda x: x[1], reverse=True)
            start = degrees[random.randint(0, min(20, len(degrees)-1))][0]
        
        burned = {start}
        active = [start]
        
        while active and len(burned) < target_size * 1.2:
            current = active.pop(0)
            
            # Get neighbors
            neighbors = list(G.neighbors(current))
            unburned_neighbors = [n for n in neighbors if n not in burned]
            
            if not unburned_neighbors:
                continue
            
            # Burn forward edges with probability
            num_to_burn = int(np.random.geometric(1 - p_forward) if p_forward < 1 else len(unburned_neighbors))
            num_to_burn = min(num_to_burn, len(unburned_neighbors), target_size * 2 - len(burned))
            
            to_burn = random.sample(unburned_neighbors, num_to_burn) if unburned_neighbors else []
            
            for node in to_burn:
                burned.add(node)
                active.append(node)
        
        # Forest fire naturally produces connected components
        if len(burned) >= target_size:
            return list(burned)[:target_size]
    
    raise RuntimeError(f"Failed to sample connected subgraph of size {target_size} after {max_attempts} attempts")


def sample_subgraph_dense_core(G: nx.Graph, target_size: int, min_avg_degree: float = 4.0,
                                max_attempts: int = 100) -> List[int]:
    """Sample a dense, connected subgraph by starting from high-degree nodes and expanding.
    
    This method prioritizes density over tree-like expansion:
    1. Start from a high-degree node
    2. Greedily add neighbors that maximize edge density
    3. Reject subgraphs that are too sparse
    
    Args:
        G: NetworkX graph
        target_size: Desired number of nodes
        min_avg_degree: Minimum average degree for the subgraph
        max_attempts: Maximum number of attempts
        
    Returns:
        List of node indices in the subgraph
    """
    nodes = list(G.nodes())
    
    # Pre-compute largest connected component
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        nodes = list(largest_cc)
    
    # Pre-compute node degrees in original graph
    node_degrees = {n: G.degree(n) for n in nodes}
    
    for attempt in range(max_attempts):
        # Start from high-degree node (top 10%)
        sorted_by_degree = sorted(nodes, key=lambda n: node_degrees[n], reverse=True)
        top_k = max(1, len(sorted_by_degree) // 10)
        start = sorted_by_degree[random.randint(0, top_k - 1)]
        
        selected = {start}
        candidates = set(G.neighbors(start)) - selected
        
        while len(selected) < target_size and candidates:
            # Score each candidate by how many edges it adds to selected set
            best_candidate = None
            best_score = -1
            
            # Sample candidates if too many (for efficiency)
            eval_candidates = list(candidates)
            if len(eval_candidates) > 50:
                eval_candidates = random.sample(eval_candidates, 50)
            
            for cand in eval_candidates:
                # Count edges to already-selected nodes
                edges_to_selected = sum(1 for neighbor in G.neighbors(cand) if neighbor in selected)
                # Bonus for high-degree nodes
                degree_bonus = min(node_degrees[cand] / 20.0, 1.0)
                score = edges_to_selected + degree_bonus * 0.5
                
                if score > best_score:
                    best_score = score
                    best_candidate = cand
            
            if best_candidate is None:
                break
            
            selected.add(best_candidate)
            candidates.discard(best_candidate)
            
            # Add new neighbors as candidates
            for neighbor in G.neighbors(best_candidate):
                if neighbor not in selected:
                    candidates.add(neighbor)
        
        # Check if we got enough nodes and if density is acceptable
        if len(selected) >= target_size:
            selected_list = list(selected)[:target_size]
            subG = G.subgraph(selected_list)
            
            if nx.is_connected(subG):
                num_edges = subG.number_of_edges()
                avg_degree = 2.0 * num_edges / target_size
                
                if avg_degree >= min_avg_degree:
                    return selected_list
    
    # Fallback: return best attempt even if below density threshold
    raise RuntimeError(f"Failed to sample dense subgraph of size {target_size} after {max_attempts} attempts")


def sample_real_world_subgraph(G: nx.Graph, labels: np.ndarray, target_size: int,
                               feature_pool: NodeFeaturePool,
                               method: str = 'mixed',
                               feature_dim: int = 100,
                               min_avg_degree: float = 4.0,
                               max_avg_degree: float = 12.0,
                               max_sampling_attempts: int = 20,
                               seed: int = None) -> Tuple[Data, Dict[str, float]]:
    """Sample a subgraph from a real-world graph and convert to PyG Data.
    
    Args:
        G: Source NetworkX graph
        labels: Node labels for the full graph
        target_size: Desired subgraph size
        feature_pool: NodeFeaturePool to sample features from
        method: Sampling method ('random_walk', 'bfs', 'forest_fire', 'dense_core', or 'mixed')
        feature_dim: Feature dimension
        min_avg_degree: Minimum average degree for subgraph (filters out sparse/spoke-like graphs)
        max_avg_degree: Maximum average degree for subgraph (filters out overly dense graphs)
        max_sampling_attempts: Number of attempts to get a graph within density range
        seed: Random seed for feature sampling
        
    Returns:
        (data, metadata): PyG Data object and metadata dict
    """
    
    best_subgraph = None
    best_avg_degree = None
    best_distance_from_target = float('inf')
    target_avg_degree = (min_avg_degree + max_avg_degree) / 2  # Ideal is middle of range
    
    # Try multiple times to get a subgraph within density range
    for attempt in range(max_sampling_attempts):
        # Sample subgraph using specified method
        # For dense graphs, prefer forest_fire and random_walk over dense_core
        if method == 'random_walk':
            subgraph_nodes = sample_subgraph_random_walk(G, target_size)
        elif method == 'bfs':
            subgraph_nodes = sample_subgraph_bfs(G, target_size)
        elif method == 'forest_fire':
            subgraph_nodes = sample_subgraph_forest_fire(G, target_size)
        elif method == 'dense_core':
            subgraph_nodes = sample_subgraph_dense_core(G, target_size, min_avg_degree=min_avg_degree)
        elif method == 'mixed':
            # Balance sampling methods - reduce dense_core weight since source graphs may already be dense
            # Weight: forest_fire (40%), random_walk (40%), dense_core (20%), bfs (0%)
            samplers = [
                (sample_subgraph_forest_fire, 0.4),
                (sample_subgraph_random_walk, 0.4),
                (sample_subgraph_dense_core, 0.2),
            ]
            r = random.random()
            cumulative = 0
            for sampler, weight in samplers:
                cumulative += weight
                if r < cumulative:
                    if sampler == sample_subgraph_dense_core:
                        subgraph_nodes = sampler(G, target_size, min_avg_degree=min_avg_degree)
                    else:
                        subgraph_nodes = sampler(G, target_size)
                    break
        else:
            raise ValueError(f"Unknown sampling method: {method}")
        
        # Create subgraph and check density
        subG = G.subgraph(subgraph_nodes)
        num_edges = subG.number_of_edges()
        avg_degree = 2.0 * num_edges / target_size if target_size > 0 else 0.0
        
        # Track best attempt (closest to target range)
        distance_from_target = 0
        if avg_degree < min_avg_degree:
            distance_from_target = min_avg_degree - avg_degree
        elif avg_degree > max_avg_degree:
            distance_from_target = avg_degree - max_avg_degree
        
        if distance_from_target < best_distance_from_target:
            best_distance_from_target = distance_from_target
            best_subgraph = subgraph_nodes
            best_avg_degree = avg_degree
        
        # Accept if within range
        if min_avg_degree <= avg_degree <= max_avg_degree:
            break
    
    # Use best attempt if we couldn't find one in range
    if best_subgraph is not None:
        subgraph_nodes = best_subgraph
        subG = G.subgraph(subgraph_nodes)
    
    # Create mapping from original node IDs to subgraph IDs
    node_mapping = {orig_id: new_id for new_id, orig_id in enumerate(subgraph_nodes)}
    
    # Extract edges
    edges = [(node_mapping[u], node_mapping[v]) for u, v in subG.edges()]

    # Filter: reject if any node is isolated (degree 0)
    connected_nodes = set()
    for u, v in edges:
        connected_nodes.add(u)
        connected_nodes.add(v)
    if len(connected_nodes) < target_size:
        raise RuntimeError(
            f"Subgraph has {target_size - len(connected_nodes)} isolated node(s); retrying"
        )

    # Get labels for subgraph nodes and remap to 3 classes
    subgraph_labels = labels[subgraph_nodes]
    # Remap to 3 classes using modulo - maintains some structure while reducing classes
    subgraph_labels = subgraph_labels % 3
    
    # Sample node features from the pool
    feature_seed = seed if seed is not None else random.randint(0, 2**31)
    node_features = feature_pool.sample(target_size, seed=feature_seed)

    # Create PyG Data
    edge_array = np.array(list(edges), dtype=np.int64)
    edge_index = torch.from_numpy(edge_array.T.copy()).long() if len(edges) > 0 else torch.zeros((2, 0), dtype=torch.long)
    
    data = Data(
        x=node_features,
        edge_index=edge_index,
        y=torch.from_numpy(subgraph_labels).long()
    )
    data.num_nodes = target_size
    
    # Compute metadata
    num_edges = len(edges)
    avg_degree = 2.0 * num_edges / target_size if target_size > 0 else 0.0
    
    # Compute homophily
    if num_edges > 0:
        same_label = sum(1 for u, v in edges if subgraph_labels[u] == subgraph_labels[v])
        homophily = same_label / num_edges
    else:
        homophily = 0.0
    
    metadata = {
        'seed': 'real_world_sample',
        'sampling_method': method,
        'target_label_hom': homophily,
        'realised_label_hom': homophily,
        'avg_degree': avg_degree,
        'num_edges': num_edges,
        'num_classes': int(subgraph_labels.max()) + 1,
    }
    
    # Compute graph statistics
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            stats_array = gen_stats(subG)
        stats_array = np.asarray(stats_array).ravel().astype(float)
        target_len = 15
        if stats_array.size < target_len:
            stats_array = np.pad(stats_array, (0, target_len - stats_array.size), constant_values=0.0)
        elif stats_array.size > target_len:
            stats_array = stats_array[:target_len]
        metadata['stats'] = stats_array
    except Exception:
        metadata['stats'] = None
    
    return data, metadata


def generate_real_world_dataset(
    dataset_names: List[str],
    node_size: int,
    num_graphs: int,
    planetoid_root: str,
    feature_pool: NodeFeaturePool,
    sampling_method: str = 'mixed',
    feature_dim: int = 100,
    min_avg_degree: float = None,
    max_avg_degree: float = None,
    checkpoint_path: str = None,
    split_name: str = "graphs",
) -> List[Tuple[Data, Dict[str, float]]]:
    """Generate dataset by sampling from real-world graphs.
    
    Args:
        dataset_names: List of Planetoid dataset names (e.g., ['Cora', 'CiteSeer', 'PubMed'])
        node_size: Target size for subgraphs
        num_graphs: Total number of graphs to generate
        planetoid_root: Root directory for Planetoid datasets
        feature_pool: NodeFeaturePool to sample features from
        sampling_method: 'random_walk', 'bfs', 'forest_fire', 'dense_core', or 'mixed'
        feature_dim: Feature dimension
        min_avg_degree: Minimum average degree for sampled subgraphs (None for adaptive)
        max_avg_degree: Maximum average degree for sampled subgraphs (None for adaptive)
        checkpoint_path: Optional path for checkpoint saving/loading
        split_name: Name of the split for logging (e.g., "S1", "S2", "test")
        
    Returns:
        List of (Data, metadata) tuples
    """
    # Use adaptive constraints if not explicitly provided
    if min_avg_degree is None or max_avg_degree is None:
        adaptive_min, adaptive_max = compute_adaptive_degree_constraints(node_size)
        if min_avg_degree is None:
            min_avg_degree = adaptive_min
        if max_avg_degree is None:
            max_avg_degree = adaptive_max
    
    print(f"\n=== Generating {num_graphs} real-world subgraphs (n={node_size}) ===")
    print(f"Source datasets: {dataset_names}")
    print(f"Sampling method: {sampling_method}")
    print(f"Avg degree range: [{min_avg_degree:.2f}, {max_avg_degree:.2f}]")
    print(f"  (Max possible avg degree for n={node_size}: {node_size - 1})")
    
    # Load checkpoint if available
    graphs = []
    if checkpoint_path:
        graphs = load_checkpoint(checkpoint_path, split_name)
    
    if len(graphs) >= num_graphs:
        print(f"  ✓ {split_name} generation complete ({len(graphs)} graphs)")
        return graphs[:num_graphs]
    
    # Load all source graphs
    source_graphs = []
    for name in dataset_names:
        print(f"Loading {name}...")
        G, labels, metadata = load_full_graph(name, planetoid_root)
        print(f"  {name}: {metadata['num_nodes']} nodes, {metadata['num_edges']} edges, "
              f"{metadata['num_classes']} classes, avg_degree={metadata['avg_degree']:.2f}")
        source_graphs.append((name, G, labels, metadata))
    
    # Generate subgraphs
    failed_attempts = 0
    max_failures = num_graphs * 5  # Allow some failures
    start_idx = len(graphs)
    
    while len(graphs) < num_graphs and failed_attempts < max_failures:
        # Randomly select a source dataset
        dataset_name, G, labels, metadata = random.choice(source_graphs)
        
        try:
            data, meta = sample_real_world_subgraph(
                G=G,
                labels=labels,
                target_size=node_size,
                feature_pool=feature_pool,
                method=sampling_method,
                feature_dim=feature_dim,
                min_avg_degree=min_avg_degree,
                max_avg_degree=max_avg_degree,
                seed=len(graphs),
            )
            
            # Add source dataset name to metadata
            meta['source_dataset'] = dataset_name
            
            graphs.append((data, meta))
            
            if len(graphs) % 100 == 0:
                print(f"  Generated {len(graphs)} / {num_graphs} subgraphs")
                if checkpoint_path:
                    save_checkpoint(checkpoint_path, graphs, split_name)
                
        except Exception as e:
            failed_attempts += 1
            if failed_attempts % 50 == 0:
                print(f"  Warning: {failed_attempts} failed attempts ({e})")
            continue
    
    # Final checkpoint save
    if checkpoint_path and len(graphs) > start_idx:
        save_checkpoint(checkpoint_path, graphs, split_name)
    
    if len(graphs) < num_graphs:
        print(f"Warning: Only generated {len(graphs)} / {num_graphs} graphs after {failed_attempts} failures")
    
    # Report statistics
    print(f"\n=== Generated {len(graphs)} subgraphs ===")
    homs = [meta['realised_label_hom'] for _, meta in graphs]
    degrees = [meta['avg_degree'] for _, meta in graphs]
    sources = [meta['source_dataset'] for _, meta in graphs]
    
    print(f"Homophily: mean={np.mean(homs):.4f}, std={np.std(homs):.4f}, range=[{np.min(homs):.4f}, {np.max(homs):.4f}]")
    print(f"Avg degree: mean={np.mean(degrees):.4f}, std={np.std(degrees):.4f}, range=[{np.min(degrees):.4f}, {np.max(degrees):.4f}]")
    
    for dataset_name in dataset_names:
        count = sources.count(dataset_name)
        print(f"{dataset_name}: {count} graphs ({100*count/len(graphs):.1f}%)")
    
    print("=" * 50)
    
    return graphs


def adjust_pair_probs(seed: SeedStats, target_hom: float, jitter: float) -> np.ndarray:
    probs = seed.pair_probs.copy()
    diag_mask = np.eye(seed.num_classes, dtype=bool)
    diag_sum = probs[diag_mask].sum()
    off_sum = probs[~diag_mask].sum()

    if target_hom <= 0.0:
        probs[diag_mask] = 1e-8
        off_value = 1.0 / (seed.num_classes * (seed.num_classes - 1))
        probs[~diag_mask] = off_value
    elif target_hom >= 1.0:
        probs[diag_mask] = 1.0 / seed.num_classes
        probs[~diag_mask] = 1e-8
    else:
        if diag_sum > 0:
            probs[diag_mask] *= target_hom / diag_sum
        else:
            probs[diag_mask] = target_hom / seed.num_classes
        if off_sum > 0:
            probs[~diag_mask] *= (1.0 - target_hom) / off_sum
        else:
            fill = (1.0 - target_hom) / max(seed.num_classes * (seed.num_classes - 1), 1)
            probs[~diag_mask] = fill

    probs = np.clip(probs, 1e-8, None)
    probs = 0.5 * (probs + probs.T)
    probs /= probs.sum()

    if jitter > 0:
        noise = np.random.dirichlet(np.ones(probs.size)).reshape(probs.shape)
        probs = (1.0 - jitter) * probs + jitter * noise
        probs = 0.5 * (probs + probs.T)
        probs = np.clip(probs, 1e-8, None)
        probs /= probs.sum()

    return probs


def sample_labels(class_probs: np.ndarray, n: int) -> np.ndarray:
    labels = np.random.choice(len(class_probs), size=n, p=class_probs)
    return labels


def sample_edge_budget(avg_degree: float, n: int, variance_scale: float = None) -> int:
    """Sample the number of edges for a graph.
    
    Args:
        avg_degree: Target average degree
        n: Number of nodes
        variance_scale: If provided, use this as the standard deviation scale.
                        If None, use adaptive scaling (higher for small graphs).
    """
    mean_edges = max(avg_degree, 1.0) * n / 2.0
    
    # For small graphs, use higher relative variance for diversity
    if variance_scale is not None:
        scale = variance_scale
    elif n <= 10:
        # For very small graphs, use 30-50% variance for high diversity
        scale = max(1.0, 0.40 * mean_edges)
    elif n <= 20:
        scale = max(1.0, 0.20 * mean_edges)
    else:
        scale = max(1.0, 0.05 * mean_edges)
    
    noise = np.random.normal(loc=0.0, scale=scale)
    total = max(1, int(round(mean_edges + noise)))
    return total


def allocate_edge_counts(pair_probs: np.ndarray, total_edges: int) -> np.ndarray:
    num_classes = pair_probs.shape[0]
    upper = np.triu(pair_probs)
    upper /= upper.sum()
    flat = upper[np.triu_indices(num_classes)]
    counts = np.random.multinomial(total_edges, flat)
    result = np.zeros_like(pair_probs, dtype=np.int64)
    idx = np.triu_indices(num_classes)
    result[idx] = counts
    result[(idx[1], idx[0])] = counts
    return result


def build_edge_set(labels: np.ndarray, pair_counts: np.ndarray) -> List[Tuple[int, int]]:
    num_classes = pair_counts.shape[0]
    class_nodes = [np.where(labels == c)[0] for c in range(num_classes)]
    edges: set[Tuple[int, int]] = set()

    for i in range(num_classes):
        nodes_i = class_nodes[i]
        if nodes_i.size == 0:
            continue
        for j in range(i, num_classes):
            need = int(pair_counts[i, j])
            if need <= 0:
                continue
            nodes_j = class_nodes[j]
            if nodes_j.size == 0:
                continue
            attempts = 0
            max_attempts = max(need * 5, 50)
            while need > 0 and attempts < max_attempts:
                u = int(np.random.choice(nodes_i))
                v = int(np.random.choice(nodes_j))
                if i == j and u == v:
                    attempts += 1
                    continue
                edge = (u, v) if u < v else (v, u)
                if edge in edges:
                    attempts += 1
                    continue
                edges.add(edge)
                need -= 1
            if need > 0:
                continue

    return list(edges)


def realised_label_hom(edges: List[Tuple[int, int]], labels: np.ndarray) -> float:
    if not edges:
        return 0.0
    same = sum(1 for u, v in edges if labels[u] == labels[v])
    return same / len(edges)


def ensure_connected(edges: List[Tuple[int, int]], n: int, labels: np.ndarray, target_hom: float) -> List[Tuple[int, int]]:
    """Ensure graph is connected by adding minimal edges between components.
    
    Uses Union-Find to identify connected components, then connects them by adding
    edges that respect the target homophily as much as possible.
    """
    # Union-Find data structure
    parent = list(range(n))
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False
    
    # Build initial components from existing edges
    edge_set = set(edges)
    for u, v in edges:
        union(u, v)
    
    # Find all components
    components = {}
    for node in range(n):
        root = find(node)
        if root not in components:
            components[root] = []
        components[root].append(node)
    
    # If already connected, return original edges
    if len(components) == 1:
        return edges
    
    # Connect components with minimal edges
    component_list = list(components.values())
    new_edges = list(edges)
    
    # Connect each component to the first one
    for i in range(1, len(component_list)):
        comp_a = component_list[0]
        comp_b = component_list[i]
        
        # Try to find an edge that respects homophily
        best_edge = None
        best_score = -1.0
        
        for u in comp_a:
            for v in comp_b:
                edge = (u, v) if u < v else (v, u)
                if edge in edge_set:
                    continue
                
                # Score: prefer same-label edges if target_hom > 0.5, else different-label
                same_label = (labels[u] == labels[v])
                if target_hom > 0.5:
                    score = 1.0 if same_label else 0.0
                else:
                    score = 0.0 if same_label else 1.0
                
                if score > best_score or best_edge is None:
                    best_score = score
                    best_edge = edge
        
        if best_edge:
            new_edges.append(best_edge)
            edge_set.add(best_edge)
            # Update union-find
            union(best_edge[0], best_edge[1])
    
    return new_edges


def compute_node_features_with_bfs(edges: List[Tuple[int, int]], n: int, spectral_dim: int = 10, n_max_nodes: int = 100) -> Tuple[torch.Tensor, np.ndarray]:
    """Compute node features using BFS ordering, degree, and Laplacian eigenvectors.
    
    This mirrors the approach in main.py:
    1. Build networkx graph
    2. Apply BFS ordering from highest-degree node in each component
    3. Compute normalized Laplacian eigenvectors
    4. Create features: [normalized_degree, eigenvector_1, ..., eigenvector_k]
    
    Args:
        edges: List of (u, v) edge tuples
        n: Number of nodes
        spectral_dim: Number of Laplacian eigenvectors to include
        n_max_nodes: Maximum nodes for normalization
        
    Returns:
        node_features: [n, spectral_dim+1] tensor
        bfs_node_list: Array mapping new_idx -> original_idx
    """
    # Build networkx graph
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from([(int(u), int(v)) for u, v in edges if u != v])
    
    # Get connected components, sorted by size
    CGs = [G.subgraph(c) for c in nx.connected_components(G)]
    CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
    
    # BFS ordering from highest-degree node in each component
    node_list_bfs = []
    for component in CGs:
        node_degree_list = [(n, d) for n, d in component.degree()]
        degree_sequence = sorted(node_degree_list, key=lambda tt: tt[1], reverse=True)
        bfs_tree = nx.bfs_tree(component, source=degree_sequence[0][0])
        node_list_bfs += list(bfs_tree.nodes())
    
    # Build adjacency matrix in BFS order
    adj_bfs = nx.to_numpy_array(G, nodelist=node_list_bfs)
    adj = torch.from_numpy(adj_bfs).float()
    
    # Compute normalized Laplacian
    diags = np.sum(adj_bfs, axis=0)
    diags = np.squeeze(np.asarray(diags))
    D = sparse.diags(diags).toarray()
    L = D - adj_bfs
    
    with np.errstate(divide="ignore"):
        diags_sqrt = 1.0 / np.sqrt(diags)
    diags_sqrt[np.isinf(diags_sqrt)] = 0
    DH = sparse.diags(diags_sqrt).toarray()
    L = np.linalg.multi_dot((DH, L, DH))
    L = torch.from_numpy(L).float()
    
    # Compute eigenvalues and eigenvectors
    eigval, eigvecs = torch.linalg.eigh(L)
    eigval = torch.real(eigval)
    eigvecs = torch.real(eigvecs)
    idx = torch.argsort(eigval)
    eigvecs = eigvecs[:, idx]
    
    # Build node features: [normalized_degree, eigenvector_1, ..., eigenvector_k]
    x = torch.zeros(n, spectral_dim + 1)
    # Normalized degree (using n_max_nodes for consistency with main.py)
    x[:, 0] = torch.mm(adj, torch.ones(n, 1))[:, 0] / (n_max_nodes - 1)
    # Eigenvectors (take up to spectral_dim)
    mn = min(n, spectral_dim)
    x[:, 1:mn+1] = eigvecs[:, :mn]
    
    return x, np.array(node_list_bfs, dtype=np.int64)


def generate_single_graph(
    templates: Sequence[SeedStats],
    n: int,
    target_hom: float,
    jitter: float,
    feature_pool: NodeFeaturePool = None,
    feature_dim: int = 100,
    seed: int = None,
) -> Tuple[Data, Dict[str, float]]:
    """Generate a single synthetic graph.
    
    Args:
        templates: Sequence of SeedStats for structure generation
        n: Number of nodes
        target_hom: Target homophily
        jitter: Homophily jitter
        feature_pool: NodeFeaturePool to sample features from (required)
        feature_dim: Feature dimension (must match pool's target_dim)
        seed: Random seed for feature sampling
        
    Returns:
        (Data, metadata): Generated graph and metadata
    """
    if feature_pool is None:
        raise ValueError("feature_pool is required - use create_feature_pools() to create one")
    
    # Compute adaptive degree constraints for small graphs
    adaptive_min_degree, adaptive_max_degree = compute_adaptive_degree_constraints(n)
    
    for attempt in range(50):
        template = random.choice(templates)
        labels = sample_labels(template.class_probs, n)
        pair_probs = adjust_pair_probs(template, target_hom, jitter)
        
        # For small graphs, sample target degree uniformly within the adaptive range
        # This creates more diversity than using template.avg_degree directly
        if n <= 10:
            # Sample target average degree uniformly in the adaptive range
            target_avg_degree = np.random.uniform(adaptive_min_degree, adaptive_max_degree)
            total_edges = sample_edge_budget(target_avg_degree, n)
        else:
            # For larger graphs, use template's average degree scaled appropriately
            scaled_avg_degree = min(max(template.avg_degree, adaptive_min_degree), adaptive_max_degree)
            total_edges = sample_edge_budget(scaled_avg_degree, n)
        
        pair_counts = allocate_edge_counts(pair_probs, total_edges)
        edges = build_edge_set(labels, pair_counts)
        if not edges:
            continue

        # Ensure connectivity
        edges = ensure_connected(edges, n, labels, target_hom)

        # Filter: reject graphs with isolated nodes (degree-0 nodes)
        connected_nodes = set()
        for u, v in edges:
            connected_nodes.add(u)
            connected_nodes.add(v)
        if len(connected_nodes) < n:
            continue

        # Sample node features from the pool
        feature_seed = seed if seed is not None else random.randint(0, 2**31)
        node_features = feature_pool.sample(n, seed=feature_seed)

        edge_array = np.array(list(edges), dtype=np.int64)
        edge_index = torch.from_numpy(edge_array.T.copy()).long()
        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=torch.from_numpy(labels).long()
        )
        data.num_nodes = n

        realised_hom = realised_label_hom(edges, labels)
        avg_degree = 2.0 * len(edges) / n if n else 0.0
        meta = {
            "seed": template.name,
            "target_label_hom": target_hom,
            "realised_label_hom": realised_hom,
            "avg_degree": avg_degree,
            "num_edges": len(edges),
        }

        # Compute and attach precomputed stats (using project's gen_stats helper)
        try:
            G = nx.Graph()
            G.add_nodes_from(range(n))
            G.add_edges_from([(int(u), int(v)) for u, v in edges if u != v])
            stats_array = gen_stats(G)
            # Ensure a 1D numpy array and coerce to length 15 (pad/truncate)
            stats_array = np.asarray(stats_array).ravel().astype(float)
            target_len = 15
            if stats_array.size < target_len:
                stats_array = np.pad(stats_array, (0, target_len - stats_array.size), constant_values=0.0)
            elif stats_array.size > target_len:
                stats_array = stats_array[:target_len]
            meta['stats'] = stats_array
        except Exception:
            # If stats computation fails, leave it out and let downstream compute
            meta['stats'] = None

        return data, meta

    raise RuntimeError("Graph sampling failed after multiple retries.")


def compute_pairwise_diversity(graphs: List[Tuple[Data, Dict[str, float]]], metric: str = 'homophily') -> float:
    """Compute diversity within a set of graphs as the std of homophily values.
    
    Args:
        graphs: List of (Data, meta) tuples
        metric: Ignored, always uses homophily std
        
    Returns:
        Diversity score (std of realised homophily, higher = more diverse)
    """
    homs = [meta.get('realised_label_hom', 0.5) for _, meta in graphs]
    return float(np.std(homs))


def compute_cross_diversity(graphs1: List[Tuple[Data, Dict[str, float]]], 
                           graphs2: List[Tuple[Data, Dict[str, float]]], 
                           metric: str = 'homophily') -> float:
    """Compute diversity between two sets of graphs as |mean(hom1) - mean(hom2)|.
    
    Args:
        graphs1, graphs2: Lists of (Data, meta) tuples
        metric: Ignored, always uses homophily difference of means
        
    Returns:
        Cross-diversity score (higher = more diverse)
    """
    homs1 = [meta.get('realised_label_hom', 0.5) for _, meta in graphs1]
    homs2 = [meta.get('realised_label_hom', 0.5) for _, meta in graphs2]
    return abs(float(np.mean(homs1)) - float(np.mean(homs2)))


def save_checkpoint(checkpoint_path: str, graphs: List[Tuple[Data, Dict[str, float]]], split_name: str) -> None:
    """Save intermediate graphs to a checkpoint file.
    
    Args:
        checkpoint_path: Path to save checkpoint
        graphs: List of (Data, meta) tuples to save
        split_name: Name of the split (for logging)
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(graphs, checkpoint_path)
    print(f"  💾 Checkpoint saved: {len(graphs)} {split_name} graphs -> {checkpoint_path}")


def load_checkpoint(checkpoint_path: str, split_name: str) -> List[Tuple[Data, Dict[str, float]]]:
    """Load graphs from a checkpoint file.
    
    Args:
        checkpoint_path: Path to checkpoint file
        split_name: Name of the split (for logging)
        
    Returns:
        List of (Data, meta) tuples, or empty list if checkpoint doesn't exist
    """
    if os.path.exists(checkpoint_path):
        graphs = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"  ✓ Resumed from checkpoint: {len(graphs)} {split_name} graphs loaded from {checkpoint_path}")
        return graphs
    return []


def generate_dataset_splits(
    templates: Sequence[SeedStats],
    node_size: int,
    total_graphs: int,
    train_per_set: int,
    test_count: int,
    target_hom: float,
    jitter: float,
    feature_pool_s1: NodeFeaturePool = None,
    feature_pool_s2: NodeFeaturePool = None,
    feature_pool_test: NodeFeaturePool = None,
    feature_dim: int = 100,
    output_dir: str = None,
) -> Dict[str, List[Tuple[Data, Dict[str, float]]]]:
    """Generate dataset splits with different feature pools for S1 and S2.
    
    Args:
        templates: Sequence of SeedStats for structure generation
        node_size: Number of nodes per graph
        total_graphs: Total graphs to generate (must be >= 2*train_per_set + test_count)
        train_per_set: Number of training graphs per set (S1 and S2)
        test_count: Number of test graphs
        target_hom: Target homophily
        jitter: Homophily jitter
        feature_pool_s1: Feature pool for S1 (Cora + Chameleon)
        feature_pool_s2: Feature pool for S2 (Citeseer + Squirrel)
        feature_pool_test: Feature pool for test set (defaults to pool_s1)
        feature_dim: Feature dimension
        
    Returns:
        Dict with 'S1', 'S2', 'test', 'extra' keys
    """
    if feature_pool_s1 is None or feature_pool_s2 is None:
        raise ValueError("feature_pool_s1 and feature_pool_s2 are required")
    
    if feature_pool_test is None:
        feature_pool_test = feature_pool_s1  # Default: test uses S1's feature distribution
    
    required = 2 * train_per_set + test_count
    if total_graphs < required:
        raise ValueError(f"Need at least {required} graphs, received {total_graphs}")

    # Setup checkpoint paths
    if output_dir:
        checkpoint_dir = os.path.join(output_dir, f"node_{node_size}", "checkpoints")
        s1_checkpoint = os.path.join(checkpoint_dir, "S1_checkpoint.pt")
        s2_checkpoint = os.path.join(checkpoint_dir, "S2_checkpoint.pt")
        test_checkpoint = os.path.join(checkpoint_dir, "test_checkpoint.pt")
    else:
        s1_checkpoint = s2_checkpoint = test_checkpoint = None

    # Generate S1 graphs with S1's feature pool (with checkpoint support)
    s1_graphs: List[Tuple[Data, Dict[str, float]]] = []
    if s1_checkpoint:
        s1_graphs = load_checkpoint(s1_checkpoint, "S1")
    
    if len(s1_graphs) < train_per_set:
        print(f"  Generating S1 graphs with Cora+Chameleon features...")
        start_idx = len(s1_graphs)
        for i in range(start_idx, train_per_set):
            graph, meta = generate_single_graph(
                templates=templates,
                n=node_size,
                target_hom=target_hom,
                jitter=jitter,
                feature_pool=feature_pool_s1,
                feature_dim=feature_dim,
                seed=i * 2,  # Deterministic seeds
            )
            s1_graphs.append((graph, meta))
            if (i + 1) % 100 == 0:
                print(f"    S1: {i + 1} / {train_per_set}")
                if s1_checkpoint:
                    save_checkpoint(s1_checkpoint, s1_graphs, "S1")
        # Final save
        if s1_checkpoint:
            save_checkpoint(s1_checkpoint, s1_graphs, "S1")
    else:
        print(f"  ✓ S1 generation complete ({len(s1_graphs)} graphs)")

    # Generate S2 graphs with S2's feature pool (with checkpoint support)
    s2_graphs: List[Tuple[Data, Dict[str, float]]] = []
    if s2_checkpoint:
        s2_graphs = load_checkpoint(s2_checkpoint, "S2")
    
    if len(s2_graphs) < train_per_set:
        print(f"  Generating S2 graphs with Citeseer+Squirrel features...")
        start_idx = len(s2_graphs)
        for i in range(start_idx, train_per_set):
            graph, meta = generate_single_graph(
                templates=templates,
                n=node_size,
                target_hom=target_hom,
                jitter=jitter,
                feature_pool=feature_pool_s2,
                feature_dim=feature_dim,
                seed=i * 2 + 1,  # Different seeds from S1
            )
            s2_graphs.append((graph, meta))
            if (i + 1) % 100 == 0:
                print(f"    S2: {i + 1} / {train_per_set}")
                if s2_checkpoint:
                    save_checkpoint(s2_checkpoint, s2_graphs, "S2")
        # Final save
        if s2_checkpoint:
            save_checkpoint(s2_checkpoint, s2_graphs, "S2")
    else:
        print(f"  ✓ S2 generation complete ({len(s2_graphs)} graphs)")

    # Generate test graphs (using specified pool, with checkpoint support)
    test_graphs: List[Tuple[Data, Dict[str, float]]] = []
    if test_checkpoint:
        test_graphs = load_checkpoint(test_checkpoint, "test")
    
    if len(test_graphs) < test_count:
        print(f"  Generating test graphs...")
        start_idx = len(test_graphs)
        for i in range(start_idx, test_count):
            graph, meta = generate_single_graph(
                templates=templates,
                n=node_size,
                target_hom=target_hom,
                jitter=jitter,
                feature_pool=feature_pool_test,
                feature_dim=feature_dim,
                seed=10000 + i,  # Different seed range for test
            )
            test_graphs.append((graph, meta))
        if test_checkpoint:
            save_checkpoint(test_checkpoint, test_graphs, "test")
    else:
        print(f"  ✓ Test generation complete ({len(test_graphs)} graphs)")

    return {"S1": s1_graphs, "S2": s2_graphs, "test": test_graphs, "extra": []}


def stratified_split(
    graphs: List[Tuple[Data, Dict[str, float]]],
    train_per_set: int,
    n_bins: int = 5,
    features_for_stratification: List[str] = None
) -> Tuple[List[Tuple[Data, Dict[str, float]]], List[Tuple[Data, Dict[str, float]]]]:
    """
    Split graphs into two balanced subsets using stratified sampling.
    
    Ensures S1 and S2 have similar distributions of key graph properties
    (homophily, degree, etc.) but are non-overlapping.
    
    Args:
        graphs: List of (Data, meta) tuples
        train_per_set: Number of graphs per split
        n_bins: Number of bins for stratification
        features_for_stratification: List of meta keys to use for stratification
                                     (default: ['realised_label_hom', 'avg_degree'])
    
    Returns:
        (s1_graphs, s2_graphs): Two non-overlapping subsets with similar distributions
    """
    if features_for_stratification is None:
        features_for_stratification = ['realised_label_hom', 'avg_degree']
    
    # Check we have enough graphs
    required = 2 * train_per_set
    if len(graphs) < required:
        raise ValueError(f"Need at least {required} graphs for stratified split, got {len(graphs)}")
    
    # Extract feature vectors for stratification
    features = []
    for _, meta in graphs:
        feat_vec = []
        for key in features_for_stratification:
            feat_vec.append(meta.get(key, 0.0))
        features.append(feat_vec)
    features = np.array(features)
    
    # Normalize features to [0, 1] for equal weighting
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features_normalized = scaler.fit_transform(features)
    
    # Assign graphs to bins using k-means clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=n_bins, random_state=42, n_init=10)
    bin_assignments = kmeans.fit_predict(features_normalized)
    
    # For each bin, split graphs alternately into S1 and S2
    s1_indices = []
    s2_indices = []
    
    for bin_id in range(n_bins):
        bin_mask = bin_assignments == bin_id
        bin_indices = np.where(bin_mask)[0]
        
        # Shuffle within bin for randomness
        np.random.shuffle(bin_indices)
        
        # Alternate assignment to S1 and S2
        for i, idx in enumerate(bin_indices):
            if i % 2 == 0:
                s1_indices.append(idx)
            else:
                s2_indices.append(idx)
    
    # Shuffle to avoid any ordering bias
    np.random.shuffle(s1_indices)
    np.random.shuffle(s2_indices)
    
    # Take exactly train_per_set from each
    s1_indices = s1_indices[:train_per_set]
    s2_indices = s2_indices[:train_per_set]
    
    s1_graphs = [graphs[i] for i in s1_indices]
    s2_graphs = [graphs[i] for i in s2_indices]
    
    # Report distribution statistics
    print("\n=== Stratified Split Statistics ===")
    for key in features_for_stratification:
        s1_vals = [meta.get(key, 0.0) for _, meta in s1_graphs]
        s2_vals = [meta.get(key, 0.0) for _, meta in s2_graphs]
        print(f"\n{key}:")
        print(f"  S1: mean={np.mean(s1_vals):.4f}, std={np.std(s1_vals):.4f}, range=[{np.min(s1_vals):.4f}, {np.max(s1_vals):.4f}]")
        print(f"  S2: mean={np.mean(s2_vals):.4f}, std={np.std(s2_vals):.4f}, range=[{np.min(s2_vals):.4f}, {np.max(s2_vals):.4f}]")
        print(f"  |mean_diff|: {abs(np.mean(s1_vals) - np.mean(s2_vals)):.4f}")
    print("=" * 40)
    
    return s1_graphs, s2_graphs


def generate_mixed_pool(
    templates: Sequence[SeedStats],
    node_size: int,
    pool_size: int,
    hom_range: Tuple[float, float] = (0.1, 0.9),
    jitter_range: Tuple[float, float] = (0.05, 0.15),
    feature_pool: NodeFeaturePool = None,
    feature_dim: int = 100,
    checkpoint_path: str = None,
) -> List[Tuple[Data, Dict[str, float]]]:
    """
    Generate a diverse pool of graphs with mixed homophily levels and statistics.
    
    Args:
        templates: List of seed graph statistics
        node_size: Number of nodes per graph
        pool_size: Total number of graphs to generate
        hom_range: Range of homophily values to sample from (low, high)
        jitter_range: Range of jitter values to sample from (low, high)
        feature_pool: NodeFeaturePool to sample features from
        feature_dim: Feature dimension
        checkpoint_path: Optional path for checkpoint saving
    
    Returns:
        List of (Data, meta) tuples with diverse properties
    """
    if feature_pool is None:
        raise ValueError("feature_pool is required")
    
    print(f"\n=== Generating mixed pool of {pool_size} graphs ===")
    print(f"Node size: {node_size}")
    print(f"Homophily range: [{hom_range[0]:.2f}, {hom_range[1]:.2f}]")
    print(f"Jitter range: [{jitter_range[0]:.2f}, {jitter_range[1]:.2f}]")
    
    # Load checkpoint if available
    graphs = []
    if checkpoint_path:
        graphs = load_checkpoint(checkpoint_path, "mixed_pool")
    
    # Generate with diverse parameters
    if len(graphs) < pool_size:
        start_idx = len(graphs)
        for i in range(start_idx, pool_size):
            # Sample homophily uniformly from range
            target_hom = random.uniform(hom_range[0], hom_range[1])
            
            # Sample jitter uniformly from range
            jitter = random.uniform(jitter_range[0], jitter_range[1])
            
            # Randomly select template
            template = random.choice(templates)
            
            try:
                graph, meta = generate_single_graph(
                    templates=[template],
                    n=node_size,
                    target_hom=target_hom,
                    jitter=jitter,
                    feature_pool=feature_pool,
                    feature_dim=feature_dim,
                    seed=i,  # Use loop index for deterministic seeds
                )
                graphs.append((graph, meta))
                
                if (i + 1) % 100 == 0:
                    print(f"  Generated {i + 1} / {pool_size} graphs")
                    if checkpoint_path:
                        save_checkpoint(checkpoint_path, graphs, "mixed_pool")
            except Exception as e:
                print(f"  Warning: Graph generation failed, retrying... ({e})")
                continue
        
        # Final checkpoint save
        if checkpoint_path:
            save_checkpoint(checkpoint_path, graphs, "mixed_pool")
    else:
        print(f"  ✓ Mixed pool generation complete ({len(graphs)} graphs)")


def generate_dataset_splits_stratified(
    templates: Sequence[SeedStats],
    node_size: int,
    train_per_set: int,
    test_count: int,
    hom_range: Tuple[float, float] = (0.1, 0.9),
    jitter_range: Tuple[float, float] = (0.05, 0.15),
    feature_pool_s1: NodeFeaturePool = None,
    feature_pool_s2: NodeFeaturePool = None,
    feature_pool_test: NodeFeaturePool = None,
    feature_dim: int = 100,
    n_stratification_bins: int = 5,
    pool_size_multiplier: float = 1.1,
    output_dir: str = None,
) -> Dict[str, List[Tuple[Data, Dict[str, float]]]]:
    """
    Generate dataset splits using stratified sampling approach.
    
    NOTE: This function generates S1 and S2 with different feature pools
    but similar structure distributions through stratified sampling.
    
    1. Generate a large pool of diverse graphs for S1 (with S1's feature pool)
    2. Generate a large pool of diverse graphs for S2 (with S2's feature pool)
    3. Use stratified sampling to ensure similar structural distributions
    4. Generate separate test set
    
    Args:
        templates: List of seed graph statistics
        node_size: Number of nodes per graph
        train_per_set: Number of graphs per training split (S1/S2)
        test_count: Number of test graphs
        hom_range: Range of homophily values (low, high)
        jitter_range: Range of jitter values (low, high)
        feature_pool_s1: Feature pool for S1 (Cora + Chameleon)
        feature_pool_s2: Feature pool for S2 (Citeseer + Squirrel)
        feature_pool_test: Feature pool for test set (defaults to S1's pool)
        feature_dim: Feature dimension
        n_stratification_bins: Number of bins for stratified sampling
        pool_size_multiplier: Generate pool_size = train_per_set * multiplier
    
    Returns:
        Dictionary with splits: {'S1': [...], 'S2': [...], 'test': [...]}
    """
    if feature_pool_s1 is None or feature_pool_s2 is None:
        raise ValueError("feature_pool_s1 and feature_pool_s2 are required")
    
    if feature_pool_test is None:
        feature_pool_test = feature_pool_s1
    
    print("\n" + "="*70)
    print("STRATIFIED SPLIT GENERATION MODE")
    print("="*70)
    
    # Generate pool for S1
    pool_size = int(train_per_set * pool_size_multiplier)
    
    # Setup checkpoint paths
    if output_dir:
        checkpoint_dir = os.path.join(output_dir, f"node_{node_size}", "checkpoints")
        s1_pool_checkpoint = os.path.join(checkpoint_dir, "S1_pool_checkpoint.pt")
        s2_pool_checkpoint = os.path.join(checkpoint_dir, "S2_pool_checkpoint.pt")
        test_checkpoint = os.path.join(checkpoint_dir, "test_checkpoint.pt")
    else:
        s1_pool_checkpoint = s2_pool_checkpoint = test_checkpoint = None
    
    print(f"\nStep 1a: Generating pool of {pool_size} graphs for S1 (Cora+Chameleon features)...")
    s1_pool = generate_mixed_pool(
        templates=templates,
        node_size=node_size,
        pool_size=pool_size,
        hom_range=hom_range,
        jitter_range=jitter_range,
        feature_pool=feature_pool_s1,
        feature_dim=feature_dim,
        checkpoint_path=s1_pool_checkpoint,
    )
    
    print(f"\nStep 1b: Generating pool of {pool_size} graphs for S2 (Citeseer+Squirrel features)...")
    s2_pool = generate_mixed_pool(
        templates=templates,
        node_size=node_size,
        pool_size=pool_size,
        hom_range=hom_range,
        jitter_range=jitter_range,
        feature_pool=feature_pool_s2,
        feature_dim=feature_dim,
        checkpoint_path=s2_pool_checkpoint,
    )
    
    # Shuffle and take train_per_set from each pool
    random.shuffle(s1_pool)
    random.shuffle(s2_pool)
    s1_graphs = s1_pool[:train_per_set]
    s2_graphs = s2_pool[:train_per_set]
    
    # Generate test set separately
    print(f"\nStep 2: Generating test set ({test_count} graphs)...")
    test_graphs = generate_mixed_pool(
        templates=templates,
        node_size=node_size,
        pool_size=test_count,
        hom_range=hom_range,
        jitter_range=jitter_range,
        feature_pool=feature_pool_test,
        feature_dim=feature_dim,
        checkpoint_path=test_checkpoint,
    )
    
    # Compute cross-similarity between S1 and S2
    print("\nStep 3: Validating split quality...")
    sample_size = min(100, len(s1_graphs), len(s2_graphs))
    s1_sample = random.sample(s1_graphs, sample_size)
    s2_sample = random.sample(s2_graphs, sample_size)
    
    # Homophily-based similarity
    s1_homs = [meta['realised_label_hom'] for _, meta in s1_sample]
    s2_homs = [meta['realised_label_hom'] for _, meta in s2_sample]
    
    print(f"\nDistribution comparison (sampled {sample_size} graphs each):")
    print(f"S1 homophily: mean={np.mean(s1_homs):.4f} ± {np.std(s1_homs):.4f}")
    print(f"S2 homophily: mean={np.mean(s2_homs):.4f} ± {np.std(s2_homs):.4f}")
    print(f"Mean difference: {abs(np.mean(s1_homs) - np.mean(s2_homs)):.4f}")
    
    if abs(np.mean(s1_homs) - np.mean(s2_homs)) < 0.05:
        print("✓ S1 and S2 have similar structural distributions!")
    else:
        print("⚠ S1 and S2 distributions differ slightly")
    
    print("\n" + "="*70)
    
    return {"S1": s1_graphs, "S2": s2_graphs, "test": test_graphs}


def generate_dataset_splits_diverse(
    templates: Sequence[SeedStats],
    node_size: int,
    train_per_set: int,
    test_count: int,
    s1_config: Dict,
    s2_config: Dict,
    test_config: Dict,
    feature_pool_s1: NodeFeaturePool = None,
    feature_pool_s2: NodeFeaturePool = None,
    feature_pool_test: NodeFeaturePool = None,
    feature_dim: int = 100,
    validate_diversity: bool = True,
    diversity_margin: float = 0.15,
    validation_sample_size: int = 100,
    generate_s1: bool = True,
    generate_s2: bool = True,
    output_dir: str = None,
) -> Dict[str, List[Tuple[Data, Dict[str, float]]]]:
    """Generate dataset splits with DIVERSE S1 and S2 configurations.
    
    Args:
        templates: List of seed graph statistics (e.g., Cora, CiteSeer)
        node_size: Number of nodes per graph
        train_per_set: Number of graphs per training split (S1/S2)
        test_count: Number of test graphs
        s1_config: Config for S1 split, e.g. {'target_hom_range': [0.65, 0.95], 'jitter': 0.1, 'templates': [0]}
        s2_config: Config for S2 split, e.g. {'target_hom_range': [0.15, 0.45], 'jitter': 0.1, 'templates': [1]}
        test_config: Config for test split, e.g. {'target_hom_range': [0.4, 0.6], 'jitter': 0.1, 'templates': None}
        feature_pool_s1: Feature pool for S1 (Cora + Chameleon)
        feature_pool_s2: Feature pool for S2 (Citeseer + Squirrel)
        feature_pool_test: Feature pool for test set (defaults to pool_s1)
        feature_dim: Feature dimension
        validate_diversity: Whether to validate diversity constraints during generation
        diversity_margin: Minimum required gap: cross_diversity > within_diversity + margin
        validation_sample_size: Sample size for diversity validation
        
    Returns:
        Dictionary with splits: {'S1': [...], 'S2': [...], 'test': [...]}
        
    Config format:
        - target_hom_range: [low, high] homophily range (will sample uniformly)
        - jitter: Dirichlet noise level for pair probabilities
        - templates: List of template indices to use (None = all templates)
    """
    if feature_pool_s1 is None or feature_pool_s2 is None:
        raise ValueError("feature_pool_s1 and feature_pool_s2 are required")
    
    if feature_pool_test is None:
        feature_pool_test = feature_pool_s1
    
    splits_to_generate = []
    if generate_s1:
        splits_to_generate.append("S1")
    if generate_s2:
        splits_to_generate.append("S2")
    
    print(f"\n=== Generating DIVERSE splits for n={node_size} ===")
    print(f"Splits to generate: {', '.join(splits_to_generate) if splits_to_generate else 'test only'}")
    if generate_s1:
        print(f"S1 config: {s1_config}")
    if generate_s2:
        print(f"S2 config: {s2_config}")
    print(f"Test config: {test_config}")
    print(f"Feature dimension: {feature_dim}")
    print(f"Validation: {'ENABLED' if validate_diversity else 'DISABLED'} (margin={diversity_margin})")
    
    # Setup checkpoint paths
    if output_dir:
        checkpoint_dir = os.path.join(output_dir, f"node_{node_size}", "checkpoints")
        s1_checkpoint = os.path.join(checkpoint_dir, "S1_checkpoint.pt")
        s2_checkpoint = os.path.join(checkpoint_dir, "S2_checkpoint.pt")
        test_checkpoint = os.path.join(checkpoint_dir, "test_checkpoint.pt")
    else:
        s1_checkpoint = s2_checkpoint = test_checkpoint = None
    
    def generate_split(config: Dict, count: int, split_name: str, feature_pool: NodeFeaturePool, checkpoint_path: str = None) -> List[Tuple[Data, Dict[str, float]]]:
        """Generate graphs for a single split with given configuration and checkpoint support."""
        hom_range = config.get('target_hom_range', [0.4, 0.6])
        jitter = config.get('jitter', 0.1)
        template_indices = config.get('templates', None)
        
        # Filter templates if specified
        if template_indices is not None:
            split_templates = [templates[i] for i in template_indices if i < len(templates)]
        else:
            split_templates = list(templates)
        
        if not split_templates:
            raise ValueError(f"No valid templates for {split_name}")
        
        # Load checkpoint if available
        graphs = []
        if checkpoint_path:
            graphs = load_checkpoint(checkpoint_path, split_name)
        
        if len(graphs) < count:
            start_idx = len(graphs)
            while len(graphs) < count:
                # Sample homophily uniformly from range
                target_hom = random.uniform(hom_range[0], hom_range[1])
                
                graph, meta = generate_single_graph(
                    templates=split_templates,
                    n=node_size,
                    target_hom=target_hom,
                    jitter=jitter,
                    feature_pool=feature_pool,
                    feature_dim=feature_dim,
                    seed=len(graphs),  # Deterministic seed based on graph index
                )
                graphs.append((graph, meta))
                
                if len(graphs) % 100 == 0:
                    print(f"  {split_name}: generated {len(graphs)} / {count} graphs")
                    if checkpoint_path:
                        save_checkpoint(checkpoint_path, graphs, split_name)
            
            # Final save
            if checkpoint_path:
                save_checkpoint(checkpoint_path, graphs, split_name)
        else:
            print(f"  ✓ {split_name} generation complete ({len(graphs)} graphs)")
        
        return graphs
    
    # Generate S1 (if requested)
    s1_graphs = []
    if generate_s1:
        print(f"\nGenerating S1 ({train_per_set} graphs) with Cora+Chameleon features...")
        s1_graphs = generate_split(s1_config, train_per_set, "S1", feature_pool_s1, s1_checkpoint)
    else:
        print(f"\nSkipping S1 generation")
    
    # Generate S2 (if requested)
    s2_graphs = []
    if generate_s2:
        print(f"\nGenerating S2 ({train_per_set} graphs) with Citeseer+Squirrel features...")
        s2_graphs = generate_split(s2_config, train_per_set, "S2", feature_pool_s2, s2_checkpoint)
    else:
        print(f"\nSkipping S2 generation")
    
    # Generate test
    print(f"\nGenerating test ({test_count} graphs)...")
    test_graphs = generate_split(test_config, test_count, "test", feature_pool_test, test_checkpoint)
    
    # Validate diversity constraints (only if both S1 and S2 are generated)
    if validate_diversity and generate_s1 and generate_s2 and s1_graphs and s2_graphs:
        print(f"\n=== Validating diversity constraints ===")
        
        # Sample for validation (full set too expensive)
        val_size = min(validation_sample_size, min(len(s1_graphs), len(s2_graphs)))
        s1_sample = random.sample(s1_graphs, val_size)
        s2_sample = random.sample(s2_graphs, val_size)
        
        # Compute diversity metrics
        print("Computing within-split diversity (homophily std)...")
        s1_within_hom = compute_pairwise_diversity(s1_sample, metric='homophily')
        s2_within_hom = compute_pairwise_diversity(s2_sample, metric='homophily')
        
        print("Computing cross-split diversity (homophily difference)...")
        cross_hom = compute_cross_diversity(s1_sample, s2_sample, metric='homophily')
        
        avg_within_hom = (s1_within_hom + s2_within_hom) / 2.0
        margin_hom = cross_hom - avg_within_hom
        
        print(f"\n--- Homophily Diversity ---")
        print(f"S1 within (std): {s1_within_hom:.4f}")
        print(f"S2 within (std): {s2_within_hom:.4f}")
        print(f"Avg within: {avg_within_hom:.4f}")
        print(f"Cross (|mean diff|): {cross_hom:.4f}")
        print(f"Margin: {margin_hom:.4f} (target: >{diversity_margin:.2f})")
        
        if margin_hom < diversity_margin:
            print(f"⚠️  WARNING: Homophily margin {margin_hom:.4f} < {diversity_margin:.2f}")
            print(f"    Consider adjusting target_hom_range configs")
        else:
            print(f"✓ Homophily diversity constraint satisfied!")
        
        print("\n" + "="*50)
    
    return {"S1": s1_graphs, "S2": s2_graphs, "test": test_graphs}


def save_split(output_dir: str, node_size: int, split: Dict[str, List[Tuple[Data, Dict[str, float]]]]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    base_dir = os.path.join(output_dir, f"node_{node_size}")
    os.makedirs(base_dir, exist_ok=True)

    for split_name, entries in split.items():
        data_path = os.path.join(base_dir, f"{split_name}.pt")
        # Entries are tuples (Data, meta) where meta may contain 'stats' (numpy array)
        torch.save(entries, data_path)

    # Compute basic summary statistics
    summary = {}
    for split_name, entries in split.items():
        homs = [meta["realised_label_hom"] for _, meta in entries]
        degrees = [meta.get("avg_degree", 0) for _, meta in entries]
        sources = [meta.get("source_dataset", "unknown") for _, meta in entries]
        
        # Count source dataset percentages
        source_counts = {}
        for src in sources:
            source_counts[src] = source_counts.get(src, 0) + 1
        source_percentages = {src: 100.0 * count / len(entries) for src, count in source_counts.items()} if entries else {}
        
        summary[split_name] = {
            "count": len(entries),
            "mean_label_hom": float(np.mean(homs)) if homs else None,
            "std_label_hom": float(np.std(homs)) if homs else None,
            "mean_avg_degree": float(np.mean(degrees)) if degrees else None,
            "std_avg_degree": float(np.std(degrees)) if degrees else None,
            "source_counts": source_counts,
            "source_percentages": source_percentages,
        }

    with open(os.path.join(base_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    
    # Also write a detailed log file
    log_path = os.path.join(base_dir, "generation_log.txt")
    with open(log_path, "w") as f:
        f.write("="*60 + "\n")
        f.write("GRAPH GENERATION LOG\n")
        f.write("="*60 + "\n\n")
        
        for split_name in ['S1', 'S2', 'test']:
            if split_name in summary:
                s = summary[split_name]
                f.write(f"{split_name}:\n")
                f.write(f"  Count: {s['count']}\n")
                f.write(f"  Homophily: mean={s['mean_label_hom']:.4f}, std={s['std_label_hom']:.4f}\n")
                f.write(f"  Avg Degree: mean={s.get('mean_avg_degree', 0):.4f}, std={s.get('std_avg_degree', 0):.4f}\n")
                
                # Source dataset percentages
                if 'source_percentages' in s and s['source_percentages']:
                    f.write("  Source Datasets:\n")
                    for src, pct in sorted(s['source_percentages'].items(), key=lambda x: -x[1]):
                        count = s['source_counts'].get(src, 0)
                        f.write(f"    {src}: {count} ({pct:.1f}%)\n")
                f.write("\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"Saved generation log to {log_path}")


def visualize_examples(output_dir: str, node_size: int, n_examples: int = 3, layout: str = 'spring') -> None:
    """Load saved splits and render example graphs from S1 and S2 side by side.

    This writes a PNG image into the split directory called `examples_node_{n}.png`.
    """
    base_dir = os.path.join(output_dir, f"node_{node_size}")
    s1_path = os.path.join(base_dir, 'S1.pt')
    s2_path = os.path.join(base_dir, 'S2.pt')
    
    if not os.path.exists(s1_path):
        print(f"No S1 split found at {s1_path}")
        return
    if not os.path.exists(s2_path):
        print(f"No S2 split found at {s2_path}")
        return

    s1_entries = torch.load(s1_path, map_location='cpu', weights_only=False)
    s2_entries = torch.load(s2_path, map_location='cpu', weights_only=False)
    
    if len(s1_entries) == 0 or len(s2_entries) == 0:
        print("Empty splits, cannot visualize")
        return

    # Sample n_examples from each set
    take = min(n_examples, len(s1_entries), len(s2_entries))
    s1_chosen = random.sample(s1_entries, take)
    s2_chosen = random.sample(s2_entries, take)

    # Create figure: 2 rows (S1, S2) x n_examples columns
    fig, axes = plt.subplots(2, take, figsize=(5 * take, 10))
    if take == 1:
        axes = axes.reshape(2, 1)

    def draw_graph(ax, data, meta, set_name):
        """Draw a single graph on the given axes."""
        G = nx.Graph()
        n = int(data.num_nodes)
        G.add_nodes_from(range(n))
        edges = data.edge_index.t().tolist()
        G.add_edges_from([(int(u), int(v)) for u, v in edges if u != v])

        # Get labels for coloring
        labels = None
        if hasattr(data, 'y') and data.y is not None:
            labels = data.y.cpu().numpy().astype(int).tolist()
        
        if labels is None:
            node_colors = ['#3498db'] * n
        else:
            uniq = sorted(set(labels))
            cmap = plt.get_cmap('Set1')
            color_map = {lab: cmap(i % 9) for i, lab in enumerate(uniq)}
            node_colors = [color_map[int(l)] for l in labels]

        # Layout
        avg_degree = 2 * G.number_of_edges() / n if n > 0 else 0
        try:
            if avg_degree < 4.0:
                pos = nx.kamada_kawai_layout(G, scale=1.0)
            else:
                k_param = max(0.3, 1.5/np.sqrt(n))
                pos = nx.spring_layout(G, k=k_param, iterations=150, seed=42, scale=1.0)
        except:
            pos = nx.circular_layout(G, scale=1.0)

        # Normalize positions
        if pos:
            node_order = list(G.nodes())
            coords = np.array([pos[node] for node in node_order], dtype=float)
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0)
            spans = np.maximum(maxs - mins, 1e-8)
            coords = (coords - mins) / spans
            coords = 2.0 * coords - 1.0
            pos = {node: (float(coords[i, 0]), float(coords[i, 1])) for i, node in enumerate(node_order)}

        # Draw edges - THICK and visible
        edge_width = max(1.0, min(2.5, 80.0 / max(n, 1)))
        nx.draw_networkx_edges(
            G, pos, ax=ax, alpha=0.6, width=edge_width, edge_color='#444444'
        )

        # Draw nodes - LARGE and visible
        node_sz = max(200, min(800, 8000 // max(n, 1)))
        nx.draw_networkx_nodes(
            G, pos, node_color=node_colors, node_size=node_sz, ax=ax,
            edgecolors='black', linewidths=1.5, alpha=0.95
        )

        # Set label showing only the set name
        ax.text(0.5, -0.08, set_name, transform=ax.transAxes, 
                ha='center', va='top', fontsize=20, fontweight='bold')
        ax.set_axis_off()
        ax.set_aspect('equal')
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)

    # Draw S1 graphs (top row)
    for i, (data, meta) in enumerate(s1_chosen):
        draw_graph(axes[0, i], data, meta, 'S1')

    # Draw S2 graphs (bottom row)
    for i, (data, meta) in enumerate(s2_chosen):
        draw_graph(axes[1, i], data, meta, 'S2')

    out_path = os.path.join(base_dir, f"examples_node_{node_size}.png")
    plt.tight_layout(pad=1.0)
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved visualization to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast seeded synthetic graph dataset generator")
    parser.add_argument("--output-dir", type=str, required=True, help="Destination directory for generated splits")
    parser.add_argument("--node-sizes", type=int, nargs="+", default=[100], help="Node counts to generate")
    parser.add_argument("--total-per-size", type=int, default=5100, help="Graphs to sample per node size (legacy mode only)")
    parser.add_argument("--train-per-set", type=int, default=2500, help="Graphs per training subset")
    parser.add_argument("--test-count", type=int, default=100, help="Graphs reserved for evaluation")
    parser.add_argument("--target-hom", type=float, default=0.5, help="Desired label homophily (legacy mode only)")
    parser.add_argument("--pair-jitter", type=float, default=0.1, help="Dirichlet noise level for pair probabilities")
    parser.add_argument("--planetoid-root", type=str, default="./planetoid", help="Planetoid dataset root directory")
    parser.add_argument("--visualize", action="store_true", help="Generate example visualizations after creating splits")
    parser.add_argument("--viz-examples", type=int, default=3, help="Number of example graphs to visualize (default: 3)")
    parser.add_argument("--feature-dim", type=int, default=100, help="Dimension of node features after PCA (default: 100)")
    
    # Diverse split generation mode
    parser.add_argument("--diverse", action="store_true", help="Enable diverse split generation (S1 and S2 have different configs) [LEGACY]")
    parser.add_argument("--stratified", action="store_true", help="Enable stratified split generation (S1 and S2 from mixed pool with similar distributions)")
    
    # Stratified mode parameters
    parser.add_argument("--hom-range", type=float, nargs=2, default=[0.1, 0.9], help="Homophily range [low, high] for mixed pool (stratified mode)")
    parser.add_argument("--jitter-range", type=float, nargs=2, default=[0.05, 0.15], help="Jitter range [low, high] for mixed pool (stratified mode)")
    parser.add_argument("--n-bins", type=int, default=5, help="Number of bins for stratification (stratified mode)")
    parser.add_argument("--pool-multiplier", type=float, default=1.1, help="Pool size multiplier for stratification (stratified mode)")
    
    # Legacy diverse mode parameters
    parser.add_argument("--s1-hom-range", type=float, nargs=2, default=[0.65, 0.95], help="S1 homophily range [low, high] (diverse mode)")
    parser.add_argument("--s2-hom-range", type=float, nargs=2, default=[0.15, 0.45], help="S2 homophily range [low, high] (diverse mode)")
    parser.add_argument("--test-hom-range", type=float, nargs=2, default=[0.4, 0.6], help="Test homophily range [low, high] (diverse mode)")
    parser.add_argument("--s1-templates", type=int, nargs="*", default=None, help="Template indices for S1 (diverse mode, None=all)")
    parser.add_argument("--s2-templates", type=int, nargs="*", default=None, help="Template indices for S2 (diverse mode, None=all)")
    parser.add_argument("--test-templates", type=int, nargs="*", default=None, help="Template indices for test (diverse mode, None=all)")
    parser.add_argument("--validate-diversity", action="store_true", help="Validate diversity constraints during generation (diverse mode)")
    parser.add_argument("--diversity-margin", type=float, default=0.15, help="Minimum cross-within diversity gap (diverse mode)")
    parser.add_argument("--validation-sample-size", type=int, default=100, help="Sample size for diversity validation (diverse mode)")
    
    # Split selection arguments
    parser.add_argument("--generate-s1", action="store_true", help="Generate S1 split (default: generate both)")
    parser.add_argument("--generate-s2", action="store_true", help="Generate S2 split (default: generate both)")
    
    # Real-world sampling mode
    parser.add_argument("--real-world", action="store_true", help="Enable real-world subgraph sampling mode")
    parser.add_argument("--source-datasets", type=str, nargs="+", default=["Cora", "CiteSeer", "PubMed", "Chameleon", "Squirrel"],
                       help="Source datasets for real-world sampling (used for both S1/S2 if separate not specified)")
    parser.add_argument("--s1-source-datasets", type=str, nargs="+", default=None,
                       help="Source datasets for S1 only (e.g., Cora Chameleon). If not set, uses --source-datasets")
    parser.add_argument("--s2-source-datasets", type=str, nargs="+", default=None,
                       help="Source datasets for S2 only (e.g., CiteSeer Squirrel). If not set, uses --source-datasets")
    parser.add_argument("--sampling-method", type=str, default="mixed",
                       choices=["random_walk", "bfs", "forest_fire", "dense_core", "mixed"],
                       help="Subgraph sampling method (default: mixed)")
    parser.add_argument("--min-avg-degree", type=float, default=None,
                       help="Minimum average degree for sampled subgraphs (None for adaptive based on node size, default: None)")
    parser.add_argument("--max-avg-degree", type=float, default=None,
                       help="Maximum average degree for sampled subgraphs (None for adaptive based on node size, default: None)")
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(0)
    np.random.seed(0)

    # Determine which splits to generate
    # If neither flag is set, generate both (default behavior)
    generate_s1 = args.generate_s1 or not (args.generate_s1 or args.generate_s2)
    generate_s2 = args.generate_s2 or not (args.generate_s1 or args.generate_s2)

    if args.real_world:
        # REAL-WORLD SAMPLING MODE: Sample subgraphs from real datasets
        print("\n" + "="*70)
        print("REAL-WORLD SUBGRAPH SAMPLING MODE ENABLED")
        # Determine source datasets for S1 and S2
        s1_sources = args.s1_source_datasets if args.s1_source_datasets else args.source_datasets
        s2_sources = args.s2_source_datasets if args.s2_source_datasets else args.source_datasets
        
        # For test set, combine all unique sources
        all_sources = list(set(s1_sources + s2_sources))
        
        print("="*70)
        print(f"S1 source datasets: {s1_sources}")
        print(f"S2 source datasets: {s2_sources}")
        print(f"Test source datasets: {all_sources}")
        print(f"Sampling method: {args.sampling_method}")
        print("="*70)
        
        # Create feature pools for S1 and S2
        print("\n=== Creating Node Feature Pools ===")
        feature_pool_s1 = NodeFeaturePool(
            dataset_names=s1_sources,
            root=args.planetoid_root,
            target_dim=args.feature_dim,
            seed=42
        )
        feature_pool_s2 = NodeFeaturePool(
            dataset_names=s2_sources,
            root=args.planetoid_root,
            target_dim=args.feature_dim,
            seed=43
        )
        # Test pool uses combined sources
        feature_pool_test = NodeFeaturePool(
            dataset_names=all_sources,
            root=args.planetoid_root,
            target_dim=args.feature_dim,
            seed=44
        )
        
        # Check if using separate source datasets (diverse mode)
        use_separate_sources = (args.s1_source_datasets is not None or args.s2_source_datasets is not None)
        
        for node_size in args.node_sizes:
            print(f"\n{'='*70}")
            print(f"Generating real-world subgraphs for node size {node_size}...")
            print(f"{'='*70}")
            
            if use_separate_sources:
                # SEPARATE SOURCE MODE: Generate S1 and S2 from different datasets
                print(f"\nUsing SEPARATE source datasets for S1 and S2")
                
                # Checkpoint paths for real-world mode
                checkpoint_dir = os.path.join(args.output_dir, f"node_{node_size}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                s1_checkpoint = os.path.join(checkpoint_dir, "checkpoint_realworld_s1.pt")
                s2_checkpoint = os.path.join(checkpoint_dir, "checkpoint_realworld_s2.pt")
                test_checkpoint = os.path.join(checkpoint_dir, "checkpoint_realworld_test.pt")
                
                print(f"\nStep 1a: Generating S1 ({args.train_per_set} graphs from {s1_sources})...")
                s1_graphs = generate_real_world_dataset(
                    dataset_names=s1_sources,
                    node_size=node_size,
                    num_graphs=args.train_per_set,
                    planetoid_root=args.planetoid_root,
                    sampling_method=args.sampling_method,
                    feature_pool=feature_pool_s1,
                    feature_dim=args.feature_dim,
                    min_avg_degree=args.min_avg_degree,
                    max_avg_degree=args.max_avg_degree,
                    checkpoint_path=s1_checkpoint,
                    split_name="S1",
                )
                
                print(f"\nStep 1b: Generating S2 ({args.train_per_set} graphs from {s2_sources})...")
                s2_graphs = generate_real_world_dataset(
                    dataset_names=s2_sources,
                    node_size=node_size,
                    num_graphs=args.train_per_set,
                    planetoid_root=args.planetoid_root,
                    sampling_method=args.sampling_method,
                    feature_pool=feature_pool_s2,
                    feature_dim=args.feature_dim,
                    min_avg_degree=args.min_avg_degree,
                    max_avg_degree=args.max_avg_degree,
                    checkpoint_path=s2_checkpoint,
                    split_name="S2",
                )
            else:
                # SHARED SOURCE MODE: Generate from same pool and stratify
                print(f"\nUsing SHARED source datasets with stratified split")
                
                required_train = 2 * args.train_per_set
                pool_size = int(required_train * 1.1)  # 10% extra for better stratification
                
                # Checkpoint paths for shared source mode
                checkpoint_dir = os.path.join(args.output_dir, f"node_{node_size}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                pool_checkpoint = os.path.join(checkpoint_dir, "checkpoint_realworld_pool.pt")
                test_checkpoint = os.path.join(checkpoint_dir, "checkpoint_realworld_test.pt")
                
                print(f"\nStep 1: Generating pool of {pool_size} graphs for training splits...")
                train_pool = generate_real_world_dataset(
                    dataset_names=args.source_datasets,
                    node_size=node_size,
                    num_graphs=pool_size,
                    planetoid_root=args.planetoid_root,
                    sampling_method=args.sampling_method,
                    feature_pool=feature_pool_s1,  # Use S1 pool for shared source mode
                    feature_dim=args.feature_dim,
                    min_avg_degree=args.min_avg_degree,
                    max_avg_degree=args.max_avg_degree,
                    checkpoint_path=pool_checkpoint,
                    split_name="train_pool",
                )
                
                # Perform stratified split
                print(f"\nStep 2: Performing stratified split into S1 and S2...")
                s1_graphs, s2_graphs = stratified_split(
                    graphs=train_pool,
                    train_per_set=args.train_per_set,
                    n_bins=10,  # More bins for better distribution matching
                    features_for_stratification=['realised_label_hom', 'avg_degree', 'num_edges']
                )
            
            # Generate test set from all sources (checkpoint path is defined in both branches above)
            print(f"\nStep 3: Generating test set ({args.test_count} graphs from {all_sources})...")
            test_graphs = generate_real_world_dataset(
                dataset_names=all_sources,
                node_size=node_size,
                num_graphs=args.test_count,
                planetoid_root=args.planetoid_root,
                sampling_method=args.sampling_method,
                feature_pool=feature_pool_test,
                feature_dim=args.feature_dim,
                min_avg_degree=args.min_avg_degree,
                max_avg_degree=args.max_avg_degree,
                checkpoint_path=test_checkpoint,
                split_name="test",
            )
            
            # Validate split quality
            print("\nStep 4: Validating split quality...")
            s1_src_list = [meta.get('source_dataset', 'unknown') for _, meta in s1_graphs]
            s2_src_list = [meta.get('source_dataset', 'unknown') for _, meta in s2_graphs]
            test_src_list = [meta.get('source_dataset', 'unknown') for _, meta in test_graphs]
            
            print(f"\nSource dataset distribution:")
            for dataset_name in all_sources:
                s1_count = s1_src_list.count(dataset_name)
                s2_count = s2_src_list.count(dataset_name)
                test_count_ds = test_src_list.count(dataset_name)
                print(f"  {dataset_name}:")
                print(f"    S1: {s1_count} ({100*s1_count/len(s1_graphs):.1f}%)")
                print(f"    S2: {s2_count} ({100*s2_count/len(s2_graphs):.1f}%)")
                print(f"    Test: {test_count_ds} ({100*test_count_ds/len(test_graphs):.1f}%)")
            
            # Save splits
            split = {"S1": s1_graphs, "S2": s2_graphs, "test": test_graphs}
            save_split(args.output_dir, node_size, split)
            print(f"\n✓ Finished node size {node_size} -> {args.output_dir}/node_{node_size}")
            
            if args.visualize:
                print(f"Creating visualizations for node size {node_size}...")
                visualize_examples(args.output_dir, node_size, n_examples=args.viz_examples, layout='kamada_kawai')
        
        return

    templates = [
        load_seed_stats("Cora", args.planetoid_root),
        load_seed_stats("CiteSeer", args.planetoid_root),
    ]
    
    # Create feature pools for S1 and S2
    print("\n=== Creating Node Feature Pools ===")
    feature_pool_s1, feature_pool_s2 = create_feature_pools(
        root=args.planetoid_root,
        target_dim=args.feature_dim,
        seed=42
    )

    if args.stratified:
        # STRATIFIED MODE: Generate mixed pool, then split with stratification
        print("\n" + "="*70)
        print("STRATIFIED SPLIT GENERATION MODE ENABLED")
        print("="*70)
        print("S1 and S2 will be created from a mixed pool with similar distributions")
        print("S1 features: Cora + Chameleon")
        print("S2 features: Citeseer + Squirrel")
        print("="*70)
        
        for node_size in args.node_sizes:
            print(f"\n{'='*70}")
            print(f"Generating STRATIFIED splits for node size {node_size}...")
            print(f"{'='*70}")
            
            split = generate_dataset_splits_stratified(
                templates=templates,
                node_size=node_size,
                train_per_set=args.train_per_set,
                test_count=args.test_count,
                hom_range=tuple(args.hom_range),
                jitter_range=tuple(args.jitter_range),
                feature_pool_s1=feature_pool_s1,
                feature_pool_s2=feature_pool_s2,
                feature_pool_test=feature_pool_s1,  # Test uses S1's feature distribution
                feature_dim=args.feature_dim,
                n_stratification_bins=args.n_bins,
                pool_size_multiplier=args.pool_multiplier,
                output_dir=args.output_dir,
            )
            
            save_split(args.output_dir, node_size, split)
            print(f"\n✓ Finished node size {node_size} -> {args.output_dir}/node_{node_size}")
            
            if args.visualize:
                print(f"Creating visualizations for node size {node_size}...")
                visualize_examples(args.output_dir, node_size, n_examples=args.viz_examples, layout='spring')
                
    elif args.diverse:
        # DIVERSE MODE: S1 and S2 have different configurations
        print("\n" + "="*70)
        print("DIVERSE SPLIT GENERATION MODE ENABLED")
        print("S1 features: Cora + Chameleon")
        print("S2 features: Citeseer + Squirrel")
        print("="*70)
        
        s1_config = {
            'target_hom_range': args.s1_hom_range,
            'jitter': args.pair_jitter,
            'templates': args.s1_templates,
        }
        s2_config = {
            'target_hom_range': args.s2_hom_range,
            'jitter': args.pair_jitter,
            'templates': args.s2_templates,
        }
        test_config = {
            'target_hom_range': args.test_hom_range,
            'jitter': args.pair_jitter,
            'templates': args.test_templates,
        }
        
        for node_size in args.node_sizes:
            print(f"\n{'='*70}")
            print(f"Generating DIVERSE splits for node size {node_size}...")
            print(f"{'='*70}")
            
            split = generate_dataset_splits_diverse(
                templates=templates,
                node_size=node_size,
                train_per_set=args.train_per_set,
                test_count=args.test_count,
                s1_config=s1_config,
                s2_config=s2_config,
                test_config=test_config,
                feature_pool_s1=feature_pool_s1,
                feature_pool_s2=feature_pool_s2,
                feature_pool_test=feature_pool_s1,
                feature_dim=args.feature_dim,
                validate_diversity=args.validate_diversity,
                diversity_margin=args.diversity_margin,
                validation_sample_size=args.validation_sample_size,
                generate_s1=generate_s1,
                generate_s2=generate_s2,
                output_dir=args.output_dir,
            )
            
            save_split(args.output_dir, node_size, split)
            print(f"\n✓ Finished node size {node_size} -> {args.output_dir}/node_{node_size}")
            
            if args.visualize:
                print(f"Creating visualizations for node size {node_size}...")
                visualize_examples(args.output_dir, node_size, n_examples=args.viz_examples, layout='spring')
    else:
        # LEGACY MODE: Identical parameters for S1 and S2
        print("\n" + "="*70)
        print("LEGACY MODE: Generating splits with IDENTICAL parameters")
        print("WARNING: This may cause S1↔S2 convergence at large n")
        print("S1 features: Cora + Chameleon")
        print("S2 features: Citeseer + Squirrel")
        print("Use --diverse or --stratified flag for better split generation")
        print("="*70)
        
        for node_size in args.node_sizes:
            print(f"\nGenerating node size {node_size}...")
            split = generate_dataset_splits(
                templates=templates,
                node_size=node_size,
                total_graphs=args.total_per_size,
                train_per_set=args.train_per_set,
                test_count=args.test_count,
                target_hom=args.target_hom,
                jitter=args.pair_jitter,
                feature_pool_s1=feature_pool_s1,
                feature_pool_s2=feature_pool_s2,
                feature_pool_test=feature_pool_s1,
                feature_dim=args.feature_dim,
                output_dir=args.output_dir,
            )
            save_split(args.output_dir, node_size, split)
            print(f"Finished node size {node_size} -> {args.output_dir}/node_{node_size}")
            
            if args.visualize:
                print(f"Creating visualizations for node size {node_size}...")
                visualize_examples(args.output_dir, node_size, n_examples=args.viz_examples, layout='spring')



if __name__ == "__main__":
    main()