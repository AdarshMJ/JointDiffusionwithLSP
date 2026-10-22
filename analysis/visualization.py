"""
Visualization utilities for non-molecular graphs.
"""
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class NonMolecularVisualization:
    """Visualization tools for non-molecular graphs."""

    def __init__(self):
        pass

    @staticmethod
    def _community_layout(G, seed=42):
        """Return (pos, node_colors) using community-aware spring layout.

        Priority:
          1. 'block' node attribute (set by nx.stochastic_block_model)
          2. Greedy modularity community detection
          3. Plain spring_layout + uniform color as fallback
        """
        palette = list(plt.cm.Set1.colors) + list(plt.cm.Set2.colors)

        # --- try block attribute first ---
        block_attrs = [G.nodes[v].get('block', None) for v in G.nodes()]
        if all(b is not None for b in block_attrs):
            community_ids = block_attrs
        else:
            # --- community detection ---
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
            pos = nx.spring_layout(G, pos=init_pos, fixed=None, seed=seed, k=0.5)
            node_colors = [palette[block_index[b] % len(palette)]
                           for b in community_ids]
        else:
            pos = nx.spring_layout(G, seed=seed)
            node_colors = 'lightblue'

        return pos, node_colors

    def visualize(self, path: str, graphs: list, num_graphs_to_visualize: int, log='',
                  node_size=50, largest_component=False):
        """Visualize generated graphs with community-aware layout."""
        os.makedirs(path, exist_ok=True)

        for i in range(min(num_graphs_to_visualize, len(graphs))):
            graph = graphs[i]
            node_types, edge_types = graph

            A = edge_types.bool().cpu().numpy()
            G = nx.from_numpy_array(A)

            if largest_component and G.number_of_nodes() > 0:
                CGs = sorted([G.subgraph(c) for c in nx.connected_components(G)],
                             key=lambda x: x.number_of_nodes(), reverse=True)
                G = CGs[0]

            pos, node_colors = self._community_layout(G, seed=42)

            fig, ax = plt.subplots(figsize=(6, 6))
            nx.draw(G, pos, ax=ax, node_size=node_size, node_color=node_colors,
                    edge_color='gray', with_labels=False)
            ax.set_axis_off()

            save_path = os.path.join(path, f'graph_{i}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

    def visualize_gt_graphs(self, path: str, gt_graphs: list, num_graphs: int = 5, 
                            node_size=100):
        """Visualize ground truth graphs."""
        os.makedirs(path, exist_ok=True)
        
        n_visualize = min(num_graphs, len(gt_graphs))
        
        print(f"\n{'='*80}")
        print(f"Visualizing {n_visualize} Ground Truth Training Graphs")
        print(f"{'='*80}")
        
        for i in range(n_visualize):
            gt_graph = gt_graphs[i]
            
            # Handle different graph types
            if isinstance(gt_graph, nx.Graph):
                G_gt = gt_graph
            else:
                # If it's a tuple (node_types, edge_types)
                node_types, edge_types = gt_graph
                A = edge_types.bool().cpu().numpy() if hasattr(edge_types, 'cpu') else edge_types
                if isinstance(A, np.ndarray) and len(A.shape) == 3:
                    A = A.sum(axis=-1) > 0
                G_gt = nx.from_numpy_array(np.array(A).astype(bool))
            
            # Compute statistics
            n_nodes = G_gt.number_of_nodes()
            n_edges = G_gt.number_of_edges()
            avg_degree = 2 * n_edges / n_nodes if n_nodes > 0 else 0
            density = nx.density(G_gt) if n_nodes > 0 else 0
            
            print(f"\nGround Truth Graph {i}:")
            print(f"  Nodes: {n_nodes}")
            print(f"  Edges: {n_edges}")
            print(f"  Avg Degree: {avg_degree:.2f}")
            print(f"  Density: {density:.4f}")
            
            # Visualize — use community-aware layout when block attributes exist
            fig, ax = plt.subplots(figsize=(8, 8))

            block_attrs = [G_gt.nodes[v].get('block', None) for v in G_gt.nodes()]
            has_blocks = all(b is not None for b in block_attrs)

            if has_blocks:
                # Pre-seed positions: place each community in its own half-plane
                # so spring_layout preserves the separation
                blocks = sorted(set(block_attrs))
                n_blocks = len(blocks)
                init_pos = {}
                for v, b in zip(G_gt.nodes(), block_attrs):
                    angle = 2 * np.pi * blocks.index(b) / n_blocks
                    cx, cy = 2.0 * np.cos(angle), 2.0 * np.sin(angle)
                    init_pos[v] = np.array([cx + np.random.default_rng(v).normal(0, 0.3),
                                            cy + np.random.default_rng(v + 1000).normal(0, 0.3)])
                pos = nx.spring_layout(G_gt, pos=init_pos, fixed=None, seed=42, k=0.5)
                palette = plt.cm.Set1.colors
                node_colors = [palette[b % len(palette)] for b in block_attrs]
            else:
                pos = nx.spring_layout(G_gt, seed=42)
                node_colors = 'lightgreen'

            nx.draw(G_gt, pos, ax=ax, node_size=node_size, node_color=node_colors,
                    edge_color='gray', with_labels=True, font_size=10)

            # Add title with statistics
            title = f'GT Graph {i}: {n_nodes} nodes, {n_edges} edges\n'
            title += f'Avg Degree: {avg_degree:.2f}, Density: {density:.4f}'
            ax.set_title(title, fontsize=14, pad=20)
            ax.set_axis_off()
            
            save_path = os.path.join(path, f'gt_graph_{i}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        print(f"\nSaved {n_visualize} ground truth graph visualizations to {path}")
        print(f"{'='*80}\n")

    def visualize_comparison(self, path: str, gt_graphs: list, gen_graphs: list,
                             num_comparisons: int = 5, node_size=50):
        """Visualize ground truth vs generated graphs side by side."""
        os.makedirs(path, exist_ok=True)

        n_compare = min(num_comparisons, len(gt_graphs), len(gen_graphs))

        for i in range(n_compare):
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # ---- Ground truth ----
            gt_graph = gt_graphs[i]
            if isinstance(gt_graph, nx.Graph):
                G_gt = gt_graph
            else:
                node_types, edge_types = gt_graph
                A = edge_types.bool().cpu().numpy() if hasattr(edge_types, 'cpu') else edge_types
                if isinstance(A, np.ndarray) and len(A.shape) == 3:
                    A = A.sum(axis=-1) > 0
                G_gt = nx.from_numpy_array(np.array(A).astype(bool))

            pos_gt, colors_gt = self._community_layout(G_gt, seed=42)
            nx.draw(G_gt, pos_gt, ax=axes[0], node_size=node_size, node_color=colors_gt,
                    edge_color='gray', with_labels=False)
            axes[0].set_title('Ground Truth', fontsize=25)
            axes[0].set_axis_off()

            # ---- Generated ----
            gen_graph = gen_graphs[i]
            node_types, edge_types = gen_graph
            A = edge_types.bool().cpu().numpy() if hasattr(edge_types, 'cpu') else edge_types
            if isinstance(A, np.ndarray) and len(A.shape) == 3:
                A = A.sum(axis=-1) > 0
            G_gen = nx.from_numpy_array(np.array(A).astype(bool))

            pos_gen, colors_gen = self._community_layout(G_gen, seed=42)
            nx.draw(G_gen, pos_gen, ax=axes[1], node_size=node_size, node_color=colors_gen,
                    edge_color='gray', with_labels=False)
            axes[1].set_title('Generated', fontsize=25)
            axes[1].set_axis_off()

            plt.tight_layout()
            save_path = os.path.join(path, f'comparison_{i}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

        print(f"Saved {n_compare} comparison plots to {path}")

    def visualize_chain(self, path, nodes_list, adjacency_matrices):
        """Visualize a chain of graphs during sampling."""
        os.makedirs(path, exist_ok=True)
        
        for i, (nodes, adj) in enumerate(zip(nodes_list, adjacency_matrices)):
            if isinstance(adj, np.ndarray):
                A = adj
            else:
                A = adj.numpy() if hasattr(adj, 'numpy') else np.array(adj)
            
            # Handle case where adj might have an extra dimension
            if len(A.shape) == 3:
                A = A.sum(axis=-1) > 0
            A = A.astype(bool)
            
            G = nx.from_numpy_array(A)
            
            fig, ax = plt.subplots(figsize=(4, 4))
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, ax=ax, node_size=30, node_color='lightblue',
                    edge_color='gray', with_labels=False)
            ax.set_axis_off()
            
            save_path = os.path.join(path, f'step_{i:04d}.png')
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
        
        return path

    def to_networkx(self, node_types, edge_types):
        """Convert node and edge types to networkx graph."""
        if hasattr(edge_types, 'numpy'):
            A = edge_types.numpy()
        else:
            A = np.array(edge_types)
        
        if len(A.shape) == 3:
            A = A.sum(axis=-1) > 0
        
        return nx.from_numpy_array(A.astype(bool))
