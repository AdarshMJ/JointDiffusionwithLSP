"""
Spectre datasets: SBM, Planar, and Community graphs.
Adapted from DiGress to remove pytorch-lightning dependency.
"""
import os
import pathlib
import random

import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from diffusion.distributions import DistributionNodes
import utils


class SpectreGraphDataset(InMemoryDataset):
    """Dataset for Spectre graphs (SBM, Planar, Community)."""
    
    def __init__(self, dataset_name, split, root, transform=None, pre_transform=None, pre_filter=None):
        self.sbm_file = 'sbm_200.pt'
        self.planar_file = 'planar_64_200.pt'
        self.comm20_file = 'community_12_21_100.pt'
        self.dataset_name = dataset_name
        self.split = split
        self.num_graphs = 200
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
        return [self.split + '.pt']

    def download(self):
        """Download raw files."""
        if self.dataset_name == 'sbm':
            raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/sbm_200.pt'
        elif self.dataset_name == 'planar':
            raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/planar_64_200.pt'
        elif self.dataset_name == 'comm20':
            raw_url = 'https://raw.githubusercontent.com/KarolisMart/SPECTRE/main/data/community_12_21_100.pt'
        else:
            raise ValueError(f'Unknown dataset {self.dataset_name}')
        file_path = download_url(raw_url, self.raw_dir)

        adjs, eigvals, eigvecs, n_nodes, max_eigval, min_eigval, same_sample, n_max = torch.load(file_path)

        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)

        test_len = int(round(self.num_graphs * 0.2))
        train_len = int(round((self.num_graphs - test_len) * 0.8))
        val_len = self.num_graphs - train_len - test_len
        indices = torch.randperm(self.num_graphs, generator=g_cpu)
        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len + val_len]
        test_indices = indices[train_len + val_len:]

        train_data = []
        val_data = []
        test_data = []

        for i, adj in enumerate(adjs):
            if i in train_indices:
                train_data.append(adj)
            elif i in val_indices:
                val_data.append(adj)
            elif i in test_indices:
                test_data.append(adj)
            else:
                raise ValueError(f'Index {i} not in any split')

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])

    def process(self):
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])

        data_list = []
        for adj in raw_dataset:
            n = adj.shape[-1]
            X = torch.ones(n, 1, dtype=torch.float)
            y = torch.zeros([1, 0]).float()
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
            edge_attr[:, 1] = 1
            num_nodes = n * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(x=X, edge_index=edge_index, edge_attr=edge_attr,
                                             y=y, n_nodes=num_nodes)
            
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        torch.save(self.collate(data_list), self.processed_paths[0])


class SpectreGraphDataModule:
    """DataModule for Spectre graphs (replaces LightningDataset)."""
    
    def __init__(self, cfg, n_graphs=200):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
        root_path = os.path.join(base_path, self.datadir)

        self.train_dataset = SpectreGraphDataset(
            dataset_name=self.cfg.dataset.name,
            split='train', root=root_path)
        self.val_dataset = SpectreGraphDataset(
            dataset_name=self.cfg.dataset.name,
            split='val', root=root_path)
        self.test_dataset = SpectreGraphDataset(
            dataset_name=self.cfg.dataset.name,
            split='test', root=root_path)
        
        # Single graph mode for overfitting test
        self.single_graph = getattr(cfg.dataset, 'single_graph', False)
        self.single_graph_idx = getattr(cfg.dataset, 'single_graph_idx', 0)
        
        if self.single_graph:
            print(f"=== SINGLE GRAPH MODE: Using graph {self.single_graph_idx} for train/val/test ===")
            # Create a subset with just one graph, repeated for proper batching
            single_data = self.train_dataset[self.single_graph_idx]
            self.single_graph_data = single_data
            # Use Subset to create a single-element dataset
            from torch.utils.data import Subset
            self.train_dataset = Subset(self.train_dataset, [self.single_graph_idx])
            self.val_dataset = Subset(self.train_dataset.dataset, [self.single_graph_idx])
            self.test_dataset = Subset(self.train_dataset.dataset, [self.single_graph_idx])
        
        self.batch_size = cfg.train.batch_size if 'debug' not in cfg.general.name else 2
        self.num_workers = getattr(cfg.train, 'num_workers', 0)
        self.pin_memory = getattr(cfg.dataset, "pin_memory", False)
        
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]
    
    def train_dataloader(self):
        from torch_geometric.loader import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self):
        from torch_geometric.loader import DataLoader
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self):
        from torch_geometric.loader import DataLoader
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def get_gt_graphs(self, num_graphs=5, split='train'):
        """Get ground truth graphs for visualization comparison."""
        import networkx as nx
        if split == 'train':
            dataset = self.train_dataset
        elif split == 'val':
            dataset = self.val_dataset
        else:
            dataset = self.test_dataset
        
        graphs = []
        for i in range(min(num_graphs, len(dataset))):
            data = dataset[i]
            # Convert to adjacency matrix
            n_nodes = data.x.shape[0]
            adj = torch.zeros(n_nodes, n_nodes)
            edge_index = data.edge_index
            adj[edge_index[0], edge_index[1]] = 1
            
            # Convert to networkx
            G = nx.from_numpy_array(adj.numpy())
            graphs.append(G)
        
        return graphs

    def node_counts(self, max_nodes_possible=None):
        """Compute distribution of node counts."""
        if max_nodes_possible is None:
            max_nodes_possible = 0
            for loader in [self.train_dataloader(), self.val_dataloader()]:
                for data in loader:
                    _, counts = torch.unique(data.batch, return_counts=True)
                    max_nodes_possible = max(max_nodes_possible, int(counts.max().item()))
            max_nodes_possible += 1  # size must exceed the largest count
        all_counts = torch.zeros(max_nodes_possible)
        for loader in [self.train_dataloader(), self.val_dataloader()]:
            for data in loader:
                unique, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[:max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts

    def node_types(self):
        """Compute distribution of node types."""
        num_classes = None
        for data in self.train_dataloader():
            num_classes = data.x.shape[1]
            break

        counts = torch.zeros(num_classes)

        for i, data in enumerate(self.train_dataloader()):
            counts += data.x.sum(dim=0)

        counts = counts / counts.sum()
        return counts

    def edge_counts(self):
        """Compute distribution of edge types."""
        num_classes = None
        for data in self.train_dataloader():
            num_classes = data.edge_attr.shape[1]
            break

        d = torch.zeros(num_classes, dtype=torch.float)

        for i, data in enumerate(self.train_dataloader()):
            unique, counts = torch.unique(data.batch, return_counts=True)

            all_pairs = 0
            for count in counts:
                all_pairs += count * (count - 1)

            num_edges = data.edge_index.shape[1]
            num_non_edges = all_pairs - num_edges

            edge_types = data.edge_attr.sum(dim=0)
            assert num_non_edges >= 0
            d[0] += num_non_edges
            d[1:] += edge_types[1:]

        d = d / d.sum()
        return d


class SpectreDatasetInfos:
    """Dataset information for Spectre graphs."""
    
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = torch.tensor([1])  # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        
        # Complete infos
        self.input_dims = None
        self.output_dims = None
        self.num_classes = len(self.node_types)
        self.max_n_nodes = len(self.n_nodes) - 1
        self.nodes_dist = DistributionNodes(self.n_nodes)

    def compute_input_output_dims(self, datamodule, extra_features, domain_features):
        """Compute input and output dimensions."""
        example_batch = next(iter(datamodule.train_dataloader()))
        ex_dense, node_mask = utils.to_dense(
            example_batch.x, example_batch.edge_index, 
            example_batch.edge_attr, example_batch.batch)
        example_data = {
            'X_t': ex_dense.X, 
            'E_t': ex_dense.E, 
            'y_t': example_batch['y'], 
            'node_mask': node_mask
        }

        self.input_dims = {
            'X': example_batch['x'].size(1),
            'E': example_batch['edge_attr'].size(1),
            'y': example_batch['y'].size(1) + 1  # + 1 due to time conditioning
        }
        
        ex_extra_feat = extra_features(example_data)
        self.input_dims['X'] += ex_extra_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_feat.y.size(-1)

        ex_extra_molecular_feat = domain_features(example_data)
        self.input_dims['X'] += ex_extra_molecular_feat.X.size(-1)
        self.input_dims['E'] += ex_extra_molecular_feat.E.size(-1)
        self.input_dims['y'] += ex_extra_molecular_feat.y.size(-1)

        self.output_dims = {
            'X': example_batch['x'].size(1),
            'E': example_batch['edge_attr'].size(1),
            'y': 0
        }

# Specific data modules for different datasets
class SBMDataModule(SpectreGraphDataModule):
    """DataModule for SBM graphs."""
    def __init__(self, cfg):
        cfg.dataset.name = 'sbm'
        super().__init__(cfg, n_graphs=200)
        self.dataset_infos = SpectreDatasetInfos(self, cfg.dataset)


class PlanarDataModule(SpectreGraphDataModule):
    """DataModule for Planar graphs."""
    def __init__(self, cfg):
        cfg.dataset.name = 'planar'
        super().__init__(cfg, n_graphs=200)
        self.dataset_infos = SpectreDatasetInfos(self, cfg.dataset)


class Comm20DataModule(SpectreGraphDataModule):
    """DataModule for Community (Comm20) graphs."""
    def __init__(self, cfg):
        cfg.dataset.name = 'comm20'
        super().__init__(cfg, n_graphs=100)
        self.dataset_infos = SpectreDatasetInfos(self, cfg.dataset)


class CoraChameleonDataset(Dataset):
    """Dataset for CoraChameleon graphs."""
    
    def __init__(self, data_list):
        """
        Args:
            data_list: List of (Data, metadata_dict) tuples
        """
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data, metadata = self.data_list[idx]
        
        # Extract number of nodes
        n_nodes = data.num_nodes
        
        # Create constant node features (instead of using the 100-dim features)
        X = torch.ones(n_nodes, 1, dtype=torch.float)
        
        # Convert edge_index to undirected (ensure symmetry for model)
        edge_index_undirected = torch_geometric.utils.to_undirected(data.edge_index)
        
        # Create edge attributes (all same type)
        edge_attr = torch.zeros(edge_index_undirected.shape[-1], 2, dtype=torch.float)
        edge_attr[:, 1] = 1  # All edges are type 1
        
        # Create empty y
        y = torch.zeros([1, 0]).float()
        
        # Create num_nodes tensor
        num_nodes_tensor = torch.tensor([n_nodes], dtype=torch.long)
        
        # Create new Data object with processed features
        processed_data = torch_geometric.data.Data(
            x=X,
            edge_index=edge_index_undirected,
            edge_attr=edge_attr,
            y=y,
            n_nodes=num_nodes_tensor
        )
        
        return processed_data


class CoraChameleonDataModule:
    """DataModule for CoraChameleon graphs."""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        
        # Get configuration parameters
        node_size = cfg.dataset.node_size  # e.g., 10, 20, 30, ..., 100, 500
        split_type = getattr(cfg.dataset, 'split_type', 'S1')  # 'S1' or 'S2'
        num_train_graphs = getattr(cfg.dataset, 'num_train_graphs', 3000)  # How many from train to use
        
        # Construct paths
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
        data_folder = os.path.join(base_path, self.datadir, f'node_{node_size}')
        
        train_path = os.path.join(data_folder, f'{split_type}.pt')
        test_path = os.path.join(data_folder, 'test.pt')
        
        print(f"Loading CoraChameleon dataset:")
        print(f"  Node size: {node_size}")
        print(f"  Split type: {split_type}")
        print(f"  Train path: {train_path}")
        print(f"  Test path: {test_path}")
        
        # Load data
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training data not found: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test data not found: {test_path}")
        
        train_data_raw = torch.load(train_path)
        test_data_raw = torch.load(test_path)
        
        # Limit training graphs if specified (0 or negative means use all)
        if num_train_graphs > 0 and num_train_graphs < len(train_data_raw):
            print(f"  Using {num_train_graphs}/{len(train_data_raw)} training graphs")
            train_data_raw = train_data_raw[:num_train_graphs]
        else:
            print(f"  Using all {len(train_data_raw)} training graphs")
        
        # Split test into val (60 graphs) and test (40 graphs)
        val_data_raw = test_data_raw[:60]
        test_data_raw = test_data_raw[60:]
        
        print(f"  Train size: {len(train_data_raw)}")
        print(f"  Val size: {len(val_data_raw)}")
        print(f"  Test size: {len(test_data_raw)}")
        
        # Create datasets
        self.train_dataset = CoraChameleonDataset(train_data_raw)
        self.val_dataset = CoraChameleonDataset(val_data_raw)
        self.test_dataset = CoraChameleonDataset(test_data_raw)
        
        # Single graph mode for overfitting test
        self.single_graph = getattr(cfg.dataset, 'single_graph', False)
        self.single_graph_idx = getattr(cfg.dataset, 'single_graph_idx', 0)
        
        if self.single_graph:
            if self.single_graph_idx >= len(self.train_dataset):
                raise IndexError(
                    f"single_graph_idx={self.single_graph_idx} but train_dataset "
                    f"only has {len(self.train_dataset)} graphs. "
                    f"Increase num_train_graphs or decrease single_graph_idx.")
            print(f"=== SINGLE GRAPH MODE: Using graph {self.single_graph_idx} for train/val/test ===")
            from torch.utils.data import Subset
            self.train_dataset = Subset(self.train_dataset, [self.single_graph_idx])
            self.val_dataset = Subset(self.train_dataset.dataset, [self.single_graph_idx])
            self.test_dataset = Subset(self.train_dataset.dataset, [self.single_graph_idx])
        
        self.batch_size = cfg.train.batch_size if 'debug' not in cfg.general.name else 2
        self.num_workers = getattr(cfg.train, 'num_workers', 0)
        self.pin_memory = getattr(cfg.dataset, "pin_memory", False)
        
        # Create dataset infos
        self.dataset_infos = SpectreDatasetInfos(self, cfg.dataset)
        
        self.inner = self.train_dataset
    
    def __getitem__(self, item):
        return self.inner[item]
    
    def train_dataloader(self):
        from torch_geometric.loader import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def val_dataloader(self):
        from torch_geometric.loader import DataLoader
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def test_dataloader(self):
        from torch_geometric.loader import DataLoader
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
    
    def get_gt_graphs(self, num_graphs=5, split='train'):
        """Get ground truth graphs for visualization comparison."""
        import networkx as nx
        if split == 'train':
            dataset = self.train_dataset
        elif split == 'val':
            dataset = self.val_dataset
        else:
            dataset = self.test_dataset
        
        graphs = []
        for i in range(min(num_graphs, len(dataset))):
            data = dataset[i]
            # Convert to adjacency matrix
            n_nodes = data.x.shape[0]
            adj = torch.zeros(n_nodes, n_nodes)
            edge_index = data.edge_index
            adj[edge_index[0], edge_index[1]] = 1
            
            # Convert to networkx
            G = nx.from_numpy_array(adj.numpy())
            graphs.append(G)
        
        return graphs
    
    def node_counts(self, max_nodes_possible=None):
        """Compute distribution of node counts."""
        if max_nodes_possible is None:
            max_nodes_possible = 0
            for loader in [self.train_dataloader(), self.val_dataloader()]:
                for data in loader:
                    _, counts = torch.unique(data.batch, return_counts=True)
                    max_nodes_possible = max(max_nodes_possible, int(counts.max().item()))
            max_nodes_possible += 1  # size must exceed the largest count
        all_counts = torch.zeros(max_nodes_possible)
        for loader in [self.train_dataloader(), self.val_dataloader()]:
            for data in loader:
                unique, counts = torch.unique(data.batch, return_counts=True)
                for count in counts:
                    all_counts[count] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[:max_index + 1]
        all_counts = all_counts / all_counts.sum()
        return all_counts
    
    def node_types(self):
        """Compute distribution of node types."""
        num_classes = None
        for data in self.train_dataloader():
            num_classes = data.x.shape[1]
            break

        counts = torch.zeros(num_classes)

        for i, data in enumerate(self.train_dataloader()):
            counts += data.x.sum(dim=0)

        counts = counts / counts.sum()
        return counts
    
    def edge_counts(self):
        """Compute distribution of edge types."""
        num_classes = None
        for data in self.train_dataloader():
            num_classes = data.edge_attr.shape[1]
            break

        d = torch.zeros(num_classes, dtype=torch.float)

        for i, data in enumerate(self.train_dataloader()):
            unique, counts = torch.unique(data.batch, return_counts=True)

            all_pairs = 0
            for count in counts:
                all_pairs += count * (count - 1)

            num_edges = data.edge_index.shape[1]
            num_non_edges = all_pairs - num_edges

            edge_types = data.edge_attr.sum(dim=0)
            assert num_non_edges >= 0
            d[0] += num_non_edges
            d[1:] += edge_types[1:]

        d = d / d.sum()
        return d


# ==============================================================================
#  SYNTHETIC EXPERIMENT DATASETS
#  Two experiments designed to show CRM prior advantage over stock DiGress
# ==============================================================================

# ------------------------------------------------------------------------------
#  Shared low-level helpers
# ------------------------------------------------------------------------------

def _nx_to_pyg(G: nx.Graph) -> torch_geometric.data.Data:
    """Convert a NetworkX graph to the PyG Data format used by this codebase."""
    n = G.number_of_nodes()
    X = torch.ones(n, 1, dtype=torch.float)
    y = torch.zeros([1, 0], dtype=torch.float)
    adj = torch.from_numpy(nx.to_numpy_array(G)).float()
    edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
    edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
    edge_attr[:, 1] = 1
    num_nodes = n * torch.ones(1, dtype=torch.long)
    return torch_geometric.data.Data(
        x=X, edge_index=edge_index, edge_attr=edge_attr,
        y=y, n_nodes=num_nodes)


def _generate_two_density_graphs(n_graphs, n_nodes, p_sparse, p_dense, seed=42):
    """Generate a balanced mix of sparse and dense Erdős-Rényi graphs.

    Returns:
        graphs: list of nx.Graph
        labels: list of str ('sparse' or 'dense'), same length
    """
    rng = np.random.default_rng(seed)
    graphs, labels = [], []
    n_each = n_graphs // 2
    for _ in range(n_each):
        s = int(rng.integers(0, 2**31))
        graphs.append(nx.erdos_renyi_graph(n_nodes, p_sparse, seed=s))
        labels.append('sparse')
    for _ in range(n_each):
        s = int(rng.integers(0, 2**31))
        graphs.append(nx.erdos_renyi_graph(n_nodes, p_dense, seed=s))
        labels.append('dense')
    perm = rng.permutation(len(graphs))
    return [graphs[i] for i in perm], [labels[i] for i in perm]


def _generate_dcsbm_graphs(n_graphs, comm_sizes, p_intra, p_inter, seed=42):
    """Generate SBM graphs: two communities with different intra-densities.

    Args:
        comm_sizes: (n0, n1) — community 0 (dense) and 1 (sparse)
        p_intra:    (p0, p1) — intra-community edge probabilities
        p_inter:    float    — inter-community edge probability

    Returns:
        graphs:      list of nx.Graph
        comm_labels: list of np.ndarray shape (n0+n1,) with values in {0,1}
    """
    rng = np.random.default_rng(seed)
    n0, n1 = comm_sizes
    n_total = n0 + n1
    base_labels = np.array([0] * n0 + [1] * n1)
    comm0 = list(range(n0))
    comm1 = list(range(n0, n_total))
    p_matrix = [[p_intra[0], p_inter],
                [p_inter,   p_intra[1]]]
    # Guarantee at least this many inter-community edges per graph
    min_inter = max(3, round(n0 * n1 * p_inter))
    graphs, comm_labels = [], []
    for _ in range(n_graphs):
        s = int(rng.integers(0, 2**31))
        G = nx.stochastic_block_model([n0, n1], p_matrix, seed=s)

        # ---- Enforce no isolated nodes (before permutation) -------------
        for v in list(nx.isolates(G)):
            pool = [u for u in (comm0 if v < n0 else comm1) if u != v]
            G.add_edge(v, int(rng.choice(pool)))

        # ---- Ensure minimum inter-community connectivity ----------------
        inter = [(u, v) for u, v in G.edges() if (u < n0) != (v < n0)]
        while len(inter) < min_inter:
            u = int(rng.choice(comm0))
            v = int(rng.choice(comm1))
            if not G.has_edge(u, v):
                G.add_edge(u, v)
                inter.append((u, v))

        # ---- Permute node labels (permutation invariance) ---------------
        perm = rng.permutation(n_total)
        G = nx.relabel_nodes(G, {int(o): int(p) for o, p in enumerate(perm)})
        G = nx.convert_node_labels_to_integers(G)
        # new node perm[o] inherits base_labels[o]
        new_labels = np.empty(n_total, dtype=int)
        new_labels[perm] = base_labels
        graphs.append(G)
        comm_labels.append(new_labels)
    return graphs, comm_labels


# ------------------------------------------------------------------------------
#  Visualisation helpers  (run once at dataset generation time)
# ------------------------------------------------------------------------------

def _visualize_two_density_stats(graphs, labels, p_sparse, p_dense, save_dir):
    """Save a 2x2 statistics figure for the two-density mixture training set.

    Key panels (all directly motivated by what CRM should preserve vs DiGress):
      (A) Per-graph density histogram  — bimodal; DiGress collapses to mean
      (B) Degree distribution per family — bimodal degree support
      (C) Per-graph degree variance     — u_0 heterogeneity signal
      (D) Avg clustering coefficient   — secondary structural property
    """
    FS = 25
    sparse_g = [G for G, l in zip(graphs, labels) if l == 'sparse']
    dense_g  = [G for G, l in zip(graphs, labels) if l == 'dense']

    sp_dens  = [nx.density(G) for G in sparse_g]
    de_dens  = [nx.density(G) for G in dense_g]
    sp_degs  = [d for G in sparse_g for _, d in G.degree()]
    de_degs  = [d for G in dense_g  for _, d in G.degree()]
    sp_vars  = [float(np.var([d for _, d in G.degree()])) for G in sparse_g]
    de_vars  = [float(np.var([d for _, d in G.degree()])) for G in dense_g]
    sp_clst  = [nx.average_clustering(G) for G in sparse_g]
    de_clst  = [nx.average_clustering(G) for G in dense_g]
    overall_mean = np.mean(sp_dens + de_dens)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # (A) Density histogram — key demo: bimodal distribution DiGress flattens
    ax = axes[0, 0]
    bins = np.linspace(0, max(max(sp_dens), max(de_dens)) * 1.05, 35)
    ax.hist(sp_dens, bins=bins, alpha=0.70, color='steelblue',
            label=f'Sparse  (p={p_sparse})')
    ax.hist(de_dens, bins=bins, alpha=0.70, color='tomato',
            label=f'Dense  (p={p_dense})')
    ax.axvline(overall_mean, ls='--', color='black', linewidth=2.5,
               label=f'Overall mean ({overall_mean:.2f})')
    ax.set_xlabel("Edge Density", fontsize=FS)
    ax.set_ylabel("Count", fontsize=FS)
    ax.tick_params(labelsize=FS - 4)
    ax.legend(fontsize=FS - 5)

    # (B) Degree distribution per family
    ax = axes[0, 1]
    max_deg = max(max(sp_degs), max(de_degs)) + 1
    dbins = np.arange(-0.5, max_deg + 1.5, 1)
    ax.hist(sp_degs, bins=dbins, density=True, alpha=0.70,
            color='steelblue', label=f'Sparse  (p={p_sparse})')
    ax.hist(de_degs, bins=dbins, density=True, alpha=0.70,
            color='tomato',    label=f'Dense  (p={p_dense})')
    ax.set_xlabel("Node Degree", fontsize=FS)
    ax.set_ylabel("Density", fontsize=FS)
    ax.tick_params(labelsize=FS - 4)
    ax.legend(fontsize=FS - 5)

    # (C) Per-graph degree variance — directly maps to u_0 signal strength
    ax = axes[1, 0]
    vmax = max(max(sp_vars), max(de_vars)) * 1.05
    vbins = np.linspace(0, vmax, 35)
    ax.hist(sp_vars, bins=vbins, alpha=0.70, color='steelblue', label='Sparse')
    ax.hist(de_vars, bins=vbins, alpha=0.70, color='tomato',    label='Dense')
    ax.set_xlabel("Per-graph Degree Variance", fontsize=FS)
    ax.set_ylabel("Count", fontsize=FS)
    ax.tick_params(labelsize=FS - 4)
    ax.legend(fontsize=FS - 5)

    # (D) Clustering coefficient
    ax = axes[1, 1]
    ax.hist(sp_clst, bins=30, alpha=0.70, color='steelblue', label='Sparse')
    ax.hist(de_clst, bins=30, alpha=0.70, color='tomato',    label='Dense')
    ax.set_xlabel("Avg Clustering Coefficient", fontsize=FS)
    ax.set_ylabel("Count", fontsize=FS)
    ax.tick_params(labelsize=FS - 4)
    ax.legend(fontsize=FS - 5)

    fig.tight_layout()
    out = os.path.join(save_dir, 'two_density_stats.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)

    print(f"\n  [Two-Density Dataset Stats]")
    print(f"    Sparse ({len(sparse_g)} graphs): "
          f"density {np.mean(sp_dens):.3f}±{np.std(sp_dens):.3f}, "
          f"mean_deg {np.mean(sp_degs):.2f}, deg_var {np.mean(sp_vars):.2f}")
    print(f"    Dense  ({len(dense_g)} graphs):  "
          f"density {np.mean(de_dens):.3f}±{np.std(de_dens):.3f}, "
          f"mean_deg {np.mean(de_degs):.2f}, deg_var {np.mean(de_vars):.2f}")
    print(f"    Stats figure -> {out}")


def _visualize_dcsbm_stats(graphs, comm_labels, comm_sizes, p_intra, p_inter,
                            save_dir):
    """Save a 2x2 statistics figure for the DC-SBM training set.

    Key panels:
      (A) Per-community degree distribution — bimodal; maps directly to u_0
      (B) Per-graph edge density
      (C) Per-graph degree variance         — u_0 heterogeneity signal
      (D) Intra-community density scatter (comm0 vs comm1) per graph
    """
    FS = 25
    n0, n1 = comm_sizes
    comm0_degs, comm1_degs = [], []
    all_dens, all_deg_vars = [], []
    intra0_dens, intra1_dens = [], []

    for G, cl in zip(graphs, comm_labels):
        deg = dict(G.degree())
        all_dens.append(nx.density(G))
        all_deg_vars.append(float(np.var(list(deg.values()))))
        for node, comm in enumerate(cl):
            (comm0_degs if comm == 0 else comm1_degs).append(deg.get(node, 0))
        e0 = sum(1 for u, v in G.edges() if cl[u] == 0 and cl[v] == 0)
        e1 = sum(1 for u, v in G.edges() if cl[u] == 1 and cl[v] == 1)
        intra0_dens.append(e0 / max(n0 * (n0 - 1) / 2, 1))
        intra1_dens.append(e1 / max(n1 * (n1 - 1) / 2, 1))

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # (A) Per-community degree distribution — key structural signal
    ax = axes[0, 0]
    max_deg = max(max(comm0_degs), max(comm1_degs)) + 1
    dbins = np.arange(-0.5, max_deg + 1.5, 1)
    ax.hist(comm0_degs, bins=dbins, density=True, alpha=0.70, color='tomato',
            label=f'Community 0  (n={n0}, p={p_intra[0]})')
    ax.hist(comm1_degs, bins=dbins, density=True, alpha=0.70, color='steelblue',
            label=f'Community 1  (n={n1}, p={p_intra[1]})')
    ax.set_xlabel("Node Degree", fontsize=FS)
    ax.set_ylabel("Density", fontsize=FS)
    ax.tick_params(labelsize=FS - 4)
    ax.legend(fontsize=FS - 5)

    # (B) Per-graph density
    ax = axes[0, 1]
    ax.hist(all_dens, bins=30, color='slategray', alpha=0.85)
    ax.axvline(np.mean(all_dens), ls='--', color='black', linewidth=2.5,
               label=f'Mean ({np.mean(all_dens):.3f})')
    ax.set_xlabel("Edge Density", fontsize=FS)
    ax.set_ylabel("Count", fontsize=FS)
    ax.tick_params(labelsize=FS - 4)
    ax.legend(fontsize=FS - 5)

    # (C) Per-graph degree variance
    ax = axes[1, 0]
    ax.hist(all_deg_vars, bins=30, color='mediumpurple', alpha=0.85)
    ax.axvline(np.mean(all_deg_vars), ls='--', color='black', linewidth=2.5,
               label=f'Mean ({np.mean(all_deg_vars):.2f})')
    ax.set_xlabel("Per-graph Degree Variance", fontsize=FS)
    ax.set_ylabel("Count", fontsize=FS)
    ax.tick_params(labelsize=FS - 4)
    ax.legend(fontsize=FS - 5)

    # (D) Intra-community density scatter
    ax = axes[1, 1]
    ax.scatter(intra0_dens, intra1_dens, alpha=0.35, s=18, color='slategray')
    ax.axvline(p_intra[0], ls='--', color='tomato',    linewidth=2, alpha=0.8,
               label=f'Target comm 0 ({p_intra[0]})')
    ax.axhline(p_intra[1], ls='--', color='steelblue', linewidth=2, alpha=0.8,
               label=f'Target comm 1 ({p_intra[1]})')
    ax.set_xlabel("Community 0 Intra-density", fontsize=FS)
    ax.set_ylabel("Community 1 Intra-density", fontsize=FS)
    ax.tick_params(labelsize=FS - 4)
    ax.legend(fontsize=FS - 5)

    fig.tight_layout()
    out = os.path.join(save_dir, 'dcsbm_stats.pdf')
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close(fig)

    print(f"\n  [DC-SBM Dataset Stats]")
    print(f"    Comm0 (n={n0}, p_intra={p_intra[0]}): "
          f"mean_deg={np.mean(comm0_degs):.2f}")
    print(f"    Comm1 (n={n1}, p_intra={p_intra[1]}): "
          f"mean_deg={np.mean(comm1_degs):.2f}")
    print(f"    Intra0: {np.mean(intra0_dens):.3f}±{np.std(intra0_dens):.3f}  "
          f"(target {p_intra[0]})")
    print(f"    Intra1: {np.mean(intra1_dens):.3f}±{np.std(intra1_dens):.3f}  "
          f"(target {p_intra[1]})")
    print(f"    Overall density: {np.mean(all_dens):.3f}±{np.std(all_dens):.3f}")
    print(f"    Stats figure -> {out}")


# ------------------------------------------------------------------------------
#  Shared DataModule base (loaders + node/edge count distributions)
# ------------------------------------------------------------------------------

class _SyntheticDataModuleBase:
    """Mixin that provides train/val/test loaders and distribution helpers.

    Subclasses must assign:
        self.train_dataset, self.val_dataset, self.test_dataset
        self.batch_size, self.num_workers, self.pin_memory
    """

    def __getitem__(self, item):
        return self.train_dataset[item]

    def train_dataloader(self):
        from torch_geometric.loader import DataLoader
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def val_dataloader(self):
        from torch_geometric.loader import DataLoader
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        from torch_geometric.loader import DataLoader
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def get_gt_graphs(self, num_graphs=5, split='train'):
        dataset = {'train': self.train_dataset,
                   'val':   self.val_dataset,
                   'test':  self.test_dataset}[split]
        graphs = []
        for i in range(min(num_graphs, len(dataset))):
            data = dataset[i]
            n = data.x.shape[0]
            adj = torch.zeros(n, n)
            adj[data.edge_index[0], data.edge_index[1]] = 1
            graphs.append(nx.from_numpy_array(adj.numpy()))
        return graphs

    def node_counts(self, max_nodes_possible=None):
        if max_nodes_possible is None:
            max_nodes_possible = 0
            for loader in [self.train_dataloader(), self.val_dataloader()]:
                for data in loader:
                    _, counts = torch.unique(data.batch, return_counts=True)
                    max_nodes_possible = max(max_nodes_possible,
                                            int(counts.max().item()))
            max_nodes_possible += 1
        all_counts = torch.zeros(max_nodes_possible)
        for loader in [self.train_dataloader(), self.val_dataloader()]:
            for data in loader:
                _, counts = torch.unique(data.batch, return_counts=True)
                for c in counts:
                    all_counts[c] += 1
        max_index = max(all_counts.nonzero())
        all_counts = all_counts[:max_index + 1]
        return all_counts / all_counts.sum()

    def node_types(self):
        counts = torch.zeros(1)
        for data in self.train_dataloader():
            counts += data.x.sum(dim=0)
        return counts / counts.sum()

    def edge_counts(self):
        d = torch.zeros(2, dtype=torch.float)
        for data in self.train_dataloader():
            _, counts = torch.unique(data.batch, return_counts=True)
            all_pairs = sum(int(c * (c - 1)) for c in counts)
            d[0] += all_pairs - data.edge_index.shape[1]
            d[1] += data.edge_index.shape[1]
        return d / d.sum()


# ==============================================================================
#  Experiment 1: Two-Density Mixture
#  50/50 mix of sparse ER (p_sparse) and dense ER (p_dense), same node count.
#  Claim: CRM preserves bimodal density; DiGress collapses to average mode.
# ==============================================================================

class TwoDensityMixtureDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class TwoDensityMixtureDataModule(_SyntheticDataModuleBase):
    """DataModule for the two-density ER mixture comparison experiment.

    Training distribution is bimodal in density (two clear peaks).
    CRM's u_0 anchoring should preserve this bimodality at generation time;
    standard DiGress uses a shared marginal and collapses to the average density.

    Config keys (under dataset):
        n_nodes:   int    node count per graph   (default 25)
        p_sparse:  float  sparse ER probability  (default 0.10)
        p_dense:   float  dense  ER probability  (default 0.50)
        n_graphs:  int    total graphs (even)    (default 400)
        seed:      int    RNG seed               (default 42)
    """

    def __init__(self, cfg):
        n_nodes  = getattr(cfg.dataset, 'n_nodes',  25)
        p_sparse = getattr(cfg.dataset, 'p_sparse', 0.10)
        p_dense  = getattr(cfg.dataset, 'p_dense',  0.50)
        n_graphs = getattr(cfg.dataset, 'n_graphs', 400)
        seed     = getattr(cfg.dataset, 'seed',     42)

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
        data_dir  = os.path.join(base_path, cfg.dataset.datadir)
        os.makedirs(data_dir, exist_ok=True)

        fname = (f'twodensity_n{n_nodes}_ps{int(p_sparse * 100)}'
                 f'_pd{int(p_dense * 100)}_N{n_graphs}_seed{seed}.pt')
        cache_path = os.path.join(data_dir, fname)

        if os.path.exists(cache_path):
            print(f"Loading cached two-density dataset from {cache_path}")
            pairs = torch.load(cache_path)
        else:
            print(f"Generating two-density mixture "
                  f"(n={n_nodes}, p_sparse={p_sparse}, p_dense={p_dense}, "
                  f"N={n_graphs})...")
            nx_graphs, labels = _generate_two_density_graphs(
                n_graphs, n_nodes, p_sparse, p_dense, seed)
            pairs = [(_nx_to_pyg(G), lbl) for G, lbl in zip(nx_graphs, labels)]
            torch.save(pairs, cache_path)
            print(f"  Saved to {cache_path}")
            stats_dir = os.path.join(data_dir, 'stats')
            os.makedirs(stats_dir, exist_ok=True)
            _visualize_two_density_stats(nx_graphs, labels, p_sparse, p_dense,
                                         stats_dir)

        # 70 / 15 / 15 split
        n_total = len(pairs)
        n_test  = max(1, int(round(n_total * 0.15)))
        n_val   = max(1, int(round(n_total * 0.15)))
        n_train = n_total - n_val - n_test
        rng = random.Random(seed)
        idx = list(range(n_total))
        rng.shuffle(idx)

        self.train_dataset = TwoDensityMixtureDataset(
            [pairs[i][0] for i in idx[:n_train]])
        self.val_dataset   = TwoDensityMixtureDataset(
            [pairs[i][0] for i in idx[n_train:n_train + n_val]])
        self.test_dataset  = TwoDensityMixtureDataset(
            [pairs[i][0] for i in idx[n_train + n_val:]])

        self.batch_size  = cfg.train.batch_size
        self.num_workers = getattr(cfg.train,   'num_workers', 0)
        self.pin_memory  = getattr(cfg.dataset, 'pin_memory',  False)
        self.dataset_infos = SpectreDatasetInfos(self, cfg.dataset)
        self.inner = self.train_dataset

        print(f"  TwoDensity: train={len(self.train_dataset)}  "
              f"val={len(self.val_dataset)}  test={len(self.test_dataset)}")


# ==============================================================================
#  Experiment 2: Degree-Corrected SBM
#  Two-community SBM: community 0 is dense, community 1 is sparse.
#  Claim: CRM's u_0 encodes per-node degree role (hub vs peripheral), allowing
#  the model to recover community degree asymmetry. DiGress treats all nodes
#  equivalently via a shared marginal.
# ==============================================================================

class DCSBMDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class DCSBMDataModule(_SyntheticDataModuleBase):
    """DataModule for the DC-SBM community degree asymmetry experiment.

    Config keys (under dataset):
        comm_sizes:      list  [n0, n1]  community sizes         (default [20, 30])
        p_intra_dense:   float intra-community p for dense comm  (default 0.70)
        p_intra_sparse:  float intra-community p for sparse comm (default 0.30)
        p_inter:         float inter-community edge probability  (default 0.03)
        n_graphs:        int   total graphs                      (default 400)
        seed:            int   RNG seed                          (default 42)
    """

    def __init__(self, cfg):
        raw_sizes      = getattr(cfg.dataset, 'comm_sizes',     [12, 18])
        comm_sizes     = tuple(int(x) for x in raw_sizes)
        p_intra_dense  = getattr(cfg.dataset, 'p_intra_dense',  0.70)
        p_intra_sparse = getattr(cfg.dataset, 'p_intra_sparse', 0.25)
        p_inter        = getattr(cfg.dataset, 'p_inter',        0.03)
        n_graphs       = getattr(cfg.dataset, 'n_graphs',       400)
        seed           = getattr(cfg.dataset, 'seed',           42)

        p_intra = (p_intra_dense, p_intra_sparse)
        n0, n1  = comm_sizes

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
        data_dir  = os.path.join(base_path, cfg.dataset.datadir)
        os.makedirs(data_dir, exist_ok=True)

        fname = (f'dcsbm_n{n0}_{n1}_pi{int(p_intra_dense * 100)}'
                 f'_{int(p_intra_sparse * 100)}'
                 f'_pinter{int(p_inter * 100)}_N{n_graphs}_seed{seed}.pt')
        cache_path = os.path.join(data_dir, fname)

        if os.path.exists(cache_path):
            print(f"Loading cached DC-SBM dataset from {cache_path}")
            pairs = torch.load(cache_path)
        else:
            print(f"Generating DC-SBM "
                  f"(comm_sizes={comm_sizes}, p_intra={p_intra}, "
                  f"p_inter={p_inter}, N={n_graphs})...")
            nx_graphs, comm_labels = _generate_dcsbm_graphs(
                n_graphs, comm_sizes, p_intra, p_inter, seed)
            pairs = [(_nx_to_pyg(G), cl)
                     for G, cl in zip(nx_graphs, comm_labels)]
            torch.save(pairs, cache_path)
            print(f"  Saved to {cache_path}")
            stats_dir = os.path.join(data_dir, 'stats')
            os.makedirs(stats_dir, exist_ok=True)
            _visualize_dcsbm_stats(nx_graphs, comm_labels, comm_sizes,
                                   p_intra, p_inter, stats_dir)

        # 70 / 15 / 15 split
        n_total = len(pairs)
        n_test  = max(1, int(round(n_total * 0.15)))
        n_val   = max(1, int(round(n_total * 0.15)))
        n_train = n_total - n_val - n_test
        rng = random.Random(seed)
        idx = list(range(n_total))
        rng.shuffle(idx)

        self.train_dataset = DCSBMDataset(
            [pairs[i][0] for i in idx[:n_train]])
        self.val_dataset   = DCSBMDataset(
            [pairs[i][0] for i in idx[n_train:n_train + n_val]])
        self.test_dataset  = DCSBMDataset(
            [pairs[i][0] for i in idx[n_train + n_val:]])

        self.batch_size  = cfg.train.batch_size
        self.num_workers = getattr(cfg.train,   'num_workers', 0)
        self.pin_memory  = getattr(cfg.dataset, 'pin_memory',  False)
        self.dataset_infos = SpectreDatasetInfos(self, cfg.dataset)
        self.inner = self.train_dataset

        print(f"  DCSBM: train={len(self.train_dataset)}  "
              f"val={len(self.val_dataset)}  test={len(self.test_dataset)}")


# ==============================================================================
#  EDGE Network Datasets: Community and Ego
#  Graph collections loaded from pickle files (originally from EDGE codebase).
#  Place the .pkl files under DigressMinimal/data/community/ and data/ego/.
#  60 / 20 / 20 train / val / test split.
# ==============================================================================

import pickle as _pkl   # avoid top-level name clash


class _EdgeNetworkDataset(Dataset):
    """Simple wrapper around a list of pre-converted PyG Data objects."""

    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


class _EdgeNetworkDataModule(_SyntheticDataModuleBase):
    """DataModule for graph collections loaded from an EDGE-style pickle file.

    The pickle file must contain a list of networkx.Graph objects.
    Data directory and pkl filename are set by subclasses.

    Splits: 60 % train / 20 % val / 20 % test (fixed seed).

    Config keys (under dataset):
        datadir:    path relative to DigressMinimal root  (e.g. 'data/community')
        pkl_file:   filename of the pickle inside datadir  (e.g. 'community.pkl')
        seed:       RNG seed for the split                 (default 42)
    """

    _dataset_label: str = 'EdgeNetwork'  # overridden by subclasses

    def __init__(self, cfg):
        seed = getattr(cfg.dataset, 'seed', 42)

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[1]
        data_dir  = os.path.join(base_path, cfg.dataset.datadir)
        pkl_name  = cfg.dataset.pkl_file

        # Search order:
        #   1. configured datadir  (e.g. data/community/community.pkl)
        #   2. EDGE graphs directory, two levels up from DigressMinimal root
        #      (../graph-generation-EDGE-main/graphs/)
        #   3. EDGE graphs directory, one level up (same parent level)
        _candidates = [
            os.path.join(data_dir, pkl_name),
            os.path.join(str(base_path.parent),
                         'graph-generation-EDGE-main', 'graphs', pkl_name),
            os.path.join(str(base_path.parent.parent),
                         'graph-generation-EDGE-main', 'graphs', pkl_name),
        ]
        pkl_path = None
        for _c in _candidates:
            if os.path.exists(_c):
                pkl_path = _c
                break

        if pkl_path is None:
            raise FileNotFoundError(
                f"[{self._dataset_label}] Could not find '{pkl_name}'. "
                f"Searched:\n" +
                "\n".join(f"  {c}" for c in _candidates) +
                f"\nCopy the file from graph-generation-EDGE-main/graphs/ "
                f"into {data_dir}/")

        print(f"[{self._dataset_label}] Loading graphs from {pkl_path} ...")
        with open(pkl_path, 'rb') as fh:
            nx_graphs = _pkl.load(fh)

        # Filter out empty / trivial graphs
        nx_graphs = [G for G in nx_graphs if G.number_of_nodes() >= 2
                     and G.number_of_edges() >= 1]

        print(f"[{self._dataset_label}] Loaded {len(nx_graphs)} valid graphs.")

        # 60 / 20 / 20 split
        n_total = len(nx_graphs)
        n_test  = max(1, int(round(n_total * 0.20)))
        n_val   = max(1, int(round(n_total * 0.20)))
        n_train = n_total - n_val - n_test
        rng = random.Random(seed)
        idx = list(range(n_total))
        rng.shuffle(idx)

        train_nx = [nx_graphs[i] for i in idx[:n_train]]
        val_nx   = [nx_graphs[i] for i in idx[n_train:n_train + n_val]]
        test_nx  = [nx_graphs[i] for i in idx[n_train + n_val:]]

        self.train_dataset = _EdgeNetworkDataset([_nx_to_pyg(G) for G in train_nx])
        self.val_dataset   = _EdgeNetworkDataset([_nx_to_pyg(G) for G in val_nx])
        self.test_dataset  = _EdgeNetworkDataset([_nx_to_pyg(G) for G in test_nx])

        self.batch_size  = cfg.train.batch_size
        self.num_workers = getattr(cfg.train,   'num_workers', 0)
        self.pin_memory  = getattr(cfg.dataset, 'pin_memory',  False)
        self.dataset_infos = SpectreDatasetInfos(self, cfg.dataset)
        self.inner = self.train_dataset

        print(f"  [{self._dataset_label}] train={len(self.train_dataset)}  "
              f"val={len(self.val_dataset)}  test={len(self.test_dataset)}")


class CommunityDataModule(_EdgeNetworkDataModule):
    """DataModule for the EDGE Community graph collection.

    Default config expects:
        dataset.datadir  = 'data/community'
        dataset.pkl_file = 'community.pkl'
    """
    _dataset_label = 'Community'

    def __init__(self, cfg):
        cfg.dataset.name = 'community'
        if not hasattr(cfg.dataset, 'pkl_file'):
            cfg.dataset.pkl_file = 'community.pkl'
        super().__init__(cfg)


class EgoDataModule(_EdgeNetworkDataModule):
    """DataModule for the EDGE Ego graph collection.

    Default config expects:
        dataset.datadir  = 'data/ego'
        dataset.pkl_file = 'Ego.pkl'
    """
    _dataset_label = 'Ego'

    def __init__(self, cfg):
        cfg.dataset.name = 'ego'
        if not hasattr(cfg.dataset, 'pkl_file'):
            cfg.dataset.pkl_file = 'Ego.pkl'
        super().__init__(cfg)