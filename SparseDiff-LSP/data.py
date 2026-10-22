"""
Dataset loaders for SparseDiffLSP.
Supports: ego, comm20, node_XX (CoraChameleon subgraphs).
"""

import os
import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torch_geometric.data
import torch_geometric.utils


# ---------------------------------------------------------------------------
#  Node count distribution (for sampling n during generation)
# ---------------------------------------------------------------------------
class DistributionNodes:
    """Empirical distribution over node counts."""

    def __init__(self, histogram):
        self.n_nodes = histogram
        self.prob = histogram / histogram.sum()

    def sample_n(self, batch_size, device):
        idx = torch.multinomial(self.prob, batch_size, replacement=True)
        return idx.to(device)


# ---------------------------------------------------------------------------
#  Spectre graphs: Ego, Comm20 — loaded from local .pkl files
# ---------------------------------------------------------------------------

def _nx_to_pyg(G):
    """Convert a NetworkX graph to a PyG Data object."""
    n = G.number_of_nodes()
    X = torch.ones(n, 1, dtype=torch.float)
    y = torch.zeros([1, 0]).float()
    adj = torch.zeros(n, n)
    mapping = {node: i for i, node in enumerate(G.nodes())}
    for u, v in G.edges():
        adj[mapping[u], mapping[v]] = 1.0
        adj[mapping[v], mapping[u]] = 1.0
    edge_index, _ = torch_geometric.utils.dense_to_sparse(adj)
    edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
    edge_attr[:, 1] = 1
    return torch_geometric.data.Data(
        x=X, edge_index=edge_index, edge_attr=edge_attr,
        y=y, n_nodes=torch.tensor([n], dtype=torch.long),
    )


class SpectreDataModule:
    """DataModule for Spectre graphs (ego, comm20) loaded from .pkl files."""

    _pkl_names = {
        "ego": "Ego.pkl",
        "comm20": "community.pkl",
    }

    def __init__(self, cfg):
        import pickle

        self.cfg = cfg
        dataset_name = cfg.dataset.name
        base_path = pathlib.Path(os.path.realpath(__file__)).parent
        pkl_file = self._pkl_names.get(dataset_name)
        if pkl_file is None:
            raise ValueError(f"Unknown Spectre dataset: {dataset_name}")

        pkl_path = os.path.join(base_path, cfg.dataset.datadir, pkl_file)
        print(f"Loading {dataset_name} from {pkl_path}")

        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Not found: {pkl_path}")

        with open(pkl_path, "rb") as f:
            nx_graphs = pickle.load(f)

        # Convert all NetworkX graphs to PyG Data
        all_data = [_nx_to_pyg(G) for G in nx_graphs]
        print(f"  Loaded {len(all_data)} graphs")

        # Deterministic split: 80% train, 10% val, 10% test
        g = torch.Generator()
        g.manual_seed(0)
        n_total = len(all_data)
        indices = torch.randperm(n_total, generator=g).tolist()

        test_len = int(round(n_total * 0.1))
        val_len = int(round(n_total * 0.1))
        train_len = n_total - val_len - test_len

        train_data = [all_data[i] for i in indices[:train_len]]
        val_data = [all_data[i] for i in indices[train_len:train_len + val_len]]
        test_data = [all_data[i] for i in indices[train_len + val_len:]]

        print(f"  Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

        self.train_dataset = train_data
        self.val_dataset = val_data
        self.test_dataset = test_data
        self.batch_size = cfg.train.batch_size
        self.num_workers = getattr(cfg.train, "num_workers", 0)

    def train_dataloader(self):
        from torch_geometric.loader import DataLoader
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
        )

    def val_dataloader(self):
        from torch_geometric.loader import DataLoader
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
        )

    def test_dataloader(self):
        from torch_geometric.loader import DataLoader
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
        )


# ---------------------------------------------------------------------------
#  CoraChameleon node_XX datasets
# ---------------------------------------------------------------------------
class CoraChameleonDataset(Dataset):
    """Dataset for CoraChameleon subgraph .pt files."""

    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data, metadata = self.data_list[idx]
        n_nodes = data.num_nodes
        X = torch.ones(n_nodes, 1, dtype=torch.float)
        edge_index = torch_geometric.utils.to_undirected(data.edge_index)
        edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float)
        edge_attr[:, 1] = 1
        y = torch.zeros([1, 0]).float()
        return torch_geometric.data.Data(
            x=X, edge_index=edge_index, edge_attr=edge_attr,
            y=y, n_nodes=torch.tensor([n_nodes], dtype=torch.long),
        )


class CoraChameleonDataModule:
    """DataModule for CoraChameleon node_XX datasets."""

    def __init__(self, cfg):
        self.cfg = cfg
        node_size = cfg.dataset.node_size
        split_type = getattr(cfg.dataset, "split_type", "S1")
        num_train_graphs = getattr(cfg.dataset, "num_train_graphs", 0)

        base_path = pathlib.Path(os.path.realpath(__file__)).parent
        data_folder = os.path.join(base_path, cfg.dataset.datadir, f"node_{node_size}")

        train_path = os.path.join(data_folder, f"{split_type}.pt")
        test_path = os.path.join(data_folder, "test.pt")

        print(f"Loading CoraChameleon: node_{node_size}, split={split_type}")
        print(f"  train: {train_path}")
        print(f"  test:  {test_path}")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Not found: {train_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Not found: {test_path}")

        train_raw = torch.load(train_path, weights_only=False)
        test_raw = torch.load(test_path, weights_only=False)

        if 0 < num_train_graphs < len(train_raw):
            train_raw = train_raw[:num_train_graphs]

        # Split test -> val (60) + test (rest)
        val_raw = test_raw[:60]
        test_raw = test_raw[60:]

        print(f"  Train: {len(train_raw)}, Val: {len(val_raw)}, Test: {len(test_raw)}")

        self.train_dataset = CoraChameleonDataset(train_raw)
        self.val_dataset = CoraChameleonDataset(val_raw)
        self.test_dataset = CoraChameleonDataset(test_raw)

        self.batch_size = cfg.train.batch_size
        self.num_workers = getattr(cfg.train, "num_workers", 0)

    def train_dataloader(self):
        from torch_geometric.loader import DataLoader
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
        )

    def val_dataloader(self):
        from torch_geometric.loader import DataLoader
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
        )

    def test_dataloader(self):
        from torch_geometric.loader import DataLoader
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
        )


# ---------------------------------------------------------------------------
#  Factory + dataset stats
# ---------------------------------------------------------------------------
def create_datamodule(cfg):
    """Create the appropriate DataModule based on config."""
    name = cfg.dataset.name
    if name in ("ego", "comm20"):
        return SpectreDataModule(cfg)
    elif name.startswith("node_"):
        return CoraChameleonDataModule(cfg)
    else:
        raise ValueError(f"Unknown dataset: {name}. Use 'ego', 'comm20', or 'node_XX'.")


def compute_dataset_stats(train_loader):
    """Compute (rho_data, n_ref, degree_sequences) from training data.

    Required for LSP calibration.
    """
    densities = []
    node_counts = []
    degree_sequences = []

    for data in train_loader:
        # Get batch sizes
        batch = data.batch
        ptr = data.ptr if hasattr(data, "ptr") else None

        if ptr is None:
            # Compute ptr from batch
            unique, counts = torch.unique(batch, return_counts=True)
            ptr = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])

        bs = ptr.shape[0] - 1
        for i in range(bs):
            start = ptr[i].item()
            end = ptr[i + 1].item()
            n_i = end - start

            if n_i <= 1:
                continue

            # Get edges for this graph
            edge_mask = (data.edge_index[0] >= start) & (data.edge_index[0] < end)
            edges_i = data.edge_index[:, edge_mask] - start

            # Compute degrees
            degrees = torch.zeros(n_i, dtype=torch.float)
            if edges_i.shape[1] > 0:
                src = edges_i[0]
                degrees.scatter_add_(0, src, torch.ones(src.shape[0]))

            # Density (undirected: each edge counted twice in edge_index)
            n_edges = edges_i.shape[1] / 2.0
            density = 2.0 * n_edges / (n_i * (n_i - 1)) if n_i > 1 else 0.0

            densities.append(density)
            node_counts.append(n_i)
            degree_sequences.append(degrees.numpy())

    if not densities:
        return 0.5, 10, []

    rho_data = sum(densities) / len(densities)
    n_ref = round(sum(node_counts) / len(node_counts))
    return rho_data, n_ref, degree_sequences


def compute_node_count_distribution(train_loader, max_n=None):
    """Compute histogram of node counts from training data."""
    all_n = []
    for data in train_loader:
        batch = data.batch
        unique, counts = torch.unique(batch, return_counts=True)
        all_n.extend(counts.tolist())

    if max_n is None:
        max_n = max(all_n) + 1

    hist = torch.zeros(max_n)
    for n in all_n:
        if n < max_n:
            hist[n] += 1

    return hist
