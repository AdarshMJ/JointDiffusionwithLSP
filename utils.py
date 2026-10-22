"""
Utility functions for DigressMinimal.
Adapted from DiGress to remove pytorch-lightning and other heavy dependencies.
"""
import os
import json
from datetime import datetime
import torch
import torch_geometric.utils
from torch_geometric.utils import to_dense_adj, to_dense_batch


def create_folders(cfg):
    """Create output directories."""
    os.makedirs(f'graphs/{cfg.name}', exist_ok=True)
    os.makedirs(f'chains/{cfg.name}', exist_ok=True)
    os.makedirs(f'checkpoints/{cfg.name}', exist_ok=True)


class PlaceHolder:
    """Container for graph data (X, E, y)."""
    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: torch.Tensor):
        """Changes the device and dtype of X, E, y."""
        self.X = self.X.type_as(x)
        self.E = self.E.type_as(x)
        self.y = self.y.type_as(x)
        return self

    def mask(self, node_mask, collapse=False):
        """Apply node mask to X, E, y."""
        x_mask = node_mask.unsqueeze(-1)          # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)             # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)             # bs, 1, n, 1

        if collapse:
            self.X = torch.argmax(self.X, dim=-1)
            self.E = torch.argmax(self.E, dim=-1)

            self.X[node_mask == 0] = -1
            self.E[(e_mask1 * e_mask2).squeeze(-1) == 0] = -1
        else:
            self.X = self.X * x_mask
            self.E = self.E * e_mask1 * e_mask2
            assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
        return self


def to_dense(x, edge_index, edge_attr, batch):
    """Convert sparse graph to dense representation."""
    X, node_mask = to_dense_batch(x=x, batch=batch)
    edge_index, edge_attr = torch_geometric.utils.remove_self_loops(edge_index, edge_attr)
    max_num_nodes = X.size(1)
    E = to_dense_adj(edge_index=edge_index, batch=batch, edge_attr=edge_attr, max_num_nodes=max_num_nodes)
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask


def encode_no_edge(E):
    """Encode no-edge as first class."""
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = torch.sum(E, dim=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt[no_edge] = 1
    E[:, :, :, 0] = first_elt
    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0
    return E


def normalize(X, E, y, norm_values, norm_biases, node_mask):
    """Normalize features."""
    X = (X - norm_biases[0]) / norm_values[0]
    E = (E - norm_biases[1]) / norm_values[1]
    y = (y - norm_biases[2]) / norm_values[2]

    diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
    E[diag] = 0

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask)


def unnormalize(X, E, y, norm_values, norm_biases, node_mask, collapse=False):
    """Unnormalize features."""
    X = (X * norm_values[0] + norm_biases[0])
    E = (E * norm_values[1] + norm_biases[1])
    y = y * norm_values[2] + norm_biases[2]

    return PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse)


def log_metrics_to_file(metrics_dict, log_dir, experiment_name, epoch=None, mode='sampling'):
    """Log metrics to a JSON file with timestamp.
    
    Args:
        metrics_dict: Dictionary of metrics to log
        log_dir: Directory to save log files
        experiment_name: Name of the experiment
        epoch: Optional epoch number
        mode: Mode of evaluation ('sampling', 'test', etc.)
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Prepare log entry
    log_entry = {
        'timestamp': timestamp,
        'experiment': experiment_name,
        'mode': mode,
        'metrics': metrics_dict
    }
    
    if epoch is not None:
        log_entry['epoch'] = epoch
    
    # Save to JSON file
    log_file = os.path.join(log_dir, f'{experiment_name}_{mode}_metrics.json')
    
    # Load existing logs if file exists
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []
    else:
        logs = []
    
    # Append new entry
    logs.append(log_entry)
    
    # Save updated logs
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)
    
    # Also create a human-readable text log
    txt_log_file = os.path.join(log_dir, f'{experiment_name}_{mode}_metrics.txt')
    with open(txt_log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Mode: {mode}\n")
        if epoch is not None:
            f.write(f"Epoch: {epoch}\n")
        f.write(f"{'-'*80}\n")
        f.write("Metrics:\n")
        _write_metrics_recursive(f, metrics_dict, indent=2)
        f.write(f"{'='*80}\n\n")
    
    print(f"\nMetrics logged to:")
    print(f"  JSON: {log_file}")
    print(f"  TXT:  {txt_log_file}")
    
    return log_file


def _write_metrics_recursive(file, metrics, indent=0):
    """Helper to write metrics recursively for nested dicts."""
    indent_str = " " * indent
    for key, value in metrics.items():
        if isinstance(value, dict):
            file.write(f"{indent_str}{key}:\n")
            _write_metrics_recursive(file, value, indent + 2)
        elif isinstance(value, float):
            file.write(f"{indent_str}{key}: {value:.6f}\n")
        else:
            file.write(f"{indent_str}{key}: {value}\n")
