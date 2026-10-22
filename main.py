"""
Main training script for DiGress Minimal.
Pure PyTorch implementation without pytorch-lightning.
"""
import os
import sys
import argparse
import yaml
import time
from datetime import datetime
import numpy as np
import networkx as nx
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from diffusion_model import DiscreteDenoisingDiffusion
from datasets.spectre_dataset import (
    SpectreGraphDataModule, SpectreDatasetInfos,
    Comm20DataModule, SBMDataModule, PlanarDataModule, CoraChameleonDataModule,
    TwoDensityMixtureDataModule, DCSBMDataModule,
    CommunityDataModule, EgoDataModule,
)
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from analysis.spectre_utils import (
    SpectreSamplingMetrics, PlanarSamplingMetrics, SBMSamplingMetrics,
    Comm20SamplingMetrics, CoraChameleonSamplingMetrics,
    TwoDensitySamplingMetrics, DCSBMSamplingMetrics,
    CommunitySamplingMetrics, EgoSamplingMetrics,
)
from analysis.visualization import NonMolecularVisualization
from metrics.train_metrics import TrainLossDiscrete
import utils


# ------------------------------------------------------------------
# Logging: tee all stdout to a single .log file
# ------------------------------------------------------------------
class TeeLogger:
    """Tees sys.stdout to both the terminal and a .log file."""
    def __init__(self, log_path):
        self.terminal = sys.__stdout__
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.logfile = open(log_path, 'a', buffering=1)

    def write(self, msg):
        self.terminal.write(msg)
        self.logfile.write(msg)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def close(self):
        sys.stdout = self.terminal
        self.logfile.close()


def setup_logging(experiment_dir, name):
    log_path = os.path.join(experiment_dir, 'logs', f'{name}.log')
    tee = TeeLogger(log_path)
    sys.stdout = tee
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Logging to: {log_path}")
    return tee


# ------------------------------------------------------------------
# Graph statistics comparison (mirrors main_crm.py)
# ------------------------------------------------------------------
def compute_expected_overlap(graphs):
    """Compute mean Chung-Lu expected self-overlap from degree sequences.

    overlap_CL(G) = (sum_i d_i^2)^2 / vol^3

    Permutation-invariant upper bound on self-overlap.  Lower value means
    the generator produces diverse structures without degree-concentrated
    memorisation.
    """
    values = []
    for G in graphs:
        if G.number_of_nodes() == 0:
            continue
        degrees = np.array([d for _, d in G.degree()], dtype=float)
        vol = degrees.sum()
        if vol < 1e-9:
            continue
        sq_sum = float((degrees ** 2).sum())
        values.append((sq_sum ** 2) / (vol ** 3))
    if not values:
        return 0.0, 0.0
    arr = np.array(values)
    return float(arr.mean()), float(arr.std())


def compute_graph_stats_summary(graphs):
    records = {k: [] for k in [
        'nodes', 'edges', 'density', 'triangles',
        'avg_clustering', 'n_communities', 'max_degree']}
    for G in graphs:
        if G.number_of_nodes() == 0:
            continue
        n = G.number_of_nodes()
        e = G.number_of_edges()
        records['nodes'].append(n)
        records['edges'].append(e)
        records['density'].append(float(nx.density(G)))
        records['triangles'].append(sum(nx.triangles(G).values()) // 3)
        records['avg_clustering'].append(float(nx.average_clustering(G)))
        degrees = [d for _, d in G.degree()]
        records['max_degree'].append(max(degrees) if degrees else 0)
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            comms = list(greedy_modularity_communities(G))
            records['n_communities'].append(len(comms))
        except Exception:
            records['n_communities'].append(1)
    summary = {}
    for key, vals in records.items():
        arr = np.array(vals, dtype=float)
        if len(arr) > 0:
            summary[key] = {'mean': float(arr.mean()), 'std': float(arr.std()),
                            'min': float(arr.min()), 'max': float(arr.max())}
        else:
            summary[key] = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
    return summary


# ------------------------------------------------------------------
# Power-law exponent (PLE) and characteristic path length (CPL)
# ------------------------------------------------------------------

def _power_law_exponent(G):
    """MLE estimate of the discrete power-law exponent (Clauset et al. 2009).
    Uses all nodes with degree >= 1.  Returns nan for trivially small graphs."""
    degrees = np.array([d for _, d in G.degree() if d > 0], dtype=float)
    if len(degrees) < 2:
        return float('nan')
    # MLE: alpha = 1 + n / sum(ln(k_i / (k_min - 0.5)))   with k_min = 1
    denom = np.sum(np.log(degrees / 0.5))
    if denom < 1e-10:
        return float('nan')
    return float(1.0 + len(degrees) / denom)


def _characteristic_path_length(G):
    """Average shortest-path length over the largest connected component."""
    if G.number_of_nodes() < 2:
        return float('nan')
    lcc_nodes = max(nx.connected_components(G), key=len)
    H = G.subgraph(lcc_nodes)
    if H.number_of_nodes() < 2:
        return float('nan')
    return float(nx.average_shortest_path_length(H))


def compute_ple_cpl_stats(graphs):
    """Return mean/std of PLE and CPL over a list of NetworkX graphs."""
    ples, cpls = [], []
    for G in graphs:
        ples.append(_power_law_exponent(G))
        cpls.append(_characteristic_path_length(G))
    ples = [v for v in ples if not (isinstance(v, float) and np.isnan(v))]
    cpls = [v for v in cpls if not (isinstance(v, float) and np.isnan(v))]
    return {
        'ple_mean': float(np.mean(ples)) if ples else float('nan'),
        'ple_std':  float(np.std(ples))  if ples else float('nan'),
        'cpl_mean': float(np.mean(cpls)) if cpls else float('nan'),
        'cpl_std':  float(np.std(cpls))  if cpls else float('nan'),
    }


def compare_graph_stats(gen_graphs, ref_graphs, ref_label):
    gen_stats = compute_graph_stats_summary(gen_graphs)
    ref_stats = compute_graph_stats_summary(ref_graphs)
    KEYS = ['nodes', 'edges', 'density', 'triangles',
            'avg_clustering', 'n_communities', 'max_degree']
    W = 70
    print(f"\n{'='*W}")
    print(f"  Graph Statistics: Generated ({len(gen_graphs)}) vs {ref_label} ({len(ref_graphs)})")
    print(f"{'='*W}")
    print(f"  {'Statistic':<22}  {'Generated (mean±std)':<26}  {ref_label+' (mean±std)':<26}  |diff|")
    print(f"  {'-'*22}  {'-'*26}  {'-'*26}  {'-'*8}")
    gen_vec, ref_vec = [], []
    for key in KEYS:
        gm, gs = gen_stats[key]['mean'], gen_stats[key]['std']
        rm, rs = ref_stats[key]['mean'], ref_stats[key]['std']
        gen_vec.append(gm)
        ref_vec.append(rm)
        diff = abs(gm - rm)
        print(f"  {key:<22}  {gm:>8.4f} \u00b1 {gs:<8.4f}    "
              f"{rm:>8.4f} \u00b1 {rs:<8.4f}    {diff:.4f}")
    gen_vec = np.array(gen_vec)
    ref_vec = np.array(ref_vec)
    mse = float(np.mean((gen_vec - ref_vec) ** 2))
    mae = float(np.mean(np.abs(gen_vec - ref_vec)))
    print(f"\n  MSE (stats vector): {mse:.6f}")
    print(f"  MAE (stats vector): {mae:.6f}")
    print(f"{'='*W}\n")
    return {'stats_mse': mse, 'stats_mae': mae,
            'gen_stats': gen_stats, 'ref_stats': ref_stats}


class DotDict(dict):
    """Dot notation access to dictionary attributes."""
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")
    
    def __setattr__(self, attr, value):
        self[attr] = value
    
    @staticmethod
    def from_dict(d):
        if isinstance(d, dict):
            return DotDict({k: DotDict.from_dict(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [DotDict.from_dict(item) for item in d]
        else:
            return d


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return DotDict.from_dict(config)


def create_datamodule(cfg):
    """Create the appropriate data module based on config."""
    dataset_name = cfg.dataset.name
    
    if dataset_name == 'sbm':
        return SBMDataModule(cfg)
    elif dataset_name == 'planar':
        return PlanarDataModule(cfg)
    elif dataset_name == 'comm20':
        return Comm20DataModule(cfg)
    elif dataset_name == 'corachameleon':
        return CoraChameleonDataModule(cfg)
    elif dataset_name == 'two_density':
        return TwoDensityMixtureDataModule(cfg)
    elif dataset_name == 'dcsbm':
        return DCSBMDataModule(cfg)
    elif dataset_name == 'community':
        return CommunityDataModule(cfg)
    elif dataset_name == 'ego':
        return EgoDataModule(cfg)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_sampling_metrics(cfg, datamodule):
    """Create the appropriate sampling metrics based on dataset."""
    dataset_name = cfg.dataset.name
    
    if dataset_name == 'sbm':
        return SBMSamplingMetrics(datamodule)
    elif dataset_name == 'planar':
        return PlanarSamplingMetrics(datamodule)
    elif dataset_name == 'comm20':
        return Comm20SamplingMetrics(datamodule)
    elif dataset_name == 'corachameleon':
        return CoraChameleonSamplingMetrics(datamodule)
    elif dataset_name == 'two_density':
        return TwoDensitySamplingMetrics(datamodule)
    elif dataset_name == 'dcsbm':
        return DCSBMSamplingMetrics(datamodule)
    elif dataset_name == 'community':
        return CommunitySamplingMetrics(datamodule)
    elif dataset_name == 'ego':
        return EgoSamplingMetrics(datamodule)
    else:
        return SpectreSamplingMetrics(datamodule)


def train_epoch(model, train_loader, optimizer, device, epoch, log_every_steps=50):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        
        step = batch_idx + epoch * len(train_loader)
        loss = model.training_step(data, log_every_steps=log_every_steps, step=step)
        
        if loss is None:
            continue
            
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % log_every_steps == 0:
            print(f'Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, val_loader, device):
    """Validate the model."""
    model.eval()
    model.reset_metrics(test=False)
    
    total_nll = 0
    num_batches = 0
    
    for data in val_loader:
        data = data.to(device)
        nll = model.validation_step(data)
        total_nll += nll if isinstance(nll, float) else nll.item()
        num_batches += 1
    
    return total_nll / max(num_batches, 1)


@torch.no_grad()
def test(model, test_loader, device):
    """Test the model."""
    model.eval()
    model.reset_metrics(test=True)
    
    total_nll = 0
    num_batches = 0
    
    for data in test_loader:
        data = data.to(device)
        nll = model.test_step(data)
        total_nll += nll if isinstance(nll, float) else nll.item()
        num_batches += 1
    
    return total_nll / max(num_batches, 1)


@torch.no_grad()
def sample_and_evaluate(model, cfg, datamodule, sampling_metrics, visualization, device, epoch=0):
    """Generate samples, compute MMD metrics and graph statistics comparison."""
    model.eval()

    n_samples = cfg.general.final_model_samples_to_generate
    batch_size = cfg.train.batch_size

    all_samples = []
    batch_id = 0
    while len(all_samples) < n_samples:
        num_this = min(batch_size, n_samples - len(all_samples))
        samples = model.sample_batch(
            batch_id=batch_id,
            batch_size=num_this,
            keep_chain=0,
            number_chain_steps=cfg.general.number_chain_steps,
            save_final=0
        )
        all_samples.extend(samples)
        batch_id += 1
        print(f'Generated {len(all_samples)}/{n_samples} samples')

    # Convert generated samples to NetworkX graphs
    gen_nx = []
    for graph in all_samples:
        node_types, edge_types = graph
        A = edge_types.bool().cpu().numpy()
        gen_nx.append(nx.from_numpy_array(A))

    if sampling_metrics is not None:
        train_graphs = sampling_metrics.train_graphs
        test_graphs  = sampling_metrics.test_graphs

        # ---- Graph Statistics ----------------------------------------
        print("\n" + "#" * 80)
        print("#  GRAPH STATISTICS COMPARISON")
        print("#" * 80)
        stats_vs_train = compare_graph_stats(gen_nx, train_graphs, "TRAIN")
        stats_vs_test  = compare_graph_stats(gen_nx, test_graphs,  "TEST")

        # ---- MMD Metrics ---------------------------------------------
        print("\n" + "#" * 80)
        print("#  MMD DISTRIBUTION METRICS: Generated vs TRAIN")
        print("#" * 80)
        metrics_train = sampling_metrics(
            generated_graphs=all_samples, name=cfg.general.name,
            current_epoch=epoch, val_counter=0, test=False, use_train=True)
        print("\nMMD vs TRAIN:")
        for k, v in metrics_train.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

        print("\n" + "#" * 80)
        print("#  MMD DISTRIBUTION METRICS: Generated vs TEST")
        print("#" * 80)
        metrics_test = sampling_metrics(
            generated_graphs=all_samples, name=cfg.general.name,
            current_epoch=epoch, val_counter=0, test=True)
        print("\nMMD vs TEST:")
        for k, v in metrics_test.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

        # ---- PLE / CPL Stats ----------------------------------------
        print("\n" + "#" * 80)
        print("#  DEGREE POWER-LAW EXPONENT (PLE) + CHARACTERISTIC PATH LENGTH (CPL)")
        print("#" * 80)
        print("  (CPL: LCC of each graph; graphs where LCC < 2 nodes are excluded from MMD)")
        print("  (PLE: MLE estimate; graphs with < 2 non-isolated nodes are excluded from MMD)")
        print("  (Triangle MMD: fraction of C(n,3) possible triangles \u2014 size-invariant, in [0,1])")
        print("  (Assortativity: undefined graphs excluded. All four use median-heuristic \u03c3)")
        ple_cpl_gen   = compute_ple_cpl_stats(gen_nx)
        ple_cpl_train = compute_ple_cpl_stats(train_graphs)
        ple_cpl_test  = compute_ple_cpl_stats(test_graphs)
        _pm = '\u00b1'
        print(f"  {'Set':<12}  {'PLE (mean'+_pm+'std)':<26}  {'CPL (mean'+_pm+'std)':<26}")
        print(f"  {'-'*12}  {'-'*26}  {'-'*26}")
        for _lbl, _st in [('Generated', ple_cpl_gen),
                          ('Train',     ple_cpl_train),
                          ('Test',      ple_cpl_test)]:
            print(f"  {_lbl:<12}  "
                  f"{_st['ple_mean']:>8.4f} \u00b1 {_st['ple_std']:<8.4f}    "
                  f"{_st['cpl_mean']:>8.4f} \u00b1 {_st['cpl_std']:<8.4f}")
        print("  (MMD\u00b2 ratios r_ple / r_cpl are reported in the MMD metrics above)")

        # ---- Triangle-Overlap Analysis ------------------------------
        print("\n" + "#" * 80)
        print("#  TRIANGLE-OVERLAP FRONTIER")
        print("#" * 80)
        gen_overlap_mean,   gen_overlap_std   = compute_expected_overlap(gen_nx)
        train_overlap_mean, train_overlap_std = compute_expected_overlap(train_graphs)
        test_overlap_mean,  test_overlap_std  = compute_expected_overlap(test_graphs)
        gen_tri   = compute_graph_stats_summary(gen_nx)['triangles']['mean']
        train_tri = compute_graph_stats_summary(train_graphs)['triangles']['mean']
        test_tri  = compute_graph_stats_summary(test_graphs)['triangles']['mean']
        print(f"  {'Set':<12}  {'CL Expected Overlap (mean±std)':<36}  {'Avg Triangles'}")
        print(f"  {'-'*12}  {'-'*36}  {'-'*14}")
        print(f"  {'Generated':<12}  {gen_overlap_mean:>10.6f} ± {gen_overlap_std:<10.6f}    {gen_tri:.2f}")
        print(f"  {'Train':<12}  {train_overlap_mean:>10.6f} ± {train_overlap_std:<10.6f}    {train_tri:.2f}")
        print(f"  {'Test':<12}  {test_overlap_mean:>10.6f} ± {test_overlap_std:<10.6f}    {test_tri:.2f}")
        print()
        print("  Interpretation: lower CL-overlap + higher triangles than edge-independent")
        print("  baselines indicates the model escapes the triangle-overlap frontier.")

        # ---- Summary ------------------------------------------------
        print("\n" + "#" * 80)
        print("#  EVALUATION SUMMARY")
        print("#" * 80)
        for label, stats, mmd in [
            ("vs TRAIN", stats_vs_train, metrics_train),
            ("vs TEST",  stats_vs_test,  metrics_test),
        ]:
            print(f"\n  [{label}]")
            print(f"    Stats MSE : {stats['stats_mse']:.6f}")
            print(f"    Stats MAE : {stats['stats_mae']:.6f}")
            # Core MMD metrics
            for k, v in mmd.items():
                if isinstance(v, float) and not k.startswith('wl_') and not k.startswith('r_') \
                        and k not in ('frac_unique', 'frac_novel'):
                    print(f"    {k}: {v:.4f}")
            # Uniqueness and novelty
            if mmd.get('frac_unique') is not None:
                print(f"    Uniqueness : {mmd['frac_unique']:.4f}")
            if mmd.get('frac_novel') is not None:
                print(f"    Novelty    : {mmd['frac_novel']:.4f}")
            # MMD ratios — only populated for vs TEST
            ratio_keys = [k for k in mmd if k.startswith('r_')]
            if ratio_keys:
                print("    MMD² ratios (r = MMD²(gen,test) / MMD²(train,test)):")
                for k in sorted(ratio_keys):
                    v = mmd[k]
                    s = f"{v:.4f}" if not (isinstance(v, float) and v != v) else "nan"
                    print(f"      {k}: {s}")
        print()

        # ---- Persist to log files -----------------------------------
        log_dir = f'{cfg.general.experiment_dir}/logs'
        utils.log_metrics_to_file(
            metrics_dict={f"test/{k}": v for k, v in metrics_test.items()},
            log_dir=log_dir, experiment_name=cfg.general.name,
            epoch=epoch, mode='sampling_vs_test')
        utils.log_metrics_to_file(
            metrics_dict={f"train/{k}": v for k, v in metrics_train.items()},
            log_dir=log_dir, experiment_name=cfg.general.name,
            epoch=epoch, mode='sampling_vs_train')
        utils.log_metrics_to_file(
            metrics_dict={'vs_test': metrics_test, 'vs_train': metrics_train,
                          'stats_vs_test': stats_vs_test, 'stats_vs_train': stats_vs_train},
            log_dir=log_dir, experiment_name=cfg.general.name,
            epoch=epoch, mode='sampling_combined')

    # Visualize GT vs Generated comparison
    if visualization is not None:
        gt_graphs = datamodule.get_gt_graphs(num_graphs=5, split='train')
        result_path = f'{cfg.general.experiment_dir}/graphs/comparisons/'
        visualization.visualize_comparison(result_path, gt_graphs, all_samples, num_comparisons=5)

    return all_samples


def save_checkpoint(model, optimizer, epoch, best_val_nll, path):
    """Save a checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_nll': best_val_nll,
    }, path)
    print(f'Checkpoint saved to {path}')


def load_checkpoint(model, optimizer, path, device):
    """Load a checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint.get('best_val_nll', float('inf'))


def main():
    parser = argparse.ArgumentParser(description='DiGress Minimal Training')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'sample'],
                        help='Mode: train, test, or sample')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for resuming/testing')
    parser.add_argument('--experiment_dir', type=str, default=None,
                        help='Use existing experiment directory (for test/sample modes)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Handle experiment directory
    if args.experiment_dir is not None:
        # Use existing experiment directory (for test/sample modes)
        experiment_dir = args.experiment_dir
        print(f'Using existing experiment directory: {experiment_dir}')
        if not os.path.exists(experiment_dir):
            print(f'Warning: Experiment directory does not exist: {experiment_dir}')
            print('Creating it now...')
            os.makedirs(f'{experiment_dir}/checkpoints', exist_ok=True)
            os.makedirs(f'{experiment_dir}/graphs', exist_ok=True)
            os.makedirs(f'{experiment_dir}/chains', exist_ok=True)
            os.makedirs(f'{experiment_dir}/logs', exist_ok=True)
    else:
        # Create new timestamped experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = f'experiments/{cfg.general.name}_{timestamp}'
        
        # Create output directories under experiment folder
        os.makedirs(f'{experiment_dir}/checkpoints', exist_ok=True)
        os.makedirs(f'{experiment_dir}/graphs', exist_ok=True)
        os.makedirs(f'{experiment_dir}/chains', exist_ok=True)
        os.makedirs(f'{experiment_dir}/logs', exist_ok=True)
        
        print(f'Created new experiment directory: {experiment_dir}')
    
    # Store experiment_dir in config for easy access
    cfg.general.experiment_dir = experiment_dir

    # Tee all stdout to a single .log file
    _tee = setup_logging(experiment_dir, cfg.general.name)

    # Save config and experiment info for new experiments
    if args.experiment_dir is None:
        # Save config to experiment directory for reproducibility
        config_save_path = f'{experiment_dir}/config.yaml'
        with open(config_save_path, 'w') as f:
            yaml.dump(dict(cfg), f, default_flow_style=False)
        
        # Create experiment info file
        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(f'{experiment_dir}/experiment_info.txt', 'w') as f:
            f.write(f"Experiment: {cfg.general.name}\n")
            f.write(f"Created: {timestamp_str}\n")
            f.write(f"Mode: {args.mode}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Config file: {args.config}\n")
            if args.checkpoint:
                f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"\nDirectory structure:\n")
            f.write(f"  - checkpoints/  (model checkpoints)\n")
            f.write(f"  - graphs/       (generated graph visualizations)\n")
            f.write(f"  - chains/       (diffusion chain visualizations)\n")
            f.write(f"  - logs/         (metrics logs in JSON and TXT format)\n")
    
    # Create data module
    datamodule = create_datamodule(cfg)
    dataset_infos = datamodule.dataset_infos
    
    # Create dataloaders
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    
    # Create extra features
    ef_type = getattr(cfg.model, 'extra_features', 'none')
    if ef_type in ['all', 'cycles', 'eigenvalues', 'norm_cycles', 'norm_all']:
        extra_features = ExtraFeatures(
            extra_features_type=ef_type,
            dataset_info=dataset_infos
        )
        domain_features = DummyExtraFeatures()
    else:
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()
    
    # Compute input/output dimensions
    dataset_infos.compute_input_output_dims(datamodule, extra_features, domain_features)
    
    # Create sampling metrics and visualization
    sampling_metrics = create_sampling_metrics(cfg, datamodule)
    visualization = NonMolecularVisualization()
    
    # Create model
    model = DiscreteDenoisingDiffusion(
        cfg=cfg,
        dataset_infos=dataset_infos,
        train_metrics=None,
        sampling_metrics=sampling_metrics,
        visualization_tools=visualization,
        extra_features=extra_features,
        domain_features=domain_features
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    
    start_epoch = 0
    best_val_nll = float('inf')

    # Auto-discover latest checkpoint when resuming from an existing experiment dir
    if args.checkpoint is None and args.experiment_dir is not None:
        ckpt_dir = os.path.join(experiment_dir, 'checkpoints')
        if os.path.isdir(ckpt_dir):
            # Prefer the latest epoch_*.pt by epoch number, fall back to best.pt
            import glob
            epoch_ckpts = glob.glob(os.path.join(ckpt_dir, 'epoch_*.pt'))
            if epoch_ckpts:
                def _epoch_num(p):
                    try:
                        return int(os.path.basename(p).split('_')[1].split('.')[0])
                    except (IndexError, ValueError):
                        return -1
                args.checkpoint = max(epoch_ckpts, key=_epoch_num)
            elif os.path.exists(os.path.join(ckpt_dir, 'best.pt')):
                args.checkpoint = os.path.join(ckpt_dir, 'best.pt')
            if args.checkpoint:
                print(f'Auto-discovered checkpoint for resume: {args.checkpoint}')

    # Load checkpoint if provided
    if args.checkpoint:
        print(f'Loading checkpoint from {args.checkpoint}')
        start_epoch, best_val_nll = load_checkpoint(model, optimizer, args.checkpoint, device)
        print(f'Resumed from epoch {start_epoch} with best val NLL {best_val_nll:.4f}')
    
    if args.mode == 'train':
        print('Starting training...')
        
        # Visualize ground truth training graphs before training starts
        num_train_graphs = getattr(cfg.dataset, 'num_train_graphs', 5)
        num_gt_to_visualize = min(5, num_train_graphs)
        gt_graphs = datamodule.get_gt_graphs(num_graphs=num_gt_to_visualize, split='train')
        gt_vis_path = f'{cfg.general.experiment_dir}/graphs/ground_truth/'
        visualization.visualize_gt_graphs(gt_vis_path, gt_graphs, num_graphs=num_gt_to_visualize)
        
        for epoch in range(start_epoch, cfg.train.n_epochs):
            model.current_epoch = epoch
            
            # Train
            train_loss = train_epoch(model, train_loader, optimizer, device, epoch, 
                                     log_every_steps=cfg.train.log_every_steps)
            print(f'Epoch {epoch} | Train Loss: {train_loss:.4f}')
            
            # Validate
            val_nll = validate(model, val_loader, device)
            print(f'Epoch {epoch} | Val NLL: {val_nll:.4f}')
            
            # Update scheduler
            scheduler.step(val_nll)
            
            # Save checkpoint
            if val_nll < best_val_nll:
                best_val_nll = val_nll
                save_checkpoint(
                    model, optimizer, epoch, best_val_nll,
                    f'{cfg.general.experiment_dir}/checkpoints/best.pt'
                )
            
            if epoch % cfg.train.save_every_epochs == 0:
                save_checkpoint(
                    model, optimizer, epoch, best_val_nll,
                    f'{cfg.general.experiment_dir}/checkpoints/epoch_{epoch}.pt'
                )
            
            # Sample periodically
            if epoch % cfg.general.sample_every_val == 0 and epoch > 0:
                print(f'Sampling at epoch {epoch}...')
                samples = model.sample_batch(
                    batch_id=0,
                    batch_size=cfg.general.samples_to_generate,
                    keep_chain=cfg.general.chains_to_save,
                    number_chain_steps=cfg.general.number_chain_steps,
                    save_final=cfg.general.final_model_samples_to_save
                )
        
        print('Training complete!')
        
        # Final test
        print('Running final test...')
        test_nll = test(model, test_loader, device)
        print(f'Test NLL: {test_nll:.4f}')
        
        # Final sampling and GT vs Generated comparison
        print('Generating final samples with GT comparison...')
        samples = sample_and_evaluate(model, cfg, datamodule, sampling_metrics, visualization, device)
        
    elif args.mode == 'test':
        if args.checkpoint is None:
            # Look for checkpoint in experiment directory or fallback to legacy path
            best_checkpoint = f'{cfg.general.experiment_dir}/checkpoints/best.pt'
            if not os.path.exists(best_checkpoint):
                best_checkpoint = f'checkpoints/{cfg.general.name}/best.pt'
            if os.path.exists(best_checkpoint):
                args.checkpoint = best_checkpoint
            else:
                print('No checkpoint provided and no best checkpoint found!')
                return
        
        load_checkpoint(model, None, args.checkpoint, device)
        test_nll = test(model, test_loader, device)
        print(f'Test NLL: {test_nll:.4f}')
        
    elif args.mode == 'sample':
        if args.checkpoint is None:
            # Look for checkpoint in experiment directory or fallback to legacy path
            best_checkpoint = f'{cfg.general.experiment_dir}/checkpoints/best.pt'
            if not os.path.exists(best_checkpoint):
                best_checkpoint = f'checkpoints/{cfg.general.name}/best.pt'
            if os.path.exists(best_checkpoint):
                args.checkpoint = best_checkpoint
            else:
                print('No checkpoint provided and no best checkpoint found!')
                return
        
        load_checkpoint(model, None, args.checkpoint, device)
        
        # Visualize ground truth graphs before sampling
        num_train_graphs = getattr(cfg.dataset, 'num_train_graphs', 5)
        num_gt_to_visualize = min(5, num_train_graphs)
        gt_graphs = datamodule.get_gt_graphs(num_graphs=num_gt_to_visualize, split='train')
        gt_vis_path = f'{cfg.general.experiment_dir}/graphs/ground_truth/'
        visualization.visualize_gt_graphs(gt_vis_path, gt_graphs, num_graphs=num_gt_to_visualize)
        
        samples = sample_and_evaluate(model, cfg, datamodule, sampling_metrics, visualization, device)
        print(f'Generated {len(samples)} samples')


if __name__ == '__main__':
    main()
