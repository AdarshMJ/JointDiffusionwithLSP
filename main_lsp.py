"""
Joint continuous-discrete diffusion with LSP.
"""
import os
import sys
import argparse
import yaml
import time
import numpy as np
import networkx as nx
from datetime import datetime
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from lsp_diffusion_model import CRMDiffusion
from datasets.spectre_dataset import (
    SpectreGraphDataModule, SpectreDatasetInfos,
    Comm20DataModule, SBMDataModule, PlanarDataModule,
    CoraChameleonDataModule,
    TwoDensityMixtureDataModule, DCSBMDataModule,
    CommunityDataModule, EgoDataModule,
)
from diffusion.extra_features import (DummyExtraFeatures, ExtraFeatures,
                                     CRMDegreeResidualFeatures, CRMDensityResidualFeatures,
                                     CRMPriorEdgeProbFeatures, CRMResidualSpectralFeatures)
from analysis.spectre_utils import (
    SpectreSamplingMetrics, PlanarSamplingMetrics,
    SBMSamplingMetrics, Comm20SamplingMetrics,
    CoraChameleonSamplingMetrics,
    TwoDensitySamplingMetrics, DCSBMSamplingMetrics,
    CommunitySamplingMetrics, EgoSamplingMetrics,
    compute_wl_kernel_similarity,
)
from analysis.visualization import NonMolecularVisualization
import utils


# ------------------------------------------------------------------
# Logging: single .log file that tees all stdout output
# ------------------------------------------------------------------
class TeeLogger:
    """Tees sys.stdout to both the terminal and a .log file."""
    def __init__(self, log_path):
        self.terminal = sys.__stdout__
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.logfile = open(log_path, 'a', buffering=1)  # line-buffered

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
    """Redirect all stdout to a single .log file inside the experiment dir.

    All print() calls will appear on the terminal AND be written to
    experiments/<name>/logs/<name>.log
    """
    log_path = os.path.join(experiment_dir, 'logs', f'{name}.log')
    tee = TeeLogger(log_path)
    sys.stdout = tee
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
          f"Logging to: {log_path}")
    return tee


# ------------------------------------------------------------------
# Graph statistics utilities
# ------------------------------------------------------------------
def compute_expected_overlap(graphs):
    """Compute mean Chung-Lu expected self-overlap from degree sequences.

    For each graph G with degree sequence d_i and volume vol = sum(d_i),
    the Chung-Lu upper-bound on self overlap is:

        overlap_CL(G) = (sum_i d_i^2)^2 / vol^3

    This is permutation-invariant (only uses degree sequences) and serves
    as an upper bound on the true self-overlap of any model with that
    degree sequence.  A lower number means the generator is *less*
    self-similar, i.e. it produces diverse structures without memorising.
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
    """Compute aggregate statistics over a list of NetworkX graphs.

    Returns a dict of stat_name -> {'mean': float, 'std': float}
    covering: nodes, edges, density, triangles, avg_clustering,
    n_communities, max_degree.
    """
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


def compare_graph_stats(gen_graphs, ref_graphs, ref_label, logger=None):
    """Print a comparison table of graph statistics and return MSE/MAE.

    Args:
        gen_graphs:  list of generated NetworkX graphs.
        ref_graphs:  list of reference NetworkX graphs (train or test).
        ref_label:   human-readable label for the reference set.

    Returns:
        dict with 'stats_mse', 'stats_mae', 'gen_stats', 'ref_stats'.
    """
    gen_stats = compute_graph_stats_summary(gen_graphs)
    ref_stats = compute_graph_stats_summary(ref_graphs)

    KEYS = ['nodes', 'edges', 'density', 'triangles',
            'avg_clustering', 'n_communities', 'max_degree']

    W = 70
    print(f"\n{'='*W}")
    print(f"  Graph Statistics: Generated ({len(gen_graphs)} graphs) "
          f"vs {ref_label} ({len(ref_graphs)} graphs)")
    print(f"{'='*W}")
    header = f"  {'Statistic':<22}  {'Generated (mean±std)':<26}  "\
             f"{ref_label+' (mean±std)':<26}  |diff|"
    print(header)
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


# ------------------------------------------------------------------
# Config helpers
# ------------------------------------------------------------------
class DotDict(dict):
    """Dot notation access to dictionary attributes."""
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{attr}'")

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


# ------------------------------------------------------------------
# Power-law exponent (PLE) and characteristic path length (CPL)
# ------------------------------------------------------------------

def _power_law_exponent(G):
    """MLE estimate of the discrete power-law exponent (Clauset et al. 2009).
    Uses all nodes with degree >= 1.  Returns nan for trivially small graphs."""
    degrees = np.array([d for _, d in G.degree() if d > 0], dtype=float)
    if len(degrees) < 2:
        return float('nan')
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


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return DotDict.from_dict(config)


# ------------------------------------------------------------------
# Data / metrics factories
# ------------------------------------------------------------------
def create_datamodule(cfg):
    name = cfg.dataset.name
    if name == 'sbm':
        return SBMDataModule(cfg)
    elif name == 'planar':
        return PlanarDataModule(cfg)
    elif name == 'comm20':
        return Comm20DataModule(cfg)
    elif name == 'corachameleon':
        return CoraChameleonDataModule(cfg)
    elif name == 'two_density':
        return TwoDensityMixtureDataModule(cfg)
    elif name == 'dcsbm':
        return DCSBMDataModule(cfg)
    elif name == 'community':
        return CommunityDataModule(cfg)
    elif name == 'ego':
        return EgoDataModule(cfg)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def create_sampling_metrics(cfg, datamodule):
    name = cfg.dataset.name
    if name == 'sbm':
        return SBMSamplingMetrics(datamodule)
    elif name == 'planar':
        return PlanarSamplingMetrics(datamodule)
    elif name == 'comm20':
        return Comm20SamplingMetrics(datamodule)
    elif name == 'corachameleon':
        return CoraChameleonSamplingMetrics(datamodule)
    elif name == 'two_density':
        return TwoDensitySamplingMetrics(datamodule)
    elif name == 'dcsbm':
        return DCSBMSamplingMetrics(datamodule)
    elif name == 'community':
        return CommunitySamplingMetrics(datamodule)
    elif name == 'ego':
        return EgoSamplingMetrics(datamodule)
    else:
        return SpectreSamplingMetrics(datamodule)


def compute_dataset_stats(train_loader):
    """Compute mean graph density (rho_data), mean node count (n_ref),
    and per-graph degree sequences.

    These are needed by the automatic prior calibration pipeline.

    Returns:
        rho_data:          float  — mean density across all training graphs.
        n_ref:             int    — mean node count (rounded to nearest int).
        degree_sequences:  list of numpy arrays — per-graph degree sequences.
    """
    densities = []
    node_counts = []
    degree_sequences = []

    for data in train_loader:
        dense_data, node_mask = utils.to_dense(
            data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)

        bs = node_mask.shape[0]
        for i in range(bs):
            n_i = int(node_mask[i].sum().item())
            if n_i <= 1:
                continue
            # Extract adjacency from one-hot edge tensor
            A_i = dense_data.E[i, :n_i, :n_i].argmax(dim=-1).float()
            A_i.fill_diagonal_(0)
            edges_i = A_i.sum().item() / 2.0   # undirected
            density_i = 2.0 * edges_i / (n_i * (n_i - 1))
            densities.append(density_i)
            node_counts.append(n_i)
            # Collect degree sequence for lognormal calibration
            degrees_i = A_i.sum(dim=1).cpu().numpy()
            degree_sequences.append(degrees_i)

    if not densities:
        print("[WARNING] No valid graphs found. Using defaults rho=0.5, n=10")
        return 0.5, 10, []

    rho_data = sum(densities) / len(densities)
    n_ref = round(sum(node_counts) / len(node_counts))
    return rho_data, n_ref, degree_sequences


# ------------------------------------------------------------------
# Training / validation / test loops
# ------------------------------------------------------------------
def train_epoch(model, train_loader, optimizer, device, epoch, log_every_steps=50):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()

        step = batch_idx + epoch * len(train_loader)
        loss = model.training_step(data, log_every_steps=log_every_steps, step=step)

        if loss is None:
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % log_every_steps == 0:
            print(f'Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | '
                  f'Loss: {loss.item():.4f}')

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()
    model.reset_metrics(test=False)
    total_loss = 0.0
    num_batches = 0
    for data in val_loader:
        data = data.to(device)
        loss = model.validation_step(data)
        total_loss += loss
        num_batches += 1
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def test(model, test_loader, device):
    model.eval()
    model.reset_metrics(test=True)
    total_loss = 0.0
    num_batches = 0
    for data in test_loader:
        data = data.to(device)
        loss = model.test_step(data)
        total_loss += loss
        num_batches += 1
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def sample_and_evaluate(model, cfg, datamodule, sampling_metrics,
                        visualization, device, epoch=0):
    """Generate samples, compute graph stat comparison and WL similarity."""
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
            save_final=0)
        all_samples.extend(samples)
        batch_id += 1
        print(f'Generated {len(all_samples)}/{n_samples} samples')

    # Convert generated graphs to NetworkX
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

        # ---- WL Kernel Similarity ------------------------------------
        print("\n" + "#" * 80)
        print("#  WL KERNEL SIMILARITY: Generated vs TRAIN")
        print("#" * 80)
        wl_vs_train = compute_wl_kernel_similarity(
            train_graphs, gen_nx, n_iter=5)

        print("\n" + "#" * 80)
        print("#  WL KERNEL SIMILARITY: Generated vs TEST")
        print("#" * 80)
        wl_vs_test = compute_wl_kernel_similarity(
            test_graphs, gen_nx, n_iter=5)

        # ---- MMD Metrics (degree / spectre / clustering / orbit) -----
        print("\n" + "#" * 80)
        print("#  MMD DISTRIBUTION METRICS: Generated vs TRAIN")
        print("#" * 80)
        metrics_train = sampling_metrics(
            generated_graphs=all_samples, name=cfg.general.name,
            current_epoch=epoch, val_counter=0, test=False, use_train=True)
        print("\nMMD vs TRAIN:")
        for k, v in metrics_train.items():
            if isinstance(v, float) and not k.startswith('wl_'):
                print(f"  {k}: {v:.4f}")

        print("\n" + "#" * 80)
        print("#  MMD DISTRIBUTION METRICS: Generated vs TEST")
        print("#" * 80)
        metrics_test = sampling_metrics(
            generated_graphs=all_samples, name=cfg.general.name,
            current_epoch=epoch, val_counter=0, test=True)
        print("\nMMD vs TEST:")
        for k, v in metrics_test.items():
            if isinstance(v, float) and not k.startswith('wl_'):
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
        for label, wl, stats, mmd in [
            ("vs TRAIN", wl_vs_train, stats_vs_train, metrics_train),
            ("vs TEST",  wl_vs_test,  stats_vs_test,  metrics_test),
        ]:
            print(f"\n  [{label}]")
            print(f"    Stats MSE : {stats['stats_mse']:.6f}")
            print(f"    Stats MAE : {stats['stats_mae']:.6f}")
            if wl.get('wl_nn_mean') is not None:
                print(f"    WL NN mean: {wl['wl_nn_mean']:.4f}  "
                      f"(median {wl['wl_nn_median']:.4f})")
                for k, v in wl.items():
                    if k.startswith('wl_retrieval_rate'):
                        print(f"    {k}: {v:.4f}")
            # Uniqueness and novelty (from the forward() call)
            if mmd.get('frac_unique') is not None:
                print(f"    Uniqueness : {mmd['frac_unique']:.4f}")
            if mmd.get('frac_novel') is not None:
                print(f"    Novelty    : {mmd['frac_novel']:.4f}")
            # MMD ratios r = MMD²(gen,test) / MMD²(train,test) — only for vs TEST
            ratio_keys = [k for k in mmd if k.startswith('r_')]
            if ratio_keys:
                print("    MMD² ratios (r = MMD²(gen,test) / MMD²(train,test)):")
                for k in sorted(ratio_keys):
                    v = mmd[k]
                    s = f"{v:.4f}" if not (isinstance(v, float) and v != v) else "nan"
                    print(f"      {k}: {s}")
        print()

    # Visualise comparison
    if visualization is not None:
        gt_graphs = datamodule.get_gt_graphs(num_graphs=5, split='train')
        result_path = f'{cfg.general.experiment_dir}/graphs/comparisons/'
        visualization.visualize_comparison(result_path, gt_graphs,
                                           all_samples, num_comparisons=5)

    return all_samples


# ------------------------------------------------------------------
# Checkpoint helpers
# ------------------------------------------------------------------
def save_checkpoint(model, optimizer, epoch, best_val_loss, path, scheduler=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    # Persist auto-calibrated prior values so they survive checkpoint reload
    if hasattr(model, 'use_crm_prior') and model.use_crm_prior:
        checkpoint['crm_calibration'] = {
            'prior_family': getattr(model, 'prior_family', 'lognormal'),
            'prior_tau': model.prior_tau,
            'prior_mu': model.prior_mu,
            'prior_sigma_scale': model.prior_sigma_scale,
            'prior_alpha': getattr(model, 'prior_alpha', None),
        }
    torch.save(checkpoint, path)
    print(f'Checkpoint saved to {path}')


def load_checkpoint(model, optimizer, path, device, scheduler=None):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        print('Restored scheduler state.')
    # Restore auto-calibrated prior values
    if 'crm_calibration' in ckpt and hasattr(model, 'use_crm_prior') and model.use_crm_prior:
        cal = ckpt['crm_calibration']
        model.prior_family = cal.get('prior_family', 'ggp')  # backward compat
        model.prior_tau = cal.get('prior_tau', None)
        model.prior_mu = cal['prior_mu']
        model.prior_sigma_scale = cal['prior_sigma_scale']
        if 'prior_alpha' in cal and cal['prior_alpha'] is not None:
            model.prior_alpha = cal['prior_alpha']
        model._prior_calibrated = True
        tau_str = f'{cal["prior_tau"]:.6f}' if cal.get('prior_tau') is not None else 'N/A'
        alpha_str = f'{cal.get("prior_alpha", "N/A")}'
        print(f'Restored calibration ({model.prior_family}): '
              f'mu_pi={cal["prior_mu"]:.4f}, '
              f'sigma_pi={cal["prior_sigma_scale"]:.4f}, '
              f'tau={tau_str}, alpha={alpha_str}')
    return ckpt['epoch'], ckpt.get('best_val_loss', float('inf'))


# ------------------------------------------------------------------
# Input / output dimension computation for CRM
# ------------------------------------------------------------------
def compute_crm_input_output_dims(datamodule, cfg, extra_features, domain_features):
    """Compute input/output dims for the CRM model.

    Input X  = u_t (1) + extra features
    Input E  = A_t one-hot (2) + extra edge features
    Input y  = graph-level features + time (1)
    Output X = 1  (epsilon prediction, per node)
    Output E = 2  (binary edge prediction)
    Output y = 0
    """
    # Use a dummy batch to figure out extra feature sizes
    example_batch = next(iter(datamodule.train_dataloader()))
    ex_dense, node_mask = utils.to_dense(
        example_batch.x, example_batch.edge_index,
        example_batch.edge_attr, example_batch.batch)

    example_data = {
        'X_t': ex_dense.X,
        'E_t': ex_dense.E,
        'y_t': example_batch['y'],
        'node_mask': node_mask,
    }

    ex_extra = extra_features(example_data)
    ex_domain = domain_features(example_data)

    input_dims = {
        'X': 1 + ex_extra.X.size(-1) + ex_domain.X.size(-1),   # u_t + extras
        'E': 2 + ex_extra.E.size(-1) + ex_domain.E.size(-1),   # A_t one-hot + extras
        'y': example_batch['y'].size(1) + 1                    # graph feats + time
             + ex_extra.y.size(-1) + ex_domain.y.size(-1),
    }

    output_dims = {
        'X': 1,    # epsilon prediction
        'E': 2,    # binary edge logits
        'y': 0,
    }

    return input_dims, output_dims


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='CRM Graph Diffusion Training')
    parser.add_argument('--config', type=str,
                        default='configs/crm_planar_config.yaml')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'sample'])
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--experiment_dir', type=str, default=None)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    print(f'Using device: {device}')

    # ---- experiment directory ----
    if args.experiment_dir is not None:
        experiment_dir = args.experiment_dir
        print(f'Using existing experiment directory: {experiment_dir}')
        for sub in ['checkpoints', 'graphs', 'chains', 'logs']:
            os.makedirs(f'{experiment_dir}/{sub}', exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = f'experiments/{cfg.general.name}_{timestamp}'
        for sub in ['checkpoints', 'graphs', 'chains', 'logs']:
            os.makedirs(f'{experiment_dir}/{sub}', exist_ok=True)
        print(f'Created new experiment directory: {experiment_dir}')

    cfg.general.experiment_dir = experiment_dir

    # ---- single .log file: tee all stdout here ----
    _tee = setup_logging(experiment_dir, cfg.general.name)

    # Save config
    if args.experiment_dir is None:
        with open(f'{experiment_dir}/config.yaml', 'w') as f:
            yaml.dump(dict(cfg), f, default_flow_style=False)
        with open(f'{experiment_dir}/experiment_info.txt', 'w') as f:
            f.write(f"Experiment: {cfg.general.name}\n")
            f.write(f"Model: CRM Joint Diffusion\n")
            f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Device: {device}\n")
            f.write(f"Config: {args.config}\n")
            f.write(f"CRM params: c=4.0 (fixed), gamma={cfg.model.crm.gamma}, "
                    f"lambda_w={cfg.model.crm.lambda_w}, u_clamp={cfg.model.crm.u_clamp}\n")
            # Log CRM prior configuration
            prior = getattr(cfg.model.crm, 'prior', None)
            if prior is not None:
                f.write(f"CRM prior: family={getattr(prior, 'family', 'ggp')}, "
                        f"sigma={prior.sigma} (tau, mu_pi, sigma_pi auto-calibrated)\n")
            else:
                f.write(f"CRM prior: LEGACY (i.i.d. lognormal, no GGP)\n")

    # ---- data ----
    datamodule = create_datamodule(cfg)
    dataset_infos = datamodule.dataset_infos

    # ---- extra features ----
    ef_type = getattr(cfg.model, 'extra_features', 'none')
    if ef_type in ['all', 'cycles', 'eigenvalues', 'norm_cycles', 'norm_all']:
        extra_features = ExtraFeatures(
            extra_features_type=ef_type, dataset_info=dataset_infos)
        domain_features = DummyExtraFeatures()
    elif ef_type == 'degree_residual':
        c = getattr(cfg.model.crm, 'c', 4.0)   # fallback; model fixes c=4 internally
        u_clamp = getattr(cfg.model.crm, 'u_clamp', 15.0)
        extra_features = CRMDegreeResidualFeatures(c=c, u_clamp=u_clamp)
        domain_features = DummyExtraFeatures()
    elif ef_type == 'density_residual':
        c = getattr(cfg.model.crm, 'c', 4.0)
        u_clamp = getattr(cfg.model.crm, 'u_clamp', 15.0)
        extra_features = CRMDensityResidualFeatures(c=c, u_clamp=u_clamp)
        domain_features = DummyExtraFeatures()
    elif ef_type == 'prior_edge_prob':
        c = getattr(cfg.model.crm, 'c', 4.0)
        u_clamp = getattr(cfg.model.crm, 'u_clamp', 15.0)
        extra_features = CRMPriorEdgeProbFeatures(c=c, u_clamp=u_clamp)
        domain_features = DummyExtraFeatures()
    elif ef_type == 'residual_spectral':
        c = getattr(cfg.model.crm, 'c', 4.0)
        u_clamp = getattr(cfg.model.crm, 'u_clamp', 15.0)
        K = getattr(cfg.model, 'residual_spectral_k', 2)
        log_every = getattr(cfg.model, 'residual_spectral_log_every', 200)
        extra_features = CRMResidualSpectralFeatures(
            c=c, u_clamp=u_clamp, K=K, log_every=log_every)
        domain_features = DummyExtraFeatures()
    else:
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

    # ---- input / output dims ----
    input_dims, output_dims = compute_crm_input_output_dims(
        datamodule, cfg, extra_features, domain_features)
    dataset_infos.input_dims = input_dims
    dataset_infos.output_dims = output_dims
    print(f"Input dims:  {input_dims}")
    print(f"Output dims: {output_dims}")

    # ---- data loaders ----
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()

    # ---- compute dataset statistics for auto-calibration ----
    rho_data, n_ref, degree_sequences = compute_dataset_stats(train_loader)
    print(f"Dataset statistics: rho_data={rho_data:.4f}, n_ref={n_ref}, "
          f"graphs_with_degrees={len(degree_sequences)}")

    # Recreate train_loader since compute_dataset_stats consumed it
    train_loader = datamodule.train_dataloader()

    # ---- sampling metrics & visualisation ----
    sampling_metrics = create_sampling_metrics(cfg, datamodule)
    visualization = NonMolecularVisualization()

    # ---- model ----
    model = CRMDiffusion(
        cfg=cfg,
        dataset_infos=dataset_infos,
        sampling_metrics=sampling_metrics,
        visualization_tools=visualization,
        extra_features=extra_features,
        domain_features=domain_features,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ---- optimizer & scheduler ----
    optimizer = optim.AdamW(model.parameters(),
                            lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    start_epoch = 0
    best_val_loss = float('inf')

    # ---- auto-detect checkpoint when resuming into an existing experiment dir ----
    # Prefer latest.pt (most recent epoch) over best.pt (best val, may be much earlier)
    if args.checkpoint is None and args.experiment_dir is not None:
        for candidate_name in ('latest.pt', 'best.pt'):
            candidate = os.path.join(experiment_dir, 'checkpoints', candidate_name)
            if os.path.exists(candidate):
                args.checkpoint = candidate
                print(f'Auto-detected checkpoint for resuming: {args.checkpoint}')
                break

    if args.checkpoint:
        print(f'Loading checkpoint from {args.checkpoint}')
        start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, args.checkpoint, device, scheduler=scheduler)
        print(f'Resumed from epoch {start_epoch}, best val loss {best_val_loss:.4f}')

    # ---- calibrate CRM prior (mu_pi + sigma_pi) ----
    # Skipped automatically when resuming, since load_checkpoint restores
    # _prior_calibrated=True from the saved crm_calibration block.
    if model.use_crm_prior and not model._prior_calibrated:
        model.calibrate_prior(
            rho_data=rho_data, n_ref=n_ref,
            degree_sequences=degree_sequences, num_samples=500)

    # ==================================================================
    # TRAIN
    # ==================================================================
    if args.mode == 'train':
        print('Starting CRM training...')

        # Visualise GT graphs
        gt_graphs = datamodule.get_gt_graphs(num_graphs=5, split='train')
        gt_vis_path = f'{experiment_dir}/graphs/ground_truth/'
        visualization.visualize_gt_graphs(gt_vis_path, gt_graphs, num_graphs=5)

        for epoch in range(start_epoch, cfg.train.n_epochs):
            model.current_epoch = epoch

            train_loss = train_epoch(
                model, train_loader, optimizer, device, epoch,
                log_every_steps=cfg.train.log_every_steps)
            print(f'Epoch {epoch} | Train Loss: {train_loss:.4f}')

            val_loss = validate(model, val_loader, device)
            print(f'Epoch {epoch} | Val Loss: {val_loss:.4f}')

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, epoch, best_val_loss,
                                f'{experiment_dir}/checkpoints/best.pt',
                                scheduler=scheduler)

            # Always overwrite latest.pt so resume picks up the most recent epoch
            save_checkpoint(model, optimizer, epoch, best_val_loss,
                            f'{experiment_dir}/checkpoints/latest.pt',
                            scheduler=scheduler)

            if epoch % cfg.train.save_every_epochs == 0:
                save_checkpoint(model, optimizer, epoch, best_val_loss,
                                f'{experiment_dir}/checkpoints/epoch_{epoch}.pt',
                                scheduler=scheduler)

            if epoch % cfg.general.sample_every_val == 0 and epoch > 0:
                print(f'Sampling at epoch {epoch}...')
                model.eval()
                with torch.no_grad():
                    samples = []
                    batch_id = 0
                    n_samples = cfg.general.samples_to_generate
                    while len(samples) < n_samples:
                        num_this = min(cfg.train.batch_size, n_samples - len(samples))
                        samples.extend(model.sample_batch(
                            batch_id=batch_id,
                            batch_size=num_this,
                            keep_chain=0,
                            number_chain_steps=cfg.general.number_chain_steps,
                            save_final=0))
                        batch_id += 1
                    print(f'  Generated {len(samples)} samples')
                gt_graphs = datamodule.get_gt_graphs(num_graphs=5, split='train')
                result_path = f'{experiment_dir}/graphs/comparisons/epoch_{epoch}/'
                visualization.visualize_comparison(
                    result_path, gt_graphs, samples, num_comparisons=5)
                print(f'  Saved comparison plots to {result_path}')
                model.train()

        print('Training complete!')

        test_loss = test(model, test_loader, device)
        print(f'Test Loss: {test_loss:.4f}')

        print('Generating final samples...')
        sample_and_evaluate(model, cfg, datamodule, sampling_metrics,
                            visualization, device)

    # ==================================================================
    # TEST
    # ==================================================================
    elif args.mode == 'test':
        if args.checkpoint is None:
            ckpt = f'{experiment_dir}/checkpoints/best.pt'
            if os.path.exists(ckpt):
                args.checkpoint = ckpt
            else:
                print('No checkpoint found!'); return
        load_checkpoint(model, None, args.checkpoint, device)
        test_loss = test(model, test_loader, device)
        print(f'Test Loss: {test_loss:.4f}')

    # ==================================================================
    # SAMPLE
    # ==================================================================
    elif args.mode == 'sample':
        if args.checkpoint is None:
            ckpt = f'{experiment_dir}/checkpoints/best.pt'
            if os.path.exists(ckpt):
                args.checkpoint = ckpt
            else:
                print('No checkpoint found!'); return
        load_checkpoint(model, None, args.checkpoint, device)

        gt_graphs = datamodule.get_gt_graphs(num_graphs=5, split='train')
        gt_vis_path = f'{experiment_dir}/graphs/ground_truth/'
        visualization.visualize_gt_graphs(gt_vis_path, gt_graphs, num_graphs=5)

        samples = sample_and_evaluate(model, cfg, datamodule, sampling_metrics,
                                      visualization, device)
        print(f'Generated {len(samples)} samples')


if __name__ == '__main__':
    main()
