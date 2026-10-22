"""
SparseDiffLSP: Sparse Diffusion with LogNormal Sociability Prior.
Entry point: training and generation via YAML config.

Usage:
    python main.py --config configs/ego.yaml
    python main.py --config configs/comm20.yaml
    python main.py --config configs/node_100.yaml
    python main.py --config configs/ego.yaml --mode sample --checkpoint path/to/ckpt.pt
"""

import os
import sys
import yaml
import argparse
import time
import math
import numpy as np
import networkx as nx
from datetime import datetime

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from diffusion_model import SparseDiffLSP
from metrics import (
    compute_mmd_ratios,
    print_mmd_report,
    visualize_graphs,
    visualize_comparison,
)
from data import (
    create_datamodule,
    compute_dataset_stats,
    compute_node_count_distribution,
    DistributionNodes,
)

# ------------------------------------------------------------------
#  Config helpers
# ------------------------------------------------------------------
class DotDict(dict):
    """Dot-notation access to nested dicts."""
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(f"No attribute '{attr}'")
    def __setattr__(self, attr, value):
        self[attr] = value
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
    @staticmethod
    def from_dict(d):
        if isinstance(d, dict):
            return DotDict({k: DotDict.from_dict(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [DotDict.from_dict(item) for item in d]
        return d


def load_config(path):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return DotDict.from_dict(raw)


# ------------------------------------------------------------------
#  Logging
# ------------------------------------------------------------------
class TeeLogger:
    def __init__(self, log_path):
        self.terminal = sys.__stdout__
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        self.logfile = open(log_path, "a", buffering=1)
    def write(self, msg):
        self.terminal.write(msg)
        self.logfile.write(msg)
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()
    def close(self):
        sys.stdout = self.terminal
        self.logfile.close()


# ------------------------------------------------------------------
#  Compute input / output dims for network
# ------------------------------------------------------------------
def compute_dims(cfg, train_loader):
    """Inspect one batch to determine feature dimensions.

    Returns:
        input_dims  dict {X: int, E: int, y: int}
        output_dims dict {X: int, E: int}
    """
    extra_name = cfg.model.extra_features
    # X dim = 1 (u_t) + extra node dim
    # E dim = 2 (one-hot edge) + extra edge dim (always 0 for our features)
    # y dim = extra graph dim + 1 (timestep)

    if extra_name in (None, "none"):
        dx_extra = 0
        de_extra = 0
        dy_extra = 0
    elif extra_name == "cycles":
        dx_extra = 3   # k3, k4, k5 node features
        de_extra = 0
        dy_extra = 4   # k3, k4, k5, k6 graph features
    elif extra_name == "eigen":
        k_ev = cfg.model.get("num_eigenvectors", 8)
        k_val = cfg.model.get("num_eigenvalues", 5)
        dx_extra = k_ev
        de_extra = 0
        dy_extra = k_val + 1  # eigenvalues + n_connected_components
    elif extra_name == "all":
        k_ev = cfg.model.get("num_eigenvectors", 8)
        k_val = cfg.model.get("num_eigenvalues", 5)
        dx_extra = 3 + k_ev
        de_extra = 0
        dy_extra = 4 + k_val + 1
    else:
        dx_extra = 0
        de_extra = 0
        dy_extra = 0

    use_lsp = cfg.model.get("use_lsp", True)

    input_dims = {
        "X": (1 if use_lsp else 0) + dx_extra if ((1 if use_lsp else 0) + dx_extra) > 0 else 1,
        "E": 2 + de_extra,    # one-hot edge (no-edge, edge) + extra edge features
        "y": dy_extra + 1,    # extra graph features + timestep
    }
    output_dims = {
        "X": 1,   # predicted u_0 (or eps) — unused in non-LSP but network needs ≥1
        "E": 2,   # predicted edge logits (no-edge, edge)
    }
    return input_dims, output_dims


# ------------------------------------------------------------------
#  Graph conversion helpers
# ------------------------------------------------------------------
def generated_to_nx(samples):
    """Convert generated samples [(node_types, adj_matrix), ...] to NetworkX."""
    graphs = []
    for node_types, adj in samples:
        A = adj.long().numpy()
        G = nx.from_numpy_array(A)
        graphs.append(G)
    return graphs


def loader_to_nx(loader):
    """Convert a PyG DataLoader to list of NetworkX graphs."""
    graphs = []
    for data in loader:
        batch = data.batch
        ptr = data.ptr if hasattr(data, "ptr") else None
        if ptr is None:
            unique, counts = torch.unique(batch, return_counts=True)
            ptr = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
        for i in range(ptr.shape[0] - 1):
            start, end = ptr[i].item(), ptr[i + 1].item()
            mask = (data.edge_index[0] >= start) & (data.edge_index[0] < end)
            ei = data.edge_index[:, mask] - start
            n_i = end - start
            adj = torch.zeros(n_i, n_i)
            if ei.shape[1] > 0:
                adj[ei[0], ei[1]] = 1.0
            G = nx.from_numpy_array(adj.numpy())
            graphs.append(G)
    return graphs


# ------------------------------------------------------------------
#  Basic graph stats comparison (lightweight, no external deps)
# ------------------------------------------------------------------
def compute_graph_stats(graphs):
    """Compute basic statistics for a list of NetworkX graphs."""
    stats = {"nodes": [], "edges": [], "density": [],
             "triangles": [], "avg_clustering": []}
    for G in graphs:
        if G.number_of_nodes() == 0:
            continue
        stats["nodes"].append(G.number_of_nodes())
        stats["edges"].append(G.number_of_edges())
        stats["density"].append(nx.density(G))
        stats["triangles"].append(sum(nx.triangles(G).values()) // 3)
        stats["avg_clustering"].append(nx.average_clustering(G))
    result = {}
    for k, v in stats.items():
        arr = np.array(v, dtype=float)
        if len(arr) > 0:
            result[k] = {"mean": float(arr.mean()), "std": float(arr.std())}
        else:
            result[k] = {"mean": 0.0, "std": 0.0}
    return result


def print_stats_comparison(gen_stats, ref_stats, ref_label):
    """Print a compact comparison table."""
    W = 70
    print(f"\n{'=' * W}")
    print(f"  Generated vs {ref_label}")
    print(f"{'=' * W}")
    keys = ["nodes", "edges", "density", "triangles", "avg_clustering"]
    for k in keys:
        gm, gs = gen_stats[k]["mean"], gen_stats[k]["std"]
        rm, rs = ref_stats[k]["mean"], ref_stats[k]["std"]
        print(f"  {k:<20} gen={gm:.3f}+/-{gs:.3f}  ref={rm:.3f}+/-{rs:.3f}  "
              f"|diff|={abs(gm - rm):.4f}")
    print(f"{'=' * W}\n")


# ------------------------------------------------------------------
#  Training
# ------------------------------------------------------------------
def train(cfg, device, resume_checkpoint=None):
    exp_name = cfg.general.name

    # If resuming, reuse the original experiment directory from checkpoint
    if resume_checkpoint is not None:
        # Expect checkpoint at <exp_dir>/checkpoints/<name>.pt
        exp_dir = os.path.dirname(os.path.dirname(os.path.abspath(resume_checkpoint)))
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_dir = os.path.join("experiments", f"{exp_name}_{timestamp}")

    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "samples"), exist_ok=True)

    tee = TeeLogger(os.path.join(exp_dir, "logs", f"{exp_name}.log"))
    sys.stdout = tee

    print(f"\n{'#' * 60}")
    print(f"  SparseDiffLSP Training: {exp_name}")
    print(f"  Device: {device}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 60}\n")

    # --- Data ---
    datamodule = create_datamodule(cfg)
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    # --- Dataset stats + prior calibration ---
    rho_data, n_ref, degree_sequences = compute_dataset_stats(train_loader)
    print(f"Dataset stats: rho={rho_data:.4f}, n_ref={n_ref}, "
          f"{len(degree_sequences)} graphs")

    node_hist = compute_node_count_distribution(train_loader)
    node_dist = DistributionNodes(node_hist)

    # --- Model ---
    model = SparseDiffLSP(cfg).to(device)
    model.calibrate_prior(rho_data, n_ref, degree_sequences)
    model.node_dist = node_dist

    input_dims, output_dims = compute_dims(cfg, train_loader)
    model.build_network(input_dims, output_dims)
    model = model.to(device)

    # --- Optimizer ---
    lr = cfg.train.lr
    wd = cfg.train.get("weight_decay", 1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # --- LR Scheduler ---
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.n_epochs, eta_min=lr * 0.01
    )

    # --- Resume from checkpoint ---
    start_epoch = 1
    best_val_loss = float("inf")
    if resume_checkpoint is not None:
        print(f"\n  Resuming from checkpoint: {resume_checkpoint}")
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        # Advance scheduler to the right epoch
        for _ in range(ckpt["epoch"]):
            scheduler.step()
        print(f"  Resumed at epoch {start_epoch} (best_val={best_val_loss:.4f})")

    # --- Training loop ---
    n_epochs = cfg.train.n_epochs
    log_every = cfg.train.get("log_every_steps", 50)
    save_every = cfg.train.get("save_every_epochs", 200)
    sample_every_val = cfg.general.get("sample_every_val", 50)
    val_every = cfg.general.get("check_val_every_n_epochs", 10)

    print(f"\nStarting training for {n_epochs} epochs (from epoch {start_epoch})...\n")

    for epoch in range(start_epoch, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()

            step = batch_idx + (epoch - 1) * len(train_loader)
            loss = model.training_step(data, step=step, log_every=log_every)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{n_epochs} | loss={avg_loss:.4f} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

        # --- Validation ---
        if epoch % val_every == 0:
            model.eval()
            val_losses = []
            for data in val_loader:
                data = data.to(device)
                vl = model.validation_step(data)
                val_losses.append(vl)
            avg_val = np.mean(val_losses) if val_losses else 0.0
            print(f"  Val loss: {avg_val:.4f}")

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                ckpt_path = os.path.join(exp_dir, "checkpoints", "best.pt")
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "cfg": cfg,
                    "prior_mu": model.prior_mu,
                    "prior_sigma": model.prior_sigma_scale,
                    "marginal_q": model.marginal_q,
                }, ckpt_path)
                print(f"  Saved best checkpoint (val={best_val_loss:.4f})")

        # --- Sample + evaluate ---
        if epoch % (val_every * sample_every_val) == 0 and epoch > 0:
            print(f"\n--- Sampling at epoch {epoch} ---")
            n_gen = cfg.general.get("samples_to_generate", 32)
            samples = model.sample_batch(batch_size=min(n_gen, 8))
            gen_nx = generated_to_nx(samples)
            ref_nx = loader_to_nx(val_loader)
            gs = compute_graph_stats(gen_nx)
            rs = compute_graph_stats(ref_nx)
            print_stats_comparison(gs, rs, "Validation")

        # --- Save periodic checkpoint ---
        if save_every > 0 and epoch % save_every == 0:
            ckpt_path = os.path.join(exp_dir, "checkpoints", f"epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cfg": cfg,
                "prior_mu": model.prior_mu,
                "prior_sigma": model.prior_sigma_scale,
                "marginal_q": model.marginal_q,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    # --- Final save ---
    final_path = os.path.join(exp_dir, "checkpoints", "final.pt")
    torch.save({
        "epoch": n_epochs,
        "model_state": model.state_dict(),
        "cfg": cfg,
        "prior_mu": model.prior_mu,
        "prior_sigma": model.prior_sigma_scale,
        "marginal_q": model.marginal_q,
    }, final_path)
    print(f"\nTraining complete. Final checkpoint: {final_path}")

    # --- Final generation + evaluation ---
    n_final = cfg.general.get("final_model_samples_to_generate", 100)
    print(f"\nGenerating {n_final} final samples...")
    model.eval()

    all_samples = []
    gen_bs = cfg.train.batch_size
    while len(all_samples) < n_final:
        n_this = min(gen_bs, n_final - len(all_samples))
        samples = model.sample_batch(batch_size=n_this)
        all_samples.extend(samples)
        print(f"  Generated {len(all_samples)}/{n_final}")

    gen_nx = generated_to_nx(all_samples)
    train_nx = loader_to_nx(train_loader)
    test_loader = datamodule.test_dataloader()
    test_nx = loader_to_nx(test_loader)

    gs = compute_graph_stats(gen_nx)
    ts_train = compute_graph_stats(train_nx)
    ts_test = compute_graph_stats(test_nx)
    print_stats_comparison(gs, ts_train, "TRAIN")
    print_stats_comparison(gs, ts_test, "TEST")

    # --- MMD ratios ---
    print("\nComputing MMD ratios...")
    mmd_results = compute_mmd_ratios(gen_nx, train_nx, test_nx)
    print_mmd_report(mmd_results)

    # --- Visualizations ---
    viz_dir = os.path.join(exp_dir, "visualizations")
    n_viz = cfg.general.get("num_graphs_to_visualize", 8)
    visualize_graphs(os.path.join(viz_dir, "generated"), gen_nx,
                     prefix="gen", num_graphs=n_viz)
    visualize_graphs(os.path.join(viz_dir, "ground_truth"), test_nx,
                     prefix="gt", num_graphs=n_viz)
    visualize_comparison(os.path.join(viz_dir, "comparison"),
                         test_nx, gen_nx, num_pairs=n_viz)

    tee.close()
    return model


# ------------------------------------------------------------------
#  Sampling from checkpoint
# ------------------------------------------------------------------
def sample(cfg, device, checkpoint_path):
    print(f"\n  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Restore config from checkpoint if available
    ckpt_cfg = ckpt.get("cfg", cfg)

    # Build model
    model = SparseDiffLSP(cfg).to(device)

    use_lsp = cfg.model.get("use_lsp", True)
    if use_lsp:
        model.prior_mu = ckpt["prior_mu"]
        model.prior_sigma_scale = ckpt["prior_sigma"]
    else:
        model.marginal_q = ckpt.get("marginal_q", 0.1)
    model._prior_calibrated = True

    # Build network (need dims)
    datamodule = create_datamodule(cfg)
    train_loader = datamodule.train_dataloader()
    input_dims, output_dims = compute_dims(cfg, train_loader)
    model.build_network(input_dims, output_dims)
    model = model.to(device)

    node_hist = compute_node_count_distribution(train_loader)
    model.node_dist = DistributionNodes(node_hist)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    n_gen = cfg.general.get("final_model_samples_to_generate", 100)
    print(f"  Generating {n_gen} samples...")

    all_samples = []
    gen_bs = cfg.train.batch_size
    while len(all_samples) < n_gen:
        n_this = min(gen_bs, n_gen - len(all_samples))
        samples = model.sample_batch(batch_size=n_this)
        all_samples.extend(samples)
        print(f"    {len(all_samples)}/{n_gen}")

    gen_nx = generated_to_nx(all_samples)
    gs = compute_graph_stats(gen_nx)

    # Compare with dataset
    train_nx = loader_to_nx(train_loader)
    test_loader = datamodule.test_dataloader()
    test_nx = loader_to_nx(test_loader)
    ts = compute_graph_stats(test_nx)
    print_stats_comparison(gs, ts, "TEST")

    # --- MMD ratios ---
    print("\nComputing MMD ratios...")
    mmd_results = compute_mmd_ratios(gen_nx, train_nx, test_nx)
    print_mmd_report(mmd_results)

    # Save
    exp_dir = os.path.join("experiments", cfg.general.name, "samples")
    os.makedirs(exp_dir, exist_ok=True)
    save_path = os.path.join(exp_dir, "generated_samples.pt")
    torch.save(all_samples, save_path)
    print(f"  Saved {len(all_samples)} samples to {save_path}")

    # --- Visualizations ---
    viz_dir = os.path.join("experiments", cfg.general.name, "visualizations")
    n_viz = cfg.general.get("num_graphs_to_visualize", 8)
    visualize_graphs(os.path.join(viz_dir, "generated"), gen_nx,
                     prefix="gen", num_graphs=n_viz)
    visualize_graphs(os.path.join(viz_dir, "ground_truth"), test_nx,
                     prefix="gt", num_graphs=n_viz)
    visualize_comparison(os.path.join(viz_dir, "comparison"),
                         test_nx, gen_nx, num_pairs=n_viz)

    return all_samples


# ------------------------------------------------------------------
#  CLI
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SparseDiffLSP")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "sample"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint for sampling")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume training from")
    parser.add_argument("--device", type=str, default=None, help="Device (auto-detected)")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Seed
    seed = args.seed if args.seed is not None else cfg.general.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if args.mode == "train":
        train(cfg, device, resume_checkpoint=args.resume)
    elif args.mode == "sample":
        if args.checkpoint is None:
            ckpt = os.path.join("experiments", cfg.general.name, "checkpoints", "best.pt")
            print(f"  No checkpoint specified, trying: {ckpt}")
        else:
            ckpt = args.checkpoint
        sample(cfg, device, ckpt)


if __name__ == "__main__":
    main()
