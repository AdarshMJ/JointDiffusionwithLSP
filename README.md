<div align="center">

# When Edge Independence Fails: Joint Graph Diffusion with Latent Sociability Prior

<h3>🎉 Accepted at NeurIPS 2026 🎉</h3>

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4B44CE.svg)](https://openreview.net/forum?id=p1zm5z8W2c)
[![arXiv](https://img.shields.io/badge/arXiv-2610.13765-b31b1b.svg)](https://arxiv.org/abs/2610.13765)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Graph_Diffusion-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)

**[Paper](https://arxiv.org/abs/2610.13765)**

Adarsh Jamadandi¹ · Nicolas Keriven¹ · Aline Roumy²  
¹CNRS, IRISA, Rennes, France · ²Inria, Rennes, France

</div>

## Overview

Discrete graph diffusion commonly corrupts edges independently. Together with exchangeability, this forces the terminal distribution to be an Erdős–Rényi graph. For unconditional generation without node features, this produces a fixed-point effect: graph denoisers tend to retain homogeneous, Erdős–Rényi-like structure and struggle to recover node heterogeneity.

This repository implements **Latent Sociability Prior (LSP)**, a joint diffusion framework that evolves continuous node-level sociability variables alongside discrete graph edges. The latent variables guide edge updates throughout the diffusion trajectory, introduce edge dependence while preserving exchangeability, and require no pretrained encoder.

The repository includes:

- the dense **Joint LSP + DiGress** model;
- a feature-free **DiGress** baseline;
- **SparseDiff-LSP** and its no-LSP baseline;
- Cora/Chameleon subgraph dataset generation;
- graph-distribution evaluation and visualization utilities.

## Repository structure

```text
.
├── main_lsp.py                 # Joint LSP + DiGress training/evaluation
├── main.py                     # DiGress baseline
├── graph_generator.py          # Cora/Chameleon subgraph generation
├── configs/                    # Dense-model configurations
├── datasets/                   # Dataset loaders
├── diffusion/                  # Discrete diffusion components
├── lsp_prior/                  # Latent sociability prior
├── models/                     # Graph Transformer components
├── analysis/                   # Metrics and visualizations
└── SparseDiff-LSP/             # Sparse implementation and configs
```

## Installation

Python 3.11 or later is recommended. Create an isolated environment and install the dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torch-geometric
python -m pip install numpy scipy networkx matplotlib scikit-learn pyyaml tqdm grakel pyemd
```

For CUDA-specific PyTorch and PyTorch Geometric builds, follow their official installation instructions for your CUDA version.

## Dataset preparation

The Cora/Chameleon graph splits are generated locally. The command below creates 3,000 training graphs for each training split and 100 evaluation graphs, all with 100 nodes:

```bash
python graph_generator.py \
  --output-dir data/CoraChameleon \
  --node-sizes 100 \
  --train-per-set 3000 \
  --test-count 100 \
  --real-world \
  --s1-source-datasets Cora Chameleon \
  --visualize
```

The resulting files are stored as:

```text
data/CoraChameleon/node_100/
├── S1.pt
├── S2.pt
└── test.pt
```

The first run downloads the source graph datasets through PyTorch Geometric. Generation supports checkpoints and resumes from existing intermediate files.

## Training

Run all commands in this section from the repository root.

### Joint LSP + DiGress

```bash
python main_lsp.py --config configs/LSPcorachameleonNode100.yaml
```

### DiGress baseline

```bash
python main.py --config configs/digresscorachameleonNode100.yaml
```

Both entry points select CUDA when available and otherwise use the CPU. To choose a device explicitly, append `--device cuda`, `--device cuda:0`, or `--device cpu`.

Each run creates a timestamped directory under `experiments/` containing the resolved configuration, logs, checkpoints, generated graphs, and diffusion-chain visualizations.

## Evaluation and sampling

Use the configuration from training and pass the saved checkpoint:

```bash
# Evaluate Joint LSP + DiGress
python main_lsp.py \
  --config configs/LSPcorachameleonNode100.yaml \
  --mode test \
  --checkpoint experiments/<run>/checkpoints/best.pt

# Generate and evaluate samples
python main_lsp.py \
  --config configs/LSPcorachameleonNode100.yaml \
  --mode sample \
  --checkpoint experiments/<run>/checkpoints/best.pt
```

Replace `main_lsp.py` and its config with `main.py` and `configs/digresscorachameleonNode100.yaml` to evaluate the DiGress baseline.

To resume dense-model training in an existing run directory:

```bash
python main_lsp.py \
  --config configs/LSPcorachameleonNode100.yaml \
  --experiment_dir experiments/<run>
```

The latest compatible checkpoint in that run is detected automatically.

## SparseDiff-LSP

The sparse implementation is self-contained in `SparseDiff-LSP/`. Run it from that directory so its relative dataset path resolves correctly:

```bash
cd SparseDiff-LSP

# SparseDiff with LSP
python main.py --config configs/node_100.yaml

# SparseDiff baseline without LSP
python main.py --config configs/node_100_nolsp.yaml
```

Sample from a sparse-model checkpoint with:

```bash
python main.py \
  --config configs/node_100.yaml \
  --mode sample \
  --checkpoint experiments/<run>/checkpoints/best.pt
```

Resume sparse training with:

```bash
python main.py \
  --config configs/node_100.yaml \
  --resume experiments/<run>/checkpoints/best.pt
```

## Configuration

Experiments are controlled through YAML files. The main settings are:

- `dataset`: data location, graph size, split, and number of training graphs;
- `model`: Graph Transformer dimensions, diffusion steps, extra features, and LSP parameters;
- `train`: epochs, batch size, learning rate, weight decay, and checkpoint frequency;
- `general`: random seed, evaluation frequency, and number of generated samples.

The supplied LSP and baseline configs use no structural node features, matching the paper's unconditional graph-generation setting.

## Citation

If this code is useful in your research, please cite:

```bibtex
@inproceedings{jamadandi2026edge,
  title     = {When Edge Independence Fails: Joint Graph Diffusion with Latent Sociability Priors},
  author    = {Jamadandi, Adarsh and Keriven, Nicolas and Roumy, Aline},
  booktitle = {The Fourtieth Annual Conference on Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

An arXiv citation is also available as `arXiv:2610.13765`.

## Acknowledgements

This codebase builds on the following open-source libraries:

- **[DiGress](https://github.com/cvignac/DiGress)** by Vignac et al.
  
- **[SparseDiff](https://github.com/qym7/SparseDiff)** by Qin et al.

We gratefully acknowledge both teams for making their code publicly available. Please cite the corresponding DiGress and SparseDiff papers when using components derived from these libraries. The complete discussion of related work and references is available in our paper.
