"""
Filter graphs with isolated nodes from a CoraChameleon .pt dataset file.

A graph passes the filter if:
  - It has exactly `--nodes` nodes (if --nodes is given), AND
  - Every node has at least one edge (no isolated nodes)

Usage:
    # Filter and enforce node count:
    python filter_isolated_nodes.py \
        --input  data/CoraChameleon/node_500/S2.pt \
        --output data/CoraChameleon/node_500/S2_filtered.pt \
        --nodes  500

    # Filter any node size (just remove isolated-node graphs):
    python filter_isolated_nodes.py \
        --input  data/CoraChameleon/node_10/S2.pt \
        --output data/CoraChameleon/node_10/S2_filtered.pt

    # Process all split files under a directory tree at once:
    python filter_isolated_nodes.py --dir data/CoraChameleon --suffix _filtered
"""

import argparse
import os
from typing import Optional
import torch


def has_isolated_nodes(data, n_nodes: int) -> bool:
    """Return True if any node in [0, n_nodes) has degree zero."""
    if data.edge_index.shape[1] == 0:
        return True  # no edges at all → all nodes isolated
    connected = torch.unique(data.edge_index)
    return connected.numel() < n_nodes


def filter_dataset(input_path: str, output_path: str, required_nodes: Optional[int]) -> None:
    print(f"Loading {input_path} ...")
    raw = torch.load(input_path)
    print(f"  Total graphs before filtering: {len(raw)}")

    kept, removed = [], 0
    for item in raw:
        # Each item is a (Data, metadata_dict) tuple
        data, meta = item
        n = data.num_nodes

        if required_nodes is not None and n != required_nodes:
            removed += 1
            continue

        if has_isolated_nodes(data, n):
            removed += 1
            continue

        kept.append(item)

    print(f"  Removed: {removed}  (wrong node count or isolated nodes)")
    print(f"  Kept:    {len(kept)}")

    torch.save(kept, output_path)
    print(f"Saved filtered dataset to {output_path}")


def process_directory(root_dir: str, suffix: str) -> None:
    """Walk root_dir and filter every .pt file that doesn't already end with suffix."""
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if not fname.endswith(".pt"):
                continue
            stem = fname[:-3]
            if stem.endswith(suffix):
                continue  # skip already-filtered files
            input_path = os.path.join(dirpath, fname)
            output_path = os.path.join(dirpath, stem + suffix + ".pt")
            filter_dataset(input_path, output_path, required_nodes=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove graphs with isolated nodes.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Path to a single input .pt file")
    group.add_argument("--dir",   help="Root directory: process all .pt files recursively")

    parser.add_argument("--output", help="Output path (only used with --input)")
    parser.add_argument("--nodes",  type=int, default=None,
                        help="If set, also discard graphs with a different node count")
    parser.add_argument("--suffix", default="_filtered",
                        help="Suffix appended to output filenames in --dir mode (default: _filtered)")
    args = parser.parse_args()

    if args.dir:
        process_directory(args.dir, args.suffix)
    else:
        if not args.output:
            parser.error("--output is required when using --input")
        filter_dataset(args.input, args.output, args.nodes)
