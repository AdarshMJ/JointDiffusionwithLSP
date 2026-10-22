"""
Sparse Graph Transformer for SparseDiffLSP.
Adapted from SparseDiff's GraphTransformerConv / TransformerConv.

Architecture:
    - Input MLPs: project (X, E, y) to hidden dims
    - Stack of XEyTransformerLayers: sparse multi-head attention + FiLM
    - Output MLPs: project back to output dims
    - Skip connection on X and E

Inputs:
    X: (N, dx_in)  node features (u_t + extras)
    edge_attr: (M, de_in)  edge features (A_t one-hot + extras)
    edge_index: (2, M)  sparse edges
    y: (bs, dy_in)  global features (extras + t)
    batch: (N,) graph assignment

Outputs:
    SparsePlaceHolder with:
        node: (N, dx_out)  — u_0 prediction (1 dim) for x0-param
        edge_attr: (M, de_out)  — edge logits (2 dims: no-edge, edge)
        edge_index, y, batch
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, LayerNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax, sort_edge_index
import torch_geometric.nn.pool as pool
from typing import Optional

import utils as U


# ---------------------------------------------------------------------------
#  Pooling layers: sparse X/E -> graph-level y
# ---------------------------------------------------------------------------
class SparseXtoy(nn.Module):
    def __init__(self, dx, dy):
        super().__init__()
        self.lin = nn.Linear(4 * dx, dy)

    def forward(self, X, batch):
        batch = batch.long()
        m = pool.global_mean_pool(X, batch)
        mi = -pool.global_max_pool(-X, batch)
        ma = pool.global_max_pool(X, batch)
        std = (X - m[batch]) ** 2
        std = pool.global_mean_pool(std, batch)
        z = torch.hstack((m, mi, ma, std))
        return self.lin(z)


class SparseEtoy(nn.Module):
    def __init__(self, de, dy):
        super().__init__()
        self.lin = nn.Linear(4 * de, dy)

    def forward(self, edge_index, edge_attr, batch):
        batch = batch.long()
        batchE = batch[edge_index[0]]
        m = pool.global_mean_pool(edge_attr, batchE)
        mi = -pool.global_max_pool(-edge_attr, batchE)
        ma = pool.global_max_pool(edge_attr, batchE)
        std = (edge_attr - m[batchE]) ** 2
        std = pool.global_mean_pool(std, batchE)
        z = torch.hstack((m, mi, ma, std))

        # Handle graphs with no edges
        n_graphs = batch.max().item() + 1
        out = self.lin.weight.new_zeros(n_graphs, self.lin.out_features)
        if z.shape[0] > 0:
            out[:z.shape[0]] = self.lin(z)
        return out


# ---------------------------------------------------------------------------
#  TransformerConv: sparse message-passing attention layer
# ---------------------------------------------------------------------------
class TransformerConv(MessagePassing):
    """Graph Transformer convolution with edge feature modulation (FiLM).

    Attention: alpha_{ij} = softmax_j( (Q_i . K_j / sqrt(d)) * (e_mul + 1) + e_add )
    Message:   m_ij = alpha_{ij} * V_j
    """

    def __init__(self, dx, de, dy, heads=8, concat=True, dropout=0.0,
                 bias=True, last_layer=True, **kwargs):
        kwargs.setdefault("aggr", "add")
        super().__init__(node_dim=0, **kwargs)

        self.dx = dx
        self.de = de
        self.dy = dy
        self._edge_attr_out = None  # buffer for edge output from message()
        self.df = int(dx / heads)
        self.heads = heads
        self.concat = concat
        self.last_layer = last_layer

        self.lin_key = Linear(dx, heads * self.df)
        self.lin_query = Linear(dx, heads * self.df)
        self.lin_value = Linear(dx, heads * self.df)

        if concat:
            self.lin_skip = Linear(dx, heads * self.df, bias=bias)
        else:
            self.lin_skip = Linear(dx, self.df, bias=bias)

        # FiLM: E -> attention
        self.e_add = Linear(de, heads)
        self.e_mul = Linear(de, heads)

        # FiLM: y -> E
        self.y_e_mul = Linear(dy, de)
        self.y_e_add = Linear(dy, de)

        # FiLM: y -> X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Graph-level prediction
        if self.last_layer:
            self.y_y = Linear(dy, dy)
            self.x_y = SparseXtoy(dx, dy)
            self.e_y = SparseEtoy(de, dy)
            self.y_out = nn.Sequential(Linear(dy, dy), nn.ReLU(), Linear(dy, dy))

        # Output projections
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)

    def forward(self, x, edge_index, edge_attr, y, batch):
        H, C = self.heads, self.df

        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(x).view(-1, H, C)
        value = self.lin_value(x).view(-1, H, C)

        # Standard propagate — message() stores edge_attr_out in self._edge_attr_out
        out = self.propagate(
            edge_index, query=query, key=key, value=value,
            edge_attr=edge_attr, size=None,
        )
        new_edge_attr = self._edge_attr_out  # retrieve stored edge output

        if self.concat:
            out = out.view(-1, self.heads * self.df)
        else:
            out = out.mean(dim=1)

        x_r = self.lin_skip(x)
        out = out + x_r

        # y -> E modulation
        batch_long = batch.long()
        batchE = batch_long[edge_index[0]]
        new_edge_attr = new_edge_attr.flatten(start_dim=1)  # (M, H*df=dx)
        new_edge_attr = self.e_out(new_edge_attr)
        ye1 = self.y_e_add(y)
        ye2 = self.y_e_mul(y)
        new_edge_attr = ye1[batchE] + (ye2[batchE] + 1) * new_edge_attr

        # y -> X modulation
        yx1 = self.y_x_add(y)
        yx2 = self.y_x_mul(y)
        new_x = yx1[batch_long] + (yx2[batch_long] + 1) * out
        new_x = self.x_out(new_x)

        # Graph-level update
        if self.last_layer:
            new_y = self.y_y(y)
            e_y = self.e_y(edge_index, edge_attr, batch)
            x_y = self.x_y(x, batch)
            new_y = new_y + x_y + e_y
            new_y = self.y_out(new_y)
        else:
            new_y = y

        return new_x, new_edge_attr, new_y

    def message(self, query_i, key_j, value_j, edge_attr, index, ptr, size_i):
        Y = (query_i * key_j) / math.sqrt(self.df)  # (M, H, C)

        e_mul = self.e_mul(edge_attr)   # (M, H)
        e_add = self.e_add(edge_attr)   # (M, H)
        Y = Y * (e_mul.unsqueeze(-1) + 1) + e_add.unsqueeze(-1)

        alpha = softmax(Y.sum(-1), index, ptr, size_i)  # (M, H)
        out = value_j * alpha.unsqueeze(-1)              # (M, H, C)

        # Store edge output as side-effect (cannot return tuple from message)
        self._edge_attr_out = Y
        return out


# ---------------------------------------------------------------------------
#  XEyTransformerLayer: one transformer block
# ---------------------------------------------------------------------------
class XEyTransformerLayer(nn.Module):
    def __init__(self, dx, de, dy, n_head, dim_ffX=256, dim_ffE=128,
                 dim_ffy=128, dropout=0.1, last_layer=True):
        super().__init__()
        self.last_layer = last_layer

        self.self_attn = TransformerConv(
            dx=dx, de=de, dy=dy, heads=n_head,
            concat=True, dropout=dropout, bias=True, last_layer=last_layer,
        )

        self.linX1 = Linear(dx, dim_ffX)
        self.linX2 = Linear(dim_ffX, dx)
        self.normX1 = LayerNorm(dx)
        self.normX2 = LayerNorm(dx)

        self.linE1 = Linear(de, dim_ffE)
        self.linE2 = Linear(dim_ffE, de)
        self.normE1 = LayerNorm(de)
        self.normE2 = LayerNorm(de)

        if self.last_layer:
            self.lin_y1 = Linear(dy, dim_ffy)
            self.lin_y2 = Linear(dim_ffy, dy)
            self.norm_y1 = LayerNorm(dy)
            self.norm_y2 = LayerNorm(dy)

    def forward(self, X, edge_index, edge_attr, y, batch):
        new_x, new_e, new_y = self.self_attn(X, edge_index, edge_attr, y, batch)

        X = self.normX1(X + new_x)
        edge_attr = self.normE1(edge_attr + new_e)

        if self.last_layer:
            y = self.norm_y1(y + new_y)
        else:
            y = new_y

        X = self.normX2(X + self.linX2(F.relu(self.linX1(X))))
        edge_attr = self.normE2(edge_attr + self.linE2(F.relu(self.linE1(edge_attr))))

        if self.last_layer:
            y = self.norm_y2(y + self.lin_y2(F.relu(self.lin_y1(y))))

        return X, edge_attr, y


# ---------------------------------------------------------------------------
#  GraphTransformerConv: full sparse transformer stack
# ---------------------------------------------------------------------------
class GraphTransformerConv(nn.Module):
    """Sparse Graph Transformer for SparseDiffLSP.

    Input:  (X, edge_attr, edge_index, y, batch)
    Output: SparsePlaceHolder(node, edge_attr, edge_index, y, batch)
    """

    def __init__(self, n_layers, input_dims, hidden_dims, output_dims, dropout=0.1):
        """
        Args:
            n_layers: number of transformer layers
            input_dims: dict with keys 'X', 'E', 'y' (int dimensions)
            hidden_dims: dict with 'dx', 'de', 'dy', 'n_head', 'dim_ffX', 'dim_ffE', 'dim_ffy'
            output_dims: dict with 'X', 'E' (int dimensions)
            dropout: dropout probability
        """
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims["X"]
        self.out_dim_E = output_dims["E"]

        self.lin_in_X = Linear(input_dims["X"], hidden_dims["dx"])
        self.lin_in_E = Linear(input_dims["E"], hidden_dims["de"])
        self.lin_in_y = Linear(input_dims["y"], hidden_dims["dy"])

        self.tf_layers = nn.ModuleList([
            XEyTransformerLayer(
                dx=hidden_dims["dx"],
                de=hidden_dims["de"],
                dy=hidden_dims["dy"],
                n_head=hidden_dims["n_head"],
                dim_ffX=hidden_dims["dim_ffX"],
                dim_ffE=hidden_dims["dim_ffE"],
                dim_ffy=hidden_dims.get("dim_ffy", 128),
                last_layer=(i < n_layers - 1),  # only inner layers predict y
            )
            for i in range(n_layers)
        ])

        self.out_ln_X = LayerNorm(hidden_dims["dx"])
        self.out_ln_E = LayerNorm(hidden_dims["de"])
        self.lin_out_X = Linear(hidden_dims["dx"], output_dims["X"])
        self.lin_out_E = Linear(hidden_dims["de"], output_dims["E"])

    def forward(self, X, edge_attr, edge_index, y, batch):
        # Save for skip connection
        X0 = X.clone()
        edge_attr0 = edge_attr.clone()

        # Input projection
        X = self.lin_in_X(X)
        edge_attr = self.lin_in_E(edge_attr)
        y = self.lin_in_y(y)

        # Transformer layers
        for layer in self.tf_layers:
            X, edge_attr, y = layer(X, edge_index, edge_attr, y, batch)

        # Output projection
        X = self.lin_out_X(self.out_ln_X(X))
        edge_attr = self.lin_out_E(self.out_ln_E(edge_attr))

        # Symmetrise edge predictions: average (i->j) and (j->i)
        top_edge_index, top_edge_attr = sort_edge_index(edge_index, edge_attr)
        _, bot_edge_attr = sort_edge_index(edge_index[[1, 0]], edge_attr)

        X = X + X0[:, :self.out_dim_X]
        edge_attr = top_edge_attr + bot_edge_attr + edge_attr0[:, :self.out_dim_E]

        return U.SparsePlaceHolder(
            node=X,
            edge_attr=edge_attr,
            edge_index=top_edge_index,
            y=y,
            batch=batch,
        )
