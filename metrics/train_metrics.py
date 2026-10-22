"""
Training metrics for discrete diffusion.
"""
import torch
import torch.nn as nn

from .abstract_metrics import CrossEntropyMetric


class TrainLossDiscrete(nn.Module):
    """Train with Cross entropy loss."""
    
    def __init__(self, lambda_train):
        super().__init__()
        self.node_loss = CrossEntropyMetric()
        self.edge_loss = CrossEntropyMetric()
        self.y_loss = CrossEntropyMetric()
        self.lambda_train = lambda_train

    def forward(self, masked_pred_X, masked_pred_E, pred_y, true_X, true_E, true_y, log: bool):
        """Compute train metrics.
        
        Args:
            masked_pred_X: tensor -- (bs, n, dx)
            masked_pred_E: tensor -- (bs, n, n, de)
            pred_y: tensor -- (bs, )
            true_X: tensor -- (bs, n, dx)
            true_E: tensor -- (bs, n, n, de)
            true_y: tensor -- (bs, )
            log: boolean
        """
        true_X = torch.reshape(true_X, (-1, true_X.size(-1)))  # (bs * n, dx)
        true_E = torch.reshape(true_E, (-1, true_E.size(-1)))  # (bs * n * n, de)
        masked_pred_X = torch.reshape(masked_pred_X, (-1, masked_pred_X.size(-1)))  # (bs * n, dx)
        masked_pred_E = torch.reshape(masked_pred_E, (-1, masked_pred_E.size(-1)))  # (bs * n * n, de)

        # Remove masked rows
        mask_X = (true_X != 0.).any(dim=-1)
        mask_E = (true_E != 0.).any(dim=-1)

        flat_true_X = true_X[mask_X, :]
        flat_pred_X = masked_pred_X[mask_X, :]

        flat_true_E = true_E[mask_E, :]
        flat_pred_E = masked_pred_E[mask_E, :]

        loss_X = self.node_loss(flat_pred_X, flat_true_X) if flat_true_X.numel() > 0 else masked_pred_X.sum() * 0
        loss_E = self.edge_loss(flat_pred_E, flat_true_E) if flat_true_E.numel() > 0 else masked_pred_E.sum() * 0
        loss_y = self.y_loss(pred_y, true_y) if true_y.numel() > 0 else pred_y.sum() * 0

        return loss_X + self.lambda_train[0] * loss_E + self.lambda_train[1] * loss_y

    def reset(self):
        self.node_loss.reset()
        self.edge_loss.reset()
        self.y_loss.reset()

    def log_epoch_metrics(self):
        epoch_node_ce = self.node_loss.compute()
        epoch_edge_ce = self.edge_loss.compute()
        epoch_y_ce = self.y_loss.compute()

        to_log = {
            "train_epoch/x_CE": epoch_node_ce,
            "train_epoch/E_CE": epoch_edge_ce,
            "train_epoch/y_CE": epoch_y_ce
        }
        return to_log
