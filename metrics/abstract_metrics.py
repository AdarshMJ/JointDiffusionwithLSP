"""
Abstract metrics for training and evaluation.
Adapted from DiGress to use pure PyTorch.
"""
import torch
from torch import Tensor
from torch.nn import functional as F


class TrainAbstractMetricsDiscrete(torch.nn.Module):
    """Abstract metrics for discrete training."""
    
    def __init__(self):
        super().__init__()

    def forward(self, masked_pred_X, masked_pred_E, true_X, true_E, log: bool):
        pass

    def reset(self):
        pass

    def log_epoch_metrics(self):
        return None, None


class SumExceptBatchMetric(torch.nn.Module):
    """Metric that sums over all dimensions except batch."""
    
    def __init__(self):
        super().__init__()
        self.total_value = 0.
        self.total_samples = 0.

    def update(self, values) -> None:
        self.total_value += torch.sum(values).item()
        self.total_samples += values.shape[0]

    def compute(self):
        if self.total_samples == 0:
            return 0.
        return self.total_value / self.total_samples

    def reset(self):
        self.total_value = 0.
        self.total_samples = 0.

    def __call__(self, values):
        self.update(values)
        return self.compute()


class SumExceptBatchKL(torch.nn.Module):
    """KL divergence metric."""
    
    def __init__(self):
        super().__init__()
        self.total_value = 0.
        self.total_samples = 0.

    def update(self, p, q) -> None:
        kl = F.kl_div(q, p, reduction='sum')
        self.total_value += kl.item()
        self.total_samples += p.size(0)

    def compute(self):
        if self.total_samples == 0:
            return 0.
        return self.total_value / self.total_samples

    def reset(self):
        self.total_value = 0.
        self.total_samples = 0.

    def __call__(self, p, q):
        self.update(p, q)
        return self.compute()


class CrossEntropyMetric(torch.nn.Module):
    """Cross entropy metric."""
    
    def __init__(self):
        super().__init__()
        self.total_ce = 0.
        self.total_samples = 0.

    def update(self, preds: Tensor, target: Tensor) -> Tensor:
        """Update state with predictions and targets. Returns differentiable loss."""
        target_classes = torch.argmax(target, dim=-1)
        ce = F.cross_entropy(preds, target_classes, reduction='mean')
        self.total_ce += ce.item() * preds.size(0)
        self.total_samples += preds.size(0)
        return ce

    def compute(self):
        if self.total_samples == 0:
            return 0.
        return self.total_ce / self.total_samples

    def reset(self):
        self.total_ce = 0.
        self.total_samples = 0.

    def __call__(self, preds: Tensor, target: Tensor) -> Tensor:
        """Compute cross entropy and update state. Returns differentiable loss."""
        return self.update(preds, target)


class ProbabilityMetric(torch.nn.Module):
    """Probability metric."""
    
    def __init__(self):
        super().__init__()

    def __call__(self, tensor):
        return tensor.mean()


class NLL(torch.nn.Module):
    """Negative log-likelihood metric."""
    
    def __init__(self):
        super().__init__()
        self.total_nll = 0.
        self.total_samples = 0.

    def update(self, nlls) -> None:
        self.total_nll += torch.sum(nlls).item()
        self.total_samples += nlls.size(0)

    def compute(self):
        if self.total_samples == 0:
            return 0.
        return self.total_nll / self.total_samples

    def reset(self):
        self.total_nll = 0.
        self.total_samples = 0.

    def __call__(self, nlls):
        self.update(nlls)
        return self.compute()
