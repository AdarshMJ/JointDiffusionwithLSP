"""
Distribution for sampling number of nodes.
"""
import torch


class DistributionNodes:
    """Distribution over number of nodes in a graph."""
    
    def __init__(self, histogram):
        """Compute the distribution of the number of nodes.
        
        Args:
            histogram: dict or tensor. Keys are num_nodes, values are counts
        """
        if type(histogram) == dict:
            max_n_nodes = max(histogram.keys())
            prob = torch.zeros(max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram

        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(prob)

    def sample_n(self, n_samples, device):
        """Sample n_samples number of nodes."""
        idx = self.m.sample((n_samples,))
        return idx.to(device)

    def log_prob(self, batch_n_nodes):
        """Compute log probability of batch of node counts."""
        assert len(batch_n_nodes.size()) == 1
        p = self.prob.to(batch_n_nodes.device)

        probas = p[batch_n_nodes]
        log_p = torch.log(probas + 1e-30)
        return log_p
