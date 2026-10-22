"""
Noise schedule for SparseDiffLSP.
Cosine beta schedule + alpha_bar computation.
"""

import numpy as np
import torch


def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """Cosine schedule betas (Nichol & Dhariwal, 2021)."""
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    return betas.squeeze()


class NoiseSchedule(torch.nn.Module):
    """Discrete cosine noise schedule.

    Stores betas and alpha_bar arrays indexed by integer timestep.
    Timestep 0 = clean, timestep T = fully noisy.
    """

    def __init__(self, timesteps):
        super().__init__()
        self.timesteps = timesteps

        betas = cosine_beta_schedule_discrete(timesteps)
        self.register_buffer("betas", torch.from_numpy(betas).float())

        alphas = 1.0 - torch.clamp(self.betas, min=0, max=0.9999)
        log_alpha = torch.log(alphas)
        log_alpha_bar = torch.cumsum(log_alpha, dim=0)
        self.register_buffer("alphas_bar", torch.exp(log_alpha_bar))

    def forward(self, t_int):
        """Get beta_t for integer timestep(s).
        t_int: (bs, 1) or (bs,) long tensor, values in [1, T]
        Returns: (bs, 1) betas
        """
        t_int = t_int.long().view(-1)
        return self.betas[t_int].unsqueeze(-1)

    def get_alpha_bar(self, t_int):
        """Get cumulative alpha_bar_t.
        t_int: (bs, 1) or (bs,) long, values in [0, T]
        Returns: (bs, 1) alpha_bar
        """
        t_int = t_int.long().view(-1)
        return self.alphas_bar[t_int].unsqueeze(-1)
