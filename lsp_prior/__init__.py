"""
CRM Prior Sampling Module.

Provides sociability-based graph priors for the CRM diffusion model:

- LogNormal prior: direct calibration from training data (recommended)
- GGP (Generalized Gamma Process): legacy Caron-Fox (2017) implementation
- Simple graph sampling from CRM sociability weights
"""

from .ggp import sample_ggp_atoms
from .caronfox_graph import sample_caronfox_simple_graph
from .terminal import sample_terminal_graph_exact_n, sample_terminal_batch, estimate_alpha_for_n
from .calibrate import calibrate_prior_stats, auto_calibrate_from_config, solve_tau
from .lognormal_prior import calibrate_lognormal_prior, sample_lognormal_terminal_batch

__all__ = [
    'sample_ggp_atoms',
    'sample_caronfox_simple_graph',
    'sample_terminal_graph_exact_n',
    'sample_terminal_batch',
    'estimate_alpha_for_n',
    'calibrate_prior_stats',
    'auto_calibrate_from_config',
    'solve_tau',
    'calibrate_lognormal_prior',
    'sample_lognormal_terminal_batch',
]
