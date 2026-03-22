"""
info_theory/__init__.py
-----------------------
Public API for the information-theoretic module.

Exposes:
  ConfigSampler     — Monte Carlo sampler of valid ship configurations
  ProbabilityEngine — Maintains belief distribution; computes marginal P(ship at cell)
  compute_expected_entropies — Vectorized E[H|action] for all candidate cells
  binary_entropy, board_entropy — Entropy utility functions
"""

from .hypothesis_space import ConfigSampler
from .probability_map import ProbabilityEngine
from .entropy import binary_entropy, board_entropy, compute_expected_entropies

__all__ = [
    "ConfigSampler",
    "ProbabilityEngine",
    "binary_entropy",
    "board_entropy",
    "compute_expected_entropies",
]
