"""
strategies — Battleship strategy implementations.

Public API:
    Strategy            — abstract base class
    GameView            — immutable game snapshot passed to strategies
    RandomStrategy      — uniform random baseline
    HuntTargetStrategy  — classical hunt-target heuristic
    ParityStrategy      — parity-filtered hunt with orientation targeting
    EntropyStrategy     — information-theoretic entropy minimisation
"""

from .base import Strategy, GameView
from .random_strategy import RandomStrategy
from .hunt_target import HuntTargetStrategy
from .parity import ParityStrategy
from .entropy_strategy import EntropyStrategy

__all__ = [
    "Strategy",
    "GameView",
    "RandomStrategy",
    "HuntTargetStrategy",
    "ParityStrategy",
    "EntropyStrategy",
]
