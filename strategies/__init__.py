"""
strategies — Battleship strategy implementations.

Public API:
    Strategy    — abstract base class
    GameView    — immutable game snapshot passed to strategies
"""

from .base import Strategy, GameView

__all__ = ["Strategy", "GameView"]
