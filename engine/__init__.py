"""
engine — Core Battleship simulation engine.

Public API (Phase 1):
    Board, CellState, BOARD_SIZE
    Ship, ShipType, Orientation, STANDARD_FLEET, TOTAL_SHIP_CELLS
    Game, GameResult, ShotRecord
"""

from .board import Board, CellState, BOARD_SIZE
from .ships import Ship, ShipType, Orientation, STANDARD_FLEET, TOTAL_SHIP_CELLS
from .game import Game, GameResult, ShotRecord

__all__ = [
    "Board",
    "CellState",
    "BOARD_SIZE",
    "Ship",
    "ShipType",
    "Orientation",
    "STANDARD_FLEET",
    "TOTAL_SHIP_CELLS",
    "Game",
    "GameResult",
    "ShotRecord",
]
