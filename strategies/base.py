"""
strategies/base.py
------------------
Abstract Strategy interface and the GameView snapshot.

Design notes:
------------
GameView is the ONLY thing a Strategy is allowed to observe.  It is a pure
value snapshot — no mutable state, no references back to Game or Board.
This enforces the information boundary between engine and strategy and makes
strategies unit-testable without a running Game.

Key contents of GameView:
  - shot_grid       : np.ndarray (read-only) — the observable board state
  - board_size      : int
  - turn            : int
  - afloat_ships    : List[ShipType] — which ships are still on the board
                      (lengths known, positions unknown — legitimate info)
  - sunk_ships      : List[ShipType] — already eliminated

The Strategy ABC requires only two methods:
  - select_action(view) -> (row, col)
  - reset()              — called at the start of each new game

This interface supports:
  - Stateless strategies    (ignore reset, derive everything from view)
  - Stateful strategies     (maintain internal probability maps across turns)
  - Future ML strategies    (encode view as feature vector, call model)

The optional name property enables experiment labeling without isinstance().
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np

from engine.board import CellState
from engine.ships import ShipType


# --------------------------------------------------------------------------- #
# GameView — the strategy's window into the game                              #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class GameView:
    """
    Immutable snapshot of observable game state passed to Strategy.select_action().

    Attributes
    ----------
    shot_grid    : np.ndarray, shape (board_size, board_size), dtype int8.
                   Values are CellState ints.  The array is a copy — strategies
                   cannot mutate board state through this reference.
    board_size   : Edge length of the square board.
    turn         : Number of shots already fired (0 on first call).
    afloat_ships : ShipTypes still afloat — lengths known, positions unknown.
                   This is public information derivable from the sinking events
                   visible in shot_grid.
    sunk_ships   : ShipTypes already sunk (in sinking order).
    """

    shot_grid: np.ndarray          # shape: (board_size, board_size)
    board_size: int
    turn: int
    afloat_ships: Tuple[ShipType, ...]
    sunk_ships: Tuple[ShipType, ...]

    # ------------------------------------------------------------------ #
    # Convenience accessors                                               #
    # ------------------------------------------------------------------ #

    @property
    def unfired_cells(self) -> List[Tuple[int, int]]:
        rows, cols = np.where(self.shot_grid == CellState.UNKNOWN)
        return list(zip(rows.tolist(), cols.tolist()))

    @property
    def hit_cells(self) -> List[Tuple[int, int]]:
        """HIT cells — ship present, ship not yet sunk."""
        rows, cols = np.where(self.shot_grid == CellState.HIT)
        return list(zip(rows.tolist(), cols.tolist()))

    @property
    def sunk_cells(self) -> List[Tuple[int, int]]:
        rows, cols = np.where(self.shot_grid == CellState.SUNK)
        return list(zip(rows.tolist(), cols.tolist()))

    @property
    def miss_cells(self) -> List[Tuple[int, int]]:
        rows, cols = np.where(self.shot_grid == CellState.MISS)
        return list(zip(rows.tolist(), cols.tolist()))

    @property
    def unfired_count(self) -> int:
        return int(np.sum(self.shot_grid == CellState.UNKNOWN))

    @property
    def has_active_hits(self) -> bool:
        """True if there are HIT cells that haven't been resolved to a sunk ship."""
        return bool(np.any(self.shot_grid == CellState.HIT))

    # ------------------------------------------------------------------ #
    # Factory                                                             #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_game(cls, game) -> "GameView":
        """
        Construct a GameView from a live Game instance.

        Called by GameRunner; not called by strategies themselves.
        The shot_grid is copied so future mutations to the board do not
        affect previously issued views (important for replay / analysis).
        """
        return cls(
            shot_grid=game.board.shot_grid.copy(),
            board_size=game.board_size,
            turn=game.turn,
            afloat_ships=tuple(game.board.afloat_ship_types),
            sunk_ships=tuple(game.board.sunk_ship_types),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"GameView(turn={self.turn}, afloat={[s.name for s in self.afloat_ships]}, "
            f"hits={len(self.hit_cells)}, unfired={self.unfired_count})"
        )


# --------------------------------------------------------------------------- #
# Strategy ABC                                                                 #
# --------------------------------------------------------------------------- #


class Strategy(ABC):
    """
    Abstract base class for all Battleship strategies.

    Implementors must define:
        select_action(view) -> (row, col)

    Implementors may override:
        reset()   — initialize / clear per-game state before each game starts
        name      — human-readable label used in experiment logs and plots

    Strategies are stateful objects (they may maintain probability maps,
    hunt-target queues, etc.) but must be fully reset between games so that
    GameRunner can reuse a single strategy instance across many games.
    """

    @abstractmethod
    def select_action(self, view: GameView) -> Tuple[int, int]:
        """
        Choose a cell to fire at.

        Parameters
        ----------
        view : GameView — current observable state.

        Returns
        -------
        (row, col) : int tuple, must be an UNKNOWN cell.

        The engine will raise ValueError if the returned cell has already
        been fired at — strategies are responsible for only returning
        unfired cells.
        """
        ...

    def reset(self) -> None:
        """
        Called by GameRunner at the start of every new game.

        Default implementation is a no-op — stateless strategies need not
        override.  Stateful strategies (probability maps, target queues)
        MUST override to avoid state leaking between games.
        """
        pass

    @property
    def name(self) -> str:
        """Human-readable strategy name.  Override for cleaner experiment labels."""
        return self.__class__.__name__

    def __repr__(self) -> str:  # pragma: no cover
        return f"Strategy({self.name})"
