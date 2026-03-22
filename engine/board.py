"""
engine/board.py
---------------
Board representation and all placement / shot mechanics.

Design notes:
-----------
Two grids are maintained together in one object:

    ship_grid  : int8 ndarray  — ground truth; cell value = ship_id (1-based)
                                 or 0 for empty.  Hidden from strategies.
    shot_grid  : int8 ndarray  — observable state; values are CellState ints.
                                 This is what strategies receive via GameView.

Keeping both grids in the same object avoids object-graph complexity while
still making the separation explicit through attribute names.

CellState encoding (int8, cheap for numpy operations):
    UNKNOWN = 0   — not yet fired at
    MISS    = 1   — fired, no ship
    HIT     = 2   — fired, ship present, ship not yet sunk
    SUNK    = 3   — fired, ship present, ship now fully sunk

When a ship sinks, ALL its cells are set to SUNK (even previously HIT ones).
This makes the sunk-ship boundary fully observable, which strategies can
exploit for re-evaluating probability maps.

Placement uses rejection sampling — statistically O(1) per ship for the
standard 10×10 fleet, and deterministic given a seeded RNG.
"""

from __future__ import annotations

import random
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

import numpy as np

from .ships import Orientation, Ship, ShipType, STANDARD_FLEET

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

BOARD_SIZE: int = 10


# --------------------------------------------------------------------------- #
# Cell state                                                                   #
# --------------------------------------------------------------------------- #


class CellState(IntEnum):
    UNKNOWN = 0
    MISS = 1
    HIT = 2
    SUNK = 3

    def symbol(self) -> str:
        return _SYMBOLS[self]


_SYMBOLS: Dict[CellState, str] = {
    CellState.UNKNOWN: ".",
    CellState.MISS: "O",
    CellState.HIT: "X",
    CellState.SUNK: "#",
}


# --------------------------------------------------------------------------- #
# Board                                                                        #
# --------------------------------------------------------------------------- #


class Board:
    """
    A single Battleship board: ship placement + shot tracking.

    The board is from the perspective of the *target* — it knows where its
    own ships are and records incoming shots.  The shooter's view is exposed
    through shot_grid (and helper properties) without revealing ship_grid.
    """

    def __init__(self, size: int = BOARD_SIZE) -> None:
        self.size = size

        # Ground truth: 0 = empty, positive int = ship_id (1-based index into
        # self.ships).  int8 supports up to 127 ships — plenty.
        self.ship_grid: np.ndarray = np.zeros((size, size), dtype=np.int8)

        # Observable state: CellState values.
        self.shot_grid: np.ndarray = np.zeros((size, size), dtype=np.int8)

        self.ships: List[Ship] = []

        # O(1) lookup from cell → Ship instance (populated by place_ship).
        self._cell_to_ship: Dict[Tuple[int, int], Ship] = {}

    # ------------------------------------------------------------------ #
    # Placement                                                            #
    # ------------------------------------------------------------------ #

    def place_ship(self, ship: Ship) -> bool:
        """
        Place *ship* on the board.

        Returns True on success, False if placement is invalid (out-of-bounds
        or overlaps an existing ship).  Does NOT raise — callers (including
        rejection sampler) rely on the boolean return.
        """
        cells = ship.cells()

        # Validate bounds and no overlap in a single pass.
        for r, c in cells:
            if not (0 <= r < self.size and 0 <= c < self.size):
                return False
            if self.ship_grid[r, c] != 0:
                return False

        ship_id = len(self.ships) + 1  # 1-based
        for r, c in cells:
            self.ship_grid[r, c] = ship_id
            self._cell_to_ship[(r, c)] = ship

        self.ships.append(ship)
        return True

    def place_fleet_randomly(
        self,
        fleet: List[ShipType] = STANDARD_FLEET,
        rng: Optional[random.Random] = None,
    ) -> None:
        """
        Place every ship in *fleet* using rejection sampling.

        Clears any existing placement first.  Ships are placed in order
        (largest first by convention) — this marginally improves acceptance
        rates for smaller boards.

        Parameters
        ----------
        fleet : ordered list of ShipType to place.
        rng   : seeded Random instance.  If None, uses a fresh (unseeded) one.
        """
        if rng is None:
            rng = random.Random()

        # Clear state for re-use / reset.
        self.ship_grid.fill(0)
        self.shot_grid.fill(0)
        self.ships.clear()
        self._cell_to_ship.clear()

        for ship_type in fleet:
            placed = False
            while not placed:
                row = rng.randint(0, self.size - 1)
                col = rng.randint(0, self.size - 1)
                orientation = rng.choice([Orientation.HORIZONTAL, Orientation.VERTICAL])
                ship = Ship(ship_type, row, col, orientation)
                placed = self.place_ship(ship)

    # ------------------------------------------------------------------ #
    # Shot mechanics                                                       #
    # ------------------------------------------------------------------ #

    def fire(self, row: int, col: int) -> CellState:
        """
        Fire a shot at (row, col).

        Returns the CellState that results:
            MISS  — empty cell
            HIT   — ship cell, ship still afloat
            SUNK  — ship cell, ship now fully sunk (all cells marked SUNK)

        Raises
        ------
        ValueError  if (row, col) is out-of-bounds or already fired at.
        """
        if not (0 <= row < self.size and 0 <= col < self.size):
            raise ValueError(
                f"Shot ({row},{col}) is out of bounds for {self.size}×{self.size} board."
            )
        if self.shot_grid[row, col] != CellState.UNKNOWN:
            raise ValueError(
                f"Cell ({row},{col}) has already been fired at "
                f"(current state: {CellState(self.shot_grid[row, col]).name})."
            )

        if self.ship_grid[row, col] != 0:
            ship = self._cell_to_ship[(row, col)]
            ship.register_hit()

            if ship.is_sunk:
                # Mark every cell of the sunk ship, including previously-HIT
                # cells, so the shot_grid fully reflects the sunk boundary.
                for r, c in ship.cells():
                    self.shot_grid[r, c] = CellState.SUNK
                return CellState.SUNK
            else:
                self.shot_grid[row, col] = CellState.HIT
                return CellState.HIT
        else:
            self.shot_grid[row, col] = CellState.MISS
            return CellState.MISS

    # ------------------------------------------------------------------ #
    # Observable properties (strategy-safe)                               #
    # ------------------------------------------------------------------ #

    @property
    def is_game_over(self) -> bool:
        return all(ship.is_sunk for ship in self.ships)

    @property
    def ships_remaining(self) -> int:
        return sum(1 for ship in self.ships if not ship.is_sunk)

    @property
    def total_shots(self) -> int:
        return int(np.sum(self.shot_grid != CellState.UNKNOWN))

    @property
    def unfired_cells(self) -> List[Tuple[int, int]]:
        rows, cols = np.where(self.shot_grid == CellState.UNKNOWN)
        return list(zip(rows.tolist(), cols.tolist()))

    @property
    def hit_cells(self) -> List[Tuple[int, int]]:
        """Cells that are HIT but whose ship has not yet been sunk."""
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

    # ------------------------------------------------------------------ #
    # Sunk ship tracking (used by GameView for strategy hints)            #
    # ------------------------------------------------------------------ #

    @property
    def sunk_ship_types(self) -> List[ShipType]:
        """ShipTypes of all ships that have been sunk, in order of sinking."""
        return [s.ship_type for s in self.ships if s.is_sunk]

    @property
    def afloat_ship_types(self) -> List[ShipType]:
        """ShipTypes of ships still afloat."""
        return [s.ship_type for s in self.ships if not s.is_sunk]

    # ------------------------------------------------------------------ #
    # Display                                                              #
    # ------------------------------------------------------------------ #

    def render_shot_grid(self) -> str:
        """Render the observable shot grid as a human-readable string."""
        header = "  " + " ".join(str(c) for c in range(self.size))
        rows = [header]
        for r in range(self.size):
            row_cells = " ".join(
                CellState(self.shot_grid[r, c]).symbol() for c in range(self.size)
            )
            rows.append(f"{r} {row_cells}")
        return "\n".join(rows)

    def render_ship_grid(self) -> str:
        """Render the true ship placement (for debugging / post-game review)."""
        header = "  " + " ".join(str(c) for c in range(self.size))
        rows = [header]
        for r in range(self.size):
            row_cells = " ".join(
                str(self.ship_grid[r, c]) if self.ship_grid[r, c] != 0 else "."
                for c in range(self.size)
            )
            rows.append(f"{r} {row_cells}")
        return "\n".join(rows)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"Board(size={self.size}, ships={len(self.ships)}, "
            f"shots={self.total_shots}, remaining={self.ships_remaining})"
        )
