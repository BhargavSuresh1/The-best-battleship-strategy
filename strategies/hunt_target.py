"""
strategies/hunt_target.py
--------------------------
Hunt-Target strategy — the classical human heuristic for Battleship.

Algorithm
---------
The strategy operates in two alternating modes:

HUNT mode (no active hits)
  Fire at a random unfired cell.  This phase continues until a HIT is
  registered on the shot_grid.

TARGET mode (one or more active HIT cells)
  Systematically explore the neighbourhood of known hits.

  Orientation inference:
    - If all active HIT cells share the same row   → the ship is horizontal.
      Only shoot left/right of the hit span.
    - If all active HIT cells share the same column → the ship is vertical.
      Only shoot above/below the hit span.
    - Mixed / single hit → no orientation known yet; try all 4 neighbours.

  After a ship sinks, its cells become SUNK in the shot_grid.  Because
  HuntTargetStrategy derives its target set fresh from the view each turn,
  the transition back to HUNT mode (when no HIT cells remain) is automatic.

Statefulness
------------
HuntTargetStrategy is intentionally *stateless* — all decisions are derived
from `view.shot_grid` on each call to `select_action`.  This has several
advantages over a queue-based approach:

  - Correct by construction: no stale queue entries after sinking events.
  - No `reset()` logic required (though `reset()` is provided for the ABC).
  - Thread-safe and trivially serialisable.

Performance
-----------
Hunt-Target typically averages 50–60 shots on the standard 10×10 board,
roughly 35–40 % better than Random.  The main gains come from not wasting
shots after the first hit on a ship.
"""

from __future__ import annotations

import random as stdlib_random
from typing import List, Optional, Tuple

from engine.board import CellState
from strategies.base import GameView, Strategy


class HuntTargetStrategy(Strategy):
    """
    Classical Hunt-Target strategy with orientation inference.

    Parameters
    ----------
    seed : Optional[int]
        Seed for the internal RNG (used in HUNT mode).
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = stdlib_random.Random(seed)

    @property
    def name(self) -> str:
        return "HuntTarget"

    # ------------------------------------------------------------------ #
    # ABC implementation                                                   #
    # ------------------------------------------------------------------ #

    def select_action(self, view: GameView) -> Tuple[int, int]:
        """
        Return the next cell to fire at.

        Delegates to TARGET mode when active HIT cells exist, otherwise HUNT.
        """
        if view.has_active_hits:
            action = self._target_action(view)
            if action is not None:
                return action
            # Fallback: orientation inference failed to find candidates
            # (can happen if all neighbours are already fired).
            # Fall through to HUNT.
        return self._hunt_action(view)

    def reset(self) -> None:
        # Stateless — nothing to reset.
        pass

    # ------------------------------------------------------------------ #
    # HUNT mode                                                            #
    # ------------------------------------------------------------------ #

    def _hunt_action(self, view: GameView) -> Tuple[int, int]:
        """Choose a random unfired cell."""
        candidates = view.unfired_cells
        if not candidates:
            raise RuntimeError("HuntTargetStrategy: no unfired cells remain.")
        return self._rng.choice(candidates)

    # ------------------------------------------------------------------ #
    # TARGET mode                                                          #
    # ------------------------------------------------------------------ #

    def _target_action(self, view: GameView) -> Optional[Tuple[int, int]]:
        """
        Determine the best next shot given active HIT cells.

        Steps:
        1. Infer ship orientation from collinear hits.
        2. Build a list of candidate cells along the inferred axis (or all
           4 neighbours if orientation is unknown).
        3. Return a random candidate, or None if the candidate list is empty.
        """
        hit_cells = view.hit_cells
        B = view.board_size

        rows = [r for r, _ in hit_cells]
        cols = [c for _, c in hit_cells]

        all_same_row = len(set(rows)) == 1
        all_same_col = len(set(cols)) == 1

        candidates: List[Tuple[int, int]] = []

        if len(hit_cells) >= 2 and all_same_row:
            # Confirmed horizontal — extend along row.
            row = rows[0]
            min_col, max_col = min(cols), max(cols)
            if min_col - 1 >= 0 and view.shot_grid[row, min_col - 1] == CellState.UNKNOWN:
                candidates.append((row, min_col - 1))
            if max_col + 1 < B and view.shot_grid[row, max_col + 1] == CellState.UNKNOWN:
                candidates.append((row, max_col + 1))

        elif len(hit_cells) >= 2 and all_same_col:
            # Confirmed vertical — extend along column.
            col = cols[0]
            min_row, max_row = min(rows), max(rows)
            if min_row - 1 >= 0 and view.shot_grid[min_row - 1, col] == CellState.UNKNOWN:
                candidates.append((min_row - 1, col))
            if max_row + 1 < B and view.shot_grid[max_row + 1, col] == CellState.UNKNOWN:
                candidates.append((max_row + 1, col))

        else:
            # Orientation unknown (single hit, or hits from multiple ships):
            # explore all 4 neighbours of every HIT cell.
            seen = set()
            for r, c in hit_cells:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < B
                        and 0 <= nc < B
                        and view.shot_grid[nr, nc] == CellState.UNKNOWN
                        and (nr, nc) not in seen
                    ):
                        candidates.append((nr, nc))
                        seen.add((nr, nc))

        if not candidates:
            return None

        return self._rng.choice(candidates)
