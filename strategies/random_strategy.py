"""
strategies/random_strategy.py
-------------------------------
Uniform random strategy — the baseline all other strategies must beat.

Algorithm
---------
On each turn, choose uniformly at random from the set of cells that have not
yet been fired at.  No memory, no inference, no structure.

Performance
-----------
On a 10×10 board with the standard 5-ship fleet (17 ship cells out of 100):

    E[total_shots] ≈ 95–96

This follows from the coupon collector's problem: to cover 17 specific cells
when sampling without replacement from 100, the expected stopping time is

    E[T] = Σ_{k=0}^{16} (100 - k) / (17 - k) ≈ 95.4

The random strategy is the only strategy that does not exploit hit/miss
information — it exists solely as a statistical baseline.

Statefulness
------------
RandomStrategy is stateless.  It derives the available cells from the view
each turn, so `reset()` is a no-op.  The internal RNG is seeded at
construction time; to get different random behaviour across games use a
different seed or leave it unseeded.
"""

from __future__ import annotations

import random as stdlib_random
from typing import Optional, Tuple

from strategies.base import GameView, Strategy


class RandomStrategy(Strategy):
    """
    Shoot uniformly at random among unfired cells.

    Parameters
    ----------
    seed : Optional[int]
        Seed for the internal RNG.  Set for reproducibility, leave None for
        non-deterministic behaviour.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = stdlib_random.Random(seed)

    @property
    def name(self) -> str:
        return "Random"

    def select_action(self, view: GameView) -> Tuple[int, int]:
        """
        Choose a uniformly random unfired cell.

        Parameters
        ----------
        view : GameView — current observable state.

        Returns
        -------
        (row, col) : random UNKNOWN cell.
        """
        candidates = view.unfired_cells
        if not candidates:
            raise RuntimeError("RandomStrategy: no unfired cells remain.")
        return self._rng.choice(candidates)

    def reset(self) -> None:
        # Stateless — nothing to reset.
        pass
