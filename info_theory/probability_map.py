"""
info_theory/probability_map.py
-------------------------------
Bayesian probability distribution over board cells.

Mathematical formulation
-------------------------
Given the hypothesis space Ω (set of valid configurations consistent with
the observed shot_grid), the marginal probability that cell (r, c) contains
a ship is:

    P(ship at (r,c)) = |{ω ∈ Ω : (r,c) is occupied in ω}| / |Ω|

Under a uniform prior P(ω) = 1/|Ω|, this is simply the fraction of valid
configurations in which (r,c) is occupied.

Monte Carlo approximation
--------------------------
Since |Ω| can be ≈ 10^8–10^10 for a full board, exact computation is
intractable.  We approximate using K i.i.d. samples ω_1, …, ω_K drawn by
ConfigSampler:

    P̂(ship at (r,c)) = (1/K) * Σ_k  𝟙[ω_k covers (r,c)]

This is an unbiased estimator with standard error O(1/√K).  For K=500 the
95 % CI width is ≤ 0.044 for any probability in [0,1] — adequate for
strategy use.

Caching
-------
The ProbabilityEngine caches its K samples keyed by (view.turn).  Since the
game state changes exactly once per shot, reusing the cache within a turn
avoids redundant sampling when both `compute_prob_map()` and `get_configs()`
are called (which is the case for EntropyStrategy).

The cache is invalidated on `reset()` (between games) and whenever
`view.turn` changes.

Update semantics
----------------
Rather than maintaining an incremental particle filter, we resample from
scratch each turn.  This is simpler and correct; the acceptance-rejection
sampler automatically incorporates all previous shots through the constraint
structure encoded in `view.shot_grid`.

For large n_samples and many remaining ships this can be slow.  The
`n_samples` parameter trades speed for accuracy:

    n_samples   Quality     Time/shot   Recommended use
    ----------  ----------  ----------  -----------------------------------
    100         Low         ~2 ms       10k+ game batch simulations
    300         Medium      ~6 ms       1k game experiments
    500         High        ~10 ms      Default strategy
    2000        Very high   ~40 ms      Per-game detailed analysis
    5000        Near-exact  ~100 ms     Research calibration
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from engine.board import CellState
from strategies.base import GameView
from .hypothesis_space import ConfigSampler


class ProbabilityEngine:
    """
    Maintains a Monte Carlo approximation to the Bayesian posterior over board
    occupancy, given all observed shots.

    Parameters
    ----------
    n_samples : int
        Number of valid configurations to sample per turn.
    rng_seed : Optional[int]
        Seed for the ConfigSampler's internal RNG.  Set for reproducibility.
    max_attempts_factor : int
        Forwarded to ConfigSampler.  Controls rejection-sampling budget.
    """

    def __init__(
        self,
        n_samples: int = 500,
        rng_seed: Optional[int] = None,
        max_attempts_factor: int = 20,
    ) -> None:
        self.n_samples = n_samples
        self._sampler = ConfigSampler(
            rng_seed=rng_seed,
            max_attempts_factor=max_attempts_factor,
        )
        # Cache: (turn, configs_array) or None
        self._cache: Optional[Tuple[int, np.ndarray]] = None

    # ------------------------------------------------------------------ #
    # Primary interface                                                    #
    # ------------------------------------------------------------------ #

    def compute_prob_map(self, view: GameView) -> np.ndarray:
        """
        Compute the marginal probability map P(ship at cell) for *view*.

        Returns
        -------
        prob_map : np.ndarray, shape (board_size, board_size), dtype=float64
            prob_map[r, c] ≈ P(cell (r,c) contains a ship).
            Fired cells (MISS, HIT, SUNK) are set to 0.0 — their state is
            known; they do not participate in probability reasoning.
            UNKNOWN cells have values in [0, 1].
        """
        configs = self._get_configs(view)
        B = view.board_size

        if len(configs) == 0:
            # No valid configuration found (degenerate state or game over).
            return np.zeros((B, B), dtype=np.float64)

        # Empirical marginal: fraction of configs occupying each cell.
        prob_map = configs.mean(axis=0).astype(np.float64)  # (B, B)

        # Known cells carry no residual uncertainty.
        fired_mask = view.shot_grid != CellState.UNKNOWN
        prob_map[fired_mask] = 0.0

        return prob_map

    def get_configs(self, view: GameView) -> np.ndarray:
        """
        Return the K sampled configurations for the current game state.

        Returns
        -------
        configs : np.ndarray, shape (K, board_size, board_size), dtype=bool
            K ≤ n_samples.  Shares the internal cache — do not mutate.
        """
        return self._get_configs(view)

    def reset(self) -> None:
        """
        Clear cached samples.  Must be called between games if the same
        ProbabilityEngine instance is reused.
        """
        self._cache = None

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    @property
    def cached_sample_count(self) -> int:
        """Number of samples in the current cache (0 if no cache)."""
        if self._cache is None:
            return 0
        return len(self._cache[1])

    @property
    def cached_turn(self) -> Optional[int]:
        """Turn number of the cached samples, or None if no cache."""
        return None if self._cache is None else self._cache[0]

    def effective_sample_size(self, view: GameView) -> int:
        """
        Return the actual number of valid samples obtained for *view*.
        May be less than n_samples when the acceptance rate is very low.
        """
        return len(self._get_configs(view))

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _get_configs(self, view: GameView) -> np.ndarray:
        """
        Return cached configs for the current turn, or resample if stale.

        Cache key is view.turn — this is correct because the view is
        constructed fresh each turn from the game state.
        """
        if self._cache is None or self._cache[0] != view.turn:
            configs = self._sampler.sample_configs(view, self.n_samples)
            self._cache = (view.turn, configs)
        return self._cache[1]

    def __repr__(self) -> str:  # pragma: no cover
        cached_info = (
            f"cached_turn={self.cached_turn}, K={self.cached_sample_count}"
            if self._cache is not None
            else "no cache"
        )
        return f"ProbabilityEngine(n_samples={self.n_samples}, {cached_info})"
