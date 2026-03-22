"""
strategies/entropy_strategy.py
--------------------------------
Information-Theoretic (Entropy Minimisation) Strategy.

This is the most mathematically rigorous strategy in the project.  It chooses
each shot to minimise the expected residual Shannon entropy of the board state,
which is equivalent to maximising the expected information gain per shot.

Mathematical foundation
-----------------------
See info_theory/entropy.py for the full derivation.  In brief:

  For each candidate cell a, compute:

      E[H | shoot(a)] = P(hit at a) * H_after_hit(a)
                      + P(miss at a) * H_after_miss(a)

  where:
      P(hit at a)  = marginal probability from Monte Carlo sample set
      H_after_hit  = Σ_{c≠a,UNKNOWN} h( P(c occ | a hit)  )
      H_after_miss = Σ_{c≠a,UNKNOWN} h( P(c occ | a miss) )
      h(p)         = binary Shannon entropy = -p log p - (1-p) log(1-p)

  Choose:  a* = argmin_a  E[H | shoot(a)]

The conditional probabilities are computed via a joint occupancy matrix
(see entropy.py), making the full computation O(K * M^2) per turn where
K = n_samples and M = number of unfired cells.

Greedy optimality
-----------------
This strategy is *greedy*: it optimises the one-step lookahead only.  True
globally-optimal play would require a full tree search over all possible
future shot sequences, which is computationally intractable.  However, greedy
information-theoretic play is near-optimal in practice and is the strongest
of the four strategies implemented here.

Exposed attributes
------------------
After each call to `select_action()`, two diagnostic arrays are available:

    strategy.last_prob_map         — (B, B) float64 probability heatmap
    strategy.last_expected_H       — (B, B) float64 expected entropy per cell
    strategy.last_info_gain_map    — (B, B) float64 information gain per cell

These are intended for visualisation (Streamlit UI, Phase 6) and per-game
analysis (Phase 4).

Fallback behaviour
------------------
When the sampler returns K=0 valid configurations (degenerate state), the
strategy falls back to choosing the cell with the highest marginal probability
from any remaining samples, and ultimately to the Hunt-Target targeting logic.
This should never occur in normal gameplay but guards against edge cases.

Performance note
----------------
  n_samples   Avg shots   Time/game   Recommended use
  ----------  ----------  ----------  -------------------------
  100         ~43         ~0.5 s      Batch simulations (1k+)
  300         ~42         ~1.5 s      Default experiments
  500         ~41         ~3 s        Research-grade analysis
  2000        ~40         ~12 s       Calibration / ground truth

Times are approximate for a standard 10×10 board on modern hardware.
"""

from __future__ import annotations

import random as stdlib_random
from typing import Optional, Tuple

import numpy as np

from engine.board import CellState
from strategies.base import GameView, Strategy
from info_theory.probability_map import ProbabilityEngine
from info_theory.entropy import (
    binary_entropy,
    board_entropy,
    compute_expected_entropies,
    information_gain_map,
)


class EntropyStrategy(Strategy):
    """
    Information-theoretic strategy: minimise expected board entropy per shot.

    Parameters
    ----------
    n_samples : int
        Number of Monte Carlo configurations to sample per turn.
        Higher values give more accurate entropy estimates at more compute cost.
    rng_seed : Optional[int]
        Seed for the ProbabilityEngine's sampler (reproducibility).
    fallback_seed : Optional[int]
        Seed for the fallback RNG (used only in degenerate states).
    """

    def __init__(
        self,
        n_samples: int = 500,
        rng_seed: Optional[int] = None,
        fallback_seed: Optional[int] = None,
    ) -> None:
        self.n_samples = n_samples
        self._engine = ProbabilityEngine(n_samples=n_samples, rng_seed=rng_seed)
        self._fallback_rng = stdlib_random.Random(fallback_seed)

        # Diagnostic state (updated each call to select_action).
        self.last_prob_map: Optional[np.ndarray] = None
        self.last_expected_H: Optional[np.ndarray] = None
        self.last_info_gain_map: Optional[np.ndarray] = None
        self.last_n_samples: int = 0

    @property
    def name(self) -> str:
        return f"Entropy(n={self.n_samples})"

    # ------------------------------------------------------------------ #
    # ABC implementation                                                   #
    # ------------------------------------------------------------------ #

    def select_action(self, view: GameView) -> Tuple[int, int]:
        """
        Choose the shot that minimises expected post-shot board entropy.

        The decision procedure:
          1. Sample K valid configurations from the hypothesis space.
          2. Compute marginal probability map from the K samples.
          3. Compute joint occupancy matrix.
          4. Derive E[H | shoot(a)] for every unfired cell a.
          5. Return argmin_a E[H | shoot(a)].

        Falls back to highest-probability cell if entropy computation
        produces degenerate results, and to random if all else fails.

        Returns
        -------
        (row, col) : the recommended next shot.
        """
        B = view.board_size
        configs = self._engine.get_configs(view)
        K = len(configs)
        self.last_n_samples = K

        # ---------------------------------------------------------- #
        # Compute probability map (always, even for fallback)        #
        # ---------------------------------------------------------- #

        prob_map = self._engine.compute_prob_map(view)
        self.last_prob_map = prob_map

        if K == 0:
            # No valid configurations sampled — degenerate state.
            # Fall back to any unfired cell.
            self.last_expected_H = None
            self.last_info_gain_map = None
            return self._fallback_action(view, prob_map)

        # ---------------------------------------------------------- #
        # Compute expected entropy for every unfired cell             #
        # ---------------------------------------------------------- #

        expected_H = compute_expected_entropies(configs, view)
        self.last_expected_H = expected_H

        # ---------------------------------------------------------- #
        # Derive information gain map without recomputing entropy     #
        # ---------------------------------------------------------- #

        current_H = board_entropy(prob_map, view)
        # IG(a) = H_current - E[H | shoot(a)].  Fired cells get -inf.
        from engine.board import CellState as _CS
        ig = current_H - expected_H
        ig = np.where(view.shot_grid == _CS.UNKNOWN, ig, -np.inf)
        self.last_info_gain_map = ig

        # ---------------------------------------------------------- #
        # Choose action: argmin expected entropy over unfired cells  #
        # ---------------------------------------------------------- #

        unfired_mask = view.shot_grid == CellState.UNKNOWN

        # Check that at least one unfired cell has finite expected H.
        if not np.any(np.isfinite(expected_H) & unfired_mask):
            return self._fallback_action(view, prob_map)

        # Mask fired cells to +inf so argmin ignores them.
        masked_H = np.where(unfired_mask, expected_H, np.inf)

        # Tie-break: when multiple cells have the same expected H (common
        # early game), prefer the cell with the highest marginal probability.
        # Implemented by adding a tiny prob-based perturbation.
        tiebreak = -prob_map * 1e-9
        masked_H_tb = masked_H + tiebreak

        best_flat = int(np.argmin(masked_H_tb))
        row, col = divmod(best_flat, B)
        return (row, col)

    def reset(self) -> None:
        """Clear per-game state.  Must be called between games."""
        self._engine.reset()
        self.last_prob_map = None
        self.last_expected_H = None
        self.last_info_gain_map = None
        self.last_n_samples = 0

    # ------------------------------------------------------------------ #
    # Fallback                                                             #
    # ------------------------------------------------------------------ #

    def _fallback_action(
        self, view: GameView, prob_map: Optional[np.ndarray]
    ) -> Tuple[int, int]:
        """
        Emergency fallback: return highest-probability unfired cell, or
        random if prob_map gives no useful signal.
        """
        B = view.board_size
        unfired = view.unfired_cells

        if not unfired:
            raise RuntimeError("EntropyStrategy: no unfired cells remain.")

        if prob_map is not None and prob_map.max() > 0:
            # Find unfired cell with highest marginal probability.
            masked = np.where(
                view.shot_grid == CellState.UNKNOWN, prob_map, -1.0
            )
            best_flat = int(np.argmax(masked))
            row, col = divmod(best_flat, B)
            if view.shot_grid[row, col] == CellState.UNKNOWN:
                return (row, col)

        # Ultimate fallback: random unfired cell.
        return self._fallback_rng.choice(unfired)

    # ------------------------------------------------------------------ #
    # Diagnostic helpers                                                   #
    # ------------------------------------------------------------------ #

    def current_board_entropy(self, view: GameView) -> float:
        """
        Return the current approximate board entropy H_board (in nats).

        Computed from the cached probability map if available.
        """
        prob_map = self.last_prob_map
        if prob_map is None:
            prob_map = self._engine.compute_prob_map(view)
        return board_entropy(prob_map, view)

    def probability_heatmap(self, view: GameView) -> np.ndarray:
        """
        Return the (B, B) probability heatmap for the current state.

        Forces computation if not already cached for this turn.
        """
        return self._engine.compute_prob_map(view)
