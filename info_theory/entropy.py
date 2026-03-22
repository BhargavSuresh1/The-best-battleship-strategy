"""
info_theory/entropy.py
-----------------------
Shannon entropy computations for the information-theoretic Battleship strategy.

Mathematical framework
-----------------------
We maintain K sampled configurations ω_1, …, ω_K of the hypothesis space Ω.
Under the uniform prior, each configuration has equal weight 1/K.

State entropy (board-level)
----------------------------
Rather than computing H(Ω) = log|Ω| (which requires knowing |Ω| exactly),
we use a cell-level approximation: sum of binary entropies over unfired cells.

For cell c with marginal occupancy probability p_c:

    h(p_c) = -p_c * log(p_c) - (1 - p_c) * log(1 - p_c)

    H_board ≈ Σ_{c: UNKNOWN} h(p_c)

This is an upper bound on the true state entropy (by independence relaxation)
and serves as a consistent proxy for expected information gain comparisons.

Expected entropy after an action (key formula)
-----------------------------------------------
For a candidate shot at cell a, the expected residual entropy is:

    E[H | shoot(a)] = P(hit at a) * H_after_hit(a) + P(miss at a) * H_after_miss(a)

where:
    P(hit at a)  = p_a  (marginal probability, directly from samples)
    H_after_hit  = board entropy under the posterior P(· | ship at a)
    H_after_miss = board entropy under the posterior P(· | no ship at a)

Conditional probability of cell b given outcome at a:

    P(b | hit at a)  = P(b AND a occupied) / P(a occupied)
                     ≈ joint[a,b] / marginal[a]

    P(b | miss at a) = P(b AND a not occupied) / P(a not occupied)
                     = (P(b) - P(b AND a)) / (1 - P(a))
                     ≈ (marginal[b] - joint[a,b]) / (1 - marginal[a])

where joint[a,b] = fraction of samples in which both a and b are occupied.

Vectorised computation
-----------------------
The naive approach loops over all M unfired cells, which is O(M * K).
We instead compute the full (M × M) joint matrix once:

    joint = (configs_unfired.T @ configs_unfired) / K   [O(K * M^2)]

This enables fully vectorised conditional probability computation, reducing
per-cell entropy evaluation to O(M^2) NumPy operations.

Total complexity per turn: O(K * M^2) dominated by the matrix multiply.
For K=500, M=100: ~5M multiply-adds → ~1 ms on modern hardware.

Information gain vs. expected entropy
--------------------------------------
We optimise for MINIMUM expected entropy, which is equivalent to maximising
EXPECTED INFORMATION GAIN:

    IG(a) = H_current - E[H | shoot(a)]

since H_current is constant for all candidate cells.  The EntropyStrategy
returns argmin_a E[H | shoot(a)].
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from engine.board import CellState
from strategies.base import GameView


# --------------------------------------------------------------------------- #
# Elementary entropy functions                                                 #
# --------------------------------------------------------------------------- #

# Minimum probability treated as nonzero (avoids log(0) while keeping the
# binary entropy near-zero for very certain cells).
_EPS: float = 1e-10


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """
    Vectorised binary (Bernoulli) Shannon entropy in nats.

        h(p) = -p * ln(p) - (1-p) * ln(1-p)

    Parameters
    ----------
    p : array-like of float in [0, 1]

    Returns
    -------
    h : np.ndarray, same shape as p.
        h(0) = h(1) = 0 by convention; h(0.5) = ln 2 ≈ 0.693.
    """
    p = np.asarray(p, dtype=np.float64)
    # Clamp to avoid log(0); clamp brings 0→ε and 1→1-ε so h≈0 at extremes.
    p_safe = np.clip(p, _EPS, 1.0 - _EPS)
    return -p_safe * np.log(p_safe) - (1.0 - p_safe) * np.log(1.0 - p_safe)


def board_entropy(prob_map: np.ndarray, view: Optional[GameView] = None) -> float:
    """
    Approximate Shannon entropy of the current board state (in nats).

    Computed as the sum of binary entropies over uncertain cells:

        H_board = Σ_{c: UNKNOWN} h(prob_map[c])

    This is an independence-relaxed upper bound on the true H(Ω).

    Parameters
    ----------
    prob_map : np.ndarray, shape (B, B)
        Marginal occupancy probabilities.  Fired cells must already be set
        to 0 (or will naturally contribute ~0 to the sum).
    view : GameView, optional
        If provided, only sums over UNKNOWN cells (correct).  If None,
        sums over all cells with p ∈ (0, 1), which is equivalent when
        prob_map is zeroed on fired cells.

    Returns
    -------
    H : float (nats)
    """
    if view is not None:
        mask = view.shot_grid == CellState.UNKNOWN
        p = prob_map[mask]
    else:
        # Include all cells with non-trivial probability.
        p = prob_map.ravel()

    return float(np.sum(binary_entropy(p)))


# --------------------------------------------------------------------------- #
# Expected entropy per candidate action                                        #
# --------------------------------------------------------------------------- #


def compute_expected_entropies(
    configs: np.ndarray,
    view: GameView,
) -> np.ndarray:
    """
    Compute E[H_board | shoot at cell a] for every unfired cell a.

    This is the core computation of the information-theoretic strategy.
    Lower values indicate more informative shots.

    Algorithm
    ---------
    1. Flatten to (K, M) matrix of float32 (M = number of unfired cells).
    2. Compute (M, M) joint occupancy matrix via matrix multiplication.
    3. Derive conditional probability matrices P(b|hit at a) and P(b|miss at a).
    4. Apply binary entropy elementwise; sum across b ≠ a per row.
    5. Weight by P(hit) and P(miss); store as expected entropy per cell.
    6. Fired cells receive +inf (never chosen by argmin).

    Parameters
    ----------
    configs : np.ndarray, shape (K, B, B), dtype=bool
        K sampled valid configurations from ProbabilityEngine.
    view : GameView
        Current observable game state.

    Returns
    -------
    expected_H : np.ndarray, shape (B, B), dtype=float64
        expected_H[r, c] = E[H_board | shoot at (r, c)].
        Fired cells contain +inf.
        UNKNOWN cells contain the expected post-shot entropy (in nats).

    Notes
    -----
    When K == 0 (no valid samples), all UNKNOWN cells receive +inf and the
    caller should fall back to a simpler strategy.
    """
    B = view.board_size
    result = np.full((B, B), np.inf, dtype=np.float64)

    K = len(configs)
    if K == 0:
        return result

    # ------------------------------------------------------------------ #
    # Step 1: identify unfired cells and extract relevant sub-matrix      #
    # ------------------------------------------------------------------ #

    unfired_mask = (view.shot_grid == CellState.UNKNOWN)      # (B, B) bool
    unfired_flat = unfired_mask.ravel()                        # (B*B,) bool
    unfired_idx = np.where(unfired_flat)[0]                    # (M,) int
    M = len(unfired_idx)

    if M == 0:
        return result

    # Extract only the unfired columns from configs; cast to float32 for speed.
    cfgs = configs.reshape(K, B * B)[:, unfired_idx].astype(np.float32)  # (K, M)

    # ------------------------------------------------------------------ #
    # Step 2: marginal and joint occupancy probabilities                  #
    # ------------------------------------------------------------------ #

    # marginal[j] = P(unfired_cell_j is occupied)
    marginal = cfgs.mean(axis=0).astype(np.float64)   # (M,)

    # joint[i, j] = P(cells i and j are both occupied)
    # = (cfgs[:, i] · cfgs[:, j]) / K
    # Computed as cfgs.T @ cfgs / K  [BLAS-optimised matrix multiply]
    joint = (cfgs.T.astype(np.float64) @ cfgs.astype(np.float64)) / K  # (M, M)

    # ------------------------------------------------------------------ #
    # Step 3: conditional probability matrices                            #
    # ------------------------------------------------------------------ #

    eps = _EPS
    # Safe denominators: clamp marginals away from 0 and 1.
    denom_hit  = np.clip(marginal, eps, 1.0)[:, None]          # (M, 1)
    denom_miss = np.clip(1.0 - marginal, eps, 1.0)[:, None]    # (M, 1)

    # p_hit_cond[i, j] = P(cell j occupied | shot at i is a HIT)
    p_hit_cond = joint / denom_hit                              # (M, M)

    # p_miss_cond[i, j] = P(cell j occupied | shot at i is a MISS)
    # = P(j AND NOT i) / P(NOT i)
    # = (marginal[j] - joint[i, j]) / (1 - marginal[i])
    p_miss_cond = (marginal[None, :] - joint) / denom_miss      # (M, M)

    # Numerical clamp: conditional probabilities must lie in [0, 1].
    np.clip(p_hit_cond,  0.0, 1.0, out=p_hit_cond)
    np.clip(p_miss_cond, 0.0, 1.0, out=p_miss_cond)

    # ------------------------------------------------------------------ #
    # Step 4: zero out the diagonal (shot cell itself becomes known)      #
    # ------------------------------------------------------------------ #
    #
    # After shooting cell a, cell a is no longer UNKNOWN.
    # h(p_aa | hit)  = h(1) = 0 and h(p_aa | miss) = h(0) = 0,
    # so zeroing the diagonal gives the correct entropy contribution (0).
    # This also prevents h(1) ≈ tiny-positive from the eps-clamp below.

    diag_idx = np.arange(M)
    p_hit_cond[diag_idx, diag_idx]  = 0.0
    p_miss_cond[diag_idx, diag_idx] = 0.0

    # ------------------------------------------------------------------ #
    # Step 5: binary entropy of conditional probability matrices          #
    # ------------------------------------------------------------------ #

    h_hit_cond  = binary_entropy(p_hit_cond)    # (M, M)
    h_miss_cond = binary_entropy(p_miss_cond)   # (M, M)

    # Sum across other cells j (axis=1).
    # H_after_hit[i]  = Σ_{j≠i} h(P(j occupied | hit at i))
    # H_after_miss[i] = Σ_{j≠i} h(P(j occupied | miss at i))
    H_hit  = h_hit_cond.sum(axis=1)            # (M,)
    H_miss = h_miss_cond.sum(axis=1)           # (M,)

    # ------------------------------------------------------------------ #
    # Step 6: expected entropy — weight by P(hit) and P(miss)            #
    # ------------------------------------------------------------------ #

    expected_H_unfired = marginal * H_hit + (1.0 - marginal) * H_miss  # (M,)

    # ------------------------------------------------------------------ #
    # Step 7: map back to (B, B) result grid                              #
    # ------------------------------------------------------------------ #

    result_flat = result.ravel()
    result_flat[unfired_idx] = expected_H_unfired
    return result.reshape(B, B)


def information_gain_map(
    configs: np.ndarray,
    view: GameView,
    current_H: Optional[float] = None,
) -> np.ndarray:
    """
    Compute expected information gain IG(a) = H_current - E[H | shoot(a)]
    for every unfired cell.

    Parameters
    ----------
    configs : np.ndarray, shape (K, B, B)
    view : GameView
    current_H : float, optional
        Pre-computed current board entropy.  If None, computed from configs.

    Returns
    -------
    ig_map : np.ndarray, shape (B, B)
        ig_map[r, c] = expected information gained by shooting (r, c).
        Fired cells contain -inf.
        Higher values → more informative shots.
    """
    expected_H = compute_expected_entropies(configs, view)

    if current_H is None:
        B = view.board_size
        if len(configs) > 0:
            prob_map = configs.mean(axis=0).astype(np.float64)
            fired_mask = view.shot_grid != CellState.UNKNOWN
            prob_map[fired_mask] = 0.0
            current_H = board_entropy(prob_map, view)
        else:
            current_H = 0.0

    ig = current_H - expected_H           # +inf for fired cells becomes -inf
    ig = np.where(np.isinf(expected_H), -np.inf, ig)
    return ig
