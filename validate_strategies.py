"""
validate_strategies.py
-----------------------
Standalone validation script for all four Phase-2 strategies.

Usage
-----
    python validate_strategies.py            # default: 500 games per strategy
    python validate_strategies.py --n 200    # faster run
    python validate_strategies.py --n 1000   # higher statistical power

Output
------
Prints a formatted table:

    Strategy       | Games | Mean shots | Std  | Min | Median | Max | Hit rate
    Random         |   500 |      95.4  | 5.7  |  72 |    96  | 100 |  17.8 %
    HuntTarget     |   500 |      55.3  | 8.2  |  35 |    55  |  88 |  30.8 %
    Parity         |   500 |      43.1  | 7.5  |  24 |    42  |  76 |  39.5 %
    Entropy(n=300) |   500 |      41.2  | 7.0  |  22 |    40  |  70 |  41.3 %

Expected ordering: Random > HuntTarget > Parity > Entropy  (mean shots).

Each strategy is run on the same board seeds (base_seed=0) for fair comparison.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import List

from engine.ships import STANDARD_FLEET
from engine.board import BOARD_SIZE
from engine.game import GameResult
from simulation.runner import GameRunner
from strategies.random_strategy import RandomStrategy
from strategies.hunt_target import HuntTargetStrategy
from strategies.parity import ParityStrategy
from strategies.entropy_strategy import EntropyStrategy


# --------------------------------------------------------------------------- #
# Configuration                                                                #
# --------------------------------------------------------------------------- #

DEFAULT_N_GAMES = 500
BASE_SEED = 0

# EntropyStrategy n_samples: balance quality vs speed for validation.
# 300 gives good accuracy; reduce to 100 for quick validation.
ENTROPY_N_SAMPLES = 300


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #


def run_strategy(strategy, n_games: int, base_seed: int) -> tuple[List[GameResult], float]:
    """Run *n_games* games and return (results, elapsed_seconds)."""
    runner = GameRunner(fleet=STANDARD_FLEET, board_size=BOARD_SIZE)
    t0 = time.perf_counter()
    results = runner.run_batch(strategy, n_games, base_seed=base_seed, verbose=False)
    elapsed = time.perf_counter() - t0
    return results, elapsed


def summarise(results: List[GameResult]) -> dict:
    shots = [r.total_shots for r in results]
    n = len(shots)
    mean = statistics.mean(shots)
    std = statistics.stdev(shots) if n > 1 else 0.0
    sorted_shots = sorted(shots)
    median = sorted_shots[n // 2]
    hit_rate = statistics.mean(r.hit_rate for r in results)
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "min": sorted_shots[0],
        "median": median,
        "max": sorted_shots[-1],
        "hit_rate": hit_rate,
    }


# --------------------------------------------------------------------------- #
# Formatted output                                                             #
# --------------------------------------------------------------------------- #

HEADER = (
    f"{'Strategy':<22} | {'Games':>5} | {'Mean':>8} | {'Std':>5} | "
    f"{'Min':>4} | {'Median':>6} | {'Max':>4} | {'Hit rate':>9} | {'Time':>7}"
)
SEP = "-" * len(HEADER)


def print_row(name: str, s: dict, elapsed: float) -> None:
    print(
        f"{name:<22} | {s['n']:>5} | {s['mean']:>8.2f} | {s['std']:>5.2f} | "
        f"{s['min']:>4} | {s['median']:>6} | {s['max']:>4} | "
        f"{s['hit_rate']:>8.1%} | {elapsed:>6.1f}s"
    )


# --------------------------------------------------------------------------- #
# Validation assertions                                                        #
# --------------------------------------------------------------------------- #


def validate_ordering(means: dict[str, float]) -> bool:
    """
    Return True if the expected ordering holds:
        Random > HuntTarget > Parity > Entropy
    """
    order = ["Random", "HuntTarget", "Parity"]
    entropy_key = [k for k in means if k.startswith("Entropy")][0]
    order.append(entropy_key)

    passed = True
    for i in range(len(order) - 1):
        better = order[i]
        worse = order[i + 1]
        if means[better] <= means[worse]:
            print(
                f"  [ORDERING FAIL] {better} ({means[better]:.2f}) should be > "
                f"{worse} ({means[worse]:.2f})"
            )
            passed = False
        else:
            print(
                f"  [OK] {better} ({means[better]:.2f}) > {worse} ({means[worse]:.2f})"
            )
    return passed


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Phase-2 Battleship strategies."
    )
    parser.add_argument(
        "--n", type=int, default=DEFAULT_N_GAMES,
        help=f"Games per strategy (default: {DEFAULT_N_GAMES})"
    )
    parser.add_argument(
        "--entropy-samples", type=int, default=ENTROPY_N_SAMPLES,
        help=f"n_samples for EntropyStrategy (default: {ENTROPY_N_SAMPLES})"
    )
    parser.add_argument(
        "--seed", type=int, default=BASE_SEED,
        help=f"Base seed for board generation (default: {BASE_SEED})"
    )
    args = parser.parse_args()

    n = args.n
    seed = args.seed
    es_n = args.entropy_samples

    strategies = [
        RandomStrategy(seed=1),
        HuntTargetStrategy(seed=1),
        ParityStrategy(seed=1),
        EntropyStrategy(n_samples=es_n, rng_seed=1),
    ]

    print(f"\nBattleship Strategy Validation")
    print(f"  Board: {BOARD_SIZE}×{BOARD_SIZE}, Fleet: {[st.name for st in STANDARD_FLEET]}")
    print(f"  Games per strategy: {n}")
    print(f"  Base seed: {seed}")
    print(f"  EntropyStrategy n_samples: {es_n}")
    print()
    print(HEADER)
    print(SEP)

    means = {}
    for strategy in strategies:
        results, elapsed = run_strategy(strategy, n, base_seed=seed)
        s = summarise(results)
        print_row(strategy.name, s, elapsed)
        means[strategy.name] = s["mean"]

    print(SEP)
    print()
    print("Ordering validation:")
    ok = validate_ordering(means)
    print()
    if ok:
        print("All ordering assertions PASSED.")
    else:
        print("WARNING: Some ordering assertions FAILED.")
        print("  This may be due to insufficient n_games or high variance.")
        print("  Try --n 1000 for more reliable results.")


if __name__ == "__main__":
    main()
