"""
simulation/runner.py
--------------------
GameRunner: drives automated games by coupling a Strategy to a Game.

Design notes:
------------
GameRunner is the only place where Strategy and Game interact.  It:
  1. Creates a fresh Game (with a unique seed derived from base_seed + game_index)
  2. Calls strategy.reset() to clear per-game state
  3. Loops: builds a GameView → calls strategy.select_action() → calls game.fire()
  4. Packages the GameResult

Per-game seeds are deterministic: seed_i = base_seed + i (when base_seed is
provided).  This gives full reproducibility while ensuring each game has an
independent board layout.

run_batch() returns a list of GameResult — callers (analysis layer, Phase 4)
convert these to DataFrames.  Keeping raw GameResult objects here avoids
coupling runner to pandas.

Parallelization note:
  Phase 1 is single-threaded.  The interface is designed so that run_batch()
  can be replaced with a multiprocessing.Pool.starmap() call in Phase 5
  without changing the Strategy or Game APIs.
"""

from __future__ import annotations

import time
from typing import List, Optional

from engine.game import Game, GameResult
from engine.ships import ShipType, STANDARD_FLEET
from engine.board import BOARD_SIZE
from strategies.base import Strategy, GameView


class GameRunner:
    """
    Runs one or many automated games using a given Strategy.

    Parameters
    ----------
    fleet      : ships to place (default: standard 5-ship fleet)
    board_size : board edge length (default: 10)
    """

    def __init__(
        self,
        fleet: List[ShipType] = STANDARD_FLEET,
        board_size: int = BOARD_SIZE,
    ) -> None:
        self.fleet = list(fleet)
        self.board_size = board_size

    # ------------------------------------------------------------------ #
    # Single game                                                          #
    # ------------------------------------------------------------------ #

    def run_game(
        self,
        strategy: Strategy,
        seed: Optional[int] = None,
    ) -> GameResult:
        """
        Run a single complete game and return its GameResult.

        Parameters
        ----------
        strategy : Strategy instance.  reset() is called before play begins.
        seed     : Board placement seed.  None → random (not reproducible).
        """
        game = Game(fleet=self.fleet, board_size=self.board_size, seed=seed)
        strategy.reset()

        while not game.is_over:
            view = GameView.from_game(game)
            row, col = strategy.select_action(view)
            game.fire(row, col)

        return game.get_result()

    # ------------------------------------------------------------------ #
    # Batch                                                                #
    # ------------------------------------------------------------------ #

    def run_batch(
        self,
        strategy: Strategy,
        n_games: int,
        base_seed: Optional[int] = None,
        verbose: bool = False,
    ) -> List[GameResult]:
        """
        Run *n_games* games and return all GameResults.

        Seeds are assigned as base_seed + i for i in [0, n_games) when
        base_seed is provided, guaranteeing independent reproducible boards.
        If base_seed is None, games are non-reproducible.

        Parameters
        ----------
        strategy   : Strategy instance (reused across games; reset() called each time).
        n_games    : Number of games to simulate.
        base_seed  : Starting seed for deterministic experiments.
        verbose    : Print progress every 10% of games.
        """
        results: List[GameResult] = []
        milestone = max(1, n_games // 10)
        t_start = time.perf_counter()

        for i in range(n_games):
            seed = (base_seed + i) if base_seed is not None else None
            result = self.run_game(strategy, seed=seed)
            results.append(result)

            if verbose and (i + 1) % milestone == 0:
                elapsed = time.perf_counter() - t_start
                rate = (i + 1) / elapsed
                print(
                    f"  [{strategy.name}] {i+1:>{len(str(n_games))}}/{n_games} games "
                    f"| {rate:.0f} games/s "
                    f"| avg shots: {sum(r.total_shots for r in results) / len(results):.1f}"
                )

        return results

    # ------------------------------------------------------------------ #
    # Quick summary                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def summarize(results: List[GameResult]) -> dict:
        """
        Compute basic statistics over a batch of results.

        Returns a dict suitable for printing or DataFrame construction.
        Heavier analysis (confidence intervals, distribution fitting) lives
        in the analysis layer (Phase 4).
        """
        if not results:
            return {}

        shots = [r.total_shots for r in results]
        n = len(shots)
        mean = sum(shots) / n
        variance = sum((s - mean) ** 2 for s in shots) / (n - 1) if n > 1 else 0.0
        std = variance ** 0.5
        shots_sorted = sorted(shots)

        return {
            "n_games": n,
            "mean_shots": round(mean, 3),
            "std_shots": round(std, 3),
            "min_shots": shots_sorted[0],
            "median_shots": shots_sorted[n // 2],
            "max_shots": shots_sorted[-1],
            "mean_hit_rate": round(sum(r.hit_rate for r in results) / n, 4),
        }
