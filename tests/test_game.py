"""
tests/test_game.py
------------------
Tests for Game orchestration, ShotRecord, GameResult, and GameRunner.
"""

import random

import pytest

from engine.board import CellState
from engine.game import Game, GameResult, ShotRecord
from engine.ships import ShipType, STANDARD_FLEET
from strategies.base import Strategy, GameView
from simulation.runner import GameRunner


# --------------------------------------------------------------------------- #
# Minimal concrete strategy for testing (fires cells left-to-right, top-down) #
# --------------------------------------------------------------------------- #


class SweepStrategy(Strategy):
    """Fires every cell in row-major order.  Guaranteed to finish any game."""

    @property
    def name(self) -> str:
        return "Sweep"

    def select_action(self, view: GameView):
        return view.unfired_cells[0]


class RandomStrategy(Strategy):
    """Fires randomly from unfired cells.  Deterministic given RNG seed."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "Random"

    def reset(self) -> None:
        pass  # rng state not reset — intentional for variety

    def select_action(self, view: GameView):
        cells = view.unfired_cells
        return cells[self._rng.randrange(len(cells))]


# --------------------------------------------------------------------------- #
# Game basics                                                                  #
# --------------------------------------------------------------------------- #


class TestGame:
    def test_game_starts_not_over(self):
        game = Game(seed=1)
        assert not game.is_over
        assert game.turn == 0

    def test_fire_increments_turn(self):
        game = Game(seed=2)
        game.fire(0, 0)
        assert game.turn == 1
        game.fire(0, 1)
        assert game.turn == 2

    def test_fire_after_game_over_raises(self):
        strategy = SweepStrategy()
        runner = GameRunner()
        game = Game(seed=3)
        # Play to completion manually via sweep
        cells = [(r, c) for r in range(10) for c in range(10)]
        for r, c in cells:
            if game.is_over:
                break
            game.fire(r, c)
        assert game.is_over
        with pytest.raises(RuntimeError, match="already over"):
            game.fire(0, 0)

    def test_shot_records_length(self):
        game = Game(seed=4)
        # Fire 5 shots
        cells = [(r, c) for r in range(10) for c in range(10)][:5]
        for r, c in cells:
            game.fire(r, c)
        assert len(game.shot_records) == 5

    def test_shot_records_turn_numbering(self):
        game = Game(seed=5)
        game.fire(0, 0)
        game.fire(1, 0)
        records = game.shot_records
        assert records[0].turn == 1
        assert records[1].turn == 2

    def test_reset_clears_state(self):
        game = Game(seed=6)
        game.fire(0, 0)
        game.reset()
        assert game.turn == 0
        assert game.shot_records == []
        assert not game.is_over

    def test_reset_with_new_seed(self):
        game = Game(seed=7)
        original_grid = game.board.ship_grid.copy()
        game.reset(new_seed=999)
        # Different seed → board may differ (not guaranteed but probable)
        assert game.seed == 999

    def test_reproducibility(self):
        """Same seed → identical board layout."""
        import numpy as np
        g1 = Game(seed=42)
        g2 = Game(seed=42)
        np.testing.assert_array_equal(g1.board.ship_grid, g2.board.ship_grid)


# --------------------------------------------------------------------------- #
# ShotRecord                                                                   #
# --------------------------------------------------------------------------- #


class TestShotRecord:
    def test_is_hit_true_for_hit(self):
        r = ShotRecord(1, 0, 0, CellState.HIT)
        assert r.is_hit()

    def test_is_hit_true_for_sunk(self):
        r = ShotRecord(1, 0, 0, CellState.SUNK)
        assert r.is_hit()

    def test_is_hit_false_for_miss(self):
        r = ShotRecord(1, 0, 0, CellState.MISS)
        assert not r.is_hit()

    def test_immutable(self):
        r = ShotRecord(1, 0, 0, CellState.MISS)
        with pytest.raises((AttributeError, TypeError)):
            r.turn = 99


# --------------------------------------------------------------------------- #
# GameResult                                                                   #
# --------------------------------------------------------------------------- #


class TestGameResult:
    def _complete_game(self, seed: int = 0) -> Game:
        game = Game(seed=seed)
        for r in range(10):
            for c in range(10):
                if game.is_over:
                    return game
                game.fire(r, c)
        return game

    def test_get_result_raises_if_not_over(self):
        game = Game(seed=1)
        with pytest.raises(RuntimeError, match="not over"):
            game.get_result()

    def test_result_total_shots_matches_turn(self):
        game = self._complete_game(seed=10)
        result = game.get_result()
        assert result.total_shots == game.turn

    def test_result_shots_to_sink_length(self):
        game = self._complete_game(seed=11)
        result = game.get_result()
        assert len(result.shots_to_sink) == len(STANDARD_FLEET)

    def test_result_shots_to_sink_monotone(self):
        game = self._complete_game(seed=12)
        result = game.get_result()
        for i in range(1, len(result.shots_to_sink)):
            assert result.shots_to_sink[i] > result.shots_to_sink[i - 1]

    def test_result_shots_to_sink_last_equals_total(self):
        game = self._complete_game(seed=13)
        result = game.get_result()
        assert result.shots_to_sink[-1] == result.total_shots

    def test_hits_plus_misses_equals_total(self):
        game = self._complete_game(seed=14)
        result = game.get_result()
        assert result.hits + result.misses == result.total_shots

    def test_hit_rate_range(self):
        game = self._complete_game(seed=15)
        result = game.get_result()
        assert 0.0 < result.hit_rate <= 1.0

    def test_to_dict_keys(self):
        game = self._complete_game(seed=16)
        result = game.get_result()
        d = result.to_dict()
        assert "total_shots" in d
        assert "hits" in d
        assert "misses" in d
        assert "hit_rate" in d

    def test_result_immutable(self):
        game = self._complete_game(seed=17)
        result = game.get_result()
        with pytest.raises((AttributeError, TypeError)):
            result.total_shots = 0


# --------------------------------------------------------------------------- #
# GameView                                                                     #
# --------------------------------------------------------------------------- #


class TestGameView:
    def test_view_from_fresh_game(self):
        game = Game(seed=1)
        view = GameView.from_game(game)
        assert view.turn == 0
        assert view.unfired_count == 100
        assert view.hit_cells == []
        assert len(view.afloat_ships) == 5

    def test_view_shot_grid_is_copy(self):
        """Mutating the view's shot_grid must not affect the live board."""
        import numpy as np
        game = Game(seed=2)
        view = GameView.from_game(game)
        view.shot_grid[0, 0] = 99  # noqa — intentional mutation of copy
        assert game.board.shot_grid[0, 0] == 0  # board unaffected

    def test_view_updates_after_shot(self):
        game = Game(seed=3)
        game.fire(0, 0)
        view = GameView.from_game(game)
        assert view.turn == 1
        assert view.unfired_count == 99

    def test_view_immutable(self):
        game = Game(seed=4)
        view = GameView.from_game(game)
        with pytest.raises((AttributeError, TypeError)):
            view.turn = 99


# --------------------------------------------------------------------------- #
# GameRunner                                                                   #
# --------------------------------------------------------------------------- #


class TestGameRunner:
    def test_run_game_completes(self):
        runner = GameRunner()
        result = runner.run_game(SweepStrategy(), seed=1)
        assert result.total_shots > 0
        assert len(result.shots_to_sink) == 5

    def test_run_game_reproducible(self):
        runner = GameRunner()
        r1 = runner.run_game(SweepStrategy(), seed=42)
        r2 = runner.run_game(SweepStrategy(), seed=42)
        assert r1.total_shots == r2.total_shots
        assert r1.shots_to_sink == r2.shots_to_sink

    def test_run_batch_length(self):
        runner = GameRunner()
        results = runner.run_batch(SweepStrategy(), n_games=20, base_seed=0)
        assert len(results) == 20

    def test_run_batch_all_complete(self):
        runner = GameRunner()
        results = runner.run_batch(SweepStrategy(), n_games=10, base_seed=0)
        for r in results:
            assert len(r.shots_to_sink) == 5

    def test_run_batch_seeds_are_independent(self):
        """Each game in batch should have a different board layout."""
        runner = GameRunner()
        results = runner.run_batch(SweepStrategy(), n_games=5, base_seed=0)
        # Total shots may vary — just confirm seeds differ
        seeds = [r.seed for r in results]
        assert len(set(seeds)) == 5

    def test_summarize_keys(self):
        runner = GameRunner()
        results = runner.run_batch(SweepStrategy(), n_games=5, base_seed=0)
        summary = runner.summarize(results)
        for key in ("n_games", "mean_shots", "std_shots", "min_shots", "max_shots"):
            assert key in summary

    def test_sweep_all_shots_leq_100(self):
        """Sweep can never need more than 100 shots on a 10×10 board."""
        runner = GameRunner()
        results = runner.run_batch(SweepStrategy(), n_games=50, base_seed=100)
        assert all(r.total_shots <= 100 for r in results)

    def test_random_strategy_finishes(self):
        runner = GameRunner()
        strategy = RandomStrategy(seed=7)
        result = runner.run_game(strategy, seed=77)
        assert result.total_shots > 0
        assert len(result.shots_to_sink) == 5
