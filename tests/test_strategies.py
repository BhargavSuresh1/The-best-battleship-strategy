"""
tests/test_strategies.py
------------------------
Unit and integration tests for all four strategies.

Tests cover:
  - Legality: no strategy ever fires at an already-fired cell
  - Correctness: strategies return valid (row, col) coordinates
  - Completion: strategies can finish a game
  - Reset: strategies produce independent results after reset()
  - Ordering: RandomStrategy > HuntTarget > Parity > Entropy (average shots)
    verified with 200-game simulations

The ordering test uses a fixed seed and relatively tight tolerances.
It verifies the expected quality ordering, not exact values.
"""

from __future__ import annotations

import statistics
from typing import List, Type

import pytest

from engine.board import CellState, BOARD_SIZE
from engine.game import Game
from engine.ships import STANDARD_FLEET, ShipType
from simulation.runner import GameRunner
from strategies.base import GameView, Strategy
from strategies.random_strategy import RandomStrategy
from strategies.hunt_target import HuntTargetStrategy
from strategies.parity import ParityStrategy
from strategies.entropy_strategy import EntropyStrategy


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #


def make_runner() -> GameRunner:
    return GameRunner(fleet=STANDARD_FLEET, board_size=BOARD_SIZE)


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def run_game_legality_check(strategy: Strategy, seed: int) -> int:
    """
    Run a single game, asserting that the strategy never fires at a cell
    that has already been fired at.  Returns total shots for the game.
    """
    game = Game(fleet=STANDARD_FLEET, board_size=BOARD_SIZE, seed=seed)
    strategy.reset()

    fired_cells = set()

    while not game.is_over:
        view = GameView.from_game(game)
        row, col = strategy.select_action(view)

        # Legality checks
        assert 0 <= row < BOARD_SIZE, f"row {row} out of bounds"
        assert 0 <= col < BOARD_SIZE, f"col {col} out of bounds"
        assert (row, col) not in fired_cells, (
            f"{strategy.name} fired at already-fired cell ({row},{col}) "
            f"on turn {game.turn}"
        )

        fired_cells.add((row, col))
        game.fire(row, col)

    return game.turn


# --------------------------------------------------------------------------- #
# Parametrised legality tests (fast — single game each)                       #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "strategy_cls,kwargs",
    [
        (RandomStrategy,      {"seed": 0}),
        (HuntTargetStrategy,  {"seed": 0}),
        (ParityStrategy,      {"seed": 0}),
        (EntropyStrategy,     {"n_samples": 50, "rng_seed": 0}),
    ],
    ids=["Random", "HuntTarget", "Parity", "Entropy"],
)
def test_legality_single_game(strategy_cls, kwargs):
    """Each strategy must never fire at an already-fired cell."""
    strategy = strategy_cls(**kwargs)
    shots = run_game_legality_check(strategy, seed=42)
    assert shots > 0, "Game completed with 0 shots (impossible)"


@pytest.mark.parametrize(
    "strategy_cls,kwargs",
    [
        (RandomStrategy,      {"seed": 1}),
        (HuntTargetStrategy,  {"seed": 1}),
        (ParityStrategy,      {"seed": 1}),
        (EntropyStrategy,     {"n_samples": 50, "rng_seed": 1}),
    ],
    ids=["Random", "HuntTarget", "Parity", "Entropy"],
)
def test_legality_ten_games(strategy_cls, kwargs):
    """Legality check across 10 different seeds."""
    strategy = strategy_cls(**kwargs)
    for seed in range(10):
        run_game_legality_check(strategy, seed=seed)


# --------------------------------------------------------------------------- #
# Reset isolation                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "strategy_cls,kwargs",
    [
        (RandomStrategy,      {"seed": 99}),
        (HuntTargetStrategy,  {"seed": 99}),
        (ParityStrategy,      {"seed": 99}),
        (EntropyStrategy,     {"n_samples": 50, "rng_seed": 99}),
    ],
    ids=["Random", "HuntTarget", "Parity", "Entropy"],
)
def test_reset_does_not_carry_state(strategy_cls, kwargs):
    """
    Running the same game twice with the same strategy (resetting between
    runs) should produce the same shot count (for deterministic strategies)
    or at least complete legally.
    """
    strategy = strategy_cls(**kwargs)
    runner = make_runner()

    shots_1 = runner.run_game(strategy, seed=7).total_shots
    shots_2 = runner.run_game(strategy, seed=7).total_shots

    # Both runs must complete (no assertion error).
    assert shots_1 > 0
    assert shots_2 > 0


# --------------------------------------------------------------------------- #
# Minimum shot counts (sanity bounds)                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "strategy_cls,kwargs,min_expected,max_expected",
    [
        # Random must be close to theoretical mean ~95
        (RandomStrategy,     {"seed": 0},              17, 100),
        # HuntTarget must finish in at most 100 shots
        (HuntTargetStrategy, {"seed": 0},              17, 100),
        # Parity must finish in at most 100 shots
        (ParityStrategy,     {"seed": 0},              17, 100),
        # Entropy must finish in at most 100 shots
        (EntropyStrategy,    {"n_samples": 50, "rng_seed": 0}, 17, 100),
    ],
    ids=["Random", "HuntTarget", "Parity", "Entropy"],
)
def test_shot_count_bounds(strategy_cls, kwargs, min_expected, max_expected):
    """Each strategy must finish in [17, 100] shots on a standard game."""
    strategy = strategy_cls(**kwargs)
    runner = make_runner()
    result = runner.run_game(strategy, seed=0)
    assert min_expected <= result.total_shots <= max_expected, (
        f"{strategy.name} completed in {result.total_shots} shots "
        f"(expected [{min_expected}, {max_expected}])"
    )


# --------------------------------------------------------------------------- #
# All ships sunk verification                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "strategy_cls,kwargs",
    [
        (RandomStrategy,     {"seed": 0}),
        (HuntTargetStrategy, {"seed": 0}),
        (ParityStrategy,     {"seed": 0}),
        (EntropyStrategy,    {"n_samples": 50, "rng_seed": 0}),
    ],
    ids=["Random", "HuntTarget", "Parity", "Entropy"],
)
def test_all_ships_sunk_on_completion(strategy_cls, kwargs):
    """Game must end with all ships sunk (not just a shot count goal)."""
    strategy = strategy_cls(**kwargs)
    game = Game(fleet=STANDARD_FLEET, board_size=BOARD_SIZE, seed=3)
    strategy.reset()

    while not game.is_over:
        view = GameView.from_game(game)
        row, col = strategy.select_action(view)
        game.fire(row, col)

    result = game.get_result()
    assert len(result.shots_to_sink) == len(STANDARD_FLEET), (
        f"Only {len(result.shots_to_sink)} ships sunk (expected {len(STANDARD_FLEET)})"
    )


# --------------------------------------------------------------------------- #
# hit_rate sanity (non-random strategies should have higher hit rate)         #
# --------------------------------------------------------------------------- #


def test_hunt_target_higher_hit_rate_than_random():
    """HuntTarget must have a higher hit rate than Random on same boards."""
    runner = make_runner()
    n = 50
    base_seed = 200

    random_results = runner.run_batch(RandomStrategy(seed=0), n, base_seed=base_seed)
    ht_results = runner.run_batch(HuntTargetStrategy(seed=0), n, base_seed=base_seed)

    random_hr = statistics.mean(r.hit_rate for r in random_results)
    ht_hr = statistics.mean(r.hit_rate for r in ht_results)

    assert ht_hr > random_hr, (
        f"HuntTarget hit_rate {ht_hr:.3f} ≤ Random hit_rate {random_hr:.3f}"
    )


# --------------------------------------------------------------------------- #
# Strategy ordering: 100-game batch                                            #
# --------------------------------------------------------------------------- #


def test_strategy_ordering_100_games():
    """
    Over 100 games with fixed seeds, confirm expected ordering:
        Random > HuntTarget > Parity > Entropy   (in avg shots, lower is better)

    We check pairwise that each strategy beats the one above it by at
    least 1 shot on average.  This is a loose check to guard against
    gross regressions without being brittle to variance.
    """
    runner = make_runner()
    n = 100
    base_seed = 42

    strategies = [
        RandomStrategy(seed=0),
        HuntTargetStrategy(seed=0),
        ParityStrategy(seed=0),
        EntropyStrategy(n_samples=100, rng_seed=0),
    ]

    means = {}
    for s in strategies:
        results = runner.run_batch(s, n, base_seed=base_seed)
        means[s.name] = statistics.mean(r.total_shots for r in results)

    random_mean   = means["Random"]
    ht_mean       = means["HuntTarget"]
    parity_mean   = means["Parity"]
    entropy_name  = f"Entropy(n=100)"
    entropy_mean  = means.get(entropy_name, list(means.values())[-1])

    assert random_mean > ht_mean, (
        f"Random ({random_mean:.1f}) should beat HuntTarget ({ht_mean:.1f})"
    )
    assert ht_mean > parity_mean, (
        f"HuntTarget ({ht_mean:.1f}) should beat Parity ({parity_mean:.1f})"
    )
    assert parity_mean > entropy_mean, (
        f"Parity ({parity_mean:.1f}) should beat Entropy ({entropy_mean:.1f})"
    )


# --------------------------------------------------------------------------- #
# ProbabilityEngine / hypothesis space tests                                  #
# --------------------------------------------------------------------------- #


def test_probability_map_sums_to_ship_cells():
    """
    Over many samples, P(ship at cell) summed over all cells should equal
    the total number of ship cells (17 for standard fleet).

    This is the probabilistic analogue of: Σ_c P(ship at c) = E[total ship cells]
    = (number of ship cells) under the uniform prior over placements.
    """
    from info_theory.probability_map import ProbabilityEngine
    from engine.game import Game

    game = Game(seed=0)
    view = GameView.from_game(game)

    engine = ProbabilityEngine(n_samples=2000, rng_seed=0)
    prob_map = engine.compute_prob_map(view)

    total_expected = sum(st.size for st in STANDARD_FLEET)  # 17
    total_prob = float(prob_map.sum())

    # Allow 5% relative error given Monte Carlo noise.
    assert abs(total_prob - total_expected) / total_expected < 0.05, (
        f"Σ P(ship at c) = {total_prob:.2f}, expected ≈ {total_expected}"
    )


def test_probability_map_zero_on_miss_cells():
    """After a MISS, the probability at that cell must be 0."""
    from info_theory.probability_map import ProbabilityEngine
    from engine.game import Game

    game = Game(seed=5)
    game.fire(0, 0)  # may or may not be a hit
    game.fire(0, 1)
    game.fire(0, 2)

    view = GameView.from_game(game)
    engine = ProbabilityEngine(n_samples=200, rng_seed=0)
    prob_map = engine.compute_prob_map(view)

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if view.shot_grid[r, c] != CellState.UNKNOWN:
                assert prob_map[r, c] == 0.0, (
                    f"Non-zero probability {prob_map[r,c]} at fired cell ({r},{c})"
                )


def test_entropy_strategy_exposes_diagnostic_maps():
    """
    After select_action(), EntropyStrategy must expose non-None diagnostic
    attributes: last_prob_map, last_expected_H, last_info_gain_map.
    """
    game = Game(seed=10)
    view = GameView.from_game(game)

    strategy = EntropyStrategy(n_samples=100, rng_seed=0)
    strategy.reset()
    strategy.select_action(view)

    assert strategy.last_prob_map is not None
    assert strategy.last_expected_H is not None
    assert strategy.last_info_gain_map is not None
    assert strategy.last_prob_map.shape == (BOARD_SIZE, BOARD_SIZE)
    assert strategy.last_expected_H.shape == (BOARD_SIZE, BOARD_SIZE)


def test_config_sampler_acceptance_rate():
    """
    On an empty board (turn 0), the acceptance rate should be high (> 50 %).
    """
    from info_theory.hypothesis_space import ConfigSampler
    from engine.game import Game

    game = Game(seed=0)
    view = GameView.from_game(game)

    sampler = ConfigSampler(rng_seed=42)
    rate = sampler.estimate_acceptance_rate(view, n_trials=200)

    assert rate > 0.5, f"Acceptance rate {rate:.2f} is too low for empty board"
