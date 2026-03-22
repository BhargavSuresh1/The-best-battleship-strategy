"""
tests/test_board.py
-------------------
Unit tests for Board placement, shot mechanics, and observable state.
"""

import random

import numpy as np
import pytest

from engine.board import Board, CellState, BOARD_SIZE
from engine.ships import Ship, ShipType, Orientation, STANDARD_FLEET


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def make_board_with_ship(ship: Ship, size: int = 10) -> Board:
    board = Board(size)
    success = board.place_ship(ship)
    assert success, f"Expected valid placement for {ship}"
    return board


# --------------------------------------------------------------------------- #
# Placement                                                                    #
# --------------------------------------------------------------------------- #


class TestPlacement:
    def test_valid_horizontal(self):
        ship = Ship(ShipType.DESTROYER, row=0, col=0, orientation=Orientation.HORIZONTAL)
        board = Board()
        assert board.place_ship(ship)
        assert board.ship_grid[0, 0] == 1
        assert board.ship_grid[0, 1] == 1

    def test_valid_vertical(self):
        ship = Ship(ShipType.CRUISER, row=2, col=5, orientation=Orientation.VERTICAL)
        board = Board()
        assert board.place_ship(ship)
        for i in range(3):
            assert board.ship_grid[2 + i, 5] == 1

    def test_out_of_bounds_horizontal(self):
        ship = Ship(ShipType.CARRIER, row=0, col=8, orientation=Orientation.HORIZONTAL)
        board = Board()
        assert not board.place_ship(ship)

    def test_out_of_bounds_vertical(self):
        ship = Ship(ShipType.BATTLESHIP, row=8, col=0, orientation=Orientation.VERTICAL)
        board = Board()
        assert not board.place_ship(ship)

    def test_overlap_rejected(self):
        ship1 = Ship(ShipType.CARRIER, row=0, col=0, orientation=Orientation.HORIZONTAL)
        ship2 = Ship(ShipType.BATTLESHIP, row=0, col=2, orientation=Orientation.VERTICAL)
        board = Board()
        assert board.place_ship(ship1)
        assert not board.place_ship(ship2)  # overlaps carrier at (0,2)

    def test_adjacent_ships_allowed(self):
        """Ships touching (but not overlapping) is legal in standard rules."""
        ship1 = Ship(ShipType.DESTROYER, row=0, col=0, orientation=Orientation.HORIZONTAL)
        ship2 = Ship(ShipType.DESTROYER, row=1, col=0, orientation=Orientation.HORIZONTAL)
        board = Board()
        assert board.place_ship(ship1)
        assert board.place_ship(ship2)

    def test_place_fleet_randomly_fills_all_ships(self):
        board = Board()
        board.place_fleet_randomly(STANDARD_FLEET, rng=random.Random(42))
        assert len(board.ships) == len(STANDARD_FLEET)

    def test_place_fleet_randomly_no_overlap(self):
        """Every occupied cell should map to exactly one ship."""
        board = Board()
        board.place_fleet_randomly(STANDARD_FLEET, rng=random.Random(99))
        occupied = np.sum(board.ship_grid != 0)
        assert occupied == sum(st.size for st in STANDARD_FLEET)  # 17

    def test_place_fleet_reproducible(self):
        b1, b2 = Board(), Board()
        b1.place_fleet_randomly(STANDARD_FLEET, rng=random.Random(7))
        b2.place_fleet_randomly(STANDARD_FLEET, rng=random.Random(7))
        np.testing.assert_array_equal(b1.ship_grid, b2.ship_grid)

    def test_place_fleet_reset_clears_previous(self):
        board = Board()
        board.place_fleet_randomly(STANDARD_FLEET, rng=random.Random(1))
        first_grid = board.ship_grid.copy()
        board.place_fleet_randomly(STANDARD_FLEET, rng=random.Random(2))
        # Different seed → different (or at least valid) layout
        # Just check that ships list was repopulated correctly
        assert len(board.ships) == len(STANDARD_FLEET)


# --------------------------------------------------------------------------- #
# Shot mechanics                                                               #
# --------------------------------------------------------------------------- #


class TestShotMechanics:
    def test_miss_returns_miss(self):
        # Place ship at (0,0)-(0,1); fire at (9,9)
        ship = Ship(ShipType.DESTROYER, 0, 0, Orientation.HORIZONTAL)
        board = make_board_with_ship(ship)
        result = board.fire(9, 9)
        assert result == CellState.MISS
        assert board.shot_grid[9, 9] == CellState.MISS

    def test_hit_returns_hit(self):
        ship = Ship(ShipType.CRUISER, 3, 3, Orientation.HORIZONTAL)
        board = make_board_with_ship(ship)
        result = board.fire(3, 3)
        assert result == CellState.HIT
        assert board.shot_grid[3, 3] == CellState.HIT

    def test_sinking_returns_sunk(self):
        ship = Ship(ShipType.DESTROYER, 5, 5, Orientation.HORIZONTAL)
        board = make_board_with_ship(ship)
        board.fire(5, 5)
        result = board.fire(5, 6)
        assert result == CellState.SUNK

    def test_sunk_marks_all_cells(self):
        """When a ship sinks, all its cells become SUNK (including prior HIT cells)."""
        ship = Ship(ShipType.CRUISER, 0, 0, Orientation.HORIZONTAL)
        board = make_board_with_ship(ship)
        board.fire(0, 0)  # HIT
        board.fire(0, 1)  # HIT
        result = board.fire(0, 2)  # SUNK
        assert result == CellState.SUNK
        assert board.shot_grid[0, 0] == CellState.SUNK
        assert board.shot_grid[0, 1] == CellState.SUNK
        assert board.shot_grid[0, 2] == CellState.SUNK

    def test_fire_out_of_bounds_raises(self):
        board = Board()
        board.place_fleet_randomly(rng=random.Random(1))
        with pytest.raises(ValueError, match="out of bounds"):
            board.fire(-1, 0)

    def test_fire_already_fired_raises(self):
        ship = Ship(ShipType.DESTROYER, 2, 2, Orientation.HORIZONTAL)
        board = make_board_with_ship(ship)
        board.fire(9, 9)
        with pytest.raises(ValueError):
            board.fire(9, 9)

    def test_game_not_over_until_all_sunk(self):
        ship1 = Ship(ShipType.DESTROYER, 0, 0, Orientation.HORIZONTAL)
        ship2 = Ship(ShipType.DESTROYER, 5, 5, Orientation.HORIZONTAL)
        board = Board()
        board.place_ship(ship1)
        board.place_ship(ship2)
        board.fire(0, 0)
        board.fire(0, 1)  # ship1 sunk
        assert not board.is_game_over
        board.fire(5, 5)
        board.fire(5, 6)  # ship2 sunk
        assert board.is_game_over

    def test_ships_remaining_decrements(self):
        ship1 = Ship(ShipType.DESTROYER, 0, 0, Orientation.HORIZONTAL)
        ship2 = Ship(ShipType.DESTROYER, 9, 0, Orientation.HORIZONTAL)
        board = Board()
        board.place_ship(ship1)
        board.place_ship(ship2)
        assert board.ships_remaining == 2
        board.fire(0, 0)
        board.fire(0, 1)
        assert board.ships_remaining == 1


# --------------------------------------------------------------------------- #
# Observable properties                                                        #
# --------------------------------------------------------------------------- #


class TestObservableProperties:
    def test_unfired_cells_initial(self):
        board = Board(size=4)
        assert len(board.unfired_cells) == 16  # 4×4

    def test_unfired_cells_decrements(self):
        board = Board()
        board.place_fleet_randomly(rng=random.Random(3))
        board.fire(0, 0)
        board.fire(0, 1)
        assert len(board.unfired_cells) == BOARD_SIZE * BOARD_SIZE - 2

    def test_total_shots_tracks_fires(self):
        board = Board()
        board.place_fleet_randomly(rng=random.Random(5))
        for i in range(5):
            board.fire(0, i)
        assert board.total_shots == 5

    def test_hit_cells_only_includes_non_sunk_hits(self):
        ship = Ship(ShipType.CRUISER, 0, 0, Orientation.HORIZONTAL)
        board = make_board_with_ship(ship)
        board.fire(0, 0)  # HIT
        assert (0, 0) in board.hit_cells
        board.fire(0, 1)  # HIT
        board.fire(0, 2)  # SUNK — all cells become SUNK
        assert board.hit_cells == []
        assert len(board.sunk_cells) == 3

    def test_miss_cells(self):
        ship = Ship(ShipType.DESTROYER, 0, 0, Orientation.HORIZONTAL)
        board = make_board_with_ship(ship)
        board.fire(9, 9)
        board.fire(8, 8)
        assert len(board.miss_cells) == 2

    def test_afloat_ship_types(self):
        board = Board()
        board.place_fleet_randomly(STANDARD_FLEET, rng=random.Random(42))
        assert len(board.afloat_ship_types) == 5
        assert len(board.sunk_ship_types) == 0
