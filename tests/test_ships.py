"""
tests/test_ships.py
-------------------
Unit tests for ShipType, Orientation, and Ship.
"""

import pytest
from engine.ships import (
    Ship,
    ShipType,
    Orientation,
    STANDARD_FLEET,
    TOTAL_SHIP_CELLS,
)


class TestShipType:
    def test_sizes(self):
        assert ShipType.CARRIER.size == 5
        assert ShipType.BATTLESHIP.size == 4
        assert ShipType.CRUISER.size == 3
        assert ShipType.SUBMARINE.size == 3
        assert ShipType.DESTROYER.size == 2

    def test_display_names(self):
        assert ShipType.CARRIER.display_name == "Carrier"
        assert ShipType.DESTROYER.display_name == "Destroyer"

    def test_standard_fleet_length(self):
        assert len(STANDARD_FLEET) == 5

    def test_total_ship_cells(self):
        assert TOTAL_SHIP_CELLS == 17  # 5+4+3+3+2


class TestShipCells:
    def test_horizontal_cells(self):
        ship = Ship(ShipType.DESTROYER, row=3, col=2, orientation=Orientation.HORIZONTAL)
        assert ship.cells() == [(3, 2), (3, 3)]

    def test_vertical_cells(self):
        ship = Ship(ShipType.CRUISER, row=1, col=4, orientation=Orientation.VERTICAL)
        assert ship.cells() == [(1, 4), (2, 4), (3, 4)]

    def test_carrier_horizontal_cells(self):
        ship = Ship(ShipType.CARRIER, row=0, col=0, orientation=Orientation.HORIZONTAL)
        assert ship.cells() == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]

    def test_cells_length_matches_size(self):
        for ship_type in ShipType:
            ship = Ship(ship_type, 0, 0, Orientation.HORIZONTAL)
            assert len(ship.cells()) == ship_type.size


class TestShipHitTracking:
    def test_initial_state(self):
        ship = Ship(ShipType.DESTROYER, 0, 0, Orientation.HORIZONTAL)
        assert ship.hits == 0
        assert not ship.is_sunk

    def test_single_hit(self):
        ship = Ship(ShipType.DESTROYER, 0, 0, Orientation.HORIZONTAL)
        ship.register_hit()
        assert ship.hits == 1
        assert not ship.is_sunk

    def test_sinks_on_full_hits(self):
        ship = Ship(ShipType.DESTROYER, 0, 0, Orientation.HORIZONTAL)
        ship.register_hit()
        ship.register_hit()
        assert ship.is_sunk

    def test_carrier_requires_five_hits(self):
        ship = Ship(ShipType.CARRIER, 0, 0, Orientation.HORIZONTAL)
        for _ in range(4):
            ship.register_hit()
        assert not ship.is_sunk
        ship.register_hit()
        assert ship.is_sunk

    def test_hits_capped_at_size(self):
        """register_hit beyond size should not exceed size (defensive cap)."""
        ship = Ship(ShipType.DESTROYER, 0, 0, Orientation.HORIZONTAL)
        for _ in range(5):  # more than size=2
            ship.register_hit()
        assert ship.hits == 2
