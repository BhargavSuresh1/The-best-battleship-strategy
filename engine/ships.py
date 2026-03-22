"""
engine/ships.py
---------------
Ship definitions: types, orientations, and the Ship dataclass.

Design notes:
- ShipType is an Enum where each member carries (display_name, size), so size
  is always co-located with identity — no external lookup tables needed.
- Orientation is a plain Enum; stored per-ship so placement is self-describing.
- Ship is a dataclass rather than a plain tuple so hit-tracking is encapsulated.
  The board maps each cell → Ship instance, allowing O(1) hit registration.
"""

from __future__ import annotations

import dataclasses
from enum import Enum
from typing import List, Tuple


class Orientation(Enum):
    HORIZONTAL = "H"
    VERTICAL = "V"


class ShipType(Enum):
    """
    Standard Battleship fleet.

    Each member stores (display_name, size) as its value so that size is
    always derivable from the type alone — useful for combinatorial
    enumeration in the information-theoretic module.
    """

    CARRIER = ("Carrier", 5)
    BATTLESHIP = ("Battleship", 4)
    CRUISER = ("Cruiser", 3)
    SUBMARINE = ("Submarine", 3)
    DESTROYER = ("Destroyer", 2)

    def __init__(self, display_name: str, size: int) -> None:
        self.display_name = display_name
        self.size = size

    def __repr__(self) -> str:  # pragma: no cover
        return f"ShipType.{self.name}(size={self.size})"


# Canonical fleet ordered largest → smallest (matches standard rules).
STANDARD_FLEET: List[ShipType] = [
    ShipType.CARRIER,
    ShipType.BATTLESHIP,
    ShipType.CRUISER,
    ShipType.SUBMARINE,
    ShipType.DESTROYER,
]

# Total ship cells in the standard fleet (used for theoretical bounds).
TOTAL_SHIP_CELLS: int = sum(st.size for st in STANDARD_FLEET)  # 17


@dataclasses.dataclass
class Ship:
    """
    A placed ship instance with mutable hit-tracking state.

    Attributes
    ----------
    ship_type   : The kind of ship.
    row, col    : Top-left anchor cell (0-indexed).
    orientation : HORIZONTAL or VERTICAL placement.
    hits        : Number of cells that have been hit (managed by Board).
    """

    ship_type: ShipType
    row: int
    col: int
    orientation: Orientation
    hits: int = dataclasses.field(default=0, init=False, repr=False)

    # ------------------------------------------------------------------ #
    # Derived properties                                                   #
    # ------------------------------------------------------------------ #

    @property
    def size(self) -> int:
        return self.ship_type.size

    @property
    def name(self) -> str:
        return self.ship_type.display_name

    @property
    def is_sunk(self) -> bool:
        return self.hits >= self.size

    # ------------------------------------------------------------------ #
    # Geometry                                                             #
    # ------------------------------------------------------------------ #

    def cells(self) -> List[Tuple[int, int]]:
        """
        Return the list of (row, col) cells occupied by this ship.

        Cells are ordered from anchor outward, which matters for the
        hypothesis-space enumerator in the info-theory module.
        """
        if self.orientation == Orientation.HORIZONTAL:
            return [(self.row, self.col + i) for i in range(self.size)]
        else:
            return [(self.row + i, self.col) for i in range(self.size)]

    # ------------------------------------------------------------------ #
    # State mutation (called only by Board.fire)                          #
    # ------------------------------------------------------------------ #

    def register_hit(self) -> None:
        """Increment hit counter. Board is responsible for calling this."""
        self.hits = min(self.hits + 1, self.size)

    # ------------------------------------------------------------------ #
    # Repr                                                                 #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        status = "SUNK" if self.is_sunk else f"{self.hits}/{self.size} hits"
        return (
            f"Ship({self.name}, ({self.row},{self.col}), "
            f"{self.orientation.value}, {status})"
        )
