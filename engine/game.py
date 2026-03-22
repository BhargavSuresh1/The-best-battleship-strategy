"""
engine/game.py
--------------
Game orchestration layer: turn execution, record keeping, and result packaging.

Design notes:
------------
Game separates *what happened* (Board) from *why* (Strategy) and *when*
(turn counter here).  This makes the engine reusable for:

  - Automated strategy benchmarking (GameRunner injects a Strategy)
  - Interactive play (UI calls game.fire() directly)
  - Replays (feed shot_records back in sequence)

GameResult is a value object (frozen dataclass) suitable for immediate
serialization to DataFrame rows in the simulation pipeline.

ShotRecord captures full provenance per shot, enabling per-turn analysis
(e.g. information gain curves, conditional hit-rate by turn number).

The Game class deliberately does NOT own a Strategy — injection happens at
the GameRunner level.  This keeps Game testable in isolation and avoids
circular imports with the strategies package.
"""

from __future__ import annotations

import dataclasses
import random
from typing import List, Optional

from .board import Board, CellState, BOARD_SIZE
from .ships import ShipType, STANDARD_FLEET


# --------------------------------------------------------------------------- #
# Value objects                                                                #
# --------------------------------------------------------------------------- #


@dataclasses.dataclass(frozen=True)
class ShotRecord:
    """Immutable record of a single shot."""

    turn: int           # 1-indexed turn number
    row: int
    col: int
    result: CellState   # MISS / HIT / SUNK

    def is_hit(self) -> bool:
        return self.result in (CellState.HIT, CellState.SUNK)

    def __repr__(self) -> str:
        return f"ShotRecord(t={self.turn}, ({self.row},{self.col}), {self.result.name})"


@dataclasses.dataclass(frozen=True)
class GameResult:
    """
    Complete summary of a finished game.

    Designed to map cleanly onto a single DataFrame row:

        total_shots         — primary performance metric
        shots_to_sink       — list[int], one entry per ship sunk, in order
        shot_records        — full shot history (for deep analysis)
        seed                — reproducibility handle
        fleet               — which ships were on the board

    Computed properties are not stored to keep serialization clean.
    """

    total_shots: int
    shots_to_sink: tuple[int, ...]   # shots at time each ship was sunk
    shot_records: tuple[ShotRecord, ...]
    seed: Optional[int]
    fleet: tuple[ShipType, ...]

    # ------------------------------------------------------------------ #
    # Derived statistics                                                   #
    # ------------------------------------------------------------------ #

    @property
    def hits(self) -> int:
        return sum(1 for s in self.shot_records if s.is_hit())

    @property
    def misses(self) -> int:
        return sum(1 for s in self.shot_records if not s.is_hit())

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total_shots if self.total_shots > 0 else 0.0

    @property
    def ships_count(self) -> int:
        return len(self.fleet)

    def shots_for_ship(self, ship_index: int) -> int:
        """
        Shots taken to sink the *ship_index*-th ship (0-indexed in sinking
        order).  Returns the marginal shots, not the cumulative total.
        """
        cumulative = self.shots_to_sink[ship_index]
        prev = self.shots_to_sink[ship_index - 1] if ship_index > 0 else 0
        return cumulative - prev

    def to_dict(self) -> dict:
        """Flat dict for DataFrame row construction."""
        d = {
            "total_shots": self.total_shots,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hit_rate, 4),
            "seed": self.seed,
        }
        for i, t in enumerate(self.shots_to_sink):
            d[f"sink_{i+1}_at_shot"] = t
        return d


# --------------------------------------------------------------------------- #
# Game                                                                         #
# --------------------------------------------------------------------------- #


class Game:
    """
    Orchestrates a single game of Battleship.

    Responsibilities:
      - Own the Board and its RNG
      - Expose fire() for shot execution
      - Track turn number and per-ship sinking milestones
      - Produce a GameResult when complete

    Non-responsibilities (deliberately excluded):
      - Deciding *where* to fire — delegated to Strategy via GameRunner
      - Rendering / display — delegated to UI or Board.render_*
    """

    def __init__(
        self,
        fleet: List[ShipType] = STANDARD_FLEET,
        board_size: int = BOARD_SIZE,
        seed: Optional[int] = None,
    ) -> None:
        self.fleet: List[ShipType] = list(fleet)
        self.board_size: int = board_size
        self.seed: Optional[int] = seed

        self._rng = random.Random(seed)
        self.board: Board = Board(board_size)
        self.board.place_fleet_randomly(self.fleet, self._rng)

        self._turn: int = 0
        self._shot_records: List[ShotRecord] = []
        self._shots_to_sink: List[int] = []
        self._prev_ships_remaining: int = len(self.fleet)

    # ------------------------------------------------------------------ #
    # State queries                                                        #
    # ------------------------------------------------------------------ #

    @property
    def is_over(self) -> bool:
        return self.board.is_game_over

    @property
    def turn(self) -> int:
        """Current turn number (0 before any shot, increments after each fire)."""
        return self._turn

    @property
    def shot_records(self) -> List[ShotRecord]:
        return list(self._shot_records)  # defensive copy

    # ------------------------------------------------------------------ #
    # Core mechanic                                                        #
    # ------------------------------------------------------------------ #

    def fire(self, row: int, col: int) -> CellState:
        """
        Execute a shot at (row, col).

        Increments turn counter, appends a ShotRecord, and records sinking
        milestones.  Returns the CellState result.

        Raises ValueError (from Board.fire) if coordinates are invalid or
        the cell has already been fired at.
        """
        if self.is_over:
            raise RuntimeError("Cannot fire: game is already over.")

        result = self.board.fire(row, col)
        self._turn += 1
        self._shot_records.append(ShotRecord(self._turn, row, col, result))

        # Record cumulative shot count at the moment each ship sinks.
        current_remaining = self.board.ships_remaining
        if current_remaining < self._prev_ships_remaining:
            self._shots_to_sink.append(self._turn)
            self._prev_ships_remaining = current_remaining

        return result

    # ------------------------------------------------------------------ #
    # Result packaging                                                     #
    # ------------------------------------------------------------------ #

    def get_result(self) -> GameResult:
        """
        Package the completed game into an immutable GameResult.

        Raises RuntimeError if the game is not yet over.
        """
        if not self.is_over:
            raise RuntimeError(
                f"Game is not over yet (turn {self._turn}, "
                f"{self.board.ships_remaining} ships remaining)."
            )
        return GameResult(
            total_shots=self._turn,
            shots_to_sink=tuple(self._shots_to_sink),
            shot_records=tuple(self._shot_records),
            seed=self.seed,
            fleet=tuple(self.fleet),
        )

    # ------------------------------------------------------------------ #
    # Reset                                                                #
    # ------------------------------------------------------------------ #

    def reset(self, new_seed: Optional[int] = None) -> None:
        """
        Reset the game for a fresh run on the same configuration.

        Useful for interactive exploration; GameRunner creates new Game
        instances instead of calling reset() for cleaner state isolation.
        """
        if new_seed is not None:
            self.seed = new_seed
        self._rng = random.Random(self.seed)
        self.board = Board(self.board_size)
        self.board.place_fleet_randomly(self.fleet, self._rng)
        self._turn = 0
        self._shot_records.clear()
        self._shots_to_sink.clear()
        self._prev_ships_remaining = len(self.fleet)

    def __repr__(self) -> str:  # pragma: no cover
        status = "OVER" if self.is_over else f"turn {self._turn}"
        return (
            f"Game({status}, {self.board.ships_remaining}/{len(self.fleet)} ships, "
            f"seed={self.seed})"
        )
