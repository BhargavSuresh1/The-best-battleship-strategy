# Battleship Optimal Strategy — System Design Document

## Project Goal

Model Battleship as an **information acquisition problem** and empirically + theoretically determine the most efficient strategy using simulation, information theory, and statistical analysis.

---

## 1. System Architecture

### High-Level Component Map

```
┌─────────────────────────────────────────────────────────────────┐
│                        UI Layer (Phase 6)                        │
│   Streamlit app — board visualization, heatmaps, step-through   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     Analysis Layer (Phase 4)                     │
│   stats.py — CI, hypothesis tests, distribution fitting         │
│   visualization.py — histograms, boxplots, convergence curves   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                  Simulation Framework (Phase 5)                  │
│   runner.py — GameRunner (single + batch + parallel)            │
│   experiment.py — versioned experiment runs, parameter sweeps   │
└──────────┬───────────────────────────────────┬──────────────────┘
           │                                   │
┌──────────▼──────────┐             ┌──────────▼──────────────────┐
│  Strategy Framework │             │  Info-Theory Module (Ph. 3) │
│  (Phase 2)          │             │  hypothesis_space.py        │
│  base.py — ABC      │◄────────────│  probability_map.py         │
│  random.py          │  uses       │  entropy.py                 │
│  hunt_target.py     │  prob maps  │                             │
│  parity.py          │             └─────────────────────────────┘
│  entropy.py         │
└──────────┬──────────┘
           │ select_action(GameView)
┌──────────▼──────────────────────────────────────────────────────┐
│                       Core Engine (Phase 1)  ✓                  │
│                                                                  │
│   ships.py   — ShipType, Orientation, Ship                      │
│   board.py   — Board (ship_grid + shot_grid), CellState         │
│   game.py    — Game, ShotRecord, GameResult                     │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
GameRunner.run_game(strategy, seed)
    │
    ├─► Game(fleet, board_size, seed)
    │       └─► Board.place_fleet_randomly(fleet, rng)
    │
    └─► loop until game.is_over:
            │
            ├─► GameView.from_game(game)      # immutable snapshot
            │       └─► shot_grid.copy()       # defensive copy
            │
            ├─► strategy.select_action(view)  # (row, col)
            │
            └─► game.fire(row, col)
                    └─► board.fire(row, col)
                            ├─► CellState result
                            └─► ship.register_hit() if hit
                                    └─► mark all cells SUNK if ship.is_sunk

    └─► game.get_result() → GameResult (frozen)
```

---

## 2. File / Folder Structure

```
battleship_project/
│
├── engine/                      # Phase 1 — Core simulation engine
│   ├── __init__.py
│   ├── ships.py                 # ShipType enum, Orientation, Ship dataclass
│   ├── board.py                 # Board, CellState, placement + shot mechanics
│   └── game.py                  # Game, ShotRecord, GameResult
│
├── strategies/                  # Phase 2 — Strategy implementations
│   ├── __init__.py
│   ├── base.py                  # Strategy ABC + GameView snapshot
│   ├── random_strategy.py       # Uniform random baseline
│   ├── hunt_target.py           # Hunt-target heuristic
│   ├── parity.py                # Checkerboard parity optimization
│   └── entropy.py               # Information-theoretic (entropy minimization)
│
├── info_theory/                 # Phase 3 — Information-theoretic module
│   ├── __init__.py
│   ├── hypothesis_space.py      # Enumerate valid ship placements
│   ├── probability_map.py       # Bayesian posterior probability grid
│   └── entropy.py               # H computation, expected info gain
│
├── simulation/                  # Phase 5 — Experiment runner
│   ├── __init__.py
│   ├── runner.py                # GameRunner (single + batch)
│   ├── experiment.py            # Versioned experiments, parameter sweeps
│   └── parallel.py              # multiprocessing.Pool wrapper
│
├── analysis/                    # Phase 4 — Statistical analysis
│   ├── __init__.py
│   ├── stats.py                 # Confidence intervals, hypothesis tests
│   └── visualization.py        # Matplotlib / seaborn plots
│
├── ui/                          # Phase 6 — Streamlit interface
│   └── app.py
│
├── tests/
│   ├── __init__.py
│   ├── test_ships.py
│   ├── test_board.py
│   └── test_game.py
│
├── design.md                    # This file
├── requirements.txt
└── README.md
```

---

## 3. Core Engine Design (Phase 1)

### 3.1 Board Representation

**Decision:** Two `numpy` int8 arrays in one `Board` object.

| Array | Contents | Visibility |
|---|---|---|
| `ship_grid` | True ship positions (ship_id per cell, 0 = empty) | Engine-internal only |
| `shot_grid` | Observable state (CellState int per cell) | Exposed to strategies via GameView |

**Why two arrays in one object?**
Keeping both in one object avoids an object-graph where Board and HiddenBoard cross-reference each other. The information boundary is enforced by *access control* (GameView only copies `shot_grid`), not by object separation.

**Why int8?**
- Minimal memory footprint — critical for storing thousands of board states in the info-theory hypothesis space
- Direct numpy indexing with no boxing overhead
- Values fit in int8: 5 ships (IDs 1–5), CellState values 0–3

### 3.2 CellState Encoding

```
UNKNOWN = 0   — not yet fired at
MISS    = 1   — fired, no ship
HIT     = 2   — fired, ship present, ship not yet sunk
SUNK    = 3   — fired, ship present, ship now fully sunk
```

**Critical design: SUNK marks ALL ship cells.** When a ship is sunk, every cell it occupies is set to `SUNK` — even cells that were previously marked `HIT`. This is intentional:

- Strategies can read the exact contour of sunk ships from `shot_grid` alone
- The info-theory module can immediately remove all placements involving those cells
- The information boundary between HIT and SUNK is precise, not ambiguous

### 3.3 Ship Placement

**Decision:** Rejection sampling.

```python
while not placed:
    row, col = rng.randint(0, size-1), rng.randint(0, size-1)
    orientation = rng.choice([H, V])
    ship = Ship(ship_type, row, col, orientation)
    placed = board.place_ship(ship)
```

**Why rejection sampling over constraint-satisfaction or precomputed tables?**
- Acceptance rate for the standard fleet on 10×10 is very high (~95%+ per ship)
- Implementation is trivial and correct by construction
- Seeded RNG makes it fully reproducible
- No precomputed table to maintain if fleet or board size changes

**Fleet placement order:** Largest ship first (`CARRIER → BATTLESHIP → ...`). This marginally improves acceptance rates since larger ships constrain placement more — placing them first avoids failed attempts late in the sequence.

### 3.4 Cell-to-Ship Mapping

```python
self._cell_to_ship: Dict[Tuple[int, int], Ship] = {}
```

O(1) lookup from `(row, col) → Ship` instance. This is populated during `place_ship()` and used in `fire()` to call `ship.register_hit()` directly without scanning the ship list. For 100k+ simulations, this matters.

### 3.5 Ship Hit Tracking

`Ship.hits` is a mutable counter on the dataclass. `register_hit()` increments it; `is_sunk` checks `hits >= size`. The Board owns when `register_hit()` is called — Ship does not know about cells or boards.

`hits` is capped at `size` as a defensive guard against double-hit bugs in strategies being developed.

---

## 4. Strategy Framework Design

### 4.1 The GameView Contract

`GameView` is the **only** interface between the strategy and the game. It is a frozen dataclass constructed by copying `board.shot_grid` before every `select_action()` call.

```python
@dataclasses.dataclass(frozen=True)
class GameView:
    shot_grid: np.ndarray          # copy — mutations do not affect board
    board_size: int
    turn: int
    afloat_ships: Tuple[ShipType, ...]   # lengths known, positions unknown
    sunk_ships: Tuple[ShipType, ...]
```

**Why frozen + copy?**
- Prevents strategies from accidentally mutating game state
- Enables replay: a sequence of GameViews is a complete game record
- Enables future serialization for ML training data
- `afloat_ships` exposes ship lengths (public information: you know which ships haven't sunk yet) without revealing positions

### 4.2 Strategy ABC

```python
class Strategy(ABC):
    @abstractmethod
    def select_action(self, view: GameView) -> Tuple[int, int]: ...

    def reset(self) -> None: ...      # called at start of every game

    @property
    def name(self) -> str: ...        # used in experiment labels / plots
```

**Why `reset()` instead of constructing a new Strategy per game?**
Constructing strategies can be expensive (e.g. precomputing probability maps). `reset()` lets the strategy reinitialize only what changes per game, while keeping pre-computed static data (e.g. all valid placements for an empty board) in memory.

**Why `name` as a property?**
Experiment logs and plot labels need human-readable strategy names. Using a property rather than a class attribute allows subclasses to include runtime parameters in the name (e.g. `"Entropy(n_samples=500)"`).

### 4.3 Strategy Hierarchy (Phase 2 Plan)

```
Strategy (ABC)
├── RandomStrategy          — uniform random over unfired cells
├── HuntTargetStrategy      — random until hit, then target neighbors
│   └── uses: hit_cells from GameView
├── ParityStrategy          — parity-filtered random + hunt-target
│   └── uses: afloat_ships (minimum ship size drives parity stride)
└── EntropyStrategy         — maximize expected information gain per shot
    └── uses: ProbabilityMap from info_theory module
```

The first three strategies need only `GameView`. `EntropyStrategy` additionally depends on the `info_theory` module but is still injected into `GameRunner` identically — the coupling is internal to the strategy class.

---

## 5. Information-Theoretic Module Design (Phase 3)

### 5.1 Mathematical Formulation

Battleship is a **hypothesis elimination game**. At any point:

- Let Ω be the set of all valid ship configurations consistent with the current `shot_grid`
- A configuration ω ∈ Ω assigns a placement to each afloat ship
- Prior: uniform distribution over Ω (standard assumption)
- Posterior after shot at cell c with result r: P(ω | r) ∝ P(r | ω) · P(ω)

**Probability map:**

For each unfired cell c, the marginal probability that c contains a ship:

```
P(ship at c) = |{ω ∈ Ω : c is occupied in ω}| / |Ω|
```

**Entropy of current state:**

```
H(Ω) = -Σ_{ω ∈ Ω} p(ω) log p(ω)
```

With uniform prior: `H(Ω) = log |Ω|` (bits if log₂, nats if ln)

**Expected information gain for firing at cell c:**

```
E[IG(c)] = H(Ω) - E[H(Ω | result at c)]
         = H(Ω) - [P(hit|c) · H(Ω_hit) + P(miss|c) · H(Ω_miss)]
```

where Ω_hit and Ω_miss are the subsets of Ω consistent with each outcome.

**Optimal greedy shot:** `c* = argmax_c E[IG(c)]`

Note: this is greedy (one-step lookahead), not globally optimal. Global optimization is intractable (exponential state space). The greedy information-theoretic strategy is near-optimal in practice.

### 5.2 The Tractability Problem

For a standard 10×10 board with 5 ships, |Ω| can be on the order of 10⁸–10¹⁰ at the start of the game. Exact computation is infeasible.

### 5.3 Approximation Plan

**Approach: Monte Carlo sampling over hypothesis space**

```python
class ProbabilityMap:
    def __init__(self, n_samples: int = 500):
        self.n_samples = n_samples

    def compute(self, view: GameView) -> np.ndarray:
        """
        Sample n_samples valid configurations from Ω,
        return probability map as (board_size, board_size) float array.
        """
        prob = np.zeros((view.board_size, view.board_size))
        valid = 0
        for _ in range(self.n_samples):
            config = self._sample_config(view)
            if config is not None:
                prob += config
                valid += 1
        return prob / valid if valid > 0 else prob
```

Each sample:
1. Randomly place each afloat ship on cells consistent with `shot_grid` (not on MISS cells, must cover all HIT cells)
2. Accept if valid (no overlaps, all constraints satisfied)
3. Increment probability map

**Tradeoffs:**

| n_samples | Accuracy | Time per shot | Use case |
|---|---|---|---|
| 100 | Low | ~1ms | Fast interactive UI |
| 500 | Medium | ~5ms | Default strategy |
| 5000 | High | ~50ms | Research analysis |
| Exact | Perfect | Intractable | Theoretical baseline only |

**Importance sampling (future):** Weight samples by how well they explain observed hits, rather than pure rejection sampling. Improves accuracy at no extra sample cost.

---

## 6. Simulation Framework Design

### 6.1 GameRunner

`GameRunner` is the coupling point between `Strategy` and `Game`. It owns no state beyond fleet and board configuration.

```
GameRunner.run_batch(strategy, n_games, base_seed)
    │
    ├─► for i in range(n_games):
    │       seed = base_seed + i         # independent, reproducible
    │       game = Game(seed=seed)
    │       strategy.reset()
    │       loop: view → strategy → fire
    │       results.append(game.get_result())
    │
    └─► return List[GameResult]
```

**Seed scheme:** `seed_i = base_seed + i` ensures:
- Every run with the same `base_seed` and `n_games` is identical (reproducibility)
- Each game has an independent board (no correlation between games)
- Experiments can be extended: running `base_seed=0, n=1000` then `base_seed=1000, n=1000` gives 2000 independent games

### 6.2 Parallelization (Phase 5)

The `run_batch` interface is designed for zero-friction parallelization:

```python
# Sequential (Phase 1):
results = [run_game(strategy, seed) for seed in seeds]

# Parallel (Phase 5) — same external API:
with Pool(cpu_count()) as pool:
    results = pool.starmap(run_game_worker, [(fleet, board_size, strategy_cls, seed) for seed in seeds])
```

Strategies must be picklable for multiprocessing. The ABC design (no file handles, no threads) makes this straightforward.

### 6.3 Metrics

| Metric | Why it matters |
|---|---|
| `total_shots` | Primary performance measure |
| `shots_to_sink[i]` | Marginal cost to eliminate each ship |
| `hit_rate` | Efficiency of the strategy (hits / total shots) |
| `variance of total_shots` | Strategy stability, not just average performance |
| `distribution shape` | Reveals whether strategy occasionally catastrophically fails |

---

## 7. Key Design Decisions and Tradeoffs

### 7.1 Engine knows nothing about strategies

`Game` and `Board` have zero imports from `strategies/`. The engine is a pure state machine. This means:

- Engine tests need no strategy mocks
- Strategies are hot-swappable
- Engine can be used for non-strategy purposes (e.g. generating training data, solving puzzles)

### 7.2 GameResult is a frozen dataclass, not a class with methods

`GameResult` is deliberately minimal: fields only, plus a few computed properties. Heavy analysis (confidence intervals, distribution fitting) lives in `analysis/stats.py`. This keeps results serializable and avoids pulling analysis dependencies into the engine.

### 7.3 No pandas in engine or strategies

The engine returns `List[GameResult]`. Callers convert to DataFrames. This means the engine works in environments without pandas and the conversion layer can be swapped (e.g. to Polars, Arrow, etc.).

### 7.4 `random.Random` not `numpy.random`

Per-game `random.Random(seed)` instances are used for board placement. `numpy.random` has a global (or module-level) state that is harder to isolate per game. `random.Random` instances are independent, lightweight, and picklable — essential for parallel simulation.

### 7.5 Extensibility for ML strategies

The `Strategy` interface is compatible with ML-based strategies:

```python
class NeuralStrategy(Strategy):
    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    def select_action(self, view: GameView) -> Tuple[int, int]:
        features = encode_board(view.shot_grid)   # → feature vector
        logits = self.model(features)             # → (100,) scores
        # mask fired cells, argmax
        return decode_action(logits, view.shot_grid)

    def reset(self) -> None:
        pass  # model weights unchanged; no per-game state
```

The engine and runner require zero changes.

---

## 8. Mathematical Depth Notes

### 8.1 Lower bound on expected shots

The theoretical minimum expected shots is bounded below by the number of ship cells (17) — you must hit every cell to win. The random baseline averages ~96 shots on a 10×10 board (with 17 ship cells). A perfect information-theoretic strategy approaches ~40–45 shots in practice. The gap is the "inefficiency budget" all strategies compete to close.

### 8.2 Information content of a shot

A shot at cell c provides:
- `P(hit) = P(ship at c)` — from the probability map
- `P(miss) = 1 - P(ship at c)`

Shannon information content: `I(c) = -log P(outcome)` bits per shot. A shot at a 50/50 cell provides maximum entropy (1 bit). A shot at a cell with P(hit)=0.99 provides very little information on hit (0.014 bits) but a lot on miss (6.6 bits, very surprising).

The entropy strategy does not maximize information content of the shot — it maximizes expected reduction in hypothesis space size, which is subtly different.

### 8.3 Bayesian update structure

When a ship sinks, the posterior update is particularly powerful: the exact cells of one ship are now known, which eliminates all hypotheses where any other ship overlaps those cells. This is why marking all cells SUNK (not just the killing shot) is a design requirement, not just a display choice.

---

## 9. Implementation Roadmap

| Phase | Content | Status |
|---|---|---|
| 1 | Core engine (ships, board, game, strategy ABC, basic runner) | **Complete** |
| 2 | Strategy implementations: Random, HuntTarget, Parity, Entropy | Planned |
| 3 | Info-theory module: hypothesis space, probability maps, entropy | Planned |
| 4 | Analysis layer: stats, CI, visualization | Planned |
| 5 | Parallelization, experiment versioning, parameter sweeps | Planned |
| 6 | Streamlit UI: board vis, heatmaps, step-through debugger | Planned |
