# Optimal Strategies for Battleship
### A Probabilistic and Information-Theoretic Analysis

A research-grade Python simulation framework that models the Battleship board
game as a sequential Bayesian search problem. Four AI strategies of increasing
theoretical sophistication are implemented, evaluated through large-scale Monte
Carlo experiments, and analysed in a full 20-page research paper.

---

## Key Results

Experiments over up to **2,000 games** on the standard 10x10 board
(fleet: Carrier 5, Battleship 4, Cruiser 3, Submarine 3, Destroyer 2):

| Strategy | Mean shots | Std | vs Random | Time / game |
|---|---|---|---|---|
| Random | 95.3 | 4.9 | -- | ~2 ms |
| Hunt / Target | 60.2 | 14.7 | -36.9% | ~2 ms |
| Parity | 51.6 | 8.7 | -45.8% | ~2 ms |
| Entropy (K=100) | 51.8 | 9.1 | -45.6% | ~1,570 ms |

**Central finding:** the full information-theoretic entropy strategy costs
x860 more compute per game but yields no measurable improvement over the
simpler parity strategy. High-probability cells are simultaneously
high-information-gain cells, making the cheaper probability criterion a
near-optimal proxy for entropy minimisation.

---

## Research Paper

**"Optimal Strategies for Battleship: A Probabilistic and
Information-Theoretic Analysis"**

| File | Description |
|---|---|
| `battleship_paper.tex` | LaTeX source |
| `battleship_paper.pdf` | Compiled PDF, 20 pages |

---

### Paper outline

1. Mathematical framework -- Bayesian posterior over hypothesis space Omega
2. Monte Carlo approximation -- adaptive acceptance-rejection sampling
3. Mean-field entropy approximation -- sum of binary entropies (proved upper bound)
4. Algorithm pseudocode for all four strategies
5. Experimental results with full statistics tables
6. Discussion -- near-equivalence of probability and entropy criteria
7. Complexity analysis -- O(K * M^2) per turn, |Omega| ~ 10^8 to 10^10
8. Appendices -- coupon-collector derivation, mean-field bound proof


## Project Structure

```
battleship_project/
|
+-- engine/                   # Core game mechanics
|   +-- ships.py              # ShipType enum (name + size), Orientation, Ship dataclass
|   +-- board.py              # Board: ship_grid (truth) + shot_grid (observable)
|   +-- game.py               # Game orchestrator, ShotRecord, GameResult
|
+-- strategies/               # Strategy implementations
|   +-- base.py               # Strategy ABC + GameView (immutable snapshot)
|   +-- random_strategy.py    # Uniform random baseline
|   +-- hunt_target.py        # Classical hunt-then-target heuristic
|   +-- parity.py             # Adaptive parity-filtered hunt + oriented targeting
|   +-- entropy_strategy.py   # Information-theoretic entropy minimisation
|
+-- info_theory/              # Bayesian inference layer
|   +-- hypothesis_space.py   # ConfigSampler: Monte Carlo acceptance-rejection
|   +-- probability_map.py    # ProbabilityEngine: marginal P(ship at cell)
|   +-- entropy.py            # Binary entropy, board entropy, expected entropies
|
+-- simulation/
|   +-- runner.py             # GameRunner: run_game(), run_batch(), summarize()
|
+-- ui/
|   +-- app.py                # Streamlit interactive dashboard
|
+-- tests/                    # pytest test suite
|   +-- test_ships.py
|   +-- test_board.py
|   +-- test_game.py
|   +-- test_strategies.py
|
+-- figures/                  # Generated PNGs (via generate_figures.py)
|   +-- entropy_curve.png
|   +-- probability_heatmap.png
|   +-- shots_histogram.png
|   +-- shots_histogram_overlay.png
|
+-- battleship_paper.tex      # LaTeX research paper source
+-- battleship_paper.pdf      # Compiled paper (20 pages)
+-- generate_figures.py       # Regenerate all figures from live simulation data
+-- design.md                 # Architecture and design decisions
+-- requirements.txt          # Python dependencies
```

---

## Installation

Requires **Python 3.10+**.

```bash
git clone <repo-url>
cd battleship_project
pip install -r requirements.txt
```

The core engine and strategies depend only on NumPy. For the UI and figure
generation:

```bash
pip install matplotlib streamlit pandas
```

---

## Quick Start

### Run a single game

```python
from simulation.runner import GameRunner
from strategies.parity import ParityStrategy

runner = GameRunner()
result = runner.run_game(ParityStrategy(), seed=42)
print(f"Sank all ships in {result.total_shots} shots")
```

### Run a batch and collect statistics

```python
from simulation.runner import GameRunner
from strategies.entropy_strategy import EntropyStrategy

runner = GameRunner()
results = runner.run_batch(
    EntropyStrategy(n_samples=100),
    n_games=200,
    base_seed=42,
    verbose=True,
)
stats = GameRunner.summarize(results)
print(stats)
# {'n_games': 200, 'mean_shots': 51.8, 'std_shots': 9.1, ...}
```

### Compare all four strategies

```python
from simulation.runner import GameRunner
from strategies.random_strategy import RandomStrategy
from strategies.hunt_target import HuntTargetStrategy
from strategies.parity import ParityStrategy
from strategies.entropy_strategy import EntropyStrategy

runner = GameRunner()
for name, strat, n in [
    ("Random",     RandomStrategy(),               500),
    ("HuntTarget", HuntTargetStrategy(),           500),
    ("Parity",     ParityStrategy(),               500),
    ("Entropy",    EntropyStrategy(n_samples=100), 100),
]:
    results = runner.run_batch(strat, n, base_seed=42)
    s = GameRunner.summarize(results)
    print(f"{name:12s}  mean={s['mean_shots']:.1f}  std={s['std_shots']:.1f}")
```

---

## Strategies

### 1. Random
Fires at a uniformly random unfired cell each turn. Makes no use of hit/miss
feedback. Theoretical mean: **~95.4 shots** (coupon-collector formula).

### 2. Hunt / Target
Operates in two alternating modes:

- **Hunt** -- random unfired cell until a hit is registered.
- **Target** -- extends shots along the detected ship's axis, inferring
  orientation from collinear hits, until the ship sinks.

Fully stateless: all decisions are re-derived from the observable board each
turn.

### 3. Parity
Augments Hunt/Target with a combinatorial insight: any ship of length >= L must
cover at least one cell of the periodic sub-lattice
`{(r, c) : (r + c) % L == 0}`. The stride L tracks the minimum alive ship size
and grows as ships sink, progressively shrinking the hunt search space from 100
cells down to ~25.

Reduces wasted hunt shots dramatically with zero additional compute cost.

### 4. Entropy (Information-Theoretic)
Selects each shot to **minimise expected post-shot board entropy**:

```
a* = argmin_a  E[H | shoot(a)]
   = argmin_a  p_a * H_after_hit(a) + (1 - p_a) * H_after_miss(a)
```

Uses a Monte Carlo probability engine that samples K valid configurations
consistent with all observations, then computes the O(K * M^2) joint occupancy
matrix to derive conditional entropies. K is configurable:

| K | Quality | Time / game | Recommended use |
|---|---|---|---|
| 100 | Good | ~1.6 s | Batch experiments |
| 300 | High | ~5 s | Default research |
| 500 | Very high | ~9 s | Calibration |

---

## Interactive UI

```bash
streamlit run ui/app.py
```

The dashboard provides:

- Step-by-step game replay with live board visualisation
- Probability heatmaps updated each turn
- Information gain maps (per-cell expected IG)
- Entropy reduction curve over the game
- Head-to-head strategy comparison charts

---

## Tests

```bash
pytest tests/ -v
```

Covers ship geometry, board mechanics, game state transitions, and strategy
legality (no repeated shots, valid coordinates, reset isolation between games).

---

## Design Notes

- **`GameView` is immutable** -- strategies receive a frozen snapshot and
  cannot mutate game state, enforcing the information boundary and enabling
  isolated unit testing.
- **Strategies are stateless by default** -- Random, Hunt/Target, and Parity
  re-derive every decision from the current `shot_grid`, eliminating
  stale-state bugs.
- **`GameRunner` is the sole coupling point** between Strategy and Game.
- **Per-game seeds** follow `base_seed + i`, giving fully reproducible batch
  experiments.
- **SUNK marks all ship cells**, not just the killing shot -- strategies can
  exploit the full sunk-ship boundary for probability map pruning.

See `design.md` for the full architectural rationale.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `numpy` | >= 1.24 | Board arrays, Monte Carlo matrix operations |
| `pytest` | >= 7.4 | Test suite |
| `matplotlib` | any | Figure generation |
| `streamlit` | any | Interactive UI dashboard |
| `pandas` | any | Results DataFrames in UI |
