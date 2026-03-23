"""
ui/app.py — Battleship Research Dashboard
==========================================
Interactive Streamlit dashboard for visualizing Battleship AI strategies,
inspecting Monte Carlo probability maps, tracking information-theoretic entropy
reduction, and running head-to-head strategy comparisons.

Integration points:
  - engine/        : Game, CellState, GameView, STANDARD_FLEET
  - strategies/    : RandomStrategy, HuntTargetStrategy, ParityStrategy, EntropyStrategy
  - info_theory/   : ProbabilityEngine, board_entropy
  - simulation/    : GameRunner

Run from project root:
    streamlit run ui/app.py
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend required for Streamlit
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.colors import ListedColormap

# Ensure project root is on sys.path so engine/strategies/etc. are importable
# when running `streamlit run ui/app.py` from any working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine import (  # noqa: E402
    BOARD_SIZE,
    STANDARD_FLEET,
    CellState,
    Game,
    GameResult,
    ShipType,
)
# strategies must be imported before info_theory to avoid a circular import:
# info_theory.hypothesis_space imports strategies.base.GameView, while
# strategies.entropy_strategy imports info_theory.probability_map.
# Importing strategies first ensures the strategies package is fully
# initialised before info_theory triggers a re-import of it.
from strategies import (  # noqa: E402
    EntropyStrategy,
    GameView,
    HuntTargetStrategy,
    ParityStrategy,
    RandomStrategy,
    Strategy,
)
from info_theory import ProbabilityEngine, board_entropy  # noqa: E402
from simulation import GameRunner  # noqa: E402

# =============================================================================
# CONSTANTS
# =============================================================================

STRATEGY_NAMES: List[str] = ["Random", "HuntTarget", "Parity", "Entropy"]

# Column labels (A–J) and row labels (1–10) for axis annotation
COL_LABELS: List[str] = list("ABCDEFGHIJ")
ROW_LABELS: List[str] = [str(i + 1) for i in range(BOARD_SIZE)]

# Per-cell color scheme — maps CellState integer value to hex colour
_CELL_HEX = {
    int(CellState.UNKNOWN): "#C8C8C8",  # light grey
    int(CellState.MISS):    "#4169E1",  # royal blue
    int(CellState.HIT):     "#FF4500",  # orange-red
    int(CellState.SUNK):    "#8B0000",  # dark red
}

# Custom matplotlib colormap indexed by CellState integer (0–3)
BOARD_CMAP = ListedColormap([
    _CELL_HEX[0],
    _CELL_HEX[1],
    _CELL_HEX[2],
    _CELL_HEX[3],
])

# Comparison chart palette (one colour per strategy, cycles if needed)
_PALETTE = ["#00d4ff", "#FF4500", "#7CFC00", "#FFD700", "#FF69B4"]


# =============================================================================
# SESSION STATE INITIALISATION
# =============================================================================

def init_session_state() -> None:
    """
    Initialise all session_state keys with safe defaults on first load.
    Does nothing if keys already exist, preserving state across reruns.
    """
    defaults: Dict = {
        # Game viewer state
        "game":           None,   # Live engine.Game instance
        "strategy":       None,   # Live Strategy instance
        "prob_engine":    None,   # ProbabilityEngine (for non-Entropy strategies)
        "frames":         [],     # List[dict] — one frame per turn (see run_single_game_step)
        "replay_slider":  0,      # Controlled by st.slider via key=
        "game_started":   False,
        "game_over":      False,
        "autoplay":       False,
        # Comparison tab
        "sim_results":    {},     # {strategy_name: List[GameResult]}
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# =============================================================================
# STRATEGY FACTORY
# =============================================================================

def make_strategy(name: str, n_samples: int, seed: Optional[int]) -> Strategy:
    """Instantiate a Strategy by display name with the given hyperparameters."""
    if name == "Random":
        return RandomStrategy(seed=seed)
    if name == "HuntTarget":
        return HuntTargetStrategy(seed=seed)
    if name == "Parity":
        return ParityStrategy(seed=seed)
    if name == "Entropy":
        return EntropyStrategy(n_samples=n_samples, rng_seed=seed)
    raise ValueError(f"Unknown strategy: {name!r}")


# =============================================================================
# PROBABILITY MAP & ENTROPY HELPERS
# =============================================================================

def get_prob_map(
    view: GameView,
    strategy: Strategy,
    prob_engine: ProbabilityEngine,
    show_prob: bool,
) -> Optional[np.ndarray]:
    """
    Retrieve the (B, B) probability map for the current view.

    For EntropyStrategy the map is a free by-product of select_action() —
    we read strategy.last_prob_map rather than re-running the sampler.

    For all other strategies the shared ProbabilityEngine is used only if
    the user has enabled 'Show Probability Map' in the sidebar.
    """
    if isinstance(strategy, EntropyStrategy):
        # Already computed inside select_action(); free to read.
        return strategy.last_prob_map
    if show_prob:
        return prob_engine.compute_prob_map(view)
    return None


def entropy_from_prob_map(
    prob_map: Optional[np.ndarray],
    view: GameView,
) -> Optional[float]:
    """Return board entropy in nats, or None if no probability map is available."""
    if prob_map is None:
        return None
    return board_entropy(prob_map, view)


# =============================================================================
# BOARD RENDERING
# =============================================================================

def render_board(
    view: GameView,
    last_action: Optional[Tuple[int, int]] = None,
    prob_map: Optional[np.ndarray] = None,
    ig_map: Optional[np.ndarray] = None,
    show_prob: bool = True,
    show_ig: bool = False,
    figsize: Tuple[float, float] = (5.5, 5.5),
) -> plt.Figure:
    """
    Render the 10×10 board as a matplotlib Figure.

    Layers (bottom to top):
      1. Base board  — coloured by CellState (imshow with BOARD_CMAP)
      2. Overlay     — probability heatmap (YlOrRd) OR information-gain heatmap (plasma)
                       applied only to UNKNOWN cells via masked array
      3. Grid lines  — thin white lines separating cells
      4. Highlight   — gold rectangle around last_action cell

    Arguments:
        view        — post-shot GameView whose shot_grid is displayed
        last_action — (row, col) of the most recent shot (highlighted in gold)
        prob_map    — (B, B) float array of occupancy probabilities [0, 1]
        ig_map      — (B, B) float array of expected information gain
        show_prob   — render probability overlay when prob_map is provided
        show_ig     — render IG overlay instead of prob overlay when ig_map provided
        figsize     — matplotlib figure size
    """
    fig, ax = plt.subplots(figsize=figsize, facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    B = view.board_size
    shot_grid = view.shot_grid.astype(float)

    # --- Layer 1: base board ---
    ax.imshow(
        shot_grid,
        cmap=BOARD_CMAP,
        vmin=0,
        vmax=3,
        aspect="equal",
        interpolation="nearest",
        zorder=1,
    )

    # --- Layer 2: overlay (IG takes priority over probability) ---
    unknown_mask = view.shot_grid == int(CellState.UNKNOWN)

    if show_ig and ig_map is not None:
        # Information gain — higher = more informative shot
        overlay = np.where(unknown_mask, ig_map, np.nan)
        masked = np.ma.masked_invalid(overlay)
        if masked.count() > 0:
            vmax = float(np.nanmax(overlay[np.isfinite(overlay)])) or 1e-6
            ax.imshow(
                masked,
                cmap="plasma",
                alpha=0.70,
                aspect="equal",
                interpolation="nearest",
                vmin=0,
                vmax=vmax,
                zorder=2,
            )

    elif show_prob and prob_map is not None:
        # Occupancy probability — higher = more likely a ship is here
        overlay = np.where(unknown_mask, prob_map, np.nan)
        masked = np.ma.masked_invalid(overlay)
        if masked.count() > 0:
            finite = overlay[np.isfinite(overlay)]
            vmax = float(finite.max()) if finite.size > 0 else 1e-6
            ax.imshow(
                masked,
                cmap="YlOrRd",
                alpha=0.65,
                aspect="equal",
                interpolation="nearest",
                vmin=0,
                vmax=vmax,
                zorder=2,
            )

    # --- Layer 3: grid lines ---
    for i in range(B + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.5, alpha=0.6, zorder=3)
        ax.axvline(i - 0.5, color="white", linewidth=0.5, alpha=0.6, zorder=3)

    # --- Layer 4: highlight last action ---
    if last_action is not None:
        r, c = last_action
        rect = plt.Rectangle(
            (c - 0.48, r - 0.48),
            0.96,
            0.96,
            linewidth=2.5,
            edgecolor="#FFD700",
            facecolor="none",
            zorder=5,
        )
        ax.add_patch(rect)

    # --- Axis labels: columns (A–J) on top, rows (1–10) on left ---
    ax.set_xticks(range(B))
    ax.set_xticklabels(COL_LABELS, color="white", fontsize=9)
    ax.set_yticks(range(B))
    ax.set_yticklabels(ROW_LABELS, color="white", fontsize=9)
    ax.tick_params(length=0)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")

    # --- Legend ---
    legend_patches = [
        mpatches.Patch(color=_CELL_HEX[int(CellState.UNKNOWN)], label="Unknown"),
        mpatches.Patch(color=_CELL_HEX[int(CellState.MISS)],    label="Miss"),
        mpatches.Patch(color=_CELL_HEX[int(CellState.HIT)],     label="Hit"),
        mpatches.Patch(color=_CELL_HEX[int(CellState.SUNK)],    label="Sunk"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=4,
        fontsize=7.5,
        framealpha=0.5,
        facecolor="#1a1a2e",
        labelcolor="white",
        handlelength=1.0,
    )

    fig.tight_layout(pad=0.4)
    return fig


# =============================================================================
# ENTROPY PLOT
# =============================================================================

def render_entropy_plot(entropy_history: List[float]) -> plt.Figure:
    """
    Line chart of board entropy (nats) over game turns.

    Entropy is computed from the marginal probability map BEFORE each shot.
    It should decrease monotonically as shots eliminate hypotheses about ship
    placement — visually demonstrating information acquisition.
    """
    fig, ax = plt.subplots(figsize=(5, 2.8), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")

    if len(entropy_history) >= 2:
        turns = list(range(len(entropy_history)))
        ax.plot(
            turns,
            entropy_history,
            color="#00d4ff",
            linewidth=2,
            marker="o",
            markersize=3,
            zorder=2,
        )
        ax.fill_between(turns, entropy_history, alpha=0.15, color="#00d4ff", zorder=1)
        ax.axhline(0, color="#555", linewidth=0.8, linestyle="--", alpha=0.6)
    elif len(entropy_history) == 1:
        ax.scatter([0], entropy_history, color="#00d4ff", s=30)

    ax.set_xlabel("Turn (shot number)", color="white", fontsize=9)
    ax.set_ylabel("H (nats)", color="white", fontsize=9)
    ax.set_title("Board Entropy Reduction", color="white", fontsize=10, pad=4)
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    fig.tight_layout(pad=0.4)
    return fig


# =============================================================================
# INFORMATION GAIN TABLE
# =============================================================================

def render_ig_table(ig_map: np.ndarray, view: GameView) -> pd.DataFrame:
    """
    Build a DataFrame of the top 5 UNKNOWN cells ranked by expected information gain.

    Information gain IG(a) = H_current − E[H | shoot(a)] measures how much
    uncertainty a shot at cell a is expected to remove.  Higher is better.

    Only considers cells where view.shot_grid == UNKNOWN and IG is finite & positive.
    """
    rows = []
    B = view.board_size
    for r in range(B):
        for c in range(B):
            if view.shot_grid[r, c] == int(CellState.UNKNOWN):
                ig = float(ig_map[r, c])
                if np.isfinite(ig) and ig > 0:
                    rows.append({
                        "Cell": f"{COL_LABELS[c]}{ROW_LABELS[r]}",
                        "Info Gain (nats)": round(ig, 4),
                    })

    if not rows:
        return pd.DataFrame()

    df = (
        pd.DataFrame(rows)
        .sort_values("Info Gain (nats)", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )
    df.insert(0, "Rank", range(1, len(df) + 1))
    return df


# =============================================================================
# GAME LOGIC
# =============================================================================

def start_new_game(strategy_name: str, n_samples: int, seed: Optional[int]) -> None:
    """
    Reset all game-viewer session state and start a fresh game.

    Called when the user clicks "Start New Game". Creates:
      - A new Game instance (seeds the board placement RNG)
      - A new Strategy instance (calls strategy.reset())
      - A new ProbabilityEngine (used for non-Entropy strategies when
        the probability overlay is enabled)
      - An initial frame representing the empty board (no shots yet)
    """
    game = Game(fleet=STANDARD_FLEET, seed=seed)
    strategy = make_strategy(strategy_name, n_samples, seed)
    strategy.reset()
    prob_engine = ProbabilityEngine(n_samples=n_samples, rng_seed=seed)

    # Frame 0: empty board — no shots, no probability map yet
    initial_view = GameView.from_game(game)
    initial_frame: Dict = {
        "view":     initial_view,   # post-shot view (= initial empty board)
        "prob_map": None,
        "ig_map":   None,
        "entropy":  None,
        "action":   None,
        "result":   None,
    }

    st.session_state.game         = game
    st.session_state.strategy     = strategy
    st.session_state.prob_engine  = prob_engine
    st.session_state.frames       = [initial_frame]
    st.session_state.replay_slider = 0
    st.session_state.game_started = True
    st.session_state.game_over    = False
    st.session_state.autoplay     = False


def run_single_game_step(show_prob: bool) -> None:
    """
    Advance the game by exactly one shot and record a new frame.

    Frame storage model
    -------------------
    Each frame stores the **post-shot** board view (for display) together with
    the probability / IG maps that were computed from the **pre-shot** view
    (i.e. the board state the strategy actually observed when making its
    decision).  This lets the replay slider correctly show "what the strategy
    was thinking" at each turn while also displaying where the shot landed.

    Step-by-step flow:
      1. Snapshot pre-shot state  → pre_view
      2. strategy.select_action(pre_view)
             For EntropyStrategy this populates last_prob_map,
             last_info_gain_map, last_expected_H (diagnostic maps).
      3. Extract diagnostic maps from strategy (post select_action).
      4. Compute board entropy from the pre-shot probability map.
      5. game.fire(row, col)  →  CellState result
      6. Snapshot post-shot state  → post_view  (for board display)
      7. Append frame and advance replay cursor to latest turn.
    """
    game: Game               = st.session_state.game
    strategy: Strategy       = st.session_state.strategy
    prob_engine: ProbabilityEngine = st.session_state.prob_engine

    if game.is_over:
        st.session_state.game_over = True
        st.session_state.autoplay  = False
        return

    # Step 1 — pre-shot snapshot (what the strategy sees)
    pre_view = GameView.from_game(game)

    # Step 2 — strategy decides (may be expensive for EntropyStrategy)
    action: Tuple[int, int] = strategy.select_action(pre_view)

    # Step 3 — collect diagnostic maps
    if isinstance(strategy, EntropyStrategy):
        # EntropyStrategy caches these after every select_action call
        prob_map = strategy.last_prob_map
        ig_map   = strategy.last_info_gain_map
    elif show_prob:
        prob_map = prob_engine.compute_prob_map(pre_view)
        ig_map   = None
    else:
        prob_map = None
        ig_map   = None

    # Step 4 — board entropy (in nats) based on pre-shot probability map
    entropy = entropy_from_prob_map(prob_map, pre_view)

    # Step 5 — fire the chosen cell
    result: CellState = game.fire(*action)

    # Step 6 — post-shot snapshot (for rendering the updated board)
    post_view = GameView.from_game(game)

    # Step 7 — record and advance cursor
    frame: Dict = {
        "view":     post_view,   # post-shot board state
        "prob_map": prob_map,    # pre-shot occupancy probabilities
        "ig_map":   ig_map,      # pre-shot information gain map
        "entropy":  entropy,     # H_board before this shot
        "action":   action,
        "result":   result,
    }
    st.session_state.frames.append(frame)
    st.session_state.replay_slider = len(st.session_state.frames) - 1

    if game.is_over:
        st.session_state.game_over = True
        st.session_state.autoplay  = False


# =============================================================================
# SIMULATION BATCH
# =============================================================================

def run_simulation_batch(
    strategy_names: List[str],
    n_games: int,
    base_seed: int,
    n_samples: int,
) -> Dict[str, List[GameResult]]:
    """
    Run n_games for each named strategy using GameRunner.run_batch().

    Each strategy gets its own fresh instance; seeds are deterministic
    (seed_i = base_seed + i) for reproducibility.

    Results are stored in st.session_state.sim_results and returned.

    Note: EntropyStrategy is slow (~0.5–3 s/game depending on n_samples).
    Keep n_games ≤ 100 when Entropy is included.
    """
    runner = GameRunner(fleet=STANDARD_FLEET, board_size=BOARD_SIZE)
    results: Dict[str, List[GameResult]] = {}
    for name in strategy_names:
        strat = make_strategy(name, n_samples, seed=base_seed)
        results[name] = runner.run_batch(strat, n_games=n_games, base_seed=base_seed)
    st.session_state.sim_results = results
    return results


# =============================================================================
# STRATEGY COMPARISON DASHBOARD
# =============================================================================

def render_strategy_comparison(results: Dict[str, List[GameResult]]) -> None:
    """
    Render the full strategy comparison dashboard from batch simulation results.

    Outputs:
      A. Summary statistics table  — mean, std, median, P10/P25/P75/P90, min/max
      B. Histogram                 — shot-count distributions (overlaid, alpha)
      C. Boxplot                   — side-by-side per-strategy box-and-whisker
      D. Convergence curve         — running mean over games to show stability
    """
    if not results:
        return

    strategy_list = list(results.keys())
    colors = {name: _PALETTE[i % len(_PALETTE)] for i, name in enumerate(strategy_list)}

    # --- A. Statistics table ---
    rows = []
    for name, game_results in results.items():
        shots = np.array([r.total_shots for r in game_results], dtype=float)
        rows.append({
            "Strategy":    name,
            "N":           len(game_results),
            "Mean":        round(float(shots.mean()), 1),
            "Std":         round(float(shots.std(ddof=1)), 1),
            "Median":      round(float(np.median(shots)), 1),
            "P10":         int(np.percentile(shots, 10)),
            "P25":         int(np.percentile(shots, 25)),
            "P75":         int(np.percentile(shots, 75)),
            "P90":         int(np.percentile(shots, 90)),
            "Min":         int(shots.min()),
            "Max":         int(shots.max()),
        })
    df_stats = pd.DataFrame(rows).set_index("Strategy")
    st.dataframe(df_stats, use_container_width=True)

    # --- B & C: Histogram + Boxplot side-by-side ---
    col_hist, col_box = st.columns(2)

    # Shared bin edges across all strategies for fair comparison
    all_shots_flat = [r.total_shots for gr in results.values() for r in gr]
    bins = np.linspace(min(all_shots_flat), max(all_shots_flat), 35)

    with col_hist:
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#1a1a2e")
        ax.set_facecolor("#16213e")
        for name, game_results in results.items():
            shots = [r.total_shots for r in game_results]
            ax.hist(shots, bins=bins, alpha=0.55, label=name,
                    color=colors[name], edgecolor="none")
        ax.set_xlabel("Shots to Win", color="white", fontsize=9)
        ax.set_ylabel("Frequency", color="white", fontsize=9)
        ax.set_title("Shot Distribution", color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=8)
        ax.legend(fontsize=8, facecolor="#16213e", labelcolor="white", framealpha=0.6)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        fig.tight_layout(pad=0.4)
        st.pyplot(fig)
        plt.close(fig)

    with col_box:
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#1a1a2e")
        ax.set_facecolor("#16213e")
        shot_lists = [[r.total_shots for r in results[name]] for name in strategy_list]
        bp = ax.boxplot(
            shot_lists,
            labels=strategy_list,
            patch_artist=True,
            medianprops=dict(color="white", linewidth=2),
            whiskerprops=dict(color="#aaa", linewidth=1.2),
            capprops=dict(color="#aaa", linewidth=1.2),
            flierprops=dict(marker="o", markerfacecolor="#aaa", alpha=0.4, markersize=3),
        )
        for patch, name in zip(bp["boxes"], strategy_list):
            patch.set_facecolor(colors[name])
            patch.set_alpha(0.75)
        ax.set_ylabel("Shots to Win", color="white", fontsize=9)
        ax.set_title("Strategy Comparison (Boxplot)", color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        fig.tight_layout(pad=0.4)
        st.pyplot(fig)
        plt.close(fig)

    # --- D. Convergence curve (running mean) ---
    st.subheader("Convergence — Running Mean Shots")
    fig, ax = plt.subplots(figsize=(9, 3), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    for name, game_results in results.items():
        shots = np.array([r.total_shots for r in game_results], dtype=float)
        running_mean = np.cumsum(shots) / np.arange(1, len(shots) + 1)
        ax.plot(running_mean, label=name, color=colors[name], linewidth=2)
    ax.set_xlabel("Games", color="white", fontsize=9)
    ax.set_ylabel("Running Mean Shots", color="white", fontsize=9)
    ax.set_title("Convergence of Strategy Performance Estimate", color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=8)
    ax.legend(fontsize=9, facecolor="#16213e", labelcolor="white", framealpha=0.6)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    fig.tight_layout(pad=0.4)
    st.pyplot(fig)
    plt.close(fig)


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main() -> None:
    # ------------------------------------------------------------------
    # Page config — must be the first Streamlit call
    # ------------------------------------------------------------------
    st.set_page_config(
        page_title="Battleship Research Dashboard",
        page_icon="⚓",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Minimal dark-theme polish via injected CSS
    st.markdown(
        """
        <style>
            .stApp { background-color: #1a1a2e; color: #e0e0e0; }
            .block-container { padding-top: 0.8rem; padding-bottom: 0.5rem; }
            h1, h2, h3 { color: #00d4ff; }
            .stMetric { background: #16213e; border-radius: 8px; padding: 8px; }
            .stButton > button {
                background: #0f3460; color: white;
                border: 1px solid #00d4ff; border-radius: 6px;
            }
            .stButton > button:hover { background: #00d4ff; color: #1a1a2e; }
            .stTabs [data-baseweb="tab"] { color: #aaa; }
            .stTabs [aria-selected="true"] { color: #00d4ff !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    init_session_state()

    # ==================================================================
    # SIDEBAR — all user-configurable parameters
    # ==================================================================
    with st.sidebar:
        st.title("⚓ Controls")
        st.markdown("---")

        st.subheader("Game Settings")
        strategy_name: str = st.selectbox(
            "Strategy",
            options=STRATEGY_NAMES,
            index=2,              # Default: Parity (good balance of speed vs performance)
            help="AI strategy used to choose shots",
        )
        n_samples: int = st.slider(
            "Monte Carlo Samples",
            min_value=50,
            max_value=1000,
            value=300,
            step=50,
            help=(
                "Samples used by ProbabilityEngine / EntropyStrategy. "
                "Higher = more accurate probability maps but slower per move."
            ),
        )
        seed_val = st.number_input(
            "Random Seed", value=42, min_value=0, max_value=99999, step=1
        )
        seed: Optional[int] = int(seed_val)

        st.markdown("---")
        st.subheader("Visualisation")
        show_prob: bool = st.toggle(
            "Show Probability Map",
            value=True,
            help="Overlay ship-occupancy probability heatmap on the board",
        )
        show_entropy: bool = st.toggle(
            "Show Entropy Plot",
            value=True,
            help="Display the entropy-reduction curve below the board",
        )
        show_ig: bool = st.toggle(
            "Show Information Gain",
            value=False,
            help="Overlay expected IG instead of probability (EntropyStrategy only)",
        )

        st.markdown("---")
        st.subheader("Autoplay")
        autoplay_speed: float = st.slider(
            "Seconds / Move",
            min_value=0.1,
            max_value=3.0,
            value=0.8,
            step=0.1,
        )

        st.markdown("---")
        st.subheader("Simulation")
        n_sim_games = int(
            st.number_input(
                "Games per Strategy",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Tip: keep ≤ 50 when Entropy is included (slow)",
            )
        )
        sim_seed = int(
            st.number_input(
                "Simulation Seed", value=0, min_value=0, max_value=99999, step=1
            )
        )

    # ==================================================================
    # PAGE HEADER
    # ==================================================================
    st.title("Battleship Research Dashboard")
    st.caption(
        "Interactive visualisation of probabilistic reasoning and "
        "information-theoretic strategy analysis"
    )

    # ==================================================================
    # TABS
    # ==================================================================
    tab_game, tab_compare = st.tabs(["🎮  Game Viewer", "📊  Strategy Comparison"])

    # ==================================================================
    # TAB 1 — GAME VIEWER
    # ==================================================================
    with tab_game:

        # ---- Top controls row ----------------------------------------
        c1, c2, c3, c4 = st.columns([2, 2, 2, 4])

        with c1:
            if st.button("▶  Start New Game", use_container_width=True):
                start_new_game(strategy_name, n_samples, seed)

        with c2:
            step_disabled = (
                not st.session_state.game_started
                or st.session_state.game_over
                or st.session_state.autoplay
            )
            if st.button("→  Next Move", disabled=step_disabled, use_container_width=True):
                run_single_game_step(show_prob)

        with c3:
            ap_disabled = (
                not st.session_state.game_started
                or st.session_state.game_over
            )
            ap_label = "⏸  Stop" if st.session_state.autoplay else "⏩  Auto Play"
            if st.button(ap_label, disabled=ap_disabled, use_container_width=True):
                st.session_state.autoplay = not st.session_state.autoplay

        # ---- Status metrics ------------------------------------------
        if st.session_state.game_started:
            game: Game   = st.session_state.game
            frames: List = st.session_state.frames

            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Turn", game.turn)
            with m2:
                strat_obj = st.session_state.strategy
                st.metric("Strategy", strat_obj.name if strat_obj else "—")
            with m3:
                latest = frames[-1]
                if latest["action"]:
                    r, c = latest["action"]
                    shot_str = f"{COL_LABELS[c]}{ROW_LABELS[r]}"
                else:
                    shot_str = "—"
                st.metric("Last Shot", shot_str)
            with m4:
                res = latest["result"]
                st.metric("Result", res.name if res else "—")
            with m5:
                current_view = GameView.from_game(game)
                st.metric("Ships Left", len(current_view.afloat_ships))

            if st.session_state.game_over:
                st.success(
                    f"Game over — won in **{game.turn} shots** "
                    f"(strategy: {st.session_state.strategy.name})"
                )

        # ---- Replay slider -------------------------------------------
        if st.session_state.game_started and len(st.session_state.frames) > 1:
            max_idx = len(st.session_state.frames) - 1
            # st.slider with key= writes to st.session_state.replay_slider automatically
            st.slider(
                "Replay: Turn",
                min_value=0,
                max_value=max_idx,
                key="replay_slider",
                help="Scrub through game history",
            )

        # ---- Main display: board + info panel -----------------------
        if st.session_state.game_started:
            frames: List    = st.session_state.frames
            replay_idx: int = st.session_state.get("replay_slider", len(frames) - 1)
            replay_idx      = max(0, min(replay_idx, len(frames) - 1))

            display_frame   = frames[replay_idx]
            board_view      = display_frame["view"]      # post-shot board state
            disp_prob       = display_frame["prob_map"] if show_prob else None
            disp_ig         = display_frame["ig_map"]   if show_ig   else None
            disp_action     = display_frame["action"]

            col_board, col_info = st.columns([3, 2])

            # -- Board ------------------------------------------------
            with col_board:
                overlay_label = ""
                if show_ig and disp_ig is not None:
                    overlay_label = " + Info Gain"
                elif show_prob and disp_prob is not None:
                    overlay_label = " + Prob Map"
                st.subheader(f"Board — Turn {replay_idx}{overlay_label}")

                fig_board = render_board(
                    view=board_view,
                    last_action=disp_action,
                    prob_map=disp_prob,
                    ig_map=disp_ig,
                    show_prob=show_prob and disp_prob is not None,
                    show_ig=show_ig and disp_ig is not None,
                )
                st.pyplot(fig_board)
                plt.close(fig_board)

            # -- Info panel -------------------------------------------
            with col_info:

                # Entropy sub-section
                if show_entropy:
                    entropy_history = [
                        f["entropy"]
                        for f in frames[: replay_idx + 1]
                        if f["entropy"] is not None
                    ]
                    st.subheader("Entropy")
                    if entropy_history:
                        current_H = entropy_history[-1]
                        st.metric(
                            "H (nats)",
                            f"{current_H:.3f}",
                            delta=f"{entropy_history[-1] - entropy_history[-2]:.3f}"
                            if len(entropy_history) >= 2
                            else None,
                            delta_color="inverse",
                        )
                        fig_e = render_entropy_plot(entropy_history)
                        st.pyplot(fig_e)
                        plt.close(fig_e)
                    else:
                        st.caption(
                            "Enable 'Show Probability Map' to track entropy, "
                            "or advance at least one move."
                        )

                # Ship status sub-section
                st.subheader("Ship Status")
                for ship_type in STANDARD_FLEET:
                    sunk_here = ship_type in board_view.sunk_ships
                    icon  = "🔴" if sunk_here else "🟢"
                    label = (
                        f"~~{ship_type.display_name} ({ship_type.size})~~"
                        if sunk_here
                        else f"{ship_type.display_name} ({ship_type.size})"
                    )
                    st.markdown(f"{icon} {label}")

                # Information gain table (EntropyStrategy only)
                if show_ig and disp_ig is not None:
                    st.subheader("Top 5 Cells by Info Gain")
                    ig_df = render_ig_table(disp_ig, board_view)
                    if not ig_df.empty:
                        st.dataframe(ig_df, use_container_width=True, hide_index=True)
                    else:
                        st.caption("No IG data for this turn.")
                elif show_ig and not isinstance(st.session_state.strategy, EntropyStrategy):
                    st.info(
                        "Information gain overlay is only available when using "
                        "the **Entropy** strategy."
                    )

        else:
            st.info(
                "Select a strategy in the sidebar and click **▶ Start New Game** to begin."
            )

    # ==================================================================
    # TAB 2 — STRATEGY COMPARISON
    # ==================================================================
    with tab_compare:
        st.subheader("Strategy Comparison Dashboard")
        st.caption(
            "Simulate many games per strategy to compare performance distributions "
            "and verify convergence."
        )

        selected: List[str] = st.multiselect(
            "Strategies to compare",
            options=STRATEGY_NAMES,
            default=["Random", "HuntTarget", "Parity"],
        )

        col_desc, col_btn = st.columns([4, 1])
        with col_desc:
            entropy_warn = (
                "  ⚠️ *Entropy is slow — consider n ≤ 50.*"
                if "Entropy" in selected
                else ""
            )
            st.markdown(
                f"Run **{n_sim_games}** games per strategy, seed={sim_seed}.{entropy_warn}"
            )
        with col_btn:
            run_pressed = st.button(
                "▶  Run",
                use_container_width=True,
                disabled=len(selected) == 0,
            )

        if run_pressed and selected:
            with st.spinner(
                f"Running {n_sim_games} × {len(selected)} games…"
            ):
                results = run_simulation_batch(
                    selected,
                    n_games=n_sim_games,
                    base_seed=sim_seed,
                    n_samples=n_samples,
                )
            st.success(
                f"Done — {n_sim_games} games × {len(selected)} strategies."
            )

        if st.session_state.sim_results:
            render_strategy_comparison(st.session_state.sim_results)
        else:
            st.info("Select strategies above and click **▶ Run** to start.")

    # ==================================================================
    # AUTOPLAY LOOP
    # Runs at the *end* of each Streamlit script execution.
    # If autoplay is active and the game is not over, execute one step
    # then sleep briefly before triggering a full rerun — creating a
    # controlled playback loop without blocking the event loop.
    # ==================================================================
    if (
        st.session_state.get("autoplay", False)
        and st.session_state.get("game_started", False)
        and not st.session_state.get("game_over", False)
    ):
        run_single_game_step(show_prob)
        time.sleep(autoplay_speed)
        st.rerun()


if __name__ == "__main__":
    main()
