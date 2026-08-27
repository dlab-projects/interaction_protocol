"""Plot synchronous Llama 3.1 70B debate outcomes for Figure 21."""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from mpl_lego.labels import apply_subplot_labels
from mpl_lego.style import use_latex_style

from interaction_protocol.data import load_debate_data
from interaction_protocol.plotting import (
    SynchronousOutcomePlotConfig,
    plot_synchronous_change_rates,
    plot_synchronous_round_outcomes,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "experiments"
OUTPUT_PATH = REPO_ROOT / "artifacts" / "figures" / "appF_llama70b_sync.pdf"

DATA_PATHS = (
    DATA_DIR / "sync_h2h_gpt_vs_llama70b.parquet",
    DATA_DIR / "sync_h2h_cla_vs_llama70b.parquet",
    DATA_DIR / "sync_h2h_gem_vs_llama70b.parquet",
)
COMPARISON_MODELS = ("GPT-4.1", "Claude 3.7 Sonnet", "Gemini 2.0 Flash")
FOCAL_MODEL = "Llama-70b"
PAIR_LABELS = tuple(f"{FOCAL_MODEL} vs {model}" for model in COMPARISON_MODELS)
PAIR_COLORS = ("#DCDCDC", "#A9A9A9", "#2F4F4F")
COLOR_CYCLE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")

ROUND_VALUES = (1, 2, 3, 4)
ROUND_TICK_LABELS = ("1", "2", "3", "4", "No\nConsensus")
FIGURE_SIZE = (13, 3.6)
DPI = 300
GRID_WIDTH_RATIOS = (1.25, 1.0)
GRID_WSPACE = 0.35
AXIS_FACE_COLOR = "0.98"
GRID_COLOR = "0.8"
EDGE_COLOR = "black"

N_BOOTSTRAP = 1_000
CONFIDENCE_LEVEL = 0.95
RANDOM_STATE = 42
ROUND_BAR_WIDTH = 0.22
CHANGE_BAR_HEIGHT = 0.60
CHANGE_GROUP_SPACING = 1.40
CHANGE_WITHIN_GROUP_OFFSET = 0.35
ROUND_ERROR_CAPSIZE = 1.5
CHANGE_ERROR_CAPSIZE = 1.8
ROUND_Y_LIMITS = (0.0, 1.0)
CHANGE_X_LIMITS = (0.0, 0.5)
AXIS_LABEL_SIZE = 12
TICK_LABEL_SIZE = 10
LEGEND_SIZE = 9
SUBPLOT_LABEL_SIZE = 13
SUBPLOT_LABEL_Y = 1.08
SAVE_PAD_INCHES = 0.10

PLOT_CONFIG = SynchronousOutcomePlotConfig(
    pair_labels=PAIR_LABELS,
    comparison_models=COMPARISON_MODELS,
    focal_model=FOCAL_MODEL,
    pair_colors=PAIR_COLORS,
    round_values=ROUND_VALUES,
    round_tick_labels=ROUND_TICK_LABELS,
    n_bootstrap=N_BOOTSTRAP,
    confidence_level=CONFIDENCE_LEVEL,
    random_state=RANDOM_STATE,
    round_bar_width=ROUND_BAR_WIDTH,
    change_bar_height=CHANGE_BAR_HEIGHT,
    change_group_spacing=CHANGE_GROUP_SPACING,
    change_within_group_offset=CHANGE_WITHIN_GROUP_OFFSET,
    round_error_capsize=ROUND_ERROR_CAPSIZE,
    change_error_capsize=CHANGE_ERROR_CAPSIZE,
    round_y_limits=ROUND_Y_LIMITS,
    change_x_limits=CHANGE_X_LIMITS,
    axis_face_color=AXIS_FACE_COLOR,
    grid_color=GRID_COLOR,
    edge_color=EDGE_COLOR,
    axis_label_size=AXIS_LABEL_SIZE,
    tick_label_size=TICK_LABEL_SIZE,
    legend_size=LEGEND_SIZE,
)


def build_figure(dataframes: tuple[pd.DataFrame, ...]) -> plt.Figure:
    """Build the two-panel Llama 3.1 70B figure."""
    figure = plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    grid = gridspec.GridSpec(
        1,
        2,
        figure=figure,
        width_ratios=GRID_WIDTH_RATIOS,
        wspace=GRID_WSPACE,
    )
    round_axis = figure.add_subplot(grid[0, 0])
    change_axis = figure.add_subplot(grid[0, 1])
    plot_synchronous_round_outcomes(round_axis, dataframes, PLOT_CONFIG)
    plot_synchronous_change_rates(change_axis, dataframes, PLOT_CONFIG)
    apply_subplot_labels(
        [round_axis, change_axis],
        y=SUBPLOT_LABEL_Y,
        size=SUBPLOT_LABEL_SIZE,
        bold=True,
    )
    return figure


def main() -> None:
    """Load Llama 3.1 70B data, build Figure 21, and save it."""
    use_latex_style()
    plt.rcParams["axes.prop_cycle"] = cycler(color=COLOR_CYCLE)

    dataframes = tuple(load_debate_data(path) for path in DATA_PATHS)
    figure = build_figure(dataframes)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_PATH,
        bbox_inches="tight",
        pad_inches=SAVE_PAD_INCHES,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
