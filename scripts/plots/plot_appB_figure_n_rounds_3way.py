"""Plot consensus timing for three-way round-robin debate (Appendix Figure 8)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.axes import Axes
from mpl_lego.labels import apply_subplot_labels, bold_text
from mpl_lego.style import use_latex_style

from interaction_protocol.data import load_debate_data


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "experiments"
OUTPUT_PATH = (
    REPO_ROOT / "artifacts" / "figures" / "appB_figure_n_rounds_3way.pdf"
)

DATA_PATHS = (
    DATA_DIR / "round_robin_3way_gpt_cla_gem.parquet",
    DATA_DIR / "round_robin_3way_cla_gpt_gem.parquet",
    DATA_DIR / "round_robin_3way_gem_gpt_cla.parquet",
    DATA_DIR / "round_robin_3way_gpt_gem_cla.parquet",
    DATA_DIR / "round_robin_3way_cla_gem_gpt.parquet",
    DATA_DIR / "round_robin_3way_gem_cla_gpt.parquet",
)
PANEL_TITLES = (
    "(1) GPT\n(2) Claude\n(3) Gemini",
    "(1) Claude\n(2) GPT\n(3) Gemini",
    "(1) Gemini\n(2) GPT\n(3) Claude",
    "(1) GPT\n(2) Gemini\n(3) Claude",
    "(1) Claude\n(2) Gemini\n(3) GPT",
    "(1) Gemini\n(2) Claude\n(3) GPT",
)

N_ROUNDS_COLUMN = "n_rounds"
FINAL_VERDICT_COLUMN = "final_verdict"
ROUND_CATEGORIES = (1, 2, 3, 4, "No Consensus")
ROUND_TICK_LABELS = ("1", "2", "3", "4", "No\nConsensus")

FIGURE_SIZE = (8, 4)
DPI = 300
SUBPLOT_HSPACE = 0.60
SUBPLOT_WSPACE = 0.20
AXIS_FACE_COLOR = "white"
GRID_COLOR = "#B0B0B0"
Y_LIMITS = (0, 1)
X_POSITIONS = np.arange(len(ROUND_CATEGORIES))
X_LIMITS = (-0.5, 4.5)
BAR_COLOR = "#708090"
BAR_EDGE_COLOR = "black"
BAR_WIDTH = 0.50
COLOR_CYCLE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")

TITLE_SIZE = 10
AXIS_LABEL_SIZE = 10
TICK_LABEL_SIZE = 10
SUBPLOT_LABEL_SIZE = 13
SUBPLOT_LABEL_X = -0.05
SUBPLOT_LABEL_Y = 1.13
SAVE_PAD_INCHES = 0.10


def round_outcome_proportions(dataframe: pd.DataFrame) -> pd.Series:
    """Return proportions for consensus round 1-4 and no consensus."""
    outcomes = dataframe[N_ROUNDS_COLUMN].astype(object).copy()
    outcomes.loc[dataframe[FINAL_VERDICT_COLUMN].isna()] = "No Consensus"
    return (
        outcomes.value_counts(normalize=True)
        .reindex(ROUND_CATEGORIES, fill_value=0.0)
        .astype(float)
    )


def style_axis(axis: Axes) -> None:
    """Apply the shared Appendix Figure 8 axis styling."""
    axis.set_ylim(*Y_LIMITS)
    axis.set_xlim(*X_LIMITS)
    axis.set_xticks(X_POSITIONS, ROUND_TICK_LABELS)
    axis.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    axis.grid(axis="y", color=GRID_COLOR)
    axis.set_axisbelow(True)
    axis.set_facecolor(AXIS_FACE_COLOR)


def plot_round_outcomes(axis: Axes, dataframe: pd.DataFrame) -> None:
    """Plot consensus timing and no-consensus proportions for one model order."""
    proportions = round_outcome_proportions(dataframe)
    axis.bar(
        X_POSITIONS,
        proportions.to_numpy(),
        width=BAR_WIDTH,
        color=BAR_COLOR,
        edgecolor=BAR_EDGE_COLOR,
    )
    style_axis(axis)


def build_figure(dataframes: tuple[pd.DataFrame, ...]) -> plt.Figure:
    """Build the six-panel Appendix Figure 8 layout."""
    figure, axes = plt.subplots(
        2,
        3,
        figsize=FIGURE_SIZE,
        dpi=DPI,
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(hspace=SUBPLOT_HSPACE, wspace=SUBPLOT_WSPACE)

    for axis, dataframe, title in zip(
        axes.ravel(), dataframes, PANEL_TITLES, strict=True
    ):
        plot_round_outcomes(axis, dataframe)
        axis.set_title(bold_text(title), fontsize=TITLE_SIZE)

    for axis in axes[1]:
        axis.set_xlabel(bold_text("Number of Rounds"), fontsize=AXIS_LABEL_SIZE)
    for axis in axes[:, 0]:
        axis.set_ylabel(
            bold_text("Fraction of Dilemmas"),
            fontsize=AXIS_LABEL_SIZE,
        )

    apply_subplot_labels(
        axes.ravel(),
        x=SUBPLOT_LABEL_X,
        y=SUBPLOT_LABEL_Y,
        size=SUBPLOT_LABEL_SIZE,
        bold=True,
    )
    return figure


def main() -> None:
    """Load three-way debate data, build Appendix Figure 8, and save it."""
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
