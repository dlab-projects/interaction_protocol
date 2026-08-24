"""Plot verdict distributions before and after debate for Figure 3."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.axes import Axes
from mpl_lego.labels import bold_text
from mpl_lego.style import use_latex_style

from interaction_protocol.data import load_debate_data
from interaction_protocol.utils import first_round_verdicts


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "experiments"
OUTPUT_PATH = REPO_ROOT / "artifacts" / "figures" / "figure_3.pdf"

DATA_PATHS = (
    DATA_DIR / "sync_h2h_cla_vs_gpt.parquet",
    DATA_DIR / "sync_h2h_cla_vs_gem.parquet",
    DATA_DIR / "sync_h2h_gpt_vs_gem.parquet",
)
PANEL_TITLES = ("Claude vs GPT", "Claude vs Gemini", "Gemini vs GPT")
PANEL_AGENT_NAMES = (
    ("Claude", "GPT"),
    ("Claude", "Gemini"),
    ("GPT", "Gemini"),
)
FINAL_DISTRIBUTION_DROPNA = (False, True, False)

VERDICT_ORDER = ("NTA", "YTA", "NAH", "ESH")
VERDICT_Y_POSITIONS = np.array([5, 4, 3, 2])
NO_CONSENSUS_Y_POSITION = 1.3
MODEL_COLORS = {
    "Claude": "#1F77B4",
    "GPT": "#FF7F0E",
    "Gemini": "#2CA02C",
}
NO_CONSENSUS_COLOR = "#D62728"
COLOR_CYCLE = (
    MODEL_COLORS["Claude"],
    MODEL_COLORS["GPT"],
    MODEL_COLORS["Gemini"],
    NO_CONSENSUS_COLOR,
)

FIGURE_SIZE = (8, 2)
DPI = 300
PANEL_WSPACE = 0.10
AXIS_FACE_COLOR = "0.97"
GRID_COLOR = "0.85"
GRID_LINE_WIDTH = 0.8
CONNECTOR_COLOR = "black"
CONNECTOR_LINE_WIDTH = 1.25
INITIAL_POINT_ALPHA = 0.75
INITIAL_POINT_SIZE = 25
FINAL_MARKER = r"$\downarrow$"
FINAL_MARKER_COLOR = "black"
FINAL_MARKER_SIZE = 80
FINAL_MARKER_Y_SHIFT = 0.165
NO_CONSENSUS_MARKER = "v"
NO_CONSENSUS_MARKER_SIZE = 50

X_LIMITS = (0, 0.9)
Y_LIMITS = (1.15, 5.5)
X_TICKS = np.arange(0, 0.81, 0.2)
AXIS_LABEL = "Proportion of Dilemmas"
AXIS_LABEL_SIZE = 9
TITLE_SIZE = 11
LEGEND_SIZE = 8
LEGEND_LOCATION = "center left"
LEGEND_BBOX = (1.05, 0.5)
LEGEND_HANDLE_LENGTH = 1.0
LEGEND_HANDLE_TEXT_PAD = 0.3
LEGEND_BORDER_AXES_PAD = 0.2
LEGEND_LABEL_SPACING = 0.5
SAVE_PAD_INCHES = 0.02


def normalized_verdict_counts(
    verdicts: pd.Series,
    *,
    dropna: bool,
) -> np.ndarray:
    """Return verdict proportions in the display order."""
    return (
        verdicts.value_counts(normalize=True, dropna=dropna)
        .reindex(VERDICT_ORDER, fill_value=0)
        .to_numpy()
    )


def initial_verdict_proportions(
    dataframe: pd.DataFrame,
    agent_number: int,
) -> np.ndarray:
    """Return an agent's first-round verdict proportions."""
    verdicts = first_round_verdicts(dataframe[f"Agent_{agent_number}_verdicts"])
    return normalized_verdict_counts(verdicts, dropna=False)


def plot_initial_distributions(
    axis: Axes,
    dataframe: pd.DataFrame,
    agent_names: tuple[str, str],
) -> None:
    """Plot both agents' first-round distributions and connecting lines."""
    first_distribution = initial_verdict_proportions(dataframe, 1)
    second_distribution = initial_verdict_proportions(dataframe, 2)

    for distribution, agent_name in zip(
        (first_distribution, second_distribution), agent_names, strict=True
    ):
        axis.scatter(
            distribution,
            VERDICT_Y_POSITIONS,
            color=MODEL_COLORS[agent_name],
            alpha=INITIAL_POINT_ALPHA,
            s=INITIAL_POINT_SIZE,
            zorder=2,
        )

    axis.hlines(
        VERDICT_Y_POSITIONS,
        first_distribution,
        second_distribution,
        color=CONNECTOR_COLOR,
        linewidth=CONNECTOR_LINE_WIDTH,
        zorder=1,
    )


def plot_final_distribution(
    axis: Axes,
    dataframe: pd.DataFrame,
    *,
    dropna: bool,
) -> None:
    """Plot final-verdict and no-consensus proportions."""
    final_distribution = normalized_verdict_counts(
        dataframe["final_verdict"], dropna=dropna
    )
    axis.scatter(
        final_distribution,
        VERDICT_Y_POSITIONS + FINAL_MARKER_Y_SHIFT,
        color=FINAL_MARKER_COLOR,
        marker=FINAL_MARKER,
        s=FINAL_MARKER_SIZE,
        zorder=3,
    )
    axis.scatter(
        dataframe["final_verdict"].isna().mean(),
        NO_CONSENSUS_Y_POSITION,
        color=NO_CONSENSUS_COLOR,
        marker=NO_CONSENSUS_MARKER,
        s=NO_CONSENSUS_MARKER_SIZE,
        zorder=3,
    )


def style_axis(axis: Axes, title: str) -> None:
    """Apply the shared Figure 3 axis styling."""
    axis.grid(True, color=GRID_COLOR, linewidth=GRID_LINE_WIDTH)
    axis.set_axisbelow(True)
    axis.set_facecolor(AXIS_FACE_COLOR)
    axis.set_xlim(*X_LIMITS)
    axis.set_ylim(*Y_LIMITS)
    axis.set_xticks(X_TICKS)
    axis.set_yticks(
        np.sort(VERDICT_Y_POSITIONS),
        [bold_text(verdict) for verdict in reversed(VERDICT_ORDER)],
    )
    axis.set_xlabel(bold_text(AXIS_LABEL), fontsize=AXIS_LABEL_SIZE)
    axis.set_title(bold_text(title), fontsize=TITLE_SIZE)


def add_legend(axis: Axes) -> None:
    """Add the model and outcome legend to the final panel."""
    for model_name in ("Claude", "GPT", "Gemini"):
        axis.scatter(
            [],
            [],
            color=MODEL_COLORS[model_name],
            s=INITIAL_POINT_SIZE,
            label=bold_text(model_name),
        )
    axis.scatter(
        [],
        [],
        color=FINAL_MARKER_COLOR,
        marker=FINAL_MARKER,
        s=FINAL_MARKER_SIZE / 2,
        label=bold_text("Final"),
    )
    axis.scatter(
        [],
        [],
        color=NO_CONSENSUS_COLOR,
        marker=NO_CONSENSUS_MARKER,
        s=NO_CONSENSUS_MARKER_SIZE * 0.8,
        label=bold_text("No\nConsensus"),
    )
    axis.legend(
        loc=LEGEND_LOCATION,
        bbox_to_anchor=LEGEND_BBOX,
        fontsize=LEGEND_SIZE,
        handlelength=LEGEND_HANDLE_LENGTH,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
        borderaxespad=LEGEND_BORDER_AXES_PAD,
        labelspacing=LEGEND_LABEL_SPACING,
    )


def build_figure(dataframes: tuple[pd.DataFrame, ...]) -> plt.Figure:
    """Build the three-panel Figure 3 layout."""
    figure, axes = plt.subplots(
        1,
        3,
        figsize=FIGURE_SIZE,
        dpi=DPI,
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(wspace=PANEL_WSPACE)

    for axis, dataframe, title, agent_names, dropna in zip(
        axes,
        dataframes,
        PANEL_TITLES,
        PANEL_AGENT_NAMES,
        FINAL_DISTRIBUTION_DROPNA,
        strict=True,
    ):
        plot_initial_distributions(axis, dataframe, agent_names)
        plot_final_distribution(axis, dataframe, dropna=dropna)
        style_axis(axis, title)

    add_legend(axes[-1])
    return figure


def main() -> None:
    """Load the analysis data, build Figure 3, and save it as a PDF."""
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
