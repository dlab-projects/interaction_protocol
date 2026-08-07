"""Plot value use and inheritance during synchronous debate for Figure 4."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.axes import Axes
from mpl_lego.labels import apply_subplot_labels, bold_text
from mpl_lego.style import use_latex_style

from interaction_protocol.data import load_debate_data
from interaction_protocol.utils import bootstrap_statistic_df


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "analysis"
OUTPUT_PATH = REPO_ROOT / "artifacts" / "figures" / "figure_4.pdf"

PAIR_DATA_PATHS = (
    DATA_DIR / "sync_h2h_cla_vs_gpt.parquet",
    DATA_DIR / "sync_h2h_cla_vs_gem.parquet",
    DATA_DIR / "sync_h2h_gpt_vs_gem.parquet",
)
PAIR_AGENT_NAMES = (
    ("Claude", "GPT"),
    ("Claude", "Gemini"),
    ("GPT", "Gemini"),
)
PAIR_ROW_LABELS = (
    "GPT\nvs.\nClaude",
    "Claude\nvs.\nGemini",
    "Gemini\nvs.\nGPT",
)
AGENT_COLORS = {
    "Claude": "#1F77B4",
    "GPT": "#FF7F0E",
    "Gemini": "#2CA02C",
}
COLOR_CYCLE = (
    AGENT_COLORS["Claude"],
    AGENT_COLORS["GPT"],
    AGENT_COLORS["Gemini"],
)

AGENT_VERDICT_COLUMNS = ("Agent_1_verdicts", "Agent_2_verdicts")
AGENT_VALUE_COLUMNS = ("Agent_1_values", "Agent_2_values")
ROUND_COLUMNS = AGENT_VERDICT_COLUMNS + AGENT_VALUE_COLUMNS

FIGURE_SIZE = (14, 7.5)
DPI = 300
HSPACE = 0.50
WSPACE = 0.10
AXIS_FACE_COLOR = "0.98"
GRID_LINESTYLE = "--"
GRID_ALPHA = 0.50
BAR_ERROR_LINE_WIDTH = 1
BAR_ERROR_CAPSIZE = 2

N_BOOTSTRAP = 1_000
CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_SEED = 42
N_VALUES_TO_CONSIDER = 5
N_INHERITED_VALUES_TO_CONSIDER = 5

VALUE_LABEL_SIZE = 10
AXIS_LABEL_SIZE = 10
DIRECTION_LABEL_SIZE = 9
PAIR_LABEL_SIZE = 14
PAIR_LABEL_PAD = 28
VALUE_LABEL_PAD_POINTS = 3
DIRECTION_LABEL_Y = -0.18
DIRECTION_LEFT_X = 0.05
DIRECTION_RIGHT_X = 0.95
SUBPLOT_LABEL_X = -0.03
SUBPLOT_LABEL_Y = 1.05
SUBPLOT_LABEL_SIZE = 15

DIFFERENCE_X_LIMITS = (-21, 21)
DIFFERENCE_X_TICKS = (-15, -10, -5, 0, 5, 10, 15)
DIFFERENCE_X_TICK_LABELS = ("15", "10", "5", "0", "5", "10", "15")
INHERITED_X_LIMITS = (-21, 21)
INHERITED_X_TICKS = DIFFERENCE_X_TICKS
INHERITED_X_TICK_LABELS = DIFFERENCE_X_TICK_LABELS
TOP_INHERITED_X_LIMITS = (-40, 40)
TOP_INHERITED_X_TICKS = (-30, -20, -10, 0, 10, 20, 30)
TOP_INHERITED_X_TICK_LABELS = ("30", "20", "10", "0", "10", "20", "30")

DIFFERENCE_X_LABEL = r"Difference in Value Occurrence (\%)"
INHERITED_X_LABEL = r"Inherited Value Occurrence (\%)"
SAVE_PAD_INCHES = 0.02


def explode_debate_rounds(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Expand each debate into one row per synchronous round."""
    return dataframe.explode(list(ROUND_COLUMNS)).reset_index(drop=True)


def value_proportions(exploded: pd.DataFrame, agent_index: int) -> pd.Series:
    """Return per-message value occurrence rates for one agent."""
    value_column = AGENT_VALUE_COLUMNS[agent_index]
    return (
        exploded[value_column].explode().value_counts() / exploded.shape[0]
    ).sort_index()


def value_differences(exploded: pd.DataFrame) -> pd.Series:
    """Return Agent 1 minus Agent 2 value occurrence rates."""
    agent_1 = value_proportions(exploded, 0)
    agent_2 = value_proportions(exploded, 1)
    return (agent_1 - agent_2).sort_values().dropna()


def bootstrap_value_difference(
    exploded: pd.DataFrame,
    value: str,
) -> dict[str, float]:
    """Bootstrap the occurrence-rate difference for one value."""

    def statistic(dataframe: pd.DataFrame) -> float:
        agent_1_counts = dataframe[AGENT_VALUE_COLUMNS[0]].explode().value_counts()
        agent_2_counts = dataframe[AGENT_VALUE_COLUMNS[1]].explode().value_counts()
        return (
            agent_1_counts.get(value, 0) - agent_2_counts.get(value, 0)
        ) / dataframe.shape[0]

    return bootstrap_statistic_df(
        df=exploded,
        statistic_func=statistic,
        n_bootstrap=N_BOOTSTRAP,
        confidence_level=CONFIDENCE_LEVEL,
        random_state=BOOTSTRAP_SEED,
    )


def estimate_and_errors(
    bootstrap_result: dict[str, float],
) -> tuple[float, float, float]:
    """Return a bootstrap estimate and asymmetric errors in percentage points."""
    estimate = 100 * float(bootstrap_result["mean"])
    lower_error = 100 * float(bootstrap_result["lower_err"])
    upper_error = 100 * float(bootstrap_result["upper_err"])
    return estimate, lower_error, upper_error


def interval_crosses_zero(
    estimate: float,
    lower_error: float,
    upper_error: float,
) -> bool:
    """Return whether an asymmetric interval includes zero."""
    return estimate - lower_error <= 0 <= estimate + upper_error


def significant_difference_entries(
    exploded: pd.DataFrame,
    differences: pd.Series,
) -> tuple[list[tuple[str, float, float, float]], list[tuple[str, float, float, float]]]:
    """Return significant negative and positive value-difference entries."""
    negative_entries: list[tuple[str, float, float, float]] = []
    positive_entries: list[tuple[str, float, float, float]] = []

    for value in differences.head(N_VALUES_TO_CONSIDER).index:
        estimate, lower_error, upper_error = estimate_and_errors(
            bootstrap_value_difference(exploded, value)
        )
        if not interval_crosses_zero(estimate, lower_error, upper_error):
            negative_entries.append((value, estimate, lower_error, upper_error))

    for value in differences.tail(N_VALUES_TO_CONSIDER).index:
        estimate, lower_error, upper_error = estimate_and_errors(
            bootstrap_value_difference(exploded, value)
        )
        if not interval_crosses_zero(estimate, lower_error, upper_error):
            positive_entries.append((value, estimate, lower_error, upper_error))

    return negative_entries, positive_entries


def changed_verdict_debates(
    dataframe: pd.DataFrame,
    agent_index: int,
) -> pd.DataFrame:
    """Return debates in which one agent changed verdict at least once."""
    verdict_column = AGENT_VERDICT_COLUMNS[agent_index]
    changed = dataframe[verdict_column].map(lambda verdicts: len(set(verdicts)) > 1)
    return dataframe.loc[changed]


def inherited_values(row: pd.Series, agent_index: int) -> set[str]:
    """Return values newly used after appearing in the opponent's first round."""
    agent_values = row[AGENT_VALUE_COLUMNS[agent_index]]
    opponent_values = row[AGENT_VALUE_COLUMNS[1 - agent_index]]
    newly_used = agent_values[-1] - agent_values[0]
    return newly_used.intersection(opponent_values[0])


def inherited_value_proportions(
    dataframe: pd.DataFrame,
    agent_index: int,
) -> pd.Series:
    """Return inherited-value rates among debates where an agent changed verdict."""
    changed_debates = changed_verdict_debates(dataframe, agent_index)
    inherited = changed_debates.apply(
        lambda row: inherited_values(row, agent_index), axis=1
    )
    return inherited.explode().value_counts() / changed_debates.shape[0]


def bootstrap_inherited_value(
    dataframe: pd.DataFrame,
    agent_index: int,
    value: str,
) -> dict[str, float]:
    """Bootstrap an inherited-value rate for one agent."""
    changed_debates = changed_verdict_debates(dataframe, agent_index)

    def statistic(sample: pd.DataFrame) -> float:
        inherited = sample.apply(
            lambda row: inherited_values(row, agent_index), axis=1
        )
        return inherited.explode().value_counts().get(value, 0) / sample.shape[0]

    return bootstrap_statistic_df(
        df=changed_debates,
        statistic_func=statistic,
        n_bootstrap=N_BOOTSTRAP,
        confidence_level=CONFIDENCE_LEVEL,
        random_state=BOOTSTRAP_SEED,
    )


def significant_inherited_entries(
    dataframe: pd.DataFrame,
    inherited_proportions: pd.Series,
    agent_index: int,
    *,
    mirror: bool,
) -> list[tuple[str, float, float, float]]:
    """Return significant inherited-value entries for one agent."""
    entries: list[tuple[str, float, float, float]] = []
    top_values = inherited_proportions.nlargest(
        N_INHERITED_VALUES_TO_CONSIDER
    ).index

    for value in top_values:
        estimate, lower_error, upper_error = estimate_and_errors(
            bootstrap_inherited_value(dataframe, agent_index, value)
        )
        plotted_estimate = -estimate if mirror else estimate
        if not interval_crosses_zero(
            plotted_estimate, lower_error, upper_error
        ):
            entries.append((value, plotted_estimate, lower_error, upper_error))
    return entries


def draw_horizontal_bars(
    axis: Axes,
    entries: list[tuple[str, float, float, float]],
    y_positions: list[int],
    color: str,
) -> None:
    """Draw horizontal bars and asymmetric bootstrap errors."""
    if not entries:
        return
    estimates = [estimate for _, estimate, _, _ in entries]
    errors = np.array(
        [
            [lower for _, _, lower, _ in entries],
            [upper for _, _, _, upper in entries],
        ]
    )
    axis.barh(
        y_positions,
        estimates,
        color=color,
        xerr=errors,
        error_kw={
            "lw": BAR_ERROR_LINE_WIDTH,
            "capsize": BAR_ERROR_CAPSIZE,
            "zorder": 2,
        },
    )


def annotate_value_labels(
    axis: Axes,
    entries: list[tuple[str, float, float, float]],
    y_positions: list[int],
    *,
    on_left: bool,
) -> None:
    """Place value names beside the zero line, opposite their bars."""
    horizontal_alignment = "right" if on_left else "left"
    offset = -VALUE_LABEL_PAD_POINTS if on_left else VALUE_LABEL_PAD_POINTS
    for y_position, (label, _, _, _) in zip(y_positions, entries, strict=True):
        axis.annotate(
            label,
            xy=(0, y_position),
            xycoords="data",
            xytext=(offset, 0),
            textcoords="offset points",
            fontsize=VALUE_LABEL_SIZE,
            ha=horizontal_alignment,
            va="center",
            zorder=3,
        )


def style_mirrored_axis(
    axis: Axes,
    x_limits: tuple[int, int],
    x_ticks: tuple[int, ...],
    x_tick_labels: tuple[str, ...],
    x_label: str,
) -> None:
    """Apply the shared mirrored-bar axis styling."""
    axis.spines["top"].set_visible(False)
    axis.spines["left"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.xaxis.grid(True, linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)
    axis.set_axisbelow(True)
    axis.set_yticks([])
    axis.set_xlim(*x_limits)
    axis.set_xticks(x_ticks, x_tick_labels)
    axis.set_facecolor(AXIS_FACE_COLOR)
    axis.set_xlabel(bold_text(x_label), fontsize=AXIS_LABEL_SIZE)


def add_direction_annotations(
    axis: Axes,
    left_model: str,
    right_model: str,
    action: str,
) -> None:
    """Label which model corresponds to each side of a mirrored axis."""
    axis.annotate(
        bold_text(r"$\uparrow$" + f"\n{action}\nby {left_model}"),
        xy=(DIRECTION_LEFT_X, DIRECTION_LABEL_Y),
        xycoords="axes fraction",
        fontsize=DIRECTION_LABEL_SIZE,
        ha="center",
        va="center",
        zorder=3,
    )
    axis.annotate(
        bold_text(r"$\uparrow$" + f"\n{action}\nby {right_model}"),
        xy=(DIRECTION_RIGHT_X, DIRECTION_LABEL_Y),
        xycoords="axes fraction",
        fontsize=DIRECTION_LABEL_SIZE,
        ha="center",
        va="center",
        zorder=3,
    )


def plot_value_differences(
    axis: Axes,
    exploded: pd.DataFrame,
    agent_names: tuple[str, str],
    row_label: str,
) -> None:
    """Plot significant differences in value occurrence for a model pair."""
    negative_entries, positive_entries = significant_difference_entries(
        exploded, value_differences(exploded)
    )
    negative_positions = list(range(len(negative_entries)))
    positive_positions = list(
        range(len(negative_entries), len(negative_entries) + len(positive_entries))
    )

    draw_horizontal_bars(
        axis,
        negative_entries,
        negative_positions,
        AGENT_COLORS[agent_names[1]],
    )
    draw_horizontal_bars(
        axis,
        positive_entries,
        positive_positions,
        AGENT_COLORS[agent_names[0]],
    )
    annotate_value_labels(
        axis, negative_entries, negative_positions, on_left=False
    )
    annotate_value_labels(
        axis, positive_entries, positive_positions, on_left=True
    )
    style_mirrored_axis(
        axis,
        DIFFERENCE_X_LIMITS,
        DIFFERENCE_X_TICKS,
        DIFFERENCE_X_TICK_LABELS,
        DIFFERENCE_X_LABEL,
    )
    add_direction_annotations(axis, agent_names[1], agent_names[0], "Used more")
    axis.set_ylabel(
        bold_text(row_label),
        fontsize=PAIR_LABEL_SIZE,
        labelpad=PAIR_LABEL_PAD,
        va="center",
        rotation=0,
    )


def plot_inherited_values(
    axis: Axes,
    dataframe: pd.DataFrame,
    agent_names: tuple[str, str],
    *,
    top_row: bool,
) -> None:
    """Plot significant inherited-value rates for a model pair."""
    agent_1_proportions = inherited_value_proportions(dataframe, 0)
    agent_2_proportions = inherited_value_proportions(dataframe, 1)
    left_entries = significant_inherited_entries(
        dataframe, agent_2_proportions, 1, mirror=True
    )
    right_entries = list(
        reversed(
            significant_inherited_entries(
                dataframe, agent_1_proportions, 0, mirror=False
            )
        )
    )
    left_positions = list(range(len(left_entries)))
    right_positions = list(
        range(len(left_entries), len(left_entries) + len(right_entries))
    )

    draw_horizontal_bars(
        axis,
        left_entries,
        left_positions,
        AGENT_COLORS[agent_names[1]],
    )
    draw_horizontal_bars(
        axis,
        right_entries,
        right_positions,
        AGENT_COLORS[agent_names[0]],
    )
    annotate_value_labels(axis, left_entries, left_positions, on_left=False)
    annotate_value_labels(axis, right_entries, right_positions, on_left=True)

    x_limits = TOP_INHERITED_X_LIMITS if top_row else INHERITED_X_LIMITS
    x_ticks = TOP_INHERITED_X_TICKS if top_row else INHERITED_X_TICKS
    x_tick_labels = (
        TOP_INHERITED_X_TICK_LABELS if top_row else INHERITED_X_TICK_LABELS
    )
    style_mirrored_axis(
        axis,
        x_limits,
        x_ticks,
        x_tick_labels,
        INHERITED_X_LABEL,
    )
    add_direction_annotations(axis, agent_names[1], agent_names[0], "Inherited")


def build_figure(dataframes: tuple[pd.DataFrame, ...]) -> plt.Figure:
    """Build the complete six-panel Figure 4 layout."""
    figure, axes = plt.subplots(3, 2, figsize=FIGURE_SIZE, dpi=DPI)
    figure.subplots_adjust(hspace=HSPACE, wspace=WSPACE)

    for row_index, (dataframe, agent_names, row_label) in enumerate(
        zip(dataframes, PAIR_AGENT_NAMES, PAIR_ROW_LABELS, strict=True)
    ):
        plot_value_differences(
            axes[row_index, 0],
            explode_debate_rounds(dataframe),
            agent_names,
            row_label,
        )
        plot_inherited_values(
            axes[row_index, 1],
            dataframe,
            agent_names,
            top_row=row_index == 0,
        )

    apply_subplot_labels(
        [
            axes[0, 0],
            axes[1, 0],
            axes[2, 0],
            axes[0, 1],
            axes[1, 1],
            axes[2, 1],
        ],
        bold=True,
        x=SUBPLOT_LABEL_X,
        y=SUBPLOT_LABEL_Y,
        size=SUBPLOT_LABEL_SIZE,
    )
    return figure


def main() -> None:
    """Load synchronous debate data, build Figure 4, and save it as a PDF."""
    use_latex_style()
    plt.rcParams["axes.prop_cycle"] = cycler(color=COLOR_CYCLE)

    dataframes = tuple(load_debate_data(path) for path in PAIR_DATA_PATHS)
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
