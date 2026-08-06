"""Plot deliberation outcomes for Figure 2."""

import pickle
from collections.abc import Callable
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from mpl_lego.labels import apply_subplot_labels, bold_text
from mpl_lego.style import use_latex_style

from interaction_protocol.utils import bootstrap_statistic_df, change_of_minds


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "analysis"
OUTPUT_PATH = REPO_ROOT / "artifacts" / "figures" / "figure_2.pdf"

SYNCHRONOUS_DATA_PATHS = (
    DATA_DIR / "exp1_sync_h2h.pkl",
    DATA_DIR / "exp5_sync_h2h.pkl",
    DATA_DIR / "exp6_sync_h2h.pkl",
)
ROUND_ROBIN_DATA_PATHS = (
    (DATA_DIR / "exp3b_round_robin_h2h.pkl", DATA_DIR / "exp3a_round_robin_h2h.pkl"),
    (DATA_DIR / "exp2a_round_robin_h2h.pkl", DATA_DIR / "exp2b_round_robin_h2h.pkl"),
    (DATA_DIR / "exp1b_round_robin_h2h.pkl", DATA_DIR / "exp1a_round_robin_h2h.pkl"),
)
MODEL_PAIRS = (("Claude", "GPT"), ("Claude", "Gemini"), ("GPT", "Gemini"))
PAIR_LABELS = tuple(f"{first} vs. {second}" for first, second in MODEL_PAIRS)
PAIR_COLORS = ("#DCDCDC", "#A9A9A9", "#2F4F4F")
COLOR_CYCLE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")

FIGURE_SIZE = (16, 3)
DPI = 300
OUTER_WIDTH_RATIOS = (1, 1)
INNER_WIDTH_RATIOS = (1, 0.75)
OUTER_WSPACE = 0.12
INNER_WSPACE = 0.30
AXIS_FACE_COLOR = "0.98"
EDGE_COLOR = "black"
GRID_AXIS_COLOR = "0.8"

N_BOOTSTRAP = 1_000
CONFIDENCE_LEVEL = 0.95
ROUND_VALUES = (1, 2, 3, 4)
SYNCHRONOUS_BAR_WIDTH = 0.22
ROUND_ROBIN_BAR_WIDTH = 0.18
ROUND_ROBIN_ORDER_GAP = 0.03
ROUND_ROBIN_GROUP_GAP = 0.12
ROUND_ROBIN_CATEGORY_SEPARATION_SCALE = 1.2
REVERSE_ORDER_HATCH = "///"
HORIZONTAL_BAR_HEIGHT = 0.50
ROUND_ROBIN_PAIR_BAR_HEIGHT = 0.36
ROUND_ROBIN_PAIR_BAR_GAP = 0.02
ERROR_CAPSIZE = 1.5
HORIZONTAL_ERROR_CAPSIZE = 1.8

AXIS_LABEL_SIZE = 12
TICK_LABEL_SIZE = 10
LEGEND_SIZE = 9
ROUND_ROBIN_LEGEND_SIZE = 8
BLOCK_TITLE_SIZE = 14
SUBPLOT_LABEL_SIZE = 15
SUBPLOT_LABEL_Y = 1.08
SYNCHRONOUS_TITLE_POSITION = (0.30, 1.04)
ROUND_ROBIN_TITLE_POSITION = (0.73, 1.04)
SAVE_PAD_INCHES = 0.02

PROPORTION_LIMITS = (0, 1)
SYNCHRONOUS_COV_LIMITS = (0, 0.5)
ROUND_ROBIN_COV_LIMITS = (0, 0.6)
ROUND_ROBIN_COV_TICKS = np.arange(0, 0.61, 0.1)
SYNCHRONOUS_COV_Y_LIMITS = (0, 6)
ROUND_ROBIN_COV_Y_LIMITS = (0, 9)


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a deliberation results dataframe from a local pickle file."""
    with path.open("rb") as file:
        return pickle.load(file)


def bootstrap_errors(
    dataframe: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
) -> tuple[float, float]:
    """Return lower and upper bootstrap errors for a dataframe statistic."""
    result = bootstrap_statistic_df(
        df=dataframe,
        statistic_func=statistic,
        n_bootstrap=N_BOOTSTRAP,
        confidence_level=CONFIDENCE_LEVEL,
    )
    return result["lower_err"], result["upper_err"]


def consensus_round_proportion(dataframe: pd.DataFrame, round_value: int) -> float:
    """Return the share reaching consensus in a specified round."""
    reached_in_round = dataframe["n_rounds"].eq(round_value)
    return (reached_in_round & dataframe["final_verdict"].notna()).mean()


def no_consensus_proportion(dataframe: pd.DataFrame) -> float:
    """Return the share of debates that did not reach consensus."""
    return dataframe["final_verdict"].isna().mean()


def change_of_verdict_rate(dataframe: pd.DataFrame, verdict_column: str) -> float:
    """Return the change-of-verdict rate for one agent column."""
    return change_of_minds(dataframe[verdict_column], mean=True)


def style_axis(axis: Axes, grid_axis: str) -> None:
    """Apply the shared background and grid treatment to an axis."""
    axis.set_facecolor(AXIS_FACE_COLOR)
    axis.grid(axis=grid_axis, color=GRID_AXIS_COLOR)
    axis.set_axisbelow(True)
    axis.tick_params(labelsize=TICK_LABEL_SIZE)


def plot_synchronous_rounds(axis: Axes, dataframes: tuple[pd.DataFrame, ...]) -> None:
    """Plot synchronous consensus timing and non-consensus rates."""
    x_positions = np.arange(len(ROUND_VALUES) + 1)

    for index, (dataframe, color, label) in enumerate(
        zip(dataframes, PAIR_COLORS, PAIR_LABELS, strict=True)
    ):
        statistics = [
            lambda frame, round_value=round_value: consensus_round_proportion(
                frame, round_value
            )
            for round_value in ROUND_VALUES
        ]
        statistics.append(no_consensus_proportion)
        heights = [statistic(dataframe) for statistic in statistics]
        errors = np.array(
            [bootstrap_errors(dataframe, statistic) for statistic in statistics]
        ).T

        axis.bar(
            x_positions + (index - 1) * SYNCHRONOUS_BAR_WIDTH,
            heights,
            width=SYNCHRONOUS_BAR_WIDTH,
            color=color,
            edgecolor=EDGE_COLOR,
            label=bold_text(label),
            yerr=errors,
            capsize=ERROR_CAPSIZE,
        )

    axis.set_ylim(*PROPORTION_LIMITS)
    axis.set_xticks(x_positions, ["1", "2", "3", "4", "No\nConsensus"])
    axis.set_xlabel(bold_text("Number of Rounds"), fontsize=AXIS_LABEL_SIZE)
    axis.set_ylabel(bold_text("Proportion of Dilemmas"), fontsize=AXIS_LABEL_SIZE)
    axis.legend(fontsize=LEGEND_SIZE)
    style_axis(axis, "y")


def plot_synchronous_changes(axis: Axes, dataframes: tuple[pd.DataFrame, ...]) -> None:
    """Plot agent change-of-verdict rates for synchronous deliberation."""
    tick_positions: list[float] = []
    tick_labels: list[str] = []

    for index, (dataframe, color, model_pair) in enumerate(
        zip(dataframes, PAIR_COLORS, MODEL_PAIRS, strict=True)
    ):
        positions = np.array([2 * index + 0.6, 2 * index + 1.4])
        columns = ("Agent_1_verdicts", "Agent_2_verdicts")
        rates = [change_of_verdict_rate(dataframe, column) for column in columns]
        errors = np.array(
            [
                bootstrap_errors(
                    dataframe,
                    lambda frame, column=column: change_of_verdict_rate(frame, column),
                )
                for column in columns
            ]
        ).T

        axis.barh(
            positions,
            rates,
            height=HORIZONTAL_BAR_HEIGHT,
            color=color,
            edgecolor=EDGE_COLOR,
            xerr=errors,
            error_kw={"capsize": HORIZONTAL_ERROR_CAPSIZE},
        )
        tick_positions.extend(positions)
        tick_labels.extend(model_pair)

    axis.axhline(2, color=EDGE_COLOR)
    axis.axhline(4, color=EDGE_COLOR)
    axis.set_xlim(*SYNCHRONOUS_COV_LIMITS)
    axis.set_ylim(*SYNCHRONOUS_COV_Y_LIMITS)
    axis.set_yticks(tick_positions, [bold_text(label) for label in tick_labels])
    axis.set_xlabel(bold_text("Change-of-Verdict Rate"), fontsize=AXIS_LABEL_SIZE)
    style_axis(axis, "x")


def round_robin_group_offset(group_index: int, reverse_order: bool) -> float:
    """Return the horizontal offset for a round-robin bar group."""
    within_group = int(reverse_order)
    group_center = (group_index - (len(MODEL_PAIRS) - 1) / 2) * (
        2 * ROUND_ROBIN_BAR_WIDTH
        + ROUND_ROBIN_ORDER_GAP
        + ROUND_ROBIN_GROUP_GAP
    )
    order_offset = (within_group - 0.5) * (
        ROUND_ROBIN_BAR_WIDTH + ROUND_ROBIN_ORDER_GAP
    )
    return group_center + order_offset


def round_robin_category_positions() -> np.ndarray:
    """Return centered positions for the round-robin outcome categories."""
    max_group_span = (len(MODEL_PAIRS) - 1) / 2 * (
        2 * ROUND_ROBIN_BAR_WIDTH
        + ROUND_ROBIN_ORDER_GAP
        + ROUND_ROBIN_GROUP_GAP
    )
    max_order_offset = 0.5 * (ROUND_ROBIN_BAR_WIDTH + ROUND_ROBIN_ORDER_GAP)
    max_offset = max_group_span + max_order_offset
    separation = ROUND_ROBIN_CATEGORY_SEPARATION_SCALE * (
        max_offset + ROUND_ROBIN_BAR_WIDTH / 2
    )
    return np.array([-separation, separation])


def plot_round_robin_rounds(
    axis: Axes,
    dataframe_pairs: tuple[tuple[pd.DataFrame, pd.DataFrame], ...],
) -> None:
    """Plot round-robin first-round consensus and non-consensus rates."""
    category_positions = round_robin_category_positions()

    for group_index, (dataframe_pair, color) in enumerate(
        zip(dataframe_pairs, PAIR_COLORS, strict=True)
    ):
        for reverse_order, dataframe in enumerate(dataframe_pair):
            statistics = (
                lambda frame: frame["n_rounds"].eq(1).mean(),
                no_consensus_proportion,
            )
            heights = [statistic(dataframe) for statistic in statistics]
            errors = np.array(
                [bootstrap_errors(dataframe, statistic) for statistic in statistics]
            ).T
            offset = round_robin_group_offset(group_index, bool(reverse_order))

            axis.bar(
                category_positions + offset,
                heights,
                width=ROUND_ROBIN_BAR_WIDTH,
                color=color,
                edgecolor=EDGE_COLOR,
                hatch=REVERSE_ORDER_HATCH if reverse_order else None,
                yerr=errors,
                capsize=ERROR_CAPSIZE,
            )

    legend_handles = [
        Patch(
            facecolor=color,
            edgecolor=EDGE_COLOR,
            label=bold_text(label),
        )
        for color, label in zip(PAIR_COLORS, PAIR_LABELS, strict=True)
    ]
    legend_handles.append(
        Patch(
            facecolor="white",
            edgecolor=EDGE_COLOR,
            hatch=REVERSE_ORDER_HATCH,
            label=bold_text("Reverse Order"),
        )
    )
    axis.set_ylim(*PROPORTION_LIMITS)
    axis.set_xticks(category_positions, ["1", "No\nConsensus"])
    axis.set_xlabel(bold_text("Number of Rounds"), fontsize=AXIS_LABEL_SIZE)
    axis.set_ylabel(bold_text("Proportion of Dilemmas"), fontsize=AXIS_LABEL_SIZE)
    axis.legend(
        handles=legend_handles,
        fontsize=ROUND_ROBIN_LEGEND_SIZE,
        frameon=True,
        loc="upper right",
    )
    style_axis(axis, "y")


def plot_round_robin_agent_bars(
    axis: Axes,
    dataframe_a: pd.DataFrame,
    dataframe_b: pd.DataFrame,
    model_pair: tuple[str, str],
    color: str,
    pair_index: int,
) -> tuple[list[float], list[str]]:
    """Plot both agents' order-specific bars for one round-robin model pair."""
    pair_bar_offset = ROUND_ROBIN_PAIR_BAR_HEIGHT / 2 + ROUND_ROBIN_PAIR_BAR_GAP
    tick_positions = [3 * pair_index + 1, 3 * pair_index + 2]
    specifications = (
        (tick_positions[0] - pair_bar_offset, dataframe_a, "Agent_1_verdicts", True),
        (tick_positions[0] + pair_bar_offset, dataframe_b, "Agent_2_verdicts", False),
        (tick_positions[1] - pair_bar_offset, dataframe_b, "Agent_1_verdicts", True),
        (tick_positions[1] + pair_bar_offset, dataframe_a, "Agent_2_verdicts", False),
    )

    for position, dataframe, column, reverse_order in specifications:
        rate = change_of_verdict_rate(dataframe, column)
        errors = np.array(
            bootstrap_errors(
                dataframe,
                lambda frame, column=column: change_of_verdict_rate(frame, column),
            )
        ).reshape(2, 1)
        axis.barh(
            position,
            rate,
            height=ROUND_ROBIN_PAIR_BAR_HEIGHT,
            color=color,
            edgecolor=EDGE_COLOR,
            hatch=REVERSE_ORDER_HATCH if reverse_order else None,
        )
        axis.errorbar(
            rate,
            position,
            xerr=errors,
            fmt="none",
            capsize=HORIZONTAL_ERROR_CAPSIZE,
            ecolor=EDGE_COLOR,
        )

    return tick_positions, list(model_pair)


def plot_round_robin_changes(
    axis: Axes,
    dataframe_pairs: tuple[tuple[pd.DataFrame, pd.DataFrame], ...],
) -> None:
    """Plot order-specific change-of-verdict rates for round-robin deliberation."""
    tick_positions: list[float] = []
    tick_labels: list[str] = []

    for pair_index, (dataframe_pair, model_pair, color) in enumerate(
        zip(dataframe_pairs, MODEL_PAIRS, PAIR_COLORS, strict=True)
    ):
        positions, labels = plot_round_robin_agent_bars(
            axis,
            dataframe_pair[0],
            dataframe_pair[1],
            model_pair,
            color,
            pair_index,
        )
        tick_positions.extend(positions)
        tick_labels.extend(labels)

    axis.axhline(3, color=EDGE_COLOR)
    axis.axhline(6, color=EDGE_COLOR)
    axis.set_xlim(*ROUND_ROBIN_COV_LIMITS)
    axis.set_xticks(ROUND_ROBIN_COV_TICKS)
    axis.set_ylim(*ROUND_ROBIN_COV_Y_LIMITS)
    axis.set_yticks(tick_positions, [bold_text(label) for label in tick_labels])
    axis.set_xlabel(bold_text("Change-of-Verdict Rate"), fontsize=AXIS_LABEL_SIZE)
    axis.legend(
        handles=[
            Patch(
                facecolor="white",
                edgecolor=EDGE_COLOR,
                label=bold_text("First"),
            ),
            Patch(
                facecolor="white",
                edgecolor=EDGE_COLOR,
                hatch=REVERSE_ORDER_HATCH,
                label=bold_text("Second"),
            ),
        ],
        fontsize=ROUND_ROBIN_LEGEND_SIZE,
        frameon=True,
        loc="upper right",
    )
    style_axis(axis, "x")


def build_figure(
    synchronous_dataframes: tuple[pd.DataFrame, ...],
    round_robin_dataframes: tuple[tuple[pd.DataFrame, pd.DataFrame], ...],
) -> plt.Figure:
    """Build the complete four-panel Figure 2 layout."""
    figure = plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    outer_grid = gridspec.GridSpec(
        1,
        2,
        figure=figure,
        width_ratios=OUTER_WIDTH_RATIOS,
        wspace=OUTER_WSPACE,
    )
    left_grid = gridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=outer_grid[0],
        width_ratios=INNER_WIDTH_RATIOS,
        wspace=INNER_WSPACE,
    )
    right_grid = gridspec.GridSpecFromSubplotSpec(
        1,
        2,
        subplot_spec=outer_grid[1],
        width_ratios=INNER_WIDTH_RATIOS,
        wspace=INNER_WSPACE,
    )

    synchronous_rounds_axis = figure.add_subplot(left_grid[0, 0])
    synchronous_changes_axis = figure.add_subplot(left_grid[0, 1])
    round_robin_rounds_axis = figure.add_subplot(right_grid[0, 0])
    round_robin_changes_axis = figure.add_subplot(right_grid[0, 1])

    plot_synchronous_rounds(synchronous_rounds_axis, synchronous_dataframes)
    plot_synchronous_changes(synchronous_changes_axis, synchronous_dataframes)
    plot_round_robin_rounds(round_robin_rounds_axis, round_robin_dataframes)
    plot_round_robin_changes(round_robin_changes_axis, round_robin_dataframes)

    apply_subplot_labels(
        [synchronous_rounds_axis, synchronous_changes_axis],
        labels=["a", "b"],
        bold=True,
        y=SUBPLOT_LABEL_Y,
        size=SUBPLOT_LABEL_SIZE,
    )
    apply_subplot_labels(
        [round_robin_rounds_axis, round_robin_changes_axis],
        labels=["c", "d"],
        bold=True,
        y=SUBPLOT_LABEL_Y,
        size=SUBPLOT_LABEL_SIZE,
    )
    figure.text(
        *SYNCHRONOUS_TITLE_POSITION,
        bold_text("Synchronous Deliberation"),
        ha="center",
        va="top",
        fontsize=BLOCK_TITLE_SIZE,
    )
    figure.text(
        *ROUND_ROBIN_TITLE_POSITION,
        bold_text("Round-Robin Deliberation"),
        ha="center",
        va="top",
        fontsize=BLOCK_TITLE_SIZE,
    )
    return figure


def main() -> None:
    """Load the analysis data, build Figure 2, and save it as a PDF."""
    use_latex_style()
    plt.rcParams["axes.prop_cycle"] = cycler(color=COLOR_CYCLE)

    synchronous_dataframes = tuple(
        load_dataframe(path) for path in SYNCHRONOUS_DATA_PATHS
    )
    round_robin_dataframes = tuple(
        (load_dataframe(first_path), load_dataframe(second_path))
        for first_path, second_path in ROUND_ROBIN_DATA_PATHS
    )

    figure = build_figure(synchronous_dataframes, round_robin_dataframes)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_PATH,
        bbox_inches="tight",
        pad_inches=SAVE_PAD_INCHES,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
