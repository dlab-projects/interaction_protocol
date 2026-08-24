"""Plot synchronous DeepSeek debate outcomes for Appendix E Figure 17."""

from collections.abc import Callable
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.axes import Axes
from mpl_lego.labels import apply_subplot_labels, bold_text
from mpl_lego.style import use_latex_style

from interaction_protocol.data import load_debate_data
from interaction_protocol.utils import bootstrap_statistic_df, change_of_minds


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "experiments"
OUTPUT_PATH = REPO_ROOT / "artifacts" / "figures" / "appE_deepseek_sync.pdf"

DATA_PATHS = (
    DATA_DIR / "sync_h2h_gpt_vs_deepseek.parquet",
    DATA_DIR / "sync_h2h_cla_vs_deepseek.parquet",
    DATA_DIR / "sync_h2h_gem_vs_deepseek.parquet",
)
COMPARISON_MODELS = ("GPT-4.1", "Claude 3.7 Sonnet", "Gemini 2.0 Flash")
PAIR_LABELS = tuple(f"DeepSeek vs. {model}" for model in COMPARISON_MODELS)
PAIR_COLORS = ("#DCDCDC", "#A9A9A9", "#2F4F4F")
COLOR_CYCLE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")

N_ROUNDS_COLUMN = "n_rounds"
FINAL_VERDICT_COLUMN = "final_verdict"
COMPARISON_VERDICT_COLUMN = "Agent_1_verdicts"
DEEPSEEK_VERDICT_COLUMN = "Agent_2_verdicts"
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
ERROR_CAPSIZE = 1.8

ROUND_Y_LIMITS = (0.0, 1.0)
CHANGE_X_LIMITS = (0.0, 0.5)
ROUND_X_POSITIONS = np.arange(len(ROUND_VALUES) + 1)
AXIS_LABEL_SIZE = 12
TICK_LABEL_SIZE = 10
LEGEND_SIZE = 9
SUBPLOT_LABEL_SIZE = 13
SUBPLOT_LABEL_Y = 1.08
SAVE_PAD_INCHES = 0.02


def consensus_round_proportion(dataframe: pd.DataFrame, round_value: int) -> float:
    """Return the fraction reaching consensus in a specified round."""
    return float(
        (
            dataframe[N_ROUNDS_COLUMN].eq(round_value)
            & dataframe[FINAL_VERDICT_COLUMN].notna()
        ).mean()
    )


def no_consensus_proportion(dataframe: pd.DataFrame) -> float:
    """Return the fraction of debates that did not reach consensus."""
    return float(dataframe[FINAL_VERDICT_COLUMN].isna().mean())


def change_of_verdict_rate(dataframe: pd.DataFrame, verdict_column: str) -> float:
    """Return one agent's change-of-verdict rate."""
    return float(change_of_minds(dataframe[verdict_column], mean=True))


def bootstrap_errors(
    dataframe: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
) -> tuple[float, float]:
    """Return lower and upper bootstrap errors for a statistic."""
    result = bootstrap_statistic_df(
        df=dataframe,
        statistic_func=statistic,
        n_bootstrap=N_BOOTSTRAP,
        confidence_level=CONFIDENCE_LEVEL,
        random_state=RANDOM_STATE,
    )
    return result["lower_err"], result["upper_err"]


def style_axis(axis: Axes, grid_axis: str) -> None:
    """Apply the shared Figure 17 axis styling."""
    axis.set_facecolor(AXIS_FACE_COLOR)
    axis.grid(axis=grid_axis, color=GRID_COLOR)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)


def plot_round_outcomes(axis: Axes, dataframes: tuple[pd.DataFrame, ...]) -> None:
    """Plot consensus timing and no-consensus rates by model pair."""
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
            ROUND_X_POSITIONS + (index - 1) * ROUND_BAR_WIDTH,
            heights,
            width=ROUND_BAR_WIDTH,
            color=color,
            edgecolor=EDGE_COLOR,
            label=bold_text(label),
            yerr=errors,
            capsize=ERROR_CAPSIZE,
        )

    axis.set_ylim(*ROUND_Y_LIMITS)
    axis.set_xticks(ROUND_X_POSITIONS, ROUND_TICK_LABELS)
    axis.set_xlabel(bold_text("Number of Rounds"), fontsize=AXIS_LABEL_SIZE)
    axis.set_ylabel(bold_text("Proportion of Dilemmas"), fontsize=AXIS_LABEL_SIZE)
    axis.legend(fontsize=LEGEND_SIZE)
    style_axis(axis, "y")


def plot_change_rates(axis: Axes, dataframes: tuple[pd.DataFrame, ...]) -> None:
    """Plot DeepSeek and comparison-model change-of-verdict rates."""
    tick_positions: list[float] = []
    tick_labels: list[str] = []

    for index, (dataframe, color, comparison_model) in enumerate(
        zip(dataframes, PAIR_COLORS, COMPARISON_MODELS, strict=True)
    ):
        center = index * CHANGE_GROUP_SPACING
        positions = np.array(
            [
                center + CHANGE_WITHIN_GROUP_OFFSET,
                center - CHANGE_WITHIN_GROUP_OFFSET,
            ]
        )
        columns = (COMPARISON_VERDICT_COLUMN, DEEPSEEK_VERDICT_COLUMN)
        rates = [
            change_of_verdict_rate(dataframe, verdict_column)
            for verdict_column in columns
        ]
        errors = np.array(
            [
                bootstrap_errors(
                    dataframe,
                    lambda frame, verdict_column=verdict_column: (
                        change_of_verdict_rate(frame, verdict_column)
                    ),
                )
                for verdict_column in columns
            ]
        ).T

        axis.barh(
            positions,
            rates,
            height=CHANGE_BAR_HEIGHT,
            color=color,
            edgecolor=EDGE_COLOR,
            xerr=errors,
            error_kw={"capsize": ERROR_CAPSIZE},
        )
        tick_positions.extend(positions)
        tick_labels.extend((comparison_model, "DeepSeek"))

    sorted_ticks = sorted(zip(tick_positions, tick_labels, strict=True))
    axis.set_xlim(*CHANGE_X_LIMITS)
    axis.set_ylim(
        -0.7,
        CHANGE_GROUP_SPACING * (len(dataframes) - 1) + 0.7,
    )
    axis.set_yticks(
        [position for position, _ in sorted_ticks],
        [bold_text(label) for _, label in sorted_ticks],
    )
    axis.set_xlabel(bold_text("Change-of-Verdict Rate"), fontsize=AXIS_LABEL_SIZE)
    style_axis(axis, "x")


def build_figure(dataframes: tuple[pd.DataFrame, ...]) -> plt.Figure:
    """Build the two-panel synchronous DeepSeek figure."""
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
    plot_round_outcomes(round_axis, dataframes)
    plot_change_rates(change_axis, dataframes)
    apply_subplot_labels(
        [round_axis, change_axis],
        y=SUBPLOT_LABEL_Y,
        size=SUBPLOT_LABEL_SIZE,
        bold=True,
    )
    return figure


def main() -> None:
    """Load DeepSeek data, build Figure 17, and save it."""
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
