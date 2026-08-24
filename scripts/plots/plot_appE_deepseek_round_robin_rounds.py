"""Plot DeepSeek round-robin consensus timing for Appendix E Figure 18."""

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from mpl_lego.labels import bold_text
from mpl_lego.style import use_latex_style

from interaction_protocol.data import load_debate_data
from interaction_protocol.utils import bootstrap_statistic_df


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "experiments"
OUTPUT_PATH = (
    REPO_ROOT / "artifacts" / "figures" / "appE_deepseek_round_robin_rounds.pdf"
)

MODEL_CONFIGS = (
    (
        "Claude 3.7",
        "#6C7A89",
        "#95A5A6",
        DATA_DIR / "round_robin_h2h_cla_vs_deepseek.parquet",
        DATA_DIR / "round_robin_h2h_deepseek_vs_cla.parquet",
    ),
    (
        "Gemini 2.0",
        "#5D8CAE",
        "#2C3E50",
        DATA_DIR / "round_robin_h2h_gem_vs_deepseek.parquet",
        DATA_DIR / "round_robin_h2h_deepseek_vs_gem.parquet",
    ),
    (
        "GPT-4.1",
        "#B96B7F",
        "#7F3A44",
        DATA_DIR / "round_robin_h2h_gpt_vs_deepseek.parquet",
        DATA_DIR / "round_robin_h2h_deepseek_vs_gpt.parquet",
    ),
)
COLOR_CYCLE = (
    "#6C7A89",
    "#95A5A6",
    "#5D8CAE",
    "#2C3E50",
    "#B96B7F",
    "#7F3A44",
)

N_ROUNDS_COLUMN = "n_rounds"
FINAL_VERDICT_COLUMN = "final_verdict"
ROUND_VALUES = (1, 2, 3, 4)
ROUND_TICK_LABELS = ("1", "2", "3", "4", "No\nConsensus")
X_POSITIONS = np.arange(len(ROUND_TICK_LABELS))

FIGURE_SIZE = (9.5, 4.0)
DPI = 300
AXIS_FACE_COLOR = "0.98"
GRID_COLOR = "0.8"
EDGE_COLOR = "black"
BAR_WIDTH = 0.12
ERROR_CAPSIZE = 2.0
N_BOOTSTRAP = 1_000
CONFIDENCE_LEVEL = 0.95
RANDOM_STATE = 42

Y_LIMITS = (0.0, 1.0)
AXIS_LABEL_SIZE = 12
TICK_LABEL_SIZE = 10
LEGEND_SIZE = 8
LEGEND_NCOL = 3
LEGEND_ANCHOR = (0.5, 1.03)
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
    """Return the fraction of debates without a consensus verdict."""
    return float(dataframe[FINAL_VERDICT_COLUMN].isna().mean())


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


def plot_experiment(
    axis: Axes,
    dataframe: pd.DataFrame,
    *,
    offset: float,
    color: str,
) -> None:
    """Plot one model-order condition across all outcome categories."""
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
        X_POSITIONS + offset,
        heights,
        width=BAR_WIDTH,
        color=color,
        edgecolor=EDGE_COLOR,
        yerr=errors,
        capsize=ERROR_CAPSIZE,
    )


def build_figure() -> plt.Figure:
    """Build the round-robin consensus-timing figure."""
    figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    condition_index = 0
    for (
        _,
        comparison_first_color,
        deepseek_first_color,
        comparison_first_path,
        deepseek_first_path,
    ) in MODEL_CONFIGS:
        for path, color in (
            (comparison_first_path, comparison_first_color),
            (deepseek_first_path, deepseek_first_color),
        ):
            centered_index = condition_index - (2 * len(MODEL_CONFIGS) - 1) / 2
            plot_experiment(
                axis,
                load_debate_data(path),
                offset=centered_index * BAR_WIDTH,
                color=color,
            )
            condition_index += 1

    legend_handles = []
    for model, first_color, second_color, _, _ in MODEL_CONFIGS:
        legend_handles.extend(
            [
                Patch(
                    facecolor=first_color,
                    edgecolor=EDGE_COLOR,
                    label=bold_text(f"{model} ({model.split()[0]} first)"),
                ),
                Patch(
                    facecolor=second_color,
                    edgecolor=EDGE_COLOR,
                    label=bold_text(f"{model} (DeepSeek first)"),
                ),
            ]
        )
    axis.legend(
        handles=legend_handles,
        fontsize=LEGEND_SIZE,
        ncol=LEGEND_NCOL,
        loc="lower center",
        bbox_to_anchor=LEGEND_ANCHOR,
    )
    axis.set_ylim(*Y_LIMITS)
    axis.set_xticks(X_POSITIONS, ROUND_TICK_LABELS)
    axis.set_xlabel(bold_text("Number of Rounds"), fontsize=AXIS_LABEL_SIZE)
    axis.set_ylabel(bold_text("Proportion of Dilemmas"), fontsize=AXIS_LABEL_SIZE)
    axis.set_facecolor(AXIS_FACE_COLOR)
    axis.grid(axis="y", color=GRID_COLOR)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    return figure


def main() -> None:
    """Build and save Appendix E Figure 18."""
    use_latex_style()
    plt.rcParams["axes.prop_cycle"] = cycler(color=COLOR_CYCLE)

    figure = build_figure()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_PATH,
        bbox_inches="tight",
        pad_inches=SAVE_PAD_INCHES,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
