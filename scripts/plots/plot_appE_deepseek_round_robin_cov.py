"""Plot corrected DeepSeek round-robin change rates for Figure 19."""

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
from interaction_protocol.utils import bootstrap_statistic_df, change_of_minds


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "analysis"
OUTPUT_PATH = (
    REPO_ROOT / "artifacts" / "figures" / "appE_deepseek_round_robin_cov.pdf"
)

MODEL_CONFIGS = (
    (
        "Claude 3.7",
        "#6C7A89",
        DATA_DIR / "round_robin_h2h_cla_vs_deepseek.parquet",
        DATA_DIR / "round_robin_h2h_deepseek_vs_cla.parquet",
    ),
    (
        "Gemini 2.0",
        "#5D8CAE",
        DATA_DIR / "round_robin_h2h_gem_vs_deepseek.parquet",
        DATA_DIR / "round_robin_h2h_deepseek_vs_gem.parquet",
    ),
    (
        "GPT-4.1",
        "#B96B7F",
        DATA_DIR / "round_robin_h2h_gpt_vs_deepseek.parquet",
        DATA_DIR / "round_robin_h2h_deepseek_vs_gpt.parquet",
    ),
)
COLOR_CYCLE = ("#6C7A89", "#5D8CAE", "#B96B7F")
COMPARISON_MODEL_HATCH = None
DEEPSEEK_HATCH = "///"

AGENT_1_VERDICTS = "Agent_1_verdicts"
AGENT_2_VERDICTS = "Agent_2_verdicts"
FIGURE_SIZE = (11.0, 3.5)
DPI = 300
AXIS_FACE_COLOR = "0.98"
GRID_COLOR = "0.8"
EDGE_COLOR = "black"
GRID_ALPHA = 0.6
BAR_WIDTH = 0.36
BAR_LABEL_OFFSET = 0.012
ERROR_CAPSIZE = 2.0
ERROR_LINE_WIDTH = 1.2
N_BOOTSTRAP = 1_000
CONFIDENCE_LEVEL = 0.95
RANDOM_STATE = 42

Y_LIMITS = (0.0, 0.60)
Y_TICKS = np.arange(0.0, 0.61, 0.1)
AXIS_LABEL_SIZE = 12
TITLE_SIZE = 12
TICK_LABEL_SIZE = 10
BAR_LABEL_SIZE = 9
LEGEND_SIZE = 9
LEGEND_LOCATION = "upper right"
SAVE_PAD_INCHES = 0.02


def change_rate(dataframe: pd.DataFrame, column: str) -> float:
    """Return the change-of-verdict rate for one agent column."""
    return float(change_of_minds(dataframe[column], mean=True))


def bootstrap_errors(dataframe: pd.DataFrame, column: str) -> tuple[float, float]:
    """Return 95% bootstrap errors for one change-of-verdict rate."""
    result = bootstrap_statistic_df(
        df=dataframe,
        statistic_func=lambda sample: change_rate(sample, column),
        n_bootstrap=N_BOOTSTRAP,
        confidence_level=CONFIDENCE_LEVEL,
        random_state=RANDOM_STATE,
    )
    return result["lower_err"], result["upper_err"]


def plot_order_panel(axis: Axes, *, deepseek_first: bool) -> None:
    """Plot change rates for one round-robin speaking-order condition."""
    positions = np.arange(len(MODEL_CONFIGS))
    for index, (model, color, comparison_first_path, deepseek_first_path) in enumerate(
        MODEL_CONFIGS
    ):
        path = deepseek_first_path if deepseek_first else comparison_first_path
        dataframe = load_debate_data(path)
        if deepseek_first:
            deepseek_column = AGENT_1_VERDICTS
            comparison_column = AGENT_2_VERDICTS
        else:
            comparison_column = AGENT_1_VERDICTS
            deepseek_column = AGENT_2_VERDICTS

        comparison_rate = change_rate(dataframe, comparison_column)
        deepseek_rate = change_rate(dataframe, deepseek_column)
        comparison_errors = bootstrap_errors(dataframe, comparison_column)
        deepseek_errors = bootstrap_errors(dataframe, deepseek_column)
        comparison_bar = axis.bar(
            positions[index] - BAR_WIDTH / 2,
            comparison_rate,
            width=BAR_WIDTH,
            color=color,
            edgecolor=EDGE_COLOR,
            hatch=COMPARISON_MODEL_HATCH,
            yerr=np.array(comparison_errors).reshape(2, 1),
            error_kw={
                "capsize": ERROR_CAPSIZE,
                "elinewidth": ERROR_LINE_WIDTH,
                "capthick": ERROR_LINE_WIDTH,
            },
        )[0]
        deepseek_bar = axis.bar(
            positions[index] + BAR_WIDTH / 2,
            deepseek_rate,
            width=BAR_WIDTH,
            color=color,
            edgecolor=EDGE_COLOR,
            hatch=DEEPSEEK_HATCH,
            yerr=np.array(deepseek_errors).reshape(2, 1),
            error_kw={
                "capsize": ERROR_CAPSIZE,
                "elinewidth": ERROR_LINE_WIDTH,
                "capthick": ERROR_LINE_WIDTH,
            },
        )[0]

        for bar, errors in (
            (comparison_bar, comparison_errors),
            (deepseek_bar, deepseek_errors),
        ):
            height = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                height + errors[1] + BAR_LABEL_OFFSET,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=BAR_LABEL_SIZE,
            )

    axis.set_xticks(
        positions,
        [bold_text(model) for model, _, _, _ in MODEL_CONFIGS],
    )
    axis.set_ylim(*Y_LIMITS)
    axis.set_yticks(Y_TICKS)
    axis.set_title(
        bold_text("DeepSeek First" if deepseek_first else "DeepSeek Second"),
        fontsize=TITLE_SIZE,
    )
    axis.set_facecolor(AXIS_FACE_COLOR)
    axis.grid(axis="y", color=GRID_COLOR, alpha=GRID_ALPHA)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)


def build_figure() -> plt.Figure:
    """Build the corrected two-panel change-of-verdict figure."""
    figure, axes = plt.subplots(
        1,
        2,
        figsize=FIGURE_SIZE,
        dpi=DPI,
        sharey=True,
    )
    plot_order_panel(axes[0], deepseek_first=False)
    plot_order_panel(axes[1], deepseek_first=True)
    axes[0].set_ylabel(
        bold_text("Change-of-Verdict Rate"),
        fontsize=AXIS_LABEL_SIZE,
    )
    legend_handles = [
        Patch(
            facecolor="white",
            edgecolor=EDGE_COLOR,
            label=bold_text("Comparison model"),
        ),
        Patch(
            facecolor="white",
            edgecolor=EDGE_COLOR,
            hatch=DEEPSEEK_HATCH,
            label=bold_text("DeepSeek"),
        ),
    ]
    axes[0].legend(
        handles=legend_handles,
        fontsize=LEGEND_SIZE,
        frameon=False,
        loc=LEGEND_LOCATION,
    )
    figure.tight_layout()
    return figure


def main() -> None:
    """Build and save corrected Appendix E Figure 19."""
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
