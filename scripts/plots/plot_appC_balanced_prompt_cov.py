"""Plot change-of-verdict rates for the balanced system-prompt experiment."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from mpl_lego.labels import add_significance_bracket_inplot, bold_text
from mpl_lego.style import use_latex_style
from statsmodels.stats.proportion import proportions_ztest

from interaction_protocol.data import load_debate_data
from interaction_protocol.plotting import significance_label
from interaction_protocol.utils import bootstrap_statistic_df, change_of_minds


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "experiments"
OUTPUT_PATH = REPO_ROOT / "artifacts" / "figures" / "appC_balanced_prompt_cov.pdf"

ORIGINAL_DATA_PATHS = (
    DATA_DIR / "sync_h2h_cla_vs_gpt.parquet",
    DATA_DIR / "sync_h2h_cla_vs_gem.parquet",
    DATA_DIR / "sync_h2h_gpt_vs_gem.parquet",
)
BALANCED_DATA_PATHS = (
    DATA_DIR / "sync_h2h_balanced_cla_vs_gpt.parquet",
    DATA_DIR / "sync_h2h_balanced_cla_vs_gem.parquet",
    DATA_DIR / "sync_h2h_balanced_gpt_vs_gem.parquet",
)
MODEL_PAIRS = (
    ("Claude", "GPT"),
    ("Claude", "Gemini"),
    ("GPT", "Gemini"),
)
PANEL_TITLES = ("Claude vs. GPT", "Claude vs. Gemini", "Gemini vs. GPT")
VERDICT_COLUMNS = ("Agent_1_verdicts", "Agent_2_verdicts")
CONDITION_LABELS = ("Original\nPrompt", "Balanced\nPrompt")

MODEL_COLORS = {
    "Claude": "#0072B2",
    "GPT": "#D55E00",
    "Gemini": "#009E73",
}
COLOR_CYCLE = tuple(MODEL_COLORS.values())

FIGURE_SIZE = (6, 3)
DPI = 300
SUBPLOT_WSPACE = 0.25
AXIS_FACE_COLOR = "0.96"
GRID_COLOR = "0.8"
GRID_LINE_WIDTH = 0.8
X_POSITIONS = np.array([0.0, 1.0])
X_OFFSETS = ((0.0, 0.0), (-0.05, 0.05), (0.0, 0.0))
X_LIMITS = (-0.25, 1.25)
Y_LIMITS = (0.0, 0.5)

N_BOOTSTRAP = 1_000
CONFIDENCE_LEVEL = 0.95
RANDOM_STATE = 42
LINE_WIDTH = 1.5
MARKER_SIZE = 5
MARKER_EDGE_COLOR = "black"
MARKER_EDGE_WIDTH = 0.4
ERROR_CAPSIZE = 3

BRACKET_LAYOUTS = (
    ((0.365, "up"), (0.225, "up")),
    ((0.300, "down"), (0.250, "down")),
    ((0.140, "up"), (0.370, "down")),
)
BRACKET_HEIGHT = 0.005
BRACKET_TEXT_OFFSETS = {"up": -0.010, "down": 0.009}
BRACKET_FONT_SIZE = 10

TITLE_SIZE = 11
AXIS_LABEL_SIZE = 11
TICK_LABEL_SIZE = 9
LEGEND_SIZE = 9
LEGEND_LOCATION = "center left"
LEGEND_ANCHOR = (1.05, 0.5)
SAVE_PAD_INCHES = 0.02


def bootstrap_change_rate(
    dataframe: pd.DataFrame,
    verdict_column: str,
) -> dict[str, float]:
    """Return a bootstrapped change-of-verdict estimate and errors."""
    return bootstrap_statistic_df(
        df=dataframe,
        statistic_func=lambda frame: change_of_minds(
            frame[verdict_column], mean=True
        ),
        n_bootstrap=N_BOOTSTRAP,
        confidence_level=CONFIDENCE_LEVEL,
        random_state=RANDOM_STATE,
    )


def change_rate_p_value(
    original: pd.DataFrame,
    balanced: pd.DataFrame,
    verdict_column: str,
) -> float:
    """Compare original and balanced change rates with a two-sided z-test."""
    counts = (
        change_of_minds(original[verdict_column]),
        change_of_minds(balanced[verdict_column]),
    )
    sample_sizes = (len(original), len(balanced))
    _, p_value = proportions_ztest(counts, sample_sizes)
    return float(p_value)


def plot_model_change_rates(
    axis: Axes,
    original: pd.DataFrame,
    balanced: pd.DataFrame,
    *,
    model: str,
    verdict_column: str,
    x_offset: float,
    bracket_y: float,
    bracket_direction: str,
) -> None:
    """Plot and compare one model's original and balanced change rates."""
    estimates = (
        bootstrap_change_rate(original, verdict_column),
        bootstrap_change_rate(balanced, verdict_column),
    )
    y_values = [estimate["original"] for estimate in estimates]
    y_errors = np.array(
        [
            [estimate["lower_err"] for estimate in estimates],
            [estimate["upper_err"] for estimate in estimates],
        ]
    )
    x_values = X_POSITIONS + x_offset

    axis.errorbar(
        x_values,
        y_values,
        yerr=y_errors,
        color=MODEL_COLORS[model],
        linewidth=LINE_WIDTH,
        marker="o",
        markersize=MARKER_SIZE,
        markeredgecolor=MARKER_EDGE_COLOR,
        markeredgewidth=MARKER_EDGE_WIDTH,
        capsize=ERROR_CAPSIZE,
    )

    label = significance_label(
        change_rate_p_value(original, balanced, verdict_column)
    )
    add_significance_bracket_inplot(
        ax=axis,
        x1=x_values[0],
        x2=x_values[1],
        y=bracket_y,
        direction=bracket_direction,
        color=MODEL_COLORS[model],
        h=BRACKET_HEIGHT,
        text_offset=BRACKET_TEXT_OFFSETS[bracket_direction],
        fontsize=BRACKET_FONT_SIZE,
        label=bold_text(label),
    )


def style_axis(axis: Axes, title: str) -> None:
    """Apply the shared system-prompt comparison axis styling."""
    axis.set_title(bold_text(title), fontsize=TITLE_SIZE)
    axis.set_xlim(*X_LIMITS)
    axis.set_ylim(*Y_LIMITS)
    axis.set_xticks(X_POSITIONS, [bold_text(label) for label in CONDITION_LABELS])
    axis.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    axis.grid(axis="y", color=GRID_COLOR, linewidth=GRID_LINE_WIDTH)
    axis.set_axisbelow(True)
    axis.set_facecolor(AXIS_FACE_COLOR)


def build_figure(
    originals: tuple[pd.DataFrame, ...],
    balanced: tuple[pd.DataFrame, ...],
) -> plt.Figure:
    """Build the three-panel balanced system-prompt comparison figure."""
    figure, axes = plt.subplots(
        1,
        3,
        figsize=FIGURE_SIZE,
        dpi=DPI,
        sharey=True,
    )
    figure.subplots_adjust(wspace=SUBPLOT_WSPACE)

    panel_data = zip(
        axes,
        originals,
        balanced,
        MODEL_PAIRS,
        PANEL_TITLES,
        X_OFFSETS,
        BRACKET_LAYOUTS,
        strict=True,
    )
    for panel_index, (
        axis,
        original,
        balanced_condition,
        models,
        title,
        offsets,
        bracket_layouts,
    ) in enumerate(panel_data):
        model_data = zip(
            models,
            VERDICT_COLUMNS,
            offsets,
            bracket_layouts,
            strict=True,
        )
        for model, verdict_column, x_offset, bracket_layout in model_data:
            bracket_y, bracket_direction = bracket_layout
            plot_model_change_rates(
                axis,
                original,
                balanced_condition,
                model=model,
                verdict_column=verdict_column,
                x_offset=x_offset,
                bracket_y=bracket_y,
                bracket_direction=bracket_direction,
            )
        style_axis(axis, title)
        if panel_index == 0:
            axis.set_ylabel(
                bold_text("Change-of-Verdict Rate"),
                fontsize=AXIS_LABEL_SIZE,
            )

    legend_handles = [
        Line2D(
            [],
            [],
            color=color,
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            markeredgecolor=MARKER_EDGE_COLOR,
            markeredgewidth=MARKER_EDGE_WIDTH,
            label=bold_text(model),
        )
        for model, color in MODEL_COLORS.items()
    ]
    axes[-1].legend(
        handles=legend_handles,
        loc=LEGEND_LOCATION,
        bbox_to_anchor=LEGEND_ANCHOR,
        fontsize=LEGEND_SIZE,
        handlelength=1.0,
        handletextpad=0.3,
        borderaxespad=0.2,
        labelspacing=0.5,
    )
    return figure


def main() -> None:
    """Load debate data, build the balanced comparison, and save it."""
    use_latex_style()
    plt.rcParams["axes.prop_cycle"] = cycler(color=COLOR_CYCLE)

    originals = tuple(load_debate_data(path) for path in ORIGINAL_DATA_PATHS)
    balanced = tuple(load_debate_data(path) for path in BALANCED_DATA_PATHS)
    figure = build_figure(originals, balanced)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_PATH,
        bbox_inches="tight",
        pad_inches=SAVE_PAD_INCHES,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
