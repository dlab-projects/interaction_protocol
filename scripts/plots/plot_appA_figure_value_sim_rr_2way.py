"""Plot value similarity in head-to-head round-robin debate (Appendix Figure 6)."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from matplotlib.axes import Axes
from mpl_lego.labels import (
    add_significance_bracket_inplot,
    apply_subplot_labels,
    bold_text,
)
from mpl_lego.style import use_latex_style

from interaction_protocol.data import load_debate_data
from interaction_protocol.plotting import (
    SimilarityComparison,
    compare_similarity_samples,
    comparison_y_errors,
    comparison_y_values,
    pairwise_jaccard,
    significance_label,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "analysis"
OUTPUT_PATH = (
    REPO_ROOT / "artifacts" / "figures" / "appA_figure_value_sim_rr_2way.pdf"
)

DATA_PATHS = (
    DATA_DIR / "round_robin_h2h_gem_vs_gpt.parquet",
    DATA_DIR / "round_robin_h2h_cla_vs_gem.parquet",
    DATA_DIR / "round_robin_h2h_gpt_vs_cla.parquet",
    DATA_DIR / "round_robin_h2h_gpt_vs_gem.parquet",
    DATA_DIR / "round_robin_h2h_gem_vs_cla.parquet",
    DATA_DIR / "round_robin_h2h_cla_vs_gpt.parquet",
)
PANEL_TITLES = (
    "Gemini\nvs.\nGPT",
    "Claude\nvs.\nGemini",
    "GPT\nvs.\nClaude",
    "GPT\nvs.\nGemini",
    "Gemini\nvs.\nClaude",
    "Claude\nvs.\nGPT",
)

AGENT_VERDICT_COLUMNS = ("Agent_1_verdicts", "Agent_2_verdicts")
AGENT_VALUE_COLUMNS = ("Agent_1_values", "Agent_2_values")
ROUND_COLUMNS = AGENT_VERDICT_COLUMNS + AGENT_VALUE_COLUMNS

FIGURE_SIZE = (4, 5)
DPI = 300
SUBPLOT_HSPACE = 0.50
SUBPLOT_WSPACE = 0.25
AXIS_FACE_COLOR = "0.96"
GRID_COLOR = "0.8"
Y_LIMITS = (0, 0.65)
Y_TICKS = (0, 0.2, 0.4, 0.6)
X_LIMITS = (-0.5, 1.5)
X_POSITIONS = (0, 1)

N_BOOTSTRAP = 1_000
CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_SEED = 2_332

MARKER_SIZE = 3
ERROR_CAPSIZE = 3
POINT_COLOR = "black"
COLOR_CYCLE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
TICK_LABEL_SIZE = 8
TICK_LABEL_ROTATION = 18
TITLE_SIZE = 10
Y_LABEL_SIZE = 9
SUBPLOT_LABEL_SIZE = 12
SUBPLOT_LABEL_X = -0.05
SUBPLOT_LABEL_Y = 1.10
SIGNIFICANCE_Y = 0.55
BRACKET_HEIGHT = 0.01
SAVE_PAD_INCHES = 0.02


def analyze_pair(dataframe: pd.DataFrame) -> SimilarityComparison:
    """Compare message-level value similarity during agreement and disagreement."""
    exploded = dataframe.explode(list(ROUND_COLUMNS)).reset_index(drop=True)
    similarities = pairwise_jaccard(
        exploded[AGENT_VALUE_COLUMNS[0]],
        exploded[AGENT_VALUE_COLUMNS[1]],
    )
    agreement = exploded[AGENT_VERDICT_COLUMNS[0]].eq(
        exploded[AGENT_VERDICT_COLUMNS[1]]
    )
    return compare_similarity_samples(
        similarities.loc[agreement],
        similarities.loc[~agreement],
        n_bootstrap=N_BOOTSTRAP,
        confidence_level=CONFIDENCE_LEVEL,
        random_state=BOOTSTRAP_SEED,
    )


def style_axis(axis: Axes) -> None:
    """Apply the shared Appendix Figure 6 axis styling."""
    axis.set_ylim(*Y_LIMITS)
    axis.set_xlim(*X_LIMITS)
    axis.set_xticks(
        X_POSITIONS,
        [bold_text("Agreement"), bold_text("Disagreement")],
        ha="right",
        rotation=TICK_LABEL_ROTATION,
        fontsize=TICK_LABEL_SIZE,
    )
    axis.set_yticks(Y_TICKS)
    axis.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    axis.grid(axis="y", color=GRID_COLOR)
    axis.set_axisbelow(True)
    axis.set_facecolor(AXIS_FACE_COLOR)


def plot_comparison(axis: Axes, comparison: SimilarityComparison) -> None:
    """Plot one agreement-versus-disagreement comparison."""
    axis.errorbar(
        X_POSITIONS,
        comparison_y_values(comparison),
        yerr=comparison_y_errors(comparison),
        fmt="o",
        markersize=MARKER_SIZE,
        capsize=ERROR_CAPSIZE,
        color=POINT_COLOR,
    )
    style_axis(axis)
    add_significance_bracket_inplot(
        ax=axis,
        x1=0,
        x2=1,
        y=SIGNIFICANCE_Y,
        h=BRACKET_HEIGHT,
        label=bold_text(significance_label(comparison.p_value)),
    )


def build_figure(
    comparisons: tuple[SimilarityComparison, ...],
) -> plt.Figure:
    """Build the six-panel Appendix Figure 6 layout."""
    figure, axes = plt.subplots(
        2,
        3,
        figsize=FIGURE_SIZE,
        dpi=DPI,
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(hspace=SUBPLOT_HSPACE, wspace=SUBPLOT_WSPACE)

    for axis, comparison, title in zip(
        axes.ravel(), comparisons, PANEL_TITLES, strict=True
    ):
        plot_comparison(axis, comparison)
        axis.set_title(bold_text(title), fontsize=TITLE_SIZE)

    for axis in axes[:, 0]:
        axis.set_ylabel(bold_text("Value Similarity"), fontsize=Y_LABEL_SIZE)

    apply_subplot_labels(
        axes.ravel(),
        size=SUBPLOT_LABEL_SIZE,
        x=SUBPLOT_LABEL_X,
        y=SUBPLOT_LABEL_Y,
        bold=True,
    )
    return figure


def main() -> None:
    """Load round-robin data, build Appendix Figure 6, and save it as a PDF."""
    use_latex_style()
    plt.rcParams["axes.prop_cycle"] = cycler(color=COLOR_CYCLE)

    dataframes = tuple(load_debate_data(path) for path in DATA_PATHS)
    comparisons = tuple(analyze_pair(dataframe) for dataframe in dataframes)
    figure = build_figure(comparisons)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_PATH,
        bbox_inches="tight",
        pad_inches=SAVE_PAD_INCHES,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
