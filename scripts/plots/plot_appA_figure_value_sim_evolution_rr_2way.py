"""Plot value-similarity evolution in round-robin debate (Appendix Figure 7)."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
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
    significance_label,
)
from interaction_protocol.utils import jaccard


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "analysis"
OUTPUT_PATH = (
    REPO_ROOT
    / "artifacts"
    / "figures"
    / "appA_figure_value_sim_evolution_rr_2way.pdf"
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

N_ROUNDS_COLUMN = "n_rounds"
FINAL_VERDICT_COLUMN = "final_verdict"
AGENT_VALUE_COLUMNS = ("Agent_1_values", "Agent_2_values")
ROUND_1_SIMILARITY_COLUMN = "round_1_value_similarity"
LAST_ROUND_SIMILARITY_COLUMN = "last_round_value_similarity"

FIGURE_SIZE = (4, 5)
DPI = 300
SUBPLOT_HSPACE = 0.40
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
CONSENSUS_COLOR = "black"
NO_CONSENSUS_COLOR = "#708090"
TRAJECTORY_SHIFT = 0.10
COLOR_CYCLE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
TICK_LABEL_SIZE = 8
TICK_LABEL_ROTATION = 18
TITLE_SIZE = 10
Y_LABEL_SIZE = 9
LEGEND_SIZE = 7.5
LEGEND_LOCATION = "center"
LEGEND_BBOX = (1.10, 0.50)
SUBPLOT_LABEL_SIZE = 12
SUBPLOT_LABEL_X = -0.05
SUBPLOT_LABEL_Y = 1.10
CONSENSUS_BRACKET_Y = (0.50, 0.40, 0.50, 0.42, 0.50, 0.56)
NO_CONSENSUS_BRACKET_Y = (0.15, 0.20, 0.25, 0.12, 0.20, 0.12)
BRACKET_HEIGHT = 0.01
NO_CONSENSUS_TEXT_OFFSET = 0.02
SIGNIFICANCE_SIZE = 10
SAVE_PAD_INCHES = 0.02


@dataclass(frozen=True)
class EvolutionAnalysis:
    """Consensus and no-consensus trajectory comparisons for one debate order."""

    consensus: SimilarityComparison
    no_consensus: SimilarityComparison


def add_trajectory_similarities(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add first- and last-round similarity columns to multi-round debates."""
    trajectories = dataframe.loc[dataframe[N_ROUNDS_COLUMN] > 1].copy()
    trajectories[ROUND_1_SIMILARITY_COLUMN] = [
        jaccard(agent_1[0], agent_2[0])
        for agent_1, agent_2 in zip(
            trajectories[AGENT_VALUE_COLUMNS[0]],
            trajectories[AGENT_VALUE_COLUMNS[1]],
            strict=True,
        )
    ]
    trajectories[LAST_ROUND_SIMILARITY_COLUMN] = [
        jaccard(agent_1[-1], agent_2[-1])
        for agent_1, agent_2 in zip(
            trajectories[AGENT_VALUE_COLUMNS[0]],
            trajectories[AGENT_VALUE_COLUMNS[1]],
            strict=True,
        )
    ]
    return trajectories


def analyze_trajectory(dataframe: pd.DataFrame) -> SimilarityComparison:
    """Compare Round 1 and last-round similarity within a debate subset."""
    return compare_similarity_samples(
        dataframe[ROUND_1_SIMILARITY_COLUMN],
        dataframe[LAST_ROUND_SIMILARITY_COLUMN],
        n_bootstrap=N_BOOTSTRAP,
        confidence_level=CONFIDENCE_LEVEL,
        random_state=BOOTSTRAP_SEED,
    )


def analyze_pair(dataframe: pd.DataFrame) -> EvolutionAnalysis:
    """Analyze value-similarity evolution for one round-robin debate order."""
    trajectories = add_trajectory_similarities(dataframe)
    reached_consensus = trajectories[FINAL_VERDICT_COLUMN].notna()
    return EvolutionAnalysis(
        consensus=analyze_trajectory(trajectories.loc[reached_consensus]),
        no_consensus=analyze_trajectory(trajectories.loc[~reached_consensus]),
    )


def style_axis(axis: Axes) -> None:
    """Apply the shared Appendix Figure 7 axis styling."""
    axis.set_ylim(*Y_LIMITS)
    axis.set_xlim(*X_LIMITS)
    axis.set_xticks(
        X_POSITIONS,
        [bold_text("Round 1"), bold_text("Last Round")],
        ha="right",
        rotation=TICK_LABEL_ROTATION,
        fontsize=TICK_LABEL_SIZE,
    )
    axis.set_yticks(Y_TICKS)
    axis.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    axis.grid(axis="y", color=GRID_COLOR)
    axis.set_axisbelow(True)
    axis.set_facecolor(AXIS_FACE_COLOR)


def plot_trajectory(
    axis: Axes,
    analysis: EvolutionAnalysis,
    panel_index: int,
) -> None:
    """Plot consensus and no-consensus value-similarity trajectories."""
    consensus_x = (-TRAJECTORY_SHIFT, 1 - TRAJECTORY_SHIFT)
    no_consensus_x = (TRAJECTORY_SHIFT, 1 + TRAJECTORY_SHIFT)
    axis.errorbar(
        consensus_x,
        comparison_y_values(analysis.consensus),
        yerr=comparison_y_errors(analysis.consensus),
        fmt="o-",
        markersize=MARKER_SIZE,
        capsize=ERROR_CAPSIZE,
        color=CONSENSUS_COLOR,
    )
    axis.errorbar(
        no_consensus_x,
        comparison_y_values(analysis.no_consensus),
        yerr=comparison_y_errors(analysis.no_consensus),
        fmt="o-",
        markersize=MARKER_SIZE,
        capsize=ERROR_CAPSIZE,
        color=NO_CONSENSUS_COLOR,
        zorder=10 + panel_index,
    )
    style_axis(axis)
    add_significance_bracket_inplot(
        ax=axis,
        x1=-TRAJECTORY_SHIFT,
        x2=1 - TRAJECTORY_SHIFT,
        y=CONSENSUS_BRACKET_Y[panel_index],
        h=BRACKET_HEIGHT,
        label=bold_text(significance_label(analysis.consensus.p_value)),
    )
    add_significance_bracket_inplot(
        ax=axis,
        x1=TRAJECTORY_SHIFT,
        x2=1 + TRAJECTORY_SHIFT,
        y=NO_CONSENSUS_BRACKET_Y[panel_index],
        direction="down",
        color=NO_CONSENSUS_COLOR,
        h=BRACKET_HEIGHT,
        text_offset=NO_CONSENSUS_TEXT_OFFSET,
        fontsize=SIGNIFICANCE_SIZE,
        label=bold_text(significance_label(analysis.no_consensus.p_value)),
    )


def add_legend(figure: plt.Figure) -> None:
    """Add the consensus-outcome legend beside the panels."""
    handles = [
        Line2D([], [], color=CONSENSUS_COLOR, label=bold_text("Consensus")),
        Line2D(
            [],
            [],
            color=NO_CONSENSUS_COLOR,
            label=bold_text("No Consensus"),
        ),
    ]
    figure.legend(
        handles=handles,
        loc=LEGEND_LOCATION,
        bbox_to_anchor=LEGEND_BBOX,
        framealpha=1,
        prop={"size": LEGEND_SIZE},
    )


def build_figure(analyses: tuple[EvolutionAnalysis, ...]) -> plt.Figure:
    """Build the six-panel Appendix Figure 7 layout."""
    figure, axes = plt.subplots(
        2,
        3,
        figsize=FIGURE_SIZE,
        dpi=DPI,
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(hspace=SUBPLOT_HSPACE, wspace=SUBPLOT_WSPACE)

    for index, (axis, analysis, title) in enumerate(
        zip(axes.ravel(), analyses, PANEL_TITLES, strict=True)
    ):
        plot_trajectory(axis, analysis, index)
        axis.set_title(bold_text(title), fontsize=TITLE_SIZE)

    for axis in axes[:, 0]:
        axis.set_ylabel(bold_text("Value Similarity"), fontsize=Y_LABEL_SIZE)

    add_legend(figure)
    apply_subplot_labels(
        axes.ravel(),
        size=SUBPLOT_LABEL_SIZE,
        x=SUBPLOT_LABEL_X,
        y=SUBPLOT_LABEL_Y,
        bold=True,
    )
    return figure


def main() -> None:
    """Load round-robin data, build Appendix Figure 7, and save it as a PDF."""
    use_latex_style()
    plt.rcParams["axes.prop_cycle"] = cycler(color=COLOR_CYCLE)

    dataframes = tuple(load_debate_data(path) for path in DATA_PATHS)
    analyses = tuple(analyze_pair(dataframe) for dataframe in dataframes)
    figure = build_figure(analyses)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_PATH,
        bbox_inches="tight",
        pad_inches=SAVE_PAD_INCHES,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
