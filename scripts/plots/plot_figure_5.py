"""Plot value similarity during synchronous debate for Figure 5."""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from mpl_lego.labels import (
    add_significance_bracket_inplot,
    apply_subplot_labels,
    bold_text,
)
from mpl_lego.style import use_latex_style
from scipy.stats import mannwhitneyu

from interaction_protocol.data import load_debate_data
from interaction_protocol.utils import bootstrap_statistic_df, jaccard


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "experiments"
OUTPUT_PATH = REPO_ROOT / "artifacts" / "figures" / "figure_5.pdf"

DATA_PATHS = (
    DATA_DIR / "sync_h2h_cla_vs_gpt.parquet",
    DATA_DIR / "sync_h2h_cla_vs_gem.parquet",
    DATA_DIR / "sync_h2h_gpt_vs_gem.parquet",
)
PANEL_TITLES = (
    "GPT\nvs.\nClaude",
    "Claude\nvs.\nGemini",
    "Gemini\nvs.\nGPT",
)

N_ROUNDS_COLUMN = "n_rounds"
FINAL_VERDICT_COLUMN = "final_verdict"
AGENT_VERDICT_COLUMNS = ("Agent_1_verdicts", "Agent_2_verdicts")
AGENT_VALUE_COLUMNS = ("Agent_1_values", "Agent_2_values")
ROUND_COLUMNS = AGENT_VERDICT_COLUMNS + AGENT_VALUE_COLUMNS
SIMILARITY_COLUMN = "value_similarity"
ROUND_1_SIMILARITY_COLUMN = "round_1_value_similarity"
LAST_ROUND_SIMILARITY_COLUMN = "last_round_value_similarity"

FIGURE_SIZE = (7.2, 2.25)
DPI = 300
GRID_WIDTH_RATIOS = (1, 1, 1, 0.30, 1, 1, 1)
GRID_WSPACE = 0.30
AXIS_FACE_COLOR = "0.96"
Y_LIMITS = (0, 0.65)
Y_TICKS = (0, 0.2, 0.4, 0.6)
X_LIMITS = (-0.5, 1.5)
X_POSITIONS = (0, 1)

N_BOOTSTRAP = 1_000
CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_SEED = 2_332
P_STRONG = 1e-3
P_WEAK = 1e-1

MARKER_SIZE = 3
ERROR_CAPSIZE = 3
CONSENSUS_COLOR = "black"
NO_CONSENSUS_COLOR = "#708090"
TRAJECTORY_SHIFT = 0.10
COLOR_CYCLE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")

TICK_LABEL_SIZE = 8
TICK_LABEL_ROTATION = 18
TITLE_SIZE = 9
Y_LABEL_SIZE = 9
LEGEND_SIZE = 7.5
LEGEND_LOCATION = "center"
LEGEND_BBOX = (1, 0.50)
SUBPLOT_LABEL_X = -0.20
SUBPLOT_LABEL_Y = 1.15

AGREEMENT_BRACKET_Y = 0.51
CONSENSUS_BRACKET_Y = 0.50
NO_CONSENSUS_BRACKET_Y = (0.23, 0.22, 0.18)
BRACKET_HEIGHT = 0.01
NO_CONSENSUS_TEXT_OFFSET = 0.02
NO_CONSENSUS_SIGNIFICANCE_SIZE = 10
SAVE_PAD_INCHES = 0.02


@dataclass(frozen=True)
class Comparison:
    """Two bootstrap estimates and their Mann-Whitney comparison."""

    first: dict[str, float]
    second: dict[str, float]
    p_value: float


@dataclass(frozen=True)
class PairAnalysis:
    """All value-similarity comparisons for one model pair."""

    agreement: Comparison
    consensus_trajectory: Comparison
    no_consensus_trajectory: Comparison


def pairwise_jaccard(
    first_values: pd.Series,
    second_values: pd.Series,
) -> pd.Series:
    """Calculate row-wise Jaccard similarity between two value-set columns."""
    similarities = [
        jaccard(first, second)
        for first, second in zip(first_values, second_values, strict=True)
    ]
    return pd.Series(similarities, index=first_values.index, dtype=float)


def bootstrap_mean(dataframe: pd.DataFrame, column: str) -> dict[str, float]:
    """Bootstrap the mean of a precomputed similarity column."""
    return bootstrap_statistic_df(
        df=dataframe,
        statistic_func=lambda sample: sample[column].mean(),
        n_bootstrap=N_BOOTSTRAP,
        confidence_level=CONFIDENCE_LEVEL,
        random_state=BOOTSTRAP_SEED,
    )


def analyze_agreement(dataframe: pd.DataFrame) -> Comparison:
    """Compare message-level value similarity during agreement and disagreement."""
    exploded = dataframe.explode(list(ROUND_COLUMNS)).reset_index(drop=True)
    exploded[SIMILARITY_COLUMN] = pairwise_jaccard(
        exploded[AGENT_VALUE_COLUMNS[0]], exploded[AGENT_VALUE_COLUMNS[1]]
    )
    agreement = exploded[AGENT_VERDICT_COLUMNS[0]].eq(
        exploded[AGENT_VERDICT_COLUMNS[1]]
    )
    agreement_rows = exploded.loc[agreement]
    disagreement_rows = exploded.loc[~agreement]
    p_value = mannwhitneyu(
        agreement_rows[SIMILARITY_COLUMN],
        disagreement_rows[SIMILARITY_COLUMN],
    ).pvalue
    return Comparison(
        first=bootstrap_mean(agreement_rows, SIMILARITY_COLUMN),
        second=bootstrap_mean(disagreement_rows, SIMILARITY_COLUMN),
        p_value=float(p_value),
    )


def add_trajectory_similarities(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add first- and last-round similarity columns to multi-round debates."""
    multi_round = dataframe.loc[dataframe[N_ROUNDS_COLUMN] > 1].copy()
    multi_round[ROUND_1_SIMILARITY_COLUMN] = [
        jaccard(agent_1[0], agent_2[0])
        for agent_1, agent_2 in zip(
            multi_round[AGENT_VALUE_COLUMNS[0]],
            multi_round[AGENT_VALUE_COLUMNS[1]],
            strict=True,
        )
    ]
    multi_round[LAST_ROUND_SIMILARITY_COLUMN] = [
        jaccard(agent_1[-1], agent_2[-1])
        for agent_1, agent_2 in zip(
            multi_round[AGENT_VALUE_COLUMNS[0]],
            multi_round[AGENT_VALUE_COLUMNS[1]],
            strict=True,
        )
    ]
    return multi_round


def analyze_trajectory(dataframe: pd.DataFrame) -> Comparison:
    """Compare Round 1 and last-round similarity for a debate subset."""
    p_value = mannwhitneyu(
        dataframe[ROUND_1_SIMILARITY_COLUMN],
        dataframe[LAST_ROUND_SIMILARITY_COLUMN],
    ).pvalue
    return Comparison(
        first=bootstrap_mean(dataframe, ROUND_1_SIMILARITY_COLUMN),
        second=bootstrap_mean(dataframe, LAST_ROUND_SIMILARITY_COLUMN),
        p_value=float(p_value),
    )


def analyze_pair(dataframe: pd.DataFrame) -> PairAnalysis:
    """Calculate all Figure 5 comparisons for one model pair."""
    trajectories = add_trajectory_similarities(dataframe)
    consensus = trajectories.loc[trajectories[FINAL_VERDICT_COLUMN].notna()]
    no_consensus = trajectories.loc[trajectories[FINAL_VERDICT_COLUMN].isna()]
    return PairAnalysis(
        agreement=analyze_agreement(dataframe),
        consensus_trajectory=analyze_trajectory(consensus),
        no_consensus_trajectory=analyze_trajectory(no_consensus),
    )


def significance_label(p_value: float) -> str:
    """Return the significance notation used in the paper."""
    if p_value < P_STRONG:
        return "***"
    if p_value < P_WEAK:
        return "*"
    return "n.s."


def comparison_y_values(comparison: Comparison) -> list[float]:
    """Return the original statistics for a two-point comparison."""
    return [comparison.first["original"], comparison.second["original"]]


def comparison_y_errors(comparison: Comparison) -> list[list[float]]:
    """Return asymmetric bootstrap errors for a two-point comparison."""
    return [
        [comparison.first["lower_err"], comparison.second["lower_err"]],
        [comparison.first["upper_err"], comparison.second["upper_err"]],
    ]


def style_axis(axis: Axes, tick_labels: tuple[str, str]) -> None:
    """Apply the shared Figure 5 axis styling."""
    axis.set_ylim(*Y_LIMITS)
    axis.set_xlim(*X_LIMITS)
    axis.set_xticks(
        X_POSITIONS,
        [bold_text(label) for label in tick_labels],
        ha="right",
        rotation=TICK_LABEL_ROTATION,
        fontsize=TICK_LABEL_SIZE,
    )
    axis.set_yticks(Y_TICKS)
    axis.grid(axis="y")
    axis.set_axisbelow(True)
    axis.set_facecolor(AXIS_FACE_COLOR)


def plot_agreement_axis(axis: Axes, comparison: Comparison) -> None:
    """Plot agreement versus disagreement value similarity."""
    axis.errorbar(
        X_POSITIONS,
        comparison_y_values(comparison),
        yerr=comparison_y_errors(comparison),
        fmt="o",
        markersize=MARKER_SIZE,
        capsize=ERROR_CAPSIZE,
        color=CONSENSUS_COLOR,
    )
    style_axis(axis, ("Consensus", "Disagreement"))
    add_significance_bracket_inplot(
        ax=axis,
        x1=0,
        x2=1,
        y=AGREEMENT_BRACKET_Y,
        h=BRACKET_HEIGHT,
        label=bold_text(significance_label(comparison.p_value)),
    )


def plot_trajectory_axis(
    axis: Axes,
    consensus: Comparison,
    no_consensus: Comparison,
    panel_index: int,
) -> None:
    """Plot Round 1 to last-round trajectories by consensus outcome."""
    consensus_x = (-TRAJECTORY_SHIFT, 1 - TRAJECTORY_SHIFT)
    no_consensus_x = (TRAJECTORY_SHIFT, 1 + TRAJECTORY_SHIFT)
    axis.errorbar(
        consensus_x,
        comparison_y_values(consensus),
        yerr=comparison_y_errors(consensus),
        fmt="o-",
        markersize=MARKER_SIZE,
        capsize=ERROR_CAPSIZE,
        color=CONSENSUS_COLOR,
    )
    axis.errorbar(
        no_consensus_x,
        comparison_y_values(no_consensus),
        yerr=comparison_y_errors(no_consensus),
        fmt="o-",
        markersize=MARKER_SIZE,
        capsize=ERROR_CAPSIZE,
        color=NO_CONSENSUS_COLOR,
        zorder=10 + panel_index,
    )
    style_axis(axis, ("Round 1", "Last Round"))
    add_significance_bracket_inplot(
        ax=axis,
        x1=-TRAJECTORY_SHIFT,
        x2=1 - TRAJECTORY_SHIFT,
        y=CONSENSUS_BRACKET_Y,
        h=BRACKET_HEIGHT,
        label=bold_text(significance_label(consensus.p_value)),
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
        fontsize=NO_CONSENSUS_SIGNIFICANCE_SIZE,
        label=bold_text(significance_label(no_consensus.p_value)),
    )


def build_axes(figure: plt.Figure) -> list[Axes]:
    """Create two three-panel blocks separated by a spacer column."""
    grid = GridSpec(
        1,
        7,
        figure=figure,
        width_ratios=GRID_WIDTH_RATIOS,
        wspace=GRID_WSPACE,
    )
    axes = [figure.add_subplot(grid[0, 0])]
    axes.extend(
        figure.add_subplot(grid[0, index], sharey=axes[0])
        for index in (1, 2, 4, 5, 6)
    )
    return axes


def add_legend(figure: plt.Figure) -> None:
    """Add the consensus-outcome legend beside the trajectory panels."""
    handles = [
        Line2D([], [], color=CONSENSUS_COLOR, label=bold_text("Consensus")),
        Line2D([], [], color=NO_CONSENSUS_COLOR, label=bold_text("No Consensus")),
    ]
    figure.legend(
        handles=handles,
        loc=LEGEND_LOCATION,
        bbox_to_anchor=LEGEND_BBOX,
        framealpha=1,
        prop={"size": LEGEND_SIZE},
    )


def build_figure(analyses: tuple[PairAnalysis, ...]) -> plt.Figure:
    """Build the complete six-panel Figure 5 layout."""
    figure = plt.figure(figsize=FIGURE_SIZE, dpi=DPI)
    axes = build_axes(figure)

    for index, (analysis, title) in enumerate(
        zip(analyses, PANEL_TITLES, strict=True)
    ):
        plot_agreement_axis(axes[index], analysis.agreement)
        plot_trajectory_axis(
            axes[index + 3],
            analysis.consensus_trajectory,
            analysis.no_consensus_trajectory,
            index,
        )
        axes[index].set_title(bold_text(title), fontsize=TITLE_SIZE)
        axes[index + 3].set_title(bold_text(title), fontsize=TITLE_SIZE)

    for axis in axes:
        axis.tick_params(axis="y", labelleft=False)
    for axis in (axes[0], axes[3]):
        axis.tick_params(axis="y", labelleft=True, labelsize=TICK_LABEL_SIZE)
        axis.set_ylabel(bold_text("Average Value Similarity"), fontsize=Y_LABEL_SIZE)

    add_legend(figure)
    apply_subplot_labels(
        [axes[0], axes[3]],
        bold=True,
        x=SUBPLOT_LABEL_X,
        y=SUBPLOT_LABEL_Y,
    )
    return figure


def main() -> None:
    """Load synchronous debate data, build Figure 5, and save it as a PDF."""
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
