"""Plot three-way value-similarity evolution (Appendix Figure 12).

Note: Unfortunately, we were unable to save the values extracted from the original
Gemini batch run, so we have archived the point estimates and confidence intervals
in this file. The values can be re-obtained by running the value batch analysis in
scripts/analysis/value_classify_batch.py.
"""
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_lego.labels import (
    add_significance_bracket_inplot,
    apply_subplot_labels,
    bold_text,
)
from mpl_lego.style import use_latex_style


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = (
    REPO_ROOT / "artifacts" / "figures" / "appA_figure_value_evolution_3way.pdf"
)

FIGURE_SIZE = (4, 2)
DPI = 300
SUBPLOT_HSPACE = 0.40
AXIS_FACE_COLOR = "0.96"
Y_LIMITS = (0, 0.65)
Y_TICKS = (0, 0.2, 0.4, 0.6)
X_LIMITS = (-0.5, 1.5)
X_TICKS = (0, 1)
TRAJECTORY_SHIFT = 0.10
X_POSITIONS = (-TRAJECTORY_SHIFT, 1 - TRAJECTORY_SHIFT)

MARKER_SIZE = 3
ERROR_CAPSIZE = 3
POINT_COLOR = "black"
TICK_LABEL_SIZE = 8
TICK_LABEL_ROTATION = 18
TITLE_SIZE = 8
Y_LABEL_SIZE = 9
SUBPLOT_LABEL_SIZE = 12
SUBPLOT_LABEL_X = -0.09
SUBPLOT_LABEL_Y = 1.10
SIGNIFICANCE_Y = 0.50
BRACKET_HEIGHT = 0.01
SAVE_PAD_INCHES = 0.10


@dataclass(frozen=True)
class Estimate:
    """A notebook-recorded point estimate and percentile confidence interval."""

    value: float
    ci_lower: float
    ci_upper: float

    @property
    def lower_error(self) -> float:
        """Return the distance from the point estimate to the lower bound."""
        return self.value - self.ci_lower

    @property
    def upper_error(self) -> float:
        """Return the distance from the point estimate to the upper bound."""
        return self.ci_upper - self.value


@dataclass(frozen=True)
class EvolutionComparison:
    """First- and last-round estimates for one model pair."""

    title: str
    round_1: Estimate
    last_round: Estimate


# Archived outputs from round_robin_values_3way.ipynb. The original Gemini batch
# result IDs no longer resolve, and the converted Parquet does not contain values.
COMPARISONS = (
    EvolutionComparison(
        title="Gemini-Claude",
        round_1=Estimate(
            0.29823978703289045,
            0.2830507351843558,
            0.31293282268497785,
        ),
        last_round=Estimate(
            0.405686794048863,
            0.3895496840324426,
            0.42086693598547054,
        ),
    ),
    EvolutionComparison(
        title="Gemini-GPT",
        round_1=Estimate(
            0.33794845001741547,
            0.3227145531671393,
            0.35366278238045473,
        ),
        last_round=Estimate(
            0.4188883913021844,
            0.40249647024431506,
            0.43385839615365485,
        ),
    ),
    EvolutionComparison(
        title="Claude-GPT",
        round_1=Estimate(
            0.4352869831318107,
            0.4178588533114395,
            0.45465282442653127,
        ),
        last_round=Estimate(
            0.46814449917898193,
            0.45375415173906564,
            0.4840496871423595,
        ),
    ),
)


def style_axis(axis: Axes) -> None:
    """Apply the original Appendix Figure 12 axis styling."""
    axis.set_ylim(*Y_LIMITS)
    axis.set_xlim(*X_LIMITS)
    axis.set_xticks(
        X_TICKS,
        [bold_text("Round 1"), bold_text("Last Round")],
        ha="right",
        rotation=TICK_LABEL_ROTATION,
        fontsize=TICK_LABEL_SIZE,
    )
    axis.grid(axis="y")
    axis.set_axisbelow(True)
    axis.set_yticks(Y_TICKS)
    axis.set_facecolor(AXIS_FACE_COLOR)


def plot_comparison(axis: Axes, comparison: EvolutionComparison) -> None:
    """Plot one pair's first-to-last-round value-similarity trajectory."""
    estimates = (comparison.round_1, comparison.last_round)
    axis.errorbar(
        X_POSITIONS,
        [estimate.value for estimate in estimates],
        yerr=[
            [estimate.lower_error for estimate in estimates],
            [estimate.upper_error for estimate in estimates],
        ],
        fmt="o-",
        markersize=MARKER_SIZE,
        capsize=ERROR_CAPSIZE,
        color=POINT_COLOR,
    )
    style_axis(axis)
    add_significance_bracket_inplot(
        ax=axis,
        x1=X_POSITIONS[0],
        x2=X_POSITIONS[1],
        y=SIGNIFICANCE_Y,
        h=BRACKET_HEIGHT,
        label=bold_text("***"),
    )
    axis.set_title(bold_text(comparison.title), fontsize=TITLE_SIZE)


def build_figure() -> plt.Figure:
    """Build Appendix Figure 12 using the notebook's original parameters."""
    figure, axes = plt.subplots(
        1,
        3,
        figsize=FIGURE_SIZE,
        dpi=DPI,
        sharex=True,
        sharey=True,
    )
    figure.subplots_adjust(hspace=SUBPLOT_HSPACE)
    for axis, comparison in zip(axes, COMPARISONS, strict=True):
        plot_comparison(axis, comparison)

    axes[0].set_ylabel(bold_text("Value Similarity"), fontsize=Y_LABEL_SIZE)
    apply_subplot_labels(
        axes,
        size=SUBPLOT_LABEL_SIZE,
        x=SUBPLOT_LABEL_X,
        y=SUBPLOT_LABEL_Y,
        bold=True,
    )
    return figure


def main() -> None:
    """Build Appendix Figure 12 and save it as a PDF."""
    use_latex_style()
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
