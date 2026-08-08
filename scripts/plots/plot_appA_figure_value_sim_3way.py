"""Plot three-way value similarity by verdict agreement (Appendix Figure 11).

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
    REPO_ROOT / "artifacts" / "figures" / "appA_figure_value_sim_3way.pdf"
)

FIGURE_SIZE = (4, 2)
DPI = 300
SUBPLOT_HSPACE = 0.50
AXIS_FACE_COLOR = "0.97"
Y_LIMITS = (0, 0.65)
Y_TICKS = (0, 0.2, 0.4, 0.6)
X_LIMITS = (-0.5, 1.5)
X_POSITIONS = (0, 1)

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
SIGNIFICANCE_Y = 0.55
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
class AgreementComparison:
    """Agreement and disagreement estimates for one model pair."""

    title: str
    agreement: Estimate
    disagreement: Estimate


# Archived outputs from round_robin_values_3way.ipynb. The original Gemini batch
# result IDs no longer resolve, and the converted Parquet does not contain values.
COMPARISONS = (
    AgreementComparison(
        title="Gemini-Claude",
        agreement=Estimate(
            0.4424981103552532,
            0.42976311091812613,
            0.45529668427654757,
        ),
        disagreement=Estimate(
            0.3014374463503703,
            0.2907572326975398,
            0.3124850508489372,
        ),
    ),
    AgreementComparison(
        title="Gemini-GPT",
        agreement=Estimate(
            0.4420239084827614,
            0.43019313488764865,
            0.45328650925596065,
        ),
        disagreement=Estimate(
            0.3028172335600907,
            0.29027948979591833,
            0.31557089569160995,
        ),
    ),
    AgreementComparison(
        title="Claude-GPT",
        agreement=Estimate(
            0.4947726423902894,
            0.4851707166199813,
            0.5042334033613446,
        ),
        disagreement=Estimate(
            0.3306479381876207,
            0.3119776969429748,
            0.3485577811371462,
        ),
    ),
)


def style_axis(axis: Axes) -> None:
    """Apply the original Appendix Figure 11 axis styling."""
    axis.set_ylim(*Y_LIMITS)
    axis.set_xlim(*X_LIMITS)
    axis.set_xticks(
        X_POSITIONS,
        [bold_text("Agreement"), bold_text("Disagreement")],
        ha="right",
        rotation=TICK_LABEL_ROTATION,
        fontsize=TICK_LABEL_SIZE,
    )
    axis.grid(axis="y")
    axis.set_axisbelow(True)
    axis.set_yticks(Y_TICKS)
    axis.set_facecolor(AXIS_FACE_COLOR)


def plot_comparison(axis: Axes, comparison: AgreementComparison) -> None:
    """Plot one pair's agreement-versus-disagreement estimates."""
    estimates = (comparison.agreement, comparison.disagreement)
    axis.errorbar(
        X_POSITIONS,
        [estimate.value for estimate in estimates],
        yerr=[
            [estimate.lower_error for estimate in estimates],
            [estimate.upper_error for estimate in estimates],
        ],
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
        label=bold_text("***"),
    )
    axis.set_title(bold_text(comparison.title), fontsize=TITLE_SIZE)


def build_figure() -> plt.Figure:
    """Build Appendix Figure 11 using the notebook's original parameters."""
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
    """Build Appendix Figure 11 and save it as a PDF."""
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
