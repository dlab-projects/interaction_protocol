"""Plot held-out calibration for the selected verdict model."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from matplotlib.axes import Axes
from mpl_lego.labels import bold_text
from mpl_lego.style import use_latex_style


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = REPO_ROOT / "artifacts" / "analysis" / "heldout_calibration_bins.csv"
PNG_OUTPUT_PATH = (
    REPO_ROOT / "artifacts" / "figures" / "heldout_calibration_plot.png"
)
PDF_OUTPUT_PATH = (
    REPO_ROOT / "artifacts" / "figures" / "heldout_calibration_plot.pdf"
)

FIGURE_SIZE = (3.5, 2.8)
DPI = 300
LINE_WIDTH = 1.4
REFERENCE_LINE_WIDTH = 0.9
MARKER_SIZE = 4.0
AXIS_LABEL_SIZE = 9
TICK_LABEL_SIZE = 8
LEGEND_FONT_SIZE = 7
LEGEND_HANDLE_LENGTH = 1.4
SAVE_PAD_INCHES = 0.02
AXIS_FACE_COLOR = "#F7F7F7"
FIGURE_FACE_COLOR = "#F7F7F7"
AXIS_LIMITS = (0.0, 1.0)
AXIS_TICKS = (0.0, 0.25, 0.5, 0.75, 1.0)
VERDICT_ORDER = ("NTA", "YTA", "NAH", "ESH", "INFO")
COLOR_CYCLE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
)


def plot_calibration_bins(axis: Axes, calibration_bins: pd.DataFrame) -> None:
    """Plot observed verdict frequencies against predicted probabilities."""
    axis.plot(
        AXIS_LIMITS,
        AXIS_LIMITS,
        color="black",
        linewidth=REFERENCE_LINE_WIDTH,
        linestyle="--",
        label=bold_text("Perfect calibration"),
    )
    for verdict in VERDICT_ORDER:
        verdict_bins = calibration_bins.loc[
            calibration_bins["verdict"].eq(verdict)
        ].sort_values("mean_predicted_probability")
        if verdict_bins.empty:
            continue
        axis.plot(
            verdict_bins["mean_predicted_probability"],
            verdict_bins["observed_frequency"],
            marker="o",
            markersize=MARKER_SIZE,
            linewidth=LINE_WIDTH,
            label=bold_text(verdict),
        )


def main() -> None:
    """Load calibration bins and save the supplementary calibration plot."""
    use_latex_style()
    plt.rcParams["axes.prop_cycle"] = cycler(color=COLOR_CYCLE)
    calibration_bins = pd.read_csv(INPUT_PATH)

    fig, axis = plt.subplots(
        figsize=FIGURE_SIZE,
        dpi=DPI,
        facecolor=FIGURE_FACE_COLOR,
    )
    axis.set_facecolor(AXIS_FACE_COLOR)
    plot_calibration_bins(axis, calibration_bins)
    axis.set_xlabel(
        bold_text("Mean predicted probability"),
        fontsize=AXIS_LABEL_SIZE,
    )
    axis.set_ylabel(bold_text("Observed frequency"), fontsize=AXIS_LABEL_SIZE)
    axis.set_xlim(*AXIS_LIMITS)
    axis.set_ylim(*AXIS_LIMITS)
    axis.set_xticks(AXIS_TICKS)
    axis.set_yticks(AXIS_TICKS)
    axis.tick_params(axis="both", labelsize=TICK_LABEL_SIZE)
    axis.legend(
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handlelength=LEGEND_HANDLE_LENGTH,
        loc="upper left",
    )
    fig.tight_layout()

    PNG_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        PNG_OUTPUT_PATH,
        bbox_inches="tight",
        pad_inches=SAVE_PAD_INCHES,
        dpi=DPI,
        facecolor=FIGURE_FACE_COLOR,
    )
    fig.savefig(
        PDF_OUTPUT_PATH,
        bbox_inches="tight",
        pad_inches=SAVE_PAD_INCHES,
        facecolor=FIGURE_FACE_COLOR,
    )
    plt.close(fig)
    print(f"Wrote {PNG_OUTPUT_PATH}")
    print(f"Wrote {PDF_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
