"""Plot verdict distributions in three-way round-robin debate (Appendix Figure 9)."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mpl_lego.labels import bold_text
from mpl_lego.style import use_latex_style

from interaction_protocol.data import load_debate_data


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "analysis"
OUTPUT_PATH = (
    REPO_ROOT / "artifacts" / "figures" / "appB_figure_verdict_dist_3way.pdf"
)

DATASETS = (
    (
        "GPT→Claude→Gemini",
        DATA_DIR / "round_robin_3way_gpt_cla_gem.parquet",
    ),
    (
        "GPT→Gemini→Claude",
        DATA_DIR / "round_robin_3way_gpt_gem_cla.parquet",
    ),
    (
        "Claude→GPT→Gemini",
        DATA_DIR / "round_robin_3way_cla_gpt_gem.parquet",
    ),
    (
        "Claude→Gemini→GPT",
        DATA_DIR / "round_robin_3way_cla_gem_gpt.parquet",
    ),
    (
        "Gemini→GPT→Claude",
        DATA_DIR / "round_robin_3way_gem_gpt_cla.parquet",
    ),
    (
        "Gemini→Claude→GPT",
        DATA_DIR / "round_robin_3way_gem_cla_gpt.parquet",
    ),
)

FINAL_VERDICT_COLUMN = "final_verdict"
VERDICT_ORDER = ("NTA", "YTA", "NAH", "ESH", "INFO")

FIGURE_SIZE = (6, 3)
DPI = 300
BAR_WIDTH = 0.80
BAR_EDGE_COLOR = "black"
BAR_COLORS = tuple(plt.cm.tab10.colors[: len(DATASETS)])
AXIS_FACE_COLOR = "0.96"
Y_LIMIT_SCALE = 1.10
GRID_AXIS = "y"
LEGEND_LOCATION = "best"
SAVE_PAD_INCHES = 0.10


def verdict_distribution(dataframes: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return normalized final-verdict distributions in the paper's order."""
    return pd.DataFrame(
        {
            label: dataframe[FINAL_VERDICT_COLUMN]
            .value_counts(normalize=True)
            .reindex(VERDICT_ORDER, fill_value=0.0)
            for label, dataframe in dataframes.items()
        },
        index=VERDICT_ORDER,
    )


def build_figure(distribution: pd.DataFrame) -> plt.Figure:
    """Build Appendix Figure 9 using the notebook's original plot parameters."""
    figure, axis = plt.subplots(1, 1, figsize=FIGURE_SIZE, dpi=DPI)
    distribution.plot(
        kind="bar",
        width=BAR_WIDTH,
        edgecolor=BAR_EDGE_COLOR,
        color=BAR_COLORS,
        ax=axis,
    )
    axis.tick_params(rotation=0, axis="x")
    axis.set_xticklabels([bold_text(verdict) for verdict in VERDICT_ORDER])
    axis.set_ylabel(bold_text("Proportion of Dilemmas"))
    axis.legend(loc=LEGEND_LOCATION)
    axis.set_ylim(0, distribution.to_numpy().max() * Y_LIMIT_SCALE)
    axis.grid("on", axis=GRID_AXIS)
    axis.set_axisbelow(True)
    axis.set_facecolor(AXIS_FACE_COLOR)
    figure.tight_layout()
    return figure


def main() -> None:
    """Load three-way debate data, build Appendix Figure 9, and save it."""
    use_latex_style()

    dataframes = {
        label: load_debate_data(path)
        for label, path in DATASETS
    }
    figure = build_figure(verdict_distribution(dataframes))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_PATH,
        bbox_inches="tight",
        pad_inches=SAVE_PAD_INCHES,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
