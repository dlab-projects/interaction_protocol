"""Plot change-of-verdict rates in three-way debate (Appendix Figure 10)."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from mpl_lego.labels import bold_text
from mpl_lego.style import use_latex_style

from interaction_protocol.data import load_debate_data
from interaction_protocol.utils import change_of_minds


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "analysis"
OUTPUT_PATH = REPO_ROOT / "artifacts" / "figures" / "appA_figure_cov_3way.pdf"

EXPERIMENTS = (
    ("gem_cla_gpt", DATA_DIR / "round_robin_3way_gem_cla_gpt.parquet"),
    ("cla_gem_gpt", DATA_DIR / "round_robin_3way_cla_gem_gpt.parquet"),
    ("gpt_cla_gem", DATA_DIR / "round_robin_3way_gpt_cla_gem.parquet"),
    ("gpt_gem_cla", DATA_DIR / "round_robin_3way_gpt_gem_cla.parquet"),
    ("gem_gpt_cla", DATA_DIR / "round_robin_3way_gem_gpt_cla.parquet"),
    ("cla_gpt_gem", DATA_DIR / "round_robin_3way_cla_gpt_gem.parquet"),
)
MODELS = ("gpt", "cla", "gem")
MODEL_NAMES = {"gpt": "GPT", "cla": "Claude", "gem": "Gemini"}
POSITION_LABELS = ("First", "Second", "Third")
POSITION_INDICES = (0, 1, 2)

FIGURE_SIZE = (5, 6)
DPI = 300
X_POSITIONS = (0, 1, 2)
BAR_WIDTH = 0.30
BAR_GAP = 0.13
BAR_OFFSET = BAR_WIDTH / 2 + BAR_GAP / 2
BAR_EDGE_COLOR = "black"
BAR_COLORS = tuple(plt.cm.tab10.colors[:6])
AXIS_FACE_COLOR = "0.97"
Y_LIMITS = (0.0, 0.55)
Y_TICKS = (0, 0.1, 0.2, 0.3, 0.4, 0.5)
GRID_LINESTYLE = "--"
GRID_ALPHA = 0.40

ORDER_LABEL_SIZE = 6
ORDER_LABEL_OFFSET = 0.01
SAVE_PAD_INCHES = 0.10


def full_order_label(parts: list[str]) -> str:
    """Format a three-model order as a multiline plot annotation."""
    return "\n".join(
        f"{index + 1}) {MODEL_NAMES[model]}"
        for index, model in enumerate(parts)
    )


def collect_model_position_data(
    dataframes: dict[str, pd.DataFrame],
) -> dict[str, dict[int, list[tuple[str, float]]]]:
    """Collect two change-of-verdict rates per model and order position."""
    model_position_data = {
        model: {position: [] for position in POSITION_INDICES}
        for model in MODELS
    }
    for experiment_name, dataframe in dataframes.items():
        parts = experiment_name.split("_")
        for model in MODELS:
            position = parts.index(model)
            verdict_column = f"Agent_{position + 1}_verdicts"
            rate = change_of_minds(dataframe[verdict_column], mean=True)
            model_position_data[model][position].append(
                (full_order_label(parts), float(rate))
            )
    return model_position_data


def plot_model_axis(
    axis: Axes,
    model: str,
    position_data: dict[int, list[tuple[str, float]]],
) -> None:
    """Plot one model's rates across first, second, and third positions."""
    for position in POSITION_INDICES:
        items = sorted(position_data[position], key=lambda item: item[0])
        if len(items) != 2:
            message = (
                f"Expected two orders for {model} in position {position}, "
                f"got {len(items)}"
            )
            raise ValueError(message)
        for item_index, (label, rate) in enumerate(items):
            direction = -1 if item_index == 0 else 1
            x_position = X_POSITIONS[position] + direction * BAR_OFFSET
            axis.bar(
                x_position,
                rate,
                width=BAR_WIDTH,
                edgecolor=BAR_EDGE_COLOR,
                color=BAR_COLORS[2 * position + item_index],
            )
            axis.text(
                x_position,
                rate + ORDER_LABEL_OFFSET,
                bold_text(label),
                multialignment="left",
                ha="center",
                va="bottom",
                fontsize=ORDER_LABEL_SIZE,
                rotation=0,
                clip_on=False,
            )

    axis.set_xticks(
        X_POSITIONS,
        [bold_text(label) for label in POSITION_LABELS],
    )
    axis.set_title(bold_text(MODEL_NAMES[model]))
    axis.set_ylabel(bold_text("Change-of-Verdict Rate"))
    axis.grid(axis="y", linestyle=GRID_LINESTYLE, alpha=GRID_ALPHA)
    axis.set_ylim(*Y_LIMITS)
    axis.set_yticks(Y_TICKS)
    axis.set_facecolor(AXIS_FACE_COLOR)


def build_figure(
    model_position_data: dict[str, dict[int, list[tuple[str, float]]]],
) -> plt.Figure:
    """Build Appendix Figure 10 using the notebook's original parameters."""
    figure, axes = plt.subplots(
        3,
        1,
        figsize=FIGURE_SIZE,
        dpi=DPI,
        sharey=True,
        sharex=True,
    )
    for axis, model in zip(axes, MODELS, strict=True):
        plot_model_axis(axis, model, model_position_data[model])
    axes[-1].set_xlabel(bold_text("Position in Order"))
    figure.tight_layout()
    return figure


def main() -> None:
    """Load three-way debate data, build Appendix Figure 10, and save it."""
    use_latex_style()

    dataframes = {
        name: load_debate_data(path)
        for name, path in EXPERIMENTS
    }
    figure = build_figure(collect_model_position_data(dataframes))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT_PATH,
        bbox_inches="tight",
        pad_inches=SAVE_PAD_INCHES,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
