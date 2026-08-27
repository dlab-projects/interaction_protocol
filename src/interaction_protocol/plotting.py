"""Shared helpers for statistical figures."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from mpl_lego.labels import bold_text
from scipy.stats import mannwhitneyu

from interaction_protocol.utils import bootstrap_statistic_df, change_of_minds, jaccard


@dataclass(frozen=True)
class SimilarityComparison:
    """Two bootstrapped similarity estimates and their rank-test p-value."""

    first: dict[str, float]
    second: dict[str, float]
    p_value: float


@dataclass(frozen=True)
class SynchronousOutcomePlotConfig:
    """Labels and styling for a two-panel synchronous-outcome figure."""

    pair_labels: tuple[str, ...]
    comparison_models: tuple[str, ...]
    focal_model: str
    pair_colors: tuple[str, ...]
    round_values: tuple[int, ...]
    round_tick_labels: tuple[str, ...]
    n_bootstrap: int
    confidence_level: float
    random_state: int
    round_bar_width: float
    change_bar_height: float
    change_group_spacing: float
    change_within_group_offset: float
    round_error_capsize: float
    change_error_capsize: float
    round_y_limits: tuple[float, float]
    change_x_limits: tuple[float, float]
    axis_face_color: str
    grid_color: str
    edge_color: str
    axis_label_size: float
    tick_label_size: float
    legend_size: float
    n_rounds_column: str = "n_rounds"
    final_verdict_column: str = "final_verdict"
    comparison_verdict_column: str = "Agent_1_verdicts"
    focal_verdict_column: str = "Agent_2_verdicts"


def consensus_round_proportion(
    dataframe: pd.DataFrame,
    round_value: int,
    *,
    n_rounds_column: str,
    final_verdict_column: str,
) -> float:
    """Return the fraction reaching consensus in a specified round."""
    return float(
        (
            dataframe[n_rounds_column].eq(round_value)
            & dataframe[final_verdict_column].notna()
        ).mean()
    )


def no_consensus_proportion(
    dataframe: pd.DataFrame,
    *,
    final_verdict_column: str,
) -> float:
    """Return the fraction of debates that did not reach consensus."""
    return float(dataframe[final_verdict_column].isna().mean())


def change_of_verdict_rate(
    dataframe: pd.DataFrame,
    *,
    verdict_column: str,
) -> float:
    """Return one agent's change-of-verdict rate."""
    return float(change_of_minds(dataframe[verdict_column], mean=True))


def bootstrap_dataframe_errors(
    dataframe: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    n_bootstrap: int,
    confidence_level: float,
    random_state: int,
) -> tuple[float, float]:
    """Return lower and upper bootstrap errors for a dataframe statistic."""
    result = bootstrap_statistic_df(
        df=dataframe,
        statistic_func=statistic,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
    )
    return result["lower_err"], result["upper_err"]


def _style_synchronous_outcome_axis(
    axis: Axes,
    *,
    grid_axis: str,
    config: SynchronousOutcomePlotConfig,
) -> None:
    """Apply shared styling to one synchronous-outcome axis."""
    axis.set_facecolor(config.axis_face_color)
    axis.grid(axis=grid_axis, color=config.grid_color)
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", labelsize=config.tick_label_size)


def plot_synchronous_round_outcomes(
    axis: Axes,
    dataframes: tuple[pd.DataFrame, ...],
    config: SynchronousOutcomePlotConfig,
) -> None:
    """Plot consensus timing and no-consensus rates by model pair."""
    x_positions = np.arange(len(config.round_values) + 1)
    for index, (dataframe, color, label) in enumerate(
        zip(dataframes, config.pair_colors, config.pair_labels, strict=True)
    ):
        statistics = [
            lambda frame, round_value=round_value: consensus_round_proportion(
                frame,
                round_value,
                n_rounds_column=config.n_rounds_column,
                final_verdict_column=config.final_verdict_column,
            )
            for round_value in config.round_values
        ]
        statistics.append(
            lambda frame: no_consensus_proportion(
                frame,
                final_verdict_column=config.final_verdict_column,
            )
        )
        heights = [statistic(dataframe) for statistic in statistics]
        errors = np.array(
            [
                bootstrap_dataframe_errors(
                    dataframe,
                    statistic,
                    n_bootstrap=config.n_bootstrap,
                    confidence_level=config.confidence_level,
                    random_state=config.random_state,
                )
                for statistic in statistics
            ]
        ).T

        axis.bar(
            x_positions + (index - 1) * config.round_bar_width,
            heights,
            width=config.round_bar_width,
            color=color,
            edgecolor=config.edge_color,
            label=bold_text(label),
            yerr=errors,
            capsize=config.round_error_capsize,
        )

    axis.set_ylim(*config.round_y_limits)
    axis.set_xticks(x_positions, config.round_tick_labels)
    axis.set_xlabel(
        bold_text("Number of Rounds"),
        fontsize=config.axis_label_size,
    )
    axis.set_ylabel(
        bold_text("Proportion of Dilemmas"),
        fontsize=config.axis_label_size,
    )
    axis.legend(fontsize=config.legend_size)
    _style_synchronous_outcome_axis(axis, grid_axis="y", config=config)


def plot_synchronous_change_rates(
    axis: Axes,
    dataframes: tuple[pd.DataFrame, ...],
    config: SynchronousOutcomePlotConfig,
) -> None:
    """Plot focal- and comparison-model change-of-verdict rates."""
    tick_positions: list[float] = []
    tick_labels: list[str] = []

    for index, (dataframe, color, comparison_model) in enumerate(
        zip(
            dataframes,
            config.pair_colors,
            config.comparison_models,
            strict=True,
        )
    ):
        center = index * config.change_group_spacing
        positions = np.array(
            [
                center + config.change_within_group_offset,
                center - config.change_within_group_offset,
            ]
        )
        columns = (
            config.comparison_verdict_column,
            config.focal_verdict_column,
        )
        rates = [
            change_of_verdict_rate(dataframe, verdict_column=verdict_column)
            for verdict_column in columns
        ]
        errors = np.array(
            [
                bootstrap_dataframe_errors(
                    dataframe,
                    lambda frame, verdict_column=verdict_column: (
                        change_of_verdict_rate(
                            frame,
                            verdict_column=verdict_column,
                        )
                    ),
                    n_bootstrap=config.n_bootstrap,
                    confidence_level=config.confidence_level,
                    random_state=config.random_state,
                )
                for verdict_column in columns
            ]
        ).T

        axis.barh(
            positions,
            rates,
            height=config.change_bar_height,
            color=color,
            edgecolor=config.edge_color,
            xerr=errors,
            error_kw={"capsize": config.change_error_capsize},
        )
        tick_positions.extend(positions)
        tick_labels.extend((comparison_model, config.focal_model))

    sorted_ticks = sorted(zip(tick_positions, tick_labels, strict=True))
    axis.set_xlim(*config.change_x_limits)
    axis.set_ylim(
        -0.7,
        config.change_group_spacing * (len(dataframes) - 1) + 0.7,
    )
    axis.set_yticks(
        [position for position, _ in sorted_ticks],
        [bold_text(label) for _, label in sorted_ticks],
    )
    axis.set_xlabel(
        bold_text("Change-of-Verdict Rate"),
        fontsize=config.axis_label_size,
    )
    _style_synchronous_outcome_axis(axis, grid_axis="x", config=config)


def pairwise_jaccard(
    first_values: pd.Series,
    second_values: pd.Series,
) -> pd.Series:
    """Calculate row-wise Jaccard similarity between two value-set series."""
    similarities = [
        jaccard(first, second)
        for first, second in zip(first_values, second_values, strict=True)
    ]
    return pd.Series(similarities, index=first_values.index, dtype=float)


def bootstrap_series_mean(
    values: pd.Series,
    *,
    n_bootstrap: int,
    confidence_level: float,
    random_state: int,
) -> dict[str, float]:
    """Bootstrap the mean of a numeric series."""
    dataframe = values.rename("value").reset_index(drop=True).to_frame()
    return bootstrap_statistic_df(
        df=dataframe,
        statistic_func=lambda sample: sample["value"].mean(),
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
    )


def compare_similarity_samples(
    first_values: pd.Series,
    second_values: pd.Series,
    *,
    n_bootstrap: int,
    confidence_level: float,
    random_state: int,
) -> SimilarityComparison:
    """Bootstrap two similarity samples and compare them with Mann-Whitney U."""
    first_values = first_values.dropna()
    second_values = second_values.dropna()
    p_value = mannwhitneyu(first_values, second_values).pvalue
    return SimilarityComparison(
        first=bootstrap_series_mean(
            first_values,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=random_state,
        ),
        second=bootstrap_series_mean(
            second_values,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_state=random_state,
        ),
        p_value=float(p_value),
    )


def comparison_y_values(comparison: SimilarityComparison) -> list[float]:
    """Return the two point estimates from a similarity comparison."""
    return [comparison.first["original"], comparison.second["original"]]


def comparison_y_errors(comparison: SimilarityComparison) -> list[list[float]]:
    """Return asymmetric bootstrap errors for a similarity comparison."""
    return [
        [comparison.first["lower_err"], comparison.second["lower_err"]],
        [comparison.first["upper_err"], comparison.second["upper_err"]],
    ]


def significance_label(
    p_value: float,
    *,
    strong_threshold: float = 1e-3,
    weak_threshold: float = 1e-1,
) -> str:
    """Return the significance notation used by the paper figures."""
    if p_value < strong_threshold:
        return "***"
    if p_value < weak_threshold:
        return "*"
    return "n.s."
