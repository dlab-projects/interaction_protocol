"""Shared helpers for statistical figures."""

from dataclasses import dataclass

import pandas as pd
from scipy.stats import mannwhitneyu

from interaction_protocol.utils import bootstrap_statistic_df, jaccard


@dataclass(frozen=True)
class SimilarityComparison:
    """Two bootstrapped similarity estimates and their rank-test p-value."""

    first: dict[str, float]
    second: dict[str, float]
    p_value: float


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
