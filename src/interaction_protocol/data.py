"""Read and write debate datasets with stable Python-level types."""

from pathlib import Path

import pandas as pd


PICKLE_SUFFIXES = {".pickle", ".pkl"}
SEQUENCE_COLUMN_SUFFIXES = ("_messages", "_verdicts")
SEQUENCE_COLUMN_NAMES = {"verdict_chain"}
VALUE_COLUMN_SUFFIX = "_values"


def _sequence_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return columns whose cells contain one item per debate round."""
    return [
        column
        for column in dataframe.columns
        if column in SEQUENCE_COLUMN_NAMES
        or column.endswith(SEQUENCE_COLUMN_SUFFIXES)
    ]


def _value_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return columns whose cells contain one set of values per round."""
    return [
        column for column in dataframe.columns if column.endswith(VALUE_COLUMN_SUFFIX)
    ]


def _as_list(value):
    """Convert an Arrow-backed sequence to a plain Python list."""
    if value is None:
        return None
    return list(value)


def _values_for_storage(round_values):
    """Convert per-round value sets to deterministic lists for Parquet storage."""
    if round_values is None:
        return None
    return [sorted(values) for values in round_values]


def _values_for_analysis(round_values):
    """Restore per-round value collections to sets for analysis operations."""
    if round_values is None:
        return None
    return [set(values) for values in round_values]


def _restore_python_types(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Restore list and set cells after reading an Arrow-backed format."""
    dataframe = dataframe.copy()
    for column in _sequence_columns(dataframe):
        dataframe[column] = dataframe[column].map(_as_list)
    for column in _value_columns(dataframe):
        dataframe[column] = dataframe[column].map(_values_for_analysis)
    return dataframe


def _prepare_for_parquet(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert Python-only containers into deterministic Parquet-compatible values."""
    dataframe = dataframe.copy()
    for column in _sequence_columns(dataframe):
        dataframe[column] = dataframe[column].map(_as_list)
    for column in _value_columns(dataframe):
        dataframe[column] = dataframe[column].map(_values_for_storage)
    return dataframe


def load_debate_data(path: str | Path) -> pd.DataFrame:
    """Load debate data from Parquet or a trusted legacy pickle file.

    Parquet stores per-round value sets as nested lists. This function restores
    those inner collections to sets so existing set-based analyses are unchanged.
    Pickle files should only be loaded from trusted sources.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return _restore_python_types(pd.read_parquet(path))
    if suffix in PICKLE_SUFFIXES:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported debate data format: {path.suffix}")


def save_debate_data(dataframe: pd.DataFrame, path: str | Path) -> None:
    """Save debate data as Parquet or a legacy pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        _prepare_for_parquet(dataframe).to_parquet(path, index=False)
        return
    if suffix in PICKLE_SUFFIXES:
        dataframe.to_pickle(path)
        return
    raise ValueError(f"Unsupported debate data format: {path.suffix}")
