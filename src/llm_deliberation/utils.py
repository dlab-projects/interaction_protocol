from collections import Counter

import numpy as np
import pandas as pd


def move_column(df: pd.DataFrame, col: str, idx: int, *, inplace: bool = False) -> pd.DataFrame | None:
    """
    Move *col* to position *idx*.

    Parameters
    ----------
    df : pd.DataFrame
    col : str
        Column name to move.
    idx : int
        Target position (0-based). Negative values count from the end.
    inplace : bool, default False
        If True, modify *df* in place and return None.
        If False, return a modified copy.

    Returns
    -------
    pd.DataFrame | None
        Updated DataFrame when ``inplace=False``; otherwise None.
    """
    if col not in df.columns:
        raise KeyError(f"{col!r} not in DataFrame columns")

    target = df if inplace else df.copy()
    series = target.pop(col)

    n = len(target.columns)
    if idx < 0:
        idx = max(n + 1 + idx, 0)      # negative indexing
    idx = min(max(idx, 0), n)          # clamp to valid range

    target.insert(idx, col, series)
    return None if inplace else target


def label_to_num(df):
    """
    Converts categorical labels in a DataFrame to numerical values.

    This function replaces specific string labels in a DataFrame with their corresponding
    numerical representations. The mapping is as follows:
        - 'NTA' (Not the Asshole) -> 0
        - 'YTA' (You're the Asshole) -> 1
        - 'ESH' (Everyone Sucks Here) -> 2
        - 'NAH' (No Assholes Here) -> 3
        - 'INF' (Not Enough Info) -> 4

    Args:
        df (pd.DataFrame): The input DataFrame containing categorical labels.

    Returns:
        pd.DataFrame: A DataFrame with the categorical labels replaced by their numerical values.
    """
    # Replace the categorical labels with their corresponding numerical values
    return df.replace({
        'NTA': 0,  # Not the Asshole
        'YTA': 1,  # You're the Asshole
        'ESH': 2,  # Everyone Sucks Here
        'NAH': 3,  # No Assholes Here
        'INF': 4,  # Not Enough Info
        'INFO': 4,
        'REFUSAL': 4
    }).infer_objects(copy=False)  # Ensure the DataFrame's object types are inferred without copying


def jaccard(set1, set2):
    """
    Calculate the Jaccard similarity between two sets.

    Args:
        set1 (set): First set of elements.
        set2 (set): Second set of elements.

    Returns:
        float: Jaccard similarity coefficient, ranging from 0 to 1.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0


def jaccard_modified(set1, set2, value2group, weight):
    """
    Calculate a modified Jaccard similarity that allows for weighted group matches.

    Parameters
    ----------
    set1, set2 : Collection[str]
        Sets (or iterables convertible to sets) of values to compare.
    value2group : Mapping[str, Hashable]
        Mapping from value to a group identifier. Values sharing a group can
        contribute to a weighted "diluted" intersection when they are not exact matches.
    weight : float
        Weight contributed by each group-based match. Must be between 0 and 1.

    Returns
    -------
    float
        Modified Jaccard similarity coefficient in [0, 1].
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be between 0 and 1")

    set1 = set(set1)
    set2 = set(set2)

    union = len(set1.union(set2))
    if union == 0:
        return 0.0

    true_matches = set1.intersection(set2)

    remaining1 = set1.difference(true_matches)
    remaining2 = set2.difference(true_matches)

    groups1 = Counter(value2group.get(v) for v in remaining1 if value2group.get(v) is not None)
    groups2 = Counter(value2group.get(v) for v in remaining2 if value2group.get(v) is not None)

    diluted_matches = sum(min(groups1[group], groups2[group]) for group in groups1.keys() & groups2.keys())

    weighted_intersection = len(true_matches) + weight * diluted_matches
    return weighted_intersection / union


def bootstrap_statistic_df(df, statistic_func, n_bootstrap=1000, confidence_level=0.95, random_state=42):
    """
    Perform bootstrap resampling on a dataframe and calculate confidence intervals.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe to resample
    statistic_func : function
        Function that takes a dataframe and returns a single statistic
        Example: lambda df: df['column'].mean()
    n_bootstrap : int, default=1000
        Number of bootstrap samples
    confidence_level : float, default=0.95
        Confidence level for intervals (e.g., 0.95 for 95% CI)
    random_state : int, default=42
        Random seed for reproducibility
    
    Returns:
    --------
    dict with keys:
        - 'original': original statistic value
        - 'mean': mean of bootstrap distribution
        - 'std': standard error
        - 'ci_lower': lower confidence interval bound
        - 'ci_upper': upper confidence interval bound
    """
    
    # Calculate original statistic
    original_stat = statistic_func(df)
    
    # Bootstrap sampling
    bootstrap_stats = []
    np.random.seed(random_state)
    
    for i in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = df.sample(n=len(df), replace=True)
        
        # Calculate statistic for this bootstrap sample
        bootstrap_stat = statistic_func(bootstrap_sample)
        bootstrap_stats.append(bootstrap_stat)
    
    # Convert to numpy array for calculations
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = np.percentile(bootstrap_stats, lower_percentile)
    ci_upper = np.percentile(bootstrap_stats, upper_percentile)
    standard_error = np.std(bootstrap_stats)
    bootstrap_mean = np.mean(bootstrap_stats)
    
    return {
        'original': original_stat,
        'mean': bootstrap_mean,
        'std': standard_error,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'upper_err': ci_upper - bootstrap_mean,
        'lower_err': bootstrap_mean - ci_lower
    }

def change_of_minds(verdicts, mean=False):
    if mean:
        return verdicts.apply(lambda x: (len(x) > 1) & (len(set(x)) > 1)).mean()
    return verdicts.apply(lambda x: (len(x) > 1) & (len(set(x)) > 1)).sum()

def first_round_verdicts(verdicts):
    return verdicts.apply(lambda x: x[0])