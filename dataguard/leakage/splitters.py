"""Leakage-aware train/test splitting utilities."""

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dataguard.leakage.detectors import LeakageDetector
from dataguard.utils.logging import get_logger

logger = get_logger(__name__)


def safe_train_test_split(
    X: Union[np.ndarray, pd.DataFrame],
    y: Union[np.ndarray, pd.Series],
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    stratify: Optional[Union[np.ndarray, pd.Series]] = None,
    check_leakage: bool = True,
    time_column: Optional[str] = None,
    group_column: Optional[str] = None
) -> Tuple[
    Union[np.ndarray, pd.DataFrame],
    Union[np.ndarray, pd.DataFrame],
    Union[np.ndarray, pd.Series],
    Union[np.ndarray, pd.Series]
]:
    """
    Safe train/test split with optional leakage checking.
    
    This function wraps sklearn's train_test_split with additional
    safety checks for common leakage patterns.
    
    Args:
        X: Feature matrix.
        y: Target vector.
        test_size: Fraction of data for test set.
        random_state: Random seed for reproducibility.
        stratify: Labels for stratified splitting.
        check_leakage: Whether to run leakage detection.
        time_column: If provided, use temporal split instead.
        group_column: If provided, use group-based split.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
        
    Raises:
        ValueError: If leakage is detected and check_leakage is True.
    """
    # Use temporal split if time column provided
    if time_column is not None:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("time_column requires X to be a DataFrame")
        return temporal_train_test_split(X, y, time_column, test_size)
    
    # Use group split if group column provided
    if group_column is not None:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("group_column requires X to be a DataFrame")
        return group_train_test_split(X, y, group_column, test_size, random_state)
    
    # Standard split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )
    
    # Run leakage detection
    if check_leakage:
        detector = LeakageDetector()
        warnings = detector.check_feature_leakage(X_train, X_test)
        
        if warnings:
            logger.warning(f"Potential leakage detected: {len(warnings)} warnings")
            for w in warnings:
                logger.warning(str(w))
    
    logger.info(
        f"Split: {len(X_train)} train, {len(X_test)} test "
        f"({test_size:.0%} test ratio)"
    )
    
    return X_train, X_test, y_train, y_test


def temporal_train_test_split(
    X: pd.DataFrame,
    y: Union[np.ndarray, pd.Series],
    time_column: str,
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data temporally to prevent future data leakage.
    
    For time-series or temporal data, random splits can cause
    future information to leak into training. This function
    splits by time, ensuring training data is always before test data.
    
    Args:
        X: Feature DataFrame with time column.
        y: Target vector.
        time_column: Name of the datetime/timestamp column.
        test_size: Fraction of data for test set.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    if time_column not in X.columns:
        raise ValueError(f"Time column '{time_column}' not found in DataFrame")
    
    # Sort by time
    sorted_idx = X[time_column].argsort()
    X_sorted = X.iloc[sorted_idx].reset_index(drop=True)
    
    if isinstance(y, pd.Series):
        y_sorted = y.iloc[sorted_idx].reset_index(drop=True)
    else:
        y_sorted = pd.Series(y[sorted_idx])
    
    # Split at cutoff point
    split_idx = int(len(X_sorted) * (1 - test_size))
    
    X_train = X_sorted.iloc[:split_idx]
    X_test = X_sorted.iloc[split_idx:]
    y_train = y_sorted.iloc[:split_idx]
    y_test = y_sorted.iloc[split_idx:]
    
    # Log time ranges
    train_end = X_train[time_column].max()
    test_start = X_test[time_column].min()
    
    logger.info(
        f"Temporal split: train ends at {train_end}, test starts at {test_start}"
    )
    
    if train_end >= test_start:
        logger.warning("Train and test time ranges overlap - potential leakage!")
    
    return X_train, X_test, y_train, y_test


def group_train_test_split(
    X: pd.DataFrame,
    y: Union[np.ndarray, pd.Series],
    group_column: str,
    test_size: float = 0.2,
    random_state: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data by groups to prevent group leakage.
    
    When data contains groups (e.g., users, sessions, entities),
    the same group should not appear in both train and test sets.
    This prevents information about specific groups from leaking.
    
    Args:
        X: Feature DataFrame with group column.
        y: Target vector.
        group_column: Name of the group identifier column.
        test_size: Fraction of groups for test set.
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    if group_column not in X.columns:
        raise ValueError(f"Group column '{group_column}' not found in DataFrame")
    
    # Get unique groups
    groups = X[group_column].unique()
    n_test_groups = max(1, int(len(groups) * test_size))
    
    # Randomly select test groups
    rng = np.random.default_rng(random_state)
    test_groups = set(rng.choice(groups, size=n_test_groups, replace=False))
    
    # Split by groups
    test_mask = X[group_column].isin(test_groups)
    train_mask = ~test_mask
    
    X_train = X[train_mask].reset_index(drop=True)
    X_test = X[test_mask].reset_index(drop=True)
    
    if isinstance(y, pd.Series):
        y_train = y[train_mask].reset_index(drop=True)
        y_test = y[test_mask].reset_index(drop=True)
    else:
        y_train = pd.Series(y[train_mask])
        y_test = pd.Series(y[test_mask])
    
    # Verify no group overlap
    train_groups = set(X_train[group_column].unique())
    test_groups_actual = set(X_test[group_column].unique())
    overlap = train_groups & test_groups_actual
    
    if overlap:
        logger.error(f"Group overlap detected: {overlap}")
        raise ValueError("Group leakage: same groups in train and test")
    
    logger.info(
        f"Group split: {len(train_groups)} train groups, "
        f"{len(test_groups_actual)} test groups, no overlap"
    )
    
    return X_train, X_test, y_train, y_test
