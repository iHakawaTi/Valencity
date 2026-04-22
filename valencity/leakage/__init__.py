"""
Data Leakage Prevention module.

This module provides tools for preventing and detecting data leakage
in machine learning pipelines, including safe cross-validation wrappers
and train/test split utilities.
"""

from valencity.leakage.cv import LeakageWarning, SafeCrossValidator
from valencity.leakage.detectors import LeakageDetector, LeakageType
from valencity.leakage.splitters import (
    group_train_test_split,
    safe_train_test_split,
    temporal_train_test_split,
)

__all__ = [
    "SafeCrossValidator",
    "LeakageWarning",
    "LeakageDetector",
    "LeakageType",
    "safe_train_test_split",
    "temporal_train_test_split",
    "group_train_test_split",
]
