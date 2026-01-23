"""
Data Leakage Prevention module.

This module provides tools for preventing and detecting data leakage
in machine learning pipelines, including safe cross-validation wrappers
and train/test split utilities.
"""

from dataguard.leakage.cv import SafeCrossValidator, LeakageWarning
from dataguard.leakage.splitters import (
    safe_train_test_split,
    temporal_train_test_split,
    group_train_test_split,
)
from dataguard.leakage.detectors import LeakageDetector, LeakageType

__all__ = [
    "SafeCrossValidator",
    "LeakageWarning",
    "LeakageDetector",
    "LeakageType",
    "safe_train_test_split",
    "temporal_train_test_split",
    "group_train_test_split",
]
