"""
DataGuard - A privacy & ML safety toolkit for production ML pipelines.

Modules:
    - pii: Detect and mask personally identifiable information
    - validation: Schema validation, data quality, and drift detection
    - leakage: Prevent data leakage in ML pipelines
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from dataguard import pii, validation, leakage

__all__ = ["pii", "validation", "leakage", "__version__"]
