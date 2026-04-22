"""Data quality checking utilities."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from valencity.utils.logging import get_logger

logger = get_logger(__name__)


class QualityCheckType(Enum):
    """Types of quality checks."""
    
    NULLS = "nulls"
    DUPLICATES = "duplicates"
    OUTLIERS = "outliers"
    CARDINALITY = "cardinality"
    COMPLETENESS = "completeness"


class QualityStatus(Enum):
    """Status of a quality check."""
    
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class QualityCheckResult:
    """Result of a single quality check."""
    
    check_type: QualityCheckType
    column: Optional[str]
    status: QualityStatus
    metric_value: float
    threshold: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    """Comprehensive data quality report."""
    
    checks: List[QualityCheckResult] = field(default_factory=list)
    total_rows: int = 0
    total_columns: int = 0
    
    @property
    def passed(self) -> bool:
        """Returns True if all checks passed."""
        return all(c.status == QualityStatus.PASS for c in self.checks)
    
    @property
    def failed_checks(self) -> List[QualityCheckResult]:
        """Get list of failed checks."""
        return [c for c in self.checks if c.status == QualityStatus.FAIL]
    
    @property
    def warning_checks(self) -> List[QualityCheckResult]:
        """Get list of warning checks."""
        return [c for c in self.checks if c.status == QualityStatus.WARN]
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        passed = sum(1 for c in self.checks if c.status == QualityStatus.PASS)
        warned = sum(1 for c in self.checks if c.status == QualityStatus.WARN)
        failed = sum(1 for c in self.checks if c.status == QualityStatus.FAIL)
        
        icon = "✅" if failed == 0 else "❌"
        lines = [
            f"{icon} Data Quality Report ({self.total_rows} rows, {self.total_columns} cols)",
            f"   Passed: {passed} | Warnings: {warned} | Failed: {failed}",
        ]
        
        if failed > 0:
            lines.append("\nFailed checks:")
            for check in self.failed_checks:
                col = f"[{check.column}]" if check.column else ""
                lines.append(f"  ❌ {check.check_type.value} {col}: {check.message}")
        
        if warned > 0:
            lines.append("\nWarnings:")
            for check in self.warning_checks:
                col = f"[{check.column}]" if check.column else ""
                lines.append(f"  ⚠️  {check.check_type.value} {col}: {check.message}")
        
        return "\n".join(lines)


class DataQualityChecker:
    """
    Check data quality metrics for DataFrames.
    
    Performs various quality checks including:
    - Null value detection
    - Duplicate detection
    - Outlier detection (IQR, Z-score methods)
    - Cardinality checks
    - Completeness scoring
    
    Example:
        >>> checker = DataQualityChecker()
        >>> report = checker.full_report(df)
        >>> print(report.summary())
        
        >>> # Or run individual checks
        >>> nulls = checker.check_nulls(df, threshold=0.05)
        >>> if not nulls.passed:
        ...     print("Too many nulls!")
    """
    
    def __init__(
        self,
        null_threshold: float = 0.1,
        duplicate_threshold: float = 0.05,
        outlier_threshold: float = 0.01,
        low_cardinality_threshold: int = 10,
        high_cardinality_ratio: float = 0.95
    ):
        """
        Initialize the quality checker.
        
        Args:
            null_threshold: Max acceptable null fraction per column.
            duplicate_threshold: Max acceptable duplicate row fraction.
            outlier_threshold: Max acceptable outlier fraction.
            low_cardinality_threshold: Min unique values for warning.
            high_cardinality_ratio: Max unique ratio for categorical.
        """
        self.null_threshold = null_threshold
        self.duplicate_threshold = duplicate_threshold
        self.outlier_threshold = outlier_threshold
        self.low_cardinality_threshold = low_cardinality_threshold
        self.high_cardinality_ratio = high_cardinality_ratio
    
    def check_nulls(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        threshold: Optional[float] = None
    ) -> QualityReport:
        """
        Check for null values in DataFrame.
        
        Args:
            df: DataFrame to check.
            columns: Specific columns to check. Defaults to all.
            threshold: Override default null threshold.
            
        Returns:
            QualityReport with null check results.
        """
        threshold = threshold or self.null_threshold
        columns = columns or df.columns.tolist()
        
        report = QualityReport(total_rows=len(df), total_columns=len(df.columns))
        
        for col in columns:
            if col not in df.columns:
                continue
            
            null_count = df[col].isna().sum()
            null_ratio = null_count / len(df) if len(df) > 0 else 0
            
            if null_ratio > threshold:
                status = QualityStatus.FAIL
            elif null_ratio > threshold / 2:
                status = QualityStatus.WARN
            else:
                status = QualityStatus.PASS
            
            report.checks.append(QualityCheckResult(
                check_type=QualityCheckType.NULLS,
                column=col,
                status=status,
                metric_value=null_ratio,
                threshold=threshold,
                message=f"{null_ratio:.1%} null values ({null_count}/{len(df)})",
                details={"null_count": null_count}
            ))
        
        return report
    
    def check_duplicates(
        self,
        df: pd.DataFrame,
        subset: Optional[List[str]] = None,
        threshold: Optional[float] = None
    ) -> QualityReport:
        """
        Check for duplicate rows.
        
        Args:
            df: DataFrame to check.
            subset: Columns to consider for duplicates.
            threshold: Override default duplicate threshold.
            
        Returns:
            QualityReport with duplicate check results.
        """
        threshold = threshold or self.duplicate_threshold
        
        report = QualityReport(total_rows=len(df), total_columns=len(df.columns))
        
        dup_mask = df.duplicated(subset=subset)
        dup_count = dup_mask.sum()
        dup_ratio = dup_count / len(df) if len(df) > 0 else 0
        
        if dup_ratio > threshold:
            status = QualityStatus.FAIL
        elif dup_ratio > threshold / 2:
            status = QualityStatus.WARN
        else:
            status = QualityStatus.PASS
        
        subset_str = f" on columns {subset}" if subset else ""
        
        report.checks.append(QualityCheckResult(
            check_type=QualityCheckType.DUPLICATES,
            column=None,
            status=status,
            metric_value=dup_ratio,
            threshold=threshold,
            message=f"{dup_ratio:.1%} duplicate rows{subset_str} ({dup_count}/{len(df)})",
            details={"duplicate_count": dup_count, "subset": subset}
        ))
        
        return report
    
    def check_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: str = "iqr",
        threshold: Optional[float] = None
    ) -> QualityReport:
        """
        Check for outliers in numeric columns.
        
        Args:
            df: DataFrame to check.
            columns: Numeric columns to check. Defaults to all numeric.
            method: Detection method - 'iqr' or 'zscore'.
            threshold: Override default outlier threshold.
            
        Returns:
            QualityReport with outlier check results.
        """
        threshold = threshold or self.outlier_threshold
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        report = QualityReport(total_rows=len(df), total_columns=len(df.columns))
        
        for col in columns:
            if col not in df.columns:
                continue
            
            series = df[col].dropna()
            if len(series) == 0:
                continue
            
            if method == "iqr":
                Q1 = series.quantile(0.25)
                Q3 = series.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers = (series < lower) | (series > upper)
            elif method == "zscore":
                mean = series.mean()
                std = series.std()
                if std == 0:
                    outliers = pd.Series([False] * len(series))
                else:
                    z_scores = np.abs((series - mean) / std)
                    outliers = z_scores > 3
            else:
                raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'")
            
            outlier_count = outliers.sum()
            outlier_ratio = outlier_count / len(series)
            
            if outlier_ratio > threshold:
                status = QualityStatus.FAIL
            elif outlier_ratio > threshold / 2:
                status = QualityStatus.WARN
            else:
                status = QualityStatus.PASS
            
            report.checks.append(QualityCheckResult(
                check_type=QualityCheckType.OUTLIERS,
                column=col,
                status=status,
                metric_value=outlier_ratio,
                threshold=threshold,
                message=f"{outlier_ratio:.1%} outliers detected ({method} method)",
                details={"outlier_count": int(outlier_count), "method": method}
            ))
        
        return report
    
    def check_cardinality(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> QualityReport:
        """
        Check cardinality of columns for potential issues.
        
        Flags:
        - Very low cardinality (might be constant/near-constant)
        - Very high cardinality in object columns (might be IDs)
        
        Args:
            df: DataFrame to check.
            columns: Columns to check. Defaults to all.
            
        Returns:
            QualityReport with cardinality check results.
        """
        columns = columns or df.columns.tolist()
        report = QualityReport(total_rows=len(df), total_columns=len(df.columns))
        
        for col in columns:
            if col not in df.columns:
                continue
            
            series = df[col].dropna()
            unique_count = series.nunique()
            unique_ratio = unique_count / len(series) if len(series) > 0 else 0
            
            status = QualityStatus.PASS
            message = f"{unique_count} unique values ({unique_ratio:.1%})"
            
            # Check for constant/near-constant columns
            if unique_count <= 1:
                status = QualityStatus.WARN
                message = f"Constant column - only {unique_count} unique value(s)"
            
            # Check for potential ID columns in object dtype
            elif series.dtype == object and unique_ratio > self.high_cardinality_ratio:
                status = QualityStatus.WARN
                message = f"High cardinality ({unique_ratio:.1%}) - might be ID column"
            
            report.checks.append(QualityCheckResult(
                check_type=QualityCheckType.CARDINALITY,
                column=col,
                status=status,
                metric_value=unique_ratio,
                threshold=self.high_cardinality_ratio,
                message=message,
                details={"unique_count": unique_count}
            ))
        
        return report
    
    def full_report(
        self,
        df: pd.DataFrame,
        include_cardinality: bool = True
    ) -> QualityReport:
        """
        Run all quality checks and return comprehensive report.
        
        Args:
            df: DataFrame to check.
            include_cardinality: Whether to include cardinality checks.
            
        Returns:
            Complete QualityReport with all checks.
        """
        report = QualityReport(total_rows=len(df), total_columns=len(df.columns))
        
        # Run all checks
        null_report = self.check_nulls(df)
        dup_report = self.check_duplicates(df)
        outlier_report = self.check_outliers(df)
        
        report.checks.extend(null_report.checks)
        report.checks.extend(dup_report.checks)
        report.checks.extend(outlier_report.checks)
        
        if include_cardinality:
            card_report = self.check_cardinality(df)
            report.checks.extend(card_report.checks)
        
        logger.info(report.summary())
        
        return report
