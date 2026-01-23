"""Distribution drift detection between datasets."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from dataguard.utils.logging import get_logger

logger = get_logger(__name__)


class DriftMethod(Enum):
    """Methods for detecting distribution drift."""
    
    KS_TEST = "ks_test"           # Kolmogorov-Smirnov test (numeric)
    CHI_SQUARED = "chi_squared"   # Chi-squared test (categorical)
    PSI = "psi"                   # Population Stability Index
    JENSEN_SHANNON = "jensen_shannon"  # Jensen-Shannon divergence


class DriftStatus(Enum):
    """Drift severity levels."""
    
    NO_DRIFT = "no_drift"
    LOW_DRIFT = "low_drift"
    MEDIUM_DRIFT = "medium_drift"
    HIGH_DRIFT = "high_drift"


@dataclass
class ColumnDriftResult:
    """Drift detection result for a single column."""
    
    column: str
    method: DriftMethod
    statistic: float
    p_value: Optional[float]
    status: DriftStatus
    threshold: float
    message: str


@dataclass
class DriftReport:
    """Complete drift detection report."""
    
    columns: Dict[str, ColumnDriftResult] = field(default_factory=dict)
    reference_rows: int = 0
    current_rows: int = 0
    
    @property
    def has_drift(self) -> bool:
        """Returns True if any significant drift detected."""
        return any(
            c.status in (DriftStatus.MEDIUM_DRIFT, DriftStatus.HIGH_DRIFT)
            for c in self.columns.values()
        )
    
    @property
    def drifted_columns(self) -> List[str]:
        """List of columns with significant drift."""
        return [
            name for name, result in self.columns.items()
            if result.status in (DriftStatus.MEDIUM_DRIFT, DriftStatus.HIGH_DRIFT)
        ]
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        drifted = self.drifted_columns
        
        if not drifted:
            return f"✅ No significant drift detected across {len(self.columns)} columns"
        
        lines = [
            f"⚠️  Drift detected in {len(drifted)}/{len(self.columns)} columns:",
        ]
        
        for col_name in drifted:
            result = self.columns[col_name]
            icon = "🔴" if result.status == DriftStatus.HIGH_DRIFT else "🟡"
            lines.append(f"  {icon} {col_name}: {result.message}")
        
        return "\n".join(lines)


class DriftDetector:
    """
    Detect distribution drift between reference and current datasets.
    
    Supports multiple detection methods:
    - KS Test: For numeric columns (Kolmogorov-Smirnov test)
    - Chi-Squared: For categorical columns
    - PSI: Population Stability Index (works for both)
    - Jensen-Shannon: Divergence metric
    
    Example:
        >>> detector = DriftDetector()
        >>> detector.fit(reference_df)
        >>> report = detector.detect(current_df)
        >>> print(report.summary())
        
        >>> if report.has_drift:
        ...     print(f"Drifted columns: {report.drifted_columns}")
    """
    
    # PSI thresholds (industry standard)
    PSI_LOW = 0.1
    PSI_MEDIUM = 0.2
    
    # P-value thresholds for statistical tests
    P_VALUE_THRESHOLD = 0.05
    
    def __init__(
        self,
        method: DriftMethod = DriftMethod.KS_TEST,
        categorical_method: DriftMethod = DriftMethod.CHI_SQUARED,
        threshold: float = 0.05,
        n_bins: int = 10
    ):
        """
        Initialize drift detector.
        
        Args:
            method: Detection method for numeric columns.
            categorical_method: Detection method for categorical columns.
            threshold: P-value threshold for significance.
            n_bins: Number of bins for PSI calculation.
        """
        self.method = method
        self.categorical_method = categorical_method
        self.threshold = threshold
        self.n_bins = n_bins
        
        self._reference: Optional[pd.DataFrame] = None
        self._reference_stats: Dict[str, Dict] = {}
    
    def fit(self, reference_df: pd.DataFrame) -> "DriftDetector":
        """
        Fit the detector on reference (training) data.
        
        Args:
            reference_df: The reference DataFrame to compare against.
            
        Returns:
            Self for method chaining.
        """
        self._reference = reference_df.copy()
        self._reference_stats = {}
        
        for col in reference_df.columns:
            series = reference_df[col].dropna()
            
            if len(series) == 0:
                continue
            
            stats_dict: Dict = {"dtype": str(series.dtype)}
            
            if np.issubdtype(series.dtype, np.number):
                stats_dict["type"] = "numeric"
                stats_dict["values"] = series.values
                
                # Pre-compute bins for PSI
                try:
                    _, bin_edges = pd.cut(series, bins=self.n_bins, retbins=True)
                    stats_dict["bin_edges"] = bin_edges
                except ValueError:
                    stats_dict["bin_edges"] = None
            else:
                stats_dict["type"] = "categorical"
                stats_dict["value_counts"] = series.value_counts(normalize=True)
            
            self._reference_stats[col] = stats_dict
        
        logger.info(f"Fitted drift detector on {len(self._reference_stats)} columns")
        return self
    
    def _compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bin_edges: Optional[np.ndarray] = None
    ) -> float:
        """Compute Population Stability Index."""
        if bin_edges is None:
            try:
                _, bin_edges = pd.cut(reference, bins=self.n_bins, retbins=True)
            except ValueError:
                return 0.0
        
        # Compute bin frequencies
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        cur_counts, _ = np.histogram(current, bins=bin_edges)
        
        # Normalize to proportions
        ref_props = ref_counts / len(reference)
        cur_props = cur_counts / len(current)
        
        # Avoid division by zero
        ref_props = np.where(ref_props == 0, 0.0001, ref_props)
        cur_props = np.where(cur_props == 0, 0.0001, cur_props)
        
        # PSI formula
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        
        return psi
    
    def _compute_categorical_psi(
        self,
        ref_counts: pd.Series,
        cur_counts: pd.Series
    ) -> float:
        """Compute PSI for categorical variables."""
        # Align categories
        all_categories = set(ref_counts.index) | set(cur_counts.index)
        
        ref_aligned = ref_counts.reindex(all_categories, fill_value=0.0001)
        cur_aligned = cur_counts.reindex(all_categories, fill_value=0.0001)
        
        # Normalize
        ref_props = ref_aligned / ref_aligned.sum()
        cur_props = cur_aligned / cur_aligned.sum()
        
        # PSI
        psi = np.sum((cur_props - ref_props) * np.log(cur_props / ref_props))
        
        return psi
    
    def _get_drift_status(
        self,
        statistic: float,
        p_value: Optional[float],
        method: DriftMethod
    ) -> DriftStatus:
        """Determine drift status based on test results."""
        if method == DriftMethod.PSI:
            if statistic < self.PSI_LOW:
                return DriftStatus.NO_DRIFT
            elif statistic < self.PSI_MEDIUM:
                return DriftStatus.LOW_DRIFT
            elif statistic < self.PSI_MEDIUM * 1.5:
                return DriftStatus.MEDIUM_DRIFT
            else:
                return DriftStatus.HIGH_DRIFT
        
        # For statistical tests, use p-value
        if p_value is not None:
            if p_value > self.threshold:
                return DriftStatus.NO_DRIFT
            elif p_value > self.threshold / 5:
                return DriftStatus.LOW_DRIFT
            elif p_value > self.threshold / 50:
                return DriftStatus.MEDIUM_DRIFT
            else:
                return DriftStatus.HIGH_DRIFT
        
        return DriftStatus.NO_DRIFT
    
    def detect_column(
        self,
        current_series: pd.Series,
        column_name: str
    ) -> Optional[ColumnDriftResult]:
        """
        Detect drift for a single column.
        
        Args:
            current_series: Current data for the column.
            column_name: Name of the column.
            
        Returns:
            ColumnDriftResult or None if column not in reference.
        """
        if column_name not in self._reference_stats:
            return None
        
        ref_stats = self._reference_stats[column_name]
        current = current_series.dropna()
        
        if len(current) == 0:
            return ColumnDriftResult(
                column=column_name,
                method=self.method,
                statistic=0.0,
                p_value=None,
                status=DriftStatus.NO_DRIFT,
                threshold=self.threshold,
                message="No data in current column"
            )
        
        if ref_stats["type"] == "numeric":
            return self._detect_numeric_drift(current.values, column_name, ref_stats)
        else:
            return self._detect_categorical_drift(current, column_name, ref_stats)
    
    def _detect_numeric_drift(
        self,
        current: np.ndarray,
        column_name: str,
        ref_stats: Dict
    ) -> ColumnDriftResult:
        """Detect drift in numeric column."""
        reference = ref_stats["values"]
        method = self.method
        
        statistic: float = 0.0
        p_value: Optional[float] = None
        
        if method == DriftMethod.KS_TEST:
            ks_stat, p_value = stats.ks_2samp(reference, current)
            statistic = ks_stat
            message = f"KS statistic: {ks_stat:.4f}, p-value: {p_value:.4f}"
        
        elif method == DriftMethod.PSI:
            psi = self._compute_psi(reference, current, ref_stats.get("bin_edges"))
            statistic = psi
            message = f"PSI: {psi:.4f}"
        
        elif method == DriftMethod.JENSEN_SHANNON:
            # Compute JS divergence using histograms
            bins = ref_stats.get("bin_edges")
            if bins is None:
                _, bins = np.histogram(np.concatenate([reference, current]), bins=self.n_bins)
            
            ref_hist, _ = np.histogram(reference, bins=bins, density=True)
            cur_hist, _ = np.histogram(current, bins=bins, density=True)
            
            # Normalize
            ref_hist = ref_hist / ref_hist.sum() if ref_hist.sum() > 0 else ref_hist
            cur_hist = cur_hist / cur_hist.sum() if cur_hist.sum() > 0 else cur_hist
            
            # Add small value to avoid log(0)
            ref_hist = np.where(ref_hist == 0, 1e-10, ref_hist)
            cur_hist = np.where(cur_hist == 0, 1e-10, cur_hist)
            
            m = 0.5 * (ref_hist + cur_hist)
            js = 0.5 * (stats.entropy(ref_hist, m) + stats.entropy(cur_hist, m))
            statistic = js
            message = f"JS divergence: {js:.4f}"
        
        else:
            raise ValueError(f"Unsupported method for numeric: {method}")
        
        status = self._get_drift_status(statistic, p_value, method)
        
        return ColumnDriftResult(
            column=column_name,
            method=method,
            statistic=statistic,
            p_value=p_value,
            status=status,
            threshold=self.threshold,
            message=message
        )
    
    def _detect_categorical_drift(
        self,
        current: pd.Series,
        column_name: str,
        ref_stats: Dict
    ) -> ColumnDriftResult:
        """Detect drift in categorical column."""
        ref_counts = ref_stats["value_counts"]
        cur_counts = current.value_counts(normalize=True)
        method = self.categorical_method
        
        statistic: float = 0.0
        p_value: Optional[float] = None
        
        if method == DriftMethod.CHI_SQUARED:
            # Align categories
            all_cats = list(set(ref_counts.index) | set(cur_counts.index))
            
            ref_freq = ref_counts.reindex(all_cats, fill_value=0)
            cur_freq = cur_counts.reindex(all_cats, fill_value=0)
            
            # Scale to counts for chi-squared
            cur_scaled = cur_freq * len(current)
            ref_scaled = ref_freq * len(current)  # Expected based on reference proportions
            
            # Filter out zero expected values
            mask = ref_scaled > 0
            if mask.sum() > 1:
                chi2, p_value = stats.chisquare(
                    cur_scaled[mask].values,
                    ref_scaled[mask].values
                )
                statistic = chi2
                message = f"Chi-squared: {chi2:.4f}, p-value: {p_value:.4f}"
            else:
                message = "Insufficient categories for chi-squared test"
        
        elif method == DriftMethod.PSI:
            psi = self._compute_categorical_psi(ref_counts, cur_counts)
            statistic = psi
            message = f"PSI: {psi:.4f}"
        
        else:
            raise ValueError(f"Unsupported method for categorical: {method}")
        
        status = self._get_drift_status(statistic, p_value, method)
        
        return ColumnDriftResult(
            column=column_name,
            method=method,
            statistic=statistic,
            p_value=p_value,
            status=status,
            threshold=self.threshold,
            message=message
        )
    
    def detect(
        self,
        current_df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> DriftReport:
        """
        Detect drift between reference and current DataFrame.
        
        Args:
            current_df: Current DataFrame to check for drift.
            columns: Specific columns to check. Defaults to all fitted columns.
            
        Returns:
            DriftReport with detection results.
            
        Raises:
            ValueError: If detector hasn't been fitted.
        """
        if self._reference is None:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        columns = columns or list(self._reference_stats.keys())
        
        report = DriftReport(
            reference_rows=len(self._reference),
            current_rows=len(current_df)
        )
        
        for col in columns:
            if col not in current_df.columns:
                logger.warning(f"Column '{col}' not in current DataFrame, skipping")
                continue
            
            result = self.detect_column(current_df[col], col)
            if result:
                report.columns[col] = result
        
        if report.has_drift:
            logger.warning(report.summary())
        else:
            logger.info(f"No significant drift detected in {len(report.columns)} columns")
        
        return report
