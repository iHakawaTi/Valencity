"""
Data Profiling module for automated statistical analysis.
"""

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from valencity.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ColumnProfile:
    """Statistical profile of a single column."""
    name: str
    dtype: str
    total_count: int
    missing_count: int
    missing_percentage: float
    unique_count: int
    unique_percentage: float
    
    # Numeric stats
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    median: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    
    # Distribution
    distribution_type: str = "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ProfileReport:
    """Comprehensive data profile report."""
    total_rows: int
    total_columns: int
    columns: Dict[str, ColumnProfile]
    correlations: Optional[Dict[str, Dict[str, float]]] = None
    
    def summary(self) -> str:
        """Generate a text summary of the profile."""
        lines = [
            "Data Profile Report",
            "-------------------",
            f"Rows:    {self.total_rows}",
            f"Columns: {self.total_columns}",
            "",
            "Column Summaries:",
        ]
        
        for col in self.columns.values():
            lines.append(f"\n[{col.name}] ({col.dtype})")
            lines.append(f"  Missing: {col.missing_count} ({col.missing_percentage:.1f}%)")
            lines.append(f"  Unique:  {col.unique_count} ({col.unique_percentage:.1f}%)")
            
            if col.mean is not None:
                lines.append(f"  Stats:   mean={col.mean:.2f}, std={col.std:.2f}, range=[{col.min_val:.2f}, {col.max_val:.2f}]")
                lines.append(f"  Dist:    {col.distribution_type}")
                
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """Export profile to JSON string."""
        return json.dumps({
            "total_rows": self.total_rows,
            "total_columns": self.total_columns,
            "columns": {k: v.to_dict() for k, v in self.columns.items()},
            "correlations": self.correlations
        }, indent=2, default=str)


class DataProfiler:
    """
    Automated data profiler.
    
    Calculates statistics, detects distributions, and analyzes correlations.
    """
    
    def __init__(self, correlation_threshold: float = 0.5):
        self.correlation_threshold = correlation_threshold
        
    def profile(self, df: pd.DataFrame) -> ProfileReport:
        """
        Generate a profile report for the DataFrame.
        
        Args:
            df: DataFrame to profile.
            
        Returns:
            ProfileReport object.
            
        Raises:
            ValueError: If input is not a DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
             raise ValueError(f"Expected pandas DataFrame, got {type(df)}")
             
        columns = {}
        total_rows = len(df)
        
        if total_rows == 0:
             # Return empty report
             return ProfileReport(0, len(df.columns), {})
        
        for col_name in df.columns:
            series = df[col_name]
            stats_dict = self._get_column_stats(series)
            
            profile = ColumnProfile(
                name=col_name,
                dtype=str(series.dtype),
                total_count=total_rows,
                **stats_dict
            )
            columns[col_name] = profile
            
        # Calculate correlations for numeric columns
        correlations = self._get_correlations(df)
        
        logger.info(f"Profiled {len(columns)} columns")
        return ProfileReport(
            total_rows=total_rows,
            total_columns=len(columns),
            columns=columns,
            correlations=correlations
        )
    
    def _get_column_stats(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate statistics for a single column."""
        missing = series.isna().sum()
        total = len(series)
        
        stats_dict = {
            "missing_count": missing,
            "missing_percentage": (missing / total) * 100 if total > 0 else 0,
            "unique_count": series.nunique(),
            "unique_percentage": (series.nunique() / total) * 100 if total > 0 else 0,
        }
        
        # Numeric stats
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if not clean.empty:
                stats_dict.update({
                    "mean": float(clean.mean()),
                    "std": float(clean.std()),
                    "min_val": float(clean.min()),
                    "max_val": float(clean.max()),
                    "median": float(clean.median()),
                    "skewness": float(clean.skew()),
                    "kurtosis": float(clean.kurtosis()),
                    "distribution_type": self._detect_distribution(clean)
                })
        
        return stats_dict
    
    def _detect_distribution(self, data: pd.Series) -> str:
        """
        Detect best fitting distribution using Kolmogorov-Smirnov test.
        
        Checks: Normal, Uniform, Log-Normal, Exponential.
        """
        if len(data) < 50:  # Not enough data for reliable test
            return "unknown (insufficient data)"
            
        distributions = {
            "normal": stats.norm,
            "uniform": stats.uniform,
            "lognorm": stats.lognorm,
            "expon": stats.expon
        }
        
        best_dist = "unknown"
        best_p = 0.0
        
        # Normalize data for comparison
        y = (data - data.mean()) / (data.std() + 1e-9) # Avoid div by zero
        
        for name, dist in distributions.items():
            try:
                # Estimate parameters
                params = dist.fit(y)
                # Perform KS test
                _, p_value = stats.kstest(y, name, args=params)
                
                # If p-value > 0.05, we cannot reject the null hypothesis
                # (that it came from this distribution)
                if p_value > 0.05 and p_value > best_p:
                    best_p = p_value
                    best_dist = name
            except Exception:
                continue
                
        return best_dist
    
    def _get_correlations(self, df: pd.DataFrame) -> Optional[Dict[str, Dict[str, float]]]:
        """Calculate correlation matrix for numeric columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.shape[1] < 2:
            return None
            
        corr_matrix = numeric_df.corr(method="pearson").round(4).to_dict()
        
        # Filter diagonals and low correlations to save space if needed
        # For now return full matrix structure
        return corr_matrix
