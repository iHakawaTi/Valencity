"""Data leakage detection utilities."""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from dataguard.leakage.cv import LeakageRisk, LeakageWarning
from dataguard.utils.logging import get_logger

logger = get_logger(__name__)


class LeakageType(Enum):
    """Types of data leakage."""
    
    TARGET_LEAKAGE = "target_leakage"       # Feature derived from target
    TRAIN_TEST_LEAKAGE = "train_test"       # Train info in test set
    TEMPORAL_LEAKAGE = "temporal"           # Future info in past
    DUPLICATE_LEAKAGE = "duplicate"         # Same rows in train/test
    FEATURE_LEAKAGE = "feature"             # Suspicious feature correlations


class LeakageDetector:
    """
    Detect potential data leakage in ML pipelines.
    
    Checks for common leakage patterns:
    - Target leakage: Features that directly encode the target
    - Train/test leakage: Overlapping samples between sets
    - Temporal leakage: Future information leaking to past
    - Duplicate leakage: Same rows in train and test
    
    Example:
        >>> detector = LeakageDetector()
        >>> 
        >>> # Check for target leakage
        >>> warnings = detector.check_target_leakage(X, y)
        >>> 
        >>> # Check for train/test overlap
        >>> warnings = detector.check_feature_leakage(X_train, X_test)
        >>> 
        >>> # Full pipeline check
        >>> all_warnings = detector.full_check(X_train, X_test, y_train, y_test)
    """
    
    # High correlation threshold for target leakage detection
    TARGET_CORR_THRESHOLD = 0.95
    
    # Threshold for suspicious train/test similarity
    SIMILARITY_THRESHOLD = 0.99
    
    def __init__(
        self,
        target_corr_threshold: float = 0.95,
        similarity_threshold: float = 0.99
    ):
        """
        Initialize leakage detector.
        
        Args:
            target_corr_threshold: Correlation threshold for target leakage.
            similarity_threshold: Similarity threshold for feature leakage.
        """
        self.target_corr_threshold = target_corr_threshold
        self.similarity_threshold = similarity_threshold
    
    def check_target_leakage(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        feature_names: Optional[List[str]] = None
    ) -> List[LeakageWarning]:
        """
        Check for features that might encode the target directly.
        
        Flags features that have suspiciously high correlation with
        the target, which often indicates they were derived from it.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            feature_names: Optional feature names.
            
        Returns:
            List of LeakageWarning objects.
        """
        warnings: List[LeakageWarning] = []
        
        # Convert to DataFrame for easier handling
        if isinstance(X, np.ndarray):
            if feature_names:
                X = pd.DataFrame(X, columns=feature_names)
            else:
                X = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        # Check correlation with target for numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            try:
                # Handle NaN values
                mask = ~(X[col].isna() | y.isna())
                if mask.sum() < 10:
                    continue
                
                corr, _ = stats.pearsonr(X[col][mask], y[mask])
                
                if abs(corr) > self.target_corr_threshold:
                    warnings.append(LeakageWarning(
                        risk=LeakageRisk.CRITICAL,
                        source=f"Feature '{col}'",
                        message=f"Correlation with target: {corr:.4f}",
                        suggestion="This feature may be derived from the target. Remove or investigate."
                    ))
                elif abs(corr) > self.target_corr_threshold * 0.9:
                    warnings.append(LeakageWarning(
                        risk=LeakageRisk.HIGH,
                        source=f"Feature '{col}'",
                        message=f"High correlation with target: {corr:.4f}",
                        suggestion="Investigate if this feature encodes target information."
                    ))
            except Exception as e:
                logger.debug(f"Could not compute correlation for {col}: {e}")
        
        # Check for target-like column names
        target_keywords = ["target", "label", "y_", "outcome", "result", "answer"]
        for col in X.columns:
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in target_keywords):
                warnings.append(LeakageWarning(
                    risk=LeakageRisk.MEDIUM,
                    source=f"Feature '{col}'",
                    message="Column name suggests it may contain target information",
                    suggestion="Verify this feature doesn't encode the target."
                ))
        
        if warnings:
            logger.warning(f"Found {len(warnings)} potential target leakage issues")
        else:
            logger.info("No obvious target leakage detected")
        
        return warnings
    
    def check_feature_leakage(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        X_test: Union[np.ndarray, pd.DataFrame]
    ) -> List[LeakageWarning]:
        """
        Check for suspicious similarities between train and test features.
        
        Flags issues like:
        - Identical feature distributions (might indicate scaling on full data)
        - Same min/max values (might indicate normalization on full data)
        
        Args:
            X_train: Training features.
            X_test: Test features.
            
        Returns:
            List of LeakageWarning objects.
        """
        warnings: List[LeakageWarning] = []
        
        # Convert to DataFrames
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)
        
        # Check for suspiciously similar statistics
        for col in X_train.columns:
            if col not in X_test.columns:
                continue
            
            train_col = X_train[col].dropna()
            test_col = X_test[col].dropna()
            
            if len(train_col) == 0 or len(test_col) == 0:
                continue
            
            if not np.issubdtype(train_col.dtype, np.number):
                continue
            
            # Check if min/max are suspiciously identical
            train_min, train_max = train_col.min(), train_col.max()
            test_min, test_max = test_col.min(), test_col.max()
            
            if train_min == test_min == 0 and train_max == test_max == 1:
                warnings.append(LeakageWarning(
                    risk=LeakageRisk.HIGH,
                    source=f"Feature '{col}'",
                    message="Both train and test have exact [0,1] range",
                    suggestion="MinMax scaling may have been applied to full data before split"
                ))
            
            # Check for identical mean/std (suggests StandardScaler on full data)
            train_mean, train_std = train_col.mean(), train_col.std()
            test_mean, test_std = test_col.mean(), test_col.std()
            
            if (abs(train_mean) < 0.01 and abs(test_mean) < 0.01 and 
                abs(train_std - 1) < 0.01 and abs(test_std - 1) < 0.01):
                warnings.append(LeakageWarning(
                    risk=LeakageRisk.MEDIUM,
                    source=f"Feature '{col}'",
                    message="Both train and test have mean≈0, std≈1",
                    suggestion="StandardScaler may have been applied to full data before split"
                ))
        
        if warnings:
            logger.warning(f"Found {len(warnings)} potential feature leakage issues")
        
        return warnings
    
    def check_duplicate_leakage(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        X_test: Union[np.ndarray, pd.DataFrame],
        y_train: Optional[Union[np.ndarray, pd.Series]] = None,
        y_test: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> List[LeakageWarning]:
        """
        Check for duplicate rows between train and test sets.
        
        Same samples in both sets is a form of leakage that leads
        to overoptimistic performance estimates.
        
        Args:
            X_train: Training features.
            X_test: Test features.
            y_train: Optional training targets.
            y_test: Optional test targets.
            
        Returns:
            List of LeakageWarning objects.
        """
        warnings: List[LeakageWarning] = []
        
        # Convert to DataFrames
        if isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train)
            X_test = pd.DataFrame(X_test)
        
        # Find duplicate rows
        train_tuples = set(map(tuple, X_train.values))
        test_tuples = set(map(tuple, X_test.values))
        
        duplicates = train_tuples & test_tuples
        
        if duplicates:
            dup_ratio = len(duplicates) / len(test_tuples)
            
            risk = LeakageRisk.CRITICAL if dup_ratio > 0.1 else LeakageRisk.HIGH
            
            warnings.append(LeakageWarning(
                risk=risk,
                source="Train/Test Split",
                message=f"{len(duplicates)} duplicate rows ({dup_ratio:.1%} of test set)",
                suggestion="Remove duplicates or use group-based splitting"
            ))
        
        if warnings:
            logger.warning(f"Found {len(duplicates)} duplicate rows between train and test")
        else:
            logger.info("No duplicate rows between train and test")
        
        return warnings
    
    def check_temporal_leakage(
        self,
        df: pd.DataFrame,
        time_column: str,
        target_column: str,
        feature_columns: Optional[List[str]] = None
    ) -> List[LeakageWarning]:
        """
        Check for temporal leakage in time-series data.
        
        Flags features that might contain future information,
        such as rolling statistics computed incorrectly.
        
        Args:
            df: DataFrame with time and feature columns.
            time_column: Name of timestamp column.
            target_column: Name of target column.
            feature_columns: Feature columns to check.
            
        Returns:
            List of LeakageWarning objects.
        """
        warnings: List[LeakageWarning] = []
        
        if time_column not in df.columns:
            raise ValueError(f"Time column '{time_column}' not found")
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        if feature_columns is None:
            feature_columns = [c for c in df.columns if c not in [time_column, target_column]]
        
        # Sort by time
        df_sorted = df.sort_values(time_column).reset_index(drop=True)
        
        # Check for features with suspiciously high correlation with future target
        target = df_sorted[target_column]
        
        for col in feature_columns:
            if col not in df_sorted.columns:
                continue
            
            feature = df_sorted[col]
            
            if not np.issubdtype(feature.dtype, np.number):
                continue
            
            try:
                # Check correlation with future target (shifted)
                future_target = target.shift(-1).dropna()
                feature_aligned = feature.iloc[:-1]
                
                if len(future_target) < 10:
                    continue
                
                corr, _ = stats.pearsonr(feature_aligned, future_target)
                
                if abs(corr) > self.target_corr_threshold:
                    warnings.append(LeakageWarning(
                        risk=LeakageRisk.CRITICAL,
                        source=f"Feature '{col}'",
                        message=f"High correlation with FUTURE target: {corr:.4f}",
                        suggestion="This feature may contain future information. Check computation."
                    ))
            except Exception as e:
                logger.debug(f"Could not check temporal leakage for {col}: {e}")
        
        # Check for keyword patterns suggesting future info
        future_keywords = ["future", "next", "lead", "forward", "_t+"]
        for col in feature_columns:
            col_lower = str(col).lower()
            if any(kw in col_lower for kw in future_keywords):
                warnings.append(LeakageWarning(
                    risk=LeakageRisk.HIGH,
                    source=f"Feature '{col}'",
                    message="Column name suggests future information",
                    suggestion="Verify this feature doesn't contain future data."
                ))
        
        if warnings:
            logger.warning(f"Found {len(warnings)} potential temporal leakage issues")
        else:
            logger.info("No obvious temporal leakage detected")
        
        return warnings
    
    def full_check(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        X_test: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        y_test: Union[np.ndarray, pd.Series]
    ) -> List[LeakageWarning]:
        """
        Run all leakage checks.
        
        Args:
            X_train: Training features.
            X_test: Test features.
            y_train: Training target.
            y_test: Test target.
            
        Returns:
            Combined list of all warnings.
        """
        all_warnings: List[LeakageWarning] = []
        
        # Target leakage (on training data)
        all_warnings.extend(self.check_target_leakage(X_train, y_train))
        
        # Feature leakage
        all_warnings.extend(self.check_feature_leakage(X_train, X_test))
        
        # Duplicate leakage
        all_warnings.extend(self.check_duplicate_leakage(X_train, X_test, y_train, y_test))
        
        logger.info(f"Full leakage check complete: {len(all_warnings)} warnings")
        
        return all_warnings
