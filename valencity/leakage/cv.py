"""Safe cross-validation wrappers that prevent data leakage."""

from collections.abc import Generator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold

from valencity.utils.logging import get_logger

logger = get_logger(__name__)


class LeakageRisk(Enum):
    """Risk levels for potential leakage."""
    
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LeakageWarning:
    """Warning about potential data leakage."""
    
    risk: LeakageRisk
    source: str
    message: str
    suggestion: str
    
    def __str__(self) -> str:
        icon = {
            LeakageRisk.LOW: "ℹ️",
            LeakageRisk.MEDIUM: "⚠️",
            LeakageRisk.HIGH: "🔴",
            LeakageRisk.CRITICAL: "🚨",
        }[self.risk]
        return f"{icon} [{self.risk.value.upper()}] {self.source}: {self.message}"


class SafeCrossValidator:
    """
    Cross-validation wrapper that prevents common data leakage patterns.
    
    This wrapper ensures that preprocessing steps (scaling, imputation, etc.)
    are fit only on training data within each fold, preventing information
    from the validation set from leaking into the training process.
    
    Common leakage patterns prevented:
    - Fitting scalers/normalizers on full dataset before CV
    - Computing statistics (mean, std) on full dataset
    - Target encoding using full dataset
    - Feature selection using full dataset
    
    Example:
        >>> from sklearn.preprocessing import StandardScaler
        >>> from sklearn.linear_model import LogisticRegression
        >>> 
        >>> cv = SafeCrossValidator(
        ...     cv=KFold(n_splits=5),
        ...     preprocessor=StandardScaler()
        ... )
        >>> 
        >>> # Get safe splits with preprocessing applied correctly
        >>> for X_train, X_val, y_train, y_val in cv.split(X, y):
        ...     model.fit(X_train, y_train)
        ...     score = model.score(X_val, y_val)
        
        >>> # Or validate an existing pipeline for leakage risks
        >>> warnings = cv.validate_pipeline(my_pipeline)
    """
    
    # Steps that commonly cause leakage when fit on full data
    RISKY_TRANSFORMERS = {
        "StandardScaler": "Fits mean/std on full data - should be per-fold",
        "MinMaxScaler": "Fits min/max on full data - should be per-fold",
        "RobustScaler": "Fits median/IQR on full data - should be per-fold",
        "Normalizer": "May use global statistics",
        "SimpleImputer": "Fits fill values on full data - should be per-fold",
        "KNNImputer": "Uses neighbors from full data - should be per-fold",
        "TargetEncoder": "Uses target info from full data - high leakage risk",
        "SelectKBest": "Selects features using full data - should be per-fold",
        "SelectFromModel": "Fits model on full data for selection",
        "RFE": "Recursive elimination on full data - should be per-fold",
        "PCA": "Fits components on full data - should be per-fold",
        "TruncatedSVD": "Fits on full data - should be per-fold",
    }
    
    def __init__(
        self,
        cv: Optional[BaseCrossValidator] = None,
        preprocessor: Optional[TransformerMixin] = None,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        stratify: bool = False
    ):
        """
        Initialize safe cross-validator.
        
        Args:
            cv: Sklearn cross-validator. Defaults to KFold.
            preprocessor: Optional preprocessor to apply per-fold.
            n_splits: Number of folds if cv not provided.
            shuffle: Whether to shuffle data.
            random_state: Random seed for reproducibility.
            stratify: Whether to use stratified splits (for classification).
        """
        if cv is not None:
            self.cv = cv
        elif stratify:
            self.cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state
            )
        else:
            self.cv = KFold(
                n_splits=n_splits,
                shuffle=shuffle,
                random_state=random_state
            )
        
        self.preprocessor = preprocessor
        self._warnings: List[LeakageWarning] = []
    
    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], None, None]:
        """
        Generate safe train/validation splits with preprocessing.
        
        Unlike standard CV, this ensures preprocessing is fit only on
        training data within each fold.
        
        Args:
            X: Feature matrix.
            y: Target vector.
            groups: Group labels for group-based CV.
            
        Yields:
            Tuple of (X_train, X_val, y_train, y_val) for each fold.
        """
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = y.values if isinstance(y, pd.Series) else y
        
        for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X_arr, y_arr, groups)):
            X_train = X_arr[train_idx]
            X_val = X_arr[val_idx]
            y_train = y_arr[train_idx] if y_arr is not None else None
            y_val = y_arr[val_idx] if y_arr is not None else None
            
            # Apply preprocessing per-fold (fit on train, transform both)
            if self.preprocessor is not None:
                preprocessor = clone(self.preprocessor)
                X_train = preprocessor.fit_transform(X_train)
                X_val = preprocessor.transform(X_val)
                
                logger.debug(f"Fold {fold_idx + 1}: Fitted preprocessor on {len(train_idx)} samples")
            
            yield X_train, X_val, y_train, y_val
    
    def get_n_splits(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None
    ) -> int:
        """Get number of splits."""
        return self.cv.get_n_splits(X, y, groups)
    
    def validate_pipeline(
        self,
        pipeline: Any,
        strict: bool = False
    ) -> List[LeakageWarning]:
        """
        Validate a sklearn Pipeline for potential leakage risks.
        
        Checks if the pipeline contains transformers that could cause
        data leakage if the pipeline is fit before cross-validation.
        
        Args:
            pipeline: Sklearn Pipeline or similar.
            strict: If True, treat all warnings as high risk.
            
        Returns:
            List of LeakageWarning objects.
        """
        warnings: List[LeakageWarning] = []
        
        # Get pipeline steps
        steps = []
        if hasattr(pipeline, 'steps'):
            steps = pipeline.steps
        elif hasattr(pipeline, 'named_steps'):
            steps = list(pipeline.named_steps.items())
        
        for name, transformer in steps:
            class_name = transformer.__class__.__name__
            
            if class_name in self.RISKY_TRANSFORMERS:
                risk = LeakageRisk.HIGH if strict else LeakageRisk.MEDIUM
                warnings.append(LeakageWarning(
                    risk=risk,
                    source=f"Pipeline step '{name}'",
                    message=self.RISKY_TRANSFORMERS[class_name],
                    suggestion="Use SafeCrossValidator or fit pipeline inside CV loop"
                ))
        
        # Check for target-related leakage
        for name, transformer in steps:
            if "target" in name.lower() or "label" in name.lower():
                warnings.append(LeakageWarning(
                    risk=LeakageRisk.CRITICAL,
                    source=f"Pipeline step '{name}'",
                    message="Target-related transformer detected - high leakage risk",
                    suggestion="Ensure target encoding is done per-fold, not on full data"
                ))
        
        if warnings:
            logger.warning(f"Found {len(warnings)} potential leakage risks in pipeline")
            for w in warnings:
                logger.warning(str(w))
        else:
            logger.info("No obvious leakage risks found in pipeline")
        
        return warnings
    
    def cross_val_score(
        self,
        estimator: BaseEstimator,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        scoring: Optional[Callable] = None
    ) -> np.ndarray:
        """
        Perform cross-validation with safe preprocessing.
        
        Args:
            estimator: Sklearn estimator to evaluate.
            X: Feature matrix.
            y: Target vector.
            scoring: Optional scoring function. Defaults to estimator.score().
            
        Returns:
            Array of scores for each fold.
        """
        scores = []
        
        for fold_idx, (X_train, X_val, y_train, y_val) in enumerate(self.split(X, y)):
            est = clone(estimator)
            est.fit(X_train, y_train)
            
            if scoring is not None:
                score = scoring(y_val, est.predict(X_val))
            else:
                score = est.score(X_val, y_val)
            
            scores.append(score)
            logger.debug(f"Fold {fold_idx + 1} score: {score:.4f}")
        
        scores_arr = np.array(scores)
        logger.info(f"CV scores: {scores_arr.mean():.4f} (+/- {scores_arr.std() * 2:.4f})")
        
        return scores_arr
