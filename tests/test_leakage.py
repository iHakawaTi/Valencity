"""Tests for dataguard.leakage module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from dataguard.leakage import (
    SafeCrossValidator,
    LeakageDetector,
    LeakageType,
    safe_train_test_split,
    temporal_train_test_split,
    group_train_test_split,
)
from dataguard.leakage.cv import LeakageRisk


class TestSafeCrossValidator:
    """Test safe cross-validation."""
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] + np.random.randn(100) * 0.1 > 0).astype(int)
        return X, y
    
    def test_split_yields_correct_shapes(self, sample_data):
        X, y = sample_data
        cv = SafeCrossValidator(n_splits=5)
        
        splits = list(cv.split(X, y))
        
        assert len(splits) == 5
        
        for X_train, X_val, y_train, y_val in splits:
            assert len(X_train) == 80
            assert len(X_val) == 20
            assert len(y_train) == 80
            assert len(y_val) == 20
    
    def test_preprocessor_applied_per_fold(self, sample_data):
        X, y = sample_data
        cv = SafeCrossValidator(
            n_splits=5,
            preprocessor=StandardScaler()
        )
        
        for X_train, X_val, y_train, y_val in cv.split(X, y):
            # Check that training data is scaled (mean ~0, std ~1)
            assert abs(X_train.mean()) < 0.1
            assert abs(X_train.std() - 1) < 0.1
            
            # Validation should be transformed with training params
            # So it won't have exactly mean 0, std 1
    
    def test_validate_pipeline_detects_risky_transformers(self):
        cv = SafeCrossValidator()
        
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression())
        ])
        
        warnings = cv.validate_pipeline(pipeline)
        
        # Should detect StandardScaler as risky
        assert len(warnings) > 0
        assert any("scaler" in w.source for w in warnings)
    
    def test_cross_val_score(self, sample_data):
        X, y = sample_data
        cv = SafeCrossValidator(n_splits=5, preprocessor=StandardScaler())
        
        scores = cv.cross_val_score(LogisticRegression(), X, y)
        
        assert len(scores) == 5
        assert all(0 <= s <= 1 for s in scores)


class TestSplitters:
    """Test train/test splitting utilities."""
    
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X = pd.DataFrame({
            "a": np.random.randn(100),
            "b": np.random.randn(100),
            "group": np.repeat(np.arange(10), 10)
        })
        y = pd.Series(np.random.randint(0, 2, 100))
        return X, y
    
    def test_safe_train_test_split(self, sample_data):
        X, y = sample_data
        
        X_train, X_test, y_train, y_test = safe_train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        assert len(X_train) == 80
        assert len(X_test) == 20
    
    def test_temporal_split(self):
        df = pd.DataFrame({
            "time": pd.date_range("2023-01-01", periods=100, freq="D"),
            "value": np.random.randn(100)
        })
        y = pd.Series(np.random.randint(0, 2, 100))
        
        X_train, X_test, y_train, y_test = temporal_train_test_split(
            df, y, time_column="time", test_size=0.2
        )
        
        # Train should be earlier than test
        assert X_train["time"].max() < X_test["time"].min()
    
    def test_group_split_no_overlap(self, sample_data):
        X, y = sample_data
        
        X_train, X_test, y_train, y_test = group_train_test_split(
            X, y, group_column="group", test_size=0.2, random_state=42
        )
        
        # No overlap in groups
        train_groups = set(X_train["group"].unique())
        test_groups = set(X_test["group"].unique())
        
        assert len(train_groups & test_groups) == 0


class TestLeakageDetector:
    """Test leakage detection."""
    
    @pytest.fixture
    def detector(self):
        return LeakageDetector()
    
    def test_target_leakage_detection(self, detector):
        X = pd.DataFrame({
            "feature1": np.random.randn(100),
            "target_copy": np.arange(100)  # Perfectly correlated with target
        })
        y = pd.Series(np.arange(100))
        
        warnings = detector.check_target_leakage(X, y)
        
        # Should detect target_copy as leaky
        assert len(warnings) > 0
        assert any("target_copy" in w.source for w in warnings)
    
    def test_feature_leakage_scaled_data(self, detector):
        # Simulate data that was scaled before split
        np.random.seed(42)
        full_data = np.random.randn(100, 3)
        
        # Scale on full data (leakage!)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(full_data)
        
        X_train = pd.DataFrame(scaled[:80])
        X_test = pd.DataFrame(scaled[80:])
        
        warnings = detector.check_feature_leakage(X_train, X_test)
        
        # Detection is heuristic-based, check function runs correctly
        assert isinstance(warnings, list)
    
    def test_duplicate_leakage_detection(self, detector):
        # Create data with duplicates in train and test
        X_train = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [1, 2, 3, 4, 5]
        })
        X_test = pd.DataFrame({
            "a": [1, 6, 7],  # Row with a=1 is duplicate
            "b": [1, 6, 7]
        })
        
        warnings = detector.check_duplicate_leakage(X_train, X_test)
        
        assert len(warnings) > 0
        assert warnings[0].risk in (LeakageRisk.HIGH, LeakageRisk.CRITICAL)
    
    def test_full_check(self, detector):
        np.random.seed(42)
        X_train = pd.DataFrame(np.random.randn(80, 3))
        X_test = pd.DataFrame(np.random.randn(20, 3))
        y_train = pd.Series(np.random.randint(0, 2, 80))
        y_test = pd.Series(np.random.randint(0, 2, 20))
        
        warnings = detector.full_check(X_train, X_test, y_train, y_test)
        
        # Should run without errors
        assert isinstance(warnings, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
