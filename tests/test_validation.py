"""Tests for dataguard.validation module."""

import numpy as np
import pandas as pd
import pytest

from dataguard.validation import (
    DataSchema, ColumnSpec, DataType, ValidationResult,
    DataQualityChecker, QualityStatus,
    DriftDetector, DriftMethod, DriftStatus
)


class TestDataSchema:
    """Test schema validation."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "score": [0.8, 0.9, 0.7]
        })
    
    def test_valid_schema_passes(self, sample_df):
        schema = DataSchema([
            ColumnSpec("id", DataType.INTEGER),
            ColumnSpec("name", DataType.STRING),
            ColumnSpec("age", DataType.INTEGER),
        ])
        
        result = schema.validate(sample_df)
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_missing_column_fails(self, sample_df):
        schema = DataSchema([
            ColumnSpec("id", DataType.INTEGER),
            ColumnSpec("missing_col", DataType.STRING),
        ])
        
        result = schema.validate(sample_df)
        assert not result.is_valid
        assert "missing_col" in result.missing_columns
    
    def test_nullable_constraint(self):
        df = pd.DataFrame({"col": [1, None, 3]})
        schema = DataSchema([
            ColumnSpec("col", DataType.FLOAT, nullable=False)
        ])
        
        result = schema.validate(df)
        assert not result.is_valid
    
    def test_unique_constraint(self):
        df = pd.DataFrame({"col": [1, 1, 2]})
        schema = DataSchema([
            ColumnSpec("col", DataType.INTEGER, unique=True)
        ])
        
        result = schema.validate(df)
        assert not result.is_valid
    
    def test_min_max_constraint(self):
        df = pd.DataFrame({"age": [25, -5, 200]})
        schema = DataSchema([
            ColumnSpec("age", DataType.INTEGER, min_value=0, max_value=150)
        ])
        
        result = schema.validate(df)
        assert not result.is_valid
    
    def test_from_dataframe_inference(self, sample_df):
        schema = DataSchema.from_dataframe(sample_df)
        
        assert len(schema.columns) == 4
        assert "id" in schema.columns
        assert schema.columns["id"].dtype == DataType.INTEGER


class TestDataQualityChecker:
    """Test data quality checking."""
    
    @pytest.fixture
    def checker(self):
        return DataQualityChecker()
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [1, None, None, 4, 5],  # 40% null
            "c": ["x", "x", "y", "y", "z"]
        })
    
    def test_check_nulls_detects_high_nulls(self, checker, sample_df):
        report = checker.check_nulls(sample_df, threshold=0.3)
        
        # Column 'b' has 40% nulls, should fail
        b_check = next(c for c in report.checks if c.column == "b")
        assert b_check.status == QualityStatus.FAIL
    
    def test_check_duplicates(self, checker):
        df = pd.DataFrame({
            "a": [1, 1, 2, 3],
            "b": [1, 1, 2, 3]
        })
        
        report = checker.check_duplicates(df, threshold=0.1)
        
        # Should detect duplicates
        assert any(c.status != QualityStatus.PASS for c in report.checks)
    
    def test_check_outliers_iqr(self, checker):
        df = pd.DataFrame({
            "values": [1, 2, 3, 4, 5, 100]  # 100 is outlier
        })
        
        report = checker.check_outliers(df, method="iqr")
        
        values_check = next(c for c in report.checks if c.column == "values")
        assert values_check.details["outlier_count"] > 0
    
    def test_full_report(self, checker, sample_df):
        report = checker.full_report(sample_df)
        
        assert report.total_rows == 5
        assert report.total_columns == 3
        assert len(report.checks) > 0


class TestDriftDetector:
    """Test distribution drift detection."""
    
    @pytest.fixture
    def reference_df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "numeric": np.random.normal(0, 1, 1000),
            "category": np.random.choice(["a", "b", "c"], 1000)
        })
    
    def test_no_drift_same_distribution(self, reference_df):
        detector = DriftDetector()
        detector.fit(reference_df)
        
        # Same distribution should not drift
        np.random.seed(43)
        current = pd.DataFrame({
            "numeric": np.random.normal(0, 1, 500),
            "category": np.random.choice(["a", "b", "c"], 500)
        })
        
        report = detector.detect(current)
        
        # Most columns should not show significant drift
        assert not report.has_drift or len(report.drifted_columns) == 0
    
    def test_drift_different_distribution(self, reference_df):
        detector = DriftDetector()
        detector.fit(reference_df)
        
        # Very different distribution should drift
        current = pd.DataFrame({
            "numeric": np.random.normal(10, 1, 500),  # Mean shifted
            "category": np.random.choice(["a", "b", "c"], 500)
        })
        
        report = detector.detect(current)
        
        # Numeric column should show drift
        assert "numeric" in report.drifted_columns
    
    def test_psi_method(self, reference_df):
        detector = DriftDetector(method=DriftMethod.PSI)
        detector.fit(reference_df)
        
        # Shifted distribution
        current = pd.DataFrame({
            "numeric": np.random.normal(5, 2, 500),
            "category": np.random.choice(["a", "b", "c"], 500)
        })
        
        report = detector.detect(current)
        
        # Check that numeric column was flagged
        numeric_result = report.columns.get("numeric")
        assert numeric_result is not None
        assert numeric_result.method == DriftMethod.PSI


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
