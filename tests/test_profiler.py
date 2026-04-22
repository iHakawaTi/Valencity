"""Tests for valencity.validation.profiler module."""

import json

import numpy as np
import pandas as pd
import pytest

from valencity.validation import DataProfiler


class TestDataProfiler:
    """Test data profiling functionality."""
    
    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "normal": np.random.normal(0, 1, 100),
            "uniform": np.random.uniform(0, 1, 100),
            "category": ["A", "B", "A", "C"] * 25,
            "missing": [1, 2, None, 4] * 25
        })
        
    def test_profile_basic_stats(self, sample_df):
        profiler = DataProfiler()
        report = profiler.profile(sample_df)
        
        assert report.total_rows == 100
        assert report.total_columns == 4
        
        # Check numeric column
        norm_prof = report.columns["normal"]
        assert norm_prof.missing_count == 0
        assert -0.5 < norm_prof.mean < 0.5  # Should be close to 0
        
        # Check categorical column
        cat_prof = report.columns["category"]
        assert cat_prof.unique_count == 3
        assert cat_prof.mean is None
        
        # Check missing column
        miss_prof = report.columns["missing"]
        assert miss_prof.missing_count == 25
        assert miss_prof.total_count == 100
        
    def test_distribution_detection(self, sample_df):
        profiler = DataProfiler()
        report = profiler.profile(sample_df)
        
        # Normal distribution should likely identify as normal or unknown (if sample small/noisy)
        # But specifically checking it runs without error and returns string
        assert isinstance(report.columns["normal"].distribution_type, str)
        
        # Uniform
        assert isinstance(report.columns["uniform"].distribution_type, str)
        
    def test_correlations(self, sample_df):
        profiler = DataProfiler()
        report = profiler.profile(sample_df)
        
        assert report.correlations is not None
        # Should have correlations for numeric columns: normal, uniform, missing
        assert "normal" in report.correlations
        assert "uniform" in report.correlations
        
        # Self correlation is 1.0
        assert report.correlations["normal"]["normal"] == pytest.approx(1.0)
        
    def test_json_export(self, sample_df):
        profiler = DataProfiler()
        report = profiler.profile(sample_df)
        
        json_str = report.to_json()
        data = json.loads(json_str)
        
        assert data["total_rows"] == 100
        assert "normal" in data["columns"]
        assert data["columns"]["normal"]["mean"] is not None

    def test_empty_df(self):
        df = pd.DataFrame()
        profiler = DataProfiler()
        report = profiler.profile(df)
        
        assert report.total_rows == 0
        assert report.total_columns == 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
