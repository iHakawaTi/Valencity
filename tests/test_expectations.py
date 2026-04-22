"""Tests for valencity.validation.expectations module."""

import pandas as pd
import pytest

from valencity.validation import expect


class TestExpectations:
    """Test fluent expectations API."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "age": [20, 30, 40, 150],
            "email": ["a@b.com", "valid@email.org", "bad-email", "test@co.uk"],
            "status": ["active", "inactive", "active", "deleted"],
            "id": [1, 2, 3, 4],
            "score": [0.1, 0.2, None, 0.4]
        })
        
    def test_expect_between(self, sample_df):
        report = (
            expect(sample_df)
            .column("age").to_be_between(0, 100)
            .run()
        )
        
        assert not report.passed
        failed = report.failed_results[0]
        assert failed.column == "age"
        assert "Found 1 values outside range" in failed.details  # 150 is > 100
        
    def test_expect_regex(self, sample_df):
        report = (
            expect(sample_df)
            .column("email").to_match_regex(r"^[\w\.-]+@[\w\.-]+\.\w+$")
            .run()
        )
        
        assert not report.passed
        failed = report.failed_results[0]
        assert failed.column == "email"
        assert "Found 1 non-matching" in failed.details # "bad-email"
        
    def test_expect_in(self, sample_df):
        report = (
            expect(sample_df)
            .column("status").to_be_in(["active", "inactive"])
            .run()
        )
        
        assert not report.passed
        failed = report.failed_results[0]
        assert failed.column == "status"
        assert "Found 1 invalid values" in failed.details # "deleted"
        
    def test_expect_not_null(self, sample_df):
        report = (
            expect(sample_df)
            .column("score").to_not_be_null()
            .run()
        )
        
        assert not report.passed
        failed = report.failed_results[0]
        assert failed.column == "score"
        assert "Found 1 null values" in failed.details
        
    def test_expect_unique(self, sample_df):
        report = (
            expect(sample_df)
            .column("id").to_be_unique()
            .run()
        )
        
        assert report.passed
        
    def test_expect_type(self, sample_df):
        report = (
            expect(sample_df)
            .column("age").to_have_type("int")
            .run()
        )
        assert report.passed
        
    def test_chained_expectations(self, sample_df):
        # All valid expectations
        report = (
            expect(sample_df)
            .column("id").to_be_unique()
            .column("id").to_be_between(0, 10)
            .run()
        )
        assert report.passed
        assert len(report.results) == 2
        
    def test_missing_column(self, sample_df):
        report = (
            expect(sample_df)
            .column("invalid_col").to_not_be_null()
            .run()
        )
        assert not report.passed
        assert "Column not found" in report.failed_results[0].details

    def test_raise_if_failed(self, sample_df):
        with pytest.raises(ValueError):
            (
                expect(sample_df)
                .column("age").to_be_between(0, 10) # Fails
                .run()
                .raise_if_failed()
            )

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
