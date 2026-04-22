"""Tests for valencity.privacy module."""

import numpy as np
import pandas as pd
import pytest

from valencity.privacy import ComplianceChecker, DifferentialPrivacy


class TestDifferentialPrivacy:
    """Test DP mechanisms."""
    
    def test_laplace_mechanism_stats(self):
        # With high epsilon (low noise), values should be close to original
        data = np.zeros(1000)
        noisy = DifferentialPrivacy.laplace_mechanism(
            data, sensitivity=1.0, epsilon=10.0, random_state=42
        )
        
        # Mean should be close to 0
        assert abs(noisy.mean()) < 0.2
        # Variance should be roughly 2 * (1/10)^2 = 0.02
        assert abs(noisy.var() - 0.02) < 0.01

    def test_laplace_series(self):
        s = pd.Series([1, 2, 3])
        noisy = DifferentialPrivacy.laplace_mechanism(s, sensitivity=1, epsilon=1)
        assert len(noisy) == 3
        assert isinstance(noisy, pd.Series)

    def test_gaussian_mechanism(self):
        data = np.zeros(1000)
        noisy = DifferentialPrivacy.gaussian_mechanism(
            data, sensitivity=1.0, epsilon=1.0, delta=1e-5, random_state=42
        )
        assert abs(noisy.mean()) < 0.5


class TestComplianceChecker:
    """Test compliance verification."""
    
    def test_gdpr_check_clean(self):
        df = pd.DataFrame({"age": [20, 30], "city": ["NY", "SF"]})
        checker = ComplianceChecker()
        report = checker.check_gdpr(df)
        
        assert report.satisfied
        assert len(report.violations) == 0
        
    def test_gdpr_check_violation(self):
        df = pd.DataFrame({
            "email": ["user@example.com"], 
            "score": [10]
        })
        checker = ComplianceChecker()
        report = checker.check_gdpr(df)
        
        assert not report.satisfied
        # Should find email as critical PII
        assert any(v.severity == "High" for v in report.violations)
        assert "Unmasked critical PII" in report.violations[0].description

    def test_consent_check(self):
        # Email exists (PII violation), but we add 'consent' column
        # The PII violation remains (High severity), but the "missing consent" warning (Low) 
        # might be suppressed depending on logic.
        # Current logic:
        # 1. Finds PII -> High violation
        # 2. Checks consent -> If present, no Low violation.
        
        df = pd.DataFrame({
            "email": ["user@example.com"],
            "has_consent": [True]
        })
        checker = ComplianceChecker()
        report = checker.check_gdpr(df)
        
        # Should still fail due to unmasked PII
        assert not report.satisfied
        # But shouldn't have the "no consent found" violation
        descriptions = [v.description for v in report.violations]
        assert not any("no obvious 'consent'" in d for d in descriptions)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
