"""Tests for valencity.pii module."""

import pandas as pd
import pytest

from valencity.pii import MaskingStrategy, PIIDetector, PIIMasker, PIIPatterns, PIIType


class TestPIIPatterns:
    """Test regex patterns for PII detection."""
    
    def test_email_detection(self):
        pattern = PIIPatterns.EMAIL
        
        # Should match
        assert pattern.search("john@example.com")
        assert pattern.search("user.name+tag@domain.org")
        assert pattern.search("contact@sub.domain.co.uk")
        
        # Should not match
        assert not pattern.search("not an email")
        assert not pattern.search("missing@domain")
    
    def test_phone_detection(self):
        pattern = PIIPatterns.PHONE
        
        # Should match various formats
        assert pattern.search("555-123-4567")
        assert pattern.search("(555) 123-4567")
        assert pattern.search("+1 555 123 4567")
    
    def test_ssn_detection(self):
        pattern = PIIPatterns.SSN
        
        # Should match
        assert pattern.search("123-45-6789")
        assert pattern.search("123 45 6789")
        
        # Should not match invalid SSNs
        assert not pattern.search("000-12-3456")  # Invalid area
        assert not pattern.search("123-00-4567")  # Invalid group
    
    def test_credit_card_detection(self):
        pattern = PIIPatterns.CREDIT_CARD
        
        # Should match
        assert pattern.search("4111111111111111")  # Visa
        assert pattern.search("5500000000000004")  # Mastercard
        assert pattern.search("4111-1111-1111-1111")  # With dashes
    
    def test_ip_address_detection(self):
        pattern = PIIPatterns.IP_ADDRESS
        
        # Should match
        assert pattern.search("192.168.1.1")
        assert pattern.search("10.0.0.255")
        
        # Should not match invalid IPs
        assert not pattern.search("256.1.1.1")
        assert not pattern.search("1.2.3")

    def test_iban_detection(self):
        pattern = PIIPatterns.IBAN
        # Valid IBAN-like strings (simplified regex)
        assert pattern.search("GB29WCBD12345678")
        assert pattern.search("DE89370000440532013000")
        
        # Invalid
        assert not pattern.search("GB29")
        assert not pattern.search("NOT AN IBAN")

    def test_passport_detection(self):
        pattern = PIIPatterns.PASSPORT
        # US Passport (9 digits)
        assert pattern.search("123456789")
        # Generic Passport (Letter + Digits)
        assert pattern.search("A12345678")
        
        # Too short
        assert not pattern.search("123")

    def test_api_key_detection(self):
        pattern = PIIPatterns.API_KEY
        # AWS Key
        assert pattern.search("AKIAIOSFODNN7EXAMPLE")
        # GitHub Token
        assert pattern.search("ghp_123456789012345678901234567890123456")
        
        # Not a key
        assert not pattern.search("just-some-text")

    def test_medical_id_detection(self):
        code_pattern = PIIPatterns.MEDICAL_CODE
        npi_pattern = PIIPatterns.MEDICAL_LICENSE
        
        # ICD-10
        assert code_pattern.search("A01.0")
        assert code_pattern.search("J01")
        
        # NPI
        assert npi_pattern.search("1234567890")
        
        # Invalid
        assert not code_pattern.search("invalid")


class TestPIIDetector:
    """Test PII detection functionality."""
    
    @pytest.fixture
    def detector(self):
        return PIIDetector()
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "name": ["John Doe", "Jane Smith"],
            "email": ["john@example.com", "jane@company.org"],
            "phone": ["555-123-4567", "(555) 987-6543"],
            "notes": ["Regular text", "My SSN is 123-45-6789"],
            "age": [30, 25]
        })
    
    def test_scan_text_finds_email(self, detector):
        matches = detector.scan_text("Contact me at john@example.com")
        
        assert len(matches) == 1
        assert matches[0].pii_type == PIIType.EMAIL
        assert matches[0].value == "john@example.com"
    
    def test_scan_text_finds_multiple_pii(self, detector):
        text = "Email: john@test.com, Phone: 555-123-4567"
        matches = detector.scan_text(text)
        
        assert len(matches) == 2
        pii_types = {m.pii_type for m in matches}
        assert PIIType.EMAIL in pii_types
        assert PIIType.PHONE in pii_types
    
    def test_scan_dataframe_returns_report(self, detector, sample_df):
        report = detector.scan_dataframe(sample_df)
        
        assert report.has_pii
        assert "email" in report.pii_columns
        assert "phone" in report.pii_columns
        assert "notes" in report.pii_columns
        assert "age" not in report.pii_columns  # Numeric column
    
    def test_scan_empty_text(self, detector):
        assert detector.scan_text("") == []
        assert detector.scan_text("   ") == []
    
    def test_scan_no_pii(self, detector):
        matches = detector.scan_text("This is regular text with no personal info.")
        assert len(matches) == 0


class TestPIIMasker:
    """Test PII masking functionality."""
    
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            "email": ["john@example.com", "jane@company.org"],
            "notes": ["Call 555-123-4567", "Regular text"]
        })
    
    def test_mask_text_redact(self):
        masker = PIIMasker(strategy=MaskingStrategy.REDACT)
        result = masker.mask_text("Email: john@example.com")
        
        assert "john@example.com" not in result
        assert "[REDACTED]" in result
    
    def test_mask_text_partial(self):
        masker = PIIMasker(strategy=MaskingStrategy.PARTIAL)
        result = masker.mask_text("Email: john@example.com")
        
        assert "john@example.com" not in result
        # Partial should keep some characters
        assert "jo" in result or "**" in result
    
    def test_mask_text_hash(self):
        masker = PIIMasker(strategy=MaskingStrategy.HASH)
        result = masker.mask_text("Email: john@example.com")
        
        assert "john@example.com" not in result
        assert "[" in result  # Hash format is [XXXXXXXX]
    
    def test_mask_dataframe(self, sample_df):
        masker = PIIMasker(strategy=MaskingStrategy.REDACT)
        masked = masker.mask_dataframe(sample_df)
        
        # Original should be unchanged
        assert sample_df["email"].iloc[0] == "john@example.com"
        
        # Masked should have redacted values
        assert "[REDACTED]" in masked["email"].iloc[0]
    
    def test_mask_preserves_non_pii(self, sample_df):
        masker = PIIMasker(strategy=MaskingStrategy.REDACT)
        masked = masker.mask_dataframe(sample_df)
        
        # Non-PII text should be preserved
        assert masked["notes"].iloc[1] == "Regular text"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
