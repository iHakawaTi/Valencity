
from valencity.pii.detector import PIIDetector
from valencity.pii.patterns import PIIPatterns, PIIType


def test_register_custom_pattern():
    # Register a custom pattern for "SKU"
    PIIPatterns.register_pattern("sku", r"\bSKU-\d{4}\b")
    
    # Check if it's in get_all_patterns
    patterns = PIIPatterns.get_all_patterns()
    assert "sku" in patterns
    
    # Check detection
    detector = PIIDetector(pii_types=["sku"])
    matches = detector.scan_text("Product SKU-1234 is available")
    
    assert len(matches) == 1
    assert matches[0].value == "SKU-1234"
    assert matches[0].pii_type == "sku"

def test_detector_with_mixed_types():
    # Detect both standard EMAIL and custom SKU
    PIIPatterns.register_pattern("sku_mixed", r"\bSKU-\d{4}\b")
    
    detector = PIIDetector(pii_types=[PIIType.EMAIL, "sku_mixed"])
    text = "Email me at test@example.com about SKU-9999"
    matches = detector.scan_text(text)
    
    assert len(matches) == 2
    types = {m.pii_type for m in matches}
    assert PIIType.EMAIL in types
    assert "sku_mixed" in types
