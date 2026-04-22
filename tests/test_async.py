
import pandas as pd
import pytest

from valencity.pii.async_detector import AsyncPIIDetector
from valencity.pii.patterns import PIIType


@pytest.mark.asyncio
async def test_async_scan_text():
    detector = AsyncPIIDetector()
    text = "Contact: test@example.com"
    matches = await detector.scan_text(text)
    
    assert len(matches) == 1
    assert matches[0].pii_type == PIIType.EMAIL

@pytest.mark.asyncio
async def test_async_scan_dataframe():
    detector = AsyncPIIDetector()
    df = pd.DataFrame({"email": ["test@example.com", "safe"]})
    
    report = await detector.scan_dataframe(df)
    assert report.has_pii
    assert "email" in report.pii_columns
