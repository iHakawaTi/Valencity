"""
Async wrapper for PII Detector.
"""

import asyncio
from typing import List, Optional

import pandas as pd

from valencity.pii.detector import PIIDetector, PIIMatch, PIIReport


class AsyncPIIDetector:
    """
    Asynchronous wrapper for PIIDetector.
    
    Offloads CPU-intensive regex scanning to a thread pool to avoid blocking 
    the event loop.
    """
    
    def __init__(self, detector: Optional[PIIDetector] = None, **kwargs):
        """
        Initialize async detector.
        
        Args:
            detector: Existing PIIDetector instance.
            **kwargs: Arguments to pass to PIIDetector constructor if detector is None.
        """
        self.detector = detector or PIIDetector(**kwargs)
        
    async def scan_text(self, text: str) -> List[PIIMatch]:
        """Async version of scan_text."""
        return await asyncio.to_thread(self.detector.scan_text, text)
        
    async def scan_dataframe(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None
    ) -> PIIReport:
        """Async version of scan_dataframe."""
        return await asyncio.to_thread(self.detector.scan_dataframe, df, columns)
