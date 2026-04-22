"""PII Masking/Anonymization strategies."""

import hashlib
import random
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional

import pandas as pd

from valencity.pii.detector import PIIDetector
from valencity.pii.patterns import PIIType
from valencity.utils.logging import get_logger

logger = get_logger(__name__)


class MaskingStrategy(Enum):
    """Available masking strategies for PII."""
    
    REDACT = "redact"           # Replace with [REDACTED]
    HASH = "hash"               # Replace with SHA256 hash (truncated)
    PARTIAL = "partial"         # Keep first/last chars, mask middle
    FAKE = "fake"               # Replace with realistic fake data
    CUSTOM = "custom"           # User-provided masking function


@dataclass
class MaskingConfig:
    """Configuration for masking behavior."""
    
    strategy: MaskingStrategy = MaskingStrategy.REDACT
    redact_string: str = "[REDACTED]"
    hash_length: int = 8
    partial_keep: int = 2        # Chars to keep at start/end
    custom_func: Optional[Callable[[str, PIIType], str]] = None


class PIIMasker:
    """
    Mask/anonymize detected PII in text and DataFrames.
    
    Supports multiple masking strategies:
    - REDACT: Replace PII with [REDACTED]
    - HASH: Replace with a consistent hash (useful for joining)
    - PARTIAL: Keep some characters visible (e.g., j***@e***.com)
    - FAKE: Replace with realistic fake data
    - CUSTOM: Use a custom masking function
    
    Example:
        >>> masker = PIIMasker(strategy=MaskingStrategy.PARTIAL)
        >>> masked_text = masker.mask_text("Email: john@example.com")
        >>> print(masked_text)  # Email: jo***@ex***.com
        
        >>> masked_df = masker.mask_dataframe(df, columns=["email", "phone"])
    """
    
    # Fake data templates for FAKE strategy
    FAKE_DATA: Dict[PIIType, List[str]] = {
        PIIType.EMAIL: [
            "user@example.com", "contact@demo.org", "info@sample.net"
        ],
        PIIType.PHONE: [
            "(555) 123-4567", "(555) 987-6543", "(555) 456-7890"
        ],
        PIIType.SSN: [
            "XXX-XX-XXXX"
        ],
        PIIType.CREDIT_CARD: [
            "XXXX-XXXX-XXXX-XXXX"
        ],
        PIIType.IP_ADDRESS: [
            "192.168.1.1", "10.0.0.1", "172.16.0.1"
        ],
        PIIType.DATE_OF_BIRTH: [
            "01/01/1990", "12/31/1985", "06/15/2000"
        ],
        PIIType.NAME: [
            "John Doe", "Jane Smith", "Alex Johnson"
        ],
        PIIType.ADDRESS: [
            "123 Main St, Anytown, USA"
        ],
    }
    
    def __init__(
        self,
        strategy: MaskingStrategy = MaskingStrategy.REDACT,
        config: Optional[MaskingConfig] = None,
        pii_types: Optional[List[PIIType]] = None,
        use_nlp: bool = False
    ):
        """
        Initialize the PII masker.
        
        Args:
            strategy: The masking strategy to use.
            config: Optional detailed configuration.
            pii_types: List of PII types to mask. Defaults to all.
            use_nlp: Whether to use NLP for name/address detection.
        """
        self.config = config or MaskingConfig(strategy=strategy)
        self.detector = PIIDetector(pii_types=pii_types, use_nlp=use_nlp)
        
        # Deterministic seed for consistent fake data
        self._fake_seed: Dict[str, str] = {}
    
    def _mask_value(self, value: str, pii_type: PIIType) -> str:
        """Apply masking to a single PII value."""
        strategy = self.config.strategy
        
        if strategy == MaskingStrategy.REDACT:
            return self.config.redact_string
        
        elif strategy == MaskingStrategy.HASH:
            hash_val = hashlib.sha256(value.encode()).hexdigest()
            return f"[{hash_val[:self.config.hash_length].upper()}]"
        
        elif strategy == MaskingStrategy.PARTIAL:
            if len(value) <= self.config.partial_keep * 2 + 3:
                return "*" * len(value)
            keep = self.config.partial_keep
            return value[:keep] + "*" * (len(value) - keep * 2) + value[-keep:]
        
        elif strategy == MaskingStrategy.FAKE:
            # Use cached fake value for consistency
            if value in self._fake_seed:
                return self._fake_seed[value]
            fake_options = self.FAKE_DATA.get(pii_type, ["[ANONYMIZED]"])
            fake_val = random.choice(fake_options)
            self._fake_seed[value] = fake_val
            return fake_val
        
        elif strategy == MaskingStrategy.CUSTOM:
            if self.config.custom_func is None:
                raise ValueError("CUSTOM strategy requires custom_func in config")
            return self.config.custom_func(value, pii_type)
        
        return self.config.redact_string
    
    def mask_text(self, text: str) -> str:
        """
        Mask all PII in a text string.
        
        Args:
            text: The text to mask.
            
        Returns:
            Text with all detected PII masked.
        """
        if not isinstance(text, str) or not text.strip():
            return text
        
        matches = self.detector.scan_text(text)
        
        if not matches:
            return text
        
        # Sort matches by position in reverse to replace from end to start
        # This preserves positions during replacement
        sorted_matches = sorted(matches, key=lambda m: m.start, reverse=True)
        
        result = text
        for match in sorted_matches:
            masked = self._mask_value(match.value, match.pii_type)
            result = result[:match.start] + masked + result[match.end:]
        
        return result
    
    def mask_series(self, series: pd.Series) -> pd.Series:
        """
        Mask PII in a pandas Series.
        
        Args:
            series: The Series to mask.
            
        Returns:
            New Series with PII masked.
        """
        if series.dtype != object:
            return series
        
        return series.apply(
            lambda x: self.mask_text(x) if isinstance(x, str) else x
        )
    
    def mask_dataframe(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Mask PII in specified DataFrame columns.
        
        Args:
            df: The DataFrame to mask.
            columns: Columns to mask. Defaults to all object columns.
            inplace: Whether to modify the DataFrame in place.
            
        Returns:
            DataFrame with PII masked in specified columns.
        """
        if not inplace:
            df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[object]).columns.tolist()
        
        masked_count = 0
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found, skipping")
                continue
            
            original = df[col].copy()
            df[col] = self.mask_series(df[col])
            masked_count += (original != df[col]).sum()
        
        logger.info(f"Masked PII in {len(columns)} columns ({masked_count} values modified)")
        
        return df
