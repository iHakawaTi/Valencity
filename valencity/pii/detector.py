"""PII Detection logic for DataFrames and text."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union

import pandas as pd

from valencity.pii.patterns import PIIPatterns, PIIType
from valencity.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PIIMatch:
    """Represents a single PII match found in text."""
    
    pii_type: Union[PIIType, str]
    value: str
    start: int
    end: int
    confidence: float = 1.0  # Regex matches are high confidence
    
    def __repr__(self) -> str:
        type_str = self.pii_type.value if isinstance(self.pii_type, PIIType) else self.pii_type
        masked = self.value[:2] + "***" + self.value[-2:] if len(self.value) > 4 else "***"
        return f"PIIMatch(type={type_str}, value='{masked}')"


@dataclass
class ColumnPIIReport:
    """PII detection report for a single column."""
    
    column_name: str
    pii_types_found: Set[Union[PIIType, str]] = field(default_factory=set)
    match_count: int = 0
    sample_matches: List[PIIMatch] = field(default_factory=list)
    rows_with_pii: int = 0
    total_rows: int = 0
    
    @property
    def pii_percentage(self) -> float:
        """Percentage of rows containing PII."""
        if self.total_rows == 0:
            return 0.0
        return (self.rows_with_pii / self.total_rows) * 100


@dataclass
class PIIReport:
    """PII detection report for an entire DataFrame."""
    
    columns_with_pii: Dict[str, ColumnPIIReport] = field(default_factory=dict)
    total_matches: int = 0
    scanned_columns: int = 0
    total_columns: int = 0
    
    @property
    def has_pii(self) -> bool:
        """Returns True if any PII was detected."""
        return self.total_matches > 0
    
    @property
    def pii_columns(self) -> List[str]:
        """List of column names containing PII."""
        return list(self.columns_with_pii.keys())
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        if not self.has_pii:
            return f"✅ No PII detected in {self.scanned_columns} columns."
        
        lines = [
            f"⚠️  PII detected in {len(self.columns_with_pii)}/{self.scanned_columns} columns:",
        ]
        for col_name, col_report in self.columns_with_pii.items():
            types_str = []
            for t in col_report.pii_types_found:
                 types_str.append(t.value if isinstance(t, PIIType) else t)
            types = ", ".join(types_str)
            
            lines.append(
                f"  - {col_name}: {col_report.match_count} matches "
                f"({col_report.pii_percentage:.1f}% of rows) [{types}]"
            )
        return "\n".join(lines)


class PIIDetector:
    """
    Detect personally identifiable information in DataFrames and text.
    
    Supports both regex-based detection (default) and NLP-based detection
    (requires optional spacy dependency).
    
    Example:
        >>> detector = PIIDetector()
        >>> report = detector.scan_dataframe(df)
        >>> print(report.summary())
        
        >>> matches = detector.scan_text("Contact me at john@example.com")
        >>> print(matches)
    """
    
    def __init__(
        self,
        pii_types: Optional[List[Union[PIIType, str]]] = None,
        use_nlp: bool = False,
        sample_size: int = 5
    ):
        """
        Initialize the PII detector.
        
        Args:
            pii_types: List of PII types (Enum or customs strings) to detect.
                       Defaults to all regex-detectable types.
            use_nlp: Whether to use NLP for name/address detection (requires spacy).
            sample_size: Number of sample matches to include in reports.
        """
        if pii_types:
             self.pii_types = pii_types
        else:
             # Default to all available patterns (including plugins)
             self.pii_types = list(PIIPatterns.get_all_patterns().keys())
             
        self.use_nlp = use_nlp
        self.sample_size = sample_size
        self._nlp = None
        
        if use_nlp:
            self._init_nlp()
    
    def _init_nlp(self) -> None:
        """Initialize spaCy NLP model for NER-based detection."""
        try:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model for NLP-based PII detection")
            except OSError:
                logger.warning(
                    "spaCy model 'en_core_web_sm' not found. "
                    "Install with: python -m spacy download en_core_web_sm"
                )
                self._nlp = None
        except ImportError:
            logger.warning(
                "spaCy not installed. NLP-based detection disabled. "
                "Install with: pip install valencity[nlp]"
            )
            self._nlp = None
    
    def scan_text(self, text: str) -> List[PIIMatch]:
        """
        Scan text for PII matches.
        
        Args:
            text: The text to scan for PII.
            
        Returns:
            List of PIIMatch objects found in the text.
        """
        if not isinstance(text, str):
             # Graceful handling for non-string inputs
             return []
             
        if not text.strip():
            return []
        
        matches: List[PIIMatch] = []
        
        # Regex-based detection
        for pii_type in self.pii_types:
            try:
                # This works for both Enum and registered string patterns
                pattern = PIIPatterns.get_pattern(pii_type)
                for match in pattern.finditer(text):
                    matches.append(PIIMatch(
                        pii_type=pii_type,
                        value=match.group(),
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9  # Regex matches are high confidence
                    ))
            except ValueError:
                # PII type requires NLP, handled separately
                pass
        
        # NLP-based detection for names and addresses
        if self._nlp is not None:
             check_nlp = False
             # Check if NLP types are in requested types
             if any(t == PIIType.NAME or t == "name" for t in self.pii_types):
                 check_nlp = True
             if any(t == PIIType.ADDRESS or t == "address" for t in self.pii_types):
                 check_nlp = True
                 
             if check_nlp:
                matches.extend(self._scan_text_nlp(text))
        
        return matches
    
    def _scan_text_nlp(self, text: str) -> List[PIIMatch]:
        """Scan text using NLP for name/address detection."""
        if self._nlp is None:
            return []
        
        matches: List[PIIMatch] = []
        doc = self._nlp(text)
        
        entity_map = {
            "PERSON": PIIType.NAME,
            "GPE": PIIType.ADDRESS,  # Geo-political entity (cities, countries)
            "LOC": PIIType.ADDRESS,  # Location
        }
        
        for ent in doc.ents:
            if ent.label_ in entity_map:
                pii_type = entity_map[ent.label_]
                
                # Check if this specific NLP type is requested
                # Handle both Enum comparisons and potential string equivalents if we were using strings for all
                is_requested = False
                if pii_type in self.pii_types or pii_type.value in self.pii_types:
                    is_requested = True
                    
                if is_requested:
                    matches.append(PIIMatch(
                        pii_type=pii_type,
                        value=ent.text,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=0.7  # NLP matches are lower confidence
                    ))
        
        return matches
    
    def scan_column(self, series: pd.Series, column_name: str = "column") -> ColumnPIIReport:
        """
        Scan a pandas Series (column) for PII.
        
        Args:
            series: The pandas Series to scan.
            column_name: Name of the column for reporting.
            
        Returns:
            ColumnPIIReport with detection results.
        """
        report = ColumnPIIReport(
            column_name=column_name,
            total_rows=len(series)
        )
        
        # Validate input
        if series.empty:
             return report
             
        # Only scan string columns
        if series.dtype != object and not pd.api.types.is_string_dtype(series):
            return report
        
        all_matches: List[PIIMatch] = []
        rows_with_pii = 0
        
        # Drop NAs and ensure we iterate over strings
        for value in series.dropna():
            if not isinstance(value, str):
                continue
            
            matches = self.scan_text(value)
            if matches:
                rows_with_pii += 1
                all_matches.extend(matches)
                for match in matches:
                    report.pii_types_found.add(match.pii_type)
        
        report.match_count = len(all_matches)
        report.rows_with_pii = rows_with_pii
        report.sample_matches = all_matches[:self.sample_size]
        
        return report
    
    def scan_dataframe(
        self,
        df: pd.DataFrame,
        columns: Optional[List[str]] = None
    ) -> PIIReport:
        """
        Scan a DataFrame for PII across specified columns.
        
        Args:
            df: The DataFrame to scan.
            columns: Optional list of columns to scan. Defaults to all object columns.
            
        Returns:
            PIIReport with detection results for the entire DataFrame.
            
        Raises:
            ValueError: If df is not a pandas DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected pandas DataFrame, got {type(df)}")
            
        report = PIIReport(total_columns=len(df.columns))
        
        if df.empty:
            logger.warning("DataFrame is empty, nothing to scan.")
            return report
        
        # Default to all string/object columns
        if columns is None:
            columns = df.select_dtypes(include=[object, "string"]).columns.tolist()
        
        report.scanned_columns = len(columns)
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found in DataFrame, skipping")
                continue
            
            col_report = self.scan_column(df[col], column_name=col)
            
            if col_report.match_count > 0:
                report.columns_with_pii[col] = col_report
                report.total_matches += col_report.match_count
        
        if report.has_pii:
            logger.warning(report.summary())
        else:
            logger.info(f"No PII detected in {report.scanned_columns} columns")
        
        return report
