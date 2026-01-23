"""Regex patterns for common PII types."""

import re
from enum import Enum
from typing import Dict, Pattern


class PIIType(Enum):
    """Types of personally identifiable information."""
    
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    DATE_OF_BIRTH = "date_of_birth"
    NAME = "name"  # Requires NLP, regex fallback is weak
    ADDRESS = "address"  # Requires NLP, regex fallback is weak


class PIIPatterns:
    """
    Collection of regex patterns for detecting PII.
    
    Each pattern is designed to catch common formats while minimizing
    false positives. Patterns are compiled for performance.
    """
    
    # Email: Standard RFC 5322 simplified pattern
    EMAIL: Pattern = re.compile(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        re.IGNORECASE
    )
    
    # Phone: Multiple formats (US-centric, international prefix optional)
    PHONE: Pattern = re.compile(
        r'''
        (?:
            (?:\+?1[-.\s]?)?              # Optional US country code
            (?:\(?\d{3}\)?[-.\s]?)        # Area code with optional parens
            \d{3}[-.\s]?\d{4}             # Main number
        )
        |
        (?:
            \+?\d{1,3}[-.\s]?             # International prefix
            \(?\d{2,4}\)?[-.\s]?          # Area code
            \d{3,4}[-.\s]?\d{3,4}         # Number
        )
        ''',
        re.VERBOSE
    )
    
    # SSN: US Social Security Number (XXX-XX-XXXX or XXXXXXXXX)
    SSN: Pattern = re.compile(
        r'\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b'
    )
    
    # Credit Card: Major card formats (Visa, MC, Amex, Discover)
    CREDIT_CARD: Pattern = re.compile(
        r'''
        \b(?:
            4[0-9]{12}(?:[0-9]{3})?|           # Visa
            5[1-5][0-9]{14}|                    # Mastercard
            3[47][0-9]{13}|                     # American Express
            6(?:011|5[0-9]{2})[0-9]{12}|       # Discover
            (?:\d{4}[-\s]?){3}\d{4}            # Any 16-digit with separators
        )\b
        ''',
        re.VERBOSE
    )
    
    # IP Address: IPv4 format
    IP_ADDRESS: Pattern = re.compile(
        r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}'
        r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    )
    
    # Date of Birth: Common date formats
    DATE_OF_BIRTH: Pattern = re.compile(
        r'''
        \b(?:
            \d{1,2}[-/]\d{1,2}[-/]\d{2,4}|     # MM/DD/YYYY or DD/MM/YYYY
            \d{4}[-/]\d{1,2}[-/]\d{1,2}|       # YYYY-MM-DD
            (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*
            \s+\d{1,2},?\s+\d{4}               # Month DD, YYYY
        )\b
        ''',
        re.VERBOSE | re.IGNORECASE
    )
    
    @classmethod
    def get_pattern(cls, pii_type: PIIType) -> Pattern:
        """
        Get the regex pattern for a specific PII type.
        
        Args:
            pii_type: The type of PII to get pattern for
            
        Returns:
            Compiled regex pattern
            
        Raises:
            ValueError: If PII type has no regex pattern (requires NLP)
        """
        pattern_map: Dict[PIIType, Pattern] = {
            PIIType.EMAIL: cls.EMAIL,
            PIIType.PHONE: cls.PHONE,
            PIIType.SSN: cls.SSN,
            PIIType.CREDIT_CARD: cls.CREDIT_CARD,
            PIIType.IP_ADDRESS: cls.IP_ADDRESS,
            PIIType.DATE_OF_BIRTH: cls.DATE_OF_BIRTH,
        }
        
        if pii_type not in pattern_map:
            raise ValueError(
                f"PII type '{pii_type.value}' requires NLP-based detection. "
                f"Install with: pip install dataguard[nlp]"
            )
        
        return pattern_map[pii_type]
    
    @classmethod
    def get_all_patterns(cls) -> Dict[PIIType, Pattern]:
        """
        Get all available regex patterns.
        
        Returns:
            Dictionary mapping PIIType to compiled patterns
        """
        return {
            PIIType.EMAIL: cls.EMAIL,
            PIIType.PHONE: cls.PHONE,
            PIIType.SSN: cls.SSN,
            PIIType.CREDIT_CARD: cls.CREDIT_CARD,
            PIIType.IP_ADDRESS: cls.IP_ADDRESS,
            PIIType.DATE_OF_BIRTH: cls.DATE_OF_BIRTH,
        }
