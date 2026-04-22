"""Regex patterns for common PII types."""

import re
from enum import Enum
from re import Pattern
from typing import Dict, Union


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
    IBAN = "iban"
    PASSPORT = "passport"
    API_KEY = "api_key"
    MEDICAL_LICENSE = "medical_license"  # US NPI, etc.
    MEDICAL_CODE = "medical_code"  # ICD-10

class PIIPatterns:
    """
    Collection of regex patterns for detecting PII.
    Supports dynamic registration of new patterns (Plugin system).
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

    # IBAN: General format (Country Code + Check Digits + BBAN)
    # Simplified validation: 2 letters, 2 digits, then 10-30 alphanumeric
    IBAN: Pattern = re.compile(
        r'\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b',
        re.IGNORECASE
    )

    # Passport: Common passport number formats
    # US: 9 digits
    # UK: 9 digits
    # Generic: 1 char + 7-9 digits, or 2 chars + 7 digits
    PASSPORT: Pattern = re.compile(
        r'''
        \b(?:
            [A-Z]\d{7,9}|           # Generic 1 letter + digits (e.g. US Passport Card)
            [A-Z]{2}\d{7}|          # Generic 2 letters + digits
            \d{9}                   # US/UK standard 9 digits (high false positive risk, use with context)
        )\b
        ''',
        re.VERBOSE
    )

    # API Keys & Secrets:
    # AWS Access Key ID, GitHub Tokens, Slack Tokens, Private Keys
    API_KEY: Pattern = re.compile(
        r'''
        (?:
            (?<![A-Z0-9])AKIA[0-9A-Z]{16}(?![A-Z0-9])|      # AWS Access Key ID
            gh[pousr]_[A-Za-z0-9_]{36,}|                    # GitHub Tokens
            xox[baprs]-[0-9A-Za-z-]{10,}|                   # Slack Tokens
            sk_live_[0-9a-zA-Z]{24,}|                       # Stripe/Generic Secret Keys
            -----BEGIN\s[A-Z\s]+\sKEY-----                  # PEM Private Keys
        )
        ''',
        re.VERBOSE
    )

    # Medical:
    # ICD-10 Codes: Letter + 2 digits + dot + 0-3 alphanumeric
    # NPI (US): 10 digit number starting with 1 or 2 (Luhn algorithm not checked here)
    MEDICAL_CODE: Pattern = re.compile(
        r'\b[A-TV-Z][0-9][0-9AB](?:\.[0-9A-K]{1,4})?\b',
        re.IGNORECASE
    )

    MEDICAL_LICENSE: Pattern = re.compile(
        r'\b(?:1|2)\d{9}\b'  # Simple NPI check (10 digits starting with 1 or 2)
    )
    
    # Registry for custom patterns
    _custom_patterns: Dict[str, Pattern] = {}
    
    @classmethod
    def register_pattern(cls, name: str, pattern: Union[str, Pattern]) -> None:
        """
        Register a new PII pattern.
        
        Args:
            name: Unique name for the pattern (slug-like).
            pattern: Regex string or compiled Pattern.
        """
        if isinstance(pattern, str):
            pattern = re.compile(pattern)
        cls._custom_patterns[name] = pattern
        
    @classmethod
    def get_pattern(cls, pii_type: Union[PIIType, str]) -> Pattern:
        """
        Get the regex pattern for a specific PII type.
        
        Args:
            pii_type: The PIIType enum or string name of custom pattern.
            
        Returns:
            Compiled regex pattern
            
        Raises:
            ValueError: If PII type not found or requires NLP.
        """
        # 1. Custom patterns check first
        if isinstance(pii_type, str):
             if pii_type in cls._custom_patterns:
                 return cls._custom_patterns[pii_type]
             # If passed a string matching a built-in Enum name
             try:
                 pii_type = PIIType(pii_type)
             except ValueError:
                  raise ValueError(f"Unknown PII type: {pii_type}")

        # 2. Built-in patterns
        pattern_map: Dict[PIIType, Pattern] = {
            PIIType.EMAIL: cls.EMAIL,
            PIIType.PHONE: cls.PHONE,
            PIIType.SSN: cls.SSN,
            PIIType.CREDIT_CARD: cls.CREDIT_CARD,
            PIIType.IP_ADDRESS: cls.IP_ADDRESS,
            PIIType.DATE_OF_BIRTH: cls.DATE_OF_BIRTH,
            PIIType.IBAN: cls.IBAN,
            PIIType.PASSPORT: cls.PASSPORT,
            PIIType.API_KEY: cls.API_KEY,
            PIIType.MEDICAL_LICENSE: cls.MEDICAL_LICENSE,
            PIIType.MEDICAL_CODE: cls.MEDICAL_CODE,
        }
        
        if pii_type not in pattern_map:
             # It might be a custom pattern registered with the enum value as key?
             # Unlikely given typing, but let's check custom again just in case
             if pii_type.value in cls._custom_patterns:
                 return cls._custom_patterns[pii_type.value]
                 
             raise ValueError(
                f"PII type '{pii_type.value}' requires NLP-based detection. "
                f"Install with: pip install valencity[nlp]"
            )
        
        return pattern_map[pii_type]
    
    @classmethod
    def get_all_patterns(cls) -> Dict[Union[PIIType, str], Pattern]:
        """
        Get all available regex patterns (built-in + custom).
        
        Returns:
            Dictionary mapping PIIType/str to compiled patterns
        """
        patterns = {
            PIIType.EMAIL: cls.EMAIL,
            PIIType.PHONE: cls.PHONE,
            PIIType.SSN: cls.SSN,
            PIIType.CREDIT_CARD: cls.CREDIT_CARD,
            PIIType.IP_ADDRESS: cls.IP_ADDRESS,
            PIIType.DATE_OF_BIRTH: cls.DATE_OF_BIRTH,
            PIIType.IBAN: cls.IBAN,
            PIIType.PASSPORT: cls.PASSPORT,
            PIIType.API_KEY: cls.API_KEY,
            PIIType.MEDICAL_LICENSE: cls.MEDICAL_LICENSE,
            PIIType.MEDICAL_CODE: cls.MEDICAL_CODE,
        }
        # Add custom patterns
        patterns.update(cls._custom_patterns)
        return patterns
