"""
Privacy compliance verification.
"""

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from valencity.pii import PIIDetector, PIIType


@dataclass
class ComplianceViolation:
    rule: str
    description: str
    severity: str  # High, Medium, Low

@dataclass
class ComplianceReport:
    satisfied: bool
    violations: List[ComplianceViolation] = field(default_factory=list)

class ComplianceChecker:
    """
    Check dataset for potential privacy regulation violations (GDPR/CCPA context).
    Note: This is a technical helper, not legal advice.
    """
    
    def __init__(self):
        self.pii_detector = PIIDetector()
        
    def check_gdpr(self, df: pd.DataFrame) -> ComplianceReport:
        """
        Perform basic technical checks relevant to GDPR.
        
        Checks:
        1. Presence of unmasked direct identifiers (Email, SSN, Phone).
        2. (Heuristic) Presence of potential 'consent' tracking.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
            
        violations = []
        
        # 1. Check for unmasked PII
        pii_report = self.pii_detector.scan_dataframe(df)
        
        if pii_report.has_pii:
            # Check for high risk PII
            critical_types = {PIIType.EMAIL, PIIType.SSN, PIIType.PHONE, PIIType.PASSPORT, PIIType.IBAN}
            found_critical = False
            
            for col, report in pii_report.columns_with_pii.items():
                found_types = set()
                # Normalize types to Enum if possible for comparison
                for t in report.pii_types_found:
                    if isinstance(t, PIIType):
                        found_types.add(t)
                    # We could try to map strings back to Enums if needed, or just ignore custom types for "Critical" check unless defined
                
                intersection = found_types.intersection(critical_types)
                if intersection:
                    found_critical = True
                    types_list = [t.value for t in intersection]
                    types_str = ", ".join(types_list)
                    violations.append(ComplianceViolation(
                        rule="GDPR Art. 32 (Security of Processing)",
                        description=f"Unmasked critical PII found in column '{col}': {types_str}. Storage of clear-text PII increases breach risk.",
                        severity="High"
                    ))
            
            # If only low risk PII found, might still be an issue but lower severity
            if not found_critical and pii_report.has_pii:
                 violations.append(ComplianceViolation(
                        rule="Data Minimization",
                        description=f"Potential PII detected in columns: {', '.join(pii_report.pii_columns)}. Verify necessity.",
                        severity="Medium"
                    ))

        # 2. Check for consent fields (heuristics)
        # Looking for columns like 'consent', 'opt_in', 'agreed'
        columns_lower = [str(c).lower() for c in df.columns]
        has_consent = any(term in col for col in columns_lower for term in ['consent', 'opt_in', 'legal_basis'])
        
        # This is a weak check, but useful reminder
        if pii_report.has_pii and not has_consent:
             violations.append(ComplianceViolation(
                rule="GDPR Art. 6 (Lawfulness of Processing)",
                description="PII detected but no obvious 'consent' or 'legal_basis' column found in dataset. Ensure you track the legal basis for processing this data.",
                severity="Low" # It's low because we might just miss the column name
            ))

        return ComplianceReport(
            satisfied=len(violations) == 0,
            violations=violations
        )
