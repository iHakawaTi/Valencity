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
    violations: List[ComplianceViolation] = field(default_factory=list)

    @property
    def satisfied(self) -> bool:
        """True if no violations found."""
        return len(self.violations) == 0

    @property
    def is_compliant(self) -> bool:
        """Alias for satisfied."""
        return self.satisfied


class ComplianceChecker:
    """
    Check a dataset for potential privacy regulation violations.

    Performs heuristic technical checks for GDPR/CCPA patterns.
    Note: This is a technical helper tool, not legal advice.

    Example:
        >>> checker = ComplianceChecker()
        >>> report = checker.check(df)          # or check_gdpr(df)
        >>> if not report.is_compliant:
        ...     for v in report.violations:
        ...         print(v.rule, v.severity)
    """

    def __init__(self) -> None:
        self.pii_detector = PIIDetector()

    def check_gdpr(self, df: pd.DataFrame) -> ComplianceReport:
        """
        Check a DataFrame for GDPR compliance violations.

        Checks:
          1. Unmasked high-risk PII (emails, SSNs, phones, etc.)
          2. Absence of consent / legal_basis tracking columns
          3. General data minimization heuristic

        Args:
            df: DataFrame to audit.

        Returns:
            ComplianceReport with a list of violations.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        report = ComplianceReport()
        pii_report = self.pii_detector.scan_dataframe(df)

        if pii_report.has_pii:
            critical_types = {
                PIIType.EMAIL, PIIType.SSN, PIIType.PHONE,
                PIIType.PASSPORT, PIIType.IBAN,
            }

            for col, col_report in pii_report.columns_with_pii.items():
                found_types = set()
                for t in col_report.pii_types_found:
                    if isinstance(t, PIIType):
                        found_types.add(t)

                critical_found = found_types & critical_types
                if critical_found:
                    types_str = ", ".join(t.value for t in critical_found)
                    report.violations.append(
                        ComplianceViolation(
                            rule="GDPR Art. 32 (Security of Processing)",
                            description=(
                                f"Unmasked high-risk PII in column '{col}': {types_str}. "
                                "Clear-text PII storage increases breach risk."
                            ),
                            severity="High",
                        )
                    )
                elif found_types:
                    report.violations.append(
                        ComplianceViolation(
                            rule="Data Minimization",
                            description=(
                                f"Potential PII detected in column '{col}'. "
                                "Verify data is necessary and properly protected."
                            ),
                            severity="Medium",
                        )
                    )

            # Check for consent / legal basis columns (heuristic)
            cols_lower = [str(c).lower() for c in df.columns]
            has_consent = any(
                term in col
                for col in cols_lower
                for term in ("consent", "opt_in", "legal_basis", "processing_purpose")
            )
            if not has_consent:
                report.violations.append(
                    ComplianceViolation(
                        rule="GDPR Art. 6 (Lawfulness of Processing)",
                        description=(
                            "PII detected but no 'consent', 'opt_in', or 'legal_basis' column found. "
                            "Ensure you track the legal basis for processing personal data."
                        ),
                        severity="Low",
                    )
                )

        return report

    def check(self, df: pd.DataFrame) -> ComplianceReport:
        """Convenience alias for check_gdpr()."""
        return self.check_gdpr(df)
