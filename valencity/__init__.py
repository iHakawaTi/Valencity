"""
Valencity - A privacy & ML safety toolkit for production ML pipelines.

Modules:
    - pii: Detect and mask personally identifiable information
    - validation: Schema validation, data quality, and drift detection
    - leakage: Prevent data leakage in ML pipelines
    - synthetic: Generate synthetic data
    - privacy: Differential privacy and compliance checking
    - reports: Generate HTML reports
"""

__version__ = "0.1.0"
__author__ = "Abdallah El-Hakawati"

from valencity import leakage, pii, privacy, reports, synthetic, validation

__all__ = [
    "pii", 
    "validation", 
    "leakage", 
    "synthetic", 
    "privacy", 
    "reports",
    "__version__",
    "show_api",
]


def show_api() -> None:
    """
    Display all available Valencity functions and classes.
    
    Example:
        >>> import valencity
        >>> valencity.show_api()
    """
    api_info = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                         🛡️  VALENCITY API REFERENCE                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  PII MODULE (valencity.pii)                                                  ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║    PIIDetector      - Scan text/DataFrames for PII                           ║
║    PIIMasker        - Mask/anonymize detected PII                            ║
║    PIIType          - Enum of supported PII types                            ║
║    PIIPatterns      - Registry of regex patterns (supports plugins)          ║
║    AsyncPIIDetector - Async wrapper for non-blocking scans                   ║
║                                                                              ║
║  VALIDATION MODULE (valencity.validation)                                    ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║    DataSchema         - Define and validate data schemas                     ║
║    DataQualityChecker - Run quality checks (nulls, duplicates, etc.)         ║
║    DataProfiler       - Generate statistical profiles                        ║
║    DriftDetector      - Detect data drift between datasets                   ║
║    expect()           - Fluent API for expectations                          ║
║                                                                              ║
║  LEAKAGE MODULE (valencity.leakage)                                          ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║    SafeCrossValidator - Leakage-safe cross-validation                        ║
║    SafeSplitter       - Leakage-safe train/test splitting                    ║
║    LeakageDetector    - Detect potential data leakage                        ║
║                                                                              ║
║  SYNTHETIC MODULE (valencity.synthetic)                                      ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║    SyntheticGenerator - Generate synthetic data from rules or templates      ║
║                                                                              ║
║  PRIVACY MODULE (valencity.privacy)                                          ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║    DifferentialPrivacy - Apply differential privacy noise                    ║
║    ComplianceChecker   - Check for GDPR/CCPA violations                      ║
║                                                                              ║
║  REPORTS MODULE (valencity.reports)                                          ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║    HTMLGenerator - Generate beautiful HTML reports                           ║
║                                                                              ║
║  CLI COMMANDS                                                                ║
║  ─────────────────────────────────────────────────────────────────────────── ║
║    valencity info              - Show package info                           ║
║    valencity pii scan <file>   - Scan file for PII                           ║
║    valencity pii mask <file>   - Mask PII in file                            ║
║    valencity profile <file>    - Generate data profile                       ║
║    valencity compliance <file> - Check privacy compliance                    ║
║    valencity validate quality  - Run quality checks                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

For detailed documentation, visit: https://github.com/ihakawati/Valencity
"""
    print(api_info)
