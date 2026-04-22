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

try:
    from importlib.metadata import version as _v
    __version__ = _v("valencity")
except Exception:
    __version__ = "0.1.1"
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
+------------------------------------------------------------------------------+
|                         VALENCITY API REFERENCE  v0.1.1                     |
+------------------------------------------------------------------------------+
|
|  PII MODULE  (valencity.pii)
|  -------------------------------------------------------------------
|    PIIDetector           .scan_dataframe(df)  .scan_text(text)
|    PIIMasker             .mask_dataframe(df)  strategy='redact'|'hash'|
|                          'partial'|'fake'
|    PIIType               Enum of 50+ supported PII types
|    PIIPatterns           Pattern registry (plugin-friendly)
|    AsyncPIIDetector      async .scan_dataframe(df)
|
|  VALIDATION MODULE  (valencity.validation)
|  -------------------------------------------------------------------
|    DataSchema            .from_dataframe(df)  .validate(df)
|                          result: .is_valid  .errors  .warnings
|    DataQualityChecker    .full_report(df)  .check_nulls(df)
|                          .check_duplicates(df)  .check_outliers(df)
|                          result: .passed  .failed_checks  .total_rows
|    DataProfiler          .profile(df)
|                          result: .total_rows  .total_columns  .columns
|    DriftDetector         .fit(df)  .detect(df)
|                          method='ks_test'|'psi'|'jensen_shannon'|'chi_squared'
|                          result: .has_drift  .drifted_columns
|    expect(df)            .column(name).to_be_between(min, max)
|                          .to_be_in(values)  .to_not_be_null()
|                          .to_be_unique()  .to_have_type(dtype)
|                          .run()  -> result: .passed  .failed_results
|
|  LEAKAGE MODULE  (valencity.leakage)
|  -------------------------------------------------------------------
|    SafeCrossValidator    .cross_val_score(model, X, y)
|    LeakageDetector       .full_check(X_tr, X_te, y_tr, y_te)
|                          .check_target_leakage(X, y)
|                          .check_feature_leakage(X_tr, X_te)
|    safe_train_test_split(X, y)             -> X_tr, X_te, y_tr, y_te
|    temporal_train_test_split(X, y,          -> X_tr, X_te, y_tr, y_te
|                              time_column)
|
|  SYNTHETIC MODULE  (valencity.synthetic)
|  -------------------------------------------------------------------
|    SyntheticGenerator    .from_dataframe(df)  .generate(num_rows)
|                          .add_numeric(name, ...)  .add_categorical(...)
|
|  PRIVACY MODULE  (valencity.privacy)
|  -------------------------------------------------------------------
|    DifferentialPrivacy   .laplace_mechanism(value, sensitivity, epsilon)
|                          .gaussian_mechanism(value, sensitivity, epsilon, delta)
|    ComplianceChecker     .check(df)  /  .check_gdpr(df)
|                          result: .is_compliant  .violations
|                          violation: .rule  .description  .severity
|
|  REPORTS MODULE  (valencity.reports)
|  -------------------------------------------------------------------
|    HTMLGenerator         .render_pii_report(report, output_path)
|                          .render_quality_report(report, output_path)
|                          .render_profile_report(report, output_path)
|                          .render_compliance_report(report, output_path)
|
|  CLI COMMANDS
|  -------------------------------------------------------------------
|    valencity info
|    valencity pii scan <file>
|    valencity pii mask <file>
|    valencity profile <file>
|    valencity compliance <file>
|    valencity validate quality
|
+------------------------------------------------------------------------------+
Docs: https://github.com/iHakawaTi/Valencity
"""
    print(api_info)

