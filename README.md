# 🛡️ DataGuard

A privacy & ML safety toolkit for production ML pipelines.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

**DataGuard** helps ML engineers build safer, more robust pipelines by providing:

- **🔍 PII Detection & Masking** — Scan DataFrames for personal information and anonymize it
- **✅ Data Validation** — Schema validation, quality checks, and distribution drift detection
- **🚫 Leakage Prevention** — Safe cross-validation wrappers and train/test split utilities

## Installation

```bash
pip install dataguard
```

For NLP-based PII detection (names, addresses):
```bash
pip install dataguard[nlp]
python -m spacy download en_core_web_sm
```

## Quick Start

### PII Detection & Masking

```python
import pandas as pd
from dataguard.pii import PIIDetector, PIIMasker, MaskingStrategy

# Create sample data
df = pd.DataFrame({
    "name": ["John Doe", "Jane Smith"],
    "email": ["john@example.com", "jane@company.org"],
    "notes": ["Call me at 555-123-4567", "SSN: 123-45-6789"]
})

# Detect PII
detector = PIIDetector()
report = detector.scan_dataframe(df)
print(report.summary())

# Mask PII
masker = PIIMasker(strategy=MaskingStrategy.PARTIAL)
masked_df = masker.mask_dataframe(df)
print(masked_df)
```

### Data Validation

```python
from dataguard.validation import DataSchema, ColumnSpec, DataType, DataQualityChecker

# Define schema
schema = DataSchema([
    ColumnSpec("id", DataType.INTEGER, nullable=False, unique=True),
    ColumnSpec("email", DataType.STRING, nullable=False),
    ColumnSpec("age", DataType.INTEGER, min_value=0, max_value=150),
])

# Validate
result = schema.validate(df)
if not result.is_valid:
    print(result.summary())

# Quality checks
checker = DataQualityChecker()
quality_report = checker.full_report(df)
print(quality_report.summary())
```

### Drift Detection

```python
from dataguard.validation import DriftDetector

# Fit on reference data
detector = DriftDetector()
detector.fit(reference_df)

# Detect drift in new data
drift_report = detector.detect(current_df)
if drift_report.has_drift:
    print(f"Drifted columns: {drift_report.drifted_columns}")
```

### Safe Cross-Validation

```python
from dataguard.leakage import SafeCrossValidator, safe_train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Safe CV with per-fold preprocessing
cv = SafeCrossValidator(
    n_splits=5,
    preprocessor=StandardScaler()
)

# Get safe train/val splits
for X_train, X_val, y_train, y_val in cv.split(X, y):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print(f"Score: {model.score(X_val, y_val):.4f}")

# Or use safe_train_test_split with leakage checks
X_train, X_test, y_train, y_test = safe_train_test_split(
    X, y, 
    test_size=0.2,
    check_leakage=True
)
```

### Leakage Detection

```python
from dataguard.leakage import LeakageDetector

detector = LeakageDetector()

# Check for target leakage
warnings = detector.check_target_leakage(X_train, y_train)

# Full check
all_warnings = detector.full_check(X_train, X_test, y_train, y_test)
for warning in all_warnings:
    print(warning)
```

## Modules

| Module | Description |
|--------|-------------|
| `dataguard.pii` | PII detection and masking |
| `dataguard.validation` | Schema validation, quality checks, drift detection |
| `dataguard.leakage` | Safe CV, splitting, leakage detection |

## Roadmap

- [ ] `dataguard.synthetic` — Synthetic data generation
- [ ] `dataguard.privacy` — Differential privacy utilities
- [ ] `dataguard.encryption` — Homomorphic encryption helpers
- [ ] `dataguard.pipeline` — Full sklearn pipeline integration

## Contributing

Contributions welcome! Please read our contributing guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.
