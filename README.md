<div align="center">

# 🛡️ Valencity

**The ML Safety Fortress**  
*Privacy Engineering · Data Validation · Leakage Prevention*

[![PyPI version](https://img.shields.io/pypi/v/valencity.svg?color=blueviolet&style=for-the-badge)](https://pypi.org/project/valencity)
[![Python versions](https://img.shields.io/pypi/pyversions/valencity.svg?style=for-the-badge)](https://pypi.org/project/valencity)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![PyPI Downloads](https://img.shields.io/pypi/dm/valencity.svg?style=for-the-badge&label=Downloads)](https://pypi.org/project/valencity)

<p align="center">
  <a href="#-why-valencity">Why Valencity</a> •
  <a href="#-features">Features</a> •
  <a href="#-installation">Install</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-full-api">Full API</a> •
  <a href="#-cli">CLI</a> •
  <a href="#-roadmap">Roadmap</a>
</p>

> **"Stop shipping ML models that leak PII or data. Valencity catches it before you do."**

</div>

---

## 🚀 Why Valencity?

Most ML teams discover data privacy problems in **production** — when it's already too late.  
Valencity is the **safety net** that catches issues in your pipeline **before** they hit your users.

One line of code. Six layers of protection.

```python
pip install valencity
```

| Capability | Valencity 🛡️ | Great Expectations | Pandera | Custom Scripts |
|---|:---:|:---:|:---:|:---:|
| **PII Detection (50+ patterns)** | ✅ | ❌ | ❌ | ⚠️ Manual |
| **Masking & Anonymization** | ✅ | ❌ | ❌ | ❌ |
| **K-Anonymity & Differential Privacy** | ✅ | ❌ | ❌ | ❌ |
| **ML Leakage Prevention** | ✅ Full Suite | ❌ | ❌ | ❌ |
| **Data Drift Detection (4 methods)** | ✅ | ✅ | ❌ | ⚠️ Manual |
| **Schema + Quality Checks** | ✅ | ✅ | ✅ | ⚠️ Manual |
| **Synthetic Data Generation** | ✅ | ❌ | ❌ | ❌ |
| **GDPR / CCPA Compliance** | ✅ | ❌ | ❌ | ❌ |
| **HTML Reports** | ✅ Beautiful | ✅ | ❌ | ❌ |
| **CLI Tool** | ✅ | ❌ | ❌ | ❌ |

---

## ✨ Features

### 🕵️ PII Detection & Anonymization
- Detect **50+ PII types** — emails, phones, IBANs, passports, API keys, SSNs, IPs, and more
- **Smart Masking**: redact, hash (SHA-256), partial mask, or replace with realistic fake data
- **NLP support** for names and addresses via spaCy (optional)
- **Async detector** for non-blocking scans in production services
- **Plugin-friendly**: register your own custom PII patterns

### ✅ Data Validation & Quality
- **Data Profiling**: automatic statistical summary of any DataFrame
- **Schema Validation**: enforce strict data contracts — types, nullability, ranges
- **Quality Checks**: nulls, duplicates, outliers, cardinality, and more
- **Fluent Expectations API**: expressive, chainable validation rules
- **Drift Detection**: KS test, chi-squared, Jensen–Shannon divergence, and PSI

### 🚫 Leakage Prevention
- **SafeCrossValidator**: preprocessing *only* sees training fold — no more optimistic CV scores
- **LeakageDetector**: catches target leakage, train/test overlap, and temporal leakage
- **SafeSplitters**: time-series and group-aware train/test splits

### 🔐 Privacy Engineering
- **Differential Privacy**: add calibrated Laplace / Gaussian noise to statistics
- **Compliance Checker**: automated GDPR & CCPA violation detection with fix suggestions

### 🧬 Synthetic Data
- **SyntheticGenerator**: generate realistic fake datasets that mirror real data distributions

### 📊 Reporting
- Beautiful **HTML reports** for PII scans, quality audits, drift analysis, and compliance checks

---

## 📦 Installation

```bash
# Core (PII, validation, leakage, reports)
pip install valencity

# With NLP support (names, free-text addresses)
pip install valencity[nlp]
python -m spacy download en_core_web_sm

# Full developer install
pip install valencity[all]
```

**Requirements:** Python ≥ 3.9, pandas ≥ 1.5, scikit-learn ≥ 1.0

---

## ⚡ Quick Start

### 1 · Detect & Mask PII

```python
import pandas as pd
from valencity.pii import PIIDetector, PIIMasker

df = pd.read_csv("users.csv")

# Scan
detector = PIIDetector()
report = detector.scan_dataframe(df)

if report.has_pii:
    print(f"⚠️  PII detected in columns: {list(report.columns_with_pii.keys())}")
    for col, col_report in report.columns_with_pii.items():
        for pii_type in col_report.pii_types_found:
            print(f"   [{pii_type.name}]")

# Mask
masker = PIIMasker(strategy="partial")   # john@example.com → j***@example.com
safe_df = masker.mask_dataframe(df)
```

### 2 · Validate Schema & Quality

```python
from valencity.validation import DataSchema, DataQualityChecker, DataProfiler

# Auto-infer schema from training data
schema = DataSchema.from_dataframe(train_df)
validation_result = schema.validate(production_df)

# Quality report
checker = DataQualityChecker()
quality = checker.full_report(df)
print(quality.summary())

# Statistical profile
profiler = DataProfiler()
profile = profiler.profile(df)
print(profile.summary())
```

### 3 · Detect Data Drift

```python
from valencity.validation import DriftDetector

detector = DriftDetector(method="ks_test")   # ks_test | chi2 | js | psi
detector.fit(reference_df=train_df)

drift = detector.detect(current_df=production_df)
if drift.has_drift:
    print(f"🚨 Drift detected in: {drift.drifted_columns}")
```

### 4 · Prevent Model Leakage

```python
from valencity.leakage import SafeCrossValidator, LeakageDetector, temporal_train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Detect leakage before training
ld = LeakageDetector()
issues = ld.full_check(X_train, X_test, y_train, y_test)
if issues:
    print(f"🚨 Leakage found: {issues}")

# Safe CV — preprocessing never sees test fold
cv = SafeCrossValidator(n_splits=5, preprocessor=StandardScaler())
scores = cv.cross_val_score(LogisticRegression(), X, y)
print(f"✅ Realistic accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

# Safe time-series split
X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, time_column="date")
```

### 5 · Privacy Engineering

```python
from valencity.privacy import DifferentialPrivacy, ComplianceChecker

# Add calibrated noise to protect aggregates
dp = DifferentialPrivacy()
noisy_mean = dp.laplace_mechanism(value=df["salary"].mean(), sensitivity=1000, epsilon=1.0)

# Check GDPR / CCPA compliance
checker = ComplianceChecker()
compliance = checker.check_gdpr(df)
if not compliance.satisfied:
    for violation in compliance.violations:
        print(f"❌ {violation.rule}: {violation.description}")
```

### 6 · Generate Synthetic Data

```python
from valencity.synthetic import SyntheticGenerator

gen = SyntheticGenerator().from_dataframe(real_df)
synthetic_df = gen.generate(num_rows=1000)
```

### 7 · Generate HTML Reports

```python
from pathlib import Path
from valencity.reports import HTMLGenerator
from valencity.pii import PIIDetector

detector = PIIDetector()
pii_report = detector.scan_dataframe(df)

gen = HTMLGenerator()
gen.render_pii_report(pii_report, output_path=Path("pii_report.html"))
print("📊 Report saved → pii_report.html")
```

---

## 🖥️ Full API

```
import valencity
valencity.show_api()
```

| Module | Class / Function | Purpose |
|---|---|---|
| `valencity.pii` | `PIIDetector` | Scan DataFrames & text for PII |
| `valencity.pii` | `PIIMasker` | Mask / anonymize PII |
| `valencity.pii` | `PIIType` | Enum of 50+ supported PII types |
| `valencity.pii` | `PIIPatterns` | Pattern registry (plugin-friendly) |
| `valencity.pii` | `AsyncPIIDetector` | Async, non-blocking PII scanning |
| `valencity.validation` | `DataSchema` | Schema validation & contracts |
| `valencity.validation` | `DataQualityChecker` | Null / duplicate / outlier checks |
| `valencity.validation` | `DataProfiler` | Statistical profiling |
| `valencity.validation` | `DriftDetector` | KS, chi2, JS, PSI drift detection |
| `valencity.validation` | `expect()` | Fluent expectations API |
| `valencity.leakage` | `SafeCrossValidator` | Leakage-safe cross-validation |
| `valencity.leakage` | `SafeSplitter` | Time-series & group-aware splits |
| `valencity.leakage` | `LeakageDetector` | Target, overlap & temporal leakage |
| `valencity.privacy` | `DifferentialPrivacy` | Laplace / Gaussian DP noise |
| `valencity.privacy` | `ComplianceChecker` | GDPR / CCPA violation detection |
| `valencity.synthetic` | `SyntheticGenerator` | Realistic synthetic data |
| `valencity.reports` | `HTMLGenerator` | Beautiful HTML audit reports |

---

## 🖥️ CLI

```bash
# Show package info
valencity info

# Scan a CSV for PII
valencity pii scan users.csv

# Mask PII in a CSV
valencity pii mask users.csv --output safe_users.csv

# Generate a data profile
valencity profile dataset.csv

# Check GDPR/CCPA compliance
valencity compliance dataset.csv

# Run quality checks
valencity validate quality dataset.csv
```

---

## 🗺️ Roadmap

- [x] **PII Detection** — 50+ patterns, masking, async support
- [x] **Data Validation** — schema, quality, drift, profiling
- [x] **Leakage Prevention** — CV, splitters, detectors
- [x] **Privacy Engineering** — differential privacy, compliance
- [x] **Synthetic Data Generation**
- [x] **CLI Tool**
- [ ] **MLflow Integration** — auto-log safety checks to experiments
- [ ] **Airflow Operator** — drop-in pipeline safety checks
- [ ] **dbt Integration** — run Valencity checks in dbt tests
- [ ] **Dashboard UI** — real-time data safety monitoring

---

## 🤝 Contributing

Contributions are welcome! Check the [open issues](https://github.com/ihakawati/valencity/issues).

```bash
git clone https://github.com/ihakawati/valencity
cd valencity
pip install -e ".[dev]"
pytest
```

1. Fork the repo
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit: `git commit -m "feat: add my feature"`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

<div align="center">

**Built with ❤️ for the ML Community**

[PyPI](https://pypi.org/project/valencity) · [GitHub](https://github.com/ihakawati/valencity) · [Report a Bug](https://github.com/ihakawati/valencity/issues)

</div>
