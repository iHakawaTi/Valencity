# 🛡️ valencity

**The ML Safety Fortress**

valencity is a comprehensive Python toolkit designed to make machine learning pipelines **safer**, **compliant**, and **reliable**. It bridges the gap between data engineering and privacy compliance.

## Key Features

- **🕵️ PII Detection**: Identify 50+ patterns including IBANs, Passports, and API Keys.
- **🛡️ Privacy Engineering**: Apply Differential Privacy and check GDPR compliance.
- **✅ Data Validation**: Automated profiling, schema enforcement, and drift detection.
- **🧪 Synthetic Data**: Generate realistic, safe datasets for testing.
- **🚫 Leakage Prevention**: Safe cross-validation and rigorous splitters.

## Installation

```bash
pip install valencity
```

## Quick Start

### Scan for PII

```python
from valencity.pii import PIIDetector

df = pd.read_csv("data.csv")
report = PIIDetector().scan_dataframe(df)

if report.has_pii:
    print(f"PII found in: {report.pii_columns}")
```
