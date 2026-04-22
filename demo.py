"""
Valencity v0.1.2 - Full Feature Demo
======================================
Run:  python demo.py
pip install valencity
"""
import os
import sys

# Fix Windows terminal encoding (emoji in logging would crash otherwise)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import logging
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)  # silence internal logger emoji on Windows

import numpy as np
import pandas as pd

# ── rich output (optional but pretty) ───────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table

    console = Console(highlight=False)
    USE_RICH = True
except ImportError:
    USE_RICH = False
    console = None  # type: ignore


def header(title: str, emoji: str = "") -> None:
    if USE_RICH:
        console.print()
        console.print(Rule(f"[bold cyan]{emoji}  {title}[/bold cyan]"))
    else:
        print(f"\n{'='*60}\n  {emoji}  {title}\n{'='*60}")


def ok(msg: str) -> None:
    if USE_RICH:
        console.print(f"  [bold green]OK[/bold green]  {msg}")
    else:
        print(f"  OK  {msg}")


def warn(msg: str) -> None:
    if USE_RICH:
        console.print(f"  [bold yellow]!![/bold yellow]  {msg}")
    else:
        print(f"  !!  {msg}")


def info(msg: str) -> None:
    if USE_RICH:
        console.print(f"  [dim]{msg}[/dim]")
    else:
        print(f"     {msg}")


# ── demo data ────────────────────────────────────────────────────────
def make_df() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame(
        {
            "id": range(1, n + 1),
            "name": ["Alice Johnson", "Bob Martinez", "Carol White", "David Lee", "Eve Thompson"] * 10,
            "email": ["alice@example.com", "bob.m@company.org", "carol@mail.com", "david@test.io", "eve@example.net"] * 10,
            "phone": ["555-0101", "+1-555-0102", "(555) 010-3000", "555.0104", "555-0105"] * 10,
            "ip_address": ["192.168.1.1", "10.0.0.5", "172.16.0.1", "8.8.8.8", "1.1.1.1"] * 10,
            "salary": rng.normal(75_000, 15_000, n).clip(30_000, 200_000),
            "age": rng.integers(22, 65, n),
            "score": rng.uniform(0, 1, n),
            "date": pd.date_range("2020-01-01", periods=n, freq="W"),
            "region": rng.choice(["NA", "EU", "APAC"], n),
            "churn": rng.choice([0, 1], n, p=[0.8, 0.2]),
        }
    )
    # Inject quality issues for demos
    df.loc[0:4, "salary"] = np.nan
    df.loc[5:7, "email"] = np.nan
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)  # duplicates
    return df


# ─────────────────────────────────────────────────────────────────────
# 1. PII Detection & Masking
# ─────────────────────────────────────────────────────────────────────
def demo_pii(df: pd.DataFrame) -> None:
    header("PII Detection & Masking", "[1/8]")

    from valencity.pii import PIIDetector, PIIMasker

    detector = PIIDetector()
    report = detector.scan_dataframe(df)

    if report.has_pii:
        warn(f"PII detected in {len(report.pii_columns)} column(s): {report.pii_columns}")
        ok(f"Total PII matches found: {report.total_matches}")
        for col in report.pii_columns[:3]:
            info(f"  Column '{col}' contains PII")
    else:
        ok("No PII found.")

    # Redact strategy
    masker_redact = PIIMasker(strategy="redact")
    safe = masker_redact.mask_dataframe(df)
    ok(f"Redacted  -> email: '{safe['email'].iloc[0]}'")

    # Hash strategy
    masker_hash = PIIMasker(strategy="hash")
    safe_hash = masker_hash.mask_dataframe(df)
    ok(f"Hashed    -> email: '{safe_hash['email'].iloc[0][:20]}...'")

    # Fake replacement
    masker_fake = PIIMasker(strategy="fake")
    safe_fake = masker_fake.mask_dataframe(df)
    ok(f"Fake data -> email: '{safe_fake['email'].iloc[0]}'")


# ─────────────────────────────────────────────────────────────────────
# 2. Data Profiling
# ─────────────────────────────────────────────────────────────────────
def demo_profiling(df: pd.DataFrame) -> None:
    header("Data Profiling", "[2/8]")

    from valencity.validation import DataProfiler

    profile = DataProfiler().profile(df)

    ok(f"Rows: {profile.total_rows}  |  Columns: {profile.total_columns}")

    if USE_RICH:
        table = Table(title="Column Stats", show_lines=True)
        table.add_column("Column", style="cyan")
        table.add_column("Type")
        table.add_column("Nulls")
        table.add_column("Unique")
        for col_name, cs in list(profile.columns.items())[:5]:
            table.add_row(col_name, str(cs.dtype), str(cs.missing_count), str(cs.unique_count))
        console.print(table)
    else:
        for col_name, cs in list(profile.columns.items())[:5]:
            print(f"  {col_name}: dtype={cs.dtype}, nulls={cs.missing_count}, unique={cs.unique_count}")


# ─────────────────────────────────────────────────────────────────────
# 3. Schema Validation
# ─────────────────────────────────────────────────────────────────────
def demo_schema(df: pd.DataFrame) -> None:
    header("Schema Validation", "[3/8]")

    from valencity.validation import DataSchema

    train_df = df.iloc[:30].copy()
    prod_df = df.iloc[30:].copy()

    schema = DataSchema.from_dataframe(train_df)
    ok(f"Schema inferred from {len(train_df)} training rows")

    result = schema.validate(prod_df)
    if result.is_valid:
        ok("Production data passes all schema checks!")
    else:
        warn(f"{len(result.errors)} schema violation(s) found:")
        for e in result.errors[:3]:
            info(f"  {e}")


# ─────────────────────────────────────────────────────────────────────
# 4. Data Quality Checks
# ─────────────────────────────────────────────────────────────────────
def demo_quality(df: pd.DataFrame) -> None:
    header("Data Quality Checks", "[4/8]")

    from valencity.validation import DataQualityChecker

    dqc = DataQualityChecker()
    report = dqc.full_report(df)

    ok(f"Total rows: {report.total_rows}  |  Columns: {report.total_columns}")
    ok(f"All checks passed: {report.passed}")

    failed = report.failed_checks
    if failed:
        warn(f"{len(failed)} check(s) failed:")
        for c in failed[:3]:
            info(f"  [{c.check_type.value}] '{c.column}': {c.message}")
    else:
        ok("No quality issues found!")

    warnings_list = report.warning_checks
    if warnings_list:
        info(f"  {len(warnings_list)} warning(s): {[w.column for w in warnings_list]}")


# ─────────────────────────────────────────────────────────────────────
# 5. Drift Detection
# ─────────────────────────────────────────────────────────────────────
def demo_drift(df: pd.DataFrame) -> None:
    header("Data Drift Detection", "[5/8]")

    from valencity.validation import DriftDetector
    from valencity.validation.drift import DriftMethod

    numeric_df = df[["salary", "age", "score"]].copy()
    reference = numeric_df.iloc[:30].dropna()
    # Simulate salary drift in production
    drifted = numeric_df.iloc[30:].copy()
    rng = np.random.default_rng(99)
    drifted["salary"] = drifted["salary"].fillna(75000) * rng.uniform(1.4, 1.9, len(drifted))

    for method in (DriftMethod.KS_TEST, DriftMethod.PSI):
        detector = DriftDetector(method=method)
        detector.fit(reference_df=reference)
        result = detector.detect(current_df=drifted.dropna())

        if result.has_drift:
            warn(f"[{method.value}] Drift in: {result.drifted_columns}")
        else:
            ok(f"[{method.value}] No significant drift detected")


# ─────────────────────────────────────────────────────────────────────
# 6. Fluent Expectations API
# ─────────────────────────────────────────────────────────────────────
def demo_expectations(df: pd.DataFrame) -> None:
    header("Expectations API", "[6/8]")

    from valencity.validation import expect

    result = (
        expect(df)
        .column("age").to_be_between(18, 100)
        .column("score").to_be_between(0.0, 1.0)
        .column("region").to_be_in(["NA", "EU", "APAC"])
        .run()
    )

    if result.passed:
        ok(f"All {len(result.results)} expectations passed!")
    else:
        warn(f"{len(result.failed_results)} expectation(s) failed:")
        for f in result.failed_results[:3]:
            info(f"  Column '{f.column}': {f.details}")


# ─────────────────────────────────────────────────────────────────────
# 7. Leakage Prevention & Safe CV
# ─────────────────────────────────────────────────────────────────────
def demo_leakage(df: pd.DataFrame) -> None:
    header("ML Leakage Prevention", "[7/8]")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from valencity.leakage import (
        LeakageDetector,
        SafeCrossValidator,
        safe_train_test_split,
        temporal_train_test_split,
    )

    feature_cols = ["salary", "age", "score"]
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df["churn"]

    # Safe split
    X_tr, X_te, y_tr, y_te = safe_train_test_split(X, y)
    ok(f"SafeSplit -> train={len(X_tr)} rows, test={len(X_te)} rows")

    # Leakage check
    ld = LeakageDetector()
    issues = ld.full_check(X_tr, X_te, y_tr, y_te)
    if issues:
        warn(f"{len(issues)} leakage warning(s) detected")
        for w in issues[:2]:
            info(f"  {w}")
    else:
        ok("No data leakage detected between train/test sets")

    # Safe cross-validation (preprocessing never sees test fold)
    cv = SafeCrossValidator(n_splits=5, preprocessor=StandardScaler())
    scores = cv.cross_val_score(LogisticRegression(max_iter=500), X, y)
    ok(f"Safe CV score: {scores.mean():.4f} +/- {scores.std():.4f}  (no preprocessing leakage)")

    # Temporal split (time-series aware)
    X_with_date = df[feature_cols + ["date"]].fillna(df[feature_cols].median())
    X_tr2, X_te2, y_tr2, y_te2 = temporal_train_test_split(X_with_date, y, time_column="date")
    ok(f"Temporal split -> train={len(X_tr2)}, test={len(X_te2)} (chronological order preserved)")


# ─────────────────────────────────────────────────────────────────────
# 8. Privacy: Differential Privacy + GDPR Compliance
# ─────────────────────────────────────────────────────────────────────
def demo_privacy(df: pd.DataFrame) -> None:
    header("Privacy Engineering", "[8/8]")

    from valencity.privacy import ComplianceChecker, DifferentialPrivacy

    # Differential Privacy
    dp = DifferentialPrivacy()
    true_mean = df["salary"].dropna().mean()
    noisy_mean = dp.laplace_mechanism(value=true_mean, sensitivity=1000, epsilon=1.0)
    ok(f"True mean salary:       ${true_mean:,.0f}")
    ok(f"DP-protected mean:      ${noisy_mean:,.0f}  (Laplace, epsilon=1.0)")

    # GDPR Compliance check
    report = ComplianceChecker().check_gdpr(df)
    if report.satisfied:
        ok("GDPR compliance check: PASSED")
    else:
        warn(f"GDPR: {len(report.violations)} violation(s) found:")
        for v in report.violations[:3]:
            info(f"  [{v.severity}] {v.rule}")
            info(f"    {v.description[:90]}...")


# ─────────────────────────────────────────────────────────────────────
# BONUS: Synthetic Data + HTML Report
# ─────────────────────────────────────────────────────────────────────
def demo_extras(df: pd.DataFrame) -> None:
    header("Bonus: Synthetic Data + HTML Report", "[+]")

    # Synthetic Data
    from valencity.synthetic import SyntheticGenerator

    numeric_df = df[["salary", "age", "score"]].dropna()
    gen = SyntheticGenerator().from_dataframe(numeric_df)
    synthetic = gen.generate(num_rows=500)
    ok(f"Generated {len(synthetic)} synthetic rows (zero real data exposed)")
    ok(f"Real salary mean:      ${numeric_df['salary'].mean():,.0f}")
    ok(f"Synthetic salary mean: ${synthetic['salary'].mean():,.0f}")

    # HTML Report
    from pathlib import Path

    from valencity.pii import PIIDetector
    from valencity.reports import HTMLGenerator

    pii_report = PIIDetector().scan_dataframe(df)
    out_path = Path("demo_pii_report.html")
    HTMLGenerator().render_pii_report(pii_report, output_path=out_path)
    ok(f"PII HTML report saved -> {out_path.resolve()}")

    # Async PII Detection
    import asyncio

    from valencity.pii import AsyncPIIDetector

    async def _async_scan() -> None:
        report = await AsyncPIIDetector().scan_dataframe(df)
        ok(f"Async PII scan complete  -> has_pii={report.has_pii}, columns={report.pii_columns}")

    asyncio.run(_async_scan())


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    if USE_RICH:
        console.print(
            Panel.fit(
                "[bold cyan]Valencity - Full Feature Demo[/bold cyan]\n"
                "[dim]ML Safety Toolkit | Privacy Engineering | v0.1.2[/dim]\n"
                "[dim]pip install valencity[/dim]",
                border_style="cyan",
            )
        )
    else:
        print("=" * 60)
        print("  VALENCITY - Full Feature Demo  |  v0.1.2")
        print("  pip install valencity")
        print("=" * 60)

    df = make_df()
    info(f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns (with injected PII & quality issues)")

    demo_pii(df)
    demo_profiling(df)
    demo_schema(df)
    demo_quality(df)
    demo_drift(df)
    demo_expectations(df)
    demo_leakage(df)
    demo_privacy(df)
    demo_extras(df)

    header("All demos complete!", "[DONE]")

    summary = (
        "Valencity covers:\n"
        "  [1] PII detection & masking (50+ patterns)\n"
        "  [2] Data profiling\n"
        "  [3] Schema validation\n"
        "  [4] Quality checks\n"
        "  [5] Drift detection (KS + PSI)\n"
        "  [6] Fluent expectations API\n"
        "  [7] Leakage prevention & safe CV\n"
        "  [8] Differential privacy & GDPR compliance\n"
        "  [+] Synthetic data + HTML reports + async support\n\n"
        "  pip install valencity  |  github.com/iHakawaTi/Valencity"
    )

    if USE_RICH:
        console.print(Panel(summary, title="[bold]Valencity[/bold]", border_style="green"))
    else:
        print(summary)


if __name__ == "__main__":
    main()
