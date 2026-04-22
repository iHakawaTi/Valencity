"""
╔══════════════════════════════════════════════════════════════════════╗
║            🛡️  VALENCITY — Full Feature Demo (v0.1.1)                ║
║         pip install valencity  →  python demo.py                    ║
╚══════════════════════════════════════════════════════════════════════╝

Run this script to showcase every major feature of Valencity.
Perfect for LinkedIn demos, talks, or onboarding new team members.
"""

import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── helpers ──────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table

    console = Console()
    USE_RICH = True
except ImportError:
    USE_RICH = False
    console = None  # type: ignore


def header(title: str, emoji: str = "🛡️") -> None:
    if USE_RICH:
        console.print()
        console.print(Rule(f"[bold cyan]{emoji}  {title}[/bold cyan]"))
    else:
        print(f"\n{'='*60}")
        print(f"  {emoji}  {title}")
        print("=" * 60)


def ok(msg: str) -> None:
    if USE_RICH:
        console.print(f"  [bold green]✅[/bold green]  {msg}")
    else:
        print(f"  ✅  {msg}")


def warn(msg: str) -> None:
    if USE_RICH:
        console.print(f"  [bold yellow]⚠️ [/bold yellow]  {msg}")
    else:
        print(f"  ⚠️   {msg}")


def err(msg: str) -> None:
    if USE_RICH:
        console.print(f"  [bold red]🚨[/bold red]  {msg}")
    else:
        print(f"  🚨  {msg}")


def info(msg: str) -> None:
    if USE_RICH:
        console.print(f"  [dim]{msg}[/dim]")
    else:
        print(f"     {msg}")


def pause() -> None:
    time.sleep(0.4)


# ── demo data ────────────────────────────────────────────────────────
def make_sample_df() -> pd.DataFrame:
    """Realistic dataset with PII and quality issues."""
    rng = np.random.default_rng(42)
    n = 50

    df = pd.DataFrame(
        {
            "id": range(1, n + 1),
            "name": [
                "Alice Johnson",
                "Bob Martinez",
                "Carol White",
                "David Lee",
                "Eve Thompson",
            ]
            * 10,
            "email": [
                "alice@example.com",
                "bob.m@company.org",
                "carol.white@mail.com",
                "david@test.io",
                "eve@example.net",
            ]
            * 10,
            "phone": [
                "555-0101",
                "+1-555-0102",
                "(555) 010-3000",
                "555.0104",
                "555-0105",
            ]
            * 10,
            "ip_address": [
                "192.168.1.1",
                "10.0.0.5",
                "172.16.0.1",
                "8.8.8.8",
                "1.1.1.1",
            ]
            * 10,
            "salary": rng.normal(75_000, 15_000, n).clip(30_000, 200_000),
            "age": rng.integers(22, 65, n),
            "score": rng.uniform(0, 1, n),
            "joined": pd.date_range("2020-01-01", periods=n, freq="W"),
            "region": rng.choice(["NA", "EU", "APAC"], n),
            "churn": rng.choice([0, 1], n, p=[0.8, 0.2]),
        }
    )

    # Inject quality issues
    df.loc[0:4, "salary"] = np.nan
    df.loc[5:7, "email"] = np.nan
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)  # duplicates
    return df


# ─────────────────────────────────────────────────────────────────────
# STEP 1 — PII Detection & Masking
# ─────────────────────────────────────────────────────────────────────
def demo_pii(df: pd.DataFrame) -> pd.DataFrame:
    header("PII Detection & Anonymization", "🕵️")

    from valencity.pii import PIIDetector, PIIMasker

    detector = PIIDetector()
    report = detector.scan_dataframe(df)
    pause()

    if report.has_pii:
        err(f"PII detected in {len(report.pii_columns)} column(s): {report.pii_columns}")
        for col, findings in report.details.items():
            for f in findings[:2]:  # show first 2 per column
                info(f"[{col}]  {f.pii_type}  →  '{f.value}'")
    else:
        ok("No PII found.")

    # Mask it
    for strategy in ("redact", "partial", "hash"):
        masker = PIIMasker(strategy=strategy)
        safe = masker.mask_dataframe(df)
        sample = safe["email"].iloc[0]
        ok(f"strategy='{strategy}'  →  email becomes  '{sample}'")
        pause()

    # Return partially-masked version for downstream demos
    masker = PIIMasker(strategy="partial")
    return masker.mask_dataframe(df)


# ─────────────────────────────────────────────────────────────────────
# STEP 2 — Data Profiling
# ─────────────────────────────────────────────────────────────────────
def demo_profiling(df: pd.DataFrame) -> None:
    header("Data Profiling", "📊")

    from valencity.validation import DataProfiler

    profiler = DataProfiler()
    profile = profiler.profile(df)
    pause()

    ok(f"Rows: {profile.row_count}  |  Columns: {profile.column_count}")
    ok(f"Missing values total: {profile.total_missing}")

    if USE_RICH:
        table = Table(title="Column Profiles", show_lines=True)
        table.add_column("Column", style="cyan")
        table.add_column("Type")
        table.add_column("Nulls")
        table.add_column("Unique")
        for col_name, col_stats in list(profile.columns.items())[:6]:
            table.add_row(
                col_name,
                str(col_stats.dtype),
                str(col_stats.null_count),
                str(col_stats.unique_count),
            )
        console.print(table)
    else:
        for col_name, col_stats in list(profile.columns.items())[:6]:
            print(
                f"  {col_name}: dtype={col_stats.dtype}, "
                f"nulls={col_stats.null_count}, unique={col_stats.unique_count}"
            )


# ─────────────────────────────────────────────────────────────────────
# STEP 3 — Schema Validation
# ─────────────────────────────────────────────────────────────────────
def demo_schema(df: pd.DataFrame) -> None:
    header("Schema Validation", "✅")

    from valencity.validation import DataSchema

    train_df = df.iloc[:30].copy()
    prod_df = df.iloc[30:].copy()

    schema = DataSchema.from_dataframe(train_df)
    ok(f"Schema inferred from {len(train_df)} training rows")
    pause()

    result = schema.validate(prod_df)
    if result.is_valid:
        ok("Production DataFrame passes all schema checks!")
    else:
        warn(f"{len(result.errors)} schema violation(s) detected:")
        for e in result.errors[:3]:
            info(f"  Column '{e.column}': {e.message}")


# ─────────────────────────────────────────────────────────────────────
# STEP 4 — Data Quality Checks
# ─────────────────────────────────────────────────────────────────────
def demo_quality(df: pd.DataFrame) -> None:
    header("Data Quality Checks", "🔍")

    from valencity.validation import DataQualityChecker

    checker = DataQualityChecker()
    quality = checker.check(df)
    pause()

    ok(f"Null rate:      {quality.null_rate:.1%}")
    ok(f"Duplicates:     {quality.duplicate_count}")
    ok(f"Outlier cols:   {quality.outlier_columns}")
    ok(f"Quality score:  {quality.quality_score:.1%}")

    if quality.issues:
        warn(f"{len(quality.issues)} issue(s) found:")
        for issue in quality.issues[:3]:
            info(f"  [{issue.severity}] {issue.column}: {issue.message}")
    else:
        ok("No quality issues found!")


# ─────────────────────────────────────────────────────────────────────
# STEP 5 — Drift Detection
# ─────────────────────────────────────────────────────────────────────
def demo_drift(df: pd.DataFrame) -> None:
    header("Data Drift Detection", "📈")

    from valencity.validation import DriftDetector

    reference = df.iloc[:30].copy()
    # Simulate drifted production data
    drifted = df.iloc[30:].copy()
    rng = np.random.default_rng(99)
    drifted["salary"] = drifted["salary"] * rng.uniform(1.2, 1.8, len(drifted))

    for method in ("ks_test", "psi"):
        detector = DriftDetector(method=method)
        detector.fit(reference_df=reference)
        result = detector.detect(current_df=drifted)
        pause()

        if result.has_drift:
            err(f"[{method}] Drift in: {result.drifted_columns}")
        else:
            ok(f"[{method}] No drift detected")


# ─────────────────────────────────────────────────────────────────────
# STEP 6 — Fluent Expectations API
# ─────────────────────────────────────────────────────────────────────
def demo_expectations(df: pd.DataFrame) -> None:
    header("Expectations API", "📋")

    from valencity.validation import expect

    pause()
    result = (
        expect(df)
        .column("age").to_be_between(18, 100)
        .column("score").to_be_between(0.0, 1.0)
        .column("region").to_be_in(["NA", "EU", "APAC"])
        .validate()
    )

    if result.passed:
        ok("All expectations passed!")
    else:
        warn(f"{result.failed_count} expectation(s) failed:")
        for failure in result.failures[:3]:
            info(f"  {failure.column}: {failure.message}")


# ─────────────────────────────────────────────────────────────────────
# STEP 7 — Leakage Detection & Safe CV
# ─────────────────────────────────────────────────────────────────────
def demo_leakage(df: pd.DataFrame) -> None:
    header("ML Leakage Prevention", "🚫")

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    from valencity.leakage import LeakageDetector, SafeCrossValidator, SafeSplitter

    feature_cols = ["salary", "age", "score"]
    X = df[feature_cols].fillna(df[feature_cols].median())
    y = df["churn"]

    # Leakage detection
    X_train = X.iloc[:40]
    X_test = X.iloc[40:]
    y_train = y.iloc[:40]

    ld = LeakageDetector()
    issues = ld.detect(X_train, X_test, y_train)
    pause()

    if issues:
        err(f"Leakage detected: {issues}")
    else:
        ok("No data leakage detected between train and test sets")

    # Safe cross-validation
    cv = SafeCrossValidator(n_splits=5, preprocessor=StandardScaler())
    scores = cv.cross_val_score(LogisticRegression(max_iter=1000), X, y)
    pause()
    ok(
        f"Safe CV scores: {scores.mean():.4f} ± {scores.std():.4f}  "
        f"(no preprocessing leakage)"
    )

    # Safe splitter
    splitter = SafeSplitter(strategy="time_series")
    X_tr, X_te, y_tr, y_te = splitter.split(X, y, time_col=None)
    ok(
        f"SafeSplitter → train={len(X_tr)} rows, test={len(X_te)} rows "
        f"(time-series aware)"
    )


# ─────────────────────────────────────────────────────────────────────
# STEP 8 — Differential Privacy & Compliance
# ─────────────────────────────────────────────────────────────────────
def demo_privacy(df: pd.DataFrame) -> None:
    header("Privacy Engineering", "🔐")

    from valencity.privacy import ComplianceChecker, DifferentialPrivacy

    # Differential Privacy
    dp = DifferentialPrivacy(epsilon=1.0)
    true_mean = df["salary"].dropna().mean()
    noisy_mean = dp.add_noise(true_mean, sensitivity=1000)
    pause()
    ok(f"True mean salary:  ${true_mean:,.0f}")
    ok(f"DP-protected mean: ${noisy_mean:,.0f}  (ε=1.0, Laplace noise)")

    # Compliance check
    checker = ComplianceChecker()
    compliance = checker.check(df)
    pause()

    if compliance.is_compliant:
        ok("GDPR / CCPA: Fully compliant ✅")
    else:
        warn(f"Found {len(compliance.violations)} compliance issue(s):")
        for v in compliance.violations[:3]:
            info(f"  [{v.regulation}] {v.column}: {v.description}")
            if v.suggestion:
                info(f"    → Fix: {v.suggestion}")


# ─────────────────────────────────────────────────────────────────────
# STEP 9 — Synthetic Data Generation
# ─────────────────────────────────────────────────────────────────────
def demo_synthetic(df: pd.DataFrame) -> None:
    header("Synthetic Data Generation", "🧬")

    from valencity.synthetic import SyntheticGenerator

    gen = SyntheticGenerator()
    synthetic = gen.generate(template_df=df[["salary", "age", "score", "region"]], n_rows=500)
    pause()

    ok(f"Generated {len(synthetic)} synthetic rows (no real data used)")
    ok(f"Mean salary (real):      ${df['salary'].dropna().mean():,.0f}")
    ok(f"Mean salary (synthetic): ${synthetic['salary'].mean():,.0f}")


# ─────────────────────────────────────────────────────────────────────
# STEP 10 — HTML Reports
# ─────────────────────────────────────────────────────────────────────
def demo_reports(df: pd.DataFrame) -> None:
    header("HTML Report Generation", "📄")

    from valencity.pii import PIIDetector
    from valencity.reports import HTMLGenerator

    pii_report = PIIDetector().scan_dataframe(df)
    gen = HTMLGenerator()
    out = "demo_pii_report.html"
    gen.generate_pii_report(pii_report, output_path=out)
    pause()
    ok(f"PII report saved → {out}  (open in browser 🌐)")


# ─────────────────────────────────────────────────────────────────────
# STEP 11 — Async PII Detection
# ─────────────────────────────────────────────────────────────────────
def demo_async(df: pd.DataFrame) -> None:
    header("Async PII Detection", "⚡")

    import asyncio

    from valencity.pii import AsyncPIIDetector

    async def _run() -> None:
        detector = AsyncPIIDetector()
        report = await detector.scan_dataframe(df)
        if report.has_pii:
            err(f"[Async] PII in: {report.pii_columns}")
        else:
            ok("[Async] No PII found")

    asyncio.run(_run())
    ok("Non-blocking scan complete — safe to use in async APIs / FastAPI routes")


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────
def main() -> None:
    if USE_RICH:
        console.print(
            Panel.fit(
                "[bold cyan]🛡️  VALENCITY — Full Feature Demo[/bold cyan]\n"
                "[dim]ML Safety Toolkit · Privacy Engineering · v0.1.1[/dim]\n"
                "[dim]pip install valencity   |   github.com/ihakawati/Valencity[/dim]",
                border_style="cyan",
            )
        )
    else:
        print(__doc__)

    df = make_sample_df()
    info(f"Sample dataset ready: {df.shape[0]} rows × {df.shape[1]} columns")

    safe_df = demo_pii(df)
    demo_profiling(df)
    demo_schema(df)
    demo_quality(df)
    demo_drift(df)
    demo_expectations(df)
    demo_leakage(df)
    demo_privacy(df)
    demo_synthetic(df)
    demo_reports(df)
    demo_async(df)

    header("All demos complete!", "🎉")
    if USE_RICH:
        console.print(
            Panel(
                "[bold green]Valencity covers every layer of ML safety:[/bold green]\n\n"
                "  🕵️  PII detection & masking\n"
                "  📊  Data profiling\n"
                "  ✅  Schema & quality validation\n"
                "  📈  Drift detection\n"
                "  🚫  Leakage prevention (CV & splits)\n"
                "  🔐  Differential privacy & compliance\n"
                "  🧬  Synthetic data\n"
                "  📄  HTML reports\n"
                "  ⚡  Async support\n\n"
                "[dim]pip install valencity   |   Star us on GitHub ⭐[/dim]",
                title="[bold]🛡️ Valencity[/bold]",
                border_style="green",
            )
        )
    else:
        print("\n  Valencity — pip install valencity")
        print("  github.com/ihakawati/Valencity\n")


if __name__ == "__main__":
    main()
