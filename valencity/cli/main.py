"""
valencity Command Line Interface.
"""

import glob
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich import print as rprint
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from valencity import __version__
from valencity.pii import MaskingStrategy, PIIDetector, PIIMasker
from valencity.privacy import ComplianceChecker
from valencity.reports.html import HTMLGenerator
from valencity.validation import DataProfiler, DataQualityChecker, DriftDetector

app = typer.Typer(
    name="valencity",
    help="🛡️ valencity: The ML Safety Fortress - PII, Validation & Leakage Prevention",
    add_completion=False,
)
pii_app = typer.Typer(name="pii", help="PII detection and masking commands")
validate_app = typer.Typer(name="validate", help="Data validation commands")

app.add_typer(pii_app, name="pii")
app.add_typer(validate_app, name="validate")

console = Console()

def load_data(file_pattern: str) -> pd.DataFrame:
    """Load data from CSV or Parquet files, supporting glob patterns."""
    files = glob.glob(file_pattern)
    if not files:
        # Try as direct path if glob returned nothing (though glob handles exact paths too)
        path = Path(file_pattern)
        if path.exists():
            files = [str(path)]
        else:
            rprint(f"[bold red]Error:[/bold red] No files found matching: {file_pattern}")
            raise typer.Exit(code=1)
            
    dfs = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task(f"Loading {len(files)} file(s)...", total=len(files))
        
        for f in files:
            path = Path(f)
            try:
                if path.suffix == ".csv":
                    dfs.append(pd.read_csv(path))
                elif path.suffix == ".parquet":
                    dfs.append(pd.read_parquet(path))
                else:
                    rprint(f"[bold yellow]Warning:[/bold yellow] Skipping unsupported file: {path}")
            except Exception as e:
                rprint(f"[bold red]Error loading {path}:[/bold red] {e}")
            progress.advance(task)
    
    if not dfs:
        rprint("[bold red]Error:[/bold red] No data loaded.")
        raise typer.Exit(code=1)
        
    if len(dfs) > 1:
        valid_dfs = [d for d in dfs if not d.empty]
        if not valid_dfs:
             rprint("[bold red]Error:[/bold red] All loaded files were empty.")
             raise typer.Exit(code=1)
        # Using concat for batch processing - might want to handle varied schemas later
        return pd.concat(valid_dfs, ignore_index=True)
        
    return dfs[0]

@app.command()
def version():
    """Show valencity version."""
    rprint(f"[bold blue]valencity {__version__}[/bold blue]")


@app.command()
def info():
    """Show valencity information."""
    rprint(f"[bold blue]🛡️ valencity v{__version__}[/bold blue]")
    rprint("A privacy & ML safety toolkit for production pipelines.")

@app.command()
def profile(
    file_pattern: str = typer.Argument(..., help="Path or glob pattern to data files"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to save HTML report"),
):
    """Generate a statistical profile of the data."""
    df = load_data(file_pattern)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task(f"Profiling {len(df)} rows...", total=None)
        profiler = DataProfiler()
        report = profiler.profile(df)
        
    rprint(f"[bold green]Profile Generated[/bold green] ({report.total_rows} rows, {report.total_columns} cols)")
    
    if output:
        generator = HTMLGenerator()
        generator.render_profile_report(report, output)
        rprint(f"[bold green]✅ HTML report saved to {output}[/bold green]")
        
    # Print summary to console
    print(report.summary())

@app.command()
def compliance(
    file_pattern: str = typer.Argument(..., help="Path or glob pattern to data files"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Path to save HTML report"),
):
    """Check dataset for privacy compliance violations (GDPR)."""
    df = load_data(file_pattern)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Running compliance checks...", total=None)
        checker = ComplianceChecker()
        report = checker.check_gdpr(df)
        
    if output:
        generator = HTMLGenerator()
        generator.render_compliance_report(report, output)
        rprint(f"[bold green]✅ HTML report saved to {output}[/bold green]")
        
    if report.satisfied:
        rprint("[bold green]✅ No automated compliance violations found.[/bold green]")
    else:
        rprint(f"[bold red]❌ Found {len(report.violations)} potential violations:[/bold red]")
        for v in report.violations:
            color = "red" if v.severity == "High" else "yellow" if v.severity == "Medium" else "blue"
            rprint(f"[{color}]• [{v.severity}] {v.rule}: {v.description}[/{color}]")

@pii_app.command("scan")
def scan_pii(
    file_pattern: str = typer.Argument(..., help="Path or glob pattern to data files"),
    sample_size: int = typer.Option(5, help="Number of sample matches to show"),
    output: Optional[Path] = typer.Option(None, help="Path to save HTML report"),
):
    """Scan a file for PII."""
    df = load_data(file_pattern)
    
    rprint(f"[bold]Scanning {len(df)} rows...[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Scanning for PII...", total=None)
        detector = PIIDetector(sample_size=sample_size)
        report = detector.scan_dataframe(df)
    
    # Generate HTML report if requested
    if output:
        generator = HTMLGenerator()
        generator.render_pii_report(report, output)
        rprint(f"[bold green]✅ HTML report saved to {output}[/bold green]")
    
    if not report.has_pii:
        rprint("[bold green]✅ No PII detected![/bold green]")
        return

    rprint(f"[bold red]⚠️ Found PII in {len(report.pii_columns)} columns:[/bold red]")
    
    table = Table(title="PII Detection Report")
    table.add_column("Column", style="cyan")
    table.add_column("Types Found", style="magenta")
    table.add_column("Match Count", style="yellow")
    table.add_column("% Rows", style="blue")
    
    for col_name in report.pii_columns:
        col_report = report.columns_with_pii[col_name]
        types = ", ".join(t.value for t in col_report.pii_types_found)
        pct = col_report.pii_percentage
        
        table.add_row(
            col_name,
            types,
            str(col_report.match_count),
            f"{pct:.1f}%"
        )
        
    console.print(table)

@pii_app.command("mask")
def mask_pii(
    file_path: Path = typer.Argument(..., help="Path to input file (Single file only for now)"),
    output: Path = typer.Option(None, help="Path to save masked file"),
    strategy: str = typer.Option("redact", help="Masking strategy: redact, hash, partial, fake"),
):
    """Mask PII in a file."""
    # Masking is safer one file at a time for now to control output paths explicitly
    df = load_data(str(file_path))
    
    # Map strategy string to Enum
    try:
        strat_enum = MaskingStrategy(strategy)
    except ValueError:
        rprint(f"[bold red]Error:[/bold red] Invalid strategy '{strategy}'. Options: redact, hash, partial, fake")
        raise typer.Exit(code=1)
        
    rprint(f"[bold]Masking PII in {file_path.name} using '{strategy}' strategy...[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Masking data...", total=None)
        masker = PIIMasker(strategy=strat_enum)
        masked_df = masker.mask_dataframe(df)
    
    if output:
        if output.suffix == ".csv":
            masked_df.to_csv(output, index=False)
        elif output.suffix == ".parquet":
            masked_df.to_parquet(output, index=False)
        rprint(f"[bold green]✅ Saved masked data to {output}[/bold green]")
    else:
        rprint("[yellow]No output path provided. Printing first 5 rows:[/yellow]")
        rprint(masked_df.head())

@validate_app.command("quality")
def check_quality(
    file_pattern: str = typer.Argument(..., help="Path or glob pattern to data files"),
    report_path: Optional[Path] = typer.Option(None, "--report", "-r", help="Path to save HTML report"),
):
    """Run data quality checks."""
    df = load_data(file_pattern)
    
    rprint(f"[bold]Checking data quality for {len(df)} rows...[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Running quality checks...", total=None)
        checker = DataQualityChecker()
        report = checker.full_report(df)
    
    # Generate HTML report if requested
    if report_path:
        generator = HTMLGenerator()
        generator.render_quality_report(report, report_path)
        rprint(f"[bold green]✅ HTML report saved to {report_path}[/bold green]")
    
    if report.passed:
        rprint("[bold green]✅ All quality checks passed![/bold green]")
    else:
        rprint("[bold red]❌ Quality checks failed:[/bold red]")
        
    # Failed checks
    if report.failed_checks:
        table = Table(title="Failed Checks", style="red")
        table.add_column("Column", style="cyan")
        table.add_column("Check", style="magenta")
        table.add_column("Message", style="white")
        
        for check in report.failed_checks:
            table.add_row(
                check.column or "Global",
                check.check_type.value,
                check.message
            )
        console.print(table)
        
    # Stats
    rprint(f"\n[dim]Analyzed {report.total_rows} rows, {report.total_columns} columns[/dim]")


@validate_app.command("drift")
def check_drift(
    reference: str = typer.Option(..., "--ref", help="Reference dataset (baseline)"),
    current: str = typer.Option(..., "--cur", help="Current dataset to check"),
    method: str = typer.Option("ks_test", "--method", "-m", help="Drift detection method: ks_test, psi, js, chi2"),
    threshold: float = typer.Option(0.05, "--threshold", "-t", help="Drift threshold"),
):
    """Detect data drift between reference and current datasets."""
    ref_df = load_data(reference)
    cur_df = load_data(current)

    rprint(f"[bold]Detecting drift (method: {method})...[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        progress.add_task("Analyzing drift...", total=None)
        detector = DriftDetector(method=method, threshold=threshold)
        detector.fit(ref_df)
        report = detector.detect(cur_df)

    if not report.has_drift:
        rprint("[bold green]PASS: No significant drift detected[/bold green]")
        return

    rprint(f"[bold red]WARN: Drift detected in {len(report.drifted_columns)} columns:[/bold red]")

    table = Table(title="Drift Report")
    table.add_column("Column", style="cyan")
    table.add_column("Method", style="magenta")
    table.add_column("Score", style="yellow")
    table.add_column("Drifted", style="red")

    for col_name, col_result in report.columns.items():
        table.add_row(
            col_name,
            col_result.method,
            f"{col_result.score:.4f}",
            "Yes" if col_result.has_drift else "No"
        )

    console.print(table)


if __name__ == "__main__":
    app()
