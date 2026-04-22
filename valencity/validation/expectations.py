"""
Expectations API for fluent data validation.

This module provides a "Great Expectations"-style API for checking
properties of data with a readable, fluent syntax.

Example:
    >>> from valencity.validation import expect
    >>> result = (
    ...     expect(df)
    ...     .column("age").to_be_between(0, 150)
    ...     .column("email").to_match_regex(r".*@.*")
    ...     .run()
    ... )
"""

from dataclasses import dataclass, field
from re import Pattern
from typing import Any, List, Union

import pandas as pd

from valencity.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ExpectationResult:
    """Result of a single expectation check."""
    
    column: str
    expectation: str
    passed: bool
    details: str
    
    def __str__(self):
        status = "✅" if self.passed else "❌"
        return f"{status} Column '{self.column}' {self.expectation}: {self.details}"


@dataclass
class ExpectationReport:
    """Report containing all expectation results."""
    
    results: List[ExpectationResult] = field(default_factory=list)
    
    @property
    def passed(self) -> bool:
        """True if all expectations passed."""
        return all(r.passed for r in self.results)
    
    @property
    def failed_results(self) -> List[ExpectationResult]:
        """List of failed expectations."""
        return [r for r in self.results if not r.passed]
    
    def summary(self) -> str:
        """Generate a text summary of the report."""
        lines = []
        if self.passed:
            lines.append("✅ All expectations passed")
        else:
            lines.append(f"❌ {len(self.failed_results)} expectations failed")
            
        for result in self.results:
            lines.append(str(result))
            
        return "\n".join(lines)
    
    def raise_if_failed(self):
        """Raise ValueError if any expectation failed."""
        if not self.passed:
            raise ValueError(self.summary())


class ColumnExpectation:
    """Fluent interface for column-level expectations."""
    
    def __init__(self, suite: "ExpectationSuite", column: str):
        self.suite = suite
        self.column = column
        
    def to_be_between(self, min_value: float, max_value: float) -> "ExpectationSuite":
        """Expect values to be between min and max (inclusive)."""
        def check(df: pd.DataFrame) -> ExpectationResult:
            if self.column not in df.columns:
                return ExpectationResult(self.column, "to_be_between", False, "Column not found")
            
            series = pd.to_numeric(df[self.column], errors='coerce').dropna()
            failures = ((series < min_value) | (series > max_value)).sum()
            passed = failures == 0
            
            return ExpectationResult(
                self.column,
                f"to be between {min_value} and {max_value}",
                passed,
                f"Found {failures} values outside range" if not passed else "All values in range"
            )
            
        self.suite.add_check(check)
        return self.suite
    
    def to_match_regex(self, pattern: Union[str, Pattern]) -> "ExpectationSuite":
        """Expect string values to match a regex pattern."""
        def check(df: pd.DataFrame) -> ExpectationResult:
            if self.column not in df.columns:
                return ExpectationResult(self.column, "to_match_regex", False, "Column not found")
            
            series = df[self.column].astype(str)
            # Use pandas str.match for vectorization
            matches = series.str.match(pattern)
            failures = (~matches).sum()
            passed = failures == 0
            
            pat_str = pattern.pattern if hasattr(pattern, "pattern") else pattern
            return ExpectationResult(
                self.column,
                f"to match regex '{pat_str}'",
                passed,
                f"Found {failures} non-matching values" if not passed else "All values matched"
            )
            
        self.suite.add_check(check)
        return self.suite

    def to_be_in(self, values: List[Any]) -> "ExpectationSuite":
        """Expect values to be in a set of allowed values."""
        def check(df: pd.DataFrame) -> ExpectationResult:
            if self.column not in df.columns:
                return ExpectationResult(self.column, "to_be_in", False, "Column not found")
            
            series = df[self.column]
            failures = (~series.isin(values)).sum()
            passed = failures == 0
            
            return ExpectationResult(
                self.column,
                f"to be in {values}",
                passed,
                f"Found {failures} invalid values" if not passed else "All values valid"
            )
            
        self.suite.add_check(check)
        return self.suite
        
    def to_not_be_null(self) -> "ExpectationSuite":
        """Expect no null values."""
        def check(df: pd.DataFrame) -> ExpectationResult:
            if self.column not in df.columns:
                return ExpectationResult(self.column, "to_not_be_null", False, "Column not found")
            
            nulls = df[self.column].isna().sum()
            passed = nulls == 0
            
            return ExpectationResult(
                self.column,
                "to not be null",
                passed,
                f"Found {nulls} null values" if not passed else "No nulls found"
            )
            
        self.suite.add_check(check)
        return self.suite
        
    def to_be_unique(self) -> "ExpectationSuite":
        """Expect all values to be unique."""
        def check(df: pd.DataFrame) -> ExpectationResult:
            if self.column not in df.columns:
                return ExpectationResult(self.column, "to_be_unique", False, "Column not found")
            
            dupes = df[self.column].duplicated().sum()
            passed = dupes == 0
            
            return ExpectationResult(
                self.column,
                "to be unique",
                passed,
                f"Found {dupes} duplicate values" if not passed else "All values unique"
            )
            
        self.suite.add_check(check)
        return self.suite
        
    def to_have_type(self, dtype: str) -> "ExpectationSuite":
        """Expect column to have a specific pandas dtype."""
        def check(df: pd.DataFrame) -> ExpectationResult:
            if self.column not in df.columns:
                return ExpectationResult(self.column, "to_have_type", False, "Column not found")
            
            actual = str(df[self.column].dtype)
            passed = dtype in actual # weak check to allow int64 vs int32
            
            return ExpectationResult(
                self.column,
                f"to have type '{dtype}'",
                passed,
                f"Actual type is '{actual}'" if not passed else f"Type is '{actual}'"
            )
            
        self.suite.add_check(check)
        return self.suite
            

class ExpectationSuite:
    """Builder for a suite of data expectations."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.checks = []
        
    def column(self, name: str) -> ColumnExpectation:
        """Start an expectation for a specific column."""
        return ColumnExpectation(self, name)
        
    def add_check(self, check_func):
        """Add a check function to the suite."""
        self.checks.append(check_func)
        
    def run(self) -> ExpectationReport:
        """Execute all expectations and return a report."""
        results = [check(self.df) for check in self.checks]
        return ExpectationReport(results)


def expect(df: pd.DataFrame) -> ExpectationSuite:
    """
    Start defining expectations for a DataFrame.
    
    Args:
        df: The DataFrame to validate.
        
    Returns:
        An ExpectationSuite builder.
    """
    return ExpectationSuite(df)
