"""Schema validation for DataFrames."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd

from valencity.utils.logging import get_logger

logger = get_logger(__name__)


class DataType(Enum):
    """Supported data types for schema validation."""
    
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    CATEGORICAL = "categorical"
    ANY = "any"
    
    @classmethod
    def from_dtype(cls, dtype: np.dtype) -> "DataType":
        """Infer DataType from numpy/pandas dtype."""
        dtype_str = str(dtype)
        
        if dtype_str == "object":
            return cls.STRING
        elif "int" in dtype_str:
            return cls.INTEGER
        elif "float" in dtype_str:
            return cls.FLOAT
        elif dtype_str == "bool":
            return cls.BOOLEAN
        elif "datetime" in dtype_str:
            return cls.DATETIME
        elif "category" in dtype_str:
            return cls.CATEGORICAL
        else:
            return cls.ANY


@dataclass
class ColumnSpec:
    """Specification for a single column."""
    
    name: str
    dtype: DataType
    nullable: bool = True
    unique: bool = False
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    allowed_values: Optional[Set[Any]] = None
    regex_pattern: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    
    def validate(self, series: pd.Series) -> List[str]:
        """
        Validate a Series against this column spec.
        
        Returns:
            List of validation error messages (empty if valid).
        """
        errors: List[str] = []
        
        # Check nullability
        if not self.nullable and series.isna().any():
            null_count = series.isna().sum()
            errors.append(f"Column '{self.name}' has {null_count} null values but is not nullable")
        
        # Check uniqueness
        if self.unique and series.dropna().duplicated().any():
            dup_count = series.dropna().duplicated().sum()
            errors.append(f"Column '{self.name}' has {dup_count} duplicate values but must be unique")
        
        # Check data type
        inferred_type = DataType.from_dtype(series.dtype)
        if self.dtype != DataType.ANY and inferred_type != self.dtype:
            # Allow numeric type coercion
            numeric_types = {DataType.INTEGER, DataType.FLOAT}
            if not (self.dtype in numeric_types and inferred_type in numeric_types):
                errors.append(
                    f"Column '{self.name}' has type {inferred_type.value}, "
                    f"expected {self.dtype.value}"
                )
        
        # Check min/max values for numeric columns
        if self.min_value is not None:
            below_min = series.dropna() < self.min_value
            if below_min.any():
                errors.append(
                    f"Column '{self.name}' has {below_min.sum()} values below minimum {self.min_value}"
                )
        
        if self.max_value is not None:
            above_max = series.dropna() > self.max_value
            if above_max.any():
                errors.append(
                    f"Column '{self.name}' has {above_max.sum()} values above maximum {self.max_value}"
                )
        
        # Check allowed values (categorical)
        if self.allowed_values is not None:
            invalid = ~series.dropna().isin(self.allowed_values)
            if invalid.any():
                sample = series.dropna()[invalid].unique()[:5]
                errors.append(
                    f"Column '{self.name}' has invalid values: {list(sample)}"
                )
        
        # Check string length
        if series.dtype == object:
            lengths = series.dropna().str.len()
            
            if self.min_length is not None:
                too_short = lengths < self.min_length
                if too_short.any():
                    errors.append(
                        f"Column '{self.name}' has {too_short.sum()} values shorter than {self.min_length}"
                    )
            
            if self.max_length is not None:
                too_long = lengths > self.max_length
                if too_long.any():
                    errors.append(
                        f"Column '{self.name}' has {too_long.sum()} values longer than {self.max_length}"
                    )
        
        return errors


@dataclass
class ValidationResult:
    """Result of schema validation."""
    
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    extra_columns: List[str] = field(default_factory=list)
    missing_columns: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Generate human-readable validation summary."""
        if self.is_valid and not self.errors:
            return "✅ Schema validation passed"
        
        lines = ["❌ Schema validation failed:"]
        
        if self.missing_columns:
            lines.append(f"  Missing columns: {self.missing_columns}")
        
        if self.extra_columns:
            lines.append(f"  Extra columns: {self.extra_columns}")
        
        for error in self.errors:
            lines.append(f"  - {error}")
        
        return "\n".join(lines)
    
    def raise_if_invalid(self) -> None:
        """Raise ValueError if validation failed."""
        if not self.is_valid:
            raise ValueError(self.summary())


class DataSchema:
    """
    Define and validate DataFrame schemas.
    
    A schema defines expected columns, their types, and constraints.
    Use it to validate incoming data or to document your data contracts.
    
    Example:
        >>> schema = DataSchema([
        ...     ColumnSpec("id", DataType.INTEGER, nullable=False, unique=True),
        ...     ColumnSpec("email", DataType.STRING, nullable=False),
        ...     ColumnSpec("age", DataType.INTEGER, min_value=0, max_value=150),
        ... ])
        >>> result = schema.validate(df)
        >>> if not result.is_valid:
        ...     print(result.summary())
        
        >>> # Or infer schema from existing data
        >>> schema = DataSchema.from_dataframe(df)
    """
    
    def __init__(
        self,
        columns: List[ColumnSpec],
        strict: bool = False,
        allow_extra_columns: bool = True
    ):
        """
        Initialize schema.
        
        Args:
            columns: List of column specifications.
            strict: If True, treat warnings as errors.
            allow_extra_columns: If False, extra columns are errors.
        """
        self.columns = {col.name: col for col in columns}
        self.strict = strict
        self.allow_extra_columns = allow_extra_columns
    
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate a DataFrame against this schema.
        
        Args:
            df: The DataFrame to validate.
            
        Returns:
            ValidationResult with validation details.
        """
        errors: List[str] = []
        warnings: List[str] = []
        
        # Check for missing columns
        expected_cols = set(self.columns.keys())
        actual_cols = set(df.columns)
        
        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols
        
        missing_columns = list(missing)
        extra_columns = list(extra)
        
        if missing:
            errors.append(f"Missing required columns: {missing_columns}")
        
        if extra and not self.allow_extra_columns:
            errors.append(f"Unexpected columns: {extra_columns}")
        elif extra:
            warnings.append(f"Extra columns found: {extra_columns}")
        
        # Validate each column
        for col_name, col_spec in self.columns.items():
            if col_name not in df.columns:
                continue  # Already reported as missing
            
            col_errors = col_spec.validate(df[col_name])
            errors.extend(col_errors)
        
        is_valid = len(errors) == 0 and (not self.strict or len(warnings) == 0)
        
        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            extra_columns=extra_columns,
            missing_columns=missing_columns
        )
        
        if is_valid:
            logger.info("Schema validation passed")
        else:
            logger.warning(result.summary())
        
        return result
    
    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        infer_nullable: bool = True,
        infer_unique: bool = False
    ) -> "DataSchema":
        """
        Infer a schema from an existing DataFrame.
        
        Args:
            df: DataFrame to infer schema from.
            infer_nullable: If True, mark columns with nulls as nullable.
            infer_unique: If True, check for unique columns.
            
        Returns:
            DataSchema inferred from the DataFrame.
        """
        columns: List[ColumnSpec] = []
        
        for col_name in df.columns:
            series = df[col_name]
            dtype = DataType.from_dtype(series.dtype)
            
            nullable = series.isna().any() if infer_nullable else True
            unique = not series.dropna().duplicated().any() if infer_unique else False
            
            spec = ColumnSpec(
                name=col_name,
                dtype=dtype,
                nullable=nullable,
                unique=unique
            )
            
            # Infer numeric bounds
            if dtype in (DataType.INTEGER, DataType.FLOAT):
                spec.min_value = series.min()
                spec.max_value = series.max()
            
            columns.append(spec)
        
        logger.info(f"Inferred schema with {len(columns)} columns")
        return cls(columns)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export schema as a dictionary for serialization."""
        return {
            "columns": [
                {
                    "name": col.name,
                    "dtype": col.dtype.value,
                    "nullable": col.nullable,
                    "unique": col.unique,
                    "min_value": col.min_value,
                    "max_value": col.max_value,
                }
                for col in self.columns.values()
            ],
            "strict": self.strict,
            "allow_extra_columns": self.allow_extra_columns,
        }
