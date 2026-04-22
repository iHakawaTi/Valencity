"""
Synthetic data generation module.
"""

import random
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from faker import Faker

from valencity.utils.logging import get_logger

logger = get_logger(__name__)


class SyntheticGenerator:
    """
    Generate synthetic data based on rules or existing data distributions.
    
    Example:
        >>> gen = SyntheticGenerator()
        >>> gen.add_numeric("age", min_val=18, max_val=90)
        >>> gen.add_categorical("status", ["active", "inactive"])
        >>> df = gen.generate(100)
    """
    
    def __init__(self, seed: Optional[int] = None):
        self.columns: Dict[str, Callable[[int], Any]] = {}
        self.faker = Faker()
        if seed is not None:
            self.faker.seed_instance(seed)
            np.random.seed(seed)
            random.seed(seed)
            
    def add_numeric(
        self, 
        name: str, 
        distribution: str = "uniform", 
        **kwargs
    ) -> "SyntheticGenerator":
        """
        Add a numeric column.
        
        Args:
            name: Column name
            distribution: 'uniform', 'normal', or 'integer'
            **kwargs: params like min_val, max_val, mean, std
            
        Raises:
            ValueError: If distribution is unknown or params are invalid.
        """
        if not name:
            raise ValueError("Column name cannot be empty")
            
        if distribution == "uniform":
            low = kwargs.get("min_val", 0)
            high = kwargs.get("max_val", 100)
            if low > high:
                raise ValueError(f"min_val ({low}) cannot be greater than max_val ({high})")
            self.columns[name] = lambda n: np.random.uniform(low, high, n)
            
        elif distribution == "normal":
            mean = kwargs.get("mean", 0)
            std = kwargs.get("std", 1)
            if std < 0:
                raise ValueError(f"Standard deviation ({std}) cannot be negative")
            self.columns[name] = lambda n: np.random.normal(mean, std, n)
            
        elif distribution == "integer":
            low = kwargs.get("min_val", 0)
            high = kwargs.get("max_val", 100)
            if low > high:
                raise ValueError(f"min_val ({low}) cannot be greater than max_val ({high})")
            self.columns[name] = lambda n: np.random.randint(low, high + 1, n)
        
        else:
            raise ValueError(f"Unknown distribution '{distribution}'. Options: uniform, normal, integer")
            
        return self
        
    def add_categorical(
        self, 
        name: str, 
        categories: List[Any], 
        weights: Optional[List[float]] = None
    ) -> "SyntheticGenerator":
        """Add a categorical column with optional weights."""
        if not name:
            raise ValueError("Column name cannot be empty")
        if not categories:
            raise ValueError("Categories list cannot be empty")
        
        if weights is not None:
            if len(weights) != len(categories):
                raise ValueError(f"Weights length ({len(weights)}) does not match categories length ({len(categories)})")
            if abs(sum(weights) - 1.0) > 1e-6:
                logger.warning(f"Weights sum to {sum(weights)}, normalizing to 1.0")
                total = sum(weights)
                weights = [w / total for w in weights]
                
        self.columns[name] = lambda n: np.random.choice(categories, n, p=weights)
        return self
        
    def add_pii(
        self, 
        name: str, 
        provider: str = "name"
    ) -> "SyntheticGenerator":
        """
        Add a PII column using Faker.
        
        Args:
            name: Column name
            provider: Faker provider (name, email, address, etc.)
        """
        if not name:
            raise ValueError("Column name cannot be empty")
            
        def generate_faker(n: int) -> List[Any]:
            # Inspect faker generic provider to find the method
            if not hasattr(self.faker, provider):
                 raise ValueError(f"Unknown Faker provider: '{provider}'")
                 
            fake_method = getattr(self.faker, provider)
            return [fake_method() for _ in range(n)]
            
        self.columns[name] = generate_faker
        return self
        
    def add_datetime(
        self, 
        name: str, 
        start_date: str = "-1y", 
        end_date: str = "now"
    ) -> "SyntheticGenerator":
        """Add a datetime column."""
        if not name:
            raise ValueError("Column name cannot be empty")
            
        def generate_dates(n: int) -> List[datetime]:
            # This is a simple implementation, generating n random dates between expected range
            # ideally we'd vectorize this, but for Faker it's per-item
            return [self.faker.date_time_between(start_date=start_date, end_date=end_date) for _ in range(n)]
            
        self.columns[name] = generate_dates
        return self

    def from_dataframe(self, df: pd.DataFrame) -> "SyntheticGenerator":
        """
        Configure generator to mimic an existing DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        if df.empty:
            logger.warning("Empty DataFrame provided, no columns added")
            return self
            
        for col in df.columns:
            dtype = df[col].dtype
            
            if pd.api.types.is_numeric_dtype(dtype):
                # Simple logic to choose between int and float
                if pd.api.types.is_integer_dtype(dtype):
                    self.add_numeric(
                        col, 
                        "integer", 
                        min_val=int(df[col].min()), 
                        max_val=int(df[col].max())
                    )
                else:
                    self.add_numeric(
                        col, 
                        "normal", 
                        mean=float(df[col].mean()), 
                        std=float(df[col].std())
                    )
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                 self.add_datetime(col)
            elif df[col].nunique() < 20 or df[col].nunique() / len(df) < 0.1:
                # Treated as categorical
                vc = df[col].value_counts(normalize=True)
                self.add_categorical(col, vc.index.tolist(), vc.values.tolist())
            else:
                # High cardinality string - likely PII or Text
                # We'll default to 'word' or 'pystr' as a fallback
                # Ideally we could inspect PII patterns here to choose better providers!
                if "email" in col.lower():
                    self.add_pii(col, "email")
                elif "name" in col.lower():
                    self.add_pii(col, "name")
                else:
                    self.add_categorical(col, ["unknown"] * 10) # Fallback
                    
        return self

    def generate(self, num_rows: int) -> pd.DataFrame:
        """Generate the synthetic dataframe."""
        if num_rows < 0:
            raise ValueError("Number of rows cannot be negative")
            
        data = {}
        logger.info(f"Generating {num_rows} rows of synthetic data...")
        
        for name, func in self.columns.items():
            data[name] = func(num_rows)
            
        return pd.DataFrame(data)
