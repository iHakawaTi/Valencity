"""Tests for valencity.synthetic module."""

import numpy as np
import pandas as pd
import pytest

from valencity.synthetic import SyntheticGenerator


class TestSyntheticGenerator:
    """Test synthetic data generation."""
    
    def test_numeric_generation(self):
        gen = SyntheticGenerator(seed=42)
        gen.add_numeric("age", "integer", min_val=0, max_val=10)
        df = gen.generate(100)
        
        assert len(df) == 100
        assert df["age"].min() >= 0
        assert df["age"].max() <= 10
        assert df["age"].dtype == int or df["age"].dtype == "int32"
        
    def test_categorical_generation(self):
        gen = SyntheticGenerator(seed=42)
        gen.add_categorical("color", ["red", "blue"], weights=[0.8, 0.2])
        df = gen.generate(1000)
        
        counts = df["color"].value_counts(normalize=True)
        assert abs(counts["red"] - 0.8) < 0.05
        
    def test_pii_generation(self):
        gen = SyntheticGenerator(seed=42)
        gen.add_pii("email", "email")
        df = gen.generate(10)
        
        assert "@" in df["email"].iloc[0]
        assert len(df) == 10
        
    def test_from_dataframe(self):
        # Create source DF
        source = pd.DataFrame({
            "age": np.random.randint(20, 60, 100),
            "score": np.random.normal(0, 1, 100),
            "group": ["A", "B"] * 50
        })
        
        gen = SyntheticGenerator(seed=42)
        gen.from_dataframe(source)
        syn_df = gen.generate(50)
        
        assert len(syn_df) == 50
        assert "age" in syn_df.columns
        assert "score" in syn_df.columns
        assert "group" in syn_df.columns
        
        # Check properties
        assert syn_df["age"].min() >= 20
        assert pd.api.types.is_numeric_dtype(syn_df["score"])
        assert set(syn_df["group"].unique()).issubset({"A", "B"})

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
