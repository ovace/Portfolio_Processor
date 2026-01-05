import pandas as pd
import pytest
from utils.processing import _calculate_total_cost

def test_calculate_total_cost_basic():
    """Test standard calculation when TC is missing."""
    df = pd.DataFrame({
        "Quantity": [10, 20],
        "Cost per Unit": [1.5, 2.0],
        "Total Cost": [None, None]
    })
    
    result = _calculate_total_cost(df)
    
    assert result.loc[0, "Total Cost"] == 15.0
    assert result.loc[1, "Total Cost"] == 40.0

def test_calculate_total_cost_partial_missing():
    """Test that existing TC values are not overwritten."""
    df = pd.DataFrame({
        "Quantity": [10, 20],
        "Cost per Unit": [1.5, 2.0],
        "Total Cost": [None, 99.0] # 99.0 should be preserved
    })
    
    result = _calculate_total_cost(df)
    
    assert result.loc[0, "Total Cost"] == 15.0
    assert result.loc[1, "Total Cost"] == 99.0

def test_calculate_total_cost_missing_factors():
    """Test that it doesn't crash if factors are missing."""
    df = pd.DataFrame({
        "Quantity": [10, None],
        "Cost per Unit": [None, 2.0],
        "Total Cost": [None, None]
    })
    
    result = _calculate_total_cost(df)
    
    # Values should remain NaN
    assert pd.isna(result.loc[0, "Total Cost"])
    assert pd.isna(result.loc[1, "Total Cost"])

def test_calculate_total_cost_non_numeric():
    """Test handling of non-numeric strings."""
    df = pd.DataFrame({
        "Quantity": ["10", "abc"],
        "Cost per Unit": ["1.5", "2.0"],
        "Total Cost": [None, None]
    })
    
    result = _calculate_total_cost(df)
    
    assert result.loc[0, "Total Cost"] == 15.0
    assert pd.isna(result.loc[1, "Total Cost"])

def test_calculate_total_cost_missing_columns():
    """Test that it returns DF as is if columns are missing."""
    df = pd.DataFrame({
        "Quantity": [10],
        "Value": [100]
    })
    
    result = _calculate_total_cost(df)
    assert "Total Cost" not in result.columns
    assert result.equals(df)
