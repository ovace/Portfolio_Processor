import pandas as pd
import pytest
from caller import _normalize_input_entries, _add_metrics_columns

def test_normalize_input_entries_cli():
    # Test when input files are provided via CLI
    args_files = ["file1.csv", "file2.csv"]
    settings = {}
    entries = _normalize_input_entries(args_files, settings)
    assert len(entries) == 2
    assert entries[0]["path"] == "file1.csv"
    assert entries[0]["tabs"] is None

def test_normalize_input_entries_settings():
    # Test when input files come from settings JSON
    args_files = None
    settings = {
        "INPUT_FILES": [
            {"path": "c:/data/f3.xlsx", "tabs": ["Sheet1"]},
            "f4.csv"
        ]
    }
    entries = _normalize_input_entries(args_files, settings)
    assert len(entries) == 2
    assert entries[0]["path"] == "c:/data/f3.xlsx"
    assert entries[0]["tabs"] == ["Sheet1"]
    assert entries[1]["path"] == "f4.csv"
    assert entries[1]["tabs"] is None

def test_caller_add_metrics_columns_cagr_row():
    # Mock some data to test vectorized CAGR calculation inside caller.py
    df = pd.DataFrame({
        "Symbol": ["AAPL", "MSFT"],
        "Quantity": [10, 20],
        "Cost per Unit": [100, 200], # Initial: 1000, 4000
        "Value": [1500, 3000],        # Final: 1500, 3000
        "Acquisition Date": [
            (pd.Timestamp.today() - pd.Timedelta(days=365)).strftime("%Y-%m-%d"), # 1 year ago
            (pd.Timestamp.today() - pd.Timedelta(days=730)).strftime("%Y-%m-%d")  # 2 years ago
        ]
    })
    
    settings = {"METRICS": {"ENABLE": True, "COMPUTE_CAGR_PER_ROW": True}}
    result = _add_metrics_columns(df, settings)
    
    assert "CAGR" in result.columns
    assert "Years Held" in result.columns
    
    # AAPL: (1500/1000)^(1/1) - 1 = 0.5 (50%)
    assert abs(result.loc[0, "CAGR"] - 0.5) < 0.05
    # MSFT: (3000/4000)^(1/2) - 1 = sqrt(0.75) - 1 = ~0.866 - 1 = -0.134
    assert abs(result.loc[1, "CAGR"] - (-0.1339745)) < 0.05
