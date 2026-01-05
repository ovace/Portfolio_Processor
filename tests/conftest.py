
import pytest
import os
import json
import pandas as pd
import numpy as np
import tempfile
import sys

# Ensure root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class SyntheticDataGenerator:
    """Helper to generate synthetic portfolio files for testing."""
    
    @staticmethod
    def create_standard_df(rows=5, missing_symbols=False):
        data = {
            "Symbol": [f"SYM{i}" for i in range(rows)],
            "Quantity": [10 * (i+1) for i in range(rows)],
            "Cost": [100.0 * (i+1) for i in range(rows)],
            "Date Acquired": [f"2023-01-{i+1:02d}" for i in range(rows)],
            "Security Description": [f"Description for SYM{i}" for i in range(rows)]
        }
        df = pd.DataFrame(data)
        if missing_symbols:
            # Introduce anomalies
            df.loc[1, "Symbol"] = None
            df.loc[2, "Symbol"] = "nan"
            df.loc[3, "Symbol"] = ""
        return df

    @staticmethod
    def create_hybrid_dfs():
        """Returns (account_df, holdings_df)"""
        acc_df = pd.DataFrame({
            "Account": ["TestAccount123"],
            "Net Account Value": [100000.0],
            "Cash": [5000.0]
        })
        
        # Hybrid holdings must have:
        # 1. Headers satisfying _is_holdings_table (Symbol, Date, Qty)
        # 2. Col 0 satisfying _parse_hybrid_holdings (Mix of Symbol strings and Date strings)
        
        holdings_df = pd.DataFrame([
            ["AAPL", "Apple", "100"],
            ["01/01/2023", "Buy", "10"],
            ["MSFT", "Microsoft", "200"],
            ["02/01/2023", "Buy", "20"]
        ], columns=["Symbol", "Date Acquired", "Qty"])
        
        return acc_df, holdings_df

    @staticmethod
    def save_csv(df, path):
        df.to_csv(path, index=False)

    @staticmethod
    def save_excel_multi_sheet(sheets_dict, path):
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            for name, df in sheets_dict.items():
                df.to_excel(writer, sheet_name=name, index=False)

@pytest.fixture
def data_gen():
    return SyntheticDataGenerator

@pytest.fixture
def mock_config():
    """Returns path to a temporary config file with standard mapping."""
    mapping = {
        "Symbol": ["symbol", "ticker", "sym"],
        "Quantity": ["quantity", "qty", "shares"],
        "Acquisition Date": ["date acquired", "purchase date"],
        "Cost per Unit": ["cost", "price"],
        "Total Cost": ["total cost", "amount"],
        "Account": ["account", "acct"],
        "Security Description": ["security description", "description"]
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as tmp:
        json.dump(mapping, tmp)
        tmp_path = tmp.name
    yield tmp_path
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
