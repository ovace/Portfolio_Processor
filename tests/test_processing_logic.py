
import pandas as pd
import pytest
from utils.portfolio_utils import (
    normalise_table,
    _fill_symbol_from_description,
    cleanup_portfolio_df,
    extract_portfolio_fields,
    forward_fill_sparse_columns
)

def test_normalise_table():
    df = pd.DataFrame({
        "Ticker": ["AAPL"],
        "price": [100],
        "Unknown": [1]
    })
    
    col_map = {
        "Symbol": ["ticker"],
        "Cost per Unit": ["price"]
    }
    
    res = normalise_table(df, col_map)
    assert "Symbol" in res.columns
    assert "Cost per Unit" in res.columns
    assert "Unknown" in res.columns # Should keep unknown

def test_normalise_table_priority():
    """Verify that priority order in the mapping is respected."""
    df = pd.DataFrame({
        "Date": ["2023-01-01"],
        "Date Acquired": ["2023-01-02"]
    })
    
    # Priority: "date acquired" > "date"
    col_map = {
        "Acquisition Date": ["date acquired", "date"]
    }
    
    res = normalise_table(df, col_map)
    assert "Acquisition Date" in res.columns
    # The value should be from "Date Acquired" (2023-01-02)
    assert res.iloc[0]["Acquisition Date"] == "2023-01-02"
    assert "Date" in res.columns # Original "Date" should remain untouched if not picked as primary

def test_fill_symbol_from_description_logic():
    df = pd.DataFrame({
        "Symbol": ["AAPL", None, "nan", "", "GOOG"],
        "Security Description": ["Apple", "Microsoft", "Amazon", "Tesla", "Google"],
        "Other": [1, 2, 3, 4, 5]
    })
    
    res = _fill_symbol_from_description(df)
    
    # AAPL kept
    assert res.iloc[0]["Symbol"] == "AAPL"
    # None -> Microsoft
    assert res.iloc[1]["Symbol"] == "Microsoft"
    # "nan" -> Amazon
    assert res.iloc[2]["Symbol"] == "Amazon"
    # "" -> Tesla
    assert res.iloc[3]["Symbol"] == "Tesla"
    # GOOG kept
    assert res.iloc[4]["Symbol"] == "GOOG"

def test_fill_symbol_no_description_col():
    df = pd.DataFrame({"Symbol": ["A", None]})
    # Should not crash, just return original
    res = _fill_symbol_from_description(df)
    assert res.iloc[1]["Symbol"] is None

def test_cleanup_portfolio_df_logic():
    df = pd.DataFrame({
        "A": [1, None, 1, 4],
        "B": [2, None, 2, 5]
    })
    # Row 1 is all None (should be dropped)
    # Row 2 is duplicate of Row 0 (should be dropped)
    
    res, blank_removed, dupes_removed = cleanup_portfolio_df(df)
    
    assert len(res) == 2
    assert res.iloc[0]["A"] == 1
    assert res.iloc[1]["A"] == 4
    assert blank_removed == 1
    assert dupes_removed == 1

def test_extract_portfolio_fields_logic():
    df = pd.DataFrame({
        "Symbol": ["AAPL"],
        "Quantity": [10],
        "Extra": [99]
    })
    
    fields = ["Symbol", "Quantity", "Account"]
    kv = {"Account": "ACC-1"}
    
    res = extract_portfolio_fields(df, fields, kv)
    
    assert list(res.columns) == fields
    assert res.iloc[0]["Symbol"] == "AAPL"
    assert res.iloc[0]["Account"] == "ACC-1" # From KV
    assert "Extra" not in res.columns  # Should drop extra

def test_forward_fill_sparse_columns():
    """Test that Security Description and Symbol are forward-filled when sparse."""
    df = pd.DataFrame({
        "Security Description": ["APPLE INC", None, None, None],
        "Symbol": ["AAPL", None, None, None],
        "Date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
        "Quantity": [10, 5, 3, 2]
    })
    
    # col_map not used in forward_fill but kept for signature consistency
    col_map = {}
    
    res = forward_fill_sparse_columns(df, col_map)
    
    # All rows should now have Security Description and Symbol filled
    assert res.iloc[0]["Security Description"] == "APPLE INC"
    assert res.iloc[1]["Security Description"] == "APPLE INC"
    assert res.iloc[2]["Security Description"] == "APPLE INC"
    assert res.iloc[3]["Security Description"] == "APPLE INC"
    
    assert res.iloc[0]["Symbol"] == "AAPL"
    assert res.iloc[1]["Symbol"] == "AAPL"
    assert res.iloc[2]["Symbol"] == "AAPL"
    assert res.iloc[3]["Symbol"] == "AAPL"

def test_forward_fill_sparse_columns_no_fill_when_dense():
    """Test that forward-fill doesn't apply when column is already dense."""
    df = pd.DataFrame({
        "Security Description": ["APPLE INC", "MICROSOFT CORP", "GOOGLE LLC", "AMAZON COM"],
        "Quantity": [10, 5, 3, 2]
    })
    
    col_map = {}
    res = forward_fill_sparse_columns(df, col_map)
    
    # Should remain unchanged (all values already present)
    assert res.iloc[0]["Security Description"] == "APPLE INC"
    assert res.iloc[1]["Security Description"] == "MICROSOFT CORP"
    assert res.iloc[2]["Security Description"] == "GOOGLE LLC"
    assert res.iloc[3]["Security Description"] == "AMAZON COM"

from utils.processing import extract_metadata_from_df, refine_metadata

def test_extract_metadata_column_based():
    df = pd.DataFrame({
        "Account": ["Joint-123", "Joint-123"],
        "Symbol": ["AAPL", "MSFT"]
    })
    col_map = {"account": "Account"}
    meta = extract_metadata_from_df(df, col_map)
    assert meta["Account"] == "Joint-123"

def test_extract_metadata_row_based():
    # Case where Account: value is in the row content
    data = [
        ["Account Number", "999888", None],
        ["Symbol", "Qty", "Price"],
        ["AAPL", 10, 150]
    ]
    df = pd.DataFrame(data)
    col_map = {"account number": "Account#"}
    
    meta = extract_metadata_from_df(df, col_map)
    assert meta["Account#"] == "999888"

def test_refine_metadata_splitting():
    meta = {"Account": "My Trust - 5678"}
    refined = refine_metadata(meta)
    assert refined["Account"] == "My Trust"
    assert refined["Account#"] == "5678"

def test_refine_metadata_splitting_hyphen():
    meta = {"Account": "Joint Account-9186"}
    refined = refine_metadata(meta)
    assert refined["Account"] == "Joint Account"
    assert refined["Account#"] == "9186"
