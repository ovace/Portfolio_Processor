
import pytest
import os
import tempfile
import pandas as pd
import json
from utils.portfolio_utils import detect_file_structure

def test_detect_structure_standard(data_gen, mock_config):
    df = data_gen.create_standard_df()
    with tempfile.NamedTemporaryFile(suffix=".csv", mode='w', newline='', delete=False) as tmp:
        df.to_csv(tmp, index=False)
        tmp_name = tmp.name
        
    try:
        struct = detect_file_structure(tmp_name, mock_config)
        assert struct == "standard"
    finally:
        os.remove(tmp_name)

def test_detect_structure_hybrid(data_gen, mock_config):
    acc, hold = data_gen.create_hybrid_dfs()
    
    # Save as multi-table CSV (separated by blank lines)
    with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", newline='', delete=False) as tmp:
        acc.to_csv(tmp, index=False)
        tmp.write("\n\n") # Blank lines
        hold.to_csv(tmp, index=False)
        tmp_name = tmp.name
        
    try:
        # Hybrid requires:
        # 1. Account summary table (Account col + Summary keywords)
        # 2. Holdings table (Symbol + Qty + Date)
        
        # NOTE: create_hybrid_dfs headers must match what logic looks for
        # logic: 
        # _is_account_summary: "account" in header AND ("net account value", "total gain", ...) in header
        # _is_holdings_table: ("date acquired"|"purchase date") AND ("qty"|"quantity") AND "symbol"
        
        # check hybrid generator headers:
        # Acc: "Account", "Net Account Value" -> matches
        # Hold: "Symbol", "Qty", "Purchase Date" -> matches
        
        struct = detect_file_structure(tmp_name, mock_config)
        assert struct == "hybrid"
    finally:
        os.remove(tmp_name)
