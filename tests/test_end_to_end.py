
import pytest
import os
import tempfile
import pandas as pd
from utils.portfolio_utils import process_file, process_hybrid_file

def test_e2e_process_standard_csv(data_gen, mock_config):
    df = data_gen.create_standard_df(rows=5)
    with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", newline='', delete=False) as tmp:
        df.to_csv(tmp, index=False)
        inp_path = tmp.name
    # Close
        
    out_dir = tempfile.mkdtemp()
    
    try:
        # Run standard process
        out_path = process_file(inp_path, mock_config, output_dir=out_dir)
        
        assert os.path.isfile(out_path)
        out_df = pd.read_csv(out_path)
        
        # Check standard columns exist
        assert "Symbol" in out_df.columns
        assert "Quantity" in out_df.columns
        assert len(out_df) == 5
        
    finally:
        os.remove(inp_path)
        # cleanup out_dir ignored (tmp)

def test_e2e_process_hybrid_xlsx(data_gen, mock_config):
    acc, hold = data_gen.create_hybrid_dfs()
    
    # Close it so pandas can open
    tmp_path = ""
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Create Excel with 2 sheets (mimic hybrid file)
        # Verify file is closed before writer
        pass
        
        # Re-open safely
        with pd.ExcelWriter(tmp_path, engine='openpyxl') as writer:
            acc.to_excel(writer, sheet_name="Summary", index=False)
            hold.to_excel(writer, sheet_name="Holdings", index=False)
            
        out_dir = tempfile.mkdtemp()
        inp_path = tmp_path
        
        # We explicitly call process_hybrid_file OR rely on detection?
        # The caller script uses structure detection.
        # Here we test `process_hybrid_file` directly to ensure it works if called.
        
        out_path = process_hybrid_file(inp_path, mock_config, output_dir=out_dir)
        
        assert os.path.isfile(out_path)
        out_df = pd.read_csv(out_path)
        
        # Check Account was propagated from summary to holdings
        assert "Account" in out_df.columns
        assert (out_df["Account"] == "TestAccount123").all()
        assert len(out_df) == 2
        
    finally:
        if os.path.exists(inp_path):
            os.remove(inp_path)

def test_e2e_backfill_feature(data_gen, mock_config):
    # Test that the backfill feature actually works in E2E flow
    df = pd.DataFrame({
        "Ticker": ["AAPL", None], # Ticker -> Symbol
        "Description": ["Apple", "Microsoft"], # Description -> Security Description
        "shares": [10, 20] # shares -> Quantity
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", newline='', delete=False) as tmp:
        df.to_csv(tmp, index=False)
        inp_path = tmp.name
    # Close
        
    out_dir = tempfile.mkdtemp()
    
    try:
        out_path = process_file(inp_path, mock_config, output_dir=out_dir)
        out_df = pd.read_csv(out_path)
        
        assert len(out_df) == 2
        # Start: AAPL, None
        # End: AAPL, Microsoft (backfilled)
        
        assert "Symbol" in out_df.columns
        symbols = out_df["Symbol"].fillna("").tolist()
        assert "Microsoft" in symbols
        
    finally:
        os.remove(inp_path)
