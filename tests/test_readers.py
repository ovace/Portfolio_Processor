
import pytest
import os
import tempfile
import pandas as pd
from utils.portfolio_utils import read_csv_file, read_excel_file

def test_read_csv_simple(data_gen):
    df = data_gen.create_standard_df(rows=3)
    with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", newline='', delete=False) as tmp:
        df.to_csv(tmp, index=False)
        tmp_name = tmp.name
        
    try:
        tables, lines = read_csv_file(tmp_name)
        # Standard CSV with no blank lines is 1 table
        assert len(tables) == 1
        assert len(tables[0]) == 3
        # lines usually contains raw rows
        assert len(lines) >= 4 # header + 3 rows
    finally:
        os.remove(tmp_name)

def test_read_csv_multi_table(data_gen):
    """Simulate CSV with blank lines separating tables"""
    with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", newline='', delete=False) as tmp:
        tmp.write("T1C1,T1C2\nA,1\nB,2\n\n") # Table 1
        tmp.write("T2C1,T2C2\nX,9\nY,8\n")   # Table 2
        tmp_name = tmp.name
        
    try:
        tables, _ = read_csv_file(tmp_name)
        assert len(tables) == 2
        assert list(tables[0].columns) == ["T1C1", "T1C2"]
        assert list(tables[1].columns) == ["T2C1", "T2C2"]
    finally:
        os.remove(tmp_name)

def test_read_excel_simple(data_gen):
    df = data_gen.create_standard_df(rows=3)
    with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
        tmp_name = tmp.name
    # Explicitly close before writing/reading on Windows
    
    try:
        df.to_excel(tmp_name, index=False)
        tables, lines = read_excel_file(tmp_name)
        assert len(tables) >= 1
        # The first table might be the sheet data
        assert any(len(t) == 3 for t in tables)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

def test_read_excel_multi_sheet(data_gen):
    df1 = data_gen.create_standard_df(rows=2)
    df2 = data_gen.create_standard_df(rows=2)
    
    # Use delete=False and close immediately
    tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    tmp_name = tmp.name
    tmp.close()
        
    try:
        with pd.ExcelWriter(tmp_name, engine='openpyxl') as writer:
            df1.to_excel(writer, sheet_name="Sheet1", index=False)
            df2.to_excel(writer, sheet_name="Sheet2", index=False)
            
        tables, _ = read_excel_file(tmp_name)
        # Should parse both sheets -> 2 tables
        assert len(tables) >= 2
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
