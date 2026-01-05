import sys
import os
import pandas as pd
import pytest
import json
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.portfolio_utils import detect_file_structure, process_file, load_column_mapping

def test_load_column_mapping():
    # creates a temp mapping file
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        json.dump({"Account": ["acct", "id"]}, tmp)
        tmp_path = tmp.name
    
    try:
        mapping = load_column_mapping(tmp_path)
        assert "Account" in mapping
        assert mapping["Account"] == ["acct", "id"]
    finally:
        os.remove(tmp_path)

def test_detect_file_structure_csv():
    # mock a simple CSV
    with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as tmp:
        # standard format
        tmp.write("Symbol,Quantity,Cost\nAAPL,10,150.0\n")
        tmp_path = tmp.name
    
    try:
        struct = detect_file_structure(tmp_path, "dummy_config")
        assert struct == "standard"
    finally:
        os.remove(tmp_path)

def test_process_file_simple():
    # mock config
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as cfg:
        json.dump({"Symbol": ["sym"], "Quantity": ["qty"]}, cfg)
        cfg_path = cfg.name
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=".csv", delete=False) as csv_file:
        csv_file.write("sym,qty,Value\nGOOG,5,1000\n")
        csv_path = csv_file.name
        
    try:
        # We need to set env var for output fields if we want specific extraction, 
        # or rely on defaults. Default expects "Account" etc.
        # Let's see if it runs without crashing and produces a file.
        output_dir = tempfile.mkdtemp()
        
        # It's hard to modify env vars safely in parallel tests, but we'll try minimal
        # The function uses `get_output_fields` which reads env.
        # We'll just run it and check it returns a path.
        
        out = process_file(csv_path, cfg_path, output_dir=output_dir)
        assert os.path.isfile(out)
        
        df = pd.read_csv(out)
        # Should have canonical columns 
        # (Symbol is in default required fields)
        assert "Symbol" in df.columns
        assert df.iloc[0]["Symbol"] == "GOOG"
        
    finally:
        os.remove(cfg_path)
        os.remove(csv_path)
        # cleanup output dir? (skip for now, messy)
