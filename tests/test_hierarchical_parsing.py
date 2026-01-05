
import pandas as pd
import pytest
from utils.detection import _parse_symbol_group_table, _is_symbol_group_table
from utils.processing import normalise_table

def test_hierarchical_symbol_group_parsing():
    """
    Test logic for hierarchical CW.xlsx style tables:
    - Category headers (Type)
    - Security headers (Aggregate rows - should be skipped)
    - Account-level lot rows (Shared columns - should be resolved)
    """
    data = {
        'Description': [
            'Cash',                                         # 0: Category
            'Advisory Retirement Sweep Program',            # 1: Security Summary (Header)
            'Ovace Mamnoon IRA NFS - PPS Custom (B37535586)', # 2: Lot Row (Account)
            'Mutual Fund',                                  # 3: Category
            'Allspring Special Mid-Cap Value Fund Cl I',    # 4: Security Summary (Header)
            'Nafees Mamnoon IRA NFS - PPS Custom (B37535582)'# 5: Lot Row (Account)
        ],
        'Symbol': [
            'Cash',         # Category indicator
            'QLFPQ',        # Security ticker
            None,           # Empty for lot
            'Mutual Fund',  # Category indicator
            'WFMIX',        # Security ticker
            None            # Empty for lot
        ],
        'Quantity': [
            None,           # Category row
            248.14,         # Aggregate total (should be excluded)
            0.01,           # Lot 1
            None,           # Category row
            639.682,        # Aggregate total (should be excluded)
            34.974          # Lot 2
        ]
    }
    df = pd.DataFrame(data)
    
    # Use real mapping to test ambiguity robustness
    from utils.portfolio_utils import load_column_mapping, flatten_column_mapping
    grouped_map = load_column_mapping("config/column_mapping.json")
    flat_map = flatten_column_mapping(grouped_map)
    
    # 1. Detection
    assert _is_symbol_group_table(df, flat_map)
    
    # 2. Parsing
    parsed = _parse_symbol_group_table(df, flat_map)
    
    # EXPECTATIONS:
    # Aggregates (Row 1, 4) and Categories (Row 0, 3) are filtered.
    # We should have 2 lot rows.
    assert len(parsed) == 2
    
    # Check extra columns added by parser
    assert 'Account' in parsed.columns
    assert 'Security Type' in parsed.columns
    
    # 3. Normalization (Priority Mapping)
    norm = normalise_table(parsed, grouped_map)
    
    # Row 1 check
    row1 = norm.iloc[0]
    assert row1['Symbol'] == 'QLFPQ'
    assert row1['Security Description'] == 'Advisory Retirement Sweep Program'
    assert row1['Account'] == 'Ovace Mamnoon IRA NFS - PPS Custom (B37535586)'
    assert row1['Type'] == 'Cash'
    assert row1['Quantity'] == 0.01
    
    # Row 2 check
    row2 = norm.iloc[1]
    assert row2['Symbol'] == 'WFMIX'
    assert row2['Security Description'] == 'Allspring Special Mid-Cap Value Fund Cl I'
    assert row2['Account'] == 'Nafees Mamnoon IRA NFS - PPS Custom (B37535582)'
    assert row2['Type'] == 'Mutual Fund'
    assert row2['Quantity'] == 34.974
