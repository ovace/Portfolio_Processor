"""
Test for ET_J.csv format processing where:
- Stock lots are grouped with symbol in first row (aggregate - should be skipped)
- Subsequent rows have acquisition dates in Symbol column
"""
import pandas as pd
import pytest
from utils.detection import _parse_symbol_group_table, _is_symbol_group_table
from utils.processing import handle_acquisition_date_from_symbol, normalise_table


from utils.portfolio_utils import process_file

def test_et_j_metadata_extraction(tmp_path):
    """Test extraction of metadata from summary tables and Account# splitting"""
    
    # 1. Create a mock CSV mimicking the ET_J.csv start
    # Table 1: Account Summary (Metadata only)
    # Table 2: Holdings (Data)
    csv_content = """Account Summary,,,
Account,Net Account Value,Total Gain $,
Joint JTWROS -9186,407888.78,105966.01,
,,,
Symbol,Quantity,Cost per Unit,Total Cost
AAPL,100,150.00,15000
"""
    input_file = tmp_path / "ET_J_test.csv"
    input_file.write_text(csv_content)
    
    config_path = "config/column_mapping.json" # Assumes this exists in the env
    
    # Process the file
    out_dir = tmp_path / "out"
    out_path = process_file(str(input_file), config_path, output_dir=str(out_dir), timestamp="test_meta")
    
    # Check the result
    df = pd.read_csv(out_path)
    
    # Verify Account extraction and refinement
    assert df['Account'].iloc[0] == "Joint JTWROS", f"Account name should be cleaned, got: {df['Account'].iloc[0]}"
    assert str(df['Account#'].iloc[0]) == "9186", f"Account number should be extracted, got: {df['Account#'].iloc[0]}"
    
    # Verify symbol and other data
    assert df['Symbol'].iloc[0] == "AAPL"
    assert df['Quantity'].iloc[0] == 100

def test_et_j_csv_format():
    """Test ET_J.csv format where acquisition dates appear in Symbol column and Account is missing"""
    
    # Create a sample DataFrame mimicking ET_J.csv structure
    # Symbol column has: Symbol (aggregate row), then dates/markers (lot rows)
    # The '--' row mimics the XLK non-dated lot.
    data = {
        'Symbol': ['AAPL', '01/15/2020', '03/22/2021', 'XLK', '--', '08/15/2020'],
        'Quantity': [200, 100, 100, 150, 75, 75],
        'Cost per Unit': [150.00, 145.00, 155.00, 200.00, 195.00, 205.00],
        'Total Cost': [30000, 14500, 15500, 30000, 14625, 15375],
    }
    df = pd.DataFrame(data)
    
    # Column mapping
    col_map = {
        'symbol': 'Symbol',
        'quantity': 'Quantity',
        'cost per unit': 'Cost per Unit',
        'total cost': 'Total Cost',
    }
    
    # Test 1: Detect as symbol-group table
    assert _is_symbol_group_table(df, col_map), "Should detect ET_J.csv as symbol-group table even without Account"
    
    # Test 2: Parse symbol-group table
    parsed = _parse_symbol_group_table(df, col_map)
    
    # Should have 4 rows (2 for AAPL, 2 for XLK) - aggregate rows excluded
    assert len(parsed) == 4, f"Expected 4 rows, got {len(parsed)}"
    
    # Should have the temporary _AcquisitionDateFromSymbol column
    assert '_AcquisitionDateFromSymbol' in parsed.columns, "Should have _AcquisitionDateFromSymbol column"
    
    # Test 3: Check that symbols are filled correctly
    assert parsed['Symbol'].iloc[0] == 'AAPL', "First row should have AAPL symbol"
    assert parsed['Symbol'].iloc[1] == 'AAPL', "Second row should have AAPL symbol"
    assert parsed['Symbol'].iloc[2] == 'XLK', "Third row should have XLK symbol"
    assert parsed['Symbol'].iloc[3] == 'XLK', "Fourth row should have XLK symbol"
    
    # Test 4: Check that acquisition dates/markers are in the temporary column
    assert parsed['_AcquisitionDateFromSymbol'].iloc[0] == '01/15/2020', "First row should have date"
    assert parsed['_AcquisitionDateFromSymbol'].iloc[1] == '03/22/2021', "Second row should have date"
    assert parsed['_AcquisitionDateFromSymbol'].iloc[2] == '--', "Third row should have -- marker"
    assert parsed['_AcquisitionDateFromSymbol'].iloc[3] == '08/15/2020', "Fourth row should have date"
    
    # Test 5: Handle acquisition date from symbol
    result = handle_acquisition_date_from_symbol(parsed)
    
    # Should have Acquisition Date column
    assert 'Acquisition Date' in result.columns, "Should have Acquisition Date column"
    
    # Should NOT have the temporary column anymore
    assert '_AcquisitionDateFromSymbol' not in result.columns, "Should remove temporary column"
    
    # Test 6: Check that acquisition dates/markers are correctly mapped
    assert result['Acquisition Date'].iloc[0] == '01/15/2020', "First row should have correct date"
    assert result['Acquisition Date'].iloc[1] == '03/22/2021', "Second row should have correct date"
    assert result['Acquisition Date'].iloc[2] == '--', "Third row should have correct marker"
    assert result['Acquisition Date'].iloc[3] == '08/15/2020', "Fourth row should have correct date"
    
    # Test 7: Verify symbols are still correct
    assert result['Symbol'].iloc[0] == 'AAPL', "Symbol should still be AAPL"
    assert result['Symbol'].iloc[2] == 'XLK', "Symbol should still be XLK"
    
    print("✓ All ET_J.csv format tests passed!")


def test_backward_compatibility_old_format():
    """Test that old symbol-group format still works (Symbol empty, Account has description)"""
    
    # Old format: Symbol column is empty for lot rows, Account has security description
    data = {
        'Symbol': ['AAPL', None, None, 'MSFT', None],
        'Account': [None, 'Apple Inc.', 'Apple Inc.', None, 'Microsoft Corp.'],
        'Quantity': [200, 100, 100, 150, 150],
        'Cost per Unit': [150.00, 145.00, 155.00, 200.00, 195.00],
    }
    df = pd.DataFrame(data)
    
    col_map = {
        'symbol': 'Symbol',
        'account': 'Account',
        'quantity': 'Quantity',
        'cost per unit': 'Cost per Unit',
    }
    
    # Parse
    parsed = _parse_symbol_group_table(df, col_map)
    
    # Should have 3 rows (2 for AAPL, 1 for MSFT)
    assert len(parsed) == 3, f"Expected 3 rows, got {len(parsed)}"
    
    # Should have the temporary column (but with None values for old format)
    assert '_AcquisitionDateFromSymbol' in parsed.columns
    
    # Symbols should be filled
    assert parsed['Symbol'].iloc[0] == 'AAPL'
    assert parsed['Symbol'].iloc[1] == 'AAPL'
    assert parsed['Symbol'].iloc[2] == 'MSFT'
    
    # Temporary column should have None for old format
    assert pd.isna(parsed['_AcquisitionDateFromSymbol'].iloc[0]) or parsed['_AcquisitionDateFromSymbol'].iloc[0] is None
    
    # Handle acquisition date - should not break
    result = handle_acquisition_date_from_symbol(parsed)
    
    # Should remove temporary column
    assert '_AcquisitionDateFromSymbol' not in result.columns
    
    print("✓ Backward compatibility test passed!")


if __name__ == '__main__':
    test_et_j_csv_format()
    test_backward_compatibility_old_format()
    print("\n✓ All tests passed successfully!")
