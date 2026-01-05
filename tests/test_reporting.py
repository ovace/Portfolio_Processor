import pandas as pd
import pytest
from utils.reporting_utils import FileReport, OverallReport

def test_file_report_initialization():
    fr = FileReport(input_path="test.csv", structure="standard")
    assert fr.input_path == "test.csv"
    assert fr.structure == "standard"
    assert fr.tables_detected == 0
    assert fr.total_rows_out == 0

def test_file_report_add_table_sample():
    fr = FileReport(input_path="test.csv", structure="standard")
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    fr.add_table_sample(0, df)
    
    assert len(fr.table_samples) == 1
    sample = fr.table_samples[0]["table_0_sample"]
    assert len(sample) == 2  # head(2)
    assert sample[0]["A"] == 1

def test_file_report_finalize_columns():
    fr = FileReport(input_path="test.csv", structure="standard")
    fr.original_columns = {"Col1", "Col2", "Col3"}
    fr.normalized_columns = {"Symbol", "Quantity"}
    
    # Mapped Col1 -> Symbol, Col2 -> Quantity. Col3 is unused.
    col_map_used = {"Col1": "Symbol", "Col2": "Quantity"}
    output_fields = ["Symbol", "Quantity", "Value"]
    
    fr.finalize_columns(output_fields, col_map_used)
    
    assert fr.rename_map == col_map_used
    assert fr.missing_canonical == ["Value"]
    assert fr.unused_input_columns == ["Col3"]

def test_overall_report_aggregation():
    fr1 = FileReport(input_path="f1.csv", structure="standard")
    fr1.total_rows_out = 10
    fr1.tables_detected = 1
    fr1.original_columns = {"A"}
    
    fr2 = FileReport(input_path="f2.csv", structure="hybrid")
    fr2.total_rows_out = 20
    fr2.tables_detected = 2
    fr2.original_columns = {"B"}
    
    overall = OverallReport()
    overall.add(fr1)
    overall.add(fr2)
    
    agg = overall.aggregate()
    assert agg["files_processed"] == 2
    assert agg["rows_out_total"] == 30
    assert agg["tables_detected_total"] == 3
    assert "A" in agg["original_columns_union"]
    assert "B" in agg["original_columns_union"]
    assert agg["row_counts_by_file"]["f1.csv"] == 10
    assert agg["row_counts_by_file"]["f2.csv"] == 20
