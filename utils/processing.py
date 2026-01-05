"""
utils/processing.py

Domain logic for normalizing, cleaning, and extracting data from partial DataFrames.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import datetime as _dt

logger = logging.getLogger(__name__)

# Replicate _audit logic or import? Better to keep self-contained for now to avoid circular deps with portfolio_utils.
# In a perfect world, we'd have a common.py.
def _audit(message: str) -> None:
    logger.info(f"AUDIT: {message}")

def _normalise_header(header: str) -> str:
    cleaned = re.sub(r"[\-_/]+", " ", str(header).strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned

def find_key_value_pairs(lines: Iterable[str], col_map: Dict[str, str]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for line in lines:
        try:
            for seg in re.split(r"[;,]", line):
                if ":" in seg:
                    k, v = seg.split(":", 1)
                    key = _normalise_header(k)
                    if key in col_map and v.strip():
                        result[col_map[key]] = v.strip()
                elif "=" in seg:
                    k, v = seg.split("=", 1)
                    key = _normalise_header(k)
                    if key in col_map and v.strip():
                        result[col_map[key]] = v.strip()
        except Exception as exc:
            logger.debug(f"Failed to parse line '{line}': {exc}")
    if result:
        _audit(f"Extracted {len(result)} key-value pairs from non-tabular data")
    return result

def normalise_table(df: pd.DataFrame, grouped_map: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Standardizes column names in the DataFrame based on a prioritized mapping.
    For each canonical field, the variants are checked in the order they appear 
    in the mapping list. The first matching column in the DataFrame is selected.
    
    Args:
        df: Input DataFrame
        grouped_map: { CanonicalName: [Variants, in, order, of, priority] }
    """
    rename_dict: Dict[pd.api.types.Hashable, str] = {}
    try:
        # Pre-normalize current column names for comparison
        col_to_norm = {col: _normalise_header(str(col)) for col in df.columns}
        norm_to_cols = {}
        for col, norm in col_to_norm.items():
            norm_to_cols.setdefault(norm, []).append(col)

        # For each target canonical name, find the best matching physical column
        for canonical, variants in grouped_map.items():
            best_match_col = None
            for variant in variants:
                norm_variant = _normalise_header(str(variant))
                if norm_variant in norm_to_cols:
                    # Found a match for this variant. High priority variants come first.
                    # If multiple columns match the SAME variant, we pick the first physical one.
                    best_match_col = norm_to_cols[norm_variant][0]
                    break
            
            if best_match_col:
                rename_dict[best_match_col] = canonical
                
        if rename_dict:
            # Note: rename(columns=...) returns a copy by default
            df = df.rename(columns=rename_dict)
            _audit(f"Renamed {len(rename_dict)} column(s) using prioritized mapping")
    except Exception as exc:
        logger.error(f"Error normalising table columns with priority: {exc}")
    return df

def forward_fill_sparse_columns(df: pd.DataFrame, col_map: Dict[str, str]) -> pd.DataFrame:
    """
    Forward-fill sparse columns where Security Description or Symbol values
    only appear in the first row(s) of the table.
    
    This handles cases like PR_O.xlsx where the security name appears once
    at the top and subsequent rows contain transaction details.
    
    Args:
        df: DataFrame with normalized column names
        col_map: Column mapping (not used directly, but kept for consistency)
    
    Returns:
        DataFrame with forward-filled Security Description and Symbol columns
    """
    try:
        if df.empty:
            return df
        
        # Target columns to forward-fill
        target_columns = ["Security Description", "Symbol"]
        
        for col in target_columns:
            if col in df.columns:
                # Check if column has any sparse data (NaN/empty values)
                non_empty = df[col].notna() & (df[col].astype(str).str.strip() != "") & (df[col].astype(str).str.lower() != "nan")
                non_empty_count = non_empty.sum()
                
                # If there are ANY empty values and at least one non-empty value, apply forward-fill
                # This handles both sparse columns and multi-table concatenation scenarios
                if non_empty_count > 0 and non_empty_count < len(df):
                    df[col] = df[col].ffill()
                    _audit(f"Forward-filled {col} column ({non_empty_count} -> {len(df)} rows)")
        
        return df
    except Exception as exc:
        logger.warning(f"Failed to forward-fill sparse columns: {exc}")
        return df

def handle_acquisition_date_from_symbol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle the special _AcquisitionDateFromSymbol column created by symbol-group parser
    for ET_J.csv files where acquisition dates appear in the Symbol column.
    
    This function:
    1. Checks if _AcquisitionDateFromSymbol column exists
    2. If Acquisition Date column doesn't exist or is empty, populate it from _AcquisitionDateFromSymbol
    3. Remove the temporary _AcquisitionDateFromSymbol column
    
    Args:
        df: DataFrame potentially containing _AcquisitionDateFromSymbol column
    
    Returns:
        DataFrame with Acquisition Date properly populated and temporary column removed
    """
    try:
        if df.empty or '_AcquisitionDateFromSymbol' not in df.columns:
            return df
        
        # Create Acquisition Date column if it doesn't exist
        if 'Acquisition Date' not in df.columns:
            df['Acquisition Date'] = None
        
        # Check which rows have dates in the temporary column
        has_date_from_symbol = df['_AcquisitionDateFromSymbol'].notna()
        
        # Check which rows have empty Acquisition Date
        acq_date_empty = df['Acquisition Date'].isna() | (df['Acquisition Date'].astype(str).str.strip() == "") | (df['Acquisition Date'].astype(str).str.lower() == "nan")
        
        # Fill Acquisition Date from _AcquisitionDateFromSymbol where applicable
        mask = has_date_from_symbol & acq_date_empty
        count = mask.sum()
        
        if count > 0:
            df.loc[mask, 'Acquisition Date'] = df.loc[mask, '_AcquisitionDateFromSymbol']
            _audit(f"Populated {count} Acquisition Date value(s) from Symbol column (ET_J.csv format)")
        
        # Remove the temporary column
        df = df.drop(columns=['_AcquisitionDateFromSymbol'])
        
        return df
    except Exception as exc:
        logger.warning(f"Failed to handle acquisition date from symbol column: {exc}")
        # Try to at least remove the temporary column
        try:
            if '_AcquisitionDateFromSymbol' in df.columns:
                df = df.drop(columns=['_AcquisitionDateFromSymbol'])
        except Exception:
            pass
        return df


def extract_portfolio_fields(df: pd.DataFrame, required_fields: Iterable[str], key_values: Dict[str, str]) -> pd.DataFrame:
    data: Dict[str, List[Any]] = {}
    try:
        for field in required_fields:
            if field in df.columns:
                series = df[field]
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                data[field] = series.tolist()
            else:
                data[field] = [key_values.get(field)] * len(df)
        _audit(f"Extracted portfolio fields for {len(df)} row(s)")
        return pd.DataFrame(data)
    except Exception as exc:
        logger.error(f"Failed to extract portfolio fields: {exc}")
        return pd.DataFrame(columns=list(required_fields))

def extract_metadata_from_df(df: pd.DataFrame, col_map: Dict[str, str]) -> Dict[str, str]:
    """
    Extract first non-empty value for each recognized field from a DataFrame.
    Searches both columns and row contents for recognized keys.
    """
    metadata: Dict[str, str] = {}
    try:
        if df.empty:
            return metadata
            
        # 1. Try column-based extraction (if headers already match)
        norm = normalise_table(df, col_map)
        for col in norm.columns:
            if col in ("Symbol", "Quantity", "Acquisition Date", "Cost per Unit", "Total Cost", "Value"):
                continue
            first_val = norm[col].dropna()
            if not first_val.empty:
                val = str(first_val.iloc[0]).strip()
                if val and val.lower() != "nan":
                    metadata[col] = val

        # 2. Try row-based extraction (if keys are inside the table body)
        # We look for a cell that matches a col_map key, and take the cell to its right or below it.
        # This handles cases where "Account" is just a cell in a summary table.
        for row_i in range(len(df)):
            for col_i in range(len(df.columns)):
                cell_val = str(df.iloc[row_i, col_i]).strip()
                norm_cell = _normalise_header(cell_val)
                if norm_cell in col_map:
                    target_field = col_map[norm_cell]
                    if target_field in ("Symbol", "Quantity", "Value"): # Skip data fields
                        continue
                        
                    # Found a key! Check value to the right
                    if col_i + 1 < len(df.columns):
                        val = str(df.iloc[row_i, col_i + 1]).strip()
                        if val and val.lower() != "nan" and _normalise_header(val) not in col_map:
                             if target_field not in metadata:
                                 metadata[target_field] = val
                    
                    # Also check value below if right was empty
                    if target_field not in metadata and row_i + 1 < len(df):
                        val = str(df.iloc[row_i + 1, col_i]).strip()
                        if val and val.lower() != "nan" and _normalise_header(val) not in col_map:
                            metadata[target_field] = val
                            
        if metadata:
            _audit(f"Extracted metadata from summary table: {metadata}")
    except Exception as exc:
        logger.debug(f"Failed to extract metadata from DF: {exc}")
    return metadata

def refine_metadata(metadata: Dict[str, str]) -> Dict[str, str]:
    """
    Refine metadata by applying rules like splitting Account strings.
    Example: 'Joint JTWROS -9186' -> Account: 'Joint JTWROS', Account#: '9186'
    """
    refined = metadata.copy()
    
    # 1. Handle Account -> Account# splitting
    acc = refined.get("Account")
    acc_num = refined.get("Account#")
    
    if acc and not acc_num:
        # Look for pattern '- \d+' or ' \d+' at the end
        match = re.search(r"[- ]+(\d+)$", acc)
        if match:
            num = match.group(1)
            base = acc[:match.start()].strip()
            # If the base ends with a hyphen, strip it too
            base = base.rstrip("-").strip()
            
            refined["Account#"] = num
            refined["Account"] = base
            _audit(f"Split Account '{acc}' into Name: '{base}' and Number: '{num}'")
            
    return refined

def cleanup_portfolio_df(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """
    Clean portfolio DataFrame by removing blank rows and duplicates.
    
    Returns:
        tuple: (cleaned_df, rows_removed_blank, rows_removed_duplicates)
    """
    try:
        if df.empty:
            return df, 0, 0
        cleaned = df.copy()
        cleaned = cleaned.replace(r"^\s*$", None, regex=True)
        before = len(cleaned)
        cleaned = cleaned.dropna(how="all")
        dropped_blank = before - len(cleaned)
        before = len(cleaned)
        cleaned = cleaned.drop_duplicates(keep="first")
        dropped_dupes = before - len(cleaned)
        if dropped_blank or dropped_dupes:
            _audit(f"Cleaned portfolio dataframe: removed {dropped_blank} blank row(s) and {dropped_dupes} duplicate row(s)")
        return cleaned, dropped_blank, dropped_dupes
    except Exception as exc:
        logger.warning(f"Failed to clean portfolio dataframe: {exc}")
        return df, 0, 0

def _fill_symbol_from_description(df: pd.DataFrame) -> pd.DataFrame:
    """
    If a row has a missing Symbol but a valid Security Description, use the Description as the Symbol.
    This should happen BEFORE cleanup/deduplication.
    """
    try:
        if df.empty or "Security Description" not in df.columns:
            return df

        if "Symbol" not in df.columns:
            df["Symbol"] = None

        sym_check = df["Symbol"].astype(str).str.strip().str.lower()
        missing_sym_mask = sym_check.isin(["", "nan", "none", "<na>"]) | df["Symbol"].isna()
        
        desc_check = df["Security Description"].astype(str).str.strip().str.lower()
        valid_desc_mask = (~desc_check.isin(["", "nan", "none", "<na>"])) & df["Security Description"].notna()
        
        mask = missing_sym_mask & valid_desc_mask
        
        count = mask.sum()
        if count > 0:
            df.loc[mask, "Symbol"] = df.loc[mask, "Security Description"]
            _audit(f"Backfilled {count} missing Symbol(s) using Security Description")

        return df
    except Exception as exc:
        logger.warning(f"Failed to backfill symbols from description: {exc}")
        return df

def _build_rename_map(original_cols, normalized_df) -> Dict[str, str]:
    try:
        orig = [str(c) for c in list(original_cols)]
        new = [str(c) for c in list(normalized_df.columns)]
        rename_map: Dict[str, str] = {}
        missing = [c for c in orig if c not in new]
        for can in new:
            if can in orig:
                continue
            tokens_can = set(str(can).lower().replace("_", " ").split())
            match = None
            for m in missing:
                tokens_m = set(str(m).lower().replace("_", " ").split())
                if tokens_can & tokens_m:
                    match = m
                    break
            if match:
                rename_map[match] = can
        return rename_map
    except Exception:
        return {}

def _calculate_total_cost(df: pd.DataFrame) -> pd.DataFrame:
    """
    Automatically calculate Total Cost if it's missing but Quantity and Cost per Unit are present.
    """
    try:
        if df.empty:
            return df
        
        required = ["Quantity", "Cost per Unit", "Total Cost"]
        if not all(col in df.columns for col in required):
            return df
            
        # Create numeric copies for calculation
        q = pd.to_numeric(df["Quantity"], errors='coerce')
        cpu = pd.to_numeric(df["Cost per Unit"], errors='coerce')
        tc = pd.to_numeric(df["Total Cost"], errors='coerce')
        
        # Mask for rows where TC is missing but Q and CPU are present
        mask = tc.isna() & q.notna() & cpu.notna()
        
        count = mask.sum()
        if count > 0:
            calculated_vals = q[mask] * cpu[mask]
            df.loc[mask, "Total Cost"] = calculated_vals
            _audit(f"Calculated Total Cost for {count} row(s) using Quantity * Cost per Unit")
            
        return df
    except Exception as exc:
        logger.warning(f"Failed to calculate total cost: {exc}")
        return df
