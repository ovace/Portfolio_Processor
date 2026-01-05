"""
utils/portfolio_utils.py

Facade and Orchestrator for portfolio processing.
Delegates specific responsibilities to:
- utils.readers (Data Access)
- utils.processing (Domain Logic)
- utils.detection (Structure Detection)
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import pandas as pd

# Re-exports for backward compatibility and API surfacing
from .readers import read_csv_file, read_excel_file
from .processing import (
    normalise_table, 
    find_key_value_pairs, 
    extract_portfolio_fields, 
    extract_metadata_from_df, # Used in process_file
    refine_metadata, # Used in process_*
    cleanup_portfolio_df, 
    _fill_symbol_from_description, # Used in process_*
    forward_fill_sparse_columns, # Used in process_*
    handle_acquisition_date_from_symbol, # Used in process_* for ET_J.csv
    _calculate_total_cost, # Automatically calculate TC
    _build_rename_map,
    _normalise_header,
    _audit as _processing_audit # We might want to unify audit
)
from .detection import (
    detect_file_structure,
    _is_symbol_group_table,
    _parse_symbol_group_table,
    _is_activity_table,
    _parse_activity_table
)

if TYPE_CHECKING:
    from utils.reporting_utils import FileReport

# -----------------------------------------------------------------------------
# Logging setup (Facade level)
# -----------------------------------------------------------------------------
DEBUG_ENV = os.getenv("DEBUG", "false").lower() == "true"
logger = logging.getLogger(__name__)
if not logger.handlers:
    level = logging.DEBUG if DEBUG_ENV else logging.INFO
    fmt = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.setLevel(level)
    try:
        log_dir = os.getenv("LOG_DIR", "logs")
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "app.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass

# Audit Shim to unify with submodules if they log differently
# For now, we reuse the pattern.
_audit_log: List[Tuple[str, str]] = []

def _audit(message: str) -> None:
    if os.getenv("AUDIT", "true").lower() == "true":
        ts = _dt.datetime.now().isoformat(timespec="seconds")
        _audit_log.append((ts, message))
        logger.info(f"AUDIT: {message}")

def get_audit_log() -> List[Tuple[str, str]]:
    return list(_audit_log)

# -----------------------------------------------------------------------------
# Configuration (Keep in Orchestrator for now)
# -----------------------------------------------------------------------------

def load_column_mapping(config_path: str) -> Dict[str, List[str]]:
    """
    Load column mapping from JSON. 
    Returns: { CanonicalName: [List, of, Variants, in, priority, order] }
    """
    env_path = os.getenv("COLUMN_MAPPING_CONFIG") or config_path
    mapping: Dict[str, List[str]] = {}
    try:
        if not os.path.isfile(env_path):
            logger.warning(f"Column mapping config not found at {env_path}; using empty mapping")
            return {}
        with open(env_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        
        for k, v in raw.items():
            if isinstance(v, list):
                # Standardize variants but keep order
                mapping[k] = [_normalise_header(variant) for variant in v]
            else:
                mapping[k] = [_normalise_header(str(v))]
        
        _audit(f"Loaded column mapping from {env_path} with {len(mapping)} canonical fields")
    except Exception as exc:
        logger.error(f"Failed to load column mapping: {exc}")
    return mapping

def flatten_column_mapping(grouped_map: Dict[str, List[str]]) -> Dict[str, str]:
    """
    Utility to convert grouped mapping {Canonical: [Variants]} to flat {Variant: Canonical}.
    Note: If multiple variants map to different canonical names (unlikely), 
    the last one in the dict will win in a flat structure.
    """
    flat: Dict[str, str] = {}
    for canonical, variants in grouped_map.items():
        for v in variants:
            flat[v] = canonical
    return flat

def get_output_fields(default: Optional[List[str]] | None = None) -> List[str]:
    env = os.getenv("OUTPUT_FIELDS")
    if env:
        s = env.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                pass
        return [p.strip() for p in s.split(",") if p.strip()]
    if default is not None:
        return default
    return [
        "Account", "Account#", "Symbol", "Quantity", "Acquisition Date",
        "Cost per Unit", "Total Cost", "Value", "Type", "Broker"
    ]

def _extract_broker_from_filename(path: str) -> Optional[str]:
    try:
        base = os.path.basename(path)
        stem, _ = os.path.splitext(base)
        return (stem.split("_")[0] or None) if stem else None
    except Exception:
        return None

# -----------------------------------------------------------------------------
# Orchestrators
# -----------------------------------------------------------------------------

def process_file(
    input_path: str,
    config_path: str,
    output_dir: str = "./out",
    timestamp: Optional[str] = None,
    *,
    tabs: Optional[List[str]] = None,
    append: bool = False,
    file_report: Optional["FileReport"] = None,
) -> str:
    if not os.path.isfile(input_path):
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(input_path)
    logger.debug(f"Input file is available: {input_path}")

    ext = os.path.splitext(input_path)[1].lower()
    if ext not in {".csv", ".xls", ".xlsx"}:
        raise ValueError(f"Unsupported file extension: {ext}")

    col_map = load_column_mapping(config_path)
    flat_map = flatten_column_mapping(col_map)
    logger.debug(f"Processing file {input_path} with {len(col_map)} canonical fields")

    if ext == ".csv":
        tables, lines = read_csv_file(input_path)
    else:
        tables, lines = read_excel_file(input_path, sheets_to_process=tabs)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Process_file: loaded {len(tables)} table(s)")

    if file_report is not None:
        try:
            file_report.tables_detected = len(tables)
            file_report.rows_read_raw = sum(len(t) for t in tables if hasattr(t, "__len__"))
            # Track original columns
            for t in tables:
                try:
                    file_report.original_columns.update(str(c) for c in t.columns)
                except Exception:
                    pass
        except Exception:
            pass

    key_values = find_key_value_pairs(lines, flat_map)

    required_fields = get_output_fields()
    if "Broker" in required_fields:
        b = _extract_broker_from_filename(input_path)
        if b:
            key_values.setdefault("Broker", b)

    global_metadata = key_values.copy()
    extracted: List[pd.DataFrame] = []
    
    for tbl in tables:
        try:
            if tbl.empty:
                continue
                
            tbl_to_use = tbl
            processed_as_holdings = False
            
            # 1. Try special realignments (Symbol-Group, Activity)
            try:
                if _is_symbol_group_table(tbl, flat_map):
                    p = _parse_symbol_group_table(tbl, flat_map)
                    if not p.empty:
                        tbl_to_use = p
                        processed_as_holdings = True
                elif _is_activity_table(tbl):
                    tbl_to_use = _parse_activity_table(tbl, flat_map)
                    processed_as_holdings = True
            except Exception:
                pass

            # 2. Normalise to see what columns we actually have
            norm = normalise_table(tbl_to_use, col_map)
            
            # 3. Determine if this table is a holdings table vs metadata/summary
            # Broaden check: Symbol, Security Description, or was already processed as special format
            has_holdings_id = "Symbol" in norm.columns or "Security Description" in norm.columns or processed_as_holdings
            
            if not has_holdings_id:
                meta = extract_metadata_from_df(tbl, flat_map)
                if meta:
                    global_metadata.update(refine_metadata(meta))
                continue # Definitely a summary/metadata table

            # 4. Process as a holdings table
            # Handle ET_J.csv format where acquisition dates are in Symbol column
            norm = handle_acquisition_date_from_symbol(norm)
            # Forward-fill sparse Security Description/Symbol columns
            norm = forward_fill_sparse_columns(norm, col_map)
            # Backfill Symbol from Security Description if missing
            norm = _fill_symbol_from_description(norm)
            
            # Calculate Total Cost if missing factors are present
            norm = _calculate_total_cost(norm)

            present = {f for f in required_fields if f in norm.columns}
            if len(present) < 2:
                # Might still have metadata
                meta = extract_metadata_from_df(tbl_to_use, flat_map)
                if meta:
                    global_metadata.update(refine_metadata(meta))
                continue

            clean, blank_removed, dupes_removed = cleanup_portfolio_df(norm)
            
            # Track cleanup metrics and normalized columns
            if file_report:
                file_report.rows_removed_blank += blank_removed
                file_report.rows_removed_duplicates += dupes_removed
                file_report.normalized_columns.update(str(c) for c in norm.columns)
            
            if clean.empty:
                # If everything was blank, maybe it was still a meta source?
                meta = extract_metadata_from_df(tbl_to_use, col_map)
                if meta:
                    global_metadata.update(refine_metadata(meta))
                continue
            
            # Refine global metadata before extraction
            final_kv = refine_metadata(global_metadata)
            
            final = extract_portfolio_fields(clean, required_fields, final_kv)
            if not final.empty:
                extracted.append(final)
        except Exception as exc:
            logger.debug(f"Skipping table due to error: {exc}")

    if not extracted:
        # Fallback: if single table CSV and failed 'present' check, maybe just raw dump?
        # But we'll respect strict processing.
        msg = "No valid portfolio tables found."
        if file_report:
            file_report.error_summary = msg
        logger.warning(msg)
        return ""

    combined = pd.concat(extracted, ignore_index=True)
    if combined.empty:
        return ""
    
    # Apply forward-fill to combined result to handle multi-table scenarios
    # where some tables have Security Description/Symbol and others don't
    combined = forward_fill_sparse_columns(combined, col_map)
    
    if not timestamp:
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"portfolio-{timestamp}.csv"
    out_path = os.path.join(output_dir, out_name)
    os.makedirs(output_dir, exist_ok=True)
    
    combined.to_csv(out_path, index=False, mode="a" if append else "w", header=not append or not os.path.exists(out_path))
    _audit(f"Wrote output CSV to {out_path}")
    
    if file_report:
        file_report.total_rows_out = len(combined)
        
    return out_path

def process_hybrid_file(
    input_path: str,
    config_path: str,
    output_dir: str = "./out",
    timestamp: Optional[str] = None,
    *,
    tabs: Optional[List[str]] = None,
    append: bool = False,
    file_report: Optional["FileReport"] = None,
) -> str:
    from .detection import _is_account_summary_table, _is_holdings_table, _parse_account_summary, _parse_hybrid_holdings

    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)
    
    col_map = load_column_mapping(config_path)
    flat_map = flatten_column_mapping(col_map)
    ext = os.path.splitext(input_path)[1].lower()
    
    if ext == ".csv":
        tables, lines = read_csv_file(input_path)
    else:
        tables, lines = read_excel_file(input_path, sheets_to_process=tabs)

    acc_idx = None
    account_name = None
    
    for i, t in enumerate(tables):
        if _is_account_summary_table(t):
            acc_idx = i
            account_name = _parse_account_summary(t, flat_map)
            break
            
    if acc_idx is None:
        logger.warning("Hybrid processing called but no Account Summary found.")
        return ""

    holdings_tables = []
    for j in range(acc_idx + 1, len(tables)):
        t = tables[j]
        if _is_holdings_table(t):
            processed = _parse_hybrid_holdings(t)
            if not processed.empty:
                holdings_tables.append(processed)

    if not holdings_tables:
        logger.warning("Hybrid processing called but no following Holdings tables found.")
        return ""
        
    combined = pd.concat(holdings_tables, ignore_index=True)
    
    # Norm + extract
    norm = normalise_table(combined, col_map)
    # Handle ET_J.csv format where acquisition dates are in Symbol column
    norm = handle_acquisition_date_from_symbol(norm)
    # Forward-fill sparse Security Description/Symbol columns
    norm = forward_fill_sparse_columns(norm, col_map)
    norm = _fill_symbol_from_description(norm)

    if file_report:
        file_report.tables_detected = len(tables)
        try:
            file_report.rows_read_raw = sum(len(t) for t in tables if hasattr(t, "__len__"))
            # Track original columns
            for t in tables:
                try:
                    file_report.original_columns.update(str(c) for c in t.columns)
                except Exception:
                    pass
        except Exception:
            pass

    required = get_output_fields()
    kv = {}
    
    # Extract Broker if required
    if "Broker" in required:
        b = _extract_broker_from_filename(input_path)
        if b:
            kv["Broker"] = b

    if account_name:
        kv["Account"] = account_name
    
    # Try to find other KVs from lines just in case
    # checking simplistic approach
    from_lines = find_key_value_pairs(lines, flat_map)
    for k, v in from_lines.items():
        if k not in kv:
            kv[k] = v
            
    # Refine the collected metadata (e.g., split combined account strings)
    kv = refine_metadata(kv)
    
    cleaned, blank_removed, dupes_removed = cleanup_portfolio_df(norm)
    
    # Track cleanup metrics and normalized columns
    if file_report:
        file_report.rows_removed_blank = blank_removed
        file_report.rows_removed_duplicates = dupes_removed
        file_report.rows_after_cleanup = len(cleaned)
        file_report.normalized_columns.update(str(c) for c in norm.columns)
    
    final = extract_portfolio_fields(cleaned, required, kv)
    
    if final.empty:
        return ""
        
    if not timestamp:
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"portfolio-{timestamp}.csv"
    out_path = os.path.join(output_dir, out_name)
    os.makedirs(output_dir, exist_ok=True)
    
    if file_report:
        file_report.total_rows_out = len(final)
        
    final.to_csv(out_path, index=False)
    _audit(f"Wrote output CSV to {out_path}")
    return out_path
