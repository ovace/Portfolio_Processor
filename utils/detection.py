"""
utils/detection.py

Logic for identifying file structures (Standard, Hybrid) and identifying table types.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Any

import pandas as pd

from .readers import read_csv_file, read_excel_file
from .processing import normalise_table, _normalise_header, _audit

logger = logging.getLogger(__name__)

def _is_account_summary_table(df: pd.DataFrame) -> bool:
    try:
        headers = [str(c).strip().lower() for c in df.columns]
        has_account = any(h == "account" for h in headers)
        has_summary = any(any(term in h for term in ("net account value", "total gain", "available", "cash")) for h in headers)
        return bool(has_account and has_summary)
    except Exception:
        return False

def _is_holdings_table(df: pd.DataFrame) -> bool:
    try:
        headers = [str(c).strip().lower() for c in df.columns]
        has_date = any(any(t in h for t in ["date acquired", "purchase date", "initial purchase date"]) for h in headers)
        has_qty = any(("qty" in h) or ("quantity" in h) for h in headers)
        has_symbol = any(h == "symbol" for h in headers)
        return bool(has_date and has_qty and has_symbol)
    except Exception:
        return False

def _parse_account_summary(df: pd.DataFrame, grouped_map: Dict[str, List[str]]) -> Optional[str]:
    try:
        if df.empty:
            return None
        dn = normalise_table(df.copy(), grouped_map)
        if "Account" in dn.columns:
            for v in dn["Account"]:
                if v is not None and str(v).strip():
                    return str(v).strip()
        return None
    except Exception as exc:
        logger.debug(f"Failed to parse account summary: {exc}")
        return None

def _parse_hybrid_holdings(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if df.empty:
            return df.copy()

        c0 = df.iloc[:, 0].astype(str).str.strip()
        
        # Regex for "Is Date" (d/m/y or m/d/y)
        is_date = c0.str.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", na=False)
        
        # Identify symbol definition rows
        is_valid_str = (c0 != "") & (c0.str.lower() != "nan")
        is_symbol_row = (~is_date) & is_valid_str
        
        symbols = c0.where(is_symbol_row).ffill()
        
        mask = is_date & symbols.notna()
        
        if not mask.any():
            return pd.DataFrame(columns=df.columns)
        
        result = df.loc[mask].copy()
        col0 = df.columns[0]
        result[col0] = symbols[mask]
        
        return result
    except Exception as exc:
        logger.debug(f"Failed to realign hybrid holdings: {exc}")
        return pd.DataFrame(columns=list(df.columns))

def _is_symbol_group_table(df: pd.DataFrame, flat_map: Dict[str, str]) -> bool:
    try:
        if df.empty:
            return False
        sym_i = -1
        desc_i = -1
        data_i = -1
        
        # Identify key columns more flexibly to handle mapping ambiguities (e.g. Description -> Account)
        for i, c in enumerate(df.columns):
            norm = _normalise_header(str(c))
            can = flat_map.get(norm)
            if can == "Symbol":
                sym_i = i
            elif can in ("Security Description", "Account") or norm == "description":
                if desc_i < 0: desc_i = i
            elif can in ("Quantity", "Value", "Cost per Unit"):
                if data_i < 0: data_i = i

        if sym_i < 0 or data_i < 0:
            return False
            
        def _looks_like_lot_marker(val: Any) -> bool:
            if val is None or pd.isna(val):
                return False
            s = str(val).strip()
            if s == "--": return True
            import re
            return any(re.match(pattern, s) for pattern in [r'^\d{1,2}/\d{1,2}/\d{4}$', r'^\d{4}-\d{2}-\d{2}$'])
        
        found_header = False
        found_lot = False
        
        account_keywords = {"IRA", "ROTH", "TRUST", "JOINT", "NFS", "ACC", "ACCOUN"}
        import re
        
        for row in df.itertuples(index=False, name=None):
            try:
                sym_val = row[sym_i]
                data_val = row[data_i]
                desc_val = row[desc_i] if desc_i >= 0 else None
            except Exception:
                continue
            
            sym_empty = (sym_val is None) or (pd.isna(sym_val)) or (str(sym_val).strip().lower() in ("", "nan"))
            data_empty = (data_val is None) or (pd.isna(data_val)) or (str(data_val).strip().lower() in ("", "nan"))
            
            if sym_empty and data_empty:
                continue
            
            # 1. Detection by Symbol/Lot Marker (ET_J.csv)
            if not sym_empty:
                if _looks_like_lot_marker(sym_val) or str(sym_val).startswith("  "):
                    found_lot = True
                else:
                    found_header = True
            
            # 2. Detection by Shared Column Pattern (CW.xlsx)
            if sym_empty and not data_empty and desc_i >= 0:
                d_str = str(desc_val).strip()
                d_upper = d_str.upper()
                is_account = any(kw in d_upper for kw in account_keywords) or bool(re.search(r'\(B\d+\)', d_str))
                if is_account:
                    found_lot = True
            
            if found_header and found_lot:
                return True
                
        return bool(found_header and found_lot)
    except Exception as exc:
        logger.debug(f"Error in _is_symbol_group_table: {exc}")
        return False

def _parse_symbol_group_table(df: pd.DataFrame, flat_map: Dict[str, str]) -> pd.DataFrame:
    """
    Overhauled parser for hierarchical symbol-group tables.
    Handles Category Type, Aggregate filtering, and Shared Column resolution.
    """
    try:
        if df.empty:
            return df.copy()
            
        sym_i = -1
        desc_i = -1
        acc_i = -1
        data_koll = [] # Indices of data columns to help detect categories/summaries
        
        for i, c in enumerate(df.columns):
            norm = _normalise_header(str(c))
            can = flat_map.get(norm)
            if can == "Symbol":
                sym_i = i
            elif can in ("Security Description", "Account") or norm == "description":
                if desc_i < 0: 
                    desc_i = i
                if can == "Account":
                    acc_i = i
            elif can in ("Quantity", "Value", "Cost per Unit"):
                data_koll.append(i)
        
        if sym_i < 0:
            return df

        new_rows = []
        current_type = None
        current_symbol = None
        current_description = None
        
        account_keywords = {"IRA", "ROTH", "TRUST", "JOINT", "NFS", "ACC", "ACCOUN"}
        type_keywords = {"CASH", "CASH EQUIVALENT", "MUTUAL FUND", "EQUITY", "ETF", "FIXED INCOME", "UNKNOWN"}
        import re

        def _is_lot_marker(val: Any) -> bool:
            if val is None or pd.isna(val):
                return False
            s = str(val).strip()
            if s == "--": return True
            return any(re.match(p, s) for p in [r'^\d{1,2}/\d{1,2}/\d{4}$', r'^\d{4}-\d{2}-\d{2}$'])

        for row in df.itertuples(index=False, name=None):
            row_list = list(row)
            try:
                raw_sym = row_list[sym_i]
                raw_desc = row_list[desc_i]
                # Is ANY numeric data present?
                data_present = False
                for di in data_koll:
                    dv = row_list[di]
                    if dv is not None and not pd.isna(dv) and str(dv).strip().lower() not in ("", "nan"):
                        data_present = True
                        break
            except Exception:
                continue
                
            sym_empty = (raw_sym is None) or (pd.isna(raw_sym)) or (str(raw_sym).strip().lower() in ("", "nan"))
            desc_empty = (raw_desc is None) or (pd.isna(raw_desc)) or (str(raw_desc).strip().lower() in ("", "nan"))
            
            if sym_empty and desc_empty and not data_present:
                continue

            s_str = str(raw_sym).strip()
            s_upper = s_str.upper()
            d_str = str(raw_desc).strip()
            d_upper = d_str.upper()
            
            # --- 1. Identify Category Headers ---
            # Broad check: If it looks like a type name and data is missing
            is_potential_type = s_upper in type_keywords or d_upper in type_keywords
            # Special case: Type matches in both columns (e.g. Cash / Cash)
            if (is_potential_type or (s_upper == d_upper and s_upper)) and not data_present:
                current_type = d_str if d_upper in type_keywords else s_str
                continue
            
            # Skip explicit Totals
            if d_upper.startswith("TOTAL") or s_upper.startswith("TOTAL"):
                continue

            # --- 2. Identify and Resolve Rows ---
            is_account_lot = any(kw in d_upper for kw in account_keywords) or bool(re.search(r'\(B\d+\)', d_str))

            if not sym_empty:
                if _is_lot_marker(raw_sym) or s_str.startswith("  "):
                    # Lot Row (ET_J.csv style)
                    if not current_symbol: continue
                    row_list[sym_i] = current_symbol
                    # Fill description from security if empty or matching the aggregate name
                    if not d_str or d_str == current_description:
                        row_list[desc_i] = current_description
                    new_rows.append(row_list + [s_str, None, current_type])
                else:
                    # Security Summary / Aggregate Row (Header)
                    current_symbol = s_str
                    current_description = d_str
                    # Do not output aggregate rows that have lots underneath
                    continue
            else:
                # sym_empty but data present -> Lot Row (CW.xlsx style)
                if not current_symbol: continue
                
                found_account = None
                # Shared Column resolution
                if is_account_lot:
                    found_account = d_str
                    row_list[desc_i] = current_description # Change account name back to security name
                elif not d_str:
                    row_list[desc_i] = current_description
                
                row_list[sym_i] = current_symbol
                # Add acquisition date (none), extracted account, and security type as extra columns
                new_rows.append(row_list + [None, found_account, current_type])
        
        new_columns = list(df.columns) + ["_AcquisitionDateFromSymbol", "Account", "Security Type"]
        result = pd.DataFrame(new_rows, columns=new_columns)
        logger.debug(f"Symbol-group parser overhauled: produced {len(result)} lot row(s)")
        return result
    except Exception as exc:
        logger.debug(f"Failed to realign symbol-group table: {exc}")
        return pd.DataFrame(columns=list(df.columns))

def _is_activity_table(df: pd.DataFrame) -> bool:
    try:
        headers = [str(c).strip().lower() for c in df.columns]
        has_activity = any("investment activity" in h for h in headers)
        from .processing import _normalise_header # circular? No, processing has it.
        # But _is_activity relies on raw headers mainly.
        return bool(has_activity and any(h == "date" for h in headers))
    except Exception:
        return False

def _parse_activity_table(df: pd.DataFrame, col_map: Dict[str, str]) -> pd.DataFrame:
    try:
        if df.empty:
            return df.copy()
        result = df.copy()
        headers = [str(c).strip().lower() for c in result.columns]
        has_symbol = False
        from .processing import _normalise_header
        for h in headers:
            norm = _normalise_header(h)
            if col_map.get(norm) == "Symbol":
                has_symbol = True
                break
        if not has_symbol:
            if len(result.columns) > 2:
                # heuristic
                date_idx = -1
                for i, c in enumerate(result.columns):
                    if str(c).strip().lower() == "date":
                        date_idx = i
                        break
                if date_idx > 0:
                    desc_col = result.columns[date_idx - 1]
                    result[desc_col] = result[desc_col].ffill()
                    if pd.isna(desc_col) or "unnamed" in str(desc_col).lower() or str(desc_col).strip() == "":
                        result.rename(columns={desc_col: "Security Description"}, inplace=True)
        return result
    except Exception as exc:
        logger.debug(f"Failed to parse activity table: {exc}")
        return df

def detect_file_structure(input_path: str, config_path: str, tabs: Optional[List[str]] = None) -> str:
    try:
        logger.debug(f"Detecting file structure for {input_path}")
        if not os.path.isfile(input_path):
            return "standard"
        ext = os.path.splitext(input_path)[1].lower()
        if ext == ".csv":
            tables, _ = read_csv_file(input_path)
        elif ext in {".xls", ".xlsx"}:
            tables, _ = read_excel_file(input_path, sheets_to_process=tabs)
        else:
            return "standard"
        logger.debug(f"Found {len(tables)} table(s) in {input_path}")
        acc_idx = None
        for i, t in enumerate(tables):
            if _is_account_summary_table(t):
                acc_idx = i
                break
        if acc_idx is None:
            logger.debug("No account summary table found; treating as standard")
            return "standard"
        for j in range(acc_idx + 1, len(tables)):
            t = tables[j]
            if _is_holdings_table(t):
                preview = _parse_hybrid_holdings(t)
                if not preview.empty:
                    logger.debug("File structure classified as hybrid")
                    return "hybrid"
        logger.debug("File structure classified as standard")
        return "standard"
    except Exception:
        return "standard"
