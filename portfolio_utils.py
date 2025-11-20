"""
portfolio_utils.py

Reusable utilities for extracting and normalizing portfolio data from CSV/XLS/XLSX
files. Supports multi‑table inputs, hybrid formats (summary + lot rows), column
synonym mapping from config, key:value fallbacks, configurable output fields,
cleansing (blank/duplicate removal), audit logging, and rich per‑file reporting
(via utils.reporting_utils.FileReport when provided).
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple, Any, TYPE_CHECKING

import pandas as pd

import logging

if TYPE_CHECKING:
    from utils.reporting_utils import FileReport

# -----------------------------------------------------------------------------
# Logging setup
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

    # optional file logging
    try:
        log_dir = os.getenv("LOG_DIR", "logs")
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, "app.log"))
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass

# In‑memory audit log
_audit_log: List[Tuple[str, str]] = []


def _audit(message: str) -> None:
    if os.getenv("AUDIT", "true").lower() == "true":
        ts = _dt.datetime.now().isoformat(timespec="seconds")
        _audit_log.append((ts, message))
        logger.info(f"AUDIT: {message}")


def get_audit_log() -> List[Tuple[str, str]]:
    return list(_audit_log)


# -----------------------------------------------------------------------------
# Config + helpers
# -----------------------------------------------------------------------------

def _normalise_header(header: str) -> str:
    cleaned = re.sub(r"[\-_/]+", " ", str(header).strip().lower())
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def load_column_mapping(config_path: str) -> Dict[str, str]:
    env_path = os.getenv("COLUMN_MAPPING_CONFIG") or config_path
    mapping: Dict[str, str] = {}
    try:
        if not os.path.isfile(env_path):
            logger.warning(f"Column mapping config not found at {env_path}; using empty mapping")
            return {}
        with open(env_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        # Support both shapes: {canonical:[variants]} OR {variant:canonical}
        for k, v in raw.items():
            if isinstance(v, list):
                for variant in v:
                    mapping[_normalise_header(variant)] = k
            else:
                mapping[_normalise_header(k)] = v
        _audit(f"Loaded column mapping from {env_path} with {len(mapping)} entries")
    except Exception as exc:
        logger.error(f"Failed to load column mapping: {exc}")
    return mapping


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
        "Account",
        "Symbol",
        "Quantity",
        "Acquisition Date",
        "Cost per Unit",
        "Total Cost",
        "Value",
        "Type",
    ]


def _extract_broker_from_filename(path: str) -> Optional[str]:
    try:
        base = os.path.basename(path)
        stem, _ = os.path.splitext(base)
        return (stem.split("_")[0] or None) if stem else None
    except Exception:
        return None


__all__ = [
    "load_column_mapping",
    "read_csv_file",
    "read_excel_file",
    "find_key_value_pairs",
    "normalise_table",
    "extract_portfolio_fields",
    "cleanup_portfolio_df",
    "process_file",
    "process_hybrid_file",
    "detect_file_structure",
    "get_output_fields",
    "get_audit_log",
]


# -----------------------------------------------------------------------------
# Readers
# -----------------------------------------------------------------------------

def read_csv_file(path: str) -> Tuple[List[pd.DataFrame], List[str]]:
    tables: List[pd.DataFrame] = []
    lines: List[str] = []
    logger.debug(f"Reading CSV file from {path}")
    if not os.path.isfile(path):
        logger.error(f"CSV file not found: {path}")
        return [], []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            reader = csv.reader(fh)
            block: List[List[str]] = []
            for row in reader:
                lines.append(",".join(row))
                # blank row breaks a table
                if all((str(c).strip() == "" for c in row)):
                    if block:
                        header, rows = block[0], block[1:]
                        try:
                            hlen = len(header)
                            aligned = [(r + [""] * (hlen - len(r)))[:hlen] for r in rows]
                            tables.append(pd.DataFrame(aligned, columns=header))
                        except Exception as exc:
                            logger.debug(f"Failed to construct DataFrame from table with header {header}: {exc}")
                        block = []
                    continue
                if len(row) == 1 and not block:
                    # single-cell line outside a table
                    continue
                block.append(row)
            if block:
                header, rows = block[0], block[1:]
                try:
                    hlen = len(header)
                    aligned = [(r + [""] * (hlen - len(r)))[:hlen] for r in rows]
                    tables.append(pd.DataFrame(aligned, columns=header))
                except Exception as exc:
                    logger.debug(f"Failed to construct DataFrame from final table with header {header}: {exc}")
        _audit(f"CSV file {path} parsed into {len(tables)} table(s)")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"CSV read complete: {len(lines)} raw line(s) and {len(tables)} table(s) extracted from {path}")
        return tables, lines
    except PermissionError:
        logger.error(f"Permission denied when reading CSV file: {path}")
        return [], []
    except Exception as exc:
        logger.error(f"Error reading CSV file {path}: {exc}")
        return [], []


def read_excel_file(path: str, sheets_to_process: Optional[List[str]] = None) -> Tuple[List[pd.DataFrame], List[str]]:
    tables: List[pd.DataFrame] = []
    lines: List[str] = []
    logger.debug(f"Reading Excel file from {path}")
    if not os.path.isfile(path):
        logger.error(f"Excel file not found: {path}")
        return [], []
    try:
        xls = pd.ExcelFile(path, engine="openpyxl")
        for sheet in xls.sheet_names:
            if sheets_to_process is not None and sheet not in sheets_to_process:
                continue
            try:
                df_sheet = xls.parse(sheet_name=sheet, header=None, dtype=str)
            except Exception as exc:
                logger.warning(f"Failed to parse sheet {sheet} in {path}: {exc}")
                continue
            block: List[List[str]] = []
            for _, row in df_sheet.iterrows():
                row_list = ["" if str(v) == "nan" else str(v) for v in row.tolist()]
                lines.append(",".join(row_list))
                if all((str(c).strip() == "" or str(c).lower() == "nan" for c in row_list)):
                    if block:
                        header, rows = block[0], block[1:]
                        try:
                            hlen = len(header)
                            aligned = [(r + [""] * (hlen - len(r)))[:hlen] for r in rows]
                            tables.append(pd.DataFrame(aligned, columns=header))
                        except Exception as exc:
                            logger.debug(f"Failed to construct DataFrame from sheet {sheet} table with header {header}: {exc}")
                        block = []
                    continue
                block.append(row_list)
            if block:
                header, rows = block[0], block[1:]
                try:
                    hlen = len(header)
                    aligned = [(r + [""] * (hlen - len(r)))[:hlen] for r in rows]
                    tables.append(pd.DataFrame(aligned, columns=header))
                except Exception as exc:
                    logger.debug(f"Failed to construct DataFrame from sheet {sheet} final table with header {header}: {exc}")
        _audit(f"Excel file {path} parsed into {len(tables)} table(s) across {len(xls.sheet_names)} sheet(s)")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Excel read complete: {len(lines)} raw line(s) and {len(tables)} table(s) extracted from {path}")
        return tables, lines
    except PermissionError:
        logger.error(f"Permission denied when reading Excel file: {path}")
        return [], []
    except Exception as exc:
        logger.error(f"Error reading Excel file {path}: {exc}")
        # fallback try as CSV (mislabelled file)
        try:
            t, l = read_csv_file(path)
            if t or l:
                logger.debug(f"Excel read failed, fallback to CSV returned {len(t)} table(s)")
                return t, l
        except Exception:
            pass
        return [], []


# -----------------------------------------------------------------------------
# Parsing helpers
# -----------------------------------------------------------------------------

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


def normalise_table(df: pd.DataFrame, col_map: Dict[str, str]) -> pd.DataFrame:
    rename_dict: Dict[str, str] = {}
    try:
        for col in df.columns:
            norm = _normalise_header(str(col))
            if norm in col_map:
                rename_dict[col] = col_map[norm]
        if rename_dict:
            df = df.rename(columns=rename_dict)
            _audit(f"Renamed {len(rename_dict)} column(s) to canonical names")
    except Exception as exc:
        logger.error(f"Error normalising table columns: {exc}")
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


def cleanup_portfolio_df(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if df.empty:
            return df
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
        return cleaned
    except Exception as exc:
        logger.warning(f"Failed to clean portfolio dataframe: {exc}")
        return df


# -----------------------------------------------------------------------------
# Hybrid detection & realignment
# -----------------------------------------------------------------------------

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


def _parse_account_summary(df: pd.DataFrame, col_map: Dict[str, str]) -> Optional[str]:
    try:
        if df.empty:
            return None
        dn = normalise_table(df.copy(), col_map)
        if "Account" not in dn.columns:
            return None
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
        headers = list(df.columns)
        new_rows: List[List[Any]] = []
        current_symbol: Optional[str] = None
        date_pat = re.compile(r"\d{1,2}/\d{1,2}/\d{4}")
        for row in df.itertuples(index=False, name=None):
            row_list = list(row)
            if not row_list:
                continue
            first = "" if row_list[0] is None or (isinstance(row_list[0], float) and pd.isna(row_list[0])) else str(row_list[0]).strip()
            is_date = bool(date_pat.fullmatch(first))
            if first and not is_date:
                current_symbol = first  # aggregated row, skip
                continue
            if current_symbol is None:
                continue
            new_rows.append([current_symbol] + row_list[1:])
        if not new_rows:
            return pd.DataFrame(columns=headers)
        return pd.DataFrame(new_rows, columns=headers)
    except Exception as exc:
        logger.debug(f"Failed to realign hybrid holdings: {exc}")
        return pd.DataFrame(columns=list(df.columns))


def _is_symbol_group_table(df: pd.DataFrame, col_map: Dict[str, str]) -> bool:
    try:
        if df.empty:
            return False
        sym_i, desc_i = -1, -1
        for i, c in enumerate(df.columns):
            norm = _normalise_header(str(c))
            if norm in col_map:
                can = col_map[norm]
                if can == "Symbol":
                    sym_i = i
                elif can == "Account":
                    desc_i = i
        if sym_i < 0 or desc_i < 0:
            return False
        found_sym = False
        found_lot_after = False
        for row in df.itertuples(index=False, name=None):
            try:
                sym_val, desc_val = row[sym_i], row[desc_i]
            except Exception:
                continue
            sym_empty = (sym_val is None) or (pd.isna(sym_val)) or (str(sym_val).strip().lower() in ("", "nan"))
            desc_empty = (desc_val is None) or (pd.isna(desc_val)) or (str(desc_val).strip().lower() in ("", "nan"))
            if not sym_empty:
                found_sym = True
                continue
            if found_sym and not desc_empty:
                found_lot_after = True
                break
        return bool(found_sym and found_lot_after)
    except Exception:
        return False


def _parse_symbol_group_table(df: pd.DataFrame, col_map: Dict[str, str]) -> pd.DataFrame:
    try:
        if df.empty:
            return df.copy()
        sym_i, desc_i = -1, -1
        for i, c in enumerate(df.columns):
            norm = _normalise_header(str(c))
            if norm in col_map:
                can = col_map[norm]
                if can == "Symbol":
                    sym_i = i
                elif can == "Account":
                    desc_i = i
        if sym_i < 0 or desc_i < 0:
            return pd.DataFrame(columns=df.columns)
        category_heads = {"total", "cash", "cash equivalent", "mutual fund"}
        new_rows: List[List[Any]] = []
        current_symbol: Optional[str] = None
        for row in df.itertuples(index=False, name=None):
            row_list = list(row)
            if not any(str(c).strip() for c in row_list if not (pd.isna(c) or (isinstance(c, str) and c.lower() == "nan"))):
                continue
            raw_sym, raw_desc = row_list[sym_i], row_list[desc_i]
            sym_empty = (raw_sym is None) or (pd.isna(raw_sym)) or (str(raw_sym).strip().lower() in ("", "nan"))
            desc_empty = (raw_desc is None) or (pd.isna(raw_desc)) or (str(raw_desc).strip().lower() in ("", "nan"))
            desc_str = "" if desc_empty else str(raw_desc).strip()
            if sym_empty and desc_str.lower() in category_heads:
                continue
            if not sym_empty:
                current_symbol = str(raw_sym).strip()
                continue
            if not current_symbol or desc_empty:
                continue
            row_list[sym_i] = current_symbol
            new_rows.append(row_list)
        if not new_rows:
            return pd.DataFrame(columns=df.columns)
        result = pd.DataFrame(new_rows, columns=df.columns)
        if logger.isEnabledFor(logging.DEBUG):
            try:
                logger.debug(
                    f"Symbol-group parser produced {len(result)} lot row(s) with columns {list(result.columns)}\n{result.head(2).to_string(index=False)}"
                )
            except Exception:
                pass
        return result
    except Exception as exc:
        logger.debug(f"Failed to realign symbol-group table: {exc}")
        return pd.DataFrame(columns=list(df.columns))


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


# -----------------------------------------------------------------------------
# Structure detection
# -----------------------------------------------------------------------------

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
                    logger.debug(
                        f"File structure classified as hybrid (account summary at table {acc_idx+1}, holdings at table {j+1})"
                    )
                    return "hybrid"
        logger.debug("File structure classified as standard")
        return "standard"
    except Exception:
        return "standard"


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
    logger.debug(f"Processing file {input_path} with {len(col_map)} column mappings")

    if ext == ".csv":
        tables, lines = read_csv_file(input_path)
    else:
        tables, lines = read_excel_file(input_path, sheets_to_process=tabs)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Process_file: loaded {len(tables)} table(s) and {len(lines)} raw line(s) from {input_path}")
        for i, t in enumerate(tables, start=1):
            try:
                if not t.empty:
                    logger.debug(f"Process_file: Table #{i} sample:\n{t.head(2).to_string(index=False)}")
            except Exception:
                pass
        if lines:
            logger.debug(f"Process_file: Sample non-tabular lines: {lines[:2] if len(lines)>=2 else lines}")

    # Reporting — table inventory
    if file_report is not None:
        try:
            file_report.tables_detected = len(tables)
            file_report.rows_read_raw = sum(len(t) for t in tables if hasattr(t, "__len__"))
            orig_cols = set()
            for t in tables:
                try:
                    orig_cols |= set(map(str, t.columns))
                except Exception:
                    pass
            file_report.original_columns = orig_cols
            if logger.isEnabledFor(logging.DEBUG):
                for idx, t in enumerate(tables, start=1):
                    file_report.add_table_sample(idx, t)
        except Exception:
            pass

    key_values = find_key_value_pairs(lines, col_map)

    required_fields = get_output_fields([
        "Account",
        "Symbol",
        "Quantity",
        "Acquisition Date",
        "Cost per Unit",
        "Total Cost",
        "Value",
        "Type",
    ])

    if "Broker" in required_fields:
        b = _extract_broker_from_filename(input_path)
        if b:
            key_values.setdefault("Broker", b)

    extracted: List[pd.DataFrame] = []
    # aggregate rename map for reporting
    col_map_used: Dict[str, str] = {}

    for tbl in tables:
        try:
            if tbl.empty:
                continue
            orig_cols = list(tbl.columns)

            # auto‑detect symbol‑group style and realign if needed
            tbl_to_use = tbl
            try:
                if _is_symbol_group_table(tbl, col_map):
                    p = _parse_symbol_group_table(tbl, col_map)
                    if not p.empty:
                        tbl_to_use = p
            except Exception:
                pass

            norm = normalise_table(tbl_to_use, col_map)

            if file_report is not None:
                try:
                    # remember normalized column universe
                    file_report.normalized_columns |= set(map(str, norm.columns))
                    # infer a best‑effort original→canonical rename map
                    inferred = _build_rename_map(orig_cols, norm)
                    col_map_used.update(inferred)
                except Exception:
                    pass

            present = {f for f in required_fields if f in norm.columns}
            if "Symbol" not in present or len(present) < 2:
                # guard against misc or filter tables
                continue

            part = extract_portfolio_fields(norm, required_fields, key_values).dropna(how="all")
            if not part.empty:
                extracted.append(part)
        except Exception as exc:
            logger.warning(f"Failed to process a table: {exc}")

    if extracted:
        combined = pd.concat(extracted, ignore_index=True)
    else:
        combined = pd.DataFrame({f: [key_values.get(f)] for f in required_fields}) if key_values else pd.DataFrame(columns=required_fields)

    cleaned = cleanup_portfolio_df(combined)

    # finalize reporting
    if file_report is not None:
        try:
            file_report.rows_after_cleanup = len(combined)
            file_report.total_rows_out = len(cleaned)
            removed_blank = combined.shape[0] - combined.dropna(how="all").shape[0]
            removed_dupes = combined.dropna(how="all").shape[0] - cleaned.shape[0]
            file_report.rows_removed_blank = max(0, removed_blank)
            file_report.rows_removed_duplicates = max(0, removed_dupes)
            file_report.finalize_columns(required_fields, col_map_used)
            file_report.log_summary()
        except Exception:
            pass

    # output
    output_dir = os.getenv("OUTPUT_DIR", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if timestamp is None:
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"portfolio-{timestamp}.csv")
    try:
        if append and os.path.isfile(out_path):
            try:
                existing = pd.read_csv(out_path)
                cleaned = cleanup_portfolio_df(pd.concat([existing, cleaned], ignore_index=True))
            except Exception as exc:
                logger.warning(f"Failed to read existing output for appending: {exc}")
        cleaned.to_csv(out_path, index=False)
        _audit(f"Wrote output CSV to {out_path}")
    except Exception as exc:
        logger.error(f"Failed to write output CSV: {exc}")
        raise
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
    if not os.path.isfile(input_path):
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(input_path)
    ext = os.path.splitext(input_path)[1].lower()
    if ext not in {".csv", ".xls", ".xlsx"}:
        raise ValueError(f"Unsupported file extension: {ext}")

    col_map = load_column_mapping(config_path)

    if ext == ".csv":
        tables, lines = read_csv_file(input_path)
    else:
        tables, lines = read_excel_file(input_path, sheets_to_process=tabs)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Hybrid processing: loaded {len(tables)} table(s) and {len(lines)} raw line(s) from {input_path}")
        for i, t in enumerate(tables, start=1):
            try:
                if not t.empty:
                    logger.debug(f"Hybrid processing: Table #{i} sample before classification:\n{t.head(2).to_string(index=False)}")
            except Exception:
                pass
        if lines:
            logger.debug(f"Hybrid processing: Sample non-tabular lines: {lines[:2] if len(lines)>=2 else lines}")

    # reporting: table inventory
    if file_report is not None:
        try:
            file_report.tables_detected = len(tables)
            file_report.rows_read_raw = sum(len(t) for t in tables if hasattr(t, "__len__"))
            orig_cols = set()
            for t in tables:
                try:
                    orig_cols |= set(map(str, t.columns))
                except Exception:
                    pass
            file_report.original_columns = orig_cols
            if logger.isEnabledFor(logging.DEBUG):
                for idx, t in enumerate(tables, start=1):
                    file_report.add_table_sample(idx, t)
        except Exception:
            pass

    # classify + realign
    account_value: Optional[str] = None
    holdings: List[pd.DataFrame] = []

    for idx, tbl in enumerate(tables):
        try:
            if tbl.empty:
                continue
            logger.debug(f"Hybrid processing: examining table #{idx+1} with columns {list(tbl.columns)} and {len(tbl)} row(s)")
            if _is_account_summary_table(tbl) and account_value is None:
                account_value = _parse_account_summary(tbl, col_map)
                logger.debug(f"Hybrid processing: identified account summary (Account={account_value})")
                continue
            if _is_holdings_table(tbl):
                try:
                    if _is_symbol_group_table(tbl, col_map):
                        realigned = _parse_symbol_group_table(tbl, col_map)
                    else:
                        realigned = _parse_hybrid_holdings(tbl)
                except Exception as exc:
                    logger.debug(f"Hybrid processing: failed to realign table #{idx+1}: {exc}")
                    realigned = pd.DataFrame(columns=tbl.columns)
                if not realigned.empty:
                    holdings.append(realigned)
        except Exception as exc:
            logger.debug(f"Failed to classify table: {exc}")

    holdings_norm = pd.DataFrame()
    if holdings:
        merged = pd.concat(holdings, ignore_index=True)
        holdings_norm = normalise_table(merged, col_map)
        if file_report is not None:
            try:
                file_report.normalized_columns |= set(map(str, holdings_norm.columns))
            except Exception:
                pass

    key_values = find_key_value_pairs(lines, col_map)
    if account_value:
        key_values.setdefault("Account", account_value)

    required_fields = get_output_fields([
        "Account",
        "Symbol",
        "Quantity",
        "Acquisition Date",
        "Cost per Unit",
        "Total Cost",
        "Value",
        "Type",
    ])

    if "Broker" in required_fields:
        b = _extract_broker_from_filename(input_path)
        if b:
            key_values.setdefault("Broker", b)

    extracted: List[pd.DataFrame] = []
    if not holdings_norm.empty:
        try:
            part = extract_portfolio_fields(holdings_norm, required_fields, key_values).dropna(how="all")
            if not part.empty:
                extracted.append(part)
        except Exception as exc:
            logger.warning(f"Failed to extract portfolio fields from hybrid holdings: {exc}")

    combined = pd.concat(extracted, ignore_index=True) if extracted else (
        pd.DataFrame({f: [key_values.get(f)] for f in required_fields}) if key_values else pd.DataFrame(columns=required_fields)
    )

    cleaned = cleanup_portfolio_df(combined)

    # finalize reporting
    if file_report is not None:
        try:
            file_report.rows_after_cleanup = len(combined)
            file_report.total_rows_out = len(cleaned)
            removed_blank = combined.shape[0] - combined.dropna(how="all").shape[0]
            removed_dupes = combined.dropna(how="all").shape[0] - cleaned.shape[0]
            file_report.rows_removed_blank = max(0, removed_blank)
            file_report.rows_removed_duplicates = max(0, removed_dupes)
            file_report.finalize_columns(required_fields, col_map_used={})
            file_report.log_summary()
        except Exception:
            pass

    # output
    output_dir = os.getenv("OUTPUT_DIR", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    if timestamp is None:
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(output_dir, f"portfolio-{timestamp}.csv")
    try:
        if append and os.path.isfile(out_path):
            try:
                existing = pd.read_csv(out_path)
                cleaned = cleanup_portfolio_df(pd.concat([existing, cleaned], ignore_index=True))
            except Exception as exc:
                logger.warning(f"Failed to read existing output file for appending: {exc}")
        cleaned.to_csv(out_path, index=False)
        _audit(f"Wrote hybrid output CSV to {out_path}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Hybrid processing: final output dataframe has {len(cleaned)} row(s) and columns {list(cleaned.columns)}")
    except Exception as exc:
        logger.error(f"Failed to write output CSV: {exc}")
        raise
    return out_path


# CLI debug helper
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract portfolio information from CSV/XLS/XLSX files.")
    parser.add_argument("input_path", help="Path to the input CSV or Excel file")
    parser.add_argument("config_path", help="Path to column mapping configuration JSON")
    parser.add_argument("--outdir", default="./out", help="Directory to write the output CSV")
    parser.add_argument("--timestamp", default=None, help="Timestamp to include in output filename (default: now)")
    args = parser.parse_args()
    output_file = process_file(args.input_path, args.config_path, args.outdir, args.timestamp)
    print(f"Output written to {output_file}")
