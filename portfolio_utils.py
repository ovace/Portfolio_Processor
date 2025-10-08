"""
portfolio_utils.py

This module provides a set of reusable functions for extracting portfolio
information from a variety of spreadsheet formats (CSV, XLS, XLSX).  The
primary goal of these utilities is to locate one or more tables within a
file, normalise the column names using a configurable mapping, and then
produce a canonical dataframe containing portfolio‐related fields.  When
columns are missing from the tabular data the utilities will also scan
the surrounding file for simple key:value pairs in order to fill in
missing values.  Finally, the consolidated dataframe can be written to
disk using a timestamped filename in a designated output directory.

The design here follows a modular, functional approach with clear
separation of concerns.  Each function does one thing well and can be
tested in isolation.  Configuration is externalised into a JSON file
(see ``config/column_mapping.json``) so that new synonyms can be added
without touching the code.  Where possible, the implementation makes
conservative assumptions about the input file structure: tables are
detected by looking for contiguous blocks of non‐empty rows and key
value pairs are extracted from lines or cells that contain a colon
separator.
"""

###############################################################################
# Metadata
#
# @file        portfolio_utils.py
# @brief       Utilities for portfolio extraction and normalisation
# @author      IRS Assistant
# @created     2025-10-07
# @modified    2025-10-07
#
# This module is part of the Python Assistant project.  It implements
# reusable functions adhering to Clean Architecture principles for
# detecting and extracting portfolio data from CSV and Excel files.  The
# utilities support configuration via environment variables and external
# JSON files, provide multi-level logging and auditing, and are designed
# to fail gracefully while cleaning up resources.  No configuration is
# hardcoded; see ``config/column_mapping.json`` and environment
# variables for customisation.
###############################################################################

from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

# Standard library logging for diagnostics.  Logging behaviour is
# configured based on the DEBUG environment variable.  Audit events are
# recorded in a global list for later inspection.
import logging
from logging.handlers import RotatingFileHandler

DEBUG_ENV = os.getenv("DEBUG", "false").lower() == "true"

# Configure the module-level logger only once.  If handlers are already
# attached (for example, when reloading in an interactive session), this
# setup is skipped to avoid duplicate log entries.
logger = logging.getLogger(__name__)
if not logger.handlers:
    _level = logging.DEBUG if DEBUG_ENV else logging.INFO
    # Console handler
    console_handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(_level)

    # Setup file logging with rotation.  Logs are written to the
    # directory specified by the LOG_DIR environment variable (default
    # "logs").  On each import (fresh run), if a log file already
    # exists it is archived with a timestamp, ensuring clean logs
    # without manual cleanup.
    def _setup_file_logging() -> None:
        log_dir = os.getenv("LOG_DIR", "logs")
        try:
            os.makedirs(log_dir, exist_ok=True)
        except Exception:
            # If directory creation fails, skip file logging
            return
        log_file = os.path.join(log_dir, "app.log")
        try:
            if os.path.exists(log_file):
                ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                archive_name = os.path.join(log_dir, f"app_{ts}.log")
                os.rename(log_file, archive_name)
        except Exception:
            # Ignore rotation errors
            pass
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(_formatter)
            logger.addHandler(file_handler)
        except Exception:
            # Ignore file handler errors
            pass

    _setup_file_logging()

# Global audit log capturing events as tuples of (timestamp, message).
_audit_log: List[Tuple[str, str]] = []


def _audit(message: str) -> None:
    """Record an audit event with the current timestamp.

    This helper appends messages to the global audit log and also
    forwards the message to the logger at INFO level.  Auditing can be
    disabled by setting the environment variable ``AUDIT`` to ``false``.

    Args:
        message: Human-readable description of the event to record.
    """
    if os.getenv("AUDIT", "true").lower() == "true":
        timestamp = _dt.datetime.now().isoformat(timespec="seconds")
        _audit_log.append((timestamp, message))
        logger.info(f"AUDIT: {message}")


def get_audit_log() -> List[Tuple[str, str]]:
    """Return a copy of the audit log.

    Returns:
        A list of (timestamp, message) tuples representing recorded events.
    """
    return list(_audit_log)

def _extract_broker_from_filename(path: str) -> Optional[str]:
    """Extract the broker identifier from a portfolio filename.

    A broker identifier is assumed to be the portion of the file name
    before the first underscore.  For example, given ``ML_10022025.csv``
    the broker would be ``ML``.  If the file name does not contain an
    underscore, the entire base name (sans extension) is returned.

    Args:
        path: The full path to the portfolio file.

    Returns:
        The broker string, or ``None`` if it cannot be determined.
    """
    try:
        base = os.path.basename(path)
        name, _ = os.path.splitext(base)
        if not name:
            return None
        parts = name.split("_")
        return parts[0] if parts else None
    except Exception:
        return None

# -------------------------------------------------------------------------
# Configuration helpers
#
def get_output_fields(default: Optional[List[str]] | None = None) -> List[str]:
    """Retrieve the list of canonical fields to include in the output.

    The order and presence of these fields determine the columns of the
    extracted portfolio dataframe.  The list may be overridden via the
    environment variable ``OUTPUT_FIELDS``.  The value of this
    environment variable can be specified as a comma-separated string or
    as a JSON list (e.g. ``["Account", "Symbol", "Quantity"]``).

    Args:
        default: Optional fallback list of canonical field names.  If
            omitted and no override is found, the built-in default list
            is used.

    Returns:
        A list of canonical field names to extract and output.
    """
    fields_env = os.getenv("OUTPUT_FIELDS")
    # If provided as a JSON list, attempt to parse it
    if fields_env:
        fields_str = fields_env.strip()
        if fields_str.startswith("[") and fields_str.endswith("]"):
            try:
                parsed = json.loads(fields_str)
                if isinstance(parsed, list):
                    return [str(f).strip() for f in parsed if str(f).strip()]
            except Exception:
                pass
        # Otherwise treat as comma-separated values
        try:
            return [f.strip() for f in fields_str.split(",") if f.strip()]
        except Exception:
            pass
    # No override found, return the provided default or the built-in list
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

# Define what gets exported when ``from portfolio_utils import *`` is used.
# Only include the public, reusable functions; internal helpers (prefixed
# with an underscore) are intentionally left out.  This explicit list
# makes it clear to consumers of the module which functions form the
# supported API surface.
__all__ = [
    "load_column_mapping",
    "read_csv_file",
    "read_excel_file",
    "find_key_value_pairs",
    "normalise_table",
    "extract_portfolio_fields",
    "process_file",
    "get_audit_log",
    # New exports for hybrid/structured file processing
    "detect_file_structure",
    "process_hybrid_file",
    "get_output_fields",
    "cleanup_portfolio_df",
]


def load_column_mapping(config_path: str) -> Dict[str, str]:
    """Load a mapping of column name variants to canonical names.

    The configuration file is expected to be JSON with a single top‐level
    dictionary mapping each variant (case‐insensitive) to its canonical
    field name.  Variants that differ only by whitespace, punctuation or
    case should each be represented explicitly in the config.  The
    function returns a dictionary keyed by the lowercased variant name
    with whitespace stripped, so that lookups can be performed in a
    normalised way.

    The location of the configuration file may be overridden via the
    ``COLUMN_MAPPING_CONFIG`` environment variable.  If the file is
    missing or cannot be parsed, an empty mapping is returned and a
    warning is logged.

    Args:
        config_path: Default path to the JSON configuration file.

    Returns:
        A dictionary mapping normalised variant names to canonical names.
    """
    # Allow environment to override the provided path.
    env_path = os.getenv("COLUMN_MAPPING_CONFIG")
    if env_path:
        config_path = env_path
    try:
        if not os.path.isfile(config_path):
            logger.warning(f"Column mapping config not found at {config_path}; returning empty mapping")
            return {}
        with open(config_path, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
        mapping: Dict[str, str] = {}
        # Support two structures: either {canonical: [variants...]} or {variant: canonical}
        for key, value in raw_map.items():
            if isinstance(value, list):
                # key is canonical name, value is list of synonyms
                canonical_name = key
                for variant in value:
                    normalised = _normalise_header(variant)
                    mapping[normalised] = canonical_name
            else:
                # key is variant, value is canonical name
                normalised = _normalise_header(key)
                mapping[normalised] = value
        _audit(f"Loaded column mapping from {config_path} with {len(mapping)} entries")
        return mapping
    except Exception as exc:
        logger.error(f"Failed to load column mapping: {exc}")
        return {}


def _normalise_header(header: str) -> str:
    """Normalise a header string for consistent comparison.

    This helper lowercases the input string, strips leading/trailing
    whitespace, removes common punctuation (underscores, hyphens, slashes)
    and collapses consecutive spaces.  The goal is to reduce the number
    of variations that need to be listed in the configuration file.

    Args:
        header: Original header string from a table or key.

    Returns:
        A normalised representation of the header suitable for lookup.
    """
    # Replace separators with spaces
    cleaned = re.sub(r"[\-_/]+", " ", header.strip().lower())
    # Collapse multiple spaces
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def read_csv_file(path: str) -> Tuple[List[pd.DataFrame], List[str]]:
    """Read a CSV file and return a list of dataframes representing tables.

    The CSV may contain multiple tables separated by completely blank
    rows.  This function preserves blank lines and splits the file into
    contiguous blocks of rows with data.  Each block becomes a separate
    pandas DataFrame with the first row used as column headers.

    Additionally, all raw lines of the file are returned as a list of
    strings so that key:value pairs outside of tables can be scanned.

    Args:
        path: Path to the CSV file.

    Returns:
        A tuple ``(tables, lines)`` where ``tables`` is a list of
        DataFrames, and ``lines`` are all lines from the file for
        subsequent key:value parsing.
    """
    tables: List[pd.DataFrame] = []
    lines: List[str] = []
    logger.debug(f"Reading CSV file from {path}")
    # Check file existence before attempting to open
    if not os.path.isfile(path):
        logger.error(f"CSV file not found: {path}")
        return [], []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f)
            current_table_rows: List[List[str]] = []
            for row in reader:
                # Preserve original row content for key:value scanning
                lines.append(",".join(row))
                if all((str(cell).strip() == "" for cell in row)):
                    if current_table_rows:
                        # Attempt to build a DataFrame from the accumulated rows
                        header = current_table_rows[0]
                        rows = current_table_rows[1:]
                        try:
                            # Align row lengths to header length by padding or truncating
                            header_len = len(header)
                            aligned_rows: List[List[str]] = []
                            for r in rows:
                                if len(r) < header_len:
                                    aligned_rows.append(r + ["" for _ in range(header_len - len(r))])
                                elif len(r) > header_len:
                                    aligned_rows.append(r[:header_len])
                                else:
                                    aligned_rows.append(r)
                            df = pd.DataFrame(aligned_rows, columns=header)
                            tables.append(df)
                        except Exception as exc:
                            logger.debug(
                                f"Failed to construct DataFrame from table with header {header}: {exc}"
                            )
                        current_table_rows = []
                    continue
                # If the row has only a single column and we are not currently building a table,
                # treat it as non-tabular data (e.g. a section header) and do not start a table.
                if len(row) == 1 and not current_table_rows:
                    continue
                current_table_rows.append(row)
            if current_table_rows:
                header = current_table_rows[0]
                rows = current_table_rows[1:]
                try:
                    header_len = len(header)
                    aligned_rows: List[List[str]] = []
                    for r in rows:
                        if len(r) < header_len:
                            aligned_rows.append(r + ["" for _ in range(header_len - len(r))])
                        elif len(r) > header_len:
                            aligned_rows.append(r[:header_len])
                        else:
                            aligned_rows.append(r)
                    df = pd.DataFrame(aligned_rows, columns=header)
                    tables.append(df)
                except Exception as exc:
                    logger.debug(
                        f"Failed to construct DataFrame from final table with header {header}: {exc}"
                    )
        _audit(f"CSV file {path} parsed into {len(tables)} table(s)")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"CSV read complete: {len(lines)} raw line(s) and {len(tables)} table(s) extracted from {path}"
            )
        return tables, lines
    except PermissionError:
        logger.error(f"Permission denied when reading CSV file: {path}")
        return [], []
    except Exception as exc:
        logger.error(f"Error reading CSV file {path}: {exc}")
        return [], []


def read_excel_file(path: str, sheets_to_process: Optional[List[str]] = None) -> Tuple[List[pd.DataFrame], List[str]]:
    """Read an Excel file (xls or xlsx) into a list of tables.

    Each sheet in the workbook is inspected.  Tables within a sheet are
    determined by splitting on completely blank rows, similar to the CSV
    reader.  Cells outside of tables are also scanned for key:value
    patterns.  The raw lines are reconstructed from the cell values for
    convenience.

    Args:
        path: Path to the Excel file.

    Args:
        path: Path to the Excel file.
        sheets_to_process: Optional list of sheet names to process.  If
            provided, only tables from these sheets will be extracted.
            Nonexistent sheet names are ignored without error.

    Returns:
        A tuple ``(tables, lines)`` where ``tables`` is a list of
        DataFrames extracted from the specified sheets (or all sheets if
        ``sheets_to_process`` is None), and ``lines`` is a list of
        strings representing raw content for key:value scanning.
    """
    tables: List[pd.DataFrame] = []
    lines: List[str] = []
    logger.debug(f"Reading Excel file from {path}")
    # Check file existence before attempting to open
    if not os.path.isfile(path):
        logger.error(f"Excel file not found: {path}")
        return [], []
    try:
        xls = pd.ExcelFile(path, engine="openpyxl")
        for sheet_name in xls.sheet_names:
            # If a list of specific sheets was provided, skip any sheets not in it
            if sheets_to_process is not None and sheet_name not in sheets_to_process:
                continue
            try:
                df_sheet = xls.parse(sheet_name=sheet_name, header=None, dtype=str)
            except Exception as exc:
                logger.warning(f"Failed to parse sheet {sheet_name} in {path}: {exc}")
                continue
            current_table_rows: List[List[str]] = []
            for _, row in df_sheet.iterrows():
                row_list = [str(cell) if cell != "nan" else "" for cell in row.tolist()]
                lines.append(",".join(row_list))
                if all((str(cell).strip() == "" or str(cell).lower() == "nan" for cell in row_list)):
                    if current_table_rows:
                        header = current_table_rows[0]
                        data_rows = current_table_rows[1:] if len(current_table_rows) > 1 else []
                        try:
                            header_len = len(header)
                            aligned_rows: List[List[str]] = []
                            for r in data_rows:
                                if len(r) < header_len:
                                    aligned_rows.append(r + ["" for _ in range(header_len - len(r))])
                                elif len(r) > header_len:
                                    aligned_rows.append(r[:header_len])
                                else:
                                    aligned_rows.append(r)
                            df = pd.DataFrame(aligned_rows, columns=header)
                            tables.append(df)
                        except Exception as exc:
                            logger.debug(
                                f"Failed to construct DataFrame from sheet {sheet_name} table with header {header}: {exc}"
                            )
                        current_table_rows = []
                    continue
                current_table_rows.append(row_list)
            if current_table_rows:
                header = current_table_rows[0]
                data_rows = current_table_rows[1:] if len(current_table_rows) > 1 else []
                try:
                    header_len = len(header)
                    aligned_rows: List[List[str]] = []
                    for r in data_rows:
                        if len(r) < header_len:
                            aligned_rows.append(r + ["" for _ in range(header_len - len(r))])
                        elif len(r) > header_len:
                            aligned_rows.append(r[:header_len])
                        else:
                            aligned_rows.append(r)
                    df = pd.DataFrame(aligned_rows, columns=header)
                    tables.append(df)
                except Exception as exc:
                    logger.debug(
                        f"Failed to construct DataFrame from sheet {sheet_name} final table with header {header}: {exc}"
                    )
        _audit(
            f"Excel file {path} parsed into {len(tables)} table(s) across {len(xls.sheet_names)} sheet(s)"
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Excel read complete: {len(lines)} raw line(s) and {len(tables)} table(s) extracted from {path}"
            )
        return tables, lines
    except PermissionError:
        logger.error(f"Permission denied when reading Excel file: {path}")
        return [], []
    except Exception as exc:
        # If Excel reading fails (e.g. invalid workbook), attempt to
        # interpret the file as a CSV.  This fallback allows processing
        # of mislabelled files with .xls/xlsx extensions.  Only perform
        # fallback if the extension indicates an Excel file.
        logger.error(f"Error reading Excel file {path}: {exc}")
        try:
            # Use the CSV reader to attempt to parse the file.  This may
            # succeed if the file is actually CSV-formatted text.
            tables, lines = read_csv_file(path)
            if tables or lines:
                logger.debug(f"Excel read failed, fallback to CSV returned {len(tables)} table(s)")
                return tables, lines
        except Exception:
            pass
        # If fallback fails, return empty results
        return [], []


def find_key_value_pairs(lines: Iterable[str], col_map: Dict[str, str]) -> Dict[str, str]:
    """Scan raw lines for key:value pairs and map them to canonical fields.

    This utility examines each line for either colon‐separated or comma‐
    separated key/value pairs.  For each pair found it attempts to
    normalise the key using the same rules as table headers and then
    consults the provided ``col_map`` to determine whether the key is a
    known portfolio field.  When a match is found, the corresponding
    value is stored.  If multiple occurrences of the same field appear
    outside of tables, the last occurrence wins.

    Args:
        lines: Iterable of raw string lines (from CSV or Excel) to scan.
        col_map: Mapping of normalised variant names to canonical names.

    Returns:
        A dictionary mapping canonical field names to their extracted
        values from the non‐tabular part of the file.
    """
    result: Dict[str, str] = {}
    for line in lines:
        try:
            segments = re.split(r"[;,]", line)
            for seg in segments:
                if ":" in seg:
                    key_part, value_part = seg.split(":", 1)
                    key = _normalise_header(key_part)
                    value = value_part.strip()
                    if key in col_map and value:
                        canonical_name = col_map[key]
                        result[canonical_name] = value
                elif "=" in seg:
                    key_part, value_part = seg.split("=", 1)
                    key = _normalise_header(key_part)
                    value = value_part.strip()
                    if key in col_map and value:
                        canonical_name = col_map[key]
                        result[canonical_name] = value
        except Exception as exc:
            logger.debug(f"Failed to parse line '{line}': {exc}")
            continue
    if result:
        _audit(f"Extracted {len(result)} key-value pairs from non-tabular data")
    return result


def normalise_table(df: pd.DataFrame, col_map: Dict[str, str]) -> pd.DataFrame:
    """Normalise a dataframe's column names using the provided mapping.

    The columns of ``df`` will be renamed in place to match the canonical
    names specified in ``col_map``.  Columns whose names do not match
    anything in the mapping will be left untouched.  To facilitate
    matching, each column name is normalised (lowercased and stripped of
    punctuation) before lookup.  If multiple columns normalise to the
    same canonical name, the last column encountered takes precedence.

    Args:
        df: Input dataframe to rename.
        col_map: Mapping of normalised variant names to canonical names.

    Returns:
        The dataframe with columns renamed according to the mapping.
    """
    rename_dict: Dict[str, str] = {}
    try:
        for col in df.columns:
            norm = _normalise_header(str(col))
            if norm in col_map:
                canonical = col_map[norm]
                rename_dict[col] = canonical
        if rename_dict:
            df = df.rename(columns=rename_dict)
            _audit(f"Renamed {len(rename_dict)} column(s) to canonical names")
        return df
    except Exception as exc:
        logger.error(f"Error normalising table columns: {exc}")
        return df


def extract_portfolio_fields(
    df: pd.DataFrame,
    required_fields: Iterable[str],
    key_values: Dict[str, str],
) -> pd.DataFrame:
    """Extract the required portfolio fields from a dataframe.

    This function selects only the columns listed in ``required_fields``.
    If a particular field is missing from the table it attempts to
    populate it with a default value from the supplied ``key_values``
    dictionary.  Columns not present in the dataframe and not found in
    ``key_values`` will be created with ``None`` values.

    Args:
        df: A normalised pandas DataFrame whose column names have been
            unified to canonical names.
        required_fields: An iterable of canonical field names to extract.
        key_values: A mapping of canonical field names to fallback
            values extracted from key:value pairs outside of tables.

    Returns:
        A new DataFrame containing exactly the ``required_fields`` in
        order.
    """
    data: Dict[str, List[Optional[str]]] = {}
    try:
        for field in required_fields:
            if field in df.columns:
                # Handle the case where multiple columns normalise to the
                # same canonical name.  pandas returns a DataFrame if
                # duplicate column names exist.  Take the first column in
                # such cases.
                series = df[field]
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                data[field] = series.tolist()
            else:
                # Use fallback value (from key-value pairs) if present,
                # otherwise fill with None.
                fallback = key_values.get(field)
                data[field] = [fallback] * len(df)
        _audit(f"Extracted portfolio fields for {len(df)} row(s)")
        return pd.DataFrame(data)
    except Exception as exc:
        logger.error(f"Failed to extract portfolio fields: {exc}")
        # Return empty DataFrame on failure
        return pd.DataFrame(columns=list(required_fields))


def cleanup_portfolio_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate and completely blank rows from a portfolio DataFrame.

    This helper performs two cleanup operations on the consolidated
    portfolio data prior to writing it to disk:

    * Empty string values (including whitespace-only strings) are
      converted to ``None`` so that they can be detected as missing
      values.
    * Rows in which all fields are missing (``None`` or ``NaN``) are
      removed entirely.
    * Duplicate rows (rows with identical values across all columns)
      are dropped.  This deduplication is useful when the same input
      file is processed multiple times or when multiple reports
      contain overlapping records.

    The cleanup preserves the order of the first occurrence of each
    unique row and makes no assumptions about the index of the
    incoming DataFrame.  If the input is empty or no cleanup is
    required, the original DataFrame is returned unchanged.

    Args:
        df: The DataFrame to clean.

    Returns:
        A new DataFrame with blank and duplicate rows removed.
    """
    try:
        if df.empty:
            return df
        cleaned = df.copy()
        # Replace empty or whitespace-only strings with None to allow dropna
        try:
            cleaned = cleaned.replace(r"^\s*$", None, regex=True)
        except Exception:
            pass
        # Drop rows where all columns are missing
        cleaned_before = len(cleaned)
        cleaned = cleaned.dropna(how="all")
        dropped_blank = cleaned_before - len(cleaned)
        # Drop duplicate rows
        dedup_before = len(cleaned)
        cleaned = cleaned.drop_duplicates(keep="first")
        dropped_dupes = dedup_before - len(cleaned)
        if dropped_blank or dropped_dupes:
            _audit(
                f"Cleaned portfolio dataframe: removed {dropped_blank} blank row(s) and {dropped_dupes} duplicate row(s)"
            )
        return cleaned
    except Exception as exc:
        logger.warning(f"Failed to clean portfolio dataframe: {exc}")
        return df


def process_file(
    input_path: str,
    config_path: str,
    output_dir: str = "./out",
    timestamp: Optional[str] = None,
    *,
    tabs: Optional[List[str]] = None,
    append: bool = False,
) -> str:
    """Main entry point for processing a portfolio file.

    This orchestrator determines the file type, loads the column mapping,
    reads all tables, normalises them, extracts the desired fields and
    writes the consolidated dataframe to a CSV file.  If the output
    directory does not exist it will be created.  The output filename
    includes a timestamp (either supplied or generated at runtime).

    Args:
        input_path: Path to the CSV or Excel file to process.
        config_path: Path to the JSON mapping of column names.
        output_dir: Directory where the resulting CSV should be written.
        timestamp: Optional timestamp string.  If omitted the current
            local time is used in YYYYMMDD_HHMMSS format.

    Keyword Args:
        append: If True and an output file with the chosen timestamp already
            exists, new portfolio rows will be appended to it instead of
            overwriting.  This is useful when processing multiple input
            files in sequence.

    Returns:
        The full path to the written CSV file.
    """
    # Validate input file path
    if not os.path.isfile(input_path):
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Input file not found: {input_path}")
    else:
        logger.debug(f"Input file is available: {input_path}")
    # Determine file extension and validate
    ext = os.path.splitext(input_path)[1].lower()
    if ext not in {".csv", ".xls", ".xlsx"}:
        logger.error(f"Unsupported file extension: {ext}")
        raise ValueError(f"Unsupported file extension: {ext}")
    # Resolve configuration path from environment if provided
    col_map = load_column_mapping(config_path)
    logger.debug(f"Processing file {input_path} with {len(col_map)} column mappings")
    # Read file contents into tables and raw lines.  When reading an
    # Excel workbook and a list of tabs has been provided, only those
    # sheets will be processed.
    if ext == ".csv":
        tables, lines = read_csv_file(input_path)
    else:
        tables, lines = read_excel_file(input_path, sheets_to_process=tabs)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"Process_file: loaded {len(tables)} table(s) and {len(lines)} raw line(s) from {input_path}"
        )
        # Output a sample of each table for inspection
        for idx, tbl in enumerate(tables):
            try:
                if not tbl.empty:
                    sample = tbl.head(2).to_string(index=False)
                    logger.debug(f"Process_file: Table #{idx+1} sample:\n{sample}")
            except Exception as exc:
                logger.debug(f"Process_file: Failed to get sample of table #{idx+1}: {exc}")
        # Output a sample of non-table raw lines
        if lines:
            sample_lines = lines[:2] if len(lines) >= 2 else lines
            logger.debug(f"Process_file: Sample non-tabular lines: {sample_lines}")
    # Extract key:value pairs for missing fields
    key_values = find_key_value_pairs(lines, col_map)
    if logger.isEnabledFor(logging.DEBUG) and key_values:
        logger.debug(f"Process_file: extracted key-value pairs: {key_values}")
    # Define required canonical fields.  Use environment override if
    # provided; otherwise fall back to the default list.  The default is
    # passed to ensure that modifications to the environment can be
    # detected, while preserving backward compatibility.
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

    # If the canonical field list includes ``Broker``, derive it from
    # the input filename.  The broker is the portion of the base file
    # name preceding the first underscore (e.g. ``ML_10022025.csv`` -> ``ML``).
    if "Broker" in required_fields:
        broker_value = _extract_broker_from_filename(input_path)
        if broker_value:
            key_values.setdefault("Broker", broker_value)
    extracted_tables: List[pd.DataFrame] = []
    # Process each detected table
    for table in tables:
        try:
            if table.empty:
                continue
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Process_file: processing table with columns {list(table.columns)} and {len(table)} row(s)"
                )
            # Determine if this table uses the symbol-group format.  If so,
            # realign the data by propagating the symbol to lot rows and
            # discarding summary or category rows.  Otherwise use the
            # table as-is.
            table_to_use = table
            try:
                if _is_symbol_group_table(table, col_map):
                    parsed_table = _parse_symbol_group_table(table, col_map)
                    if not parsed_table.empty:
                        table_to_use = parsed_table
            except Exception as exc:
                logger.debug(f"Symbol-group detection/parsing failed: {exc}")
            # Normalise column names on the (possibly realigned) table
            table_norm = normalise_table(table_to_use, col_map)
            # Determine how many canonical fields are present.  Require at
            # least the Symbol column and at least one other portfolio
            # field to consider this a valid holdings table.  This
            # prevents misc tables (e.g. filter descriptions) from being
            # processed erroneously.
            present_fields = {field for field in required_fields if field in table_norm.columns}
            if "Symbol" not in present_fields or len(present_fields) < 2:
                continue
            portfolio_df = extract_portfolio_fields(table_norm, required_fields, key_values)
            portfolio_df = portfolio_df.dropna(how="all")
            if not portfolio_df.empty:
                extracted_tables.append(portfolio_df)
        except Exception as exc:
            logger.warning(f"Failed to process a table: {exc}")
            continue
    # Concatenate or build fallback DataFrame
    if extracted_tables:
        combined_df = pd.concat(extracted_tables, ignore_index=True)
    else:
        if key_values:
            combined_df = pd.DataFrame({field: [key_values.get(field)] for field in required_fields})
        else:
            combined_df = pd.DataFrame(columns=required_fields)
    # Clean up the combined dataframe: remove blank and duplicate rows
    try:
        cleaned_df = cleanup_portfolio_df(combined_df)
    except Exception as exc:
        logger.warning(f"Failed to clean combined dataframe: {exc}")
        cleaned_df = combined_df
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"Process_file: combined dataframe has {len(cleaned_df)} row(s) and columns {list(cleaned_df.columns)} after cleanup"
        )
    # Determine output directory, allowing environment override
    out_dir_env = os.getenv("OUTPUT_DIR")
    if out_dir_env:
        output_dir = out_dir_env
    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as exc:
        logger.error(f"Failed to create output directory {output_dir}: {exc}")
        raise
    # Determine timestamp for filename
    if timestamp is None:
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"portfolio-{timestamp}.csv")
    try:
        # If append is requested and the output file already exists, read
        # existing data and concatenate with the new data before writing.
        if append and os.path.isfile(output_path):
            try:
                existing_df = pd.read_csv(output_path)
                cleaned_df = pd.concat([existing_df, cleaned_df], ignore_index=True)
                # Re-clean after concatenation to remove any new duplicates or blanks
                cleaned_df = cleanup_portfolio_df(cleaned_df)
            except Exception as exc:
                logger.warning(f"Failed to read existing output file for appending: {exc}")
        cleaned_df.to_csv(output_path, index=False)
        _audit(f"Wrote output CSV to {output_path}")
    except Exception as exc:
        logger.error(f"Failed to write output CSV: {exc}")
        raise
    return output_path

# -----------------------------------------------------------------------------
# Hybrid file support
#
# Some portfolio reports use a "hybrid" layout consisting of an account
# summary table followed by one or more holdings tables where the first
# row for each symbol summarises the lots and subsequent rows for that symbol
# omit the symbol value.  These helper functions detect such formats,
# extract account information, and realign the holdings rows so that each
# lot row contains the associated symbol.  The resulting dataframe is then
# normalised and written to an output CSV, just like in ``process_file``.

def _is_account_summary_table(df: pd.DataFrame) -> bool:
    """Heuristically determine whether a dataframe represents an account summary.

    An account summary table typically contains an ``Account`` column and
    additional columns such as ``Net Account Value`` or similar financial
    summary fields.  This helper performs a case-insensitive check on the
    column names to decide whether the table likely holds account-level
    information.

    Args:
        df: The dataframe to inspect.

    Returns:
        True if the table appears to be an account summary, False otherwise.
    """
    try:
        headers = [str(c).strip().lower() for c in df.columns]
        # Look for an Account column and any summary-related columns
        if any("account" == h for h in headers) and any(
            term in h for h in headers for term in ("net account value", "total gain", "available", "cash")
        ):
            return True
        return False
    except Exception:
        return False


def _is_holdings_table(df: pd.DataFrame) -> bool:
    """Heuristically determine whether a dataframe represents a holdings table.

    A holdings table is expected to contain per-security rows with columns
    such as ``Symbol`` and ``Date Acquired`` or ``Qty``.  This helper
    performs a case-insensitive check on the column names to decide whether
    the table likely holds position-level information.

    Args:
        df: The dataframe to inspect.

    Returns:
        True if the table appears to be a holdings table, False otherwise.
    """
    try:
        headers = [str(c).strip().lower() for c in df.columns]
        # Must contain a date acquired or similar and quantity or price
        # Determine if the table contains date and quantity columns.  Look
        # for several possible date phrases (e.g. "date acquired",
        # "purchase date", "initial purchase date") to accommodate
        # different report formats.
        has_date = any(
            any(term in h for term in ["date acquired", "purchase date", "initial purchase date"])
            for h in headers
        )
        # Quantity can appear as "qty", "quantity", or variants with '#' or numbers
        has_qty = any(("qty" in h) or ("quantity" in h) for h in headers)
        has_symbol = any("symbol" == h for h in headers)
        return has_date and has_qty and has_symbol
    except Exception:
        return False


def _parse_account_summary(df: pd.DataFrame, col_map: Dict[str, str]) -> Optional[str]:
    """Extract the account identifier from an account summary table.

    The input dataframe should contain a column that maps to the canonical
    ``Account`` field name.  The function normalises the column names
    according to ``col_map``, then returns the first non-empty value from
    the ``Account`` column.  If no such column or value exists, None is
    returned.

    Args:
        df: A dataframe suspected to contain account summary data.
        col_map: Mapping of normalised variant names to canonical names.

    Returns:
        The extracted account string, or None if unavailable.
    """
    try:
        if df.empty:
            return None
        df_norm = normalise_table(df.copy(), col_map)
        if "Account" not in df_norm.columns:
            return None
        for val in df_norm["Account"]:
            if val is not None and str(val).strip():
                return str(val).strip()
        return None
    except Exception as exc:
        logger.debug(f"Failed to parse account summary: {exc}")
        return None


def _parse_hybrid_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """Realign a holdings dataframe that omits the symbol on lot rows.

    In a hybrid holdings table the first row for each security contains a
    symbol and aggregated totals.  Subsequent rows for that security
    represent individual lots but do not repeat the symbol; instead the
    symbol cell is blank or contains whitespace.  This helper constructs a
    new dataframe where each lot row has the symbol filled in and the
    aggregated rows are discarded.  The output dataframe has the same
    columns as the input.

    Args:
        df: The holdings dataframe to realign.  Columns should match the
            original report header (e.g. ``Symbol, Date Acquired, ...``).

    Returns:
        A new dataframe containing only lot-level rows with symbols filled
        in.  If the input is empty, an empty dataframe is returned.
    """
    try:
        if df.empty:
            return df.copy()
        headers = list(df.columns)
        new_rows: List[List[str]] = []
        current_symbol: Optional[str] = None
        # Precompile a date pattern to identify lot rows
        date_pattern = re.compile(r"\d{1,2}/\d{1,2}/\d{4}")
        for row in df.itertuples(index=False, name=None):
            row_list = list(row)
            if not row_list:
                continue
            # Inspect the first cell; treat None/NaN as empty
            first_raw = row_list[0]
            second_raw = row_list[1] if len(row_list) > 1 else None
            # Determine trimmed values while handling NaN values
            if first_raw is None or (isinstance(first_raw, float) and pd.isna(first_raw)):
                first_trim = ""
            else:
                first_trim = str(first_raw).strip()
            if second_raw is None or (isinstance(second_raw, float) and pd.isna(second_raw)):
                second_trim = ""
            else:
                second_trim = str(second_raw).strip()
            # Determine whether this row is an aggregated summary.  A summary row
            # will have a non-empty first cell that does not resemble a date.
            is_first_date = bool(date_pattern.fullmatch(first_trim))
            if first_trim and not is_first_date:
                # Treat as aggregated summary row: update current_symbol and skip
                current_symbol = first_trim
                continue
            # For lot rows, the first cell is either empty or a date; we
            # require a current_symbol context to fill in.
            if current_symbol is None:
                continue
            # Realign the row by discarding the first cell and prefixing the symbol
            new_row = [current_symbol] + row_list[1:]
            new_rows.append(new_row)
        if not new_rows:
            return pd.DataFrame(columns=headers)
        return pd.DataFrame(new_rows, columns=headers)
    except Exception as exc:
        logger.debug(f"Failed to realign hybrid holdings: {exc}")
        return pd.DataFrame(columns=list(df.columns))


# -------------------------------------------------------------------------
# Symbol group parsing utilities
#
# Some portfolio reports use a grouped table format where a security's
# description and symbol appear only on a single summary row, followed by
# multiple lot rows without the symbol repeated.  These helpers detect
# such tables and realign them by propagating the symbol from the summary
# row to each subsequent lot row while discarding the summary row.  In
# addition, category headers (e.g. ``Cash``, ``Cash Equivalent``,
# ``Mutual Fund``, ``Total``) are skipped.  The ``Description`` column
# contains the account name for lot rows and is mapped to the canonical
# ``Account`` field via the column mapping configuration.

def _is_symbol_group_table(df: pd.DataFrame, col_map: Dict[str, str]) -> bool:
    """Determine whether a dataframe is a symbol-group table.

    A symbol-group table is characterised by the presence of a column
    corresponding to the canonical ``Symbol`` field and a column
    corresponding to the canonical ``Account`` field (often labelled
    ``Description``).  At least one row must have a non-empty symbol
    followed by one or more rows where the symbol cell is empty but
    the description cell is non-empty.  This pattern indicates a
    summary row followed by lot rows.

    Args:
        df: Dataframe to inspect.
        col_map: Mapping of normalised header variants to canonical
            field names.

    Returns:
        True if the dataframe appears to be a symbol-group table,
        False otherwise.
    """
    try:
        if df.empty:
            return False
        # Identify indices for symbol and description (account) columns
        symbol_idx = -1
        desc_idx = -1
        for idx, col_name in enumerate(df.columns):
            norm = _normalise_header(str(col_name))
            if norm in col_map:
                canonical = col_map[norm]
                if canonical == "Symbol":
                    symbol_idx = idx
                elif canonical == "Account":
                    desc_idx = idx
        # Require both symbol and description columns
        if symbol_idx < 0 or desc_idx < 0:
            return False
        found_nonempty_symbol = False
        found_empty_after = False
        for row in df.itertuples(index=False, name=None):
            try:
                sym_val = row[symbol_idx]
                desc_val = row[desc_idx]
            except Exception:
                continue
            # Treat None or NaN (including string 'nan') as empty
            sym_empty = True
            desc_empty = True
            try:
                # pandas will use float('nan') for blank cells; pd.isna handles this
                sym_empty = (sym_val is None) or (pd.isna(sym_val)) or (str(sym_val).strip().lower() in ("", "nan"))
            except Exception:
                sym_empty = False
            try:
                desc_empty = (desc_val is None) or (pd.isna(desc_val)) or (str(desc_val).strip().lower() in ("", "nan"))
            except Exception:
                desc_empty = False
            if not sym_empty:
                # Summary row with symbol; reset marker and continue
                found_nonempty_symbol = True
                continue
            # At this point symbol is empty
            if found_nonempty_symbol and not desc_empty:
                # Found at least one lot row following a summary row
                found_empty_after = True
                break
        return found_nonempty_symbol and found_empty_after
    except Exception:
        return False


def _parse_symbol_group_table(df: pd.DataFrame, col_map: Dict[str, str]) -> pd.DataFrame:
    """Realign a symbol-group dataframe by propagating symbols to lot rows.

    In symbol-group tables the first row for each security contains the
    description and symbol for the security along with aggregated
    totals.  Subsequent rows for that security represent individual
    lots but omit the symbol (and sometimes the description).  This
    helper constructs a new dataframe where each lot row has the symbol
    filled in from the preceding summary row and the summary rows are
    discarded.  Category headers (such as ``Cash``, ``Cash Equivalent``,
    ``Mutual Fund``, ``Total``) and entirely blank rows are also
    discarded.  The output dataframe retains the original column names.

    Args:
        df: Input dataframe to realign.  Columns should correspond
            exactly to the original report header (e.g. ``Description,
            Symbol, Quantity, ...``).
        col_map: Mapping of normalised header variants to canonical
            field names.  Used to locate the symbol and account
            (description) columns.

    Returns:
        A new dataframe containing only lot-level rows with symbols
        filled in.  If no lot rows are detected, an empty dataframe
        with the same columns is returned.
    """
    try:
        if df.empty:
            return df.copy()
        # Identify symbol and description (account) column indices
        symbol_idx = -1
        desc_idx = -1
        for idx, col_name in enumerate(df.columns):
            norm = _normalise_header(str(col_name))
            if norm in col_map:
                canonical = col_map[norm]
                if canonical == "Symbol":
                    symbol_idx = idx
                elif canonical == "Account":
                    desc_idx = idx
        if symbol_idx < 0 or desc_idx < 0:
            return pd.DataFrame(columns=df.columns)
        category_headings = {"total", "cash", "cash equivalent", "mutual fund"}
        new_rows: List[List[str]] = []
        current_symbol: Optional[str] = None
        for row in df.itertuples(index=False, name=None):
            row_list = list(row)
            # Skip completely empty rows
            if not any(str(cell).strip() for cell in row_list if not (pd.isna(cell) or (isinstance(cell, str) and cell.lower() == "nan"))):
                continue
            raw_symbol = row_list[symbol_idx]
            raw_desc = row_list[desc_idx]
            # Determine emptiness using pandas isna; treat 'nan' as empty
            sym_empty = True
            desc_empty = True
            try:
                sym_empty = (raw_symbol is None) or (pd.isna(raw_symbol)) or (str(raw_symbol).strip().lower() in ("", "nan"))
            except Exception:
                sym_empty = False
            try:
                desc_empty = (raw_desc is None) or (pd.isna(raw_desc)) or (str(raw_desc).strip().lower() in ("", "nan"))
            except Exception:
                desc_empty = False
            # Normalise description for category detection
            desc_str = "" if desc_empty else str(raw_desc).strip()
            desc_norm = desc_str.lower()
            # Skip category headings (cash, cash equivalent, mutual fund, total)
            if sym_empty and desc_norm in category_headings:
                continue
            if not sym_empty:
                # This is a summary row; store symbol and skip
                current_symbol = str(raw_symbol).strip()
                continue
            # For lot rows, ensure we have a current symbol and a non-empty description (account)
            if not current_symbol or desc_empty:
                continue
            # Assign current symbol to the symbol column
            row_list[symbol_idx] = current_symbol
            new_rows.append(row_list)
        if not new_rows:
            return pd.DataFrame(columns=df.columns)
        result_df = pd.DataFrame(new_rows, columns=df.columns)
        if logger.isEnabledFor(logging.DEBUG):
            try:
                preview = result_df.head(2).to_string(index=False)
                logger.debug(
                    f"Symbol-group parser produced {len(result_df)} lot row(s) with columns {list(result_df.columns)}. Sample:\n{preview}"
                )
            except Exception:
                pass
        return result_df
    except Exception as exc:
        logger.debug(f"Failed to realign symbol-group table: {exc}")
        return pd.DataFrame(columns=list(df.columns))


def detect_file_structure(
    input_path: str,
    config_path: str,
    tabs: Optional[List[str]] = None,
) -> str:
    """Classify the structure of a portfolio data file.

    The function examines the first few tables in the input file to decide
    whether it should be processed as a standard single-table report or
    as a hybrid multi-table report.  The hybrid format is identified by
    the presence of an account summary table followed by a holdings table
    where the symbol is only specified on the first row of each security.
    If neither of these heuristics match, ``standard`` is returned.

    Args:
        input_path: Path to the CSV or Excel file to inspect.
        config_path: Path to the column mapping configuration (unused here
            but included for future extensibility).

    Returns:
        A string indicating the file structure: either ``"hybrid"`` or
        ``"standard"``.  Unknown formats default to ``"standard"``.
    """
    try:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Detecting file structure for {input_path}")
        # Ensure input exists and has known extension
        if not os.path.isfile(input_path):
            return "standard"
        ext = os.path.splitext(input_path)[1].lower()
        if ext == ".csv":
            tables, _ = read_csv_file(input_path)
        elif ext in {".xls", ".xlsx"}:
            tables, _ = read_excel_file(input_path, sheets_to_process=tabs)
        else:
            return "standard"
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Found {len(tables)} table(s) in {input_path}")
        # Locate the account summary table and the first holdings table after it
        acc_index = None
        for idx, tbl in enumerate(tables):
            try:
                if _is_account_summary_table(tbl):
                    acc_index = idx
                    break
            except Exception:
                continue
        if acc_index is None:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("No account summary table found; treating as standard")
            return "standard"
        # Search for holdings table after the account summary
        for h_idx in range(acc_index + 1, len(tables)):
            tbl = tables[h_idx]
            try:
                if _is_holdings_table(tbl):
                    realigned_preview = _parse_hybrid_holdings(tbl)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"Hybrid parser preview produced {len(realigned_preview)} row(s) from table #{h_idx+1}"
                        )
                    if not realigned_preview.empty:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                f"File structure classified as hybrid (account summary at table {acc_index+1}, holdings at table {h_idx+1})"
                            )
                        return "hybrid"
            except Exception:
                continue
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("File structure classified as standard")
        return "standard"
    except Exception:
        return "standard"


def process_hybrid_file(
    input_path: str,
    config_path: str,
    output_dir: str = "./out",
    timestamp: Optional[str] = None,
    *,
    tabs: Optional[List[str]] = None,
    append: bool = False,
) -> str:
    """Process a portfolio report with a hybrid layout.

    This function orchestrates the extraction of data from files that
    contain both account summary and holdings tables.  It follows these
    steps:

    1. Determine file type and read all tables.
    2. Identify account summary and holdings tables.
    3. Extract the account identifier from the summary table.
    4. Realign the holdings tables so each lot row has a symbol.
    5. Normalise column names, extract portfolio fields and write the
       consolidated dataframe to CSV.

    Args:
        input_path: Path to the CSV or Excel file to process.
        config_path: Path to the JSON mapping of column names.
        output_dir: Directory where the resulting CSV should be written.
        timestamp: Optional timestamp string.  If omitted the current
            local time is used in YYYYMMDD_HHMMSS format.

    Keyword Args:
        tabs: Optional list of sheet names to process when reading an
            Excel workbook.  If provided, only tables from these tabs
            will be considered.  Ignored for CSV files.
        append: If True and an output file with the chosen timestamp
            already exists, new portfolio rows will be appended to it
            instead of overwriting.

    Returns:
        The full path to the written CSV file.
    """
    # Validate the input file exists and has a supported extension
    if not os.path.isfile(input_path):
        logger.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Input file not found: {input_path}")
    ext = os.path.splitext(input_path)[1].lower()
    if ext not in {".csv", ".xls", ".xlsx"}:
        logger.error(f"Unsupported file extension: {ext}")
        raise ValueError(f"Unsupported file extension: {ext}")
    # Load column mapping
    col_map = load_column_mapping(config_path)
    # Read tables and raw lines; restrict to specific tabs if provided.
    if ext == ".csv":
        tables, lines = read_csv_file(input_path)
    else:
        tables, lines = read_excel_file(input_path, sheets_to_process=tabs)
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"Hybrid processing: loaded {len(tables)} table(s) and {len(lines)} raw line(s) from {input_path}"
        )
        # Output a sample of each raw table for inspection
        for t_idx, tbl in enumerate(tables):
            try:
                if not tbl.empty:
                    sample = tbl.head(2).to_string(index=False)
                    logger.debug(f"Hybrid processing: Table #{t_idx+1} sample before classification:\n{sample}")
            except Exception as exc:
                logger.debug(f"Hybrid processing: Failed to get sample of table #{t_idx+1}: {exc}")
        # Output a sample of non-table raw lines
        if lines:
            sample_lines = lines[:2] if len(lines) >= 2 else lines
            logger.debug(f"Hybrid processing: Sample non-tabular lines: {sample_lines}")
    # Identify account value and realign holdings
    account_value: Optional[str] = None
    holdings_tables: List[pd.DataFrame] = []
    for idx, tbl in enumerate(tables):
        try:
            if tbl.empty:
                continue
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Hybrid processing: examining table #{idx+1} with columns {list(tbl.columns)} and {len(tbl)} row(s)"
                )
            if _is_account_summary_table(tbl) and account_value is None:
                account_value = _parse_account_summary(tbl, col_map)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Hybrid processing: identified account summary table with account {account_value}"
                    )
                continue
            if _is_holdings_table(tbl):
                # Determine whether this holdings table uses the symbol-group layout or the
                # classic hybrid layout.  If it contains symbol-group patterns, use
                # the symbol-group parser; otherwise fall back to the hybrid parser.
                try:
                    if _is_symbol_group_table(tbl, col_map):
                        realigned = _parse_symbol_group_table(tbl, col_map)
                    else:
                        realigned = _parse_hybrid_holdings(tbl)
                except Exception as exc:
                    logger.debug(f"Hybrid processing: failed to realign table #{idx+1}: {exc}")
                    realigned = pd.DataFrame(columns=tbl.columns)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Hybrid processing: realigned holdings table has {len(realigned)} lot row(s)"
                    )
                    # Output a sample of the realigned holdings rows
                    try:
                        if not realigned.empty:
                            sample_realigned = realigned.head(2).to_string(index=False)
                            logger.debug(
                                f"Hybrid processing: realigned holdings sample for table #{idx+1}:\n{sample_realigned}"
                            )
                    except Exception as exc:
                        logger.debug(f"Hybrid processing: Failed to get sample of realigned table #{idx+1}: {exc}")
                if not realigned.empty:
                    holdings_tables.append(realigned)
        except Exception as exc:
            logger.debug(f"Failed to classify table: {exc}")
            continue
    # Combine all holdings
    if holdings_tables:
        holdings_combined = pd.concat(holdings_tables, ignore_index=True)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Hybrid processing: combined holdings dataframe has {len(holdings_combined)} row(s) before normalisation"
            )
        # Normalise headers to canonical names
        holdings_norm = normalise_table(holdings_combined, col_map)
    else:
        holdings_norm = pd.DataFrame()
    # Extract key:value pairs outside of tables for fallback
    key_values = find_key_value_pairs(lines, col_map)
    # Ensure the Account field is set either from account summary or key_values
    if account_value:
        key_values.setdefault("Account", account_value)
    if logger.isEnabledFor(logging.DEBUG) and key_values:
        logger.debug(
            f"Hybrid processing: extracted key-value pairs (fallbacks): {key_values}"
        )
    # Define required canonical fields, allowing override via
    # environment variable.  Provide the default list as the fallback
    # so that custom output field lists can be configured externally.
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

    # If Broker appears in the required field list, derive it from the
    # input filename.  Use the file name prefix before the first
    # underscore to populate the ``Broker`` fallback.
    if "Broker" in required_fields:
        broker_value = _extract_broker_from_filename(input_path)
        if broker_value:
            key_values.setdefault("Broker", broker_value)
    extracted_tables: List[pd.DataFrame] = []
    if not holdings_norm.empty:
        try:
            portfolio_df = extract_portfolio_fields(holdings_norm, required_fields, key_values)
            portfolio_df = portfolio_df.dropna(how="all")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Hybrid processing: extracted portfolio dataframe with {len(portfolio_df)} row(s)"
                )
            if not portfolio_df.empty:
                extracted_tables.append(portfolio_df)
        except Exception as exc:
            logger.warning(f"Failed to extract portfolio fields from hybrid holdings: {exc}")
    # If no holdings or extraction failed, fallback to key-values only
    if not extracted_tables:
        if key_values:
            combined_df = pd.DataFrame({field: [key_values.get(field)] for field in required_fields})
        else:
            combined_df = pd.DataFrame(columns=required_fields)
    else:
        combined_df = pd.concat(extracted_tables, ignore_index=True)
    # Clean up the combined dataframe: remove blank and duplicate rows
    try:
        cleaned_df = cleanup_portfolio_df(combined_df)
    except Exception as exc:
        logger.warning(f"Failed to clean combined dataframe: {exc}")
        cleaned_df = combined_df
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"Hybrid processing: final dataframe has {len(cleaned_df)} row(s) and columns {list(cleaned_df.columns)} after cleanup"
        )
    # Determine output directory (honour OUTPUT_DIR env)
    out_dir_env = os.getenv("OUTPUT_DIR")
    if out_dir_env:
        output_dir = out_dir_env
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as exc:
        logger.error(f"Failed to create output directory {output_dir}: {exc}")
        raise
    if timestamp is None:
        timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"portfolio-{timestamp}.csv")
    try:
        # If appending and the output file exists, merge with existing data
        if append and os.path.isfile(output_path):
            try:
                existing_df = pd.read_csv(output_path)
                cleaned_df = pd.concat([existing_df, cleaned_df], ignore_index=True)
                cleaned_df = cleanup_portfolio_df(cleaned_df)
            except Exception as exc:
                logger.warning(f"Failed to read existing output file for appending: {exc}")
        cleaned_df.to_csv(output_path, index=False)
        _audit(f"Wrote hybrid output CSV to {output_path}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Hybrid processing: final output dataframe has {len(cleaned_df)} row(s) and columns {list(cleaned_df.columns)}"
            )
    except Exception as exc:
        logger.error(f"Failed to write output CSV: {exc}")
        raise
    return output_path


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