"""
utils/readers.py

Handles file reading (CSV/Excel) and parsing into raw Pandas DataFrames.
"""
from __future__ import annotations

import csv
import logging
import os
from typing import List, Optional, Tuple

import pandas as pd

# Setup basic logger (shared configuration assumed to be present in app context)
logger = logging.getLogger(__name__)

def _audit(message: str) -> None:
    # Minimal in-module audit shim; in a full clean arch, IAuditService would be injected.
    # For now, we log as info with AUDIT prefix to maintain compatibility.
    logger.info(f"AUDIT: {message}")

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
        with pd.ExcelFile(path, engine="openpyxl") as xls:
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
