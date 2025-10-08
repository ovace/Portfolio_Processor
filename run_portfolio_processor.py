#!/usr/bin/env python3
"""
run_portfolio_processor.py

Caller/orchestrator for the portfolio extraction utilities.

Enhancements in this version:
- Appends across multiple input files (single timestamp per run)
- Optional Excel sheet filtering via config INPUT_FILES[].tabs
- Config-driven OUTPUT_FIELDS and Broker-from-filename fallback
- Detailed DEBUG/AUDIT logging
- **New:** post-processing metrics integration (CAGR per-row; optional portfolio Beta)

Usage::

    python run_portfolio_processor.py <input-file> [<input-file> ...] [--config CONFIG] [--outdir OUTDIR]
                                       [--timestamp TIMESTAMP] [--debug] [--show-audit]

If no <input-file> arguments are provided, the script uses INPUT_FILES from the config.

Configuration keys considered (default_settings.json):
- OUTPUT_DIR
- DEBUG
- OUTPUT_FIELDS
- INPUT_FILES: list of {"path": str, "tabs": [str]|null}
- METRICS (optional):
    {
      "ENABLE": true,
      "COMPUTE_CAGR_PER_ROW": true,
      "RETURNS": {
        "PORTFOLIO_FILE": null,
        "BENCHMARK_FILE": null,
        "COLUMN": "return",
        "PERIODS_PER_YEAR": 252
      }
    }
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.portfolio_utils import (
    get_audit_log,
    detect_file_structure,
    process_file,
    process_hybrid_file,
)
from utils.portfolio_metrics import (
    calculate_cagr,
    calculate_beta,
)


# -------------------------- helpers --------------------------

def _load_settings(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _apply_env_from_settings(settings: Dict[str, Any]) -> None:
    # Flatten a few top-level values into env for utils
    for k in ("OUTPUT_DIR", "DEBUG", "OUTPUT_FIELDS"):
        if k in settings and settings[k] is not None:
            os.environ[k] = str(settings[k])


def _normalize_input_entries(args_files: List[str] | None, settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if args_files:
        for f in args_files:
            entries.append({"path": f, "tabs": None})
    else:
        cfg_files = settings.get("INPUT_FILES", [])
        if isinstance(cfg_files, list):
            for entry in cfg_files:
                if isinstance(entry, dict):
                    entries.append({
                        "path": entry.get("path"),
                        "tabs": entry.get("tabs") if entry.get("tabs") not in (None, []) else None,
                    })
                else:
                    entries.append({"path": entry, "tabs": None})
    # Filter out invalid
    return [e for e in entries if isinstance(e.get("path"), str) and e["path"]]


# -------------------------- metrics --------------------------

def _add_metrics_columns(df: pd.DataFrame, settings: Dict[str, Any]) -> pd.DataFrame:
    metrics_cfg = settings.get("METRICS", {}) if isinstance(settings.get("METRICS"), dict) else {}
    enable_metrics = bool(metrics_cfg.get("ENABLE", True))
    if not enable_metrics:
        return df

    out = df.copy()

    # --- CAGR per row (uses Acquisition Date for time; Total Cost -> Value) ---
    if bool(metrics_cfg.get("COMPUTE_CAGR_PER_ROW", True)):
        # Parse dates & numerics defensively
        acq = pd.to_datetime(out.get("Acquisition Date"), errors="coerce")
        today = pd.Timestamp.today().normalize()
        years = (today - acq).dt.days / 365.25

        # Initial value: prefer Total Cost; fallback to Quantity * Cost per Unit
        def _to_num(s):
            return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")

        total_cost = _to_num(out.get("Total Cost")) if "Total Cost" in out.columns else pd.Series([pd.NA] * len(out))
        quantity = _to_num(out.get("Quantity")) if "Quantity" in out.columns else pd.Series([pd.NA] * len(out))
        cpu = _to_num(out.get("Cost per Unit")) if "Cost per Unit" in out.columns else pd.Series([pd.NA] * len(out))
        initial = total_cost.where(total_cost.notna(), quantity * cpu)

        final_val = _to_num(out.get("Value")) if "Value" in out.columns else pd.Series([pd.NA] * len(out))

        cagr_vals: List[Optional[float]] = []
        for iv, fv, yrs in zip(initial.tolist(), final_val.tolist(), years.tolist() if years is not None else [None]*len(out)):
            if iv is None or pd.isna(iv) or iv <= 0:
                cagr_vals.append(None)
                continue
            if fv is None or pd.isna(fv) or fv <= 0:
                cagr_vals.append(None)
                continue
            if yrs is None or pd.isna(yrs) or yrs <= 0:
                cagr_vals.append(None)
                continue
            cagr_vals.append(calculate_cagr(float(iv), float(fv), float(yrs)))
        out["Years Held"] = years
        out["CAGR"] = cagr_vals

    # --- Portfolio-level Beta (optional, requires returns series files) ---
    returns_cfg = metrics_cfg.get("RETURNS", {}) if isinstance(metrics_cfg.get("RETURNS"), dict) else {}
    pf_file = returns_cfg.get("PORTFOLIO_FILE")
    bm_file = returns_cfg.get("BENCHMARK_FILE")
    col = returns_cfg.get("COLUMN", "return")

    beta_value: Optional[float] = None
    if pf_file and bm_file and os.path.isfile(pf_file) and os.path.isfile(bm_file):
        try:
            pf = pd.read_csv(pf_file)
            bm = pd.read_csv(bm_file)
            if col not in pf.columns:
                # fallback: first column
                pf_series = pf.iloc[:, 0].astype(float).tolist()
            else:
                pf_series = pd.to_numeric(pf[col], errors="coerce").dropna().astype(float).tolist()
            if col not in bm.columns:
                bm_series = bm.iloc[:, 0].astype(float).tolist()
            else:
                bm_series = pd.to_numeric(bm[col], errors="coerce").dropna().astype(float).tolist()
            # Align lengths
            n = min(len(pf_series), len(bm_series))
            if n >= 2:
                beta_value = calculate_beta(pf_series[:n], bm_series[:n])
        except Exception:
            beta_value = None

    if beta_value is not None:
        out["Beta"] = beta_value

    return out


# -------------------------- main --------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Portfolio Processor")
    parser.add_argument("input_files", nargs="*", help="Optional list of input files. If omitted, uses INPUT_FILES from config.")
    parser.add_argument("--config", default="config/default_settings.json", help="Path to settings JSON")
    parser.add_argument("--outdir", default=None, help="Override output directory")
    parser.add_argument("--timestamp", default=None, help="Override timestamp for output file name")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--show-audit", action="store_true", help="Print audit log at the end")

    args = parser.parse_args(argv)

    # Load and apply settings
    settings = _load_settings(args.config)
    if args.outdir:
        settings["OUTPUT_DIR"] = args.outdir
    if args.debug:
        settings["DEBUG"] = True
    _apply_env_from_settings(settings)

    # Build list of file entries
    file_entries = _normalize_input_entries(args.input_files, settings)
    if not file_entries:
        sys.stderr.write("No input files provided or configured.\n")
        return 2

    # Process each entry
    output_dir = settings.get("OUTPUT_DIR") or os.environ.get("OUTPUT_DIR") or "./out"
    os.makedirs(output_dir, exist_ok=True)

    ts = args.timestamp  # keep a single timestamp across files
    append_flag = False
    last_output_path: Optional[str] = None

    for entry in file_entries:
        file_path = entry.get("path")
        tabs = entry.get("tabs")
        if not isinstance(file_path, str) or not os.path.isfile(file_path):
            sys.stderr.write(f"Error: input file does not exist: {file_path}\n")
            continue
        # Decide structure
        structure = detect_file_structure(file_path, settings.get("COLUMN_MAPPING", "config/column_mapping.json"), tabs=tabs)
        if structure == "hybrid":
            outpath = process_hybrid_file(file_path, settings.get("COLUMN_MAPPING", "config/column_mapping.json"), output_dir=output_dir, timestamp=ts, tabs=tabs, append=append_flag)
        else:
            outpath = process_file(file_path, settings.get("COLUMN_MAPPING", "config/column_mapping.json"), output_dir=output_dir, timestamp=ts, tabs=tabs, append=append_flag)
        if ts is None and outpath:
            # derive timestamp from filename portion
            base = os.path.basename(outpath)
            # portfolio-YYYYMMDD_HHMMSS.csv
            try:
                ts = base.split("portfolio-")[1].split(".csv")[0]
            except Exception:
                pass
        last_output_path = outpath
        append_flag = True

    if not last_output_path:
        sys.stderr.write("No output was produced.\n")
        return 3

    # ---------------- Metrics & final save ----------------
    try:
        df = pd.read_csv(last_output_path)
    except Exception as exc:
        sys.stderr.write(f"Failed to load output for metrics: {exc}\n")
        return 4

    df_with_metrics = _add_metrics_columns(df, settings)

    # Overwrite final CSV with metrics columns appended to the right
    try:
        df_with_metrics.to_csv(last_output_path, index=False)
    except Exception as exc:
        sys.stderr.write(f"Failed to write output with metrics: {exc}\n")
        return 5

    if args.show_audit:
        for level, msg in get_audit_log():
            print(f"[{level}] {msg}")

    print(f"Portfolio processed and metrics added. Output: {last_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
