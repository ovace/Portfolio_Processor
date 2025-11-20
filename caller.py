#!/usr/bin/env python3
"""
caller.py — Unified CLI for the PortfolioProcessor project

This single entrypoint consolidates the previous callers:
  - run_portfolio_processor.py  (extraction/normalization/cleansing)
  - run_portfolio_metrics.py    (CAGR/Beta metrics)

It provides one cohesive command line with subcommands and flags to:
  • process portfolio files (CSV/XLS/XLSX),
  • compute metrics, and
  • emit per‑file and overall summaries (reporting utils),
  • append results across multiple inputs using a single timestamp.

Clean architecture: this script orchestrates only — it reads config/args,
then calls utilities in utils/, and writes outputs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import pandas as pd

# ---- Local imports ----
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
from utils.reporting_utils import FileReport, OverallReport

# -------------------------- Settings helpers --------------------------

def _load_settings(path: str) -> Dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _apply_env_from_settings(settings: Dict[str, Any]) -> None:
    """Expose a few top‑level settings as env for utils that consult os.environ."""
    for k in ("OUTPUT_DIR", "DEBUG", "OUTPUT_FIELDS"):
        if k in settings and settings[k] is not None:
            os.environ[k] = str(settings[k])


def _normalize_input_entries(args_files: List[str] | None, settings: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list of {path, tabs} from CLI or config INPUT_FILES."""
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
    return [e for e in entries if isinstance(e.get("path"), str) and e["path"]]


# -------------------------- Metrics helpers --------------------------

def _add_metrics_columns(df: pd.DataFrame, settings: Dict[str, Any]) -> pd.DataFrame:
    """Append metrics columns to the right of df based on settings.METRICS."""
    metrics_cfg = settings.get("METRICS", {}) if isinstance(settings.get("METRICS"), dict) else {}
    enable_metrics = bool(metrics_cfg.get("ENABLE", True))
    if not enable_metrics:
        return df

    out = df.copy()

    # --- CAGR per row (Acquisition Date -> years; Total Cost -> Value) ---
    if bool(metrics_cfg.get("COMPUTE_CAGR_PER_ROW", True)):
        acq = pd.to_datetime(out.get("Acquisition Date"), errors="coerce")
        today = pd.Timestamp.today().normalize()
        years = (today - acq).dt.days / 365.25

        def _to_num(s):
            return pd.to_numeric(s.astype(str).str.replace(",", "", regex=False), errors="coerce")

        total_cost = _to_num(out.get("Total Cost")) if "Total Cost" in out.columns else pd.Series([pd.NA] * len(out))
        quantity = _to_num(out.get("Quantity")) if "Quantity" in out.columns else pd.Series([pd.NA] * len(out))
        cpu = _to_num(out.get("Cost per Unit")) if "Cost per Unit" in out.columns else pd.Series([pd.NA] * len(out))
        initial = total_cost.where(total_cost.notna(), quantity * cpu)
        final_val = _to_num(out.get("Value")) if "Value" in out.columns else pd.Series([pd.NA] * len(out))

        cagr_vals: List[Optional[float]] = []  # type: ignore[name-defined]
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
                pf_series = pf.iloc[:, 0].astype(float).tolist()
            else:
                pf_series = pd.to_numeric(pf[col], errors="coerce").dropna().astype(float).tolist()
            if col not in bm.columns:
                bm_series = bm.iloc[:, 0].astype(float).tolist()
            else:
                bm_series = pd.to_numeric(bm[col], errors="coerce").dropna().astype(float).tolist()
            n = min(len(pf_series), len(bm_series))
            if n >= 2:
                beta_value = calculate_beta(pf_series[:n], bm_series[:n])
        except Exception:
            beta_value = None

    if beta_value is not None:
        out["Beta"] = beta_value

    return out


# -------------------------- Orchestration --------------------------

def run_processing(settings: Dict[str, Any], files: List[str] | None, *, debug: bool = False,
                   outdir_override: Optional[str] = None, timestamp: Optional[str] = None,
                   show_audit: bool = False, summaries: bool = True) -> Optional[str]:
    """Process one or more input files, append across inputs, and return output path.
    Also logs per‑file and overall summaries via reporting_utils when enabled.
    """
    _apply_env_from_settings(settings)

    # Build entries (path,tabs) from CLI or config
    file_entries = _normalize_input_entries(files, settings)
    if not file_entries:
        sys.stderr.write("No input files provided or configured.\n")
        return None

    output_dir = outdir_override or settings.get("OUTPUT_DIR") or os.environ.get("OUTPUT_DIR") or "./out"
    os.makedirs(output_dir, exist_ok=True)

    ts = timestamp  # single timestamp across appended files
    append_flag = False
    last_output_path: Optional[str] = None

    overall = OverallReport()

    for entry in file_entries:
        file_path = entry.get("path")
        tabs = entry.get("tabs")
        if not isinstance(file_path, str) or not os.path.isfile(file_path):
            sys.stderr.write(f"Error: input file does not exist: {file_path}\n")
            continue

        structure = detect_file_structure(file_path, settings.get("COLUMN_MAPPING", "config/column_mapping.json"), tabs=tabs)
        # prep file‑level reporter
        fr = FileReport(input_path=file_path, structure=structure)

        if structure == "hybrid":
            outpath = process_hybrid_file(
                file_path,
                settings.get("COLUMN_MAPPING", "config/column_mapping.json"),
                output_dir=output_dir,
                timestamp=ts,
                tabs=tabs,
                append=append_flag,
                file_report=fr,  # type: ignore[arg-type]
            )
        else:
            outpath = process_file(
                file_path,
                settings.get("COLUMN_MAPPING", "config/column_mapping.json"),
                output_dir=output_dir,
                timestamp=ts,
                tabs=tabs,
                append=append_flag,
                file_report=fr,  # type: ignore[arg-type]
            )
        if ts is None and outpath:
            base = os.path.basename(outpath)
            try:
                ts = base.split("portfolio-")[1].split(".csv")[0]
            except Exception:
                pass
        last_output_path = outpath
        append_flag = True
        overall.add(fr)

    # Log aggregated summary and optionally persist JSON rollup
    if summaries and overall.files:
        overall.log_overall()
        try:
            rollup_path = os.path.join(output_dir, f"summary-{ts or 'latest'}.json")
            with open(rollup_path, "w", encoding="utf-8") as fh:
                json.dump(overall.aggregate(), fh, indent=2)
            print(f"Summary written to: {rollup_path}")
        except Exception:
            pass

    if not last_output_path:
        sys.stderr.write("No output was produced.\n")
        return None

    if show_audit:
        for level, msg in get_audit_log():
            print(f"[{level}] {msg}")

    return last_output_path


def run_metrics_on_output(output_csv: str, settings: Dict[str, Any]) -> bool:
    """Open final CSV, append metrics columns, and overwrite in place."""
    try:
        df = pd.read_csv(output_csv)
    except Exception as exc:
        sys.stderr.write(f"Failed to load output for metrics: {exc}\n")
        return False

    df_with_metrics = _add_metrics_columns(df, settings)

    try:
        df_with_metrics.to_csv(output_csv, index=False)
        return True
    except Exception as exc:
        sys.stderr.write(f"Failed to write output with metrics: {exc}\n")
        return False


# -------------------------- CLI --------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified Portfolio Processor CLI — process files, compute metrics, and produce summaries.")

    sub = p.add_subparsers(dest="command")

    # process subcommand
    sp = sub.add_parser("process", help="Process CSV/XLS/XLSX inputs and write normalized CSV.")
    sp.add_argument("input_files", nargs="*", help="Optional list of input files. If omitted, uses INPUT_FILES from config.")
    sp.add_argument("--config", default="config/default_settings.json", help="Path to settings JSON")
    sp.add_argument("--outdir", default=None, help="Override output directory")
    sp.add_argument("--timestamp", default=None, help="Override output file timestamp")
    sp.add_argument("--debug", action="store_true", help="Enable debug logging")
    sp.add_argument("--show-audit", action="store_true", help="Print audit log at the end")
    sp.add_argument("--no-summaries", action="store_true", help="Disable JSON/log summaries generation")

    # metrics subcommand
    sm = sub.add_parser("metrics", help="Append metrics (CAGR/Beta) to an existing processed CSV.")
    sm.add_argument("output_csv", help="Path to processed portfolio CSV to augment with metrics.")
    sm.add_argument("--config", default="config/default_settings.json", help="Path to settings JSON")

    # all‑in‑one subcommand (default workflow)
    sa = sub.add_parser("all", help="Process inputs then append metrics in one go (default workflow).")
    sa.add_argument("input_files", nargs="*", help="Optional list of input files. If omitted, uses INPUT_FILES from config.")
    sa.add_argument("--config", default="config/default_settings.json", help="Path to settings JSON")
    sa.add_argument("--outdir", default=None, help="Override output directory")
    sa.add_argument("--timestamp", default=None, help="Override output file timestamp")
    sa.add_argument("--debug", action="store_true", help="Enable debug logging")
    sa.add_argument("--show-audit", action="store_true", help="Print audit log at the end")
    sa.add_argument("--no-summaries", action="store_true", help="Disable JSON/log summaries generation")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Default to "all" if no subcommand provided
    command = args.command or "all"

    # Load settings once
    cfg_path = getattr(args, "config", "config/default_settings.json")
    settings = _load_settings(cfg_path)

    if command == "metrics":
        ok = run_metrics_on_output(args.output_csv, settings)
        print("Metrics appended." if ok else "Metrics failed.")
        return 0 if ok else 5

    # process and all share the same processing path
    files = getattr(args, "input_files", None)
    outdir = getattr(args, "outdir", None)
    ts = getattr(args, "timestamp", None)
    debug = getattr(args, "debug", False)
    show_audit = getattr(args, "show_audit", False)
    summaries = not getattr(args, "no_summaries", False)

    output_csv = run_processing(
        settings,
        files,
        debug=debug,
        outdir_override=outdir,
        timestamp=ts,
        show_audit=show_audit,
        summaries=summaries,
    )

    if not output_csv:
        return 3

    if command in ("all",):
        ok = run_metrics_on_output(output_csv, settings)
        print("Portfolio processed and metrics added." if ok else "Processing done, metrics failed.")
        return 0 if ok else 5

    # command == "process"
    print(f"Portfolio processed. Output: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
