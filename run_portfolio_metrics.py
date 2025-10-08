#!/usr/bin/env python3
"""
=================================================================================
 Script: run_portfolio_metrics.py
 Project: Financial Analytics Utilities
 Role: Caller/orchestrator to compute portfolio CAGR and Beta using utils.
 Location: ProjectRoot/
 
 Clean Architecture:
   - Orchestration only: reads config/inputs, calls pure utils, writes outputs.
   - No hidden I/O in utility functions.
   - All configuration from environment or ./config/default_settings.json.
 
 Functions:
   - main(): parses args, wires config, computes metrics, emits JSON/CSV report.
 
 Author: Ovace A. Mamnoon
 Created: 2025-10-07
 Version: 1.0.0
=================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import math
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

# Local imports
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
from metrics_utils import (
    calculate_cagr,
    calculate_beta_timeseries,
    calculate_portfolio_beta_holdings,
    get_audit_log,
)

# ----------------------------
# Logging setup (recyclable)
# ----------------------------
def setup_logging(log_dir: str, debug: bool = False) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "run_portfolio_metrics.log")
    # Archive previous log on each fresh run
    if os.path.exists(log_path):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.rename(log_path, os.path.join(log_dir, f"run_portfolio_metrics_{ts}.log"))
    logger = logging.getLogger("run_portfolio_metrics")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# ----------------------------
# Config loading
# ----------------------------
def load_defaults(cfg_path: str) -> dict:
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def coerce_returns(series: pd.Series) -> pd.Series:
    """Accept decimals (0.01) or percents (1.0 = 100%). If abs(mean) > 1 => assume % and divide by 100."""
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return s
    mean_abs = s.dropna().abs().mean()
    if mean_abs > 1.0:
        return s / 100.0
    return s

def infer_years_from_dates(df: pd.DataFrame, date_col: str, first_row: int, last_row: int) -> Optional[float]:
    try:
        d0 = pd.to_datetime(df.loc[first_row, date_col])
        d1 = pd.to_datetime(df.loc[last_row, date_col])
        delta_days = (d1 - d0).days
        if delta_days <= 0:
            return None
        return float(delta_days) / 365.25
    except Exception:
        return None

def read_holdings(csv_path: str) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception:
        return None

def write_report(output_dir: str, payload: dict) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"metrics_report_{ts}.json")
    csv_path = os.path.join(output_dir, f"metrics_report_{ts}.csv")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        json_path = ""
    try:
        pd.DataFrame([payload]).to_csv(csv_path, index=False)
    except Exception:
        csv_path = ""
    return {"json": json_path, "csv": csv_path}

def main() -> int:
    project_root = os.path.dirname(__file__)
    cfg_dir = os.path.join(project_root, "config")
    logs_dir = os.getenv("LOG_DIR", os.path.join(project_root, "logs"))
    cfg_path = os.path.join(project_root, "default_settings.json")
    # fall back to config/default_settings.json if present
    if not os.path.exists(cfg_path):
        cfg_path = os.path.join(cfg_dir, "default_settings.json")

    defaults = load_defaults(cfg_path)
    # Environment overrides
    for k, v in defaults.items():
        os.environ.setdefault(k, str(v))

    parser = argparse.ArgumentParser(description="Compute portfolio CAGR and Beta.")
    parser.add_argument("--holdings", type=str, default=os.getenv("HOLDINGS_CSV", defaults.get("HOLDINGS_CSV", "")),
                        help="CSV with positions; needs Beta column and Value/Weight column.")
    parser.add_argument("--portfolio-returns", type=str, default=os.getenv("PORTFOLIO_RETURNS_CSV", defaults.get("PORTFOLIO_RETURNS_CSV", "")),
                        help="CSV file with a single column of portfolio returns.")
    parser.add_argument("--benchmark-returns", type=str, default=os.getenv("BENCHMARK_RETURNS_CSV", defaults.get("BENCHMARK_RETURNS_CSV", "")),
                        help="CSV file with a single column of benchmark returns.")
    parser.add_argument("--values-csv", type=str, default=os.getenv("PORTFOLIO_VALUES_CSV", defaults.get("PORTFOLIO_VALUES_CSV", "")),
                        help="CSV with Date and Value columns to infer CAGR (begin/end/years).")
    parser.add_argument("--date-col", type=str, default=os.getenv("DATE_COL", defaults.get("DATE_COL", "Date")), help="Date column name for values-csv.")
    parser.add_argument("--value-col", type=str, default=os.getenv("VALUE_COL", defaults.get("VALUE_COL", "Value")), help="Value column name for values-csv.")
    parser.add_argument("--begin", type=float, default=None, help="Beginning value for CAGR.")
    parser.add_argument("--end", type=float, default=None, help="Ending value for CAGR.")
    parser.add_argument("--years", type=float, default=None, help="Years between begin and end (e.g., 36 months = 3.0).")
    parser.add_argument("--output-dir", type=str, default=os.getenv("OUTPUT_DIR", defaults.get("OUTPUT_DIR", os.path.join(project_root, "logs"))),
                        help="Directory to write metrics report.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    logger = setup_logging(logs_dir, debug=args.debug)

    logger.info("Starting run_portfolio_metrics")
    logger.info(f"Args: {vars(args)}")

    # ------------------
    # Compute Beta
    # ------------------
    beta_ts = None
    if args.portfolio_returns and args.benchmark_returns and os.path.exists(args.portfolio_returns) and os.path.exists(args.benchmark_returns):
        try:
            pr = pd.read_csv(args.portfolio_returns).iloc[:, 0]
            br = pd.read_csv(args.benchmark_returns).iloc[:, 0]
            pr = coerce_returns(pr)
            br = coerce_returns(br)
            beta_ts = calculate_beta_timeseries(pr, br)
            logger.info(f"Timeseries Beta: {None if beta_ts is None else round(beta_ts, 6)}")
        except Exception as exc:
            logger.error(f"Failed to compute timeseries beta: {exc}")

    beta_holdings = None
    if beta_ts is None and args.holdings and os.path.exists(args.holdings):
        try:
            holdings_df = read_holdings(args.holdings)
            beta_holdings = calculate_portfolio_beta_holdings(holdings_df)
            logger.info(f"Holdings Beta: {None if beta_holdings is None else round(beta_holdings, 6)}")
        except Exception as exc:
            logger.error(f"Failed to compute holdings beta: {exc}")

    beta = beta_ts if beta_ts is not None else beta_holdings

    # ------------------
    # Compute CAGR
    # ------------------
    cagr = None
    if args.begin is not None and args.end is not None and args.years is not None:
        cagr = calculate_cagr(args.begin, args.end, args.years)
    elif args.values_csv and os.path.exists(args.values_csv):
        try:
            dfv = pd.read_csv(args.values_csv)
            if args.value_col not in dfv.columns or args.date_col not in dfv.columns:
                raise KeyError(f"CSV must contain '{args.date_col}' and '{args.value_col}' columns.")
            dfv = dfv.dropna(subset=[args.value_col, args.date_col]).reset_index(drop=True)
            if dfv.empty:
                raise ValueError("No usable rows in values CSV.")
            begin_value = float(dfv.loc[0, args.value_col])
            end_value = float(dfv.loc[len(dfv)-1, args.value_col])
            years = infer_years_from_dates(dfv, args.date_col, 0, len(dfv)-1) or float(len(dfv)) / 12.0
            cagr = calculate_cagr(begin_value, end_value, years)
        except Exception as exc:
            logging.getLogger().error(f"Failed to compute CAGR from values CSV: {exc}")

    # ------------------
    # Report
    # ------------------
    payload = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "metrics": {
            "beta": beta,
            "beta_method": "timeseries" if beta_ts is not None else ("holdings" if beta_holdings is not None else None),
            "cagr": cagr,
        },
        "inputs": {
            "holdings": args.holdings,
            "portfolio_returns": args.portfolio_returns,
            "benchmark_returns": args.benchmark_returns,
            "values_csv": args.values_csv,
        },
        "config": {
            "BETA_COLUMN": os.getenv("BETA_COLUMN", "Beta"),
            "BETA_WEIGHT_COLUMN": os.getenv("BETA_WEIGHT_COLUMN", "Value"),
        },
        "audit": get_audit_log(),
        "diagnostics": {"ok": beta is not None or cagr is not None}
    }
    paths = write_report(args.output_dir, payload)

    logger.info(f"Report written: {paths}")
    print(json.dumps({"report_paths": paths, "summary": payload["metrics"]}, indent=2))
    logger.info("Completed run_portfolio_metrics")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
