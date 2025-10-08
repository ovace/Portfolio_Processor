"""
===============================================================================
 Module: metrics_utils.py
 Project: Financial Analytics Utilities
 Location: ProjectRoot/utils/
 Description:
   Single-purpose, reusable utility functions to calculate portfolio metrics
   (CAGR and Beta) with robust logging, diagnostics, and audit support.
   Designed for Clean Architecture: pure functions, no hidden I/O, all
   configuration via environment variables or config/default_settings.json.

 Author: Ovace A. Mamnoon
 Created: 2025-10-07
 Version: 1.0.0
 Dependencies: numpy, pandas, logging; respects LOG_DIR, DEBUG, AUDIT envs
===============================================================================
"""

from __future__ import annotations

import os
import math
import logging
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
__author__ = "Ovace A. Mamnoon"
__version__ = "1.0.0"
__maintainer__ = "EA Automation Framework"
__status__ = "Production"

# ---------------------------------------------------------------------------
# Logging & Audit (reuses project conventions)
# ---------------------------------------------------------------------------

DEBUG_ENV = os.getenv("DEBUG", "false").lower() == "true"

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.setLevel(logging.DEBUG if DEBUG_ENV else logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    # File
    try:
        log_dir = os.getenv("LOG_DIR", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "metrics_utils.log")
        # Archive existing file at each fresh run to keep clean logs
        if os.path.exists(log_path):
            import datetime as _dt
            ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            os.rename(log_path, os.path.join(log_dir, f"metrics_utils_{ts}.log"))
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        # If file logging fails, continue with console logging
        pass

_AUDIT_ON = os.getenv("AUDIT", "true").lower() == "true"
_AUDIT_LOG: list[tuple[str, str]] = []

def _audit(msg: str) -> None:
    if not _AUDIT_ON:
        return
    import datetime as _dt
    ts = _dt.datetime.now().isoformat(timespec="seconds")
    _AUDIT_LOG.append((ts, msg))
    logger.info(f"AUDIT: {msg}")

def get_audit_log() -> list[tuple[str, str]]:
    return list(_AUDIT_LOG)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    "calculate_cagr",
    "calculate_beta_timeseries",
    "calculate_portfolio_beta_holdings",
    "get_audit_log",
]

def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float, np.floating)):
            return float(x)
        s = str(x).strip().replace(",", "")
        return float(s)
    except Exception:
        return None

def calculate_cagr(begin_value: float, end_value: float, years: float) -> Optional[float]:
    """
    Compute Compound Annual Growth Rate (CAGR).

    Formula:
        CAGR = (End / Begin) ** (1 / Years) - 1

    Args:
        begin_value: Starting portfolio value (must be > 0).
        end_value: Ending portfolio value (must be > 0).
        years: Exact number of years between the two values (> 0).
               For monthly data over N months, pass years = N / 12.

    Returns:
        CAGR as a decimal (e.g., 0.086 for 8.6%), or None if inputs invalid.

    Notes:
        - Pure function: no I/O; errors are logged and None is returned.
        - Accepts numeric-like strings and will safely coerce.
    """
    try:
        b = _to_float(begin_value)
        e = _to_float(end_value)
        y = _to_float(years)
        if b is None or e is None or y is None:
            raise ValueError("Inputs must be numeric.")
        if b <= 0 or e <= 0 or y <= 0:
            raise ValueError("All inputs must be positive and non-zero.")
        cagr = (e / b) ** (1.0 / y) - 1.0
        logger.debug(f"CAGR computed: {cagr:.6f} (Begin={b}, End={e}, Years={y})")
        return float(cagr)
    except Exception as exc:
        logger.error(f"calculate_cagr failed: {exc}")
        return None

def calculate_beta_timeseries(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    ddof: int = 1
) -> Optional[float]:
    """
    Compute portfolio Beta from synchronized time series of returns.

    Beta = Cov(R_p, R_b) / Var(R_b)

    Args:
        portfolio_returns: pandas Series of periodic returns (decimal, e.g., 0.012).
        benchmark_returns: pandas Series of corresponding benchmark returns.
        ddof: Delta degrees of freedom for covariance/variance (default 1).

    Returns:
        Beta as float, or None if insufficient/invalid data.

    Behavior:
        - Aligns on intersection of valid indices.
        - Drops NaNs.
        - Uses numpy.cov with ddof; guards for zero variance.
    """
    try:
        if not isinstance(portfolio_returns, pd.Series) or not isinstance(benchmark_returns, pd.Series):
            raise TypeError("Inputs must be pandas Series.")
        df = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        if df.shape[0] < 2:
            raise ValueError("Not enough paired observations to compute beta.")
        p = df.iloc[:, 0].astype(float).values
        b = df.iloc[:, 1].astype(float).values
        cov = np.cov(p, b, ddof=ddof)[0, 1]
        var_b = np.var(b, ddof=ddof)
        if var_b == 0 or math.isclose(var_b, 0.0, abs_tol=1e-18):
            raise ZeroDivisionError("Benchmark variance is zero.")
        beta = float(cov / var_b)
        logger.debug(f"Timeseries Beta computed: {beta:.6f} (n={len(b)})")
        return beta
    except Exception as exc:
        logger.error(f"calculate_beta_timeseries failed: {exc}")
        return None

def calculate_portfolio_beta_holdings(
    holdings: pd.DataFrame,
    beta_column: str | None = None,
    weight_column: str | None = None
) -> Optional[float]:
    """
    Compute portfolio Beta from a holdings dataframe via value-weighted average.

    Args:
        holdings: DataFrame of positions. Expected columns include a Beta column
                  (e.g., 'Beta' or configured) and a weight proxy such as
                  current market value (e.g., 'Value').
        beta_column: Column name for per-position beta. If None, uses env
                     'BETA_COLUMN' or defaults to 'Beta'.
        weight_column: Column name for weights. If None, uses env
                       'BETA_WEIGHT_COLUMN' or falls back to 'Value', then 'Weight'.

    Returns:
        Value-weighted portfolio Beta or None if inputs invalid.

    Logic:
        - Coerces beta and weights to numeric (dropping NaNs and non-positive weights).
        - Normalizes weights to sum to 1.
        - Computes sum_i (w_i * beta_i).
    """
    try:
        if holdings is None or not isinstance(holdings, pd.DataFrame) or holdings.empty:
            raise ValueError("Holdings DataFrame is empty.")
        beta_col = beta_column or os.getenv("BETA_COLUMN", "Beta")
        w_col = weight_column or os.getenv("BETA_WEIGHT_COLUMN", "Value")
        if beta_col not in holdings.columns:
            raise KeyError(f"Missing beta column '{beta_col}'.")
        if w_col not in holdings.columns:
            # try a reasonable fallback
            if "Weight" in holdings.columns:
                w_col = "Weight"
            else:
                raise KeyError(f"Missing weight/value column '{w_col}'.")

        betas = pd.to_numeric(holdings[beta_col], errors="coerce")
        weights = pd.to_numeric(holdings[w_col], errors="coerce")

        df = pd.DataFrame({"beta": betas, "w": weights}).dropna()
        df = df[df["w"] > 0]
        if df.empty:
            raise ValueError("No valid beta/weight rows to compute portfolio beta.")

        w_sum = df["w"].sum()
        if w_sum <= 0:
            raise ValueError("Non-positive total weight.")
        weights_norm = df["w"] / w_sum
        port_beta = float((weights_norm * df["beta"]).sum())
        logger.debug(f"Holdings Beta computed: {port_beta:.6f} (rows={len(df)})")
        return port_beta
    except Exception as exc:
        logger.error(f"calculate_portfolio_beta_holdings failed: {exc}")
        return None

# ---------------------------------------------------------------------------
# Diagnostics (optional, safe to run)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("metrics_utils diagnostics starting...")
    # CAGR smoke test
    cagr = calculate_cagr(10000, 16105.10, 5)
    print("CAGR:", None if cagr is None else f"{cagr:.4%}")
    # Beta from timeseries smoke test
    np.random.seed(0)
    p = pd.Series(np.random.normal(0.01, 0.05, 36))
    b = pd.Series(np.random.normal(0.008, 0.04, 36))
    beta_ts = calculate_beta_timeseries(p, b)
    print("Beta (TS):", None if beta_ts is None else f"{beta_ts:.4f}")
    # Beta from holdings smoke test
    holdings = pd.DataFrame({
        "Symbol": ["AAA", "BBB", "CCC"],
        "Value": [50000, 30000, 20000],
        "Beta": [1.2, 0.9, 1.1],
    })
    beta_holdings = calculate_portfolio_beta_holdings(holdings)
    print("Beta (Holdings):", None if beta_holdings is None else f"{beta_holdings:.4f}")
    logger.info("metrics_utils diagnostics complete.")
