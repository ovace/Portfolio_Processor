"""
utils/portfolio_metrics.py

This module introduces a small collection of single‑purpose utility functions
for computing common portfolio statistics, namely the compound annual
growth rate (CAGR) and the beta coefficient relative to a market index.
"""
from __future__ import annotations

import logging
import math
from typing import Iterable, List, Optional

try:
    # relative import works because both are now in utils package
    from .portfolio_utils import _audit
except Exception:
    def _audit(message: str) -> None:  # type: ignore[misc]
        return

__all__ = [
    "calculate_cagr",
    "calculate_cagr_from_series",
    "calculate_beta",
]


# Configure module logger.
logger = logging.getLogger(__name__)


def calculate_cagr(
    initial_value: float,
    final_value: float,
    years: float,
) -> Optional[float]:
    """Compute the compound annual growth rate (CAGR) for an investment."""
    try:
        if initial_value <= 0:
            logger.error(
                f"CAGR calculation failed: initial_value must be positive (got {initial_value})"
            )
            return None
        if years <= 0:
            logger.error(
                f"CAGR calculation failed: years must be positive (got {years})"
            )
            return None
        # Avoid division by zero when final_value is zero
        if final_value <= 0:
            logger.error(
                f"CAGR calculation failed: final_value must be positive (got {final_value})"
            )
            return None
        growth_ratio = final_value / initial_value
        cagr = growth_ratio ** (1.0 / years) - 1.0
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"CAGR calculation: initial={initial_value}, final={final_value}, years={years}, "
                f"ratio={growth_ratio}, cagr={cagr}"
            )
        _audit(
            f"Calculated CAGR: initial_value={initial_value}, final_value={final_value}, years={years}, cagr={cagr}"
        )
        return cagr
    except Exception as exc:
        logger.error(f"CAGR calculation error: {exc}")
        return None


def calculate_cagr_from_series(
    values: Iterable[float],
    periods_per_year: float,
) -> Optional[float]:
    """Compute CAGR from a series of portfolio values sampled at regular intervals."""
    try:
        values_list: List[float] = list(values)
        n = len(values_list)
        if n < 2:
            logger.error(
                "CAGR series calculation failed: at least two values are required"
            )
            return None
        if periods_per_year <= 0:
            logger.error(
                f"CAGR series calculation failed: periods_per_year must be positive (got {periods_per_year})"
            )
            return None
        initial_value = values_list[0]
        final_value = values_list[-1]
        years = (n - 1) / periods_per_year
        return calculate_cagr(initial_value, final_value, years)
    except Exception as exc:
        logger.error(f"CAGR series calculation error: {exc}")
        return None


def calculate_beta(
    portfolio_returns: Iterable[float],
    benchmark_returns: Iterable[float],
) -> Optional[float]:
    """Compute the beta of a portfolio relative to a benchmark index."""
    try:
        port: List[float] = list(portfolio_returns)
        bench: List[float] = list(benchmark_returns)
        if len(port) != len(bench):
            logger.error(
                "Beta calculation failed: portfolio and benchmark returns must have equal length"
            )
            return None
        if len(port) < 2:
            logger.error(
                "Beta calculation failed: at least two return observations are required"
            )
            return None
        # Compute means
        mean_port = sum(port) / len(port)
        mean_bench = sum(bench) / len(bench)
        # Compute covariance between portfolio and benchmark
        cov_num = sum((p - mean_port) * (b - mean_bench) for p, b in zip(port, bench))
        covariance = cov_num / (len(port) - 1)
        # Compute variance of benchmark
        var_num = sum((b - mean_bench) ** 2 for b in bench)
        variance = var_num / (len(bench) - 1)
        if variance == 0:
            logger.error("Beta calculation failed: variance of benchmark returns is zero")
            return None
        beta = covariance / variance
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Beta calculation: mean_port={mean_port}, mean_bench={mean_bench}, "
                f"covariance={covariance}, variance={variance}, beta={beta}"
            )
        _audit(
            f"Calculated beta: observations={len(port)}, covariance={covariance}, variance={variance}, beta={beta}"
        )
        return beta
    except Exception as exc:
        logger.error(f"Beta calculation error: {exc}")
        return None
