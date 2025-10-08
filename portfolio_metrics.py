"""
portfolio_metrics.py

This module introduces a small collection of single‑purpose utility functions
for computing common portfolio statistics, namely the compound annual
growth rate (CAGR) and the beta coefficient relative to a market index.
These functions are designed to be reused in various contexts and do
not make any assumptions about the structure of the broader
application.  Each calculation is accompanied by robust input
validation, detailed documentation, and optional debug logging via the
standard Python ``logging`` module.  Audit events are recorded in the
global audit log defined in ``portfolio_utils.py`` when applicable.

To calculate CAGR, supply either the initial and final portfolio
values along with the duration in years, or provide a time series of
values and specify the number of periods per year.  For beta
calculation, provide aligned sequences of portfolio returns and
benchmark (market) returns.  The functions return floating point
numbers representing the computed metric, or ``None`` if the
calculation cannot be performed due to invalid inputs.

The design of this module follows Clean Architecture principles: each
function has a single responsibility and can be tested in isolation.
Configuration (such as logging levels) is controlled via
environment variables inherited from the main application.  No
configuration values are hardcoded here; instead, logging and
auditing behaviour depend on settings established in
``portfolio_utils.py``.  The module deliberately avoids any I/O
operations and does not depend on pandas, making it lightweight and
general.

@file        portfolio_metrics.py
@brief       Financial metrics utilities (CAGR and beta)
@author      IRS Assistant
@created     2025-10-07
@modified    2025-10-07

Copyright (c) 2025. All rights reserved.

This file is part of the Python Assistant project and is licensed
under the terms described in the project root.  See the LICENSE file
for details.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, List, Optional

try:
    # Import the audit helper from portfolio_utils to record audit events.
    from .portfolio_utils import _audit
except Exception:
    # Fallback: define a no‑op audit function if import fails.  This
    # occurs when portfolio_utils is not available in the import path.
    def _audit(message: str) -> None:  # type: ignore[misc]
        return

__all__ = [
    "calculate_cagr",
    "calculate_cagr_from_series",
    "calculate_beta",
]


# Configure module logger.  This logger inherits handlers from the
# root configuration established in ``portfolio_utils.py``.  It
# defaults to INFO level unless the DEBUG environment variable is
# enabled in the main application.
logger = logging.getLogger(__name__)


def calculate_cagr(
    initial_value: float,
    final_value: float,
    years: float,
) -> Optional[float]:
    """Compute the compound annual growth rate (CAGR) for an investment.

    The CAGR is defined as the constant annual growth rate that will
    yield a final value given an initial value over a specified number
    of years.  Mathematically,

    .. math:: \text{CAGR} = \left(\frac{V_{\text{final}}}{V_{\text{initial}}}\right)^{\frac{1}{\text{years}}} - 1

    Args:
        initial_value: The starting value of the investment.  Must be
            positive.
        final_value: The ending value of the investment.
        years: The duration over which growth occurred, measured in
            years.  Must be positive.

    Returns:
        The CAGR as a decimal fraction (e.g. 0.08 for 8% growth), or
        ``None`` if the input values are invalid (e.g. non‑positive
        initial value or zero/negative years).

    Example:
        >>> calculate_cagr(1000, 1500, 3)
        0.1447...

    Notes:
        * If ``initial_value`` or ``years`` is not positive, the
          function logs an error and returns ``None``.
        * Debug messages include intermediate computations when the
          module's logger level is set to DEBUG.
    """
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
    """Compute CAGR from a series of portfolio values sampled at regular intervals.

    When you have a time series of portfolio values (e.g. monthly or
    quarterly account balances), the CAGR can be estimated by taking
    the ratio of the final value to the initial value and raising it
    to the appropriate exponent based on the number of periods per
    year.  If the series contains fewer than two values, the CAGR
    cannot be computed and ``None`` is returned.

    Args:
        values: A sequence of numeric values representing portfolio
            balances over time.  The order should be chronological.
        periods_per_year: How many periods constitute one year (e.g.
            12 for monthly data, 4 for quarterly).  Must be positive.

    Returns:
        The CAGR as a decimal fraction, or ``None`` if the series is
        invalid or too short.  An error is logged if ``periods_per_year``
        is non‑positive.

    Example:
        >>> calculate_cagr_from_series([100, 110, 121], 12)
        0.0954...
    """
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
    """Compute the beta of a portfolio relative to a benchmark index.

    Beta measures the volatility of a portfolio relative to the
    broader market.  It is calculated as the covariance of the
    portfolio and benchmark returns divided by the variance of the
    benchmark returns:

    .. math:: \beta = \frac{\mathrm{cov}(R_p, R_m)}{\mathrm{var}(R_m)}

    Args:
        portfolio_returns: A sequence of periodic returns for the
            portfolio.  Each element should correspond in time to the
            matching element in ``benchmark_returns``.
        benchmark_returns: A sequence of periodic returns for the
            market or benchmark index.

    Returns:
        The beta coefficient as a float, or ``None`` if the inputs
        are invalid (e.g. mismatched lengths or insufficient data).

    Example:
        >>> calculate_beta([0.01, 0.02, -0.01], [0.02, 0.03, 0.00])
        0.75...

    Notes:
        * Both input sequences must contain at least two data points.
        * If the variance of the benchmark returns is zero, the beta
          cannot be computed; the function logs an error and returns
          ``None``.
        * Debug messages include intermediate statistics (mean,
          covariance, variance) when the module logger is set to DEBUG.
    """
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