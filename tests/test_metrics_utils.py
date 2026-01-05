import pandas as pd
import numpy as np
import pytest
from utils.metrics_utils import (
    calculate_cagr,
    calculate_beta_timeseries,
    calculate_portfolio_beta_holdings,
    _to_float
)

def test_metrics_to_float():
    assert _to_float(10) == 10.0
    assert _to_float("123.45") == 123.45
    assert _to_float("1,234.56") == 1234.56
    assert _to_float(None) is None
    assert _to_float("abc") is None

def test_metrics_calculate_cagr():
    # Double in 1 year = 100%
    assert calculate_cagr(100, 200, 1) == 1.0
    # Double in 2 years = ~41.4%
    cagr = calculate_cagr(100, 200, 2)
    assert abs(cagr - 0.41421356) < 1e-6
    
    # Invalid inputs
    assert calculate_cagr(0, 100, 1) is None
    assert calculate_cagr(100, 0, 1) is None
    assert calculate_cagr(100, 200, 0) is None

def test_calculate_beta_timeseries():
    p = pd.Series([0.01, 0.02, 0.01, 0.03])
    b = pd.Series([0.01, 0.02, 0.01, 0.03])
    # Perfect correlation
    assert abs(calculate_beta_timeseries(p, b) - 1.0) < 1e-9

    p_inv = pd.Series([-0.01, -0.02, -0.01, -0.03])
    assert abs(calculate_beta_timeseries(p_inv, b) - (-1.0)) < 1e-9

def test_calculate_portfolio_beta_holdings():
    holdings = pd.DataFrame({
        "Symbol": ["A", "B"],
        "Value": [1000, 3000], # 25% and 75% weights
        "Beta": [1.0, 2.0]
    })
    # (0.25 * 1.0) + (0.75 * 2.0) = 0.25 + 1.5 = 1.75
    beta = calculate_portfolio_beta_holdings(holdings)
    assert abs(beta - 1.75) < 1e-9

def test_calculate_portfolio_beta_holdings_fallback():
    holdings = pd.DataFrame({
        "Symbol": ["A"],
        "Weight": [1.0], # Fallback column
        "Beta": [1.5]
    })
    assert calculate_portfolio_beta_holdings(holdings) == 1.5
