import sys
import os
import math
import pytest

# Ensure root is in path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.portfolio_metrics import calculate_cagr, calculate_beta, calculate_cagr_from_series

def test_calculate_cagr():
    # Standard case
    assert calculate_cagr(100, 200, 1) == 1.0  # 100% growth
    assert abs(calculate_cagr(100, 110, 1) - 0.1) < 1e-9
    
    # Approx check for 3 years doubling
    cagr_doubling = calculate_cagr(100, 200, 3) # ~26%
    assert 0.25 < cagr_doubling < 0.27

    # Error cases
    assert calculate_cagr(100, 200, 0) is None
    assert calculate_cagr(0, 200, 1) is None
    assert calculate_cagr(100, 200, -1) is None

def test_calculate_cagr_from_series():
    series = [100, 110, 121]
    # 2 periods, 1 per year => 2 years. 100->121 is 10% annual
    assert abs(calculate_cagr_from_series(series, 1) - 0.1) < 1e-9
    
    # 12 periods per year (monthly), 2 steps = 2/12 = 1/6 year
    # 100 -> 121 in 2 months is huge growth
    val = calculate_cagr_from_series(series, 12)
    assert val > 0

def test_calculate_beta():
    # Correlated
    pf = [0.01, 0.02, 0.03]
    bm = [0.01, 0.02, 0.03]
    assert abs(calculate_beta(pf, bm) - 1.0) < 1e-9

    # Inverse
    pf_inv = [-0.01, -0.02, -0.03]
    assert abs(calculate_beta(pf_inv, bm) - (-1.0)) < 1e-9

    # Error
    assert calculate_beta([1], [1]) is None # too short
    assert calculate_beta([1,2], [1,2,3]) is None # mismatch
