"""Tests for daily_scans.anchor.compute_rs_anchor — pure pivot math, no IO.

Feeds a synthetic benchmark frame directly (no parquet) and asserts the
(pivot_date, anchor_close, today_close) parity the repo's _compute_nifty_rs_anchor
used to assert.
"""
import datetime as dt

import pandas as pd
import pytest

from daily_scans import compute_rs_anchor

from _helpers import _make_nifty_with_pivot


def test_insufficient_history_returns_none():
    df = pd.DataFrame({
        'symbol': 'NIFTY 50',
        'date': pd.bdate_range('2024-01-01', periods=100),
        'high': 100.0, 'low': 99.0, 'close': 99.5,
        'instrument_type': 'INDEX',
    })
    assert compute_rs_anchor(df, k=3.0) is None


def test_picks_synthetic_peak():
    df = _make_nifty_with_pivot()
    result = compute_rs_anchor(df, k=3.0)
    assert result is not None
    pivot_date, anchor_close, today_close = result
    # Expected peak: the constructed peak bar (index n_warmup + n_pre).
    expected_peak_date = df.iloc[160 + 170]['date'].date()
    actual = pivot_date if isinstance(pivot_date, dt.date) else pivot_date.date()
    # Allow ±7 bars slack — ZigZag may pick an adjacent bar of equal/higher high.
    assert abs((actual - expected_peak_date).days) <= 7
    assert today_close == pytest.approx(float(df.iloc[-1]['close']))
    assert anchor_close > 0
