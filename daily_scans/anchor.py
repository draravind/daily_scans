"""Benchmark-pivot relative-strength anchor — index-AGNOSTIC.

Lifts the math from the repo's `_compute_nifty_rs_anchor`; the repo decides
WHICH index, keeps the parquet read, and passes the frame in.
"""
import pandas as pd
import talib

from daily_scans import scan_cores


def compute_rs_anchor(index_ohlc_df, *, k=3.0, warmup=132, window=252):
    """Find a benchmark index's most recent confirmed swing high.

    `index_ohlc_df` is ANY benchmark's OHLC frame (date-sorted, cols
    date/high/low/close). Runs `scan_cores._zigzag_pivots` with the given `k`
    on the most recent `window` bars (with `warmup` bars before that to warm up
    the ATR-pct SMA). k/warmup/window are tunable defaults, not index-specific.

    Returns (pivot_date, close_at_pivot, close_today) or None.
    """
    if len(index_ohlc_df) < warmup + window:
        return None

    series = index_ohlc_df.tail(warmup + window).reset_index(drop=True)
    highs = series['high'].values
    lows = series['low'].values
    closes = series['close'].values

    tr = talib.ATR(highs, lows, closes, 1)
    atr_pct = pd.Series(tr / closes).rolling(warmup, min_periods=warmup).mean().values

    pivot_start = len(series) - window
    pivots = scan_cores._zigzag_pivots(
        highs=highs[pivot_start:],
        lows=lows[pivot_start:],
        atr_pct=atr_pct[pivot_start:],
        closes=closes[pivot_start:],
        k=k,
    )
    if len(pivots) < 2:
        return None

    confirmed = pivots[:-1]
    last_h = next((p for p in reversed(confirmed) if p[2] == 'H'), None)
    if last_h is None:
        return None

    local_idx, _, _ = last_h
    global_idx = pivot_start + local_idx
    pivot_date_raw = series.at[global_idx, 'date']
    pivot_date = pivot_date_raw.date() if hasattr(pivot_date_raw, 'date') else pivot_date_raw
    return pivot_date, float(closes[global_idx]), float(closes[-1])
