"""
Pure per-stock scan logic extracted from daily_scans.py.

Each function signature: (stock: str, data_dict: dict, ctx=None) -> Optional[Any]
- Returns None  → skip (not appended to results)
- Returns value → collected into results list
- Raises exception → runner catches, logs, records failure
"""
import datetime as dt

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import talib

from daily_scans.context import ScanContext


def _ctx(ctx):
    """Resolve injected run-state, defaulting to a calendar-free local today."""
    return ctx if ctx is not None else ScanContext(today=dt.date.today())


def _zigzag_pivots(highs, lows, atr_pct, closes, k):
    """
    Volatility-adjusted ZigZag pivot detector.

    Reversal threshold at bar i is k * atr_pct[i] * closes[i] (absolute
    price units). Walks forward through the series tracking a provisional
    pivot; a candidate H or L is confirmed only when price reverses by at
    least the threshold computed at the provisional pivot's bar.

    Returns: list of (idx, price, kind) in chronological order, where
    kind is 'H' (swing high) or 'L' (swing low). The final entry is
    provisional (not yet reversed against) and may be revised by future
    bars — callers that need a fully-confirmed pivot should drop the
    last entry.
    """
    n = len(highs)
    if n < 2:
        return []

    pivots = []

    max_idx, max_price = 0, highs[0]
    min_idx, min_price = 0, lows[0]
    i = 1
    direction = None
    ext_idx, ext_price = 0, highs[0]
    while i < n:
        if highs[i] >= max_price:
            max_idx, max_price = i, highs[i]
        if lows[i] <= min_price:
            min_idx, min_price = i, lows[i]

        if np.isnan(atr_pct[max_idx]) or np.isnan(atr_pct[min_idx]):
            i += 1
            continue

        thr_from_high = k * atr_pct[max_idx] * closes[max_idx]
        thr_from_low = k * atr_pct[min_idx] * closes[min_idx]

        down_reversal = max_price - lows[i] >= thr_from_high
        up_reversal = highs[i] - min_price >= thr_from_low

        if down_reversal and up_reversal:
            if max_idx <= min_idx:
                pivots.append((max_idx, max_price, 'H'))
                direction = 'down'
                ext_idx, ext_price = i, lows[i]
            else:
                pivots.append((min_idx, min_price, 'L'))
                direction = 'up'
                ext_idx, ext_price = i, highs[i]
            i += 1
            break
        elif down_reversal:
            pivots.append((max_idx, max_price, 'H'))
            direction = 'down'
            ext_idx, ext_price = i, lows[i]
            i += 1
            break
        elif up_reversal:
            pivots.append((min_idx, min_price, 'L'))
            direction = 'up'
            ext_idx, ext_price = i, highs[i]
            i += 1
            break
        i += 1
    else:
        if max_idx >= min_idx:
            return [(max_idx, max_price, 'H')]
        return [(min_idx, min_price, 'L')]

    while i < n:
        if np.isnan(atr_pct[ext_idx]):
            i += 1
            continue
        threshold = k * atr_pct[ext_idx] * closes[ext_idx]
        if direction == 'down':
            if lows[i] <= ext_price:
                ext_idx, ext_price = i, lows[i]
            if highs[i] - ext_price >= threshold:
                pivots.append((ext_idx, ext_price, 'L'))
                direction = 'up'
                ext_idx, ext_price = i, highs[i]
        else:
            if highs[i] >= ext_price:
                ext_idx, ext_price = i, highs[i]
            if ext_price - lows[i] >= threshold:
                pivots.append((ext_idx, ext_price, 'H'))
                direction = 'down'
                ext_idx, ext_price = i, lows[i]
        i += 1

    pivots.append((ext_idx, ext_price, 'H' if direction == 'up' else 'L'))
    return pivots


def _williams_pivots(highs, lows, n=2):
    """5-bar Williams swing pivots (n=2 → 5-bar window).

    Strict inequality against each side separately, matching the form
    inlined in `higher_highs_higher_lows`. Tie equals on either side
    disqualify the centre bar.

    Returns (swing_highs, swing_lows) — each a list of (idx, price)
    tuples. Excludes the first n and last n bars (pivots are only
    confirmed once n future bars exist).
    """
    swing_h, swing_l = [], []
    for i in range(n, len(highs) - n):
        if highs[i] > max(highs[i - n:i]) and highs[i] > max(highs[i + 1:i + n + 1]):
            swing_h.append((i, highs[i]))
        if lows[i] < min(lows[i - n:i]) and lows[i] < min(lows[i + 1:i + n + 1]):
            swing_l.append((i, lows[i]))
    return swing_h, swing_l


def near_ATH_high_ATR(stock, data_dict, ctx=None):
    ctx = _ctx(ctx)
    start = ctx.today - dt.timedelta(365 * 5)
    full_data = data_dict[stock]
    ath = full_data['high'].max()

    data = full_data[full_data.index > start].copy()
    if data.empty:
        return None

    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['ATR'] = talib.SMA(data['TR'], 125)
    data['EMA_200'] = talib.EMA(data['close'], 200)

    current_high = data['high'].iloc[-1]
    condition_1 = current_high <= ath
    condition_2 = current_high > ath * min(0.85, (1 - 4 * data['ATR'].iloc[-1]))
    condition_3 = data['close'].iloc[-2:].max() >= data['EMA_200'].iloc[-1]
    condition_4 = data['ATR'].iloc[-1] >= 0.036

    if condition_1 and condition_2 and condition_3 and condition_4:
        return stock
    return None


def near_ATH_low_ATR(stock, data_dict, ctx=None):
    ctx = _ctx(ctx)
    start = ctx.today - dt.timedelta(365 * 5)
    full_data = data_dict[stock]
    ath = full_data['high'].max()

    data = full_data[full_data.index > start].copy()
    if data.empty:
        return None

    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['ATR'] = talib.SMA(data['TR'], 125)
    data['EMA_200'] = talib.EMA(data['close'], 200)

    current_high = data['high'].iloc[-1]
    condition_1 = current_high <= ath
    condition_2 = current_high > ath * min(0.85, (1 - 4 * data['ATR'].iloc[-1]))
    condition_3 = data['close'].iloc[-2:].max() >= data['EMA_200'].iloc[-1]
    condition_4 = data['ATR'].iloc[-1] < 0.036

    if condition_1 and condition_2 and condition_3 and condition_4:
        return stock
    return None


def _ath_runs(data):
    """Compute run-length encoding of ATH plateaus. Returns DataFrame of runs."""
    change_points = data['ATH'] != data['ATH'].shift()
    group_ids = change_points.cumsum()
    grouped = data.groupby(group_ids, observed=True)
    runs = grouped.agg(
        start=('ATH', lambda x: x.index[0]),
        end=('ATH', lambda x: x.index[-1]),
        value=('ATH', 'first'),
        length=('ATH', 'size'),
    ).reset_index(drop=True)
    return runs[::-1]


def _find_recent_long_run(runs, data):
    """Find the most recent ATH consolidation base (>100 bars, not the current run)."""
    long_runs = runs[runs['length'] > 100]
    if long_runs.empty:
        return None
    if long_runs.iloc[0]['end'] == data.index[-1]:
        if len(long_runs) > 1:
            return long_runs.iloc[1]
        return None
    return long_runs.iloc[0]


def making_new_ATH_high_ATR(stock, data_dict, ctx=None):
    ctx = _ctx(ctx)
    start_limit = ctx.today - dt.timedelta(365 * 5)
    full_data = data_dict[stock].copy()
    full_data['ATH'] = full_data['high'].cummax()
    data = full_data[full_data.index > start_limit].copy()
    if data.empty:
        return None

    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['ATR'] = talib.SMA(data['TR'], 125)
    data['EMA_200'] = talib.EMA(data['close'], 200)

    current_high = data['high'].iloc[-1]
    runs = _ath_runs(data)
    most_recent_long_run = _find_recent_long_run(runs, data)
    if most_recent_long_run is None:
        return None

    condition_1 = data['ATH'].iloc[-1] >= most_recent_long_run['value']
    run_end = pd.Timestamp(most_recent_long_run['end'])
    today_ts = pd.Timestamp(ctx.today)
    condition_2 = today_ts - run_end <= dt.timedelta(days=30)
    condition_3 = current_high > data['ATH'].iloc[-1] * min(0.85, (1 - 4 * data['ATR'].iloc[-1]))
    condition_4 = data['close'].iloc[-2:].max() >= data['EMA_200'].iloc[-1]
    condition_5 = data['ATR'].iloc[-1] >= 0.036

    if condition_1 and condition_2 and condition_3 and condition_4 and condition_5:
        return stock
    return None


def making_new_ATH_low_ATR(stock, data_dict, ctx=None):
    ctx = _ctx(ctx)
    start_limit = ctx.today - dt.timedelta(365 * 5)
    full_data = data_dict[stock].copy()
    full_data['ATH'] = full_data['high'].cummax()
    data = full_data[full_data.index > start_limit].copy()
    if data.empty:
        return None

    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['ATR'] = talib.SMA(data['TR'], 125)
    data['EMA_200'] = talib.EMA(data['close'], 200)

    current_high = data['high'].iloc[-1]
    runs = _ath_runs(data)
    most_recent_long_run = _find_recent_long_run(runs, data)
    if most_recent_long_run is None:
        return None

    condition_1 = data['ATH'].iloc[-1] >= most_recent_long_run['value']
    run_end = pd.Timestamp(most_recent_long_run['end'])
    today_ts = pd.Timestamp(ctx.today)
    condition_2 = today_ts - run_end <= dt.timedelta(days=30)
    condition_3 = current_high > data['ATH'].iloc[-1] * min(0.85, (1 - 4 * data['ATR'].iloc[-1]))
    condition_4 = data['close'].iloc[-2:].max() >= data['EMA_200'].iloc[-1]
    condition_5 = data['ATR'].iloc[-1] < 0.036

    if condition_1 and condition_2 and condition_3 and condition_4 and condition_5:
        return stock
    return None


def smooth_stocks(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    data['ATH'] = data['high'].cummax()
    data['EMA_200'] = talib.EMA(data['close'], 200)

    condition = data['close'].iloc[-1] > max(data['EMA_200'].iloc[-1], data['ATH'].iloc[-1] * 0.70)

    if data.shape[0] >= 400 and condition:
        if data['low'].min() != 0 and not np.isnan(data['low'].min()):
            data['turnover'] = data['close'] * data['volume']
            data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
            data['ATR'] = talib.SMA(data['TR'], 125)
            data['ATR_c'] = data['ATR'] * data['close']
            data['range'] = (data['high'] - data['low']) / data['close']
            data['tail'] = (data.apply(lambda x: min(x['open'], x['close']), axis=1) - data['low']) / data['close']
            data = data.iloc[-300:]
            range_ = (data['high'].max() - data['low'].min()) / data['low'].min()
            return {
                'stock': stock,
                "DR_to_RR_>_4%": (data['range'] >= range_ * 4 / 100).mean() * 100,
                "DR_to_RR_>_5%": (data['range'] >= range_ * 5 / 100).mean() * 100,
                "DR_to_RR_>_6%": (data['range'] >= range_ * 6 / 100).mean() * 100,
            }
    return None


def basing_stocks(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    if data.shape[0] <= 250:
        return None

    volume_cond_1 = data.iloc[-100:]['volume'].quantile(0.25) > 50000
    volume_cond_2 = data.iloc[-100:]['volume'].quantile(0.5) > 100000
    penny_stock_cond = data['close'].iloc[-100:].min() >= 10

    lookback_period = 350
    lookback_data = data.iloc[-lookback_period:]

    lookback_high = lookback_data['high'].max()
    lookback_high_date = lookback_data.index[lookback_data['high'] == lookback_high][0]

    base_data = lookback_data[lookback_data.index >= lookback_high_date].copy()
    base_low = base_data['low'].min()
    base_low_date = base_data.index[base_data['low'] == base_low][0]

    pre_base_data = lookback_data[lookback_data.index < lookback_high_date].copy()
    pre_base_low = pre_base_data['low'].min()
    pre_base_low_date = pre_base_data.index[pre_base_data['low'] == pre_base_low][0]

    pre_base_low_to_high_time = (lookback_high_date - pre_base_low_date).days
    high_to_base_low_time = (base_low_date - lookback_high_date).days
    high_to_present_time = (lookback_data.index[-1] - lookback_high_date).days

    prebase_low_to_lookback_high_perc = (lookback_high - pre_base_low) / pre_base_low * 100
    base_dd = (lookback_high - base_low) / lookback_high * 100

    condition_1 = high_to_base_low_time > 20
    condition_2 = high_to_present_time > 60
    condition_3 = 550 > pre_base_low_to_high_time > 120
    condition_4 = 10 < base_dd < 40
    condition_5 = prebase_low_to_lookback_high_perc > 100
    condition_6 = base_low > pre_base_low + (lookback_high - pre_base_low) * 0.4
    condition_7 = data[data.index > base_low_date].iloc[-5:]['close'].max() > base_low + (lookback_high - base_low) / 3

    if all([condition_1, condition_2, condition_3, condition_4, condition_5,
            condition_6, condition_7, volume_cond_1, volume_cond_2, penny_stock_cond]):
        return stock
    return None


def high_tight_flag(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()

    if data.shape[0] < 105:
        return None

    lookback = data.iloc[-105:]
    highs = lookback['high'].values
    lows = lookback['low'].values

    # Pole foot: lowest low in window, later-bar-wins on ties.
    foot_idx = len(lows) - 1 - lows[::-1].argmin()
    pole_low_initial = lows[foot_idx]

    # Pole top: highest high between foot and today, later-bar-wins.
    high_after_foot = highs[foot_idx:]
    top_offset = len(high_after_foot) - 1 - high_after_foot[::-1].argmax()
    top_idx = foot_idx + top_offset
    pole_high = highs[top_idx]

    flag_low = lows[top_idx:].min()

    running_high = highs[foot_idx]
    lowest_since_hi = lows[foot_idx]
    pole_low = pole_low_initial
    pole_low_idx = foot_idx
    price_diff = running_high / lowest_since_hi

    for i in range(foot_idx, top_idx + 1):
        if highs[i] >= running_high:
            running_high = highs[i]
            lowest_since_hi = lows[i]
        else:
            lowest_since_hi = min(lowest_since_hi, lows[i])
            if lows[i] <= pole_low:
                pole_low = lows[i]
                pole_low_idx = i

        price_diff = max(price_diff, running_high / lowest_since_hi)

        threshold = (pole_high / pole_low) ** (1 / 3)
        if price_diff > threshold:
            running_high = highs[i]
            lowest_since_hi = lows[i]
            pole_low = lows[i]
            pole_low_idx = i
            price_diff = running_high / lowest_since_hi

    last_idx = len(lookback) - 1

    # Flavor B "hold" check: in the last HOLD_WINDOW bars (which may span the
    # late pole and the flag), every low must be at or above flag_low
    # (unbroken floor) AND at least one bar's low must come within
    # HOLD_PROXIMITY_PCT of flag_low (the floor was actually tested, not just
    # coincidentally above a rising pole).
    HOLD_WINDOW = 10
    HOLD_PROXIMITY_PCT = 0.03

    window_lows = lows[-HOLD_WINDOW:]
    floor_unbroken = bool((window_lows >= flag_low).all())
    floor_tested = bool((window_lows <= flag_low * (1 + HOLD_PROXIMITY_PCT)).any())

    cond_1 = flag_low > pole_high / (pole_high / pole_low) ** (1 / 3)
    cond_2 = (last_idx - top_idx) <= 35
    cond_3 = (pole_high - pole_low) / pole_low > 0.6
    cond_4 = (last_idx - top_idx) >= 5 or (floor_unbroken and floor_tested)

    if not (cond_1 and cond_2 and cond_3 and cond_4):
        return None

    pole_gain_pct = (pole_high - pole_low) / pole_low * 100
    tier = "low" if pole_gain_pct <= 80 else "high"
    return {
        "stock": stock,
        "pole_low_date": lookback.index[pole_low_idx],
        "pole_high_date": lookback.index[top_idx],
        "flag_end_date": lookback.index[-1],
        "tier": tier,
        "pole_gain_pct": float(pole_gain_pct),
    }


def weekly_atr_contraction(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    data['ATR_5'] = talib.ATR(data['high'], data['low'], data['close'], 5)
    data['ATR_10'] = talib.ATR(data['high'], data['low'], data['close'], 10)

    if data.shape[0] <= 250:
        return None

    volume_cond_1 = data.iloc[-100:]['volume'].quantile(0.25) > 50000
    volume_cond_2 = data.iloc[-100:]['volume'].quantile(0.5) > 100000
    penny_stock_cond = data['close'].iloc[-100:].min() >= 10

    condition_1 = data['ATR_5'].iloc[-1] <= data['ATR_5'].iloc[-100:].quantile(0.1)
    condition_2 = data['ATR_10'].iloc[-1] <= data['ATR_10'].iloc[-100:].quantile(0.1)

    if all([any([condition_1, condition_2]), volume_cond_1, volume_cond_2, penny_stock_cond]):
        return stock
    return None


def ema_contraction(stock, data_dict, ctx=None):
    """Always returns a dict (no filter condition). Harness adds error handling."""
    data = data_dict[stock].copy()
    data['ATR_20'] = talib.ATR(data['high'], data['low'], data['close'], 20)
    data['EMA_20'] = talib.EMA(data['close'], 20)
    data['EMA_50'] = talib.EMA(data['close'], 50)
    data['EMA_100'] = talib.EMA(data['close'], 100)
    data['EMA_200'] = talib.EMA(data['close'], 200)

    emas = data.iloc[-1, -4:]
    contraction = (emas.max() - emas.min()) / emas.mean() * 100
    contraction_atr = (emas.max() - emas.min()) / data['ATR_20'].iloc[-1]

    return {'stock': stock, 'contraction': contraction, 'contraction_atr': contraction_atr}


def stocks_above_200(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    data['EMA_200'] = talib.EMA(data['close'], 200)

    if data['close'].iloc[-1] > data['EMA_200'].iloc[-1]:
        return stock
    return None


def relative_strength_score(stock, data_dict, ctx=None):
    """
    IBD-style raw Relative Strength score.
    RS_raw = 2*(C/C[63]) + (C/C[126]) + (C/C[189]) + (C/C[252])
    Returns None if fewer than 253 bars (need close[-253] for the 252-day ratio).
    Percentile ranking to 1-99 is done by the caller across the full universe.
    """
    data = data_dict[stock]
    if data.shape[0] < 253:
        return None
    close = data['close']
    c0 = close.iloc[-1]
    c63 = close.iloc[-64]
    c126 = close.iloc[-127]
    c189 = close.iloc[-190]
    c252 = close.iloc[-253]
    if c63 <= 0 or c126 <= 0 or c189 <= 0 or c252 <= 0:
        return None
    rs_raw = 2 * (c0 / c63) + (c0 / c126) + (c0 / c189) + (c0 / c252)
    return {'stock': stock, 'rs_raw': float(rs_raw)}


def relative_strength_benchmark_pivot_score(stock, data_dict, ctx=None):
    """RS anchored to a benchmark index's most recent confirmed pivot high.

    Reads the abstract (pivot_date, benchmark_close_at_pivot) anchor from
    `ctx.rs_anchor` (set by the repo). Per-stock return is
    `close_today / close_at_or_before_pivot_date - 1`. The 'or before' lookup
    handles IPOs, halts and exchange holidays where the stock has no bar on
    the exact pivot session.

    Percentile ranking to 1-99 is done by the caller across the universe.
    """
    ctx = _ctx(ctx)
    if ctx.rs_anchor is None:
        return None
    pivot_date, _ = ctx.rs_anchor
    data = data_dict[stock]
    on_or_before = data.index[data.index <= pivot_date]
    if len(on_or_before) == 0:
        return None
    anchor_close = float(data.at[on_or_before[-1], 'close'])
    if anchor_close <= 0:
        return None
    today_close = float(data['close'].iloc[-1])
    return {'stock': stock, 'return_pct': (today_close / anchor_close - 1.0) * 100.0}


def episodic_pivots(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['TR'] = data['TR'].rolling(window=132, min_periods=132).quantile(0.5)
    data['vol_sma'] = talib.SMA(data['volume'], 50)

    a = (data['open'].iloc[-1] - data['high'].iloc[-2]) > data['TR'].iloc[-1] * data['close'].iloc[-1]
    b = data['close'].iloc[-1] - data['open'].iloc[-1] > data['TR'].iloc[-1] * data['close'].iloc[-1]
    c = data['volume'].iloc[-1] >= 2 * data['vol_sma'].iloc[-1]

    day_range = data['high'].iloc[-1] - data['low'].iloc[-1]
    if day_range <= 0:
        return None
    d = (data['close'].iloc[-1] - data['low'].iloc[-1]) / day_range >= 0.75

    if a and b and c and d:
        return stock
    return None


def ema_uptrend_fn(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['TR'] = (
        data['TR'].rolling(window=132, min_periods=132).quantile(0.5)
        .rolling(window=132, min_periods=132).quantile(0.5)
        * data['close']
    )

    data['EMA_22'] = talib.EMA(data['close'], 22)
    data['EMA_50'] = talib.EMA(data['close'], 50)
    data['EMA_65'] = talib.EMA(data['close'], 65)
    data['EMA_100'] = talib.EMA(data['close'], 100)
    data['EMA_200'] = talib.EMA(data['close'], 200)

    condition_1 = data['EMA_50'].iloc[-1] > data['EMA_50'].iloc[-23] + data['TR'].iloc[-1] * 0.5
    condition_2 = data['EMA_100'].iloc[-1] > data['EMA_100'].iloc[-23] + data['TR'].iloc[-1] * 0.5
    condition_3 = data['EMA_200'].iloc[-1] > data['EMA_200'].iloc[-23] + data['TR'].iloc[-1] * 0.5

    condition_4 = data['EMA_50'].iloc[-23:].max() < data['EMA_50'].iloc[-1] + (data['EMA_50'].iloc[-1] - data['EMA_50'].iloc[-23]) * 0.5
    condition_5 = data['EMA_100'].iloc[-23:].max() < data['EMA_100'].iloc[-1] + (data['EMA_100'].iloc[-1] - data['EMA_100'].iloc[-23]) * 0.5
    condition_6 = data['EMA_200'].iloc[-23:].max() < data['EMA_200'].iloc[-1] + (data['EMA_200'].iloc[-1] - data['EMA_200'].iloc[-23]) * 0.5

    condition_7 = (data['EMA_22'] - data['EMA_50']).iloc[-23:].min() > 0
    condition_8 = (data['EMA_50'] - data['EMA_65']).iloc[-23:].min() > 0
    condition_9 = (data['EMA_65'] - data['EMA_100']).iloc[-23:].min() > 0

    if all([condition_1, condition_2, condition_3, condition_4, condition_5, condition_6]) or all([condition_7, condition_8, condition_9]):
        return stock
    return None


def slow_stocks(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['25_TR'] = data['TR'].rolling(window=132, min_periods=132).quantile(0.25).rolling(window=132, min_periods=132).quantile(0.5)
    data['50_TR'] = data['TR'].rolling(window=132, min_periods=132).quantile(0.5).rolling(window=132, min_periods=132).quantile(0.5)

    if data['25_TR'].iloc[-1] < 0.02 and data['50_TR'].iloc[-1] < 0.03:
        return stock
    return None


def fast_stocks(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['25_TR'] = data['TR'].rolling(window=132, min_periods=132).quantile(0.25).rolling(window=132, min_periods=132).quantile(0.5)
    data['50_TR'] = data['TR'].rolling(window=132, min_periods=132).quantile(0.5).rolling(window=132, min_periods=132).quantile(0.5)

    data['EMA_22'] = talib.EMA(data['close'], 22)
    data['EMA_200'] = talib.EMA(data['close'], 200)

    volume_cond = data.iloc[-100:]['volume'].quantile(0.25) > 100000
    uptrend = data['close'].iloc[-3:].max() > data['EMA_200'].iloc[-1] or data['EMA_22'].iloc[-1] > data['EMA_200'].iloc[-1]
    volatility = data['25_TR'].iloc[-1] > 0.025 and data['50_TR'].iloc[-1] > 0.04

    if volatility and uptrend and volume_cond:
        return stock
    return None


def launchpad_stocks(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['ATR'] = talib.SMA(data['TR'], 132)

    data['EMA_22'] = talib.EMA(data['close'], 22)
    data['EMA_50'] = talib.EMA(data['close'], 50)
    data['EMA_65'] = talib.EMA(data['close'], 65)
    data['EMA_100'] = talib.EMA(data['close'], 100)
    data['EMA_200'] = talib.EMA(data['close'], 200)

    launch_pad = (
        max(data['EMA_22'].iloc[-1], data['EMA_50'].iloc[-1], data['EMA_65'].iloc[-1], data['EMA_100'].iloc[-1])
        - min(data['EMA_22'].iloc[-1], data['EMA_50'].iloc[-1], data['EMA_65'].iloc[-1], data['EMA_100'].iloc[-1])
    ) <= data['ATR'].iloc[-1] * data['close'].iloc[-1] * 0.5

    volume_cond = data.iloc[-100:]['volume'].quantile(0.25) > 100000
    uptrend = data['close'].iloc[-3:].max() > data['EMA_200'].iloc[-1] or data['EMA_22'].iloc[-1] > data['EMA_200'].iloc[-1]
    volatility = data['ATR'].iloc[-1] > 0.04

    e22, e50, e65, e100 = (
        data['EMA_22'].iloc[-1], data['EMA_50'].iloc[-1],
        data['EMA_65'].iloc[-1], data['EMA_100'].iloc[-1],
    )
    stacking = e22 > e50 > e65 > e100

    rising = (
        data['EMA_22'].iloc[-1] > data['EMA_22'].iloc[-22]
        and data['EMA_50'].iloc[-1] > data['EMA_50'].iloc[-22]
        and data['EMA_65'].iloc[-1] > data['EMA_65'].iloc[-22]
        and data['EMA_100'].iloc[-1] > data['EMA_100'].iloc[-22]
    )

    if data.shape[0] < 200:
        return None

    prior_advance = data['close'].iloc[-1] >= 1.30 * data['close'].iloc[-60:].min()

    if launch_pad and volatility and uptrend and volume_cond and stacking and rising and prior_advance:
        return stock
    return None


def vcp(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    data['EMA_200'] = talib.EMA(data['close'], 200)
    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['ATR_c'] = talib.SMA(data['TR'], 132) * data['close']
    atr = data['ATR_c'].iloc[-1]

    temp = data.iloc[-31:].copy()
    temp['updays'] = temp['close'] > temp['open']
    upVol_avg = temp[temp['updays']]['volume'].mean()
    downVol_avg = temp[~temp['updays']]['volume'].mean()

    if data['close'].iloc[-5:].max() <= data['EMA_200'].iloc[-1]:
        return None

    for a in range(5, 10):
        for b in [a + 10, a + 15]:
            for c in [b + 15, b + 20]:
                a_high = data.iloc[-a:]['high'].max()
                a_low = data.iloc[-a:]['low'].min()
                b_high = data.iloc[-b:]['high'].max()
                b_low = data.iloc[-b:]['low'].min()
                c_high = data.iloc[-c:]['high'].max()
                c_low = data.iloc[-c:]['low'].min()

                condition_2 = a_high < b_high < c_high and a_low > b_low > c_low
                condition_3 = ((c_high - c_low) > (a_high - a_low) + atr) or ((b_high - b_low) > (a_high - a_low) + atr)
                condition_4 = data['low'].iloc[-90:-30].min() < b_low - atr * 5
                vol_condition = upVol_avg / downVol_avg >= 1.5

                if condition_2 and condition_3 and condition_4 and vol_condition:
                    return stock
    return None


def minervini_vcp(stock, data_dict, ctx=None):
    """Canonical Minervini VCP detection.

    Requires the Trend Template (7 single-stock criteria) and a base of
    2-6 progressively tighter contractions, with the final contraction
    <= 5% and total base depth <= 30%.
    """
    data = data_dict[stock].copy()
    if data.shape[0] < 252:
        return None

    data['SMA_50'] = talib.SMA(data['close'], 50)
    data['SMA_150'] = talib.SMA(data['close'], 150)
    data['SMA_200'] = talib.SMA(data['close'], 200)
    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['ATR'] = talib.SMA(data['TR'], 132)

    close_last = data['close'].iloc[-1]
    sma50_last = data['SMA_50'].iloc[-1]
    sma150_last = data['SMA_150'].iloc[-1]
    sma200_last = data['SMA_200'].iloc[-1]

    if not (close_last > sma50_last):
        return None
    if not (close_last > sma150_last):
        return None
    if not (close_last > sma200_last):
        return None
    if not (sma50_last > sma150_last > sma200_last):
        return None
    if not (data['SMA_200'].iloc[-1] > data['SMA_200'].iloc[-21]):
        return None
    if not (close_last >= 1.30 * data['low'].iloc[-252:].min()):
        return None
    if not (close_last >= 0.75 * data['high'].iloc[-252:].max()):
        return None

    window = data.iloc[-150:]
    pivots = _zigzag_pivots(
        highs=window['high'].values,
        lows=window['low'].values,
        atr_pct=window['ATR'].values,
        closes=window['close'].values,
        k=3.0,
    )

    if len(pivots) < 4:
        return None
    confirmed = pivots[:-1]
    if len(confirmed) < 3:
        return None

    if confirmed[-1][2] != 'L':
        return None

    # Extract (H, L) contraction pairs by walking back from the newest L.
    # `_zigzag_pivots` guarantees alternation, so an H always sits immediately
    # before each L. Stop when the alternation pattern breaks or pivots run out.
    pairs = []
    i = len(confirmed) - 1
    while i >= 1 and confirmed[i][2] == 'L' and confirmed[i - 1][2] == 'H':
        pairs.append((confirmed[i - 1], confirmed[i]))
        i -= 2
    pairs.reverse()  # oldest → newest

    if len(pairs) < 2 or len(pairs) > 6:
        return None

    depths = [(h[1] - l[1]) / h[1] for h, l in pairs]
    for j in range(len(depths) - 1):
        if not (depths[j] > depths[j + 1]):
            return None
    if depths[-1] > 0.05:
        return None

    all_h_prices = [h[1] for h, _ in pairs]
    all_l_prices = [l[1] for _, l in pairs]
    max_h_price = max(all_h_prices)
    min_l_price = min(all_l_prices)
    total_base_depth = (max_h_price - min_l_price) / max_h_price
    if total_base_depth > 0.30:
        return None

    base_start_idx = pairs[0][0][0]
    base_end_idx = pairs[-1][1][0]
    base_length = base_end_idx - base_start_idx
    if not (25 <= base_length <= 130):
        return None

    chain_prices = []
    for h, l in pairs:
        chain_prices.append(float(h[1]))
        chain_prices.append(float(l[1]))

    return {
        'stock': stock,
        'n_contractions': len(pairs),
        'contraction_depths_pct': [round(d * 100, 2) for d in depths],
        'pivot_prices': chain_prices,
        'final_contraction_pct': round(depths[-1] * 100, 2),
        'total_base_depth_pct': round(total_base_depth * 100, 2),
        'base_length_days': base_length,
        'base_start_date': window.index[base_start_idx],
        'base_end_date': window.index[base_end_idx],
    }


def checklist(stock, data_dict, ctx=None):
    data = data_dict[stock].copy().sort_index()
    data['ATH'] = data['high'].cummax()
    data['EMA_10'] = talib.EMA(data['close'], 10)
    data['EMA_21'] = talib.EMA(data['close'], 21)
    data['EMA_50'] = talib.EMA(data['close'], 50)
    data['EMA_100'] = talib.EMA(data['close'], 100)
    data['EMA_200'] = talib.EMA(data['close'], 200)
    data['SMA_EMA_21'] = talib.SMA(data['EMA_21'], 5)
    data['SMA_EMA_50'] = talib.SMA(data['EMA_50'], 10)
    data['SMA_EMA_100'] = talib.SMA(data['EMA_100'], 21)
    data['volume_SMA'] = talib.SMA(data['volume'], 10)

    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1)
    data['ATR'] = talib.SMA(data['TR'] / data['close'], 132)
    data['ATR_c'] = data['ATR'] * data['close']

    temp = data.iloc[-31:].copy()
    temp['updays'] = temp['close'] > temp['open']

    upVol_max = temp[temp['updays']]['volume'].max()
    upVol_avg = temp[temp['updays']]['volume'].mean()
    downVol_max = temp[~temp['updays']]['volume'].max()
    downVol_avg = temp[~temp['updays']]['volume'].mean()

    if data.iloc[-1]['EMA_50'] > data.iloc[-43]['SMA_EMA_50'] + data.iloc[-1]['ATR_c'] * 0.5:
        MT_trend_ = True
    elif data.iloc[-1]['EMA_50'] > data.iloc[-43]['SMA_EMA_50']:
        MT_trend_ = False
    else:
        MT_trend_ = False

    if data.iloc[-1]['EMA_21'] > data.iloc[-1]['EMA_50'] and data.iloc[-1]['EMA_21'] > data.iloc[-1]['EMA_100']:
        S_M_EMA = True
    elif data.iloc[-1]['EMA_21'] > data.iloc[-1]['EMA_50'] or data.iloc[-1]['EMA_21'] > data.iloc[-1]['EMA_100']:
        S_M_EMA = False
    else:
        S_M_EMA = False

    if data.iloc[-1]['EMA_50'] > data.iloc[-1]['EMA_100'] and data.iloc[-1]['EMA_50'] > data.iloc[-1]['EMA_200']:
        M_L_EMA = True
    elif data.iloc[-1]['EMA_50'] > data.iloc[-1]['EMA_100'] or data.iloc[-1]['EMA_50'] > data.iloc[-1]['EMA_200']:
        M_L_EMA = False
    else:
        M_L_EMA = False

    if data.iloc[-1]['close'] > data.iloc[-1]['EMA_50']:
        close_50EMA = True
    elif data.iloc[-1]['close'] > data.iloc[-1]['EMA_50'] - data.iloc[-1]['ATR_c'] * 0.25:
        close_50EMA = False
    else:
        close_50EMA = False

    if data.iloc[-1]['close'] > data.iloc[-1]['EMA_100']:
        close_100EMA = True  # noqa: F841 (computed but not used in final condition — preserved from original)
    elif data.iloc[-1]['close'] > data.iloc[-1]['EMA_100'] - data.iloc[-1]['ATR_c'] * 0.25:
        close_100EMA = False  # noqa: F841
    else:
        close_100EMA = False  # noqa: F841

    cond_1 = upVol_avg * 0.85 > downVol_avg
    cond_2 = upVol_max * 0.85 > downVol_max
    cond_3 = upVol_avg > downVol_avg
    cond_4 = upVol_max > downVol_max
    cond_5 = downVol_max < data.iloc[-1]['volume_SMA'] * 2

    if cond_1 and cond_2 and cond_5:
        volume_ = True
    elif cond_3 and cond_4:
        volume_ = False
    else:
        volume_ = False

    dd_ATH = (data.iloc[-1]['ATH'] - data.iloc[-1]['close']) / data.iloc[-1]['ATH'] * 100
    if dd_ATH <= 35:
        dd_ATH_ = True
    elif dd_ATH <= 40:
        dd_ATH_ = False
    else:
        dd_ATH_ = False

    if MT_trend_ and S_M_EMA and M_L_EMA and close_50EMA and volume_ and dd_ATH_:
        return stock
    return None


def tight_range(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    data['EMA_200'] = talib.EMA(data['close'], 200)

    if data['close'].iloc[-5:].max() <= data['EMA_200'].iloc[-1]:
        return None

    for i in range(5, 11):
        data[f'{i}_day_range_c'] = (
            data['high'].rolling(window=i, min_periods=i).max()
            - data['low'].rolling(window=i, min_periods=i).min()
        ) / data['close'].rolling(window=i, min_periods=i).mean()

    for i in range(5, 11):
        if data[f'{i}_day_range_c'].iloc[-1] <= data[f'{i}_day_range_c'].iloc[-400:].quantile(0.01):
            return stock
    return None


def high_atr(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    data['turnover'] = data['close'] * data['volume']
    data['EMA_200'] = talib.EMA(data['close'], 200)
    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['ATR'] = talib.SMA(data['TR'], 132)
    data['TR_50'] = data['TR'].rolling(window=132, min_periods=132).quantile(0.5)

    trend_condition = (data['close'] > data['EMA_200']).iloc[-120:].sum() >= 80
    turnover_cond = data.iloc[-100:]['turnover'].quantile(0.5) > 10000000
    volume_cond = data.iloc[-100:]['volume'].quantile(0.5) > 75000
    volatility_cond = data['TR_50'].iloc[-1] > 0.04
    penny_stock_cond = data['close'].iloc[-100:].min() >= 10

    if trend_condition and turnover_cond and volume_cond and volatility_cond and penny_stock_cond:
        return stock
    return None


def contracting_stocks(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['ATR_c'] = talib.SMA(data['TR'], 132) * data['close']
    data['EMA_200'] = talib.EMA(data['close'], 200)

    condition = data['close'].iloc[-5:].max() > data['EMA_200'].iloc[-1]

    data = data[::-1].copy()
    data['high_max'] = data['high'].cummax()
    data['low_min'] = data['low'].cummin()
    data['range'] = data['high_max'] - data['low_min']
    data['VC_1'] = data['range'] <= data['ATR_c']
    data['VC_2'] = data['range'] <= data['ATR_c'] * 1.5

    if data['VC_1'].all():
        return None
    n_days_1 = data.index.get_loc(data[data['VC_1'] == False].index[0])
    n_days_2 = data.shape[0] if data['VC_2'].all() else data.index.get_loc(data[data['VC_2'] == False].index[0])

    if condition:
        return (stock, n_days_1, n_days_2)
    return None


def bullish_confirmation(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['ATR'] = talib.SMA(data['TR'], 132)
    data['EMA_200'] = talib.EMA(data['close'], 200)
    data['volume_MA'] = talib.SMA(data['volume'], 42)

    if data['close'].iloc[-5:].max() <= data['EMA_200'].iloc[-1]:
        return None

    data['bullish_candle'] = (
        (((data['close'] - data['open']) / data['open']) / data['ATR'] >= 1.5)
        & (data['close'] > data['open'])
        & (data['volume'] >= data['volume_MA'] * 2)
    )

    if not data['bullish_candle'].any():
        return None
    bullish_candle = data[data['bullish_candle']].iloc[-1]
    iloc = data.index.get_loc(bullish_candle.name)
    n_days_ago = data.shape[0] - iloc
    _after_candle = data.iloc[iloc + 1:].copy()

    condition_1 = _after_candle['low'].min() > (bullish_candle['close'] + bullish_candle['open']) * 0.5
    condition_2 = (bullish_candle['close'] - bullish_candle['open']) > (bullish_candle['high'] - bullish_candle['low']) * 0.66
    condition_3 = _after_candle['high'].max() < bullish_candle['close'] * (1 + bullish_candle['ATR'] * 2)
    condition_4 = (_after_candle['high'] - _after_candle['low']).max() < bullish_candle['close'] * bullish_candle['ATR']
    condition_5 = (_after_candle['high'] - _after_candle['low']).max() < (bullish_candle['high'] - bullish_candle['low']) * 0.5
    condition_6 = 2 < n_days_ago <= 5

    if condition_1 and condition_2 and condition_3 and (condition_4 or condition_5) and condition_6:
        return stock
    return None


_TOP_MOVERS_LOOKBACKS = {'is_1d': 1, 'is_1w': 5, 'is_1m': 22, 'is_3m': 63, 'is_1y': 252}

def top_movers(stock, data_dict, ctx=None):
    """Compute ATR-normalized move score across multiple timeframes. Returns dict with period flags."""
    data = data_dict[stock].copy()
    if data.shape[0] < 260:
        return None
    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['ATR'] = talib.SMA(data['TR'], 132)
    data['EMA_200'] = talib.EMA(data['close'], 200)

    if data['close'].iloc[-5:].max() <= data['EMA_200'].iloc[-1]:
        return None

    atr = data['ATR'].iloc[-1]
    if atr <= 0:
        return None

    result = {'stock': stock}
    for col, lookback in _TOP_MOVERS_LOOKBACKS.items():
        if col == 'is_1d':
            score = (data['close'].iloc[-1] - data['open'].iloc[-1]) / data['open'].iloc[-1] / atr
        else:
            score = (data['close'].iloc[-1] - data['close'].iloc[-lookback]) / data['close'].iloc[-lookback] / atr
        result[col] = score

    return result


# ── Base breakout parameters ──
_BASE_BREAKOUT_LEFT_BARS = 64        # ~3 months
_BASE_BREAKOUT_RIGHT_BARS = 21       # ~1 month (also confirmation lag)
_BASE_BREAKOUT_RECENT_WINDOW = 5     # ~1 week breakout window
_BASE_BREAKOUT_LOOKBACK_DAYS = 1260  # ~5 years


_TURNOVER_LOOKBACKS = {'is_3m': 63, 'is_6m': 126, 'is_1y': 252, 'is_ath': 500}

def highest_turnover(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['ATR'] = talib.SMA(data['TR'], 132)
    data['EMA_21'] = talib.EMA(data['close'], 21)
    data['EMA_200'] = talib.EMA(data['close'], 200)
    data['volume_MA'] = talib.SMA(data['volume'], 42)
    data['turnover'] = data['volume'] * data[['open', 'high', 'low', 'close']].mean(axis=1)

    last_candle = data.iloc[-1]
    condition_1 = (last_candle['EMA_21'] > last_candle['EMA_200']) or (data['high'].iloc[-5:].max() > last_candle['EMA_200'])

    if not condition_1:
        return None

    recent_max = data['turnover'].iloc[-5:].max()
    result = {'stock': stock}
    any_match = False
    for col, lookback in _TURNOVER_LOOKBACKS.items():
        qualifies = recent_max > data['turnover'].iloc[-lookback:-5].max()
        result[col] = int(qualifies)
        if qualifies:
            any_match = True

    return result if any_match else None


def ipo(stock, data_dict, ctx=None):
    """Returns a dict for IPO stocks (<=250 bars), None otherwise."""
    data = data_dict[stock].copy()
    if data.shape[0] > 250:
        return None

    IPO_high = data['high'].iloc[:63].max()
    near_IPO_high = IPO_high * 1.25 >= data['close'].iloc[-1] >= IPO_high * 0.85

    return {
        'stock': stock,
        'near_ipo_high': near_IPO_high,
        'ipo_6m': 21 <= data.shape[0] <= 125,
        'ipo_12m': 125 < data.shape[0] <= 250,
    }


def intraday(stock, data_dict, ctx=None):
    volatility_threshold = 0.05
    volume_threshold = 20

    data = data_dict[stock].copy()
    if data.shape[0] < 250:
        return None

    data['turnover'] = data['close'] * data['volume']
    data['DR'] = (data['high'] - data['low']) / data['close']
    data['ADR'] = talib.SMA(data['DR'], 125)

    volatility_condition = data['ADR'].iloc[-1] >= volatility_threshold
    volume_condition = talib.SMA(data['volume'], 100).iloc[-1] / 25 / (2000 / (data['close'].iloc[-1] * 0.01)) >= volume_threshold

    if volatility_condition and volume_condition:
        return stock
    return None


def gapups(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['ATR'] = talib.SMA(data['TR'], 132)
    data['EMA_21'] = talib.EMA(data['close'], 21)
    data['EMA_200'] = talib.EMA(data['close'], 200)

    if data['close'].iloc[-5:].max() <= data['EMA_200'].iloc[-1]:
        return None

    data['gapup'] = data['low'] > data['high'].shift() * 1.01
    if not data['gapup'].any():
        return None
    gapup_ix = data.index.get_loc(data[data['gapup']].index[-1])
    condition_1 = data.shape[0] - gapup_ix <= 5
    condition_2 = data['low'].iloc[gapup_ix:].min() >= data['low'].iloc[gapup_ix]

    if condition_1 and condition_2:
        return stock
    return None


def base_breakout(stock, data_dict, ctx=None):
    """Flag stocks whose recent high cleared a structural pivot ceiling.

    A pivot is a bar whose high strictly exceeds the LEFT_BARS highs immediately
    before it AND the RIGHT_BARS highs immediately after it (Pine-style swing
    high with a wide left window).

    For each pivot inside the LOOKBACK_DAYS window, the first subsequent bar
    whose high exceeds the pivot's high is the breakout bar. The stock is
    flagged when that breakout bar falls inside the RECENT_WINDOW (last N bars).
    When multiple ceilings break inside the recent window, the one with the
    highest pivot_price wins.

    Returns (stock, base_days, pivot_date, pivot_price) or None.
        base_days = breakout_idx - pivot_idx (>= RIGHT_BARS + 1).
    """
    data = data_dict[stock]
    highs = data['high'].values
    n = len(highs)
    left = _BASE_BREAKOUT_LEFT_BARS
    right = _BASE_BREAKOUT_RIGHT_BARS
    recent = _BASE_BREAKOUT_RECENT_WINDOW

    if n < left + right + 2:
        return None

    win_size = left + 1 + right
    windows = sliding_window_view(highs, win_size)
    center = windows[:, left]
    left_max = windows[:, :left].max(axis=1)
    right_max = windows[:, left + 1:].max(axis=1)
    is_pivot = (center > left_max) & (center > right_max)
    pivot_indices = np.where(is_pivot)[0] + left

    start = max(0, n - _BASE_BREAKOUT_LOOKBACK_DAYS)
    pivot_indices = pivot_indices[pivot_indices >= start]

    breakout_threshold = n - recent
    best = None
    for p in pivot_indices:
        pivot_price = float(highs[p])
        after = highs[p + 1:]
        idxs = np.where(after > pivot_price)[0]
        if idxs.size == 0:
            continue
        breakout_idx = p + 1 + int(idxs[0])
        if breakout_idx < breakout_threshold:
            continue
        if best is None or pivot_price > best[0]:
            best = (pivot_price, int(p), breakout_idx)

    if best is None:
        return None

    pivot_price, pivot_idx, breakout_idx = best
    base_length = breakout_idx - pivot_idx
    return (stock, int(base_length), data.index[pivot_idx], pivot_price)


def cup_with_handle(stock, data_dict, ctx=None):
    """Canonical O'Neil Cup-with-Handle, setup-ready trigger.

    Requires a prior 30% advance, a valid cup (12-33% deep, 7-65 weeks),
    a valid handle (1-4 weeks, <=12% deep, in upper half of cup), and
    today's close within 5% below the pivot (handle high) without having
    already broken it.
    """
    data = data_dict[stock].copy()
    if data.shape[0] < 252:
        return None

    data['EMA_200'] = talib.EMA(data['close'], 200)
    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['ATR'] = talib.SMA(data['TR'], 132)

    if data.shape[0] < 325:
        return None
    prior_window = data['close'].iloc[-325:-65]
    if prior_window.empty or not (data['close'].iloc[-1] >= 1.30 * prior_window.min()):
        return None

    if not (data['close'].iloc[-1] > data['EMA_200'].iloc[-1]):
        return None

    window = data.iloc[-350:]
    pivots = _zigzag_pivots(
        highs=window['high'].values,
        lows=window['low'].values,
        atr_pct=window['ATR'].values,
        closes=window['close'].values,
        k=1.5,
    )

    if len(pivots) < 4:
        return None
    confirmed = pivots[:-1]
    if len(confirmed) < 3:
        return None

    triplet = None
    for idx in range(len(confirmed) - 1, 1, -1):
        if (confirmed[idx][2] == 'H' and confirmed[idx - 1][2] == 'L'
                and confirmed[idx - 2][2] == 'H'):
            triplet = (confirmed[idx - 2], confirmed[idx - 1], confirmed[idx])
            break

    if triplet is None:
        return None

    left_rim, cup_low, right_rim = triplet
    left_rim_idx, left_rim_price, _ = left_rim
    cup_low_idx, cup_low_price, _ = cup_low
    right_rim_idx, right_rim_price, _ = right_rim

    cup_length = right_rim_idx - left_rim_idx
    if not (35 <= cup_length <= 325):
        return None

    cup_high = max(left_rim_price, right_rim_price)
    cup_depth = (cup_high - cup_low_price) / cup_high
    if not (0.12 <= cup_depth <= 0.33):
        return None

    if abs(left_rim_price - right_rim_price) / cup_high > 0.05:
        return None

    highs_arr = window['high'].values
    lows_arr = window['low'].values
    closes_arr = window['close'].values
    n = len(window)

    if right_rim_idx + 1 >= n:
        return None
    handle_low_offset = int(np.argmin(lows_arr[right_rim_idx + 1:]))
    handle_low_idx = right_rim_idx + 1 + handle_low_offset
    handle_low_price = float(lows_arr[handle_low_idx])

    pivot = float(np.max(highs_arr[right_rim_idx:handle_low_idx + 1]))

    handle_length = n - right_rim_idx
    if not (5 <= handle_length <= 20):
        return None

    if pivot <= 0:
        return None
    handle_depth = (pivot - handle_low_price) / pivot
    if handle_depth > 0.12:
        return None

    if not (handle_low_price > (cup_high + cup_low_price) / 2):
        return None

    if not (handle_low_idx - right_rim_idx >= 5):
        return None
    if not (n - 1 - handle_low_idx >= 3):
        return None

    if handle_low_idx + 1 < n and closes_arr[handle_low_idx + 1:n].max() > pivot:
        return None

    distance_to_pivot = (pivot - closes_arr[-1]) / pivot
    if not (0 <= distance_to_pivot <= 0.05):
        return None

    return {
        'stock': stock,
        'cup_length_days': int(cup_length),
        'cup_depth_pct': round(cup_depth * 100, 2),
        'handle_length_days': int(handle_length),
        'handle_depth_pct': round(handle_depth * 100, 2),
        'pivot_price': pivot,
        'distance_to_pivot_pct': round(distance_to_pivot * 100, 2),
        'left_rim_price': float(left_rim_price),
        'right_rim_price': float(right_rim_price),
        'cup_low_price': float(cup_low_price),
        'handle_low_price': float(handle_low_price),
        'left_rim_date': window.index[left_rim_idx],
        'cup_low_date': window.index[cup_low_idx],
        'right_rim_date': window.index[right_rim_idx],
        'handle_low_date': window.index[handle_low_idx],
    }


def avg_turnover_ranked(stock, data_dict, ctx=None):
    """Compute mean daily turnover for multiple periods. Requires >= 63 bars."""
    data = data_dict[stock]
    if data.shape[0] < 63:
        return None
    turnover = (data['close'] * data['volume']).values
    result = {'stock': stock, 'quarter': float(np.mean(turnover[-63:]))}
    if len(turnover) >= 125:
        result['6_months'] = float(np.mean(turnover[-125:]))
    if len(turnover) >= 250:
        result['1_year'] = float(np.mean(turnover[-250:]))
    result['all_time'] = float(np.mean(turnover))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Momentum & Indicator scans
# ═══════════════════════════════════════════════════════════════════════════════

def golden_cross(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    if data.shape[0] < 250:
        return None
    data['EMA_50'] = talib.EMA(data['close'], 50)
    data['EMA_200'] = talib.EMA(data['close'], 200)

    if data['close'].iloc[-1] <= data['EMA_50'].iloc[-1]:
        return None
    if data['close'].iloc[-1] <= data['EMA_200'].iloc[-1]:
        return None
    if data['EMA_50'].iloc[-1] <= data['EMA_200'].iloc[-1]:
        return None
    # Recently crossed: EMA50 was <= EMA200 within last 5 bars
    if not any(data['EMA_50'].iloc[i] <= data['EMA_200'].iloc[i] for i in range(-6, -1)):
        return None
    return stock


def macd_bullish_crossover(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    if data.shape[0] < 250:
        return None
    data['EMA_200'] = talib.EMA(data['close'], 200)
    macd, signal, hist = talib.MACD(data['close'], 12, 26, 9)

    if data['close'].iloc[-1] <= data['EMA_200'].iloc[-1]:
        return None
    if macd.iloc[-1] <= signal.iloc[-1]:
        return None
    if not any(macd.iloc[i] <= signal.iloc[i] for i in range(-4, -1)):
        return None
    if hist.iloc[-1] <= 0:
        return None
    return stock


# ═══════════════════════════════════════════════════════════════════════════════
# Candlestick Pattern scans
# ═══════════════════════════════════════════════════════════════════════════════

def bullish_engulfing(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    if data.shape[0] < 250:
        return None
    data['EMA_50'] = talib.EMA(data['close'], 50)
    data['EMA_200'] = talib.EMA(data['close'], 200)
    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['ATR'] = talib.SMA(data['TR'], 125)
    data['vol_sma'] = talib.SMA(data['volume'], 20)

    pattern = talib.CDLENGULFING(data['open'], data['high'], data['low'], data['close'])

    if data['close'].iloc[-1] <= data['EMA_200'].iloc[-1]:
        return None

    for i in range(-3, 0):
        if pattern.iloc[i] > 0:
            near_ema = abs(data['close'].iloc[i] - data['EMA_50'].iloc[i]) <= 1.5 * data['ATR'].iloc[i] * data['close'].iloc[i]
            vol_confirm = data['volume'].iloc[i] > data['vol_sma'].iloc[i]
            if near_ema and vol_confirm:
                return stock
    return None


def hammer(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    if data.shape[0] < 250:
        return None
    data['EMA_50'] = talib.EMA(data['close'], 50)
    data['EMA_100'] = talib.EMA(data['close'], 100)
    data['EMA_200'] = talib.EMA(data['close'], 200)
    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['ATR'] = talib.SMA(data['TR'], 125)

    pattern = talib.CDLHAMMER(data['open'], data['high'], data['low'], data['close'])

    if data['close'].iloc[-1] <= data['EMA_200'].iloc[-1]:
        return None

    for i in range(-3, 0):
        if pattern.iloc[i] > 0:
            low = data['low'].iloc[i]
            atr_dist = data['ATR'].iloc[i] * data['close'].iloc[i]
            near_ema50 = abs(low - data['EMA_50'].iloc[i]) <= atr_dist
            near_ema100 = abs(low - data['EMA_100'].iloc[i]) <= atr_dist
            if near_ema50 or near_ema100:
                return stock
    return None


def morning_star(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    if data.shape[0] < 250:
        return None
    data['EMA_200'] = talib.EMA(data['close'], 200)

    pattern = talib.CDLMORNINGSTAR(data['open'], data['high'], data['low'], data['close'], penetration=0.3)

    if data['close'].iloc[-1] <= data['EMA_200'].iloc[-1]:
        return None

    for i in range(-3, 0):
        if pattern.iloc[i] > 0:
            return stock
    return None


def doji(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    if data.shape[0] < 250:
        return None
    data['EMA_21'] = talib.EMA(data['close'], 21)
    data['EMA_50'] = talib.EMA(data['close'], 50)
    data['EMA_100'] = talib.EMA(data['close'], 100)
    data['EMA_200'] = talib.EMA(data['close'], 200)
    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['ATR'] = talib.SMA(data['TR'], 125)

    pattern = talib.CDLDOJI(data['open'], data['high'], data['low'], data['close'])

    if data['close'].iloc[-1] <= data['EMA_200'].iloc[-1]:
        return None

    for i in range(-2, 0):
        if pattern.iloc[i] != 0:
            close = data['close'].iloc[i]
            atr_dist = data['ATR'].iloc[i] * close
            near_ema = any(
                abs(close - data[ema].iloc[i]) <= atr_dist
                for ema in ['EMA_21', 'EMA_50', 'EMA_100']
            )
            if near_ema:
                return stock
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Volatility scans
# ═══════════════════════════════════════════════════════════════════════════════

def bollinger_squeeze(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    if data.shape[0] < 250:
        return None
    data['EMA_200'] = talib.EMA(data['close'], 200)
    upper, middle, lower = talib.BBANDS(data['close'], 20, 2.0, 2.0)
    data['bandwidth'] = (upper - lower) / middle

    if data['close'].iloc[-1] <= data['EMA_200'].iloc[-1]:
        return None
    if data['bandwidth'].iloc[-1] <= data['bandwidth'].iloc[-125:].quantile(0.10):
        return stock
    return None


def narrow_range(stock, data_dict, ctx=None):
    """NR7: today's range is narrower than each of the prior 6 days."""
    data = data_dict[stock].copy()
    if data.shape[0] < 250:
        return None
    data['EMA_200'] = talib.EMA(data['close'], 200)
    data['day_range'] = data['high'] - data['low']

    if data['close'].iloc[-1] <= data['EMA_200'].iloc[-1]:
        return None
    if data.iloc[-100:]['volume'].quantile(0.5) <= 50000:
        return None

    today_range = data['day_range'].iloc[-1]
    prior_ranges = data['day_range'].iloc[-7:-1]
    if today_range < prior_ranges.min():
        return stock
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Breakout scans
# ═══════════════════════════════════════════════════════════════════════════════

def fifty_two_week_high(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    if data.shape[0] < 252:
        return None
    data['vol_sma'] = talib.SMA(data['volume'], 20)

    high_252 = data['high'].iloc[-252:].max()
    if data['high'].iloc[-1] >= high_252 and data['volume'].iloc[-1] > data['vol_sma'].iloc[-1]:
        return stock
    return None


def volume_breakout(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    if data.shape[0] < 250:
        return None
    data['EMA_200'] = talib.EMA(data['close'], 200)
    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['ATR'] = talib.SMA(data['TR'], 125)
    data['vol_sma'] = talib.SMA(data['volume'], 20)

    if data['close'].iloc[-1] <= data['EMA_200'].iloc[-1]:
        return None

    for i in range(-2, 0):
        vol_spike = data['volume'].iloc[i] > 3 * data['vol_sma'].iloc[i]
        bullish = data['close'].iloc[i] > data['open'].iloc[i]
        significant = (data['close'].iloc[i] - data['open'].iloc[i]) / data['open'].iloc[i] > data['ATR'].iloc[i]
        if vol_spike and bullish and significant:
            return stock
    return None


def consolidation_breakout(stock, data_dict, ctx=None):
    """Breakout from a tight 3-15 day consolidation range in a stacked-EMA uptrend.
    Returns (stock, consolidation_days)."""
    data = data_dict[stock].copy()
    if data.shape[0] < 250:
        return None
    data['EMA_20'] = talib.EMA(data['close'], 20)
    data['EMA_50'] = talib.EMA(data['close'], 50)
    data['EMA_200'] = talib.EMA(data['close'], 200)
    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['ATR'] = talib.SMA(data['TR'], 125)
    data['vol_sma'] = talib.SMA(data['volume'], 20)

    close = data['close'].iloc[-1]
    ema_20 = data['EMA_20'].iloc[-1]
    ema_50 = data['EMA_50'].iloc[-1]
    ema_200 = data['EMA_200'].iloc[-1]
    if not (close > ema_50 and close > ema_200 and ema_20 > ema_50 and ema_50 > ema_200):
        return None

    atr = data['ATR'].iloc[-1] * data['close'].iloc[-1]
    if atr <= 0:
        return None

    best_length = 0
    for length in range(15, 2, -1):
        window = data.iloc[-length - 1:-1]
        range_size = window['high'].max() - window['low'].min()
        tight_threshold = 2.0 * (length / 15.0) ** 0.5 * atr
        if range_size < tight_threshold:
            best_length = length
            break

    if best_length < 3:
        return None

    consol_window = data.iloc[-best_length - 1:-1]
    range_high = consol_window['high'].max()
    if data['close'].iloc[-1] <= range_high:
        return None

    if data['volume'].iloc[-1] <= 1.5 * data['vol_sma'].iloc[-1]:
        return None

    return (stock, best_length)


# ═══════════════════════════════════════════════════════════════════════════════
# Consolidation scans
# ═══════════════════════════════════════════════════════════════════════════════


def consolidation(stock, data_dict, ctx=None):
    """Detect a tight consolidation window using a combined ATR + historical-rank score.

    Returns a dict with marker geometry when a qualifying window exists,
    else None.
    """
    data = data_dict[stock].copy()
    if data.shape[0] < 414:
        return None
    data['EMA_200'] = talib.EMA(data['close'], 200)
    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['ATR'] = talib.SMA(data['TR'], 125)

    if data['close'].iloc[-5:].max() <= data['EMA_200'].iloc[-1]:
        return None

    atr_price = data['ATR'].iloc[-1] * data['close'].iloc[-1]
    if atr_price <= 0:
        return None

    for length in range(15, 4, -1):
        window = data.iloc[-length:]
        w_high = window['high'].max()
        w_low = window['low'].min()
        w_range = w_high - w_low
        ceiling = 2.0 * (length / 15.0) ** 0.5 * atr_price
        score_atr = w_range / ceiling

        roll_range = (
            data['high'].rolling(length).max()
            - data['low'].rolling(length).min()
        )
        roll_mean = data['close'].rolling(length).mean()
        coeff_series = (roll_range / roll_mean).iloc[-400:]
        score_hist = (coeff_series < coeff_series.iloc[-1]).mean()

        score = score_atr + score_hist
        if score < 1.0:
            return {
                'stock': stock,
                'consolidation_days': length,
                'combined_score': float(score),
                'score_atr': float(score_atr),
                'score_hist': float(score_hist),
                'start_date': window.index[0],
                'end_date': window.index[-1],
                'range_high': float(w_high),
                'range_low': float(w_low),
            }

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Volume scans
# ═══════════════════════════════════════════════════════════════════════════════

def volume_dryup(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    if data.shape[0] < 250:
        return None
    data['EMA_21'] = talib.EMA(data['close'], 21)
    data['EMA_65'] = talib.EMA(data['close'], 65)
    data['EMA_200'] = talib.EMA(data['close'], 200)
    data['vol_sma'] = talib.SMA(data['volume'], 50)

    close = data['close'].iloc[-1]
    if close <= data['EMA_200'].iloc[-1]:
        return None

    ema21, ema65 = data['EMA_21'].iloc[-1], data['EMA_65'].iloc[-1]
    if not (min(ema21, ema65) <= close <= max(ema21, ema65)):
        return None

    recent_high = data['high'].iloc[-20:].max()
    if (recent_high - close) / recent_high > 0.15:
        return None

    if data['volume'].iloc[-1] >= 0.5 * data['vol_sma'].iloc[-1]:
        return None

    return stock


def unusual_volume(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    if data.shape[0] < 250:
        return None
    data['EMA_200'] = talib.EMA(data['close'], 200)
    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['ATR'] = talib.SMA(data['TR'], 125)
    data['vol_sma'] = talib.SMA(data['volume'], 20)

    if data['close'].iloc[-1] <= data['EMA_200'].iloc[-1]:
        return None

    for i in range(-2, 0):
        vol_spike = data['volume'].iloc[i] > 2 * data['vol_sma'].iloc[i]
        small_candle = abs(data['close'].iloc[i] - data['open'].iloc[i]) / data['close'].iloc[i] < data['ATR'].iloc[i]
        if vol_spike and small_candle:
            ratio = round(float(data['volume'].iloc[i] / data['vol_sma'].iloc[i]), 1)
            return {'stock': stock, 'volume_ratio': ratio}
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Price Action scans
# ═══════════════════════════════════════════════════════════════════════════════

def inside_day(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    if data.shape[0] < 250:
        return None
    data['EMA_200'] = talib.EMA(data['close'], 200)

    if data['close'].iloc[-1] <= data['EMA_200'].iloc[-1]:
        return None
    if data.iloc[-100:]['volume'].quantile(0.5) <= 50000:
        return None

    if data['high'].iloc[-1] <= data['high'].iloc[-2] and data['low'].iloc[-1] >= data['low'].iloc[-2]:
        return stock
    return None


def pocket_pivot(stock, data_dict, ctx=None):
    data = data_dict[stock].copy()
    if data.shape[0] < 250:
        return None
    data['EMA_21'] = talib.EMA(data['close'], 21)
    data['EMA_50'] = talib.EMA(data['close'], 50)
    data['EMA_200'] = talib.EMA(data['close'], 200)
    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['ATR'] = talib.SMA(data['TR'], 125)

    if data['close'].iloc[-1] <= data['EMA_200'].iloc[-1]:
        return None
    # Must be an up day
    if data['close'].iloc[-1] <= data['open'].iloc[-1]:
        return None

    # Max volume of down days in last 10 bars
    window = data.iloc[-11:-1]
    down_days = window[window['close'] < window['open']]
    if down_days.empty:
        return None
    max_down_vol = down_days['volume'].max()

    # Today's volume exceeds all down-day volume
    if data['volume'].iloc[-1] <= max_down_vol:
        return None

    # Near moving average support
    close = data['close'].iloc[-1]
    atr_dist = 1.5 * data['ATR'].iloc[-1] * close
    near_ema = (
        abs(close - data['EMA_21'].iloc[-1]) <= atr_dist
        or abs(close - data['EMA_50'].iloc[-1]) <= atr_dist
    )
    if not near_ema:
        return None

    return stock


def pullback_to_ema(stock, data_dict, ctx=None):
    """Returns {stock, ema_level} if pulling back to EMA support in an uptrend."""
    data = data_dict[stock].copy()
    if data.shape[0] < 250:
        return None
    data['EMA_21'] = talib.EMA(data['close'], 21)
    data['EMA_50'] = talib.EMA(data['close'], 50)
    data['EMA_100'] = talib.EMA(data['close'], 100)
    data['EMA_200'] = talib.EMA(data['close'], 200)
    data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
    data['ATR'] = talib.SMA(data['TR'], 125)

    # Strong uptrend structure
    if not (data['EMA_50'].iloc[-1] > data['EMA_100'].iloc[-1] > data['EMA_200'].iloc[-1]):
        return None

    # Was higher recently (pulling back)
    if data['close'].iloc[-6:-1].max() <= data['close'].iloc[-1]:
        return None

    close = data['close'].iloc[-1]
    low = data['low'].iloc[-1]
    atr_dist = 0.5 * data['ATR'].iloc[-1] * close

    ema_levels = [('EMA 21', 'EMA_21'), ('EMA 50', 'EMA_50'), ('EMA 100', 'EMA_100')]
    for label, col in ema_levels:
        ema_val = data[col].iloc[-1]
        if abs(low - ema_val) <= atr_dist and close > ema_val:
            return {'stock': stock, 'ema_level': label}

    return None


def higher_highs_higher_lows(stock, data_dict, ctx=None):
    """Detect staircase uptrend via swing pivot analysis. Returns (stock, streak_count)."""
    data = data_dict[stock].copy()
    if data.shape[0] < 250:
        return None
    data['EMA_200'] = talib.EMA(data['close'], 200)

    if data['close'].iloc[-1] <= data['EMA_200'].iloc[-1]:
        return None

    highs = data['high'].values
    lows = data['low'].values

    swing_highs, swing_lows = _williams_pivots(highs, lows, n=2)

    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return None

    # Count consecutive HH+HL from the end
    recent_highs = swing_highs[-10:]
    recent_lows = swing_lows[-10:]

    hh_streak = 0
    for i in range(len(recent_highs) - 1, 0, -1):
        if recent_highs[i][1] > recent_highs[i-1][1]:
            hh_streak += 1
        else:
            break

    hl_streak = 0
    for i in range(len(recent_lows) - 1, 0, -1):
        if recent_lows[i][1] > recent_lows[i-1][1]:
            hl_streak += 1
        else:
            break

    streak = min(hh_streak, hl_streak)
    if streak >= 3:
        return (stock, streak)
    return None


def stage_2_advancing(stock, data_dict, ctx=None):
    """Stage 2 (advancing) classification mirroring market_breadth/stage_analysis.py.

    is_s2 = (slope > 0.01) AND (close > sma_150), where
    sma_150 = close.rolling(150, min_periods=100).mean() and
    slope   = (sma_150 - sma_150.shift(20)) / sma_150.shift(20).
    Returns {'stock', 'entered_stage2_date'} when today's bar is Stage 2;
    entered_stage2_date is the first day of the current uninterrupted run.
    """
    data = data_dict[stock]
    close = data['close']

    sma_150 = close.rolling(window=150, min_periods=100).mean()
    sma_150_prev = sma_150.shift(20)
    slope = (sma_150 - sma_150_prev) / sma_150_prev
    is_s2 = (slope > 0.01) & (close > sma_150)

    if is_s2.empty or not bool(is_s2.iloc[-1]):
        return None

    arr = is_s2.values
    start = len(arr) - 1
    while start > 0 and arr[start - 1]:
        start -= 1
    entered = data.index[start]

    return {'stock': stock, 'entered_stage2_date': str(entered)}
