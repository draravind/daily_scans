"""Shared synthetic-data builders for the package test suite."""
import numpy as np
import pandas as pd


def _make_ohlcv(
    n_days=400,
    start_close=100.0,
    daily_return=0.002,
    volatility=0.04,
    volume_base=200_000,
    start_date='2020-01-01',
    seed=42,
):
    """Build a synthetic OHLCV DataFrame indexed by business dates.

    daily_return: mean daily % change (0.002 = +0.2%/day)
    volatility: daily range as fraction of close (0.04 = 4% range)
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start_date, periods=n_days)

    closes = np.empty(n_days)
    closes[0] = start_close
    for i in range(1, n_days):
        closes[i] = closes[i - 1] * (1 + daily_return + rng.randn() * 0.005)

    highs = closes * (1 + volatility * rng.uniform(0.3, 1.0, n_days))
    lows = closes * (1 - volatility * rng.uniform(0.3, 1.0, n_days))
    opens = closes * (1 + rng.uniform(-0.01, 0.01, n_days))
    volumes = (volume_base * rng.uniform(0.5, 2.0, n_days)).astype(int)

    df = pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': volumes,
    }, index=dates)
    df.index.name = 'date'
    return df


def _make_flat_ohlcv(n_days=400, close=100.0, volume=200_000, start_date='2020-01-01'):
    """Flat price series — useful for testing zero-std / no-trigger scenarios."""
    dates = pd.bdate_range(start_date, periods=n_days)
    df = pd.DataFrame({
        'open': close, 'high': close * 1.001, 'low': close * 0.999,
        'close': close, 'volume': volume,
    }, index=dates)
    df.index.name = 'date'
    return df


def _pack(stock, df):
    """Wrap a single DataFrame into the data_dict format."""
    return {stock: df}


def _make_nifty_with_pivot(n_warmup=160, n_pre=170, n_post=80, base=15000.0, peak_uplift=2500.0):
    """Synthetic benchmark series: gentle ramp during warmup, then rise → peak → drop → recover.

    Returns a DataFrame with columns [symbol, date, high, low, close, instrument_type].
    Total bars = n_warmup + n_pre + 1 + n_post.
    The peak sits at index `n_warmup + n_pre` (a clean local swing high).
    """
    total = n_warmup + n_pre + 1 + n_post
    dates = pd.bdate_range('2023-01-02', periods=total)
    closes = np.empty(total)
    # Warmup: slow drift.
    closes[:n_warmup] = base + np.linspace(0, 200, n_warmup)
    # Pre-peak rise.
    closes[n_warmup:n_warmup + n_pre] = base + 200 + np.linspace(0, peak_uplift, n_pre)
    # Peak bar.
    peak_idx = n_warmup + n_pre
    closes[peak_idx] = base + 200 + peak_uplift + 100
    # Drop then recover (drop >> ATR*k threshold to confirm the H pivot).
    drop_depth = peak_uplift * 0.7
    closes[peak_idx + 1:peak_idx + 1 + n_post // 2] = np.linspace(
        closes[peak_idx], closes[peak_idx] - drop_depth, n_post // 2
    )
    closes[peak_idx + 1 + n_post // 2:] = np.linspace(
        closes[peak_idx] - drop_depth, closes[peak_idx] - drop_depth * 0.5, n_post - n_post // 2
    )
    highs = closes * 1.005
    lows = closes * 0.995
    # Make sure the peak high stands out.
    highs[peak_idx] = closes[peak_idx] * 1.02
    return pd.DataFrame({
        'symbol': 'NIFTY 50',
        'date': dates,
        'high': highs,
        'low': lows,
        'close': closes,
        'instrument_type': 'INDEX',
    })


def _make_rs_lead_case(total=260, symbol='X', start_date='2023-01-02'):
    """Stock + benchmark where the RS line (close / benchmark) makes a fresh
    252-day high inside the last 5 bars while price topped ~8 bars earlier and
    now sits ~5% below that high.

    Returns (stock_df, benchmark_close_series) — both indexed by python dates.
    """
    dates = pd.bdate_range(start_date, periods=total)

    closes = np.empty(total)
    peak_idx = total - 9                          # price high ~8 bars before the end
    closes[:peak_idx + 1] = np.linspace(100.0, 200.0, peak_idx + 1)
    closes[peak_idx + 1:] = np.linspace(200.0, 190.0, total - peak_idx - 1)

    # Benchmark tracks the stock (flat RS) until the recency window, then drops in
    # the last 5 bars so close/benchmark sets its max on the final bar.
    bench = closes * 0.5
    bench[total - 5:] = np.linspace(bench[total - 6], bench[total - 6] * 0.8, 5)

    df = pd.DataFrame({
        'open': closes, 'high': closes * 1.01, 'low': closes * 0.99,
        'close': closes, 'volume': 200_000,
    }, index=dates)
    df.index = df.index.date
    bench_series = pd.Series(bench, index=df.index)
    return df, bench_series


# India liquidity thresholds, mirrored here so package tests can exercise
# common_filters with realistic numbers without depending on any repo.
def _india_like_config():
    from daily_scans import LiquidityFilterConfig
    return LiquidityFilterConfig(
        min_bars=125, price_min=20, volatility_cutoff=0.018,
        turnover_q_high=20_000_000, turnover_mean_high=20_000_000,
        turnover_q_low=10_000_000, turnover_mean_low=30_000_000,
        vol_q_high=75_000, vol_q_mid=50_000, vol_mean_mid=100_000,
        turnover_mean_override=40_000_000, enable_uptrend_filter=False,
    )
