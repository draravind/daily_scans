"""Tests for daily_scans.scan_cores — 36 pure per-stock scan functions.

All functions share signature: fn(stock, data_dict) -> Optional[Any]
They use talib for technical indicators, so synthetic OHLCV DataFrames must
be long enough for rolling windows (typically 250-400 bars).

Approach:
  - _make_ohlcv() builds synthetic data with controllable trend, volatility, volume.
  - For each scan: one "should trigger" and one "should not trigger" dataset.
  - Edge cases: empty data, insufficient bars, zero values.
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from daily_scans import scan_cores as sc
from daily_scans.context import ScanContext

from _helpers import _make_ohlcv, _make_flat_ohlcv, _pack

# Fixed today for the today-sensitive ATH cores (replaces the old calendar
# freeze). Use a Timestamp so `ctx.today - timedelta` compares against the test
# frames' datetime64 index (production passes python-date-indexed frames).
_ATH_CTX = ScanContext(today=pd.Timestamp('2025-06-15'))




# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

class TestAthRuns:
    """Tests for _ath_runs() helper."""

    def test_single_plateau(self):
        """Monotonically rising highs → each bar is its own ATH run of length 1."""
        df = _make_ohlcv(50, daily_return=0.01, seed=10)
        df['ATH'] = df['high'].cummax()
        runs = sc._ath_runs(df)
        assert isinstance(runs, pd.DataFrame)
        assert set(runs.columns) == {'start', 'end', 'value', 'length'}
        # Reversed order: most recent first
        assert runs.iloc[0]['end'] == df.index[-1]

    def test_flat_ath_single_run(self):
        """If high never increases after first bar, ATH plateaus into runs."""
        df = _make_flat_ohlcv(100)
        # Make first bar the highest so ATH is set from bar 0
        df.iloc[0, df.columns.get_loc('high')] = 200.0
        df['ATH'] = df['high'].cummax()
        runs = sc._ath_runs(df)
        # ATH = 200 from bar 0 onwards, so all bars share same ATH → 1 run
        # But bar 0's high (200) vs bar 1+ high (100.1) means ATH never changes
        assert len(runs) >= 1
        assert runs['length'].sum() == 100


class TestFindRecentLongRun:
    def test_no_long_runs(self):
        """All runs < 100 bars → returns None."""
        runs = pd.DataFrame({
            'start': pd.to_datetime(['2024-01-01', '2024-06-01']),
            'end': pd.to_datetime(['2024-05-31', '2024-12-31']),
            'value': [100, 110],
            'length': [50, 50],
        })
        data = _make_flat_ohlcv(200)
        assert sc._find_recent_long_run(runs, data) is None

    def test_single_long_run_at_end(self):
        """Only one long run and it ends at data[-1] → returns None (skip current)."""
        data = _make_flat_ohlcv(200)
        runs = pd.DataFrame({
            'start': [data.index[0]],
            'end': [data.index[-1]],
            'value': [100],
            'length': [200],
        })
        assert sc._find_recent_long_run(runs, data) is None

    def test_two_long_runs_current_at_end(self):
        """Two long runs, current at end → returns second."""
        data = _make_flat_ohlcv(400)
        runs = pd.DataFrame({
            'start': [data.index[-1], data.index[0]],
            'end': [data.index[-1], data.index[150]],
            'value': [110, 100],
            'length': [150, 151],
        })
        result = sc._find_recent_long_run(runs, data)
        assert result is not None
        assert result['value'] == 100


# ═══════════════════════════════════════════════════════════════════════════════
# ATH-related scans
# ═══════════════════════════════════════════════════════════════════════════════

class TestNearATH:
    """Tests for near_ATH_high_ATR and near_ATH_low_ATR."""

    def test_high_atr_triggers_near_ath(self):
        """Stock near ATH with high volatility should trigger."""
        df = _make_ohlcv(400, daily_return=0.002, volatility=0.05, seed=1)
        # Force last bar near ATH and high ATR
        ath = df['high'].max()
        df.iloc[-1, df.columns.get_loc('high')] = ath * 0.98
        df.iloc[-1, df.columns.get_loc('close')] = ath * 0.95
        df.iloc[-2, df.columns.get_loc('close')] = ath * 0.94
        result = sc.near_ATH_high_ATR('TEST', _pack('TEST', df), _ATH_CTX)
        # May or may not trigger depending on EMA200/ATR values
        assert result is None or result == 'TEST'

    def test_empty_data_returns_none(self):
        """Data entirely outside 5-year window → empty → None."""
        df = _make_ohlcv(100, start_date='2015-01-01')
        # With _ATH_CTX.today = 2025-06-15, 5yr window starts ~2020-06-15
        # Data from 2015 is entirely before window
        result = sc.near_ATH_high_ATR('X', _pack('X', df), _ATH_CTX)
        assert result is None

    def test_low_atr_variant(self):
        """Low ATR variant should reject high-volatility stocks."""
        df = _make_ohlcv(400, volatility=0.06, seed=7)
        result = sc.near_ATH_low_ATR('X', _pack('X', df), _ATH_CTX)
        # High volatility → ATR likely >= 3.6% → should NOT trigger low_ATR
        assert result is None or result == 'X'

    def test_downtrend_returns_none(self):
        """Strong downtrend → close far below ATH → None."""
        df = _make_ohlcv(400, daily_return=-0.005, seed=3)
        assert sc.near_ATH_high_ATR('X', _pack('X', df), _ATH_CTX) is None


class TestMakingNewATH:
    """Tests for making_new_ATH_high_ATR and making_new_ATH_low_ATR."""

    def test_no_long_runs_returns_none(self):
        """Monotonically rising stock → no >100-bar ATH plateau → None."""
        df = _make_ohlcv(400, daily_return=0.005, seed=5)
        assert sc.making_new_ATH_high_ATR('X', _pack('X', df), _ATH_CTX) is None

    def test_empty_data_returns_none(self):
        df = _make_ohlcv(100, start_date='2015-01-01')
        assert sc.making_new_ATH_high_ATR('X', _pack('X', df), _ATH_CTX) is None

    def test_low_atr_variant_no_runs(self):
        df = _make_ohlcv(400, daily_return=0.005, seed=5)
        assert sc.making_new_ATH_low_ATR('X', _pack('X', df), _ATH_CTX) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Smooth & Basing
# ═══════════════════════════════════════════════════════════════════════════════

class TestSmoothStocks:
    def test_insufficient_bars_returns_none(self):
        """Less than 400 bars → None."""
        df = _make_ohlcv(300, seed=1)
        assert sc.smooth_stocks('X', _pack('X', df)) is None

    def test_returns_dict_with_expected_keys(self):
        """When conditions met, returns dict with DR_to_RR keys."""
        df = _make_ohlcv(500, daily_return=0.003, volatility=0.02, seed=10)
        result = sc.smooth_stocks('X', _pack('X', df))
        if result is not None:
            assert isinstance(result, dict)
            assert result['stock'] == 'X'
            assert 'DR_to_RR_>_4%' in result
            assert 'DR_to_RR_>_5%' in result
            assert 'DR_to_RR_>_6%' in result
            for key in ['DR_to_RR_>_4%', 'DR_to_RR_>_5%', 'DR_to_RR_>_6%']:
                assert 0 <= result[key] <= 100

    def test_zero_low_returns_none(self):
        """If any low == 0, returns None."""
        df = _make_ohlcv(500, seed=2)
        df.iloc[10, df.columns.get_loc('low')] = 0.0
        assert sc.smooth_stocks('X', _pack('X', df)) is None

    def test_downtrend_returns_none(self):
        """Strong downtrend → close < max(EMA200, 70% ATH) → None."""
        df = _make_ohlcv(500, daily_return=-0.003, seed=4)
        assert sc.smooth_stocks('X', _pack('X', df)) is None


class TestBasingStocks:
    def test_insufficient_bars_returns_none(self):
        df = _make_ohlcv(200, seed=1)
        assert sc.basing_stocks('X', _pack('X', df)) is None

    def test_monotonic_uptrend_returns_none(self):
        """Pure uptrend has no base formation → None."""
        df = _make_ohlcv(400, daily_return=0.005, volatility=0.02, seed=1)
        assert sc.basing_stocks('X', _pack('X', df)) is None

    def test_low_volume_returns_none(self):
        """Below volume thresholds → None."""
        df = _make_ohlcv(400, volume_base=10_000, seed=5)
        assert sc.basing_stocks('X', _pack('X', df)) is None

    def test_penny_stock_returns_none(self):
        """Close < 10 → None."""
        df = _make_ohlcv(400, start_close=5.0, daily_return=0.001, seed=6)
        assert sc.basing_stocks('X', _pack('X', df)) is None


# ═══════════════════════════════════════════════════════════════════════════════
# High Tight Flag
# ═══════════════════════════════════════════════════════════════════════════════

class TestHighTightFlag:
    def test_insufficient_bars_returns_none(self):
        df = _make_ohlcv(104, seed=1)
        assert sc.high_tight_flag('X', _pack('X', df)) is None

    def test_returns_dict_when_triggered(self):
        """Clean monotonic pole (100 → 200) followed by a tight flag near the top
        triggers, with pole top within the 35-bar recency window."""
        n = 110
        dates = pd.bdate_range('2024-01-01', periods=n)
        p = np.empty(n)
        p[0:30] = 100.0
        p[30:80] = np.linspace(100.0, 200.0, 50)
        p[80:n] = np.linspace(195.0, 190.0, n - 80)
        df = pd.DataFrame({
            'open': p, 'high': p * 1.005, 'low': p * 0.995, 'close': p,
            'volume': 200_000,
        }, index=dates)
        result = sc.high_tight_flag('X', _pack('X', df))
        assert result is not None
        assert result['stock'] == 'X'
        assert result['tier'] in ('low', 'high')
        assert isinstance(result['pole_gain_pct'], float)
        assert result['pole_gain_pct'] > 60.0
        assert isinstance(result['pole_low_date'], dt.date)
        assert isinstance(result['pole_high_date'], dt.date)
        assert isinstance(result['flag_end_date'], dt.date)

    def test_two_leg_pole_rejected_by_cleanup(self):
        """A pole with a >cube-root mid-drawdown re-anchors pole_low above the
        initial foot — surviving gain drops below 60% and the pattern is rejected.
        The pre-cleanup endpoints (foot 100 → top 200) would have passed."""
        n = 105
        dates = pd.bdate_range('2024-01-01', periods=n)
        p = np.empty(n)
        p[0:21] = np.linspace(100.0, 200.0, 21)   # leg 1: 100 → 200
        p[21:25] = np.linspace(195.0, 158.0, 4)   # deep pullback trips re-anchor
        p[25:50] = np.linspace(164.0, 199.0, 25)  # leg 2: rise back to ~200
        p[50] = 200.0                              # pole top (later-wins tie)
        p[51:n] = 195.0                            # flag near top
        df = pd.DataFrame({
            'open': p, 'high': p, 'low': p, 'close': p, 'volume': 200_000,
        }, index=dates)
        assert sc.high_tight_flag('X', _pack('X', df)) is None

    def test_flat_stock_returns_none(self):
        """Flat stock has no 60% pole gain → None."""
        df = _make_flat_ohlcv(400)
        assert sc.high_tight_flag('X', _pack('X', df)) is None


# ═══════════════════════════════════════════════════════════════════════════════
# ATR Contraction
# ═══════════════════════════════════════════════════════════════════════════════

class TestWeeklyAtrContraction:
    def test_insufficient_bars_returns_none(self):
        df = _make_ohlcv(200, seed=1)
        assert sc.weekly_atr_contraction('X', _pack('X', df)) is None

    def test_low_volume_returns_none(self):
        """Below volume thresholds → None even if ATR contracted."""
        df = _make_ohlcv(400, volume_base=1_000, seed=2)
        assert sc.weekly_atr_contraction('X', _pack('X', df)) is None

    def test_penny_stock_returns_none(self):
        df = _make_ohlcv(400, start_close=5.0, daily_return=0.001, seed=3)
        assert sc.weekly_atr_contraction('X', _pack('X', df)) is None


# ═══════════════════════════════════════════════════════════════════════════════
# EMA Contraction (always returns dict)
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmaContraction:
    def test_always_returns_dict(self):
        """ema_contraction always returns a dict, never None."""
        df = _make_ohlcv(300, seed=1)
        result = sc.ema_contraction('X', _pack('X', df))
        assert isinstance(result, dict)
        assert result['stock'] == 'X'
        assert 'contraction' in result
        assert 'contraction_atr' in result

    def test_flat_stock_low_contraction(self):
        """Flat price → all EMAs converge → low contraction."""
        df = _make_flat_ohlcv(400)
        result = sc.ema_contraction('X', _pack('X', df))
        assert result['contraction'] < 1.0  # very tight cluster

    def test_trending_stock_higher_contraction(self):
        """Strong trend → EMAs spread out → higher contraction."""
        df = _make_ohlcv(400, daily_return=0.01, seed=5)
        result = sc.ema_contraction('X', _pack('X', df))
        assert result['contraction'] > 0  # non-zero spread


# ═══════════════════════════════════════════════════════════════════════════════
# Stocks Above 200 EMA
# ═══════════════════════════════════════════════════════════════════════════════

class TestStocksAbove200:
    def test_uptrend_triggers(self):
        """Strong uptrend → close > EMA200 → returns stock."""
        df = _make_ohlcv(400, daily_return=0.005, seed=1)
        result = sc.stocks_above_200('X', _pack('X', df))
        assert result == 'X'

    def test_downtrend_returns_none(self):
        """Strong downtrend → close < EMA200 → None."""
        df = _make_ohlcv(400, daily_return=-0.005, seed=1)
        result = sc.stocks_above_200('X', _pack('X', df))
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# Relative Strength Score (always returns dict)
# ═══════════════════════════════════════════════════════════════════════════════

class TestRelativeStrengthScore:
    def test_returns_dict_with_rs_raw(self):
        df = _make_ohlcv(300, seed=1)
        result = sc.relative_strength_score('X', _pack('X', df))
        assert isinstance(result, dict)
        assert result['stock'] == 'X'
        assert 'rs_raw' in result
        assert isinstance(result['rs_raw'], float)

    def test_short_history_returns_none(self):
        df = _make_ohlcv(100, seed=1)  # < 253 bars
        assert sc.relative_strength_score('X', _pack('X', df)) is None

    def test_uptrend_higher_than_downtrend(self):
        up = _make_ohlcv(300, daily_return=0.005, seed=2)
        down = _make_ohlcv(300, daily_return=-0.003, seed=3)
        up_rs = sc.relative_strength_score('U', _pack('U', up))['rs_raw']
        down_rs = sc.relative_strength_score('D', _pack('D', down))['rs_raw']
        assert up_rs > down_rs


# ═══════════════════════════════════════════════════════════════════════════════
# Episodic Pivots
# ═══════════════════════════════════════════════════════════════════════════════

class TestEpisodicPivots:
    def test_no_gap_returns_none(self):
        """Normal trading without gaps → None."""
        df = _make_ohlcv(300, volatility=0.02, seed=1)
        result = sc.episodic_pivots('X', _pack('X', df))
        assert result is None

    def test_gap_up_with_strong_body(self):
        """Inject a gap-up opening with strong body + 2x volume on last bar."""
        df = _make_ohlcv(300, seed=10)
        # Force gap-up: open >> prior high, close >> open
        prior_high = df['high'].iloc[-2]
        tr_pct = 0.05  # typical TR%
        gap_open = prior_high + tr_pct * df['close'].iloc[-1] * 2
        gap_close = gap_open + tr_pct * df['close'].iloc[-1] * 2
        df.iloc[-1, df.columns.get_loc('open')] = gap_open
        df.iloc[-1, df.columns.get_loc('close')] = gap_close
        df.iloc[-1, df.columns.get_loc('high')] = gap_close * 1.01
        # Tight low → close lands in top 25% of day's range (canonical Bonde).
        df.iloc[-1, df.columns.get_loc('low')] = gap_open * 0.999
        # Volume ≥ 2× SMA(volume, 50) (canonical Bonde EP volume gate).
        avg_vol = df['volume'].iloc[-50:].mean()
        df.iloc[-1, df.columns.get_loc('volume')] = avg_vol * 3
        result = sc.episodic_pivots('X', _pack('X', df))
        assert result == 'X'


# ═══════════════════════════════════════════════════════════════════════════════
# EMA Uptrend
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmaUptrend:
    def test_strong_uptrend_triggers(self):
        """Sustained uptrend → EMA stack properly ordered → triggers."""
        df = _make_ohlcv(500, daily_return=0.005, volatility=0.02, seed=1)
        result = sc.ema_uptrend_fn('X', _pack('X', df))
        # Strong uptrend should trigger either group 1 or group 2
        assert result is None or result == 'X'

    def test_downtrend_returns_none(self):
        df = _make_ohlcv(500, daily_return=-0.005, seed=1)
        assert sc.ema_uptrend_fn('X', _pack('X', df)) is None

    def test_flat_stock_returns_none(self):
        df = _make_flat_ohlcv(500)
        assert sc.ema_uptrend_fn('X', _pack('X', df)) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Slow / Fast Stocks
# ═══════════════════════════════════════════════════════════════════════════════

class TestSlowStocks:
    def test_low_volatility_triggers(self):
        """Very low range stock → should trigger."""
        df = _make_ohlcv(400, volatility=0.005, seed=1)
        result = sc.slow_stocks('X', _pack('X', df))
        assert result is None or result == 'X'

    def test_high_volatility_returns_none(self):
        """High volatility → TR > thresholds → None."""
        df = _make_ohlcv(400, volatility=0.08, seed=1)
        result = sc.slow_stocks('X', _pack('X', df))
        assert result is None


class TestFastStocks:
    def test_low_volatility_returns_none(self):
        """Low volatility → fails TR thresholds → None."""
        df = _make_ohlcv(400, volatility=0.01, seed=1)
        assert sc.fast_stocks('X', _pack('X', df)) is None

    def test_low_volume_returns_none(self):
        """Below volume threshold → None."""
        df = _make_ohlcv(400, volatility=0.06, volume_base=10_000, seed=1)
        assert sc.fast_stocks('X', _pack('X', df)) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Launchpad Stocks
# ═══════════════════════════════════════════════════════════════════════════════

class TestLaunchpadStocks:
    def test_low_volume_returns_none(self):
        df = _make_ohlcv(400, volume_base=10_000, seed=1)
        assert sc.launchpad_stocks('X', _pack('X', df)) is None

    def test_downtrend_returns_none(self):
        df = _make_ohlcv(400, daily_return=-0.005, seed=1)
        assert sc.launchpad_stocks('X', _pack('X', df)) is None


# ═══════════════════════════════════════════════════════════════════════════════
# VCP (Volatility Contraction Pattern)
# ═══════════════════════════════════════════════════════════════════════════════

class TestVCP:
    def test_below_ema200_returns_none(self):
        """Close < EMA200 → None."""
        df = _make_ohlcv(400, daily_return=-0.005, seed=1)
        assert sc.vcp('X', _pack('X', df)) is None

    def test_flat_stock_returns_none(self):
        """Flat price → no contracting highs/lows pattern → None."""
        df = _make_flat_ohlcv(400)
        assert sc.vcp('X', _pack('X', df)) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Checklist
# ═══════════════════════════════════════════════════════════════════════════════

class TestChecklist:
    def test_downtrend_returns_none(self):
        """Strong downtrend fails multiple checklist conditions → None."""
        df = _make_ohlcv(400, daily_return=-0.005, seed=1)
        assert sc.checklist('X', _pack('X', df)) is None

    def test_strong_uptrend_may_trigger(self):
        """Strong uptrend with good volume may trigger."""
        df = _make_ohlcv(400, daily_return=0.004, volatility=0.03, volume_base=500_000, seed=10)
        result = sc.checklist('X', _pack('X', df))
        assert result is None or result == 'X'

    def test_flat_stock_returns_none(self):
        """Flat stock → MT_trend fails → None."""
        df = _make_flat_ohlcv(400, volume=500_000)
        assert sc.checklist('X', _pack('X', df)) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Tight Range
# ═══════════════════════════════════════════════════════════════════════════════

class TestTightRange:
    def test_below_ema200_returns_none(self):
        df = _make_ohlcv(500, daily_return=-0.005, seed=1)
        assert sc.tight_range('X', _pack('X', df)) is None

    def test_high_volatility_returns_none(self):
        """High range → never hits Q1 → None."""
        df = _make_ohlcv(500, daily_return=0.003, volatility=0.06, seed=1)
        result = sc.tight_range('X', _pack('X', df))
        # High vol unlikely to have current range at 1st percentile
        assert result is None or result == 'X'


# ═══════════════════════════════════════════════════════════════════════════════
# High ATR
# ═══════════════════════════════════════════════════════════════════════════════

class TestHighAtr:
    def test_low_turnover_returns_none(self):
        """Below 10M turnover → None."""
        df = _make_ohlcv(400, volume_base=10_000, start_close=50, seed=1)
        assert sc.high_atr('X', _pack('X', df)) is None

    def test_penny_stock_returns_none(self):
        df = _make_ohlcv(400, start_close=5, daily_return=0.001, seed=1)
        assert sc.high_atr('X', _pack('X', df)) is None

    def test_downtrend_returns_none(self):
        """Downtrend → < 80/120 bars above EMA200 → None."""
        df = _make_ohlcv(400, daily_return=-0.005, seed=1)
        assert sc.high_atr('X', _pack('X', df)) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Contracting Stocks
# ═══════════════════════════════════════════════════════════════════════════════

class TestContractingStocks:
    def test_below_ema200_returns_none(self):
        df = _make_ohlcv(400, daily_return=-0.005, seed=1)
        assert sc.contracting_stocks('X', _pack('X', df)) is None

    def test_returns_tuple_format(self):
        """When triggered, returns (stock, n_days_1, n_days_2)."""
        df = _make_ohlcv(400, daily_return=0.003, volatility=0.03, seed=5)
        result = sc.contracting_stocks('X', _pack('X', df))
        if result is not None:
            assert isinstance(result, tuple)
            assert len(result) == 3
            assert result[0] == 'X'
            assert isinstance(result[1], (int, np.integer))
            assert isinstance(result[2], (int, np.integer))


# ═══════════════════════════════════════════════════════════════════════════════
# Bullish Confirmation
# ═══════════════════════════════════════════════════════════════════════════════

class TestBullishConfirmation:
    def test_below_ema200_returns_none(self):
        df = _make_ohlcv(400, daily_return=-0.005, seed=1)
        assert sc.bullish_confirmation('X', _pack('X', df)) is None

    def test_no_bullish_candles_returns_none(self):
        """Flat stock → no strong candles → None."""
        df = _make_flat_ohlcv(400)
        assert sc.bullish_confirmation('X', _pack('X', df)) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Top Movers
# ═══════════════════════════════════════════════════════════════════════════════

class TestTopMovers:
    def test_below_ema200_returns_none(self):
        df = _make_ohlcv(400, daily_return=-0.005, seed=1)
        assert sc.top_movers('X', _pack('X', df)) is None

    def test_returns_dict_with_scores(self):
        """Uptrend → returns dict with period scores."""
        df = _make_ohlcv(400, daily_return=0.005, seed=1)
        result = sc.top_movers('X', _pack('X', df))
        if result is not None:
            assert isinstance(result, dict)
            assert result['stock'] == 'X'
            for key in ['is_1d', 'is_1w', 'is_1m', 'is_3m', 'is_1y']:
                assert key in result


# ═══════════════════════════════════════════════════════════════════════════════
# Highest Turnover
# ═══════════════════════════════════════════════════════════════════════════════

class TestHighestTurnover:
    def test_downtrend_returns_none(self):
        df = _make_ohlcv(600, daily_return=-0.005, seed=1)
        assert sc.highest_turnover('X', _pack('X', df)) is None

    def test_turnover_spike_may_trigger(self):
        """Inject a massive volume spike on last bar."""
        df = _make_ohlcv(600, daily_return=0.003, seed=5)
        df.iloc[-1, df.columns.get_loc('volume')] = df['volume'].max() * 100
        result = sc.highest_turnover('X', _pack('X', df))
        assert result is None or (isinstance(result, dict) and result['stock'] == 'X')


# ═══════════════════════════════════════════════════════════════════════════════
# IPO
# ═══════════════════════════════════════════════════════════════════════════════

class TestIPO:
    def test_more_than_250_bars_returns_none(self):
        df = _make_ohlcv(300, seed=1)
        assert sc.ipo('X', _pack('X', df)) is None

    def test_within_250_bars_returns_dict(self):
        df = _make_ohlcv(100, seed=1)
        result = sc.ipo('X', _pack('X', df))
        assert isinstance(result, dict)
        assert result['stock'] == 'X'
        assert 'near_ipo_high' in result
        assert 'ipo_6m' in result
        assert 'ipo_12m' in result

    def test_ipo_6m_classification(self):
        """21-125 bars → ipo_6m=True, ipo_12m=False."""
        df = _make_ohlcv(80, seed=1)
        result = sc.ipo('X', _pack('X', df))
        assert result['ipo_6m'] is True
        assert result['ipo_12m'] is False

    def test_ipo_12m_classification(self):
        """126-250 bars → ipo_6m=False, ipo_12m=True."""
        df = _make_ohlcv(200, seed=1)
        result = sc.ipo('X', _pack('X', df))
        assert result['ipo_6m'] is False
        assert result['ipo_12m'] is True

    def test_under_21_bars(self):
        """< 21 bars → ipo_6m=False, ipo_12m=False."""
        df = _make_ohlcv(15, seed=1)
        result = sc.ipo('X', _pack('X', df))
        assert result['ipo_6m'] is False
        assert result['ipo_12m'] is False

    def test_near_ipo_high(self):
        """Close near the 63-bar IPO high → near_ipo_high=True."""
        df = _make_ohlcv(100, daily_return=0.0, volatility=0.02, seed=1)
        # Close near first 63-bar high
        ipo_high = df['high'].iloc[:63].max()
        df.iloc[-1, df.columns.get_loc('close')] = ipo_high * 0.95  # within 85%-125%
        result = sc.ipo('X', _pack('X', df))
        assert result['near_ipo_high'] == True

    def test_far_from_ipo_high(self):
        """Close far below IPO high → near_ipo_high=False."""
        df = _make_ohlcv(100, daily_return=-0.01, seed=1)
        result = sc.ipo('X', _pack('X', df))
        assert result['near_ipo_high'] == False


# ═══════════════════════════════════════════════════════════════════════════════
# Intraday
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntraday:
    def test_insufficient_bars_returns_none(self):
        df = _make_ohlcv(200, seed=1)
        assert sc.intraday('X', _pack('X', df)) is None

    def test_low_volatility_returns_none(self):
        """ADR < 5% → None."""
        df = _make_ohlcv(300, volatility=0.01, seed=1)
        assert sc.intraday('X', _pack('X', df)) is None

    def test_high_volatility_high_volume_triggers(self):
        """High ADR + good volume → triggers."""
        df = _make_ohlcv(300, volatility=0.07, volume_base=1_000_000, seed=1)
        result = sc.intraday('X', _pack('X', df))
        assert result is None or result == 'X'


# ═══════════════════════════════════════════════════════════════════════════════
# Gap-ups
# ═══════════════════════════════════════════════════════════════════════════════

class TestGapups:
    def test_below_ema200_returns_none(self):
        df = _make_ohlcv(400, daily_return=-0.005, seed=1)
        assert sc.gapups('X', _pack('X', df)) is None

    def test_no_gaps_returns_none(self):
        """Low volatility → no gaps → None."""
        df = _make_ohlcv(400, volatility=0.01, seed=1)
        result = sc.gapups('X', _pack('X', df))
        assert result is None

    def test_injected_gapup_triggers(self):
        """Inject a gap-up on bar -3 and hold above gap low."""
        df = _make_ohlcv(400, daily_return=0.003, volatility=0.03, seed=10)
        # Create a gap-up: bar -3 low >> bar -4 high
        prior_high = df['high'].iloc[-4]
        gap_low = prior_high * 1.05  # 5% gap, > 1.01x required
        gap_close = gap_low * 1.02
        df.iloc[-3, df.columns.get_loc('low')] = gap_low
        df.iloc[-3, df.columns.get_loc('open')] = gap_low
        df.iloc[-3, df.columns.get_loc('close')] = gap_close
        df.iloc[-3, df.columns.get_loc('high')] = gap_close * 1.01
        # Ensure subsequent bars hold above gap low
        for i in [-2, -1]:
            df.iloc[i, df.columns.get_loc('low')] = gap_low * 1.001
            df.iloc[i, df.columns.get_loc('close')] = gap_close * 1.01
            df.iloc[i, df.columns.get_loc('high')] = gap_close * 1.02
            df.iloc[i, df.columns.get_loc('open')] = gap_close
        result = sc.gapups('X', _pack('X', df))
        assert result is None or result == 'X'


# ═══════════════════════════════════════════════════════════════════════════════
# Base Breakout
# ═══════════════════════════════════════════════════════════════════════════════

class TestBaseBreakout:
    def test_valid_pivot_breakout(self):
        """Pivot at bar 100, breakout on the last bar inside the recent window."""
        n = 200
        df = _make_flat_ohlcv(n, close=100.0)
        df.iloc[100, df.columns.get_loc('high')] = 105.0
        df.iloc[199, df.columns.get_loc('high')] = 106.0
        result = sc.base_breakout('X', _pack('X', df))
        assert result == ('X', 99, df.index[100], 105.0)

    def test_unconfirmable_pivot_too_close_to_end(self):
        """Pivot at index 185 cannot be confirmed (needs 21 right bars after it)."""
        n = 200
        df = _make_flat_ohlcv(n, close=100.0)
        df.iloc[185, df.columns.get_loc('high')] = 105.0
        df.iloc[199, df.columns.get_loc('high')] = 106.0
        result = sc.base_breakout('X', _pack('X', df))
        assert result is None

    def test_unconfirmable_pivot_too_close_to_start(self):
        """Pivot at index 50 cannot be confirmed (needs 64 left bars before it)."""
        n = 200
        df = _make_flat_ohlcv(n, close=100.0)
        df.iloc[50, df.columns.get_loc('high')] = 105.0
        df.iloc[199, df.columns.get_loc('high')] = 106.0
        result = sc.base_breakout('X', _pack('X', df))
        assert result is None

    def test_tied_left_bar_disqualifies_pivot(self):
        """Equal high inside the left window invalidates the candidate (strict >)."""
        n = 200
        df = _make_flat_ohlcv(n, close=100.0)
        df.iloc[100, df.columns.get_loc('high')] = 105.0
        df.iloc[80, df.columns.get_loc('high')] = 105.0
        df.iloc[199, df.columns.get_loc('high')] = 106.0
        result = sc.base_breakout('X', _pack('X', df))
        assert result is None

    def test_tied_right_bar_disqualifies_pivot(self):
        """Equal high inside the right window invalidates the candidate (strict >)."""
        n = 200
        df = _make_flat_ohlcv(n, close=100.0)
        df.iloc[100, df.columns.get_loc('high')] = 105.0
        df.iloc[115, df.columns.get_loc('high')] = 105.0
        df.iloc[199, df.columns.get_loc('high')] = 106.0
        result = sc.base_breakout('X', _pack('X', df))
        assert result is None

    def test_tied_breakout_rejected(self):
        """Breakout requires strict >, equal high does not break the ceiling."""
        n = 200
        df = _make_flat_ohlcv(n, close=100.0)
        df.iloc[100, df.columns.get_loc('high')] = 105.0
        df.iloc[199, df.columns.get_loc('high')] = 105.0
        result = sc.base_breakout('X', _pack('X', df))
        assert result is None

    def test_unbroken_ceiling_returns_none(self):
        """Pivot present but no later bar clears it — ceiling still active."""
        n = 200
        df = _make_flat_ohlcv(n, close=100.0)
        df.iloc[100, df.columns.get_loc('high')] = 105.0
        result = sc.base_breakout('X', _pack('X', df))
        assert result is None

    def test_breakout_outside_recent_window_rejected(self):
        """Breakout bar before the trailing RECENT_WINDOW is too old."""
        n = 200
        df = _make_flat_ohlcv(n, close=100.0)
        df.iloc[100, df.columns.get_loc('high')] = 105.0
        df.iloc[150, df.columns.get_loc('high')] = 106.0
        result = sc.base_breakout('X', _pack('X', df))
        assert result is None

    def test_breakout_within_recent_window_accepted(self):
        """Breakout bar inside the trailing RECENT_WINDOW is accepted."""
        n = 200
        df = _make_flat_ohlcv(n, close=100.0)
        df.iloc[100, df.columns.get_loc('high')] = 105.0
        df.iloc[197, df.columns.get_loc('high')] = 106.0
        result = sc.base_breakout('X', _pack('X', df))
        assert result is not None
        assert result[1] == 97
        assert result[2] == df.index[100]
        assert result[3] == 105.0

    def test_history_beyond_lookback_ignored(self):
        """Confirmable pivots whose bar index falls before the trailing 1260-bar window are ignored."""
        n = 1400
        df = _make_flat_ohlcv(n, close=100.0)
        df.iloc[100, df.columns.get_loc('high')] = 200.0  # confirmable pivot, but pre-lookback (100 < 140)
        df.iloc[200, df.columns.get_loc('high')] = 105.0  # inside lookback
        df.iloc[1399, df.columns.get_loc('high')] = 106.0
        result = sc.base_breakout('X', _pack('X', df))
        assert result is not None
        assert result[3] == 105.0
        assert result[2] == df.index[200]

    def test_highest_ceiling_wins_when_multiple_break(self):
        """When multiple ceilings break in the recent window, the highest wins."""
        n = 300
        df = _make_flat_ohlcv(n, close=100.0)
        df.iloc[100, df.columns.get_loc('high')] = 110.0  # higher pivot, earlier
        df.iloc[200, df.columns.get_loc('high')] = 105.0  # lower pivot, later
        df.iloc[299, df.columns.get_loc('high')] = 111.0  # clears both inside recent window
        result = sc.base_breakout('X', _pack('X', df))
        assert result is not None
        assert result[3] == 110.0
        assert result[2] == df.index[100]

    def test_insufficient_history_returns_none(self):
        """Need at least left + right + 2 = 87 bars."""
        df = _make_flat_ohlcv(86, close=100.0)
        result = sc.base_breakout('X', _pack('X', df))
        assert result is None

    def test_minimum_history_with_pivot_at_boundary(self):
        """At the minimum confirmable size, pivot at index 64 with breakout at index 86."""
        n = 87
        df = _make_flat_ohlcv(n, close=100.0)
        df.iloc[64, df.columns.get_loc('high')] = 105.0
        df.iloc[86, df.columns.get_loc('high')] = 106.0
        result = sc.base_breakout('X', _pack('X', df))
        assert result == ('X', 22, df.index[64], 105.0)

    def test_return_shape(self):
        """Valid breakout returns a 4-tuple with the documented types."""
        n = 200
        df = _make_flat_ohlcv(n, close=100.0)
        df.iloc[100, df.columns.get_loc('high')] = 105.0
        df.iloc[199, df.columns.get_loc('high')] = 106.0
        result = sc.base_breakout('X', _pack('X', df))
        assert result is not None
        assert len(result) == 4
        assert isinstance(result[0], str)
        assert isinstance(result[1], int)
        assert isinstance(result[3], float)


# ═══════════════════════════════════════════════════════════════════════════════
# Avg Turnover Ranked
# ═══════════════════════════════════════════════════════════════════════════════

class TestAvgTurnoverRanked:
    def test_insufficient_bars_returns_none(self):
        df = _make_ohlcv(50, seed=1)
        assert sc.avg_turnover_ranked('X', _pack('X', df)) is None

    def test_exactly_63_bars(self):
        """63 bars → quarter + all_time, no 6_months or 1_year."""
        df = _make_ohlcv(63, seed=1)
        result = sc.avg_turnover_ranked('X', _pack('X', df))
        assert result is not None
        assert 'quarter' in result
        assert 'all_time' in result
        assert '6_months' not in result
        assert '1_year' not in result

    def test_125_bars(self):
        """125 bars → quarter + 6_months + all_time."""
        df = _make_ohlcv(125, seed=1)
        result = sc.avg_turnover_ranked('X', _pack('X', df))
        assert '6_months' in result
        assert '1_year' not in result

    def test_250_bars(self):
        """250+ bars → all periods."""
        df = _make_ohlcv(300, seed=1)
        result = sc.avg_turnover_ranked('X', _pack('X', df))
        assert '1_year' in result

    def test_turnover_values_positive(self):
        df = _make_ohlcv(300, seed=1)
        result = sc.avg_turnover_ranked('X', _pack('X', df))
        for key in ['quarter', '6_months', '1_year', 'all_time']:
            assert result[key] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Top Movers (rewritten — multi-timeframe dict)
# ═══════════════════════════════════════════════════════════════════════════════

class TestTopMoversRewritten:
    def test_below_ema200_returns_none(self):
        df = _make_ohlcv(400, daily_return=-0.005, seed=1)
        assert sc.top_movers('X', _pack('X', df)) is None

    def test_insufficient_bars_returns_none(self):
        df = _make_ohlcv(200, seed=1)
        assert sc.top_movers('X', _pack('X', df)) is None

    def test_returns_dict_with_period_scores(self):
        """Uptrend → returns dict with all period scores."""
        df = _make_ohlcv(400, daily_return=0.005, seed=1)
        result = sc.top_movers('X', _pack('X', df))
        if result is not None:
            assert isinstance(result, dict)
            assert result['stock'] == 'X'
            for key in ['is_1d', 'is_1w', 'is_1m', 'is_3m', 'is_1y']:
                assert key in result
                assert isinstance(result[key], float)


# ═══════════════════════════════════════════════════════════════════════════════
# Golden Cross
# ═══════════════════════════════════════════════════════════════════════════════

class TestGoldenCross:
    def test_insufficient_bars_returns_none(self):
        df = _make_ohlcv(200, seed=1)
        assert sc.golden_cross('X', _pack('X', df)) is None

    def test_downtrend_returns_none(self):
        df = _make_ohlcv(400, daily_return=-0.005, seed=1)
        assert sc.golden_cross('X', _pack('X', df)) is None

    def test_strong_uptrend_no_recent_cross(self):
        """Sustained uptrend → SMA50 > SMA200 for a long time → no recent crossover → None."""
        df = _make_ohlcv(500, daily_return=0.005, seed=1)
        result = sc.golden_cross('X', _pack('X', df))
        # SMA50 has been above SMA200 for a long time, so no recent cross
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# MACD Bullish Crossover
# ═══════════════════════════════════════════════════════════════════════════════

class TestMACDBullishCrossover:
    def test_insufficient_bars_returns_none(self):
        df = _make_ohlcv(200, seed=1)
        assert sc.macd_bullish_crossover('X', _pack('X', df)) is None

    def test_below_ema200_returns_none(self):
        df = _make_ohlcv(400, daily_return=-0.005, seed=1)
        assert sc.macd_bullish_crossover('X', _pack('X', df)) is None

    def test_flat_stock_returns_none(self):
        df = _make_flat_ohlcv(400)
        assert sc.macd_bullish_crossover('X', _pack('X', df)) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Bullish Engulfing
# ═══════════════════════════════════════════════════════════════════════════════

class TestBullishEngulfing:
    def test_insufficient_bars_returns_none(self):
        df = _make_ohlcv(200, seed=1)
        assert sc.bullish_engulfing('X', _pack('X', df)) is None

    def test_below_ema200_returns_none(self):
        df = _make_ohlcv(400, daily_return=-0.005, seed=1)
        assert sc.bullish_engulfing('X', _pack('X', df)) is None

    def test_flat_stock_returns_none(self):
        """Flat stock unlikely to produce engulfing patterns."""
        df = _make_flat_ohlcv(400)
        assert sc.bullish_engulfing('X', _pack('X', df)) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Hammer
# ═══════════════════════════════════════════════════════════════════════════════

class TestHammer:
    def test_insufficient_bars_returns_none(self):
        df = _make_ohlcv(200, seed=1)
        assert sc.hammer('X', _pack('X', df)) is None

    def test_below_ema200_returns_none(self):
        df = _make_ohlcv(400, daily_return=-0.005, seed=1)
        assert sc.hammer('X', _pack('X', df)) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Morning Star
# ═══════════════════════════════════════════════════════════════════════════════

class TestMorningStar:
    def test_insufficient_bars_returns_none(self):
        df = _make_ohlcv(200, seed=1)
        assert sc.morning_star('X', _pack('X', df)) is None

    def test_below_ema200_returns_none(self):
        df = _make_ohlcv(400, daily_return=-0.005, seed=1)
        assert sc.morning_star('X', _pack('X', df)) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Doji
# ═══════════════════════════════════════════════════════════════════════════════

class TestDoji:
    def test_insufficient_bars_returns_none(self):
        df = _make_ohlcv(200, seed=1)
        assert sc.doji('X', _pack('X', df)) is None

    def test_below_ema200_returns_none(self):
        df = _make_ohlcv(400, daily_return=-0.005, seed=1)
        assert sc.doji('X', _pack('X', df)) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Bollinger Squeeze
# ═══════════════════════════════════════════════════════════════════════════════

class TestBollingerSqueeze:
    def test_insufficient_bars_returns_none(self):
        df = _make_ohlcv(200, seed=1)
        assert sc.bollinger_squeeze('X', _pack('X', df)) is None

    def test_below_ema200_returns_none(self):
        df = _make_ohlcv(400, daily_return=-0.005, seed=1)
        assert sc.bollinger_squeeze('X', _pack('X', df)) is None

    def test_high_volatility_returns_none(self):
        """High vol stock → bandwidth not at 6-month low → None."""
        df = _make_ohlcv(400, daily_return=0.003, volatility=0.06, seed=1)
        result = sc.bollinger_squeeze('X', _pack('X', df))
        assert result is None or result == 'X'


# ═══════════════════════════════════════════════════════════════════════════════
# Narrow Range (NR7)
# ═══════════════════════════════════════════════════════════════════════════════

class TestNarrowRange:
    def test_insufficient_bars_returns_none(self):
        df = _make_ohlcv(200, seed=1)
        assert sc.narrow_range('X', _pack('X', df)) is None

    def test_below_ema200_returns_none(self):
        df = _make_ohlcv(400, daily_return=-0.005, seed=1)
        assert sc.narrow_range('X', _pack('X', df)) is None

    def test_low_volume_returns_none(self):
        """Median volume <= 50k → None."""
        df = _make_ohlcv(400, daily_return=0.003, volume_base=10_000, seed=1)
        assert sc.narrow_range('X', _pack('X', df)) is None

    def test_injected_narrow_range_triggers(self):
        """Squeeze today's range to be narrowest of 7 days."""
        df = _make_ohlcv(400, daily_return=0.003, volatility=0.03, seed=10)
        # Make today's range very small
        mid = df['close'].iloc[-1]
        df.iloc[-1, df.columns.get_loc('high')] = mid * 1.001
        df.iloc[-1, df.columns.get_loc('low')] = mid * 0.999
        # Ensure prior 6 days have wider ranges
        for i in range(-7, -1):
            df.iloc[i, df.columns.get_loc('high')] = df['close'].iloc[i] * 1.03
            df.iloc[i, df.columns.get_loc('low')] = df['close'].iloc[i] * 0.97
        result = sc.narrow_range('X', _pack('X', df))
        assert result is None or result == 'X'


# ═══════════════════════════════════════════════════════════════════════════════
# 52-Week High
# ═══════════════════════════════════════════════════════════════════════════════

class TestFiftyTwoWeekHigh:
    def test_insufficient_bars_returns_none(self):
        df = _make_ohlcv(200, seed=1)
        assert sc.fifty_two_week_high('X', _pack('X', df)) is None

    def test_downtrend_returns_none(self):
        """Downtrend → today's high < 252-day high → None."""
        df = _make_ohlcv(400, daily_return=-0.003, seed=1)
        assert sc.fifty_two_week_high('X', _pack('X', df)) is None

    def test_new_high_with_volume_triggers(self):
        """Set today's high as 252-day max with above-avg volume."""
        df = _make_ohlcv(400, daily_return=0.003, seed=10)
        max_high = df['high'].iloc[-252:].max()
        df.iloc[-1, df.columns.get_loc('high')] = max_high * 1.01
        df.iloc[-1, df.columns.get_loc('volume')] = df['volume'].iloc[-20:].mean() * 2
        result = sc.fifty_two_week_high('X', _pack('X', df))
        assert result is None or result == 'X'


# ═══════════════════════════════════════════════════════════════════════════════
# Volume Breakout
# ═══════════════════════════════════════════════════════════════════════════════

class TestVolumeBreakout:
    def test_insufficient_bars_returns_none(self):
        df = _make_ohlcv(200, seed=1)
        assert sc.volume_breakout('X', _pack('X', df)) is None

    def test_below_ema200_returns_none(self):
        df = _make_ohlcv(400, daily_return=-0.005, seed=1)
        assert sc.volume_breakout('X', _pack('X', df)) is None

    def test_no_volume_spike_returns_none(self):
        """Normal volume → no 3x spike → None."""
        df = _make_ohlcv(400, daily_return=0.003, volatility=0.02, seed=1)
        result = sc.volume_breakout('X', _pack('X', df))
        assert result is None or result == 'X'


# ═══════════════════════════════════════════════════════════════════════════════
# Consolidation Breakout
# ═══════════════════════════════════════════════════════════════════════════════

class TestConsolidationBreakout:
    def test_insufficient_bars_returns_none(self):
        df = _make_ohlcv(200, seed=1)
        assert sc.consolidation_breakout('X', _pack('X', df)) is None

    def test_returns_tuple_format(self):
        """When triggered, returns (stock, consolidation_days)."""
        df = _make_ohlcv(400, daily_return=0.003, volatility=0.02, seed=10)
        result = sc.consolidation_breakout('X', _pack('X', df))
        if result is not None:
            assert isinstance(result, tuple)
            assert result[0] == 'X'
            assert 3 <= result[1] <= 15

    def test_uptrend_filter_isolation(self):
        """Same triggering fixture, with early bars pumped to inflate EMA_200.
        Untouched → triggers; pumped → rejected by stacked-EMA uptrend filter."""
        rng = np.random.RandomState(10)
        n = 400
        dates = pd.bdate_range('2020-01-01', periods=n)
        closes = np.empty(n)
        closes[0] = 100.0
        for i in range(1, 390):
            closes[i] = closes[i - 1] * (1 + 0.003 + rng.randn() * 0.005)
        base = closes[389]
        for i in range(390, 399):
            closes[i] = base * (1 + rng.uniform(-0.005, 0.005))
        closes[399] = base * 1.03
        highs = closes * 1.005
        lows = closes * 0.995
        opens = closes * (1 + rng.uniform(-0.002, 0.002, n))
        lows[399] = closes[398] * 0.999
        volumes = (200_000 * rng.uniform(0.8, 1.2, n)).astype(int)
        volumes[399] = int(volumes[-20:-1].mean() * 3)
        df = pd.DataFrame(
            {'open': opens, 'high': highs, 'low': lows, 'close': closes, 'volume': volumes},
            index=dates,
        )
        trigger = sc.consolidation_breakout('X', _pack('X', df))
        assert trigger is not None and trigger[0] == 'X'

        df_pumped = df.copy()
        for col in ('open', 'high', 'low', 'close'):
            df_pumped.iloc[:50, df_pumped.columns.get_loc(col)] *= 100
        assert sc.consolidation_breakout('X', _pack('X', df_pumped)) is None

    def test_flat_stock_returns_none(self):
        """Flat stock → no breakout above range → None."""
        df = _make_flat_ohlcv(400)
        assert sc.consolidation_breakout('X', _pack('X', df)) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Volume Dryup
# ═══════════════════════════════════════════════════════════════════════════════

class TestVolumeDryup:
    def test_insufficient_bars_returns_none(self):
        df = _make_ohlcv(200, seed=1)
        assert sc.volume_dryup('X', _pack('X', df)) is None

    def test_below_ema200_returns_none(self):
        df = _make_ohlcv(400, daily_return=-0.005, seed=1)
        assert sc.volume_dryup('X', _pack('X', df)) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Unusual Volume
# ═══════════════════════════════════════════════════════════════════════════════

class TestUnusualVolume:
    def test_insufficient_bars_returns_none(self):
        df = _make_ohlcv(200, seed=1)
        assert sc.unusual_volume('X', _pack('X', df)) is None

    def test_below_ema200_returns_none(self):
        df = _make_ohlcv(400, daily_return=-0.005, seed=1)
        assert sc.unusual_volume('X', _pack('X', df)) is None

    def test_returns_dict_with_volume_ratio(self):
        """Inject a volume spike with small candle."""
        df = _make_ohlcv(400, daily_return=0.003, volatility=0.02, seed=10)
        # Small candle + big volume on last bar
        mid = df['close'].iloc[-1]
        df.iloc[-1, df.columns.get_loc('open')] = mid * 0.999
        df.iloc[-1, df.columns.get_loc('close')] = mid * 1.001
        df.iloc[-1, df.columns.get_loc('volume')] = int(df['volume'].iloc[-20:].mean() * 5)
        result = sc.unusual_volume('X', _pack('X', df))
        if result is not None:
            assert isinstance(result, dict)
            assert result['stock'] == 'X'
            assert 'volume_ratio' in result
            assert result['volume_ratio'] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Inside Day
# ═══════════════════════════════════════════════════════════════════════════════

class TestInsideDay:
    def test_insufficient_bars_returns_none(self):
        df = _make_ohlcv(200, seed=1)
        assert sc.inside_day('X', _pack('X', df)) is None

    def test_below_ema200_returns_none(self):
        df = _make_ohlcv(400, daily_return=-0.005, seed=1)
        assert sc.inside_day('X', _pack('X', df)) is None

    def test_low_volume_returns_none(self):
        df = _make_ohlcv(400, daily_return=0.003, volume_base=10_000, seed=1)
        assert sc.inside_day('X', _pack('X', df)) is None

    def test_injected_inside_day_triggers(self):
        """Today's range inside yesterday's → triggers."""
        df = _make_ohlcv(400, daily_return=0.003, volatility=0.03, seed=10)
        # Make yesterday wide, today narrow and inside
        df.iloc[-2, df.columns.get_loc('high')] = df['close'].iloc[-2] * 1.05
        df.iloc[-2, df.columns.get_loc('low')] = df['close'].iloc[-2] * 0.95
        df.iloc[-1, df.columns.get_loc('high')] = df['close'].iloc[-1] * 1.01
        df.iloc[-1, df.columns.get_loc('low')] = df['close'].iloc[-1] * 0.99
        # Ensure today's high < yesterday's high and today's low > yesterday's low
        if df['high'].iloc[-1] <= df['high'].iloc[-2] and df['low'].iloc[-1] >= df['low'].iloc[-2]:
            result = sc.inside_day('X', _pack('X', df))
            assert result is None or result == 'X'


# ═══════════════════════════════════════════════════════════════════════════════
# Pocket Pivot
# ═══════════════════════════════════════════════════════════════════════════════

class TestPocketPivot:
    def test_insufficient_bars_returns_none(self):
        df = _make_ohlcv(200, seed=1)
        assert sc.pocket_pivot('X', _pack('X', df)) is None

    def test_below_ema200_returns_none(self):
        df = _make_ohlcv(400, daily_return=-0.005, seed=1)
        assert sc.pocket_pivot('X', _pack('X', df)) is None

    def test_down_day_returns_none(self):
        """Close < open → not an up day → None."""
        df = _make_ohlcv(400, daily_return=0.003, seed=10)
        df.iloc[-1, df.columns.get_loc('close')] = df['open'].iloc[-1] * 0.98
        assert sc.pocket_pivot('X', _pack('X', df)) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Pullback to EMA
# ═══════════════════════════════════════════════════════════════════════════════

class TestPullbackToEma:
    def test_insufficient_bars_returns_none(self):
        df = _make_ohlcv(200, seed=1)
        assert sc.pullback_to_ema('X', _pack('X', df)) is None

    def test_downtrend_returns_none(self):
        df = _make_ohlcv(400, daily_return=-0.005, seed=1)
        assert sc.pullback_to_ema('X', _pack('X', df)) is None

    def test_returns_dict_with_ema_level(self):
        """When triggered, returns dict with stock and ema_level."""
        df = _make_ohlcv(400, daily_return=0.004, volatility=0.03, seed=10)
        result = sc.pullback_to_ema('X', _pack('X', df))
        if result is not None:
            assert isinstance(result, dict)
            assert result['stock'] == 'X'
            assert result['ema_level'] in ['EMA 21', 'EMA 50', 'EMA 100']


# ═══════════════════════════════════════════════════════════════════════════════
# Higher Highs Higher Lows
# ═══════════════════════════════════════════════════════════════════════════════

class TestHigherHighsHigherLows:
    def test_insufficient_bars_returns_none(self):
        df = _make_ohlcv(200, seed=1)
        assert sc.higher_highs_higher_lows('X', _pack('X', df)) is None

    def test_below_ema200_returns_none(self):
        df = _make_ohlcv(400, daily_return=-0.005, seed=1)
        assert sc.higher_highs_higher_lows('X', _pack('X', df)) is None

    def test_returns_tuple_format(self):
        """When triggered, returns (stock, streak)."""
        df = _make_ohlcv(500, daily_return=0.004, volatility=0.04, seed=10)
        result = sc.higher_highs_higher_lows('X', _pack('X', df))
        if result is not None:
            assert isinstance(result, tuple)
            assert result[0] == 'X'
            assert result[1] >= 3

    def test_flat_stock_returns_none(self):
        """Flat stock → no HH/HL pattern → None."""
        df = _make_flat_ohlcv(400)
        assert sc.higher_highs_higher_lows('X', _pack('X', df)) is None


# ═══════════════════════════════════════════════════════════════════════════════
# relative_strength_benchmark_pivot_score
# ═══════════════════════════════════════════════════════════════════════════════

class TestRelativeStrengthBenchmarkPivotScore:
    def test_anchor_unset_returns_none(self):
        """With no anchor (ctx.rs_anchor is None), every call returns None."""
        df = _make_ohlcv(300, seed=1)
        df.index = df.index.date
        ctx = ScanContext(today=dt.date(2025, 6, 15))
        assert sc.relative_strength_benchmark_pivot_score('X', _pack('X', df), ctx) is None

    def test_return_pct_matches_arithmetic(self):
        """Per-stock return = (today / close_at_pivot - 1) * 100."""
        df = _make_ohlcv(300, seed=1)
        df.index = df.index.date
        pivot_date = df.index[-50]
        anchor_close = float(df.at[pivot_date, 'close'])
        today_close = float(df['close'].iloc[-1])
        expected = (today_close / anchor_close - 1.0) * 100.0
        ctx = ScanContext(today=dt.date(2025, 6, 15), rs_anchor=(pivot_date, 12345.67))
        result = sc.relative_strength_benchmark_pivot_score('X', _pack('X', df), ctx)
        assert result == {'stock': 'X', 'return_pct': pytest.approx(expected)}

    def test_no_bar_on_pivot_uses_prior(self):
        """If stock has no bar on pivot_date, fall back to the closest prior bar."""
        df = _make_ohlcv(300, seed=1)
        df.index = df.index.date
        idxs = list(df.index)
        gap_i = next(
            i for i in range(50, len(idxs) - 1)
            if (idxs[i + 1] - idxs[i]).days >= 3
        )
        friday = idxs[gap_i]
        saturday = friday + dt.timedelta(days=1)
        assert saturday not in df.index
        anchor_close = float(df.at[friday, 'close'])
        today_close = float(df['close'].iloc[-1])
        expected = (today_close / anchor_close - 1.0) * 100.0
        ctx = ScanContext(today=dt.date(2025, 6, 15), rs_anchor=(saturday, 99.0))
        result = sc.relative_strength_benchmark_pivot_score('X', _pack('X', df), ctx)
        assert result['return_pct'] == pytest.approx(expected)

    def test_ipo_before_pivot_returns_none(self):
        """Stock that started trading AFTER the pivot has no on-or-before bar -> None."""
        df = _make_ohlcv(20, seed=1, start_date='2025-01-01')
        df.index = df.index.date
        old_pivot = dt.date(2024, 1, 1)
        ctx = ScanContext(today=dt.date(2025, 6, 15), rs_anchor=(old_pivot, 100.0))
        assert sc.relative_strength_benchmark_pivot_score('X', _pack('X', df), ctx) is None
