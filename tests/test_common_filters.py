"""Tests for daily_scans.filters.common_filters — the parametrized liquidity gate.

Exercises the mechanics (insufficient-bars, penny, low-volume, downtrend,
missing-stock skip) with an India-like config, plus the optional uptrend branch.
"""
from dataclasses import replace

from daily_scans import common_filters

from _helpers import _make_ohlcv, _india_like_config


CFG = _india_like_config()


class TestCommonFilters:
    def test_returns_list(self):
        data_dict = {'A': _make_ohlcv(300, volume_base=500_000, start_close=50, seed=1)}
        result = common_filters(['A'], data_dict, CFG)
        assert isinstance(result, list)

    def test_insufficient_bars_excluded(self):
        """Stocks with < min_bars bars are silently skipped."""
        data_dict = {'SHORT': _make_ohlcv(100, seed=1)}
        assert common_filters(['SHORT'], data_dict, CFG) == []

    def test_penny_stock_excluded(self):
        """Close < price_min → excluded."""
        data_dict = {'PENNY': _make_ohlcv(300, start_close=5.0, daily_return=0.001, seed=1)}
        assert 'PENNY' not in common_filters(['PENNY'], data_dict, CFG)

    def test_low_volume_excluded(self):
        """Very low volume → fails turnover and volume conditions."""
        data_dict = {'LOW_VOL': _make_ohlcv(300, volume_base=1_000, start_close=50, seed=1)}
        assert 'LOW_VOL' not in common_filters(['LOW_VOL'], data_dict, CFG)

    def test_liquid_uptrend_included(self):
        data_dict = {'A': _make_ohlcv(300, daily_return=0.002, volume_base=600_000, start_close=200, seed=1)}
        assert 'A' in common_filters(['A'], data_dict, CFG)

    def test_missing_stock_skipped(self):
        """KeyError for missing stock silently skipped."""
        data_dict = {'A': _make_ohlcv(300, seed=1)}
        result = common_filters(['A', 'MISSING'], data_dict, CFG)
        assert isinstance(result, list)


class TestUptrendBranch:
    """enable_uptrend_filter=True adds max(close, EMA21, EMA50) >= EMA200 gate."""

    def test_downtrend_liquid_stock_gated_only_by_uptrend(self):
        # Liquid (high turnover/volume, well above penny) but firmly downtrending,
        # so close/EMA21/EMA50 sit below EMA200.
        data_dict = {
            'BEAR': _make_ohlcv(300, daily_return=-0.002, volume_base=800_000, start_close=300, seed=3)
        }
        no_uptrend = replace(CFG, enable_uptrend_filter=False)
        with_uptrend = replace(CFG, enable_uptrend_filter=True)

        assert 'BEAR' in common_filters(['BEAR'], data_dict, no_uptrend)
        assert 'BEAR' not in common_filters(['BEAR'], data_dict, with_uptrend)

    def test_uptrend_stock_passes_with_filter_on(self):
        data_dict = {
            'BULL': _make_ohlcv(300, daily_return=0.003, volume_base=600_000, start_close=200, seed=4)
        }
        with_uptrend = replace(CFG, enable_uptrend_filter=True)
        assert 'BULL' in common_filters(['BULL'], data_dict, with_uptrend)
