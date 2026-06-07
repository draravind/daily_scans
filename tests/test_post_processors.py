"""Tests for daily_scans.post_processors via core+pp assembly.

Each test runs a core over a synthetic data_dict to collect raw rows, then
applies the matching post-processor — asserting the same ranking / cap /
section-header / classification invariants the repo relied on.
"""
import pandas as pd
import pytest

from daily_scans import scan_cores, post_processors

from _helpers import _make_ohlcv


def _rows(core, syms, dd):
    return [r for s in syms if (r := core(s, dd)) is not None]


# ── ema_contraction + ema_contraction_pp ─────────────────────────────────────

class TestEMAContraction:
    def test_returns_dataframe(self):
        dd = {'A': _make_ohlcv(300, seed=1)}
        result = post_processors.ema_contraction_pp(_rows(scan_cores.ema_contraction, ['A'], dd))
        assert isinstance(result, pd.DataFrame)

    def test_empty_input_returns_empty_df(self):
        result = post_processors.ema_contraction_pp([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_columns_present(self):
        dd = {'A': _make_ohlcv(300, seed=1), 'B': _make_ohlcv(300, seed=2)}
        result = post_processors.ema_contraction_pp(_rows(scan_cores.ema_contraction, ['A', 'B'], dd))
        assert 'contraction' in result.columns
        assert 'contraction_atr' in result.columns
        assert len(result) == 2


# ── ipo + ipo_pp ─────────────────────────────────────────────────────────────

class TestIPOs:
    def test_returns_tuple_with_extras(self):
        dd = {
            'IPO_6M': _make_ohlcv(80, seed=1),
            'IPO_12M': _make_ohlcv(200, seed=2),
            'OLD': _make_ohlcv(300, seed=3),
        }
        result = post_processors.ipo_pp(_rows(scan_cores.ipo, ['IPO_6M', 'IPO_12M', 'OLD'], dd))
        assert isinstance(result, tuple)
        symbols, extras = result
        assert isinstance(symbols, list)
        assert isinstance(extras, dict)

    def test_classification_correct(self):
        dd = {'IPO_6M': _make_ohlcv(80, seed=1), 'IPO_12M': _make_ohlcv(200, seed=2)}
        symbols, extras = post_processors.ipo_pp(_rows(scan_cores.ipo, ['IPO_6M', 'IPO_12M'], dd))
        assert 'IPO_6M' in symbols
        assert 'IPO_12M' in symbols
        assert extras['IPO_6M']['ipo_6m'] == 1
        assert extras['IPO_6M']['ipo_12m'] == 0
        assert extras['IPO_12M']['ipo_6m'] == 0
        assert extras['IPO_12M']['ipo_12m'] == 1

    def test_old_stocks_excluded(self):
        dd = {'OLD': _make_ohlcv(300, seed=1)}
        assert post_processors.ipo_pp(_rows(scan_cores.ipo, ['OLD'], dd)) == []


# ── avg_turnover_ranked + highest_avg_turnover_pp ────────────────────────────

class TestHighestAvgTurnover:
    def test_empty_returns_empty(self):
        assert post_processors.highest_avg_turnover_pp([]) == []

    def test_includes_section_headers(self):
        dd = {'A': _make_ohlcv(300, seed=1)}
        result = post_processors.highest_avg_turnover_pp(
            _rows(scan_cores.avg_turnover_ranked, ['A'], dd))
        assert '###QUARTER' in result
        assert '###ALL TIME' in result

    def test_insufficient_bars_excluded(self):
        dd = {'SHORT': _make_ohlcv(50, seed=1)}
        assert post_processors.highest_avg_turnover_pp(
            _rows(scan_cores.avg_turnover_ranked, ['SHORT'], dd)) == []


# ── relative_strength_score + relative_strength_pp ───────────────────────────

class TestRelativeStrength:
    def test_returns_tuple_with_extras(self):
        dd = {f'S{i}': _make_ohlcv(300, daily_return=0.001 * i, seed=i) for i in range(1, 11)}
        result = post_processors.relative_strength_pp(
            _rows(scan_cores.relative_strength_score, list(dd), dd))
        assert isinstance(result, tuple)
        symbols, extras = result
        assert len(symbols) == len(dd)
        assert all(s in extras for s in symbols)
        for s in symbols:
            assert 1 <= extras[s]['rs_rank'] <= 99
            assert isinstance(extras[s]['rs_rank'], int)

    def test_sorted_by_rank_descending(self):
        dd = {f'S{i}': _make_ohlcv(300, daily_return=0.001 * i, seed=i) for i in range(1, 11)}
        symbols, extras = post_processors.relative_strength_pp(
            _rows(scan_cores.relative_strength_score, list(dd), dd))
        ranks = [extras[s]['rs_rank'] for s in symbols]
        assert ranks == sorted(ranks, reverse=True)

    def test_caps_at_200(self):
        dd = {f'S{i:03d}': _make_ohlcv(300, daily_return=0.0005 * i, seed=i) for i in range(1, 251)}
        symbols, _extras = post_processors.relative_strength_pp(
            _rows(scan_cores.relative_strength_score, list(dd), dd))
        assert len(symbols) == 200

    def test_short_history_skipped(self):
        dd = {
            'LONG': _make_ohlcv(300, daily_return=0.002, seed=1),
            'SHORT': _make_ohlcv(100, daily_return=0.002, seed=2),
        }
        symbols, _extras = post_processors.relative_strength_pp(
            _rows(scan_cores.relative_strength_score, ['LONG', 'SHORT'], dd))
        assert 'LONG' in symbols
        assert 'SHORT' not in symbols


# ── relative_strength_benchmark_pivot_pp (direct) ────────────────────────────

class TestRelativeStrengthBenchmarkPivotPP:
    def test_empty_rows_returns_empty_list(self):
        assert post_processors.relative_strength_benchmark_pivot_pp([]) == []

    def test_extras_have_rank_and_return(self):
        rows = [
            {'stock': 'A', 'return_pct': 50.0},
            {'stock': 'B', 'return_pct': 20.0},
            {'stock': 'C', 'return_pct': -10.0},
        ]
        symbols, extras = post_processors.relative_strength_benchmark_pivot_pp(rows)
        assert symbols == ['A', 'B', 'C']
        for sym in symbols:
            assert set(extras[sym].keys()) == {'rs_rank', 'return_pct'}
            assert isinstance(extras[sym]['rs_rank'], int)
            assert 1 <= extras[sym]['rs_rank'] <= 99

    def test_rank_monotonic_with_return(self):
        rows = [{'stock': f'S{i:03d}', 'return_pct': float(i)} for i in range(50)]
        symbols, extras = post_processors.relative_strength_benchmark_pivot_pp(rows)
        ranks = [extras[s]['rs_rank'] for s in symbols]
        assert ranks == sorted(ranks, reverse=True)

    def test_ties_get_same_rank(self):
        rows = [
            {'stock': 'A', 'return_pct': 10.0},
            {'stock': 'B', 'return_pct': 10.0},
            {'stock': 'C', 'return_pct': 5.0},
        ]
        _, extras = post_processors.relative_strength_benchmark_pivot_pp(rows)
        assert extras['A']['rs_rank'] == extras['B']['rs_rank']

    def test_top_200_cap(self):
        rows = [{'stock': f'S{i:04d}', 'return_pct': float(i)} for i in range(500)]
        symbols, _ = post_processors.relative_strength_benchmark_pivot_pp(rows)
        assert len(symbols) == 200
        assert symbols[0] == 'S0499'
        assert symbols[-1] == 'S0300'


# ── consolidation_pp (direct) ────────────────────────────────────────────────

class TestConsolidationPP:
    def test_empty_rows_returns_empty_list(self):
        assert post_processors.consolidation_pp([]) == []

    def test_returns_tuple_with_extras(self):
        rows = [
            {'stock': 'A', 'consolidation_days': 10, 'tightness': 0.30},
            {'stock': 'B', 'consolidation_days': 6, 'tightness': 0.10},
        ]
        result = post_processors.consolidation_pp(rows)
        assert isinstance(result, tuple)
        symbols, extras = result
        assert isinstance(symbols, list)
        assert isinstance(extras, dict)
        for sym in symbols:
            assert set(extras[sym].keys()) == {'consolidation_days', 'tightness'}

    def test_sorted_by_tightness_ascending(self):
        rows = [
            {'stock': 'LOOSE', 'consolidation_days': 8, 'tightness': 0.90},
            {'stock': 'TIGHT', 'consolidation_days': 5, 'tightness': 0.05},
            {'stock': 'MID', 'consolidation_days': 7, 'tightness': 0.40},
        ]
        symbols, _ = post_processors.consolidation_pp(rows)
        assert symbols == ['TIGHT', 'MID', 'LOOSE']

    def test_extras_carry_days_and_tightness(self):
        rows = [{'stock': 'A', 'consolidation_days': 12, 'tightness': 0.123456}]
        symbols, extras = post_processors.consolidation_pp(rows)
        assert symbols == ['A']
        assert extras['A']['consolidation_days'] == 12
        assert extras['A']['tightness'] == 0.1235  # rounded to 4 dp
