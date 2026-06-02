"""Tests for daily_scans.runner — the per-batch orchestration primitives that
have no pre-existing coverage: run_scans_on_batch (universe selection +
error_handler hook), finalize, and build_batch_data_dict.
"""
import pandas as pd

from daily_scans import (
    ScanSpec, FilesKind, ScanContext,
    run_scans_on_batch, finalize, build_batch_data_dict,
    common_filters,
)

from _helpers import _make_ohlcv, _india_like_config


CFG = _india_like_config()
CTX = ScanContext(today=None)  # cores used here don't read ctx.today


def _recorder():
    seen = []

    def core(stock, data_dict, ctx=None):
        seen.append(stock)
        return stock
    return core, seen


def _make_batch():
    """2 liquid stocks + 1 illiquid penny → common_filters keeps the 2 liquid."""
    return {
        'LIQ1': _make_ohlcv(300, daily_return=0.001, volume_base=600_000, start_close=200, seed=1),
        'LIQ2': _make_ohlcv(300, daily_return=0.0008, volume_base=700_000, start_close=250, seed=2),
        'PENNY': _make_ohlcv(300, daily_return=0.0, volume_base=900, start_close=5, seed=3),
    }


class TestRunScansOnBatch:
    def test_all_kind_sees_full_universe_filtered_sees_survivors(self):
        dd = _make_batch()
        all_core, all_seen = _recorder()
        filt_core, filt_seen = _recorder()
        selected = [
            ScanSpec('all_scan', all_core, lambda r: r, FilesKind.ALL, 'default'),
            ScanSpec('filt_scan', filt_core, lambda r: r, FilesKind.FILTERED, 'default'),
        ]
        results = run_scans_on_batch(dd, selected, {'default': CFG}, CTX)

        survivors = set(common_filters(list(dd), dd, CFG))
        assert survivors == {'LIQ1', 'LIQ2'}
        assert set(all_seen) == set(dd)          # ALL-kind: every symbol
        assert set(filt_seen) == survivors        # FILTERED-kind: only survivors
        assert set(results['all_scan']) == set(dd)
        assert set(results['filt_scan']) == survivors

    def test_results_keyed_by_output_name(self):
        dd = _make_batch()
        core, _ = _recorder()
        selected = [
            ScanSpec('algo', core, lambda r: r, FilesKind.ALL, 'default', output_name='custom_csv'),
        ]
        results = run_scans_on_batch(dd, selected, {'default': CFG}, CTX)
        assert 'custom_csv' in results
        assert 'algo' not in results

    def test_filter_computed_once_per_key(self):
        dd = _make_batch()
        c1, _ = _recorder()
        c2, _ = _recorder()
        selected = [
            ScanSpec('a', c1, lambda r: r, FilesKind.FILTERED, 'default'),
            ScanSpec('b', c2, lambda r: r, FilesKind.FILTERED, 'default'),
        ]
        # Both share filter_key 'default'; must not raise and both see survivors.
        results = run_scans_on_batch(dd, selected, {'default': CFG}, CTX)
        survivors = set(common_filters(list(dd), dd, CFG))
        assert set(results['a']) == survivors
        assert set(results['b']) == survivors


class TestErrorHandler:
    def test_raising_core_routes_to_handler(self):
        class Handler:
            def __init__(self):
                self.successes, self.failures, self.logs = [], [], []

            def record_success(self, spec):
                self.successes.append(spec.output_name)

            def record_failure(self, spec):
                self.failures.append(spec.output_name)

            def log(self, spec, stock, exc):
                self.logs.append((spec.output_name, stock, type(exc).__name__))

        dd = {'A': _make_ohlcv(300, seed=1), 'B': _make_ohlcv(300, seed=2)}

        def boom(stock, data_dict, ctx=None):
            raise KeyError(stock)

        handler = Handler()
        selected = [ScanSpec('boom', boom, lambda r: r, FilesKind.ALL, 'default')]
        results = run_scans_on_batch(dd, selected, {'default': CFG}, CTX, error_handler=handler)

        assert results['boom'] == []
        assert set(handler.failures) == {'boom'}
        assert len(handler.failures) == 2          # one per stock
        assert handler.successes == []
        assert {s for _, s, _ in handler.logs} == {'A', 'B'}
        assert all(kind == 'KeyError' for _, _, kind in handler.logs)


class TestFinalize:
    def test_dispatches_to_post_processors(self):
        selected = [
            ScanSpec('x', lambda *a: None, lambda rows: ['###H', *rows], FilesKind.ALL, 'default'),
            ScanSpec('y', lambda *a: None, lambda rows: rows[::-1], FilesKind.FILTERED, 'default',
                     output_name='y_out'),
        ]
        accumulated = {'x': ['p', 'q'], 'y_out': [1, 2, 3]}
        out = finalize(accumulated, selected)
        assert out['x'] == ['###H', 'p', 'q']
        assert out['y_out'] == [3, 2, 1]


class TestBuildBatchDataDict:
    def test_reshapes_and_drops_dotted(self):
        rows = []
        for sym in ['AAA', 'BBB', 'C.C']:
            for i, d in enumerate(pd.bdate_range('2024-01-01', periods=3)):
                rows.append({'symbol': sym, 'date': d, 'open': 1.0 + i, 'high': 2.0,
                             'low': 0.5, 'close': 1.5, 'volume': 100})
        batch = pd.DataFrame(rows)
        dd = build_batch_data_dict(batch)
        assert set(dd) == {'AAA', 'BBB'}           # dotted symbol dropped
        frame = dd['AAA']
        # 'symbol' is retained (groupby keeps the key column); 'date' becomes the index.
        assert {'open', 'high', 'low', 'close', 'volume'}.issubset(frame.columns)
        assert 'date' not in frame.columns
        import datetime as _dt
        assert all(isinstance(ix, _dt.date) for ix in frame.index)   # python dates, not datetime64
        assert len(frame) == 3
