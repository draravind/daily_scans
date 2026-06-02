"""Per-batch scan runner + finalize — pure. All IO/logging stays in the repo;
the repo injects a duck-typed `error_handler` (record_success/record_failure/
log, each taking the ScanSpec) wrapping its FailureTracker + error logger.
"""
from daily_scans.context import FilesKind
from daily_scans.filters import common_filters


def apply_core(core, stock, data_dict, ctx=None):
    """Single core invocation (used by chart_patterns' bespoke row-shaping too)."""
    return core(stock, data_dict, ctx)


def build_batch_data_dict(batch_df):
    """Convert a flat batch frame into a {symbol -> per-symbol date-indexed df} map.

    Sets the index to `date` (converted to python `date` objects, since cores
    compare against python date bounds) and drops dotted/empty symbols.
    """
    data_dict = {}
    for symbol, data in batch_df.groupby('symbol', observed=True):
        symbol_str = str(symbol)
        if data.empty or '.' in symbol_str:
            continue
        data = data.set_index('date').sort_index()
        data.index = data.index.date
        data_dict[symbol_str] = data
    return data_dict


def run_scans_on_batch(data_dict, selected, filter_configs, ctx, error_handler=None):
    """Run every selected spec over its universe, collecting raw rows.

    FILTERED-kind specs run over `common_filters(all_symbols, data_dict,
    filter_configs[spec.filter_key])` — computed ONCE per distinct filter_key.
    ALL-kind specs run over every batch symbol (no liquidity gate). Each core is
    called as `core(stock, data_dict, ctx)`. Per-stock errors of the expected
    types are routed to `error_handler`. Returns {spec.output_name: rows}.
    """
    all_symbols = list(data_dict.keys())

    filtered = {}
    for spec in selected:
        if spec.files_kind == FilesKind.FILTERED and spec.filter_key not in filtered:
            filtered[spec.filter_key] = common_filters(
                all_symbols, data_dict, filter_configs[spec.filter_key]
            )

    results = {}
    for spec in selected:
        universe = filtered[spec.filter_key] if spec.files_kind == FilesKind.FILTERED else all_symbols
        rows = []
        for stock in universe:
            try:
                value = spec.core(stock, data_dict, ctx)
                if value is not None:
                    rows.append(value)
                if error_handler is not None:
                    error_handler.record_success(spec)
            except (KeyError, IndexError, ValueError, ZeroDivisionError) as e:
                if error_handler is not None:
                    error_handler.log(spec, stock, e)
                    error_handler.record_failure(spec)
        results[spec.output_name] = rows
    return results


def finalize(accumulated, selected):
    """Apply each spec's post_process to its accumulated rows.
    Returns {spec.output_name: result}."""
    return {
        spec.output_name: spec.post_process(accumulated[spec.output_name])
        for spec in selected
    }
