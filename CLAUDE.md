# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Pure-compute daily technical scans for equity universes — the scan **math** only
(cores, liquidity filters, post-processors, a benchmark-pivot anchor, and a
per-batch runner). No IO, no logging, no market-specific identity.

Extracted from `finance_website` so the Indian and US market pipelines can share
one source of truth for scan computation, mirroring `indian-market-calendar`.

## Commands

```bash
uv run pytest -q                                  # full suite
uv run pytest tests/test_scan_cores.py -q         # one file
uv run pytest tests/test_runner.py::test_name -q  # one test
uv run pytest -k consolidation -q                 # match by keyword
.venv/bin/python -m compileall -q daily_scans     # syntax check after edits
uv sync                                            # install deps (incl. test extra: uv sync --extra test)
```

Requires Python ≥3.11. Runtime deps are `pandas`, `numpy`, `TA-Lib` only —
deliberately **no `pyarrow`**, because the package is IO-free (parquet reads live
in the host repo). `conftest.py` puts both the package root and `tests/` on
`sys.path`, so the suite runs without an editable install and `import _helpers`
resolves.

## Module map

- `scan_cores.py` (~1.9k LoC) — every per-stock scan algorithm + shared
  primitives (`_zigzag_pivots`, `_ctx`). The bulk of the logic lives here.
- `post_processors.py` — pure row→output reducers (see Adding a new scan).
- `registry.py` — `ALL_SCANS` catalog + `build_registry` selection builder.
- `filters.py` — the parametrized `common_filters` liquidity gate.
- `anchor.py` — `compute_rs_anchor`, the index-agnostic benchmark-pivot finder.
- `context.py` — `ScanContext` (injected run-state), `ScanSpec`, `FilesKind`.
- `runner.py` — `build_batch_data_dict` → `run_scans_on_batch` → `finalize`.
- `tests/_helpers.py` — synthetic OHLCV/benchmark builders all tests draw on.

## Data flow (per batch)

The host repo drives one batch through the runner in this order:

1. `build_batch_data_dict(batch_df)` → `{symbol -> date-indexed OHLCV df}`
   (index is python `date` objects; dotted/empty symbols dropped).
2. `run_scans_on_batch(data_dict, selected, filter_configs, ctx, error_handler)`
   — for each `FILTERED`-kind spec it computes `common_filters` **once per
   distinct `filter_key`** and runs the core over that universe; `ALL`-kind specs
   run over every symbol with no liquidity gate. Per-stock `KeyError/IndexError/
   ValueError/ZeroDivisionError` are routed to the injected `error_handler`
   (duck-typed: `record_success`/`record_failure`/`log`, each taking the spec).
   Returns `{output_name: raw_rows}`.
3. `finalize(accumulated, selected)` applies each spec's `post_process` →
   `{output_name: list[str] | (symbols, extras)}`.

`output_name` is the CSV stem **chosen by the repo** (defaults to the
market-neutral `name`); the repo writes the CSVs.

## Boundary

The package owns **pure computation**; the host repo owns **all IO, logging,
orchestration, and injected state**:

- Parquet reads / streaming, CSV writes, symbol normalization → repo.
- Liquidity thresholds (`LiquidityFilterConfig`), the benchmark index choice,
  and the per-scan output CSV names → supplied by the repo.
- The package carries **no** "Nifty", exchange, or currency. The
  relative-strength-vs-pivot scan takes an abstract
  `ScanContext.rs_anchor = (pivot_date, benchmark_close)`; the repo decides the
  benchmark.
- `rs_high_before_price_high` takes an abstract
  `ScanContext.benchmark_close` — a date-indexed `pd.Series` of the benchmark's
  close (python `date` keys, matching what `build_batch_data_dict` produces).
  Same neutrality contract as `rs_anchor`: the package never knows which index;
  the repo builds and injects it.

## Public API

```python
from daily_scans import (
    ScanContext, ScanSpec, FilesKind,
    LiquidityFilterConfig, common_filters,
    compute_rs_anchor,
    ALL_SCANS, get_scan, build_registry,
    run_scans_on_batch, finalize, build_batch_data_dict, apply_core,
    scan_cores, post_processors,
)
```

## Cores and `ctx`

Every core has signature `fn(stock, data_dict, ctx=None)`. Only 6 cores read
`ctx`: the 4 ATH cores use `ctx.today`,
`relative_strength_benchmark_pivot_score` uses `ctx.rs_anchor`, and
`rs_high_before_price_high` uses `ctx.benchmark_close`. When `ctx is None` the
today-sensitive cores fall back to a **local** `date.today()` (the package stays
calendar-free) — so a direct caller of a today-sensitive core that wants IST
semantics must pass `ctx` explicitly. The two benchmark-fed cores instead no-op
to `None` when their abstract input is absent. The runner always passes `ctx`.
A ctx-dependent core returns `None` for every stock when its injected input is
absent (silent empty output, not an error) — when adding such a scan, document
the required `ctx` input on the core.

## Adding a new scan — all four are required

A new core is NOT usable until every step below is done. Missing the registry
entry makes `build_registry()` raise `KeyError`; missing the post-processor
breaks the CSV writer when the core returns anything other than a bare symbol.

1. **Core** — `scan_cores.py`: `def my_scan(stock, data_dict, ctx=None) -> value | None`.
   Return `None` to skip a stock. Read `ctx` only if the scan needs today/benchmark;
   if it requires injected `ctx` state, document that input (see Cores and ctx) —
   absent state means silent empty output, not an error.
2. **Post-processor** — `post_processors.py`:
   - Core returns a **bare symbol** → reuse `identity_pp` (no new function).
   - Core returns a **dict or tuple** → a real pp is **mandatory**; it reduces the
     rows to a symbol list, decides sort order, and may return `(symbols, extras)`
     to surface extra columns. It **cannot** ride `identity_pp`.
3. **Register** — `registry.py`: add `"my_scan": ScanDef(scan_cores.my_scan, post_processors.my_pp)`
   to `ALL_SCANS`. A scan absent from `ALL_SCANS` is invisible to `build_registry`.
4. **Tests** — `tests/`: a core test in `test_scan_cores.py` (deterministic guard
   branches at minimum; a triggering fixture if cheap to build), a pp test in
   `test_post_processors.py` for any non-`identity_pp` (feed synthetic rows
   directly), and a `"my_scan" in ALL_SCANS` assertion in `test_runner.py`.

## Not this repo's job

Consuming repos own the **ordered selection**, per-market **output CSV names**,
validation, and any **UI columns**. Adding a scan here does not make it run
anywhere — the host repo must also wire it into its selection (and frontend).

## Tests

```bash
uv run pytest -q
```
