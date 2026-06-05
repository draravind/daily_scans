# daily-scans

Pure-compute daily technical scans for equity universes — the scan **math** only
(cores, liquidity filters, post-processors, a benchmark-pivot anchor, and a
per-batch runner). No IO, no logging, no market-specific identity.

Extracted from `finance_website` so the Indian and US market pipelines can share
one source of truth for scan computation, mirroring `indian-market-calendar`.

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
  the repo builds and injects it. See **Repo-side integration** below.

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

## Repo-side integration

### `rs_high_before_price_high` (needs an injected benchmark series)

This scan flags stocks whose relative-strength line (`close / benchmark_close`)
set a new ~252-day high within the last 5 bars while price did **not** (price
topped earlier and now sits within 15% below that high). It reads the benchmark
from `ctx.benchmark_close`; with no series injected the core returns `None` for
every stock, so the host repo must wire two things in the same rollout:

1. **Build and inject the benchmark close series.** From the index OHLC the repo
   already loads for `compute_rs_anchor`, build a date-indexed `pd.Series` of the
   benchmark close — python `date` keys matching `build_batch_data_dict` (e.g.
   `series.index = series.index.date`) — and pass it on the context:

   ```python
   ctx = ScanContext(today=..., rs_anchor=..., benchmark_close=bench_close)
   ```

   The package never knows which index this is (Nifty 50 / NASDAQ); the repo
   decides, exactly as it already does for `rs_anchor`.

2. **Add the scan to the ordered selection.** Recommend `FilesKind.FILTERED`
   with the same `filter_key` as the other RS scans, and per-market output names:

   ```python
   ("rs_high_before_price_high", FilesKind.FILTERED, "default", "rs_high_before_price_high_nifty")
   ("rs_high_before_price_high", FilesKind.FILTERED, "default", "rs_high_before_price_high_nasdaq")
   ```

Output is a plain symbol list (`identity_pp`).

## Tests

```bash
uv run pytest -q
```
