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

Every core has signature `fn(stock, data_dict, ctx=None)`. Only 5 cores read
`ctx`: the 4 ATH cores use `ctx.today`, and
`relative_strength_benchmark_pivot_score` uses `ctx.rs_anchor`. When `ctx is
None` they fall back to a **local** `date.today()` (the package stays
calendar-free) — so a direct caller of a today-sensitive core that wants IST
semantics must pass `ctx` explicitly. The runner always passes `ctx`.

## Tests

```bash
uv run pytest -q
```
