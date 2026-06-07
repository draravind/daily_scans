"""Scan catalog (market-NEUTRAL algorithm name → core + post-processor) and the
selection builder. The package owns the catalog; the repo owns the ordered
selection AND the per-scan output CSV name (e.g. India maps catalog
`relative_strength_benchmark_pivot` → output `relative_strength_nifty_pivot`).
"""
from dataclasses import dataclass
from typing import Callable

from daily_scans import scan_cores, post_processors
from daily_scans.context import ScanSpec, FilesKind


@dataclass(frozen=True)
class ScanDef:
    core: Callable
    post_process: Callable


ALL_SCANS: dict[str, ScanDef] = {
    "new_ATH": ScanDef(scan_cores.making_new_ATH_high_ATR, post_processors.identity_pp),
    "near_ATH": ScanDef(scan_cores.near_ATH_high_ATR, post_processors.identity_pp),
    "high_tight_flags_stocks": ScanDef(scan_cores.high_tight_flag, post_processors.high_tight_flag_pp),
    "gapups": ScanDef(scan_cores.gapups, post_processors.identity_pp),
    "relative_strength": ScanDef(scan_cores.relative_strength_score, post_processors.relative_strength_pp),
    "relative_strength_benchmark_pivot": ScanDef(
        scan_cores.relative_strength_benchmark_pivot_score,
        post_processors.relative_strength_benchmark_pivot_pp,
    ),
    "smooth_stocks": ScanDef(scan_cores.smooth_stocks, post_processors.smooth_stocks_pp),
    "ipo_stocks": ScanDef(scan_cores.ipo, post_processors.ipo_pp),
    "highest_avg_turnover": ScanDef(scan_cores.avg_turnover_ranked, post_processors.highest_avg_turnover_pp),
    "golden_cross": ScanDef(scan_cores.golden_cross, post_processors.identity_pp),
    "macd_bullish_crossover": ScanDef(scan_cores.macd_bullish_crossover, post_processors.identity_pp),
    "bullish_engulfing": ScanDef(scan_cores.bullish_engulfing, post_processors.identity_pp),
    "hammer": ScanDef(scan_cores.hammer, post_processors.identity_pp),
    "morning_star": ScanDef(scan_cores.morning_star, post_processors.identity_pp),
    "doji": ScanDef(scan_cores.doji, post_processors.identity_pp),
    "bollinger_squeeze": ScanDef(scan_cores.bollinger_squeeze, post_processors.identity_pp),
    "narrow_range": ScanDef(scan_cores.narrow_range, post_processors.identity_pp),
    "52_week_high": ScanDef(scan_cores.fifty_two_week_high, post_processors.identity_pp),
    "volume_breakout": ScanDef(scan_cores.volume_breakout, post_processors.identity_pp),
    "volume_dryup": ScanDef(scan_cores.volume_dryup, post_processors.identity_pp),
    "inside_day": ScanDef(scan_cores.inside_day, post_processors.identity_pp),
    "pocket_pivot": ScanDef(scan_cores.pocket_pivot, post_processors.identity_pp),
    "pullback_to_ema": ScanDef(scan_cores.pullback_to_ema, post_processors.pullback_to_ema_pp),
    "higher_highs_higher_lows": ScanDef(
        scan_cores.higher_highs_higher_lows, post_processors.higher_highs_higher_lows_pp),
    "stage_2_advancing": ScanDef(scan_cores.stage_2_advancing, post_processors.stage_2_advancing_pp),
    "highest_turnover": ScanDef(scan_cores.highest_turnover, post_processors.highest_turnover_pp),
    "base_breakouts": ScanDef(scan_cores.base_breakout, post_processors.base_breakouts_pp),
    "top_movers": ScanDef(scan_cores.top_movers, post_processors.top_movers_pp),
    "unusual_volume": ScanDef(scan_cores.unusual_volume, post_processors.unusual_volume_pp),
    "consolidation_breakout": ScanDef(scan_cores.consolidation_breakout, post_processors.consolidation_breakout_pp),
    "consolidation": ScanDef(scan_cores.consolidation, post_processors.consolidation_pp),
    "rs_high_before_price_high": ScanDef(
        scan_cores.rs_high_before_price_high, post_processors.identity_pp),
}


def get_scan(name: str) -> ScanDef:
    return ALL_SCANS[name]


def build_registry(selection: list[tuple]) -> list[ScanSpec]:
    """Repo passes ordered tuples (algo_name, files_kind, filter_key[, output_name]);
    output_name defaults to algo_name when omitted. Returns list[ScanSpec]."""
    specs = []
    for entry in selection:
        name, fk, key = entry[0], entry[1], entry[2]
        out = entry[3] if len(entry) > 3 else name
        d = ALL_SCANS[name]
        specs.append(ScanSpec(name, d.core, d.post_process, fk, key, output_name=out))
    return specs
