"""Injected run-state + scan descriptors. Market-neutral by construction."""
from dataclasses import dataclass
from enum import Enum
import datetime as dt
from typing import Callable


class FilesKind(str, Enum):
    FILTERED = "filtered"
    ALL = "all"


@dataclass(frozen=True)
class ScanContext:
    """Per-run injected state — replaces the removed ist_today import + the
    removed module-global benchmark anchor. rs_anchor is an ABSTRACT benchmark
    pivot — the package never knows the benchmark is Nifty."""
    today: dt.date
    rs_anchor: tuple[dt.date, float] | None = None   # (pivot_date, benchmark_close_at_pivot)


@dataclass(frozen=True)
class ScanSpec:
    name: str                      # algorithm / catalog key (market-neutral)
    core: Callable                 # fn(stock, data_dict, ctx=None) -> value | None
    post_process: Callable         # fn(rows) -> list[str] | (symbols, extras)
    files_kind: FilesKind
    filter_key: str = "default"    # which LiquidityFilterConfig gates this scan
    output_name: str = ""          # CSV stem chosen by the REPO; falls back to `name`

    def __post_init__(self):
        # Authoritative default-resolution so EVERY construction path
        # (build_registry OR direct construction in package tests) is safe —
        # never key results by "".
        if not self.output_name:
            object.__setattr__(self, "output_name", self.name)  # frozen-safe idiom
