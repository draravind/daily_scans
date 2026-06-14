"""Pure post-processors: each maps a scan's accumulated raw rows to its final
output shape — either a list of symbols (optionally with `###` section headers)
or a `(symbols, extras)` tuple. Moved verbatim from the repo; public names drop
the leading underscore, and the one market-named pp is neutralized.
"""
import pandas as pd


def identity_pp(rows):
    # Tolerant: scans that emit geometry now return dict rows {stock, ...}; the
    # screening output stays a symbol list. Bare-string scans pass through.
    return [r['stock'] if isinstance(r, dict) else r for r in rows]


def pullback_to_ema_pp(rows):
    if not rows:
        return []
    ema21 = [r['stock'] for r in rows if r['ema_level'] == 'EMA 21']
    ema50 = [r['stock'] for r in rows if r['ema_level'] == 'EMA 50']
    ema100 = [r['stock'] for r in rows if r['ema_level'] == 'EMA 100']
    return ['###EMA 21', *ema21, '###EMA 50', *ema50, '###EMA 100', *ema100]


def higher_highs_higher_lows_pp(rows):
    if not rows:
        return []
    output = pd.DataFrame(rows).sort_values(1, ascending=False)
    return output.iloc[:100][0].tolist()


def high_tight_flag_pp(rows):
    if not rows:
        return []
    df = pd.DataFrame(rows).sort_values('pole_gain_pct', ascending=False)
    return df['stock'].head(100).tolist()


def contracting_stocks_pp(rows):
    if not rows:
        return []
    output = pd.DataFrame(rows)
    output.sort_values(1, ascending=False, inplace=True)
    a = output[output[1] >= output[1].quantile(0.95)][0].tolist()
    output.sort_values(2, ascending=False, inplace=True)
    b = output[output[2] >= output[2].quantile(0.95)][0].tolist()
    b = [x for x in b if x not in a]
    return ["###ATR 1X", *a, "###ATR 1.5X", *b]


def smooth_stocks_pp(rows):
    if not rows:
        return []
    output = pd.DataFrame(rows).set_index('stock')
    output.dropna(how='any', inplace=True)
    if output.empty:
        return []
    output[output.columns.map(lambda x: x + '_p')] = output.rank(pct=True) * 100
    output['result'] = False
    for i in [4, 5, 6]:
        output['result'] = (output[f"DR_to_RR_>_{i}%"] <= 10) | (output['result'])
    return output[output['result']].index.tolist()


def relative_strength_pp(rows):
    if not rows:
        return []
    output = pd.DataFrame(rows).set_index('stock')
    output['rs_rank'] = (
        output['rs_raw'].rank(pct=True, method='min').mul(99).round().clip(lower=1, upper=99).astype(int)
    )
    # Sort by the continuous rs_raw (not rs_rank) so the top-200 cut is deterministic
    # even when many symbols tie at rs_rank == 99.
    output = output.sort_values('rs_raw', ascending=False).head(200)
    symbols = output.index.tolist()
    extras = {sym: {'rs_rank': int(output.at[sym, 'rs_rank'])} for sym in symbols}
    return symbols, extras


def relative_strength_benchmark_pivot_pp(rows):
    if not rows:
        return []
    output = pd.DataFrame(rows).set_index('stock')
    output['rs_rank'] = (
        output['return_pct']
        .rank(pct=True, method='min')
        .mul(99).round().clip(lower=1, upper=99).astype(int)
    )
    # Sort by continuous return_pct so the top-200 cut is deterministic even
    # under heavy ties at rs_rank == 99. Mirrors relative_strength_pp.
    output = output.sort_values('return_pct', ascending=False).head(200)
    symbols = output.index.tolist()
    extras = {
        sym: {
            'rs_rank': int(output.at[sym, 'rs_rank']),
            'return_pct': round(float(output.at[sym, 'return_pct']), 2),
        }
        for sym in symbols
    }
    return symbols, extras


def stage_2_advancing_pp(rows):
    if not rows:
        return []
    symbols = [r['stock'] for r in rows]
    extras = {r['stock']: {'entered_stage2_date': r['entered_stage2_date']} for r in rows}
    return symbols, extras


def ema_contraction_pp(rows):
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index('stock')


def ipo_pp(rows):
    if not rows:
        return []
    symbols = [r['stock'] for r in rows]
    extras = {r['stock']: {'ipo_6m': int(r['ipo_6m']), 'ipo_12m': int(r['ipo_12m'])} for r in rows}
    return symbols, extras


def highest_avg_turnover_pp(rows):
    if not rows:
        return []
    df = pd.DataFrame(rows).set_index('stock')
    result = []
    sections = [('quarter', '###QUARTER'), ('6_months', '###6 MONTHS'), ('1_year', '###1 YEAR'), ('all_time', '###ALL TIME')]
    for col, header in sections:
        if col not in df.columns:
            continue
        top = df[col].dropna().nlargest(50)
        result.append(header)
        result.extend(top.index.tolist())
    return result


def highest_turnover_pp(rows):
    if not rows:
        return []
    symbols = [r['stock'] for r in rows]
    extras = {
        r['stock']: {k: r[k] for k in ('is_3m', 'is_6m', 'is_1y', 'is_ath')}
        for r in rows
    }
    return symbols, extras


def base_breakouts_pp(rows):
    if not rows:
        return []
    rows = sorted(rows, key=lambda r: r["base_length"], reverse=True)
    symbols = [r["stock"] for r in rows]
    extras = {
        r["stock"]: {
            "base_days": r["base_length"],
            "pivot_date": r["pivot_date"],
            "pivot_price": r["pivot_price"],
        }
        for r in rows
    }
    return symbols, extras


def top_movers_pp(rows):
    if not rows:
        return []
    df = pd.DataFrame(rows).set_index('stock')
    period_cols = ['is_1d', 'is_1w', 'is_1m', 'is_3m', 'is_1y']
    for col in period_cols:
        q95 = df[col].quantile(0.95)
        df[col] = ((df[col] >= q95) & (df[col] >= 1)).astype(int)
    mask = df[period_cols].any(axis=1)
    df = df[mask]
    if df.empty:
        return []
    symbols = df.index.tolist()
    extras = {s: {c: int(df.loc[s, c]) for c in period_cols} for s in symbols}
    return symbols, extras


def unusual_volume_pp(rows):
    if not rows:
        return []
    symbols = [r['stock'] for r in rows]
    extras = {r['stock']: {'volume_ratio': r['volume_ratio']} for r in rows}
    return symbols, extras


def consolidation_breakout_pp(rows):
    if not rows:
        return []
    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    symbols = [r[0] for r in rows]
    extras = {r[0]: {"consolidation_days": r[1]} for r in rows}
    return symbols, extras


def consolidation_pp(rows):
    if not rows:
        return []
    rows = sorted(rows, key=lambda r: r["tightness"])  # ascending: tightest first
    symbols = [r["stock"] for r in rows]
    extras = {
        r["stock"]: {
            "consolidation_days": r["consolidation_days"],
            "tightness": round(float(r["tightness"]), 4),
        }
        for r in rows
    }
    return symbols, extras
