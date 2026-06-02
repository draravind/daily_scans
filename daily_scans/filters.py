"""Liquidity / quality gate — parametrized so each market supplies its own
thresholds via a LiquidityFilterConfig. The body is the exact shared logic;
only the numbers (and the optional uptrend branch) vary by market.

India: enable_uptrend_filter=False ⇒ the EMA block never executes and
uptrend_cond is a no-op True. US: enable_uptrend_filter=True ⇒ requires
max(close, EMA_21, EMA_50) >= EMA_200 at the last bar.

The internal talib.SMA(TR, 125) and iloc[-125:] windows are literal warmup
windows (NOT the min_bars knob) and are shared by every market.
"""
import os
from dataclasses import dataclass

import talib


@dataclass(frozen=True)
class LiquidityFilterConfig:
    min_bars: int
    price_min: float
    volatility_cutoff: float
    turnover_q_high: float
    turnover_mean_high: float
    turnover_q_low: float
    turnover_mean_low: float
    vol_q_high: float
    vol_q_mid: float
    vol_mean_mid: float
    turnover_mean_override: float
    enable_uptrend_filter: bool = False


def common_filters(input_files, data_dict, config):
    """Return the subset of `input_files` symbols passing the liquidity gate."""
    filtered = []
    for input_file in input_files:
        try:
            stock = os.path.basename(input_file)
            data = data_dict[stock].copy()

            if data.shape[0] >= config.min_bars:
                data['turnover'] = data['close'] * data['volume']
                if config.enable_uptrend_filter:
                    data['EMA_21'] = talib.EMA(data['close'], 21)
                    data['EMA_50'] = talib.EMA(data['close'], 50)
                    data['EMA_200'] = talib.EMA(data['close'], 200)
                data['TR'] = talib.ATR(data['high'], data['low'], data['close'], 1) / data['close']
                data['ATR'] = talib.SMA(data['TR'], 125)

                turnover_q = data.iloc[-125:]['turnover'].quantile(0.5)
                turnover_mean = data.iloc[-125:]['turnover'].mean()
                turnover_cond = (
                    (turnover_q > config.turnover_q_high and turnover_mean > config.turnover_mean_high)
                    or (turnover_q > config.turnover_q_low and turnover_mean > config.turnover_mean_low)
                )
                vol_q = data.iloc[-125:]['volume'].quantile(0.5)
                vol_mean = data.iloc[-125:]['volume'].mean()
                volume_cond = (
                    vol_q > config.vol_q_high
                    or (vol_q > config.vol_q_mid and vol_mean > config.vol_mean_mid)
                    or turnover_mean > config.turnover_mean_override
                )
                volatility_cond = data['ATR'].iloc[-1] > config.volatility_cutoff
                penny_stock_cond = data['close'].iloc[-100:].min() >= config.price_min
                if config.enable_uptrend_filter:
                    uptrend_cond = max(
                        data['close'].iloc[-1], data['EMA_21'].iloc[-1], data['EMA_50'].iloc[-1]
                    ) >= data['EMA_200'].iloc[-1]
                else:
                    uptrend_cond = True

                if turnover_cond and volume_cond and volatility_cond and penny_stock_cond and uptrend_cond:
                    filtered.append(stock)
        except (KeyError, IndexError, ValueError, ZeroDivisionError):
            pass

    return filtered
