"""
Feature engineering for the CSI500 stock-selection baseline.

This file starts as a copy of the root-level `features.py` and is intended for
large factor changes in the personal workspace.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# columns used downstream by the model
# 动量因子（基于图表111提到的13个因子）
FEATURE_COLUMNS = [
    "HAlpha",
    "return_1m", "return_3m", "return_6m", "return_12m",
    "wgt_return_1m", "wgt_return_3m", "wgt_return_6m", "wgt_return_12m",
    "exp_wgt_return_1m", "exp_wgt_return_3m", "exp_wgt_return_6m", "exp_wgt_return_12m",
]
TARGET_COLUMN = "target_5d"
FORWARD_HORIZON = 5

LOOKBACK = {
    "1m": 21,
    "3m": 63,
    "6m": 126,
    "12m": 252,
}


def _per_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Momentum-focused features computed from one stock's time series."""
    df = df.sort_values("date").copy()
    close = df["close"]
    ret_1d = close.pct_change(1)
    df["ret_1d"] = ret_1d

    # 传统N个月动量因子: return_Nm
    df["return_1m"] = close.pct_change(LOOKBACK["1m"])
    df["return_3m"] = close.pct_change(LOOKBACK["3m"])
    df["return_6m"] = close.pct_change(LOOKBACK["6m"])
    df["return_12m"] = close.pct_change(LOOKBACK["12m"])

    turnover = df["turnover"].astype(float) if "turnover" in df.columns else pd.Series(np.nan, index=df.index)
    weighted_daily_ret = ret_1d * turnover

    # 换手率加权N个月动量: wgt_return_Nm
    for tag, window in LOOKBACK.items():
        num = weighted_daily_ret.rolling(window).sum()
        den = turnover.rolling(window).sum().replace(0, np.nan)
        df[f"wgt_return_{tag}"] = num / den

    # 指数衰减换手率加权N个月动量: exp_wgt_return_Nm
    for tag, window in LOOKBACK.items():
        ewm_num = weighted_daily_ret.ewm(span=window, adjust=False, min_periods=window).mean()
        ewm_den = turnover.ewm(span=window, adjust=False, min_periods=window).mean().replace(0, np.nan)
        df[f"exp_wgt_return_{tag}"] = ewm_num / ewm_den

    df[TARGET_COLUMN] = close.shift(-FORWARD_HORIZON) / close - 1.0
    return df


def _add_halpha(panel: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Add HAlpha-like factor from rolling CAPM alpha:
      alpha_t = mean(r_i) - beta_t * mean(r_m)
    where beta_t = cov(r_i, r_m) / var(r_m) over rolling window.

    Note:
    - Original report uses very long horizon (60 months). With current daily
      sample length, we approximate using 252 trading days.
    """
    panel = panel.sort_values(["stock_code", "date"]).copy()
    panel["market_ret"] = panel.groupby("date")["ret_1d"].transform("mean")

    def _calc_alpha(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()
        ri = g["ret_1d"]
        rm = g["market_ret"]
        cov = ri.rolling(window).cov(rm)
        var = rm.rolling(window).var().replace(0, np.nan)
        beta = cov / var
        g["HAlpha"] = ri.rolling(window).mean() - beta * rm.rolling(window).mean()
        return g

    panel = panel.groupby("stock_code", group_keys=False).apply(_calc_alpha).reset_index(drop=True)
    return panel


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Build a (date, stock_code) panel of features + target."""
    required = {"date", "stock_code", "close", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing required columns: {missing}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    panel = (
        prices.groupby("stock_code", group_keys=False)
        .apply(_per_stock_features)
        .reset_index(drop=True)
    )
    panel = _add_halpha(panel, window=LOOKBACK["12m"])
    return panel


def training_frame(panel: pd.DataFrame, min_date=None, max_date=None) -> pd.DataFrame:
    """Rows usable for supervised training: features + target both present."""
    df = panel.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()
    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]
    return df


def prediction_frame(panel: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Rows for a single prediction date (defaults to the latest date)."""
    if as_of is None:
        as_of = panel["date"].max()
    as_of = pd.Timestamp(as_of)
    df = panel[panel["date"] == as_of].dropna(subset=FEATURE_COLUMNS).copy()
    return df
