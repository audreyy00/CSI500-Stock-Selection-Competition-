"""
Single-factor effectiveness test (t-value method only).

Scope intentionally simplified:
- Keep only Fama-MacBeth-style daily cross-sectional regression.
- Output only two core indicators per factor:
  1) factor_return_mean (beta_mean)
  2) t_value (beta_t_fm)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make my_workspace importable when running this script directly.
WORKSPACE_DIR = Path(__file__).resolve().parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

from features.features_v1 import (  # noqa: E402
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_features,
    training_frame,
)

def _daily_regression_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """
    OLS y = alpha + beta * x + eps for one date's cross-section.
    Return beta and t_stat(beta).
    """
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_centered = x - x_mean
    denom = float(np.sum(x_centered ** 2))
    if denom <= 0:
        return np.nan, np.nan

    beta = float(np.sum(x_centered * (y - y_mean)) / denom)
    alpha = y_mean - beta * x_mean
    resid = y - (alpha + beta * x)
    n = len(x)
    if n <= 2:
        return beta, np.nan
    sigma2 = float(np.sum(resid ** 2) / (n - 2))
    se_beta = np.sqrt(sigma2 / denom) if denom > 0 else np.nan
    if se_beta == 0 or np.isnan(se_beta):
        t_beta = np.nan
    else:
        t_beta = beta / se_beta
    return beta, t_beta


def fama_macbeth_univariate(df: pd.DataFrame, factor: str) -> tuple[float, float]:
    """
    Returns:
    - factor_return_mean: mean daily beta
    - t_value: Fama-MacBeth t-stat of beta series mean
    """
    rows = []
    for _, g in df.groupby("date"):
        gg = g[[factor, TARGET_COLUMN]].dropna()
        if len(gg) < 30:
            continue
        x = gg[factor].to_numpy(dtype=float)
        y = gg[TARGET_COLUMN].to_numpy(dtype=float)
        beta, _ = _daily_regression_stats(x, y)
        if not np.isnan(beta):
            rows.append(beta)

    beta_series = pd.Series(rows, dtype=float).dropna()
    if beta_series.empty:
        return np.nan, np.nan

    factor_return_mean = float(beta_series.mean())
    beta_std = float(beta_series.std(ddof=1))
    n_days = int(len(beta_series))
    t_value = factor_return_mean / (beta_std / np.sqrt(n_days)) if beta_std > 0 else np.nan
    return factor_return_mean, t_value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default="data/prices.parquet")
    p.add_argument("--outdir", default="my_workspace/results/factor_analysis")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    panel = build_features(prices)
    df = training_frame(panel)

    rows = []
    for factor in FEATURE_COLUMNS:
        factor_return_mean, t_value = fama_macbeth_univariate(df, factor)
        rows.append(
            {
                "factor": factor,
                "factor_return_mean": factor_return_mean,
                "t_value": t_value,
            }
        )

    out = pd.DataFrame(rows).sort_values("t_value", ascending=False, na_position="last")
    out.to_csv(outdir / "single_factor_tvalue.csv", index=False)

    print(">> Single-factor test done (t-value method only)")
    print(f"   output dir: {outdir}")
    print("   file: single_factor_tvalue.csv")


if __name__ == "__main__":
    main()
