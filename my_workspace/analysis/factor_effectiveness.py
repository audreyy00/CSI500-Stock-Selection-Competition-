"""
Factor effectiveness analysis toolkit for current feature set.

Includes:
1) Univariate Fama-MacBeth style daily cross-sectional regression diagnostics.
2) IC / Rank IC / IR analysis.
3) Grouping backtest (quantile portfolios and long-short).
4) Orthogonalization diagnostics (residual IC + factor correlation).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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

# Factor buckets for selection suggestions.
FACTOR_BUCKETS = {
    "momentum_reversal": ["ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d", "ret_5d_rank", "ret_20d_rank"],
    "volume_activity": ["volume_z_20d", "turnover_ma_20d", "vol_20d", "vol_20d_rank"],
    "technical_pattern": ["close_over_ma20", "close_over_ma60", "rsi_14"],
}


def _daily_regression_stats(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """
    OLS y = alpha + beta * x + eps for one date's cross-section.
    Return beta, t_stat(beta), alpha.
    """
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    x_centered = x - x_mean
    denom = float(np.sum(x_centered ** 2))
    if denom <= 0:
        return np.nan, np.nan, np.nan

    beta = float(np.sum(x_centered * (y - y_mean)) / denom)
    alpha = y_mean - beta * x_mean
    resid = y - (alpha + beta * x)
    n = len(x)
    if n <= 2:
        return beta, np.nan, alpha
    sigma2 = float(np.sum(resid ** 2) / (n - 2))
    se_beta = np.sqrt(sigma2 / denom) if denom > 0 else np.nan
    if se_beta == 0 or np.isnan(se_beta):
        t_beta = np.nan
    else:
        t_beta = beta / se_beta
    return beta, t_beta, alpha


def fama_macbeth_univariate(df: pd.DataFrame, factor: str) -> dict:
    rows = []
    for d, g in df.groupby("date"):
        gg = g[[factor, TARGET_COLUMN]].dropna()
        if len(gg) < 30:
            continue
        x = gg[factor].to_numpy(dtype=float)
        y = gg[TARGET_COLUMN].to_numpy(dtype=float)
        beta, t_beta, alpha = _daily_regression_stats(x, y)
        rows.append({"date": d, "beta": beta, "t_beta": t_beta, "alpha": alpha, "n": len(gg)})

    out = pd.DataFrame(rows)
    if out.empty:
        return {
            "factor": factor,
            "n_days": 0,
            "beta_mean": np.nan,
            "beta_t_fm": np.nan,
            "t_abs_mean": np.nan,
            "t_abs_gt_2_ratio": np.nan,
        }

    beta_mean = float(out["beta"].mean())
    beta_std = float(out["beta"].std(ddof=1))
    n_days = int(len(out))
    beta_t_fm = beta_mean / (beta_std / np.sqrt(n_days)) if beta_std > 0 else np.nan
    t_abs = out["t_beta"].abs()
    return {
        "factor": factor,
        "n_days": n_days,
        "beta_mean": beta_mean,
        "beta_t_fm": beta_t_fm,
        "t_abs_mean": float(t_abs.mean()),
        "t_abs_gt_2_ratio": float((t_abs > 2.0).mean()),
    }


def ic_ir_analysis(df: pd.DataFrame, factor: str) -> dict:
    ic_rows = []
    for _, g in df.groupby("date"):
        gg = g[[factor, TARGET_COLUMN]].dropna()
        if len(gg) < 30:
            continue
        x = gg[factor]
        y = gg[TARGET_COLUMN]
        ic = x.corr(y, method="pearson")
        rank_ic = x.corr(y, method="spearman")
        ic_rows.append((ic, rank_ic))

    if not ic_rows:
        return {
            "factor": factor,
            "n_days": 0,
            "ic_mean": np.nan,
            "ic_std": np.nan,
            "ic_ir_mean_over_std": np.nan,
            "rank_ic_mean": np.nan,
            "rank_ic_std": np.nan,
            "rank_ic_ir_mean_over_std": np.nan,
        }

    arr = np.array(ic_rows, dtype=float)
    ic_mean, rank_ic_mean = np.nanmean(arr, axis=0)
    ic_std, rank_ic_std = np.nanstd(arr, axis=0, ddof=1)
    return {
        "factor": factor,
        "n_days": int(arr.shape[0]),
        "ic_mean": float(ic_mean),
        "ic_std": float(ic_std),
        "ic_ir_mean_over_std": float(ic_mean / ic_std) if ic_std > 0 else np.nan,
        "rank_ic_mean": float(rank_ic_mean),
        "rank_ic_std": float(rank_ic_std),
        "rank_ic_ir_mean_over_std": float(rank_ic_mean / rank_ic_std) if rank_ic_std > 0 else np.nan,
    }


def grouping_backtest(df: pd.DataFrame, factor: str, n_groups: int = 10) -> dict:
    rows = []
    for d, g in df.groupby("date"):
        gg = g[[factor, TARGET_COLUMN]].dropna()
        if len(gg) < max(50, n_groups * 5):
            continue
        gg = gg.sort_values(factor).copy()
        # rank-based bins for robustness
        ranks = gg[factor].rank(method="first")
        gg["group"] = pd.qcut(ranks, n_groups, labels=False) + 1
        group_ret = gg.groupby("group")[TARGET_COLUMN].mean()
        if len(group_ret) != n_groups:
            continue
        long_short = float(group_ret.loc[n_groups] - group_ret.loc[1])
        mono_rho, _ = spearmanr(np.arange(1, n_groups + 1), group_ret.values)
        row = {"date": d, "long_short": long_short, "monotonicity_rho": mono_rho}
        for k, v in group_ret.items():
            row[f"group_{int(k)}"] = float(v)
        rows.append(row)

    if not rows:
        return {"factor": factor, "n_days": 0}

    bt = pd.DataFrame(rows).sort_values("date")
    group_cols = [c for c in bt.columns if c.startswith("group_")]
    group_means = bt[group_cols].mean()

    # cumulative long-short (overlapping forward-return convention, as diagnostic)
    ls_curve = (1.0 + bt["long_short"]).cumprod()
    cum_ls = float(ls_curve.iloc[-1] - 1.0)
    max_dd = float(((ls_curve / ls_curve.cummax()) - 1.0).min())

    out = {
        "factor": factor,
        "n_days": int(len(bt)),
        "long_short_mean": float(bt["long_short"].mean()),
        "long_short_std": float(bt["long_short"].std(ddof=1)),
        "long_short_ir_mean_over_std": float(
            bt["long_short"].mean() / bt["long_short"].std(ddof=1)
        ) if bt["long_short"].std(ddof=1) > 0 else np.nan,
        "long_short_cum_return": cum_ls,
        "long_short_max_drawdown": max_dd,
        "monotonicity_rho_mean": float(bt["monotonicity_rho"].mean()),
    }
    for col, val in group_means.items():
        out[f"{col}_mean"] = float(val)
    return out


def orthogonalized_rank_ic(df: pd.DataFrame, factor: str, controls: list[str]) -> dict:
    """
    Residualize one factor against controls, then compute rank IC of residual.
    """
    rows = []
    for _, g in df.groupby("date"):
        cols = [factor, TARGET_COLUMN] + controls
        gg = g[cols].dropna()
        if len(gg) < 40:
            continue
        y = gg[factor].to_numpy(dtype=float)
        x = gg[controls].to_numpy(dtype=float)
        x = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        rank_ic = pd.Series(resid).corr(gg[TARGET_COLUMN], method="spearman")
        rows.append(rank_ic)
    rows = pd.Series(rows, dtype=float).dropna()
    return {
        "factor": factor,
        "controls": ",".join(controls),
        "n_days": int(len(rows)),
        "orth_rank_ic_mean": float(rows.mean()) if not rows.empty else np.nan,
        "orth_rank_ic_std": float(rows.std(ddof=1)) if len(rows) > 1 else np.nan,
        "orth_rank_ic_ir_mean_over_std": float(rows.mean() / rows.std(ddof=1))
        if len(rows) > 1 and rows.std(ddof=1) > 0 else np.nan,
    }


def suggest_kept_factors(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Suggest kept factors by bucket:
    - momentum_reversal: keep top 3
    - volume_activity: keep top 2
    - technical_pattern: keep top 2-3 (here keep top 3 if available)
    Ranking score combines rank_ic and Fama-MacBeth t strength.
    """
    df = summary.copy()
    df["score"] = (
        df["rank_ic_mean"].fillna(0.0) * 100.0
        + df["rank_ic_ir_mean_over_std"].fillna(0.0) * 10.0
        + df["beta_t_fm"].fillna(0.0)
        + df["t_abs_gt_2_ratio"].fillna(0.0) * 2.0
    )
    keep_rows = []
    for bucket, factors in FACTOR_BUCKETS.items():
        sub = df[df["factor"].isin(factors)].sort_values("score", ascending=False)
        if bucket == "momentum_reversal":
            k = 3
        elif bucket == "volume_activity":
            k = 2
        else:
            k = min(3, len(sub))
        chosen = sub.head(k).copy()
        chosen["bucket"] = bucket
        keep_rows.append(chosen)
    if not keep_rows:
        return pd.DataFrame()
    return pd.concat(keep_rows, ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default="data/prices.parquet")
    p.add_argument("--outdir", default="my_workspace/results/factor_analysis")
    p.add_argument("--groups", type=int, default=10)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    panel = build_features(prices)
    df = training_frame(panel)

    fm_rows = []
    ic_rows = []
    gb_rows = []
    orth_rows = []
    corr = df[FEATURE_COLUMNS].corr(method="spearman")
    corr.to_csv(outdir / "factor_spearman_corr.csv", index=True)

    for factor in FEATURE_COLUMNS:
        fm_rows.append(fama_macbeth_univariate(df, factor))
        ic_rows.append(ic_ir_analysis(df, factor))
        gb_rows.append(grouping_backtest(df, factor, n_groups=args.groups))
        controls = [f for f in FEATURE_COLUMNS if f != factor]
        # keep only top correlated controls for a practical orthogonalization test
        top_controls = corr[factor].drop(index=factor).abs().sort_values(ascending=False).head(3).index.tolist()
        if top_controls:
            orth_rows.append(orthogonalized_rank_ic(df, factor, top_controls))

    fm_df = pd.DataFrame(fm_rows)
    ic_df = pd.DataFrame(ic_rows)
    gb_df = pd.DataFrame(gb_rows)
    orth_df = pd.DataFrame(orth_rows)

    summary = fm_df.merge(ic_df, on=["factor", "n_days"], how="outer")
    summary = summary.merge(gb_df, on=["factor", "n_days"], how="outer")
    summary.to_csv(outdir / "factor_summary.csv", index=False)
    fm_df.to_csv(outdir / "fama_macbeth_univariate.csv", index=False)
    ic_df.to_csv(outdir / "ic_ir.csv", index=False)
    gb_df.to_csv(outdir / "grouping_backtest.csv", index=False)
    orth_df.to_csv(outdir / "orthogonalized_rank_ic.csv", index=False)

    kept = suggest_kept_factors(summary)
    kept.to_csv(outdir / "factor_keep_suggestion.csv", index=False)

    print(">> Factor analysis done")
    print(f"   output dir: {outdir}")
    print("   files:")
    for name in [
        "factor_summary.csv",
        "fama_macbeth_univariate.csv",
        "ic_ir.csv",
        "grouping_backtest.csv",
        "orthogonalized_rank_ic.csv",
        "factor_spearman_corr.csv",
        "factor_keep_suggestion.csv",
    ]:
        print(f"   - {name}")


if __name__ == "__main__":
    main()
