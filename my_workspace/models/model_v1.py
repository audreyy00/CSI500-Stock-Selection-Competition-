"""
Model v1 based on the course baseline, with strict time controls.

What is added compared with baseline_xgboost.py:
1) Explicit as-of date resolution and strict trading-date checks.
2) Optional historical walk-forward backtest output (portfolio/benchmark/excess).
3) Clear separation between prediction output (CSV) and in-sample sanity metrics.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

# Make `my_workspace/features` importable when running this file directly.
WORKSPACE_DIR = Path(__file__).resolve().parents[1]
if str(WORKSPACE_DIR) not in sys.path:
    sys.path.append(str(WORKSPACE_DIR))

from features.features_v1 import (
    FEATURE_COLUMNS, TARGET_COLUMN, FORWARD_HORIZON,
    build_features, training_frame, prediction_frame,
)

DATA_DIR = Path(__file__).parent / "data"
VAL_DAYS = 10               # number of trading days in the validation window
EMBARGO_DAYS = 5            # gap between train end and val start (>= FORWARD_HORIZON
                            # so training targets don't reach into val dates)
MIN_STOCKS = 30             # rule: portfolio must hold >= 30 names
MAX_WEIGHT = 0.10           # rule: per-stock weight cap
DEFAULT_TOP_K = 50          # baseline picks top-50 by predicted score
DEFAULT_HOLD_DAYS = 5       # for optional historical backtest windows
ALLOW_CROSS_VALIDATION = False  # hard guard: this script only uses single time split

# Start with relatively high capacity, but keep explicit regularization + early stop.
MODEL_CONFIG = {
    "n_estimators": 1200,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "reg_lambda": 1.0,
    "gamma": 0.0,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 42,
}
EARLY_STOPPING_ROUNDS = 50


def _realized_return_one_stock(stock_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> float:
    in_window = stock_df[(stock_df["date"] >= start) & (stock_df["date"] <= end)].sort_values("date")
    if in_window.empty:
        return 0.0

    before = stock_df[stock_df["date"] < start].sort_values("date")
    entry = before["close"].iloc[-1] if not before.empty else in_window["open"].iloc[0]
    exit_ = in_window["close"].iloc[-1]
    if pd.isna(entry) or pd.isna(exit_) or entry <= 0:
        return 0.0
    return float(exit_ / entry - 1.0)


def score_window(
    weights: pd.Series,
    prices: pd.DataFrame,
    index_df: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[float, float, float]:
    """Compute portfolio / benchmark / excess return over one realized window."""
    rets = {}
    for code in weights.index:
        sub = prices[prices["stock_code"] == code]
        rets[code] = _realized_return_one_stock(sub, start, end)
    stock_rets = pd.Series(rets)
    portfolio_return = float((weights * stock_rets).sum())

    idx_window = index_df[(index_df["date"] >= start) & (index_df["date"] <= end)].sort_values("date")
    if idx_window.empty:
        raise RuntimeError(f"No index data in [{start.date()}, {end.date()}].")
    idx_before = index_df[index_df["date"] < start].sort_values("date")
    idx_entry = idx_before["close"].iloc[-1] if not idx_before.empty else idx_window["open"].iloc[0]
    idx_exit = idx_window["close"].iloc[-1]
    benchmark_return = float(idx_exit / idx_entry - 1.0)
    excess_return = portfolio_return - benchmark_return
    return portfolio_return, benchmark_return, excess_return


def _max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns).cumprod()
    dd = wealth / wealth.cummax() - 1.0
    return float(dd.min()) if not dd.empty else np.nan


def performance_metrics(bt: pd.DataFrame, hold_days: int) -> dict:
    """
    Compute strategy performance metrics from walk-forward windows.
    Assumes each row is one rebalance window with realized portfolio/benchmark return.
    Risk-free rate is set to 0 for Sharpe.
    """
    rp = bt["portfolio_return"].astype(float)
    rb = bt["benchmark_return"].astype(float)

    periods_per_year = 252.0 / max(1, hold_days)
    mean_rp = float(rp.mean())
    std_rp = float(rp.std(ddof=1))
    vol_ann = std_rp * np.sqrt(periods_per_year) if std_rp > 0 else np.nan
    sharpe = (mean_rp / std_rp) * np.sqrt(periods_per_year) if std_rp > 0 else np.nan

    # Regression: rp = alpha + beta * rb + eps
    rb_var = float(rb.var(ddof=1))
    if rb_var > 0:
        beta = float(np.cov(rp, rb, ddof=1)[0, 1] / rb_var)
        alpha_period = float(mean_rp - beta * float(rb.mean()))
    else:
        beta = np.nan
        alpha_period = np.nan
    alpha_ann = alpha_period * periods_per_year if not np.isnan(alpha_period) else np.nan

    cum_port = float((1.0 + rp).prod() - 1.0)
    cum_bench = float((1.0 + rb).prod() - 1.0)
    cum_excess = float((1.0 + bt["excess_return"]).prod() - 1.0)

    return {
        "strategy_cum_return": cum_port,
        "benchmark_cum_return": cum_bench,
        "cum_excess_return": cum_excess,
        "alpha_annualized": alpha_ann,
        "beta": beta,
        "sharpe_annualized": sharpe,
        "volatility_annualized": vol_ann,
        "max_drawdown": _max_drawdown(rp),
    }


def _resolve_as_of(panel: pd.DataFrame, as_of: str | None) -> tuple[pd.Timestamp, np.ndarray]:
    trading_dates = np.sort(panel["date"].unique())
    if as_of is None:
        return pd.Timestamp(trading_dates[-1]), trading_dates

    as_of_ts = pd.Timestamp(as_of)
    if np.datetime64(as_of_ts) not in set(trading_dates):
        raise ValueError(
            f"as_of={as_of_ts.date()} is not a trading date in data. "
            "Use an existing date from prices.parquet."
        )
    return as_of_ts, trading_dates


def _build_train_val(panel: pd.DataFrame, as_of_ts: pd.Timestamp, trading_dates: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Strictly time-bounded train/validation split with embargo."""
    as_of_idx = np.searchsorted(trading_dates, np.datetime64(as_of_ts))
    if as_of_idx >= len(trading_dates) or trading_dates[as_of_idx] != np.datetime64(as_of_ts):
        raise ValueError(f"as_of={as_of_ts.date()} not found in trading dates.")

    # Prevent any target from touching dates after as_of.
    cutoff_idx = as_of_idx - FORWARD_HORIZON
    if cutoff_idx < 0:
        raise RuntimeError("Not enough history before as_of to construct forward target safely.")
    train_cutoff = pd.Timestamp(trading_dates[cutoff_idx])
    train_pool = training_frame(panel, max_date=train_cutoff)

    all_dates = np.sort(train_pool["date"].unique())
    if len(all_dates) < VAL_DAYS + EMBARGO_DAYS + 20:
        raise RuntimeError("Not enough dates to train with strict embargo.")
    val_start = pd.Timestamp(all_dates[-VAL_DAYS])
    train_end = pd.Timestamp(all_dates[-(VAL_DAYS + EMBARGO_DAYS + 1)])
    train_df = train_pool[train_pool["date"] <= train_end]
    val_df = train_pool[train_pool["date"] >= val_start]
    return train_df, val_df


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame) -> xgb.XGBRegressor:
    if ALLOW_CROSS_VALIDATION:
        raise RuntimeError("Cross-validation is disabled by design in model_v1.")

    model = xgb.XGBRegressor(
        **MODEL_CONFIG,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    model.fit(
        train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN],
        eval_set=[(val_df[FEATURE_COLUMNS], val_df[TARGET_COLUMN])],
        verbose=False,
    )
    return model


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray) -> float:
    """Daily cross-sectional Spearman correlation, averaged over dates."""
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 20:
            continue
        rho, _ = spearmanr(y_true[mask], y_pred[mask])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


def build_portfolio(scores: pd.Series, top_k: int = DEFAULT_TOP_K) -> pd.Series:
    """Top-K names, weight proportional to (rank) then capped at MAX_WEIGHT.

    We use rank-weights rather than score-weights so pathological score scales
    do not produce a single dominant name.  After capping at 10% we redistribute
    spillover to uncapped names and iterate until feasible.
    """
    if top_k < MIN_STOCKS:
        raise ValueError(f"top_k must be >= {MIN_STOCKS} (rule)")
    chosen = scores.sort_values(ascending=False).head(top_k).copy()

    # Rank-based weights (best stock gets largest weight, then normalize).
    ranks = np.arange(top_k, 0, -1, dtype=float)
    w = pd.Series(ranks / ranks.sum(), index=chosen.index)

    # Iteratively cap at MAX_WEIGHT and redistribute to uncapped names.
    for _ in range(50):
        over = w > MAX_WEIGHT
        if not over.any():
            break
        excess = (w[over] - MAX_WEIGHT).sum()
        w[over] = MAX_WEIGHT
        free = ~over
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()

    assert abs(w.sum() - 1.0) < 1e-6, f"weights sum to {w.sum()}"
    assert (w <= MAX_WEIGHT + 1e-9).all(), "cap violated"
    assert (w > 0).sum() >= MIN_STOCKS, "too few names"
    return w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prices", default="data/prices.parquet")
    p.add_argument("--index", default="data/index.parquet")
    p.add_argument("--as-of", default=None, help="YYYYMMDD; defaults to latest trading date in data")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--out", default="submission.csv")
    p.add_argument("--run-backtest", action="store_true", help="run strict walk-forward historical windows")
    p.add_argument("--backtest-windows", type=int, default=6, help="number of walk-forward windows to report")
    p.add_argument("--hold-days", type=int, default=DEFAULT_HOLD_DAYS, help="holding days per backtest window")
    args = p.parse_args()

    print(f">> Loading {args.prices}")
    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    prices["stock_code"] = prices["stock_code"].astype(str).str.zfill(6)
    print(f"   {len(prices):,} rows, {prices['stock_code'].nunique()} stocks, "
          f"dates {prices['date'].min().date()} to {prices['date'].max().date()}")

    print(">> Building features")
    panel = build_features(prices)

    as_of_ts, trading_dates = _resolve_as_of(panel, args.as_of)
    train_df, val_df = _build_train_val(panel, as_of_ts, trading_dates)
    all_dates = np.sort(training_frame(panel, max_date=as_of_ts)["date"].unique())
    val_start = pd.Timestamp(all_dates[-VAL_DAYS])
    train_end = pd.Timestamp(all_dates[-(VAL_DAYS + EMBARGO_DAYS + 1)])
    print(f"   train: {len(train_df):,} rows up to {train_end.date()}")
    print(f"   embargo: {EMBARGO_DAYS} trading days (discarded)")
    print(f"   val:   {len(val_df):,} rows from {val_start.date()}")

    print(">> Training XGBoost")
    model = train_model(train_df, val_df)

    val_pred = model.predict(val_df[FEATURE_COLUMNS])
    ic = rank_ic(val_df[TARGET_COLUMN].to_numpy(), val_pred, val_df["date"].to_numpy())
    print(f"   validation rank IC: {ic:.4f}")

    print(">> Predicting portfolio")
    pred_df = prediction_frame(panel, as_of=as_of_ts)
    if pred_df.empty:
        raise RuntimeError(f"No rows available for as_of={as_of_ts.date()}. Check data.")
    pred_date = pred_df["date"].iloc[0]
    print(f"   as of {pred_date.date()}, scoring {len(pred_df)} stocks")

    pred_df = pred_df.assign(score=model.predict(pred_df[FEATURE_COLUMNS]))
    scores = pred_df.set_index("stock_code")["score"]
    weights = build_portfolio(scores, top_k=args.top_k)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({"stock_code": weights.index, "weight": weights.values})
    out.to_csv(out_path, index=False)
    print(f">> Wrote {len(out)} names to {out_path}")
    print(f"   weight summary: min={out['weight'].min():.4f} "
          f"max={out['weight'].max():.4f} sum={out['weight'].sum():.4f}")

    if not args.run_backtest:
        return

    print(">> Running strict walk-forward backtest")
    index_df = pd.read_parquet(args.index)
    index_df["date"] = pd.to_datetime(index_df["date"])

    as_of_idx = np.searchsorted(trading_dates, np.datetime64(as_of_ts))
    latest_backtest_idx = min(as_of_idx, len(trading_dates) - args.hold_days - 1)
    backtest_indices = [
        latest_backtest_idx - i * args.hold_days
        for i in range(args.backtest_windows)
        if latest_backtest_idx - i * args.hold_days >= 0
    ]
    backtest_indices = sorted(backtest_indices)
    rows = []
    for idx in backtest_indices:
        bt_as_of = pd.Timestamp(trading_dates[idx])
        bt_train_df, bt_val_df = _build_train_val(panel, bt_as_of, trading_dates)
        bt_model = train_model(bt_train_df, bt_val_df)
        bt_pred = bt_model.predict(bt_val_df[FEATURE_COLUMNS])
        bt_ic = rank_ic(bt_val_df[TARGET_COLUMN].to_numpy(), bt_pred, bt_val_df["date"].to_numpy())

        bt_frame = prediction_frame(panel, as_of=bt_as_of)
        if bt_frame.empty:
            continue
        bt_frame = bt_frame.assign(score=bt_model.predict(bt_frame[FEATURE_COLUMNS]))
        bt_weights = build_portfolio(bt_frame.set_index("stock_code")["score"], top_k=args.top_k)

        start = pd.Timestamp(trading_dates[idx + 1])
        end = pd.Timestamp(trading_dates[idx + args.hold_days])
        p_ret, b_ret, e_ret = score_window(bt_weights, prices, index_df, start, end)
        rows.append({
            "as_of": bt_as_of.date().isoformat(),
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
            "val_rank_ic": bt_ic,
            "portfolio_return": p_ret,
            "benchmark_return": b_ret,
            "excess_return": e_ret,
        })

    if not rows:
        print("   no complete historical windows available for backtest.")
        return

    bt = pd.DataFrame(rows)
    print(bt.to_string(index=False, float_format=lambda x: f"{x:+.4%}" if abs(x) < 2 else f"{x:.4f}"))

    metrics = performance_metrics(bt, hold_days=args.hold_days)
    summary_df = pd.DataFrame([metrics])
    metrics_out = out_path.with_name(out_path.stem + "_backtest_metrics.csv")
    summary_df.to_csv(metrics_out, index=False)

    print(">> Backtest summary")
    print(f"   windows: {len(bt)}")
    print(f"   strategy cumulative return:  {metrics['strategy_cum_return']:+.3%}")
    print(f"   benchmark cumulative return: {metrics['benchmark_cum_return']:+.3%}")
    print(f"   cumulative excess return:    {metrics['cum_excess_return']:+.3%}")
    print(f"   alpha (annualized):          {metrics['alpha_annualized']:+.3%}")
    print(f"   beta:                        {metrics['beta']:+.4f}")
    print(f"   sharpe (annualized):         {metrics['sharpe_annualized']:+.4f}")
    print(f"   volatility (annualized):     {metrics['volatility_annualized']:+.3%}")
    print(f"   max drawdown:                {metrics['max_drawdown']:+.3%}")
    print(f"   metrics csv:                 {metrics_out}")


if __name__ == "__main__":
    main()
