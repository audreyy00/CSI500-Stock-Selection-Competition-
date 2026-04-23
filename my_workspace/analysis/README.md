# Factor Analysis

`factor_effectiveness.py` runs four diagnostics on current factors:

1. Univariate Fama-MacBeth-style daily cross-sectional regression.
2. IC / Rank IC / IR analysis.
3. Quantile grouping backtest and long-short diagnostics.
4. Orthogonalized Rank IC diagnostics.

## Run

```bash
python my_workspace/analysis/factor_effectiveness.py \
  --prices data/prices.parquet \
  --outdir my_workspace/results/factor_analysis \
  --groups 10
```

## Outputs

- `factor_summary.csv`
- `fama_macbeth_univariate.csv`
- `ic_ir.csv`
- `grouping_backtest.csv`
- `orthogonalized_rank_ic.csv`
- `factor_spearman_corr.csv`
- `factor_keep_suggestion.csv`
