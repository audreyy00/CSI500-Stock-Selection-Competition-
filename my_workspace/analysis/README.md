# Factor Analysis

`factor_effectiveness.py` now runs only one method:

- Univariate Fama-MacBeth-style daily cross-sectional regression (t-value method).

Single-factor output keeps only two indicators per factor:

1. `factor_return_mean` (mean beta, i.e. factor return)
2. `t_value` (Fama-MacBeth t-stat of mean beta)

## Run

```bash
python my_workspace/analysis/factor_effectiveness.py \
  --prices data/prices.parquet \
  --outdir my_workspace/results/factor_analysis
```

## Outputs

- `single_factor_tvalue.csv`
