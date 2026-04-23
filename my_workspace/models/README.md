# Models

- `baseline_xgboost_reference.py`: frozen baseline snapshot for comparison.
- `model_v1.py`: strict time-split version with optional walk-forward backtest output.
- Add your custom models here, e.g. `model_v1.py`, `model_v2.py`.

Keep model files self-contained and import shared utilities from repo root when possible.

## model_v1.py quick usage

```bash
# Generate submission only
python my_workspace/models/model_v1.py --out my_workspace/results/outputs/week1_model_v1.csv

# Generate submission + historical walk-forward return table
python my_workspace/models/model_v1.py \
  --out my_workspace/results/outputs/week1_model_v1.csv \
  --run-backtest --backtest-windows 6 --hold-days 5
```
