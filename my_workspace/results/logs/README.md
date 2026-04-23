# Logs

Save plain-text logs for reproducibility.

Recommended naming:

- `2026-04-23_baseline_week1.log`
- `2026-05-02_exp001_week1.log`

Tip:

```bash
python baseline_xgboost.py --out submissions/week1.csv 2>&1 | tee my_workspace/results/logs/2026-04-23_baseline_week1.log
```
