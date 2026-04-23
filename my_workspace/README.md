# Audrey Workspace

This directory is the personal working area for model iteration, while keeping
the course baseline files in the repository root intact.

## Structure

- `models/`: model scripts owned by you.
- `experiments/`: experiment configs and notes.
- `results/outputs/`: generated CSV files for upload.
- `results/logs/`: run logs and metric snapshots.

## Current Baseline Snapshot

- Baseline script copy: `models/baseline_xgboost_reference.py`
- Week 1 baseline submission: `results/outputs/week1_baseline.csv`

## Iteration Workflow (Cursor + Kaggle)

1. Edit code under this directory in Cursor.
2. Commit and push to GitHub.
3. In Kaggle, run `git pull origin main` to sync latest code.
4. Run training and validation there.
5. Copy new CSV back into `results/submissions/` and commit.

## Notes

- Keep root-level baseline files unchanged for clean comparison.
- Put all new model ideas under `models/` and `experiments/`.
- Save each run's key metrics in `results/logs/` for report writing.
