# HTD Training Pipeline

This repository contains tools for training machine learning models to predict basketball game outcomes and evaluate betting strategies.

## Quick Start

1. **Prepare Data**
   - Place CSV feature files in the directory specified by `data.train_dir` in `training_config.yaml`.
   - Ensure `outcomes_csv` points to a CSV containing the final scores for each game.

2. **Configure Training**
   - Edit `training_config.yaml` to adjust hyperparameters.
   - Set `device` to `auto` (default) to automatically use CUDA if available. Use `cuda` to request a GPU or `cpu` to force CPU execution. If a GPU is requested but unavailable, the script now falls back to CPU with a warning.

3. **Run Training**
   ```bash
   python train.py
   ```
   The script saves models under the `models/` directory and generates detailed betting reports.

## CUDA Notes

The training script can leverage a CUDA‑enabled GPU for faster computation. When `device` is set to `cuda` but no GPU is detected, the script now falls back to CPU and prints a warning. Using `device: auto` performs this selection automatically.

## Reports

After training completes, several summary files are produced inside the model folder:

- `betting_summary_report.txt` – concise win rates and ROI metrics.
- `BETTING_PERFORMANCE_REPORT.txt` – comprehensive analysis including monthly breakdowns.
- `training_summary.json` – full training metrics and configuration snapshot.

Refer to `BETTING_REPORT_GUIDE.md` for a detailed explanation of each file.
