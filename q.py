#!/usr/bin/env python3
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

# === import your training-time utilities from train.py ===
from train import CSVDataset, MLP, temporal_train_test_split, evaluate_model

def load_model_from_checkpoint(ckpt_path: Path, device: torch.device):
    # 1) Load checkpoint (weights_only=True silences the pickle warning in newer PyTorch)
    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    # 2) Pull the exact config dictionary you used at training time
    train_cfg    = checkpoint['config']
    hidden_dims  = train_cfg['model']['hidden_dims']
    dropout      = train_cfg['model']['dropout']
    # 3) Retrieve dims you saved
    in_dim       = checkpoint['in_dim']
    out_dim      = checkpoint['out_dim']
    # 4) Rebuild your MLP exactly and load its weights
    model = MLP(in_dim, hidden_dims, out_dim, dropout).to(device)
    model.load_state_dict(checkpoint['model'])
    # 5) Get the scalers you fit at training time
    feature_scaler = checkpoint['feature_scaler']
    target_scaler  = checkpoint['target_scaler']
    return model, feature_scaler, target_scaler

def main():
    # ——— 1) Read config ———
    cfg_path = sys.argv[1] if len(sys.argv)>1 else "training_config.yaml"
    if not Path(cfg_path).exists():
        print(f"❌ Config '{cfg_path}' not found."); sys.exit(1)
    cfg = yaml.safe_load(open(cfg_path, "r"))

    # ——— 2) Select device ———
    device = torch.device(cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"🚀 Using device: {device}")

    # ——— 3) Locate your best/last checkpoint ———
    p = cfg['paths']
    best = Path(p['best_model_path'])
    last = Path(p['last_model_path'])
    if   best.exists(): ckpt = best
    elif last.exists(): ckpt = last
    else:
        print("❌ No checkpoint at", best, "or", last); sys.exit(1)
    print("📦 Loading checkpoint:", ckpt)

    # ——— 4) Load model + scalers ———
    model, feat_scaler, targ_scaler = load_model_from_checkpoint(ckpt, device)
    model.eval()

    # ——— 5) Gather CSV files & outcomes ———
    d = cfg['data']
    csv_dir = Path(d['train_dir'])
    all_files = sorted(csv_dir.glob("*.csv"))
    if not all_files:
        print("❌ No CSVs in", csv_dir); sys.exit(1)

    outcomes_csv = d['outcomes_csv']
    outcomes_df  = pd.read_csv(outcomes_csv)
    valid_names  = set(outcomes_df['file_name'])
    val_files    = [str(p) for p in all_files if p.name in valid_names]
    print(f"🔍 Found {len(val_files)} files with recorded outcomes.")

    # ——— 6) Build dataset (applies feature_scaler internally) ———
    ds = CSVDataset(val_files, outcomes_csv, scaler=feat_scaler)
    X, Y = ds.features, ds.labels

    # ——— 7) Temporal split → take only the “future” portion as validation ———
    val_frac = 1 - d.get('train_split', 0.8)
    _, X_val, _, Y_val = temporal_train_test_split(X, Y, ds.file_names, test_size=val_frac)
    val_ds = TensorDataset(X_val, Y_val)

    loader = DataLoader(
        val_ds,
        batch_size = d['batch_size'],
        shuffle    = False,
        num_workers= d['num_workers']
    )

    # ——— 8) Evaluate ———
    print(f"📝 Running validation on {len(val_ds)} samples…")
    metrics, loss, preds, trues = evaluate_model(
        model, loader, device, targ_scaler, nn.MSELoss()
    )

    # ——— 9) Print results ———
    print("\n📊 VALIDATION RESULTS\n" + "="*50)
    for name, m in metrics.items():
        print(f"{name.upper():12}  MAE: {m['mae']:.3f}  RMSE: {m['rmse']:.3f}  "
              f"R²: {m['r2']:.4f}  Acc(±{m['threshold']}): {m['accuracy_pct']:.1f}%")
    print("="*50 + f"\nMean validation loss: {loss:.6f}\n")

    # ——— 10) Save predictions CSV ———
    out_dir = Path(p['checkpoint_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "validation_predictions.csv"
    pd.DataFrame({
        'pred_final_spread': preds[:,0],
        'pred_final_total':  preds[:,1],
        'true_final_spread': trues[:,0],
        'true_final_total':  trues[:,1],
    }).to_csv(out_csv, index=False)
    print("📈 Saved predictions →", out_csv)
    print("\n✅ Validation-only run complete. No training performed.")

if __name__ == "__main__":
    main()
