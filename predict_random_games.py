import random
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import yaml
import json

# --- CONFIGURATION ---
CONFIG_PATH = "training_config.yaml"
MODEL_DIR = Path("models/model_data_v19")  # Change if using a different version
BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"
OUTCOMES_CSV = "C:/Users/admin/Desktop/training/outcomes.csv"
BM_PRED_PATH = "bm_pred_plus_outcomes.csv"  # Optional, for bookmaker lines

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    model_state = checkpoint['model']
    in_dim = checkpoint['in_dim']
    out_dim = checkpoint['out_dim']
    config = checkpoint['config']
    scaler = checkpoint['feature_scaler']
    target_scaler = checkpoint['target_scaler']
    # Model definition (must match train.py)
    class MLP(torch.nn.Module):
        def __init__(self, in_dim, hidden_dims, out_dim, dropout):
            super().__init__()
            layers = []
            last_dim = in_dim
            for h in hidden_dims:
                layers += [torch.nn.Linear(last_dim, h), torch.nn.ReLU(), torch.nn.Dropout(dropout)]
                last_dim = h
            layers.append(torch.nn.Linear(last_dim, out_dim))
            self.net = torch.nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)
    model = MLP(in_dim, config['model']['hidden_dims'], out_dim, config['model']['dropout'])
    model.load_state_dict(model_state)
    model.eval()
    model.to(device)
    return model, scaler, target_scaler, config

def get_random_games(num_games=100, data_dir=None, outcomes_csv=OUTCOMES_CSV):
    outcomes = pd.read_csv(outcomes_csv)
    all_files = outcomes['file_name'].tolist()
    selected_files = random.sample(all_files, num_games)
    return selected_files, outcomes[outcomes['file_name'].isin(selected_files)].reset_index(drop=True)

def load_game_features(game_file, scaler, max_rows, in_dim):
    df = pd.read_csv(game_file)
    df = df.select_dtypes(include=[np.number])
    # Pad/truncate to max_rows
    if len(df) > max_rows:
        df = df.iloc[:max_rows]
    elif len(df) < max_rows:
        pad = pd.DataFrame(np.zeros((max_rows - len(df), df.shape[1])), columns=df.columns)
        df = pd.concat([df, pad], ignore_index=True)
    arr = df.values.flatten()
    arr = scaler.transform([arr])
    arr = arr[:, :in_dim]  # Ensure correct input size
    return torch.from_numpy(arr).float()

def get_bookmaker_lines(game_file, bm_pred_path=BM_PRED_PATH):
    if not Path(bm_pred_path).exists():
        return None, None
    bm_df = pd.read_csv(bm_pred_path)
    row = bm_df[bm_df['filename'] == game_file]
    if row.empty:
        return None, None
    return float(row.iloc[0]['bm_spread_prediction_at_halftime']), float(row.iloc[0]['bm_total_prediction_at_halftime'])

def main():
    # Load config and model
    config = load_config(CONFIG_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler, target_scaler, cfg = load_model(BEST_MODEL_PATH, device)
    max_rows = cfg['data'].get('max_rows', 0)
    if not max_rows:
        # Try to infer max_rows from training
        max_rows = 0
        for f in Path(cfg['data']['train_dir']).glob("*.csv"):
            df = pd.read_csv(f)
            max_rows = max(max_rows, len(df))
    in_dim = cfg['model']['input_size'] or model.net[0].in_features

    # Pick random games
    selected_files, outcomes = get_random_games(100, cfg['data']['train_dir'], OUTCOMES_CSV)
    data_dir = Path(cfg['data']['train_dir'])

    results = []
    for idx, row in outcomes.iterrows():
        game_file = row['file_name']
        game_path = data_dir / game_file
        if not game_path.exists():
            continue
        features = load_game_features(game_path, scaler, max_rows, in_dim).to(device)
        with torch.no_grad():
            pred = model(features).cpu().numpy()
        pred_denorm = target_scaler.inverse_transform(pred)[0]
        pred_spread, pred_total = pred_denorm[0], pred_denorm[1]
        true_spread, true_total = row['final_spread'], row['final_total']
        bm_spread, bm_total = get_bookmaker_lines(game_file)
        # Betting logic
        spread_bet = "OVER" if bm_spread is not None and pred_spread > bm_spread else "UNDER"
        total_bet = "OVER" if bm_total is not None and pred_total > bm_total else "UNDER"
        spread_result = "OVER" if bm_spread is not None and true_spread > bm_spread else "UNDER"
        total_result = "OVER" if bm_total is not None and true_total > bm_total else "UNDER"
        spread_bet_win = (spread_bet == spread_result) if bm_spread is not None else None
        total_bet_win = (total_bet == total_result) if bm_total is not None else None
        results.append({
            "game_file": game_file,
            "pred_final_spread": pred_spread,
            "pred_final_total": pred_total,
            "true_final_spread": true_spread,
            "true_final_total": true_total,
            "bm_spread": bm_spread,
            "bm_total": bm_total,
            "spread_bet": spread_bet,
            "spread_result": spread_result,
            "spread_bet_win": spread_bet_win,
            "total_bet": total_bet,
            "total_result": total_result,
            "total_bet_win": total_bet_win,
            "spread_error": abs(pred_spread - true_spread),
            "total_error": abs(pred_total - true_total),
        })

    # Summary
    df = pd.DataFrame(results)
    print("="*60)
    print(f"Random Game Prediction Summary (n={len(df)})")
    print("="*60)
    print(f"Spread MAE: {df['spread_error'].mean():.2f}")
    print(f"Total MAE: {df['total_error'].mean():.2f}")
    if df['spread_bet_win'].notnull().any():
        print(f"Spread Bet Win Rate: {df['spread_bet_win'].mean()*100:.1f}%")
    if df['total_bet_win'].notnull().any():
        print(f"Total Bet Win Rate: {df['total_bet_win'].mean()*100:.1f}%")
    print(f"Spread Prediction ±3.0 Accuracy: {(df['spread_error'] <= 3.0).mean()*100:.1f}%")
    print(f"Total Prediction ±5.0 Accuracy: {(df['total_error'] <= 5.0).mean()*100:.1f}%")
    print("="*60)
    print("Sample predictions:")
    print(df[['game_file', 'pred_final_spread', 'true_final_spread', 'bm_spread', 'spread_bet', 'spread_result', 'spread_bet_win',
              'pred_final_total', 'true_final_total', 'bm_total', 'total_bet', 'total_result', 'total_bet_win']].head(10).to_string(index=False))
    print("="*60)

if __name__ == "__main__":
    main()
