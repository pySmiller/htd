import random
from pathlib import Path
import numpy as np
import pandas as pd
from predict import GamePredictor

def main():
    data_dir = Path("data")
    model_path = "models/main/best_model.pth"
    all_files = list(data_dir.glob("*.csv"))
    if not all_files:
        print("❌ No CSV files found in data directory")
        return

    sample_files = random.sample(all_files, min(100, len(all_files)))
    print(f"🎯 Validating model on {len(sample_files)} randomly selected games...")

    predictor = GamePredictor(model_path)
    predictions = predictor.predict_batch(sample_files)

    successful = [p for p in predictions if 'error' not in p]
    if not successful:
        print("❌ No successful predictions.")
        return

    df = pd.DataFrame(successful)
    print("\n📊 Prediction Summary (first 5):")
    print(df[["file", "final_spread", "final_total", "live_bm_spread_line", "live_bm_total_line"]].head())

    avg_spread = np.mean([p['final_spread'] for p in successful])
    avg_total = np.mean([p['final_total'] for p in successful])
    print(f"\n🎯 Average predicted spread: {avg_spread:.2f}")
    print(f"🎯 Average predicted total:  {avg_total:.2f}")

    # --- Bets won/loss calculation ---
    spread_wins = 0
    spread_losses = 0
    total_wins = 0
    total_losses = 0
    for p in successful:
        bm_spread = p.get('live_bm_spread_line')
        actual_spread = p.get('final_spread')
        if isinstance(bm_spread, (int, float)) and isinstance(actual_spread, (int, float)):
            # "Favourite" bet: actual_spread < bm_spread, "Underdog" bet: actual_spread > bm_spread
            # For validation, assume you bet "Favourite" if actual_spread < bm_spread, else "Underdog"
            bet_on_fav = actual_spread < bm_spread
            # Win if actual_spread is on correct side of bm_spread
            if (bet_on_fav and actual_spread < bm_spread) or (not bet_on_fav and actual_spread > bm_spread):
                spread_wins += 1
            else:
                spread_losses += 1
        bm_total = p.get('live_bm_total_line')
        actual_total = p.get('final_total')
        if isinstance(bm_total, (int, float)) and isinstance(actual_total, (int, float)):
            # "Over" bet: actual_total > bm_total, "Under" bet: actual_total < bm_total
            bet_on_over = actual_total > bm_total
            if (bet_on_over and actual_total > bm_total) or (not bet_on_over and actual_total < bm_total):
                total_wins += 1
            else:
                total_losses += 1

    print(f"\n🏀 Spread Bets: Won {spread_wins}, Lost {spread_losses}")
    print(f"🏀 Total Bets:  Won {total_wins}, Lost {total_losses}")

    df.to_csv("validator_predictions.csv", index=False)
    print("\n💾 Results saved to validator_predictions.csv")

if __name__ == "__main__":
    main()
