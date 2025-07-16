import os
import pandas as pd

# Set the root directory for models
root_dir = r'C:\Users\admin\Desktop\training\models'

results = []

for subdir, dirs, files in os.walk(root_dir):
    if 'detailed_betting_analysis.csv' in files:
        csv_path = os.path.join(subdir, 'detailed_betting_analysis.csv')
        try:
            df = pd.read_csv(csv_path)
            
            # Profitability for Spread Bets
            spread_win_rate = df['spread_bet_won'].mean() * 100
            spread_roi = (df['spread_bet_won'].sum() - (~df['spread_bet_won']).sum()) / len(df) * 100

            # Profitability for Total Bets
            total_win_rate = df['total_bet_won'].mean() * 100
            total_roi = (df['total_bet_won'].sum() - (~df['total_bet_won']).sum()) / len(df) * 100

            # Profitability for Either Bet
            either_win_rate = df['either_bet_won'].mean() * 100
            either_roi = (df['either_bet_won'].sum() - (~df['either_bet_won']).sum()) / len(df) * 100

            # Model/Bookmaker Accuracy
            spread_model_accuracy = df['spread_model_accuracy'].mean() * 100
            spread_bookmaker_accuracy = df['spread_bookmaker_accuracy'].mean() * 100
            total_model_accuracy = df['total_model_accuracy'].mean() * 100
            total_bookmaker_accuracy = df['total_bookmaker_accuracy'].mean() * 100

            results.append({
                "model_folder": os.path.basename(subdir),
                "spread_win_rate": spread_win_rate,
                "spread_roi": spread_roi,
                "total_win_rate": total_win_rate,
                "total_roi": total_roi,
                "either_win_rate": either_win_rate,
                "either_roi": either_roi,
                "spread_model_accuracy": spread_model_accuracy,
                "spread_bookmaker_accuracy": spread_bookmaker_accuracy,
                "total_model_accuracy": total_model_accuracy,
                "total_bookmaker_accuracy": total_bookmaker_accuracy
            })

        except Exception as e:
            print(f"Error processing {csv_path}: {e}")

# Convert results to DataFrame
summary_df = pd.DataFrame(results)

# Save to CSV (optional)
summary_df.to_csv(r'C:\Users\admin\Desktop\model_summary_stats.csv', index=False)

# Print result
print(summary_df)
