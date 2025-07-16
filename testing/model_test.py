#!/usr/bin/env python3
"""
Batch ROI and betting stats extraction for every detailed_betting_analysis.csv inside each subfolder.
Prints full stats table and saves CSV summary for easy review.
"""

import os
import pandas as pd

def extract_betting_stats(csv_path):
    """Read CSV and extract profitability & accuracy stats."""
    df = pd.read_csv(csv_path)

    # Calculate stats
    spread_win_rate = df['spread_bet_won'].mean() * 100
    spread_roi = (df['spread_bet_won'].sum() - (~df['spread_bet_won']).sum()) / len(df) * 100

    total_win_rate = df['total_bet_won'].mean() * 100
    total_roi = (df['total_bet_won'].sum() - (~df['total_bet_won']).sum()) / len(df) * 100

    either_win_rate = df['either_bet_won'].mean() * 100
    either_roi = (df['either_bet_won'].sum() - (~df['either_bet_won']).sum()) / len(df) * 100

    spread_model_accuracy = df['spread_model_accuracy'].mean() * 100
    spread_bookmaker_accuracy = df['spread_bookmaker_accuracy'].mean() * 100
    total_model_accuracy = df['total_model_accuracy'].mean() * 100
    total_bookmaker_accuracy = df['total_bookmaker_accuracy'].mean() * 100

    return {
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
    }

def main():
    root_dir = r'C:\Users\admin\Desktop\training\models'

    stats = []
    print("\n🔎 Scanning for detailed_betting_analysis.csv files...\n")

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file == 'detailed_betting_analysis.csv':
                csv_path = os.path.join(subdir, file)
                model_folder = os.path.basename(subdir)
                try:
                    res = extract_betting_stats(csv_path)
                    res['model_folder'] = model_folder
                    res['csv_path'] = csv_path
                    stats.append(res)
                    print(f"📈 {model_folder:20} | Spread ROI: {res['spread_roi']:7.2f}% | Total ROI: {res['total_roi']:7.2f}% | Either ROI: {res['either_roi']:7.2f}%")
                except Exception as e:
                    print(f"❌ Error in {csv_path}: {e}")

    # To DataFrame and save
    if stats:
        df_stats = pd.DataFrame(stats)
        df_stats = df_stats[['model_folder', 'spread_win_rate', 'spread_roi', 'total_win_rate', 'total_roi',
                             'either_win_rate', 'either_roi',
                             'spread_model_accuracy', 'spread_bookmaker_accuracy',
                             'total_model_accuracy', 'total_bookmaker_accuracy', 'csv_path']]
        df_stats.to_csv(r'C:\Users\admin\Desktop\model_summary_stats.csv', index=False)
        print("\n✅ All model stats saved to C:\\Users\\admin\\Desktop\\model_summary_stats.csv\n")
        print(df_stats[['model_folder', 'spread_roi', 'total_roi', 'either_roi']].to_string(index=False))
        print(f"\nTotal models analyzed: {len(df_stats)}")
    else:
        print("⚠️ No betting analysis CSVs found.")

if __name__ == "__main__":
    main()
