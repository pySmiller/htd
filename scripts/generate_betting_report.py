#!/usr/bin/env python
"""
Generate a comprehensive betting report with enhanced formatting and analysis
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

def generate_enhanced_betting_report(model_dir):
    """
    Generate an enhanced betting report with better formatting and insights
    """
    # Load the detailed betting analysis
    detailed_file = Path(model_dir) / 'detailed_betting_analysis.csv'
    
    if not detailed_file.exists():
        print("❌ Detailed betting analysis not found. Please run generate_detailed_analysis.py first.")
        return
    
    df = pd.read_csv(detailed_file)
    
    # Create enhanced summary report
    report_path = Path(model_dir) / 'BETTING_PERFORMANCE_REPORT.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("COMPREHENSIVE BETTING PERFORMANCE REPORT\n")
        f.write("=" * 100 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Version: {model_dir.name}\n")
        f.write("=" * 100 + "\n\n")
        
        # Executive Summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 50 + "\n")
        total_games = len(df)
        spread_win_rate = df['spread_bet_won'].mean() * 100
        total_win_rate = df['total_bet_won'].mean() * 100
        either_win_rate = df['either_bet_won'].mean() * 100
        both_win_rate = df['both_bets_won'].mean() * 100
        
        f.write(f"Total Games Analyzed: {total_games:,}\n")
        f.write(f"Overall Success Rate: {either_win_rate:.1f}% (Either bet wins)\n")
        f.write(f"Spread Betting Success: {spread_win_rate:.1f}%\n")
        f.write(f"Total Betting Success: {total_win_rate:.1f}%\n")
        f.write(f"Both Bets Success: {both_win_rate:.1f}%\n")
        
        # Profitability estimate (assuming -110 odds)
        if spread_win_rate > 52.38:  # Break-even point for -110 odds
            f.write(f"SPREAD BETTING: PROFITABLE (Need 52.38% to break even)\n")
        else:
            f.write(f"SPREAD BETTING: NOT PROFITABLE (Need 52.38% to break even)\n")
            
        if total_win_rate > 52.38:
            f.write(f"TOTAL BETTING: PROFITABLE (Need 52.38% to break even)\n")
        else:
            f.write(f"TOTAL BETTING: NOT PROFITABLE (Need 52.38% to break even)\n")
        
        f.write("\n" + "=" * 100 + "\n\n")
        
        # Detailed Performance Analysis
        f.write("DETAILED PERFORMANCE ANALYSIS\n")
        f.write("-" * 50 + "\n\n")
        
        # Spread Analysis
        f.write("SPREAD BETTING ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        spread_wins = df['spread_bet_won'].sum()
        spread_losses = total_games - spread_wins
        avg_spread_confidence = df['spread_confidence_level'].mean()
        high_conf_spread = (df['spread_confidence_level'] > 10).sum()
        
        f.write(f"Wins: {spread_wins:,}\n")
        f.write(f"Losses: {spread_losses:,}\n")
        f.write(f"Win Rate: {spread_win_rate:.1f}%\n")
        f.write(f"Average Confidence: {avg_spread_confidence:.1f} points\n")
        f.write(f"High Confidence Bets (>10 pts): {high_conf_spread:,}\n")
        
        # High confidence spread performance
        if high_conf_spread > 0:
            high_conf_spread_df = df[df['spread_confidence_level'] > 10]
            high_conf_spread_rate = high_conf_spread_df['spread_bet_won'].mean() * 100
            f.write(f"High Confidence Win Rate: {high_conf_spread_rate:.1f}%\n")
        
        f.write("\n")
        
        # Total Analysis
        f.write("TOTAL BETTING ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        total_wins = df['total_bet_won'].sum()
        total_losses = total_games - total_wins
        avg_total_confidence = df['total_confidence_level'].mean()
        high_conf_total = (df['total_confidence_level'] > 15).sum()
        
        f.write(f"Wins: {total_wins:,}\n")
        f.write(f"Losses: {total_losses:,}\n")
        f.write(f"Win Rate: {total_win_rate:.1f}%\n")
        f.write(f"Average Confidence: {avg_total_confidence:.1f} points\n")
        f.write(f"High Confidence Bets (>15 pts): {high_conf_total:,}\n")
        
        # High confidence total performance
        if high_conf_total > 0:
            high_conf_total_df = df[df['total_confidence_level'] > 15]
            high_conf_total_rate = high_conf_total_df['total_bet_won'].mean() * 100
            f.write(f"High Confidence Win Rate: {high_conf_total_rate:.1f}%\n")
        
        f.write("\n" + "=" * 100 + "\n\n")
        
        # Best Performing Bets
        f.write("BEST PERFORMING BETS\n")
        f.write("-" * 50 + "\n\n")
        
        # Top 20 spread bets
        f.write("TOP 20 SPREAD BETS (by confidence):\n")
        f.write("-" * 40 + "\n")
        f.write("Game | Date | Bet | Confidence | Result | Actual vs Predicted\n")
        f.write("-" * 70 + "\n")
        
        top_spread = df.nlargest(20, 'spread_confidence_level')
        for _, row in top_spread.iterrows():
            date = row['game_date'][:10] if len(row['game_date']) > 10 else row['game_date']
            result = "WIN" if row['spread_bet_won'] else "LOSS"
            f.write(f"{row['game_id']:4d} | {date} | {row['spread_model_bet']:5s} | {row['spread_confidence_level']:8.1f} | {result:4s} | {row['spread_actual_result']:4.0f} vs {row['spread_model_prediction']:5.1f}\n")
        
        f.write("\n")
        
        # Top 20 total bets
        f.write("TOP 20 TOTAL BETS (by confidence):\n")
        f.write("-" * 40 + "\n")
        f.write("Game | Date | Bet | Confidence | Result | Actual vs Predicted\n")
        f.write("-" * 70 + "\n")
        
        top_total = df.nlargest(20, 'total_confidence_level')
        for _, row in top_total.iterrows():
            date = row['game_date'][:10] if len(row['game_date']) > 10 else row['game_date']
            result = "WIN" if row['total_bet_won'] else "LOSS"
            f.write(f"{row['game_id']:4d} | {date} | {row['total_model_bet']:5s} | {row['total_confidence_level']:8.1f} | {result:4s} | {row['total_actual_result']:4.0f} vs {row['total_model_prediction']:5.1f}\n")
        
        f.write("\n" + "=" * 100 + "\n\n")
        
        # Monthly Performance (if dates are available)
        f.write("PERFORMANCE BY MONTH\n")
        f.write("-" * 50 + "\n")
        
        # Extract month from game_date and analyze
        df_copy = df.copy()
        df_copy['month'] = df_copy['game_date'].str[:7]  # Extract YYYY-MM
        monthly_stats = df_copy.groupby('month').agg({
            'spread_bet_won': ['count', 'sum', 'mean'],
            'total_bet_won': ['count', 'sum', 'mean'],
            'either_bet_won': ['sum', 'mean']
        }).round(3)
        
        f.write("Month    | Games | Spread W/L | Spread% | Total W/L | Total% | Either%\n")
        f.write("-" * 70 + "\n")
        
        for month in monthly_stats.index:
            games = monthly_stats.loc[month, ('spread_bet_won', 'count')]
            spread_wins = monthly_stats.loc[month, ('spread_bet_won', 'sum')]
            spread_rate = monthly_stats.loc[month, ('spread_bet_won', 'mean')] * 100
            total_wins = monthly_stats.loc[month, ('total_bet_won', 'sum')]
            total_rate = monthly_stats.loc[month, ('total_bet_won', 'mean')] * 100
            either_rate = monthly_stats.loc[month, ('either_bet_won', 'mean')] * 100
            
            f.write(f"{month} | {games:5.0f} | {spread_wins:4.0f}/{games-spread_wins:4.0f} | {spread_rate:6.1f}% | {total_wins:4.0f}/{games-total_wins:4.0f} | {total_rate:6.1f}% | {either_rate:6.1f}%\n")
        
        f.write("\n" + "=" * 100 + "\n\n")
        
        # Betting Recommendations
        f.write("BETTING RECOMMENDATIONS\n")
        f.write("-" * 50 + "\n")
        
        # Analyze patterns
        high_conf_spread_threshold = np.percentile(df['spread_confidence_level'], 75)
        high_conf_total_threshold = np.percentile(df['total_confidence_level'], 75)
        
        # Best spread opportunities
        best_spread_opportunities = df[df['spread_confidence_level'] >= high_conf_spread_threshold]
        best_spread_rate = best_spread_opportunities['spread_bet_won'].mean() * 100
        
        # Best total opportunities
        best_total_opportunities = df[df['total_confidence_level'] >= high_conf_total_threshold]
        best_total_rate = best_total_opportunities['total_bet_won'].mean() * 100
        
        f.write(f"SPREAD BETTING STRATEGY:\n")
        f.write(f"- Focus on bets with confidence >= {high_conf_spread_threshold:.1f} points\n")
        f.write(f"- This gives you {len(best_spread_opportunities)} opportunities with {best_spread_rate:.1f}% win rate\n")
        f.write(f"- Current spread betting is {'PROFITABLE' if spread_win_rate > 52.38 else 'NOT PROFITABLE'}\n\n")
        
        f.write(f"TOTAL BETTING STRATEGY:\n")
        f.write(f"- Focus on bets with confidence >= {high_conf_total_threshold:.1f} points\n")
        f.write(f"- This gives you {len(best_total_opportunities)} opportunities with {best_total_rate:.1f}% win rate\n")
        f.write(f"- Current total betting is {'PROFITABLE' if total_win_rate > 52.38 else 'NOT PROFITABLE'}\n\n")
        
        f.write("OVERALL RECOMMENDATION:\n")
        if spread_win_rate > 52.38:
            f.write("✅ PROCEED WITH SPREAD BETTING - Model shows profitable edge\n")
        else:
            f.write("❌ AVOID SPREAD BETTING - Model needs improvement\n")
        
        if total_win_rate > 52.38:
            f.write("✅ PROCEED WITH TOTAL BETTING - Model shows profitable edge\n")
        else:
            f.write("❌ AVOID TOTAL BETTING - Model needs improvement\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("End of Report\n")
        f.write("=" * 100 + "\n")
    
    print(f"✅ Enhanced betting report saved to: {report_path}")
    return report_path

if __name__ == "__main__":
    model_dir = Path("models/model_data_v8")
    if model_dir.exists():
        report_path = generate_enhanced_betting_report(model_dir)
        print(f"\n📊 Report generated successfully!")
        print(f"📄 View the comprehensive report at: {report_path}")
    else:
        print("❌ Model directory not found. Please run training first.")
