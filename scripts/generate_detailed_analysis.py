#!/usr/bin/env python
"""
Generate detailed betting analysis from existing model results
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

def create_detailed_betting_analysis(model_predictions_path, bm_predictions_path, output_dir):
    """
    Create a detailed betting analysis showing each individual bet with comprehensive details
    """
    print(f"Creating detailed betting analysis...")
    
    try:
        # Load data
        model_preds = pd.read_csv(model_predictions_path)
        bm_data = pd.read_csv(bm_predictions_path)
        
        # Sample data to match sizes (same logic as in compare_with_bookmaker)
        if len(model_preds) > len(bm_data):
            model_sample = model_preds.sample(n=len(bm_data), random_state=42).reset_index(drop=True)
            bm_sample = bm_data.copy()
        else:
            bm_sample = bm_data.sample(n=len(model_preds), random_state=42).reset_index(drop=True)
            model_sample = model_preds.copy()
        
        # Create detailed analysis for each bet
        detailed_bets = []
        
        for i in range(len(model_sample)):
            # Get predictions and actuals
            model_spread = model_sample.iloc[i]['pred_final_spread']
            model_total = model_sample.iloc[i]['pred_final_total']
            bm_spread = bm_sample.iloc[i]['bm_spread_prediction_at_halftime']
            bm_total = bm_sample.iloc[i]['bm_total_prediction_at_halftime']
            actual_spread = bm_sample.iloc[i]['final_spread']
            actual_total = bm_sample.iloc[i]['final_total']
            
            # Betting decisions and outcomes
            # SPREAD BET ANALYSIS
            model_spread_bet = "OVER" if model_spread > bm_spread else "UNDER"
            actual_spread_result = "OVER" if actual_spread > bm_spread else "UNDER"
            spread_bet_won = model_spread_bet == actual_spread_result
            spread_confidence = abs(model_spread - bm_spread)
            
            # TOTAL BET ANALYSIS  
            model_total_bet = "OVER" if model_total > bm_total else "UNDER"
            actual_total_result = "OVER" if actual_total > bm_total else "UNDER"
            total_bet_won = model_total_bet == actual_total_result
            total_confidence = abs(model_total - bm_total)
            
            # Accuracy calculations
            spread_accuracy = abs(model_spread - actual_spread) <= 3.0
            total_accuracy = abs(model_total - actual_total) <= 5.0
            bm_spread_accuracy = abs(bm_spread - actual_spread) <= 3.0
            bm_total_accuracy = abs(bm_total - actual_total) <= 5.0
            
            # Edge calculations (how much better/worse than bookmaker)
            spread_edge = abs(model_spread - actual_spread) - abs(bm_spread - actual_spread)
            total_edge = abs(model_total - actual_total) - abs(bm_total - actual_total)
            
            # Game info (if available)
            game_file = bm_sample.iloc[i].get('filename', f'game_{i+1}')
            game_date = game_file.replace('game_', '').replace('.csv', '') if 'game_' in game_file else 'unknown'
            
            bet_analysis = {
                # Game Info
                'game_id': i + 1,
                'game_file': game_file,
                'game_date': game_date,
                
                # SPREAD BET DETAILS
                'spread_bookmaker_line': bm_spread,
                'spread_model_prediction': round(model_spread, 2),
                'spread_actual_result': actual_spread,
                'spread_model_bet': model_spread_bet,
                'spread_actual_outcome': actual_spread_result,
                'spread_bet_won': spread_bet_won,
                'spread_confidence_level': round(spread_confidence, 2),
                'spread_model_accuracy': spread_accuracy,
                'spread_bookmaker_accuracy': bm_spread_accuracy,
                'spread_edge_vs_bookmaker': round(spread_edge, 2),
                'spread_prediction_error': round(abs(model_spread - actual_spread), 2),
                'spread_bookmaker_error': round(abs(bm_spread - actual_spread), 2),
                
                # TOTAL BET DETAILS
                'total_bookmaker_line': bm_total,
                'total_model_prediction': round(model_total, 2),
                'total_actual_result': actual_total,
                'total_model_bet': model_total_bet,
                'total_actual_outcome': actual_total_result,
                'total_bet_won': total_bet_won,
                'total_confidence_level': round(total_confidence, 2),
                'total_model_accuracy': total_accuracy,
                'total_bookmaker_accuracy': bm_total_accuracy,
                'total_edge_vs_bookmaker': round(total_edge, 2),
                'total_prediction_error': round(abs(model_total - actual_total), 2),
                'total_bookmaker_error': round(abs(bm_total - actual_total), 2),
                
                # OVERALL ANALYSIS
                'both_bets_won': spread_bet_won and total_bet_won,
                'either_bet_won': spread_bet_won or total_bet_won,
                'overall_confidence': round((spread_confidence + total_confidence) / 2, 2),
                'model_better_than_bookmaker': (spread_edge < 0) or (total_edge < 0)
            }
            
            detailed_bets.append(bet_analysis)
        
        # Convert to DataFrame
        detailed_df = pd.DataFrame(detailed_bets)
        
        # Create summary statistics
        summary_stats = {
            'total_games': len(detailed_df),
            'spread_bets_won': detailed_df['spread_bet_won'].sum(),
            'spread_bet_win_rate': detailed_df['spread_bet_won'].mean() * 100,
            'total_bets_won': detailed_df['total_bet_won'].sum(),
            'total_bet_win_rate': detailed_df['total_bet_won'].mean() * 100,
            'both_bets_won': detailed_df['both_bets_won'].sum(),
            'both_bets_win_rate': detailed_df['both_bets_won'].mean() * 100,
            'either_bet_won': detailed_df['either_bet_won'].sum(),
            'either_bet_win_rate': detailed_df['either_bet_won'].mean() * 100,
            'high_confidence_spread_bets': (detailed_df['spread_confidence_level'] > 5.0).sum(),
            'high_confidence_total_bets': (detailed_df['total_confidence_level'] > 8.0).sum(),
            'avg_spread_confidence': detailed_df['spread_confidence_level'].mean(),
            'avg_total_confidence': detailed_df['total_confidence_level'].mean(),
            'model_beats_bookmaker_games': detailed_df['model_better_than_bookmaker'].sum(),
            'model_beats_bookmaker_rate': detailed_df['model_better_than_bookmaker'].mean() * 100
        }
        
        # Save detailed analysis
        detailed_file = Path(output_dir) / 'detailed_betting_analysis.csv'
        detailed_df.to_csv(detailed_file, index=False)
        
        # Create a summary report
        summary_file = Path(output_dir) / 'betting_summary_report.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("DETAILED BETTING ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"OVERALL STATISTICS:\n")
            f.write(f"   Total Games Analyzed: {summary_stats['total_games']}\n")
            f.write(f"   Model vs Bookmaker Win Rate: {summary_stats['model_beats_bookmaker_rate']:.1f}%\n\n")
            
            f.write(f"SPREAD BETTING PERFORMANCE:\n")
            f.write(f"   Spread Bets Won: {summary_stats['spread_bets_won']}/{summary_stats['total_games']}\n")
            f.write(f"   Spread Win Rate: {summary_stats['spread_bet_win_rate']:.1f}%\n")
            f.write(f"   Average Confidence: {summary_stats['avg_spread_confidence']:.2f}\n")
            f.write(f"   High Confidence Bets: {summary_stats['high_confidence_spread_bets']}\n\n")
            
            f.write(f"TOTAL BETTING PERFORMANCE:\n")
            f.write(f"   Total Bets Won: {summary_stats['total_bets_won']}/{summary_stats['total_games']}\n")
            f.write(f"   Total Win Rate: {summary_stats['total_bet_win_rate']:.1f}%\n")
            f.write(f"   Average Confidence: {summary_stats['avg_total_confidence']:.2f}\n")
            f.write(f"   High Confidence Bets: {summary_stats['high_confidence_total_bets']}\n\n")

            f.write(f"COMBINED BETTING PERFORMANCE:\n")
            f.write(f"   Both Bets Won: {summary_stats['both_bets_won']}/{summary_stats['total_games']}\n")
            f.write(f"   Both Bets Win Rate: {summary_stats['both_bets_win_rate']:.1f}%\n")
            f.write(f"   Either Bet Won: {summary_stats['either_bet_won']}/{summary_stats['total_games']}\n")
            f.write(f"   Either Bet Win Rate: {summary_stats['either_bet_win_rate']:.1f}%\n\n")

            spread_roi = ((summary_stats['spread_bets_won'] * 0.91) - (summary_stats['total_games'] - summary_stats['spread_bets_won'])) / summary_stats['total_games'] * 100 if summary_stats['total_games'] else 0
            total_roi = ((summary_stats['total_bets_won'] * 0.91) - (summary_stats['total_games'] - summary_stats['total_bets_won'])) / summary_stats['total_games'] * 100 if summary_stats['total_games'] else 0
            overall_roi = (((summary_stats['spread_bets_won'] + summary_stats['total_bets_won']) * 0.91) - (2 * summary_stats['total_games'] - (summary_stats['spread_bets_won'] + summary_stats['total_bets_won']))) / (2 * summary_stats['total_games']) * 100 if summary_stats['total_games'] else 0
            f.write(f"   Estimated Spread ROI: {spread_roi:.1f}%\n")
            f.write(f"   Estimated Total ROI: {total_roi:.1f}%\n")
            f.write(f"   Combined ROI: {overall_roi:.1f}%\n\n")
            
            # Top performing bets
            f.write(f"TOP 10 HIGHEST CONFIDENCE SPREAD BETS:\n")
            top_spread = detailed_df.nlargest(10, 'spread_confidence_level')
            for _, row in top_spread.iterrows():
                f.write(f"   Game {row['game_id']}: {row['spread_model_bet']} (Confidence: {row['spread_confidence_level']}) - {'WON' if row['spread_bet_won'] else 'LOST'}\n")
            
            f.write(f"\nTOP 10 HIGHEST CONFIDENCE TOTAL BETS:\n")
            top_total = detailed_df.nlargest(10, 'total_confidence_level')
            for _, row in top_total.iterrows():
                f.write(f"   Game {row['game_id']}: {row['total_model_bet']} (Confidence: {row['total_confidence_level']}) - {'WON' if row['total_bet_won'] else 'LOST'}\n")
        
        print(f"✅ Detailed betting analysis saved to: {detailed_file}")
        print(f"✅ Summary report saved to: {summary_file}")
        
        return summary_stats
        
    except Exception as e:
        print(f"❌ Error creating detailed analysis: {e}")
        return {}

if __name__ == "__main__":
    # Use the most recent model results
    model_dir = Path("models/model_data_v8")
    model_predictions_path = model_dir / "validation_predictions.csv"
    bm_predictions_path = "bm_pred_plus_outcomes.csv"
    
    if model_predictions_path.exists() and Path(bm_predictions_path).exists():
        stats = create_detailed_betting_analysis(model_predictions_path, bm_predictions_path, model_dir)
        print(f"\n📊 SUMMARY STATISTICS:")
        print(f"   Spread Win Rate: {stats.get('spread_bet_win_rate', 0):.1f}%")
        print(f"   Total Win Rate: {stats.get('total_bet_win_rate', 0):.1f}%")
        print(f"   Overall Betting Success: {stats.get('either_bet_win_rate', 0):.1f}%")
    else:
        print("❌ Required files not found. Please run training first.")
