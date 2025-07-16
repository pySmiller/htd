#!/usr/bin/env python3
"""
Comprehensive Prediction Comparison: Betting Market vs Neural Network Model
============================================================================
This script compares:
1. Betting Market predictions (bm_predictions.csv)
2. Neural Network Model predictions (validation_predictions.csv)
3. Actual outcomes (outcomes.csv)

Generates detailed performance metrics and saves comparison results.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yaml
import os
from pathlib import Path

def load_config():
    """Load training configuration"""
    with open('training_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def calculate_metrics(predictions, actuals, name=""):
    """Calculate comprehensive metrics for predictions vs actuals"""
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    r2 = r2_score(actuals, predictions)
    
    # Calculate accuracy within tolerance
    spread_tolerance = 3.0
    total_tolerance = 5.0
    
    if "spread" in name.lower():
        accuracy = np.mean(np.abs(predictions - actuals) <= spread_tolerance) * 100
        tolerance_str = f"±{spread_tolerance}"
    else:
        accuracy = np.mean(np.abs(predictions - actuals) <= total_tolerance) * 100
        tolerance_str = f"±{total_tolerance}"
    
    return {
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'Accuracy': accuracy,
        'Tolerance': tolerance_str,
        'Mean_Pred': np.mean(predictions),
        'Mean_Actual': np.mean(actuals)
    }

def load_validation_filenames():
    """Load validation set filenames from training script logic"""
    # Load outcomes to get all filenames
    outcomes_df = pd.read_csv('outcomes.csv')
    
    # Use same train/val split as training (80/20 with seed 42)
    np.random.seed(42)
    n_samples = len(outcomes_df)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    train_size = int(0.8 * n_samples)
    val_indices = indices[train_size:]
    
    # Get validation filenames
    val_filenames = outcomes_df.iloc[val_indices]['filename'].tolist()
    return val_filenames

def main():
    print("🔍 COMPREHENSIVE PREDICTION COMPARISON")
    print("=" * 60)
    md_lines = []
    md_lines.append("# Comprehensive Prediction Comparison\n")
    md_lines.append("This report compares the performance of the Betting Market and Neural Network Model predictions against actual outcomes.\n")
    md_lines.append("---\n")
    
    # Load configuration
    config = load_config()
    
    # Load data files
    print("📊 Loading data files...")
    md_lines.append("## Data Overview\n")
    bm_df = pd.read_csv('resources/bm_predictions.csv')
    val_predictions_df = pd.read_csv('checkpoints/validation_predictions.csv')
    outcomes_df = pd.read_csv('outcomes.csv')
    
    # Get validation filenames
    val_filenames = load_validation_filenames()
    
    print(f"📈 Validation set size: {len(val_filenames)}")
    print(f"📈 Model predictions: {len(val_predictions_df)}")
    print(f"📈 Betting market predictions: {len(bm_df)}")
    print(f"📈 Actual outcomes: {len(outcomes_df)}")
    md_lines.append(f"- Validation set size: **{len(val_filenames)}**  ")
    md_lines.append(f"- Model predictions: **{len(val_predictions_df)}**  ")
    md_lines.append(f"- Betting market predictions: **{len(bm_df)}**  ")
    md_lines.append(f"- Actual outcomes: **{len(outcomes_df)}**  \n")
    
    # Filter betting market and outcomes for validation games only
    bm_val = bm_df[bm_df['filename'].isin(val_filenames)].copy()
    outcomes_val = outcomes_df[outcomes_df['filename'].isin(val_filenames)].copy()
    
    # Sort by filename to ensure alignment
    bm_val = bm_val.sort_values('filename').reset_index(drop=True)
    outcomes_val = outcomes_val.sort_values('filename').reset_index(drop=True)
    
    # Verify alignment
    if len(bm_val) != len(val_predictions_df) or len(outcomes_val) != len(val_predictions_df):
        print("⚠️  Warning: Data alignment issue detected!")
        print(f"   BM predictions: {len(bm_val)}")
        print(f"   Model predictions: {len(val_predictions_df)}")
        print(f"   Outcomes: {len(outcomes_val)}")
        
        # Take intersection
        common_files = set(bm_val['filename']) & set(outcomes_val['filename'])
        min_size = min(len(bm_val), len(outcomes_val), len(val_predictions_df))
        
        bm_val = bm_val.head(min_size)
        outcomes_val = outcomes_val.head(min_size)
        val_predictions_df = val_predictions_df.head(min_size)
        
        print(f"   Using {min_size} aligned samples")
    
    # Create comprehensive comparison DataFrame
    comparison_df = pd.DataFrame({
        'filename': bm_val['filename'] if len(bm_val) > 0 else [f"game_{i}" for i in range(len(val_predictions_df))],
        
        # Betting Market Predictions
        'bm_pred_spread': bm_val['bm_live_spread'].values if len(bm_val) > 0 else [0] * len(val_predictions_df),
        'bm_pred_total': bm_val['bm_live_total'].values if len(bm_val) > 0 else [0] * len(val_predictions_df),
        
        # Model Predictions
        'model_pred_spread': val_predictions_df['pred_final_spread'].values,
        'model_pred_total': val_predictions_df['pred_final_total'].values,
        
        # Actual Outcomes - use the true values from validation file for consistency
        'actual_spread': val_predictions_df['true_final_spread'].values,
        'actual_total': val_predictions_df['true_final_total'].values,
    })
    
    # If we have matching outcomes file, use it for betting market comparison
    if len(bm_val) > 0 and len(outcomes_val) > 0:
        # For betting market comparison, use outcomes from outcomes.csv
        comparison_df['bm_actual_spread'] = outcomes_val['final_spread'].values if len(outcomes_val) == len(bm_val) else comparison_df['actual_spread'].values
        comparison_df['bm_actual_total'] = outcomes_val['final_total'].values if len(outcomes_val) == len(bm_val) else comparison_df['actual_total'].values
    else:
        comparison_df['bm_actual_spread'] = comparison_df['actual_spread'].values
        comparison_df['bm_actual_total'] = comparison_df['actual_total'].values
    
    # Calculate errors
    comparison_df['bm_spread_error'] = comparison_df['bm_pred_spread'] - comparison_df['bm_actual_spread']
    comparison_df['bm_total_error'] = comparison_df['bm_pred_total'] - comparison_df['bm_actual_total']
    comparison_df['model_spread_error'] = comparison_df['model_pred_spread'] - comparison_df['actual_spread']
    comparison_df['model_total_error'] = comparison_df['model_pred_total'] - comparison_df['actual_total']
    
    # Absolute errors
    comparison_df['bm_spread_abs_error'] = np.abs(comparison_df['bm_spread_error'])
    comparison_df['bm_total_abs_error'] = np.abs(comparison_df['bm_total_error'])
    comparison_df['model_spread_abs_error'] = np.abs(comparison_df['model_spread_error'])
    comparison_df['model_total_abs_error'] = np.abs(comparison_df['model_total_error'])
    
    # Calculate metrics for each prediction type
    print("\n🎯 PERFORMANCE COMPARISON")
    print("=" * 60)
    md_lines.append("\n## Performance Comparison\n")
    # Spread predictions
    print("\n📊 SPREAD PREDICTIONS:")
    print("-" * 40)
    md_lines.append("### Spread Predictions\n")
    if len(bm_val) > 0:
        bm_spread_metrics = calculate_metrics(
            comparison_df['bm_pred_spread'], 
            comparison_df['bm_actual_spread'], 
            "spread"
        )
        print(f"Betting Market:")
        print(f"  MAE: {bm_spread_metrics['MAE']:.3f}")
        print(f"  RMSE: {bm_spread_metrics['RMSE']:.3f}")
        print(f"  R²: {bm_spread_metrics['R²']:.4f}")
        print(f"  Accuracy ({bm_spread_metrics['Tolerance']}): {bm_spread_metrics['Accuracy']:.1f}%")
        md_lines.append(f"**Betting Market**  ")
        md_lines.append(f"- MAE: `{bm_spread_metrics['MAE']:.3f}`  ")
        md_lines.append(f"- RMSE: `{bm_spread_metrics['RMSE']:.3f}`  ")
        md_lines.append(f"- R²: `{bm_spread_metrics['R²']:.4f}`  ")
        md_lines.append(f"- Accuracy ({bm_spread_metrics['Tolerance']}): `{bm_spread_metrics['Accuracy']:.1f}%`  ")
    model_spread_metrics = calculate_metrics(
        comparison_df['model_pred_spread'], 
        comparison_df['actual_spread'], 
        "spread"
    )
    print(f"Neural Network Model:")
    print(f"  MAE: {model_spread_metrics['MAE']:.3f}")
    print(f"  RMSE: {model_spread_metrics['RMSE']:.3f}")
    print(f"  R²: {model_spread_metrics['R²']:.4f}")
    print(f"  Accuracy ({model_spread_metrics['Tolerance']}): {model_spread_metrics['Accuracy']:.1f}%")
    md_lines.append(f"**Neural Network Model**  ")
    md_lines.append(f"- MAE: `{model_spread_metrics['MAE']:.3f}`  ")
    md_lines.append(f"- RMSE: `{model_spread_metrics['RMSE']:.3f}`  ")
    md_lines.append(f"- R²: `{model_spread_metrics['R²']:.4f}`  ")
    md_lines.append(f"- Accuracy ({model_spread_metrics['Tolerance']}): `{model_spread_metrics['Accuracy']:.1f}%`  \n")
    # Total predictions
    print("\n📊 TOTAL PREDICTIONS:")
    print("-" * 40)
    md_lines.append("### Total Predictions\n")
    if len(bm_val) > 0:
        bm_total_metrics = calculate_metrics(
            comparison_df['bm_pred_total'], 
            comparison_df['bm_actual_total'], 
            "total"
        )
        print(f"Betting Market:")
        print(f"  MAE: {bm_total_metrics['MAE']:.3f}")
        print(f"  RMSE: {bm_total_metrics['RMSE']:.3f}")
        print(f"  R²: {bm_total_metrics['R²']:.4f}")
        print(f"  Accuracy ({bm_total_metrics['Tolerance']}): {bm_total_metrics['Accuracy']:.1f}%")
        md_lines.append(f"**Betting Market**  ")
        md_lines.append(f"- MAE: `{bm_total_metrics['MAE']:.3f}`  ")
        md_lines.append(f"- RMSE: `{bm_total_metrics['RMSE']:.3f}`  ")
        md_lines.append(f"- R²: `{bm_total_metrics['R²']:.4f}`  ")
        md_lines.append(f"- Accuracy ({bm_total_metrics['Tolerance']}): `{bm_total_metrics['Accuracy']:.1f}%`  ")
    model_total_metrics = calculate_metrics(
        comparison_df['model_pred_total'], 
        comparison_df['actual_total'], 
        "total"
    )
    print(f"Neural Network Model:")
    print(f"  MAE: {model_total_metrics['MAE']:.3f}")
    print(f"  RMSE: {model_total_metrics['RMSE']:.3f}")
    print(f"  R²: {model_total_metrics['R²']:.4f}")
    print(f"  Accuracy ({model_total_metrics['Tolerance']}): {model_total_metrics['Accuracy']:.1f}%")
    md_lines.append(f"**Neural Network Model**  ")
    md_lines.append(f"- MAE: `{model_total_metrics['MAE']:.3f}`  ")
    md_lines.append(f"- RMSE: `{model_total_metrics['RMSE']:.3f}`  ")
    md_lines.append(f"- R²: `{model_total_metrics['R²']:.4f}`  ")
    md_lines.append(f"- Accuracy ({model_total_metrics['Tolerance']}): `{model_total_metrics['Accuracy']:.1f}%`  \n")
    
    # Head-to-head comparison
    if len(bm_val) > 0:
        print("\n🏆 HEAD-TO-HEAD COMPARISON:")
        print("-" * 40)
        md_lines.append("## Head-to-Head Comparison\n")
        # Count wins for each metric
        spread_wins_model = np.sum(comparison_df['model_spread_abs_error'] < comparison_df['bm_spread_abs_error'])
        spread_wins_bm = np.sum(comparison_df['bm_spread_abs_error'] < comparison_df['model_spread_abs_error'])
        spread_ties = len(comparison_df) - spread_wins_model - spread_wins_bm
        total_wins_model = np.sum(comparison_df['model_total_abs_error'] < comparison_df['bm_total_abs_error'])
        total_wins_bm = np.sum(comparison_df['bm_total_abs_error'] < comparison_df['model_total_abs_error'])
        total_ties = len(comparison_df) - total_wins_model - total_wins_bm
        print(f"Spread Predictions (better absolute error):")
        print(f"  🤖 Model wins: {spread_wins_model} ({spread_wins_model/len(comparison_df)*100:.1f}%)")
        print(f"  📈 BM wins: {spread_wins_bm} ({spread_wins_bm/len(comparison_df)*100:.1f}%)")
        print(f"  🤝 Ties: {spread_ties} ({spread_ties/len(comparison_df)*100:.1f}%)")
        print(f"Total Predictions (better absolute error):")
        print(f"  🤖 Model wins: {total_wins_model} ({total_wins_model/len(comparison_df)*100:.1f}%)")
        print(f"  📈 BM wins: {total_wins_bm} ({total_wins_bm/len(comparison_df)*100:.1f}%)")
        print(f"  🤝 Ties: {total_ties} ({total_ties/len(comparison_df)*100:.1f}%)")
        md_lines.append(f"**Spread Predictions (better absolute error):**  ")
        md_lines.append(f"- 🤖 Model wins: `{spread_wins_model}` ({spread_wins_model/len(comparison_df)*100:.1f}%)  ")
        md_lines.append(f"- 📈 BM wins: `{spread_wins_bm}` ({spread_wins_bm/len(comparison_df)*100:.1f}%)  ")
        md_lines.append(f"- 🤝 Ties: `{spread_ties}` ({spread_ties/len(comparison_df)*100:.1f}%)  ")
        md_lines.append(f"**Total Predictions (better absolute error):**  ")
        md_lines.append(f"- 🤖 Model wins: `{total_wins_model}` ({total_wins_model/len(comparison_df)*100:.1f}%)  ")
        md_lines.append(f"- 📈 BM wins: `{total_wins_bm}` ({total_wins_bm/len(comparison_df)*100:.1f}%)  ")
        md_lines.append(f"- 🤝 Ties: `{total_ties}` ({total_ties/len(comparison_df)*100:.1f}%)  \n")
    
    # Save detailed comparison
    output_path = 'checkpoints/prediction_comparison.csv'
    comparison_df.to_csv(output_path, index=False)
    print(f"\n💾 Detailed comparison saved to: {output_path}")
    md_lines.append(f"\n---\n\n**Detailed comparison saved to:** `{output_path}`\n")
    # Save summary metrics
    summary_data = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'validation_samples': len(comparison_df),
        'model_metrics': {
            'spread': model_spread_metrics,
            'total': model_total_metrics
        }
    }
    if len(bm_val) > 0:
        summary_data['betting_market_metrics'] = {
            'spread': bm_spread_metrics,
            'total': bm_total_metrics
        }
        summary_data['head_to_head'] = {
            'spread_model_wins': int(spread_wins_model),
            'spread_bm_wins': int(spread_wins_bm),
            'total_model_wins': int(total_wins_model),
            'total_bm_wins': int(total_wins_bm)
        }
    import json
    with open('checkpoints/comparison_summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    print(f"📊 Summary metrics saved to: checkpoints/comparison_summary.json")
    md_lines.append(f"**Summary metrics saved to:** `checkpoints/comparison_summary.json`\n")
    # Create a top/bottom performers analysis
    print("\n🎯 TOP/BOTTOM PERFORMERS:")
    print("-" * 40)
    md_lines.append("\n## Top/Bottom Performers\n")
    # Best model predictions
    best_spread_idx = comparison_df['model_spread_abs_error'].idxmin()
    worst_spread_idx = comparison_df['model_spread_abs_error'].idxmax()
    best_total_idx = comparison_df['model_total_abs_error'].idxmin()
    worst_total_idx = comparison_df['model_total_abs_error'].idxmax()
    print(f"Best Model Spread Prediction:")
    print(f"  Game: {comparison_df.loc[best_spread_idx, 'filename']}")
    print(f"  Predicted: {comparison_df.loc[best_spread_idx, 'model_pred_spread']:.1f}")
    print(f"  Actual: {comparison_df.loc[best_spread_idx, 'actual_spread']:.1f}")
    print(f"  Error: {comparison_df.loc[best_spread_idx, 'model_spread_abs_error']:.1f}")
    md_lines.append(f"**Best Model Spread Prediction:**  ")
    md_lines.append(f"- Game: `{comparison_df.loc[best_spread_idx, 'filename']}`  ")
    md_lines.append(f"- Predicted: `{comparison_df.loc[best_spread_idx, 'model_pred_spread']:.1f}`  ")
    md_lines.append(f"- Actual: `{comparison_df.loc[best_spread_idx, 'actual_spread']:.1f}`  ")
    md_lines.append(f"- Error: `{comparison_df.loc[best_spread_idx, 'model_spread_abs_error']:.1f}`  \n")
    print(f"Best Model Total Prediction:")
    print(f"  Game: {comparison_df.loc[best_total_idx, 'filename']}")
    print(f"  Predicted: {comparison_df.loc[best_total_idx, 'model_pred_total']:.1f}")
    print(f"  Actual: {comparison_df.loc[best_total_idx, 'actual_total']:.1f}")
    print(f"  Error: {comparison_df.loc[best_total_idx, 'model_total_abs_error']:.1f}")
    md_lines.append(f"**Best Model Total Prediction:**  ")
    md_lines.append(f"- Game: `{comparison_df.loc[best_total_idx, 'filename']}`  ")
    md_lines.append(f"- Predicted: `{comparison_df.loc[best_total_idx, 'model_pred_total']:.1f}`  ")
    md_lines.append(f"- Actual: `{comparison_df.loc[best_total_idx, 'actual_total']:.1f}`  ")
    md_lines.append(f"- Error: `{comparison_df.loc[best_total_idx, 'model_total_abs_error']:.1f}`  \n")
    print(f"\n🎊 COMPARISON COMPLETE!")
    print(f"📁 Results saved in checkpoints/ folder")
    md_lines.append("\n---\n\n🎊 **COMPARISON COMPLETE!**\n")
    # Save markdown report
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    md_report_path = os.path.join('resources', f'prediction_comparison_report_{timestamp}.md')
    
    # Add comprehensive analysis to markdown
    md_lines.append("## Executive Summary\n")
    
    # Determine overall winner
    if len(bm_val) > 0:
        model_overall_acc = (model_spread_metrics['Accuracy'] + model_total_metrics['Accuracy']) / 2
        bm_overall_acc = (bm_spread_metrics['Accuracy'] + bm_total_metrics['Accuracy']) / 2
        
        if model_overall_acc > bm_overall_acc:
            winner = "🤖 Neural Network Model"
            advantage = model_overall_acc - bm_overall_acc
        else:
            winner = "📈 Betting Market"
            advantage = bm_overall_acc - model_overall_acc
        
        md_lines.append(f"**Overall Winner:** {winner} (by {advantage:.1f}% accuracy)  \n")
        
        # Performance breakdown
        md_lines.append("### Performance Breakdown\n")
        md_lines.append("| Metric | Model | Betting Market | Winner |\n")
        md_lines.append("|--------|-------|----------------|--------|\n")
        
        # Spread comparison
        spread_winner = "🤖 Model" if model_spread_metrics['Accuracy'] > bm_spread_metrics['Accuracy'] else "📈 BM"
        md_lines.append(f"| Spread Accuracy | {model_spread_metrics['Accuracy']:.1f}% | {bm_spread_metrics['Accuracy']:.1f}% | {spread_winner} |\n")
        
        # Total comparison
        total_winner = "🤖 Model" if model_total_metrics['Accuracy'] > bm_total_metrics['Accuracy'] else "📈 BM"
        md_lines.append(f"| Total Accuracy | {model_total_metrics['Accuracy']:.1f}% | {bm_total_metrics['Accuracy']:.1f}% | {total_winner} |\n")
        
        # R² comparison
        spread_r2_winner = "🤖 Model" if model_spread_metrics['R²'] > bm_spread_metrics['R²'] else "📈 BM"
        md_lines.append(f"| Spread R² | {model_spread_metrics['R²']:.3f} | {bm_spread_metrics['R²']:.3f} | {spread_r2_winner} |\n")
        
        total_r2_winner = "🤖 Model" if model_total_metrics['R²'] > bm_total_metrics['R²'] else "📈 BM"
        md_lines.append(f"| Total R² | {model_total_metrics['R²']:.3f} | {bm_total_metrics['R²']:.3f} | {total_r2_winner} |\n")
        
        # Overall win rate
        total_games = len(comparison_df)
        model_wins = spread_wins_model + total_wins_model
        bm_wins = spread_wins_bm + total_wins_bm
        model_win_rate = (model_wins / (total_games * 2)) * 100
        bm_win_rate = (bm_wins / (total_games * 2)) * 100
        
        md_lines.append(f"| Overall Win Rate | {model_win_rate:.1f}% | {bm_win_rate:.1f}% | {'🤖 Model' if model_win_rate > bm_win_rate else '📈 BM'} |\n")
        
        # Key insights
        md_lines.append("\n### Key Insights\n")
        if model_overall_acc > 60:
            md_lines.append("- ✅ **Model Performance**: Excellent accuracy, ready for deployment\n")
        elif model_overall_acc > 50:
            md_lines.append("- ⚠️ **Model Performance**: Good accuracy, could use improvement\n")
        else:
            md_lines.append("- ❌ **Model Performance**: Poor accuracy, needs significant improvement\n")
        
        if model_overall_acc > bm_overall_acc:
            md_lines.append(f"- 🎯 **Competitive Advantage**: Model outperforms bookmakers by {advantage:.1f}%\n")
        else:
            md_lines.append(f"- 📈 **Improvement Needed**: Model underperforms bookmakers by {advantage:.1f}%\n")
        
        # Betting recommendation
        md_lines.append("\n### Betting Recommendation\n")
        if model_overall_acc > 55 and model_overall_acc > bm_overall_acc:
            md_lines.append("- 🚀 **RECOMMENDED**: Model shows edge over bookmakers, suitable for betting\n")
        elif model_overall_acc > 50:
            md_lines.append("- ⚠️ **CONDITIONAL**: Model performs adequately, use with caution\n")
        else:
            md_lines.append("- ❌ **NOT RECOMMENDED**: Model performance insufficient for betting\n")
        
    else:
        md_lines.append("**Status:** Model evaluation only (no betting market comparison available)\n")
        md_lines.append(f"**Model Overall Accuracy:** {(model_spread_metrics['Accuracy'] + model_total_metrics['Accuracy']) / 2:.1f}%\n")
    
    with open(md_report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    print(f"📄 Comprehensive markdown report saved to: {md_report_path}")

if __name__ == "__main__":
    main()
