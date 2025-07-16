#!/usr/bin/env python3
"""
Enhanced prediction script with thorough betting analysis and confidence ratings
"""

from predict import GamePredictor
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import os
import sys

def calculate_confidence_metrics(pred_spread, pred_total, bm_spread, bm_total):
    """Calculate confidence metrics based on pre    # Save comprehensive CSV
    if csv_data:
        df_csv = pd.DataFrame(csv_data)
        csv_filename = f"betting_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_csv.to_csv(csv_filename, index=False)
        print(f"\n💾 Comprehensive betting analysis saved to: {csv_filename}")
    
    # Save enhanced results to CSV (original format)
    df = pd.DataFrame(enhanced_predictions)
    df.to_csv("enhanced_predictions.csv", index=False)
    print(f"💾 Enhanced predictions saved to: enhanced_predictions.csv")
    
    # Log summary
    print(f"\n📄 Detailed betting logs saved to: {log_dir}")
    print(f"🎯 Check individual game logs and daily summary for detailed analysis!")
    
    print(f"\n\n{'='*120}")
    print(f"🎯 ANALYSIS COMPLETE - {len(successful_preds)} games processed")
    print(f"{'='*120}\n\n")ferences from bookmaker lines"""
    
    # Model performance metrics (from training_summary.json)
    spread_rmse = 8.39  # From validation metrics
    total_rmse = 9.91   # From validation metrics
    spread_mae = 6.69   # From validation metrics
    total_mae = 7.71    # From validation metrics
    
    confidence_metrics = {}
    
    # Spread confidence calculation
    if bm_spread is not None and isinstance(bm_spread, (int, float)):
        # Both model and bookmaker spreads should be for home team
        # Model: positive = home wins, negative = away wins
        # Bookmaker: negative = home favored, positive = away favored (opposite convention)
        # Convert bookmaker line to model convention for comparison
        bm_spread_model_convention = -bm_spread
        
        spread_diff = abs(pred_spread - bm_spread_model_convention)
        # Higher difference = higher confidence (more edge)
        spread_confidence = min(spread_diff / spread_rmse * 10, 10)  # Scale to 0-10
        confidence_metrics['spread_confidence'] = round(spread_confidence, 2)
        confidence_metrics['spread_edge'] = round(spread_diff, 2)
        
        # Determine confidence level
        if spread_confidence >= 7:
            confidence_metrics['spread_confidence_level'] = 'HIGH'
        elif spread_confidence >= 4:
            confidence_metrics['spread_confidence_level'] = 'MEDIUM'
        else:
            confidence_metrics['spread_confidence_level'] = 'LOW'
    else:
        confidence_metrics['spread_confidence'] = 0
        confidence_metrics['spread_edge'] = 0
        confidence_metrics['spread_confidence_level'] = 'N/A'
    
    # Total confidence calculation
    if bm_total is not None and isinstance(bm_total, (int, float)):
        total_diff = abs(pred_total - bm_total)
        # Higher difference = higher confidence (more edge)
        total_confidence = min(total_diff / total_rmse * 10, 10)  # Scale to 0-10
        confidence_metrics['total_confidence'] = round(total_confidence, 2)
        confidence_metrics['total_edge'] = round(total_diff, 2)
        
        # Determine confidence level
        if total_confidence >= 7:
            confidence_metrics['total_confidence_level'] = 'HIGH'
        elif total_confidence >= 4:
            confidence_metrics['total_confidence_level'] = 'MEDIUM'
        else:
            confidence_metrics['total_confidence_level'] = 'LOW'
    else:
        confidence_metrics['total_confidence'] = 0
        confidence_metrics['total_edge'] = 0
        confidence_metrics['total_confidence_level'] = 'N/A'
    
    # Overall confidence (weighted average)
    if confidence_metrics['spread_confidence'] > 0 and confidence_metrics['total_confidence'] > 0:
        overall_confidence = (confidence_metrics['spread_confidence'] + confidence_metrics['total_confidence']) / 2
    elif confidence_metrics['spread_confidence'] > 0:
        overall_confidence = confidence_metrics['spread_confidence']
    elif confidence_metrics['total_confidence'] > 0:
        overall_confidence = confidence_metrics['total_confidence']
    else:
        overall_confidence = 0
    
    confidence_metrics['overall_confidence'] = round(overall_confidence, 2)
    
    return confidence_metrics

def get_betting_recommendation(spread_confidence, total_confidence, spread_edge, total_edge, 
                             spread_conf_level, total_conf_level):
    """Generate betting recommendations based on confidence levels"""
    recommendations = []
    
    # Spread recommendations
    if spread_conf_level == 'HIGH' and spread_edge >= 5:
        recommendations.append("🔥 STRONG SPREAD BET")
    elif spread_conf_level == 'MEDIUM' and spread_edge >= 3:
        recommendations.append("⚡ MODERATE SPREAD BET")
    elif spread_conf_level == 'LOW':
        recommendations.append("❄️ WEAK SPREAD EDGE")
    
    # Total recommendations
    if total_conf_level == 'HIGH' and total_edge >= 5:
        recommendations.append("🔥 STRONG TOTAL BET")
    elif total_conf_level == 'MEDIUM' and total_edge >= 3:
        recommendations.append("⚡ MODERATE TOTAL BET")
    elif total_conf_level == 'LOW':
        recommendations.append("❄️ WEAK TOTAL EDGE")
    
    if not recommendations:
        recommendations.append("🚫 NO STRONG BETS")
    
    return " | ".join(recommendations)

def log_bet_analysis(bet_data, log_dir):
    """Log detailed betting analysis for each bet"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create individual bet log file
    game_name = bet_data['file'].replace('.csv', '').replace(' ', '_')
    bet_log_file = log_dir / f"{game_name}_{timestamp}.txt"
    
    # Create summary log entry
    summary_log_file = log_dir / f"betting_summary_{datetime.now().strftime('%Y%m%d')}.txt"
    
    # Convert model spread to betting convention for logging
    model_spread_betting = -bet_data['final_spread']
    
    # Detailed bet analysis content
    bet_analysis = []
    bet_analysis.append("="*80)
    bet_analysis.append(f"BETTING ANALYSIS - {game_name}")
    bet_analysis.append("="*80)
    bet_analysis.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    bet_analysis.append(f"Game File: {bet_data['file']}")
    bet_analysis.append("")
    
    # Game predictions
    bet_analysis.append("📊 GAME PREDICTIONS")
    bet_analysis.append("-"*40)
    bet_analysis.append(f"Model Spread: {model_spread_betting:+.2f} (betting convention)")
    bet_analysis.append(f"Model Total: {bet_data['final_total']:.2f}")
    bet_analysis.append(f"Bookmaker Spread: {bet_data.get('live_bm_spread_line', 'N/A')}")
    bet_analysis.append(f"Bookmaker Total: {bet_data.get('live_bm_total_line', 'N/A')}")
    bet_analysis.append("")
    
    # Confidence metrics
    bet_analysis.append("🎯 CONFIDENCE METRICS")
    bet_analysis.append("-"*40)
    bet_analysis.append(f"Spread Confidence: {bet_data['spread_confidence']:.1f}/10 ({bet_data['spread_confidence_level']})")
    bet_analysis.append(f"Total Confidence: {bet_data['total_confidence']:.1f}/10 ({bet_data['total_confidence_level']})")
    bet_analysis.append(f"Spread Edge: {bet_data['spread_edge']:.1f} points")
    bet_analysis.append(f"Total Edge: {bet_data['total_edge']:.1f} points")
    bet_analysis.append(f"Overall Confidence: {bet_data['overall_confidence']:.1f}/10")
    bet_analysis.append("")
    
    # Betting recommendations
    bet_analysis.append("💡 BETTING RECOMMENDATIONS")
    bet_analysis.append("-"*40)
    bet_analysis.append(f"Spread Position: {bet_data['spread_bet']}")
    bet_analysis.append(f"Total Position: {bet_data['total_bet']} TOTAL")
    bet_analysis.append(f"Recommendation: {bet_data['betting_recommendation']}")
    bet_analysis.append("")
    
    # Risk assessment
    bet_analysis.append("⚠️ RISK ASSESSMENT")
    bet_analysis.append("-"*40)
    
    # Determine stake size
    if bet_data['overall_confidence'] >= 8:
        stake_size = "HIGH (5-10% of bankroll)"
        risk_level = "MODERATE"
    elif bet_data['overall_confidence'] >= 6:
        stake_size = "MEDIUM (3-5% of bankroll)"
        risk_level = "MODERATE"
    elif bet_data['overall_confidence'] >= 4:
        stake_size = "LOW (1-3% of bankroll)"
        risk_level = "HIGH"
    else:
        stake_size = "MINIMAL (0.5-1% of bankroll)"
        risk_level = "VERY HIGH"
    
    bet_analysis.append(f"Recommended Stake: {stake_size}")
    bet_analysis.append(f"Risk Level: {risk_level}")
    bet_analysis.append("")
    
    # Expected value calculation
    bet_analysis.append("💰 EXPECTED VALUE ANALYSIS")
    bet_analysis.append("-"*40)
    
    # Simple EV calculation based on confidence and edge
    spread_ev = (bet_data['spread_confidence'] / 10) * bet_data['spread_edge'] * 0.95 - (1 - bet_data['spread_confidence'] / 10) * 100
    total_ev = (bet_data['total_confidence'] / 10) * bet_data['total_edge'] * 0.95 - (1 - bet_data['total_confidence'] / 10) * 100
    
    bet_analysis.append(f"Spread Expected Value: {spread_ev:.2f} units per 100 units wagered")
    bet_analysis.append(f"Total Expected Value: {total_ev:.2f} units per 100 units wagered")
    bet_analysis.append("")
    
    # Historical context
    bet_analysis.append("📈 HISTORICAL CONTEXT")
    bet_analysis.append("-"*40)
    bet_analysis.append("Model Performance:")
    bet_analysis.append("- Spread Directional Accuracy: 76.7%")
    bet_analysis.append("- Total Directional Accuracy: 100%")
    bet_analysis.append("- Overall Betting Success Rate: 69%")
    bet_analysis.append("- Historical Advantage over Bookmaker: 6%")
    bet_analysis.append("")
    
    # Action items
    bet_analysis.append("✅ ACTION ITEMS")
    bet_analysis.append("-"*40)
    
    if bet_data['overall_confidence'] >= 7:
        bet_analysis.append("🔥 STRONG BET - Consider immediate action")
        bet_analysis.append("- Monitor line movement")
        bet_analysis.append("- Place bet with recommended stake size")
        bet_analysis.append("- Set up alerts for significant changes")
    elif bet_data['overall_confidence'] >= 5:
        bet_analysis.append("⚡ MODERATE BET - Consider with caution")
        bet_analysis.append("- Wait for better line movement if possible")
        bet_analysis.append("- Use lower stake size")
        bet_analysis.append("- Monitor for additional confirmation")
    else:
        bet_analysis.append("❄️ WEAK BET - Avoid or minimal stake")
        bet_analysis.append("- Skip this bet unless part of system play")
        bet_analysis.append("- Use minimal stake if betting")
        bet_analysis.append("- Focus on higher confidence opportunities")
    
    bet_analysis.append("")
    bet_analysis.append("="*80)
    
    # Write individual bet log
    with open(bet_log_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(bet_analysis))
    
    # Append to summary log
    summary_entry = (
        f"{datetime.now().strftime('%H:%M:%S')} | {game_name[:20]:20} | "
        f"Confidence: {bet_data['overall_confidence']:4.1f}/10 | "
        f"Spread: {bet_data['spread_confidence']:4.1f}/10 | "
        f"Total: {bet_data['total_confidence']:4.1f}/10 | "
        f"Recommendation: {bet_data['betting_recommendation']}\n"
    )
    
    with open(summary_log_file, 'a', encoding='utf-8') as f:
        # Write header if file is new
        if os.path.getsize(summary_log_file) == len(summary_entry):
            f.write(f"BETTING ANALYSIS SUMMARY - {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write("="*120 + "\n")
            f.write(f"{'Time':8} | {'Game':20} | {'Overall':15} | {'Spread':12} | {'Total':11} | {'Recommendation'}\n")
            f.write("-"*120 + "\n")
        f.write(summary_entry)

def quick_predict_example(single_file=None):
    """Enhanced prediction with thorough betting analysis"""
    
    # Initialize the predictor
    predictor = GamePredictor("models/main/best_model.pth")
    
    # Handle single file or batch processing
    if single_file:
        sample_files = [Path(single_file)]
    else:
        # Find some sample data files
        data_dir = Path(r"C:\Users\admin\Desktop\NEWNEW\model_ready")
        sample_files = list(data_dir.glob("*.csv"))[:10]  # Get first 10 files for analysis
    
    if not sample_files:
        print("❌ No CSV files found in data directory")
        return
    
    print(f"🎯 Making predictions for {len(sample_files)} games...")
    
    # Make predictions
    predictions = predictor.predict_batch(sample_files)
    
    # Enhanced analysis with confidence metrics
    enhanced_predictions = []
    log_dir = Path(r"C:\Users\admin\Desktop\NEWNEW\bet_analysis_log")
    
    for pred in predictions:
        if 'error' not in pred:
            # Get bookmaker lines
            bm_total = pred.get('live_bm_total_line', None)
            bm_spread = pred.get('live_bm_spread_line', None)
            
            # Calculate confidence metrics
            confidence_metrics = calculate_confidence_metrics(
                pred['final_spread'], pred['final_total'], bm_spread, bm_total
            )
            
            # Determine betting positions
            if bm_total is not None and isinstance(bm_total, (int, float)):
                ou_bet = 'Over' if pred['final_total'] > bm_total else 'Under'
            else:
                ou_bet = 'N/A'
            
            # Fixed spread betting logic with correct sign interpretation
            if bm_spread is not None and isinstance(bm_spread, (int, float)):
                # Model predicts: positive = home wins by X, negative = away wins by X
                # Bookmaker: negative = home favored by X, positive = away favored by X
                # Convert bookmaker to model convention for comparison
                bm_spread_model_convention = -bm_spread
                
                # Compare model prediction vs bookmaker expectation (both in model convention)
                if pred['final_spread'] < bm_spread_model_convention:
                    # Model thinks home team performs worse than bookmaker expects
                    if bm_spread < 0:  # Home team favored
                        spread_bet = f'Away Team {abs(bm_spread):+.1f} (Home team wins by less than expected)'
                    else:  # Away team favored
                        spread_bet = f'Home Team {abs(bm_spread):+.1f} (Away team wins by less than expected)'
                else:
                    # Model thinks home team performs better than bookmaker expects
                    if bm_spread < 0:  # Home team favored
                        spread_bet = f'Home Team {abs(bm_spread):+.1f} (Home team wins by more than expected)'
                    else:  # Away team favored
                        spread_bet = f'Away Team {abs(bm_spread):+.1f} (Home team wins when expected to lose)'
            else:
                spread_bet = 'N/A'
            
            # Get betting recommendation
            recommendation = get_betting_recommendation(
                confidence_metrics['spread_confidence'],
                confidence_metrics['total_confidence'],
                confidence_metrics['spread_edge'],
                confidence_metrics['total_edge'],
                confidence_metrics['spread_confidence_level'],
                confidence_metrics['total_confidence_level']
            )
            
            enhanced_pred = {
                **pred,
                **confidence_metrics,
                'spread_bet': spread_bet,
                'total_bet': ou_bet,
                'betting_recommendation': recommendation
            }
            enhanced_predictions.append(enhanced_pred)
            
            # Log detailed analysis for this bet
            log_bet_analysis(enhanced_pred, log_dir)
            
        else:
            enhanced_predictions.append(pred)
    
    # Summary statistics (moved before detailed results)
    successful_preds = [p for p in enhanced_predictions if 'error' not in p]
    if successful_preds:
        avg_spread_conf = np.mean([p['spread_confidence'] for p in successful_preds])
        avg_total_conf = np.mean([p['total_confidence'] for p in successful_preds])
        high_conf_spread = len([p for p in successful_preds if p['spread_confidence_level'] == 'HIGH'])
        high_conf_total = len([p for p in successful_preds if p['total_confidence_level'] == 'HIGH'])
        
        print(f"\n\n{'='*80}")
        print(f"📈 CONFIDENCE SUMMARY")
        print(f"{'='*80}")
        print(f"🎯 Average Spread Confidence: {avg_spread_conf:.2f}/10")
        print(f"🎯 Average Total Confidence:  {avg_total_conf:.2f}/10")
        print(f"🔥 High Confidence Spread Bets: {high_conf_spread}/{len(successful_preds)}")
        print(f"🔥 High Confidence Total Bets:  {high_conf_total}/{len(successful_preds)}")
        print(f"📊 Total Games Analyzed: {len(successful_preds)}")
        print(f"{'='*80}")

    # Display detailed results
    print(f"\n\n{'='*120}")
    print(f"📊 DETAILED BETTING ANALYSIS")
    print(f"{'='*120}")
    
    for pred in enhanced_predictions:
        if 'error' not in pred:
            # Convert model spread to betting convention for display
            model_spread_betting = -pred['final_spread']  # Convert to betting convention
            
            print(f"\n🎲 Game: {pred['file']}")
            print(f"   📈 Model Spread: {model_spread_betting:>+6.2f} | BM Line: {pred.get('live_bm_spread_line', 'N/A'):>6}")
            print(f"   📊 Total:        {pred['final_total']:>6.2f} | BM Line: {pred.get('live_bm_total_line', 'N/A'):>6}")
            print(f"   🎯 Spread Confidence: {pred['spread_confidence']:>4.1f}/10 ({pred['spread_confidence_level']}) | Edge: {pred['spread_edge']:>4.1f}")
            print(f"   🎯 Total Confidence:  {pred['total_confidence']:>4.1f}/10 ({pred['total_confidence_level']}) | Edge: {pred['total_edge']:>4.1f}")
            print(f"   💡 Betting Position: {pred['spread_bet']}")
            print(f"   💡 Total Position: {pred['total_bet']} TOTAL")
            print(f"   🔥 Recommendation: {pred['betting_recommendation']}")
        else:
            print(f"\n❌ {pred['file']}: ERROR - {pred.get('error', 'Unknown error')}")
    
    # Create comprehensive CSV output
    csv_data = []
    for pred in enhanced_predictions:
        if 'error' not in pred:
            # Convert model spread to betting convention for CSV
            model_spread_betting = -pred['final_spread']
            
            csv_row = {
                'game': pred['file'],
                'model_spread': model_spread_betting,
                'bm_spread_line': pred.get('live_bm_spread_line', None),
                'model_total': pred['final_total'],
                'bm_total_line': pred.get('live_bm_total_line', None),
                'spread_confidence': pred['spread_confidence'],
                'spread_confidence_level': pred['spread_confidence_level'],
                'spread_edge': pred['spread_edge'],
                'total_confidence': pred['total_confidence'],
                'total_confidence_level': pred['total_confidence_level'],
                'total_edge': pred['total_edge'],
                'overall_confidence': pred['overall_confidence'],
                'spread_bet': pred['spread_bet'],
                'total_bet': pred['total_bet'],
                'betting_recommendation': pred['betting_recommendation'],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            csv_data.append(csv_row)
    
    # Save comprehensive CSV
    if csv_data:
        df_csv = pd.DataFrame(csv_data)
        csv_filename = f"betting_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_csv.to_csv(csv_filename, index=False)
        print(f"\n� Comprehensive betting analysis saved to: {csv_filename}")
    
    # Save enhanced results to CSV (original format)
    df = pd.DataFrame(enhanced_predictions)
    df.to_csv("enhanced_predictions.csv", index=False)
    print(f"💾 Enhanced predictions saved to: enhanced_predictions.csv")

def generate_betting_report(enhanced_predictions, successful_preds):
    """Generate a comprehensive betting report"""
    
    report = []
    report.append("🎯 COMPREHENSIVE BETTING ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"📅 Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Executive Summary
    report.append("📊 EXECUTIVE SUMMARY")
    report.append("-" * 40)
    total_games = len(successful_preds)
    high_conf_spread = len([p for p in successful_preds if p['spread_confidence_level'] == 'HIGH'])
    high_conf_total = len([p for p in successful_preds if p['total_confidence_level'] == 'HIGH'])
    strong_bets = len([p for p in successful_preds if 'STRONG' in p['betting_recommendation']])
    
    report.append(f"🎲 Total Games Analyzed: {total_games}")
    report.append(f"🔥 High Confidence Spread Opportunities: {high_conf_spread} ({high_conf_spread/total_games*100:.1f}%)")
    report.append(f"🔥 High Confidence Total Opportunities: {high_conf_total} ({high_conf_total/total_games*100:.1f}%)")
    report.append(f"💪 Strong Betting Opportunities: {strong_bets} ({strong_bets/total_games*100:.1f}%)")
    report.append("")
    
    # Top Opportunities
    report.append("⭐ TOP BETTING OPPORTUNITIES")
    report.append("-" * 40)
    
    # Sort by overall confidence
    top_bets = sorted(successful_preds, key=lambda x: x['overall_confidence'], reverse=True)[:5]
    
    for i, bet in enumerate(top_bets, 1):
        # Convert model spread to betting convention
        model_spread_betting = -bet['final_spread']
        
        report.append(f"{i}. {bet['file']}")
        report.append(f"   Overall Confidence: {bet['overall_confidence']:.1f}/10")
        report.append(f"   Spread: {model_spread_betting:+.2f} (vs BM: {bet.get('live_bm_spread_line', 'N/A')}) | Confidence: {bet['spread_confidence']:.1f}/10")
        report.append(f"   Total: {bet['final_total']:.2f} (vs BM: {bet.get('live_bm_total_line', 'N/A')}) | Confidence: {bet['total_confidence']:.1f}/10")
        report.append(f"   Recommendation: {bet['betting_recommendation']}")
        report.append("")
    
    # Risk Analysis
    report.append("⚠️ RISK ANALYSIS")
    report.append("-" * 40)
    
    spread_edges = [p['spread_edge'] for p in successful_preds]
    total_edges = [p['total_edge'] for p in successful_preds]
    
    report.append(f"📈 Average Spread Edge: {np.mean(spread_edges):.2f}")
    report.append(f"📊 Average Total Edge: {np.mean(total_edges):.2f}")
    report.append(f"🎯 Edge Consistency: {np.std(spread_edges + total_edges):.2f} (lower is better)")
    report.append("")
    
    # Betting Strategy
    report.append("💡 RECOMMENDED BETTING STRATEGY")
    report.append("-" * 40)
    
    avg_edge = np.mean(spread_edges + total_edges)
    if avg_edge > 8:
        strategy = "AGGRESSIVE"
        report.append("🔥 AGGRESSIVE APPROACH RECOMMENDED")
        report.append("   - High confidence bets: 5-10% of bankroll")
        report.append("   - Medium confidence bets: 3-5% of bankroll")
        report.append("   - Strong edge detection suggests higher stakes")
    elif avg_edge > 4:
        strategy = "CONSERVATIVE"
        report.append("⚡ CONSERVATIVE APPROACH RECOMMENDED")
        report.append("   - High confidence bets: 3-5% of bankroll")
        report.append("   - Medium confidence bets: 2-3% of bankroll")
        report.append("   - Moderate edges suggest standard stakes")
    else:
        strategy = "CAUTIOUS"
        report.append("❄️ CAUTIOUS APPROACH RECOMMENDED")
        report.append("   - High confidence bets: 2-3% of bankroll")
        report.append("   - Medium confidence bets: 1-2% of bankroll")
        report.append("   - Lower edges suggest minimal stakes")
    
    report.append("")
    
    # Model Performance Context
    report.append("🤖 MODEL PERFORMANCE CONTEXT")
    report.append("-" * 40)
    report.append("📊 Historical Performance:")
    report.append("   - Spread Directional Accuracy: 76.7%")
    report.append("   - Total Directional Accuracy: 100%")
    report.append("   - Overall Betting Success Rate: 69%")
    report.append("   - Advantage over Bookmaker: 6%")
    report.append("")
    report.append("⚡ Strengths:")
    report.append("   - Excellent total predictions (100% directional)")
    report.append("   - Strong edge detection vs bookmaker lines")
    report.append("   - Consistent outperformance in live betting scenarios")
    report.append("")
    report.append("⚠️ Limitations:")
    report.append("   - Spread predictions less accurate than totals")
    report.append("   - Model performance depends on data quality")
    report.append("   - Past performance doesn't guarantee future results")
    
    return "\n".join(report)

if __name__ == "__main__":
    # Check if a specific file was provided as command line argument
    if len(sys.argv) > 1:
        single_file = sys.argv[1]
        if os.path.exists(single_file):
            quick_predict_example(single_file)
        else:
            print(f"❌ File not found: {single_file}")
    else:
        quick_predict_example()
