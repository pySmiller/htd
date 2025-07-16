#!/usr/bin/env python3
"""
Single Game Prediction Script

This script loads a trained model and makes predictions for new game data.
Uses the model from models/main/best_model.pth
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import argparse
import warnings
warnings.filterwarnings('ignore')

class MLP(nn.Module):
    """Multi-layer Perceptron model - matches the training architecture"""
    def __init__(self, in_dim, hidden_dims, out_dim, dropout):
        super().__init__()
        layers = []
        current_dim = in_dim
        
        for hidden_dim in hidden_dims:
            layers += [
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.net(x)

class GamePredictor:
    """Game prediction class that handles model loading and prediction"""
    
    def __init__(self, model_path="models/main/best_model.pth"):
        self.model_path = Path(model_path)
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_scaler = None
        self.target_scaler = None
        self.input_size = None
        self.max_rows = None
        
        print(f"🚀 Initializing GamePredictor")
        print(f"📱 Device: {self.device}")
        print(f"📁 Model path: {self.model_path}")
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model and scalers"""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"📥 Loading model from {self.model_path}")
        
        # Load checkpoint with weights_only=False to allow sklearn objects
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Extract model configuration
        config = checkpoint['config']
        self.input_size = checkpoint['in_dim']
        output_size = checkpoint['out_dim']
        
        # Create model with same architecture
        hidden_dims = config['model']['hidden_dims']
        dropout = config['model']['dropout']
        
        self.model = MLP(
            in_dim=self.input_size,
            hidden_dims=hidden_dims,
            out_dim=output_size,
            dropout=dropout
        ).to(self.device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model'])
        self.model.eval()
        
        # Load scalers
        self.feature_scaler = checkpoint['feature_scaler']
        self.target_scaler = checkpoint['target_scaler']
        
        # Calculate max_rows from input_size and number of feature columns
        # This is a reasonable assumption based on the training data structure
        self.max_rows = self.input_size // 150  # Approximate columns per row
        
        print(f"✅ Model loaded successfully")
        print(f"📊 Input size: {self.input_size}")
        print(f"📊 Output size: {output_size}")
        print(f"📊 Hidden layers: {hidden_dims}")
        print(f"📊 Dropout: {dropout}")
        print(f"📊 Max rows per game: {self.max_rows}")
    
    def preprocess_game_data(self, csv_path):
        """Preprocess a single game CSV file for prediction"""
        print(f"📄 Processing game data: {csv_path}")
        
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Select numeric columns only
        numeric_data = df.select_dtypes(include=[np.number])
        
        # Fill NaN values with 0
        numeric_data = numeric_data.fillna(0)
        
        # Adjust rows to match training data format
        if len(numeric_data) > self.max_rows:
            # Truncate if too many rows
            numeric_data = numeric_data.iloc[:self.max_rows]
            print(f"⚠️  Truncated to {self.max_rows} rows")
        elif len(numeric_data) < self.max_rows:
            # Pad with zeros if too few rows
            rows_to_add = self.max_rows - len(numeric_data)
            zero_padding = pd.DataFrame(
                np.zeros((rows_to_add, len(numeric_data.columns))),
                columns=numeric_data.columns
            )
            numeric_data = pd.concat([numeric_data, zero_padding], ignore_index=True)
            print(f"📝 Padded to {self.max_rows} rows")
        
        # Flatten the data to match training format
        flattened_data = numeric_data.values.flatten()
        
        # Ensure correct dimensionality
        if len(flattened_data) != self.input_size:
            print(f"⚠️  Warning: Input size mismatch")
            print(f"    Expected: {self.input_size}")
            print(f"    Got: {len(flattened_data)}")
            
            # Truncate or pad as needed
            if len(flattened_data) > self.input_size:
                flattened_data = flattened_data[:self.input_size]
            else:
                padding = np.zeros(self.input_size - len(flattened_data))
                flattened_data = np.concatenate([flattened_data, padding])
        
        # Normalize using the same scaler from training
        normalized_data = self.feature_scaler.transform(flattened_data.reshape(1, -1))
        
        return normalized_data
    
    def predict_game(self, csv_path):
        """Make prediction for a single game"""
        # Preprocess the data
        processed_data = self.preprocess_game_data(csv_path)
        
        # Convert to tensor
        input_tensor = torch.from_numpy(processed_data.astype(np.float32)).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            normalized_prediction = self.model(input_tensor)
        
        # Denormalize the prediction
        prediction = self.target_scaler.inverse_transform(
            normalized_prediction.cpu().numpy()
        )
        
        return prediction[0]  # Return first (and only) prediction
    
    def predict_batch(self, csv_files):
        """Make predictions for multiple games, including bookmaker lines if available"""
        predictions = []
        for csv_file in csv_files:
            try:
                pred = self.predict_game(csv_file)
                # Try to extract bookmaker lines from the last row of the CSV
                bm_total = None
                bm_spread = None
                try:
                    df = pd.read_csv(csv_file)
                    if len(df) > 0:
                        last_row = df.iloc[-1]
                        if 'live_bm_total_line' in last_row:
                            bm_total = last_row['live_bm_total_line']
                        if 'live_bm_spread_line' in last_row:
                            bm_spread = last_row['live_bm_spread_line']
                        # Convert to float if possible
                        if pd.notnull(bm_total):
                            bm_total = float(bm_total)
                        else:
                            bm_total = None
                        if pd.notnull(bm_spread):
                            bm_spread = float(bm_spread)
                        else:
                            bm_spread = None
                except Exception as e:
                    print(f"⚠️  Could not extract bookmaker lines from {csv_file}: {e}")
                predictions.append({
                    'file': Path(csv_file).name,
                    'final_spread': pred[0],
                    'final_total': pred[1],
                    'live_bm_total_line': bm_total,
                    'live_bm_spread_line': bm_spread
                })
                print(f"✅ {Path(csv_file).name}: Spread={pred[0]:.2f}, Total={pred[1]:.2f}")
            except Exception as e:
                print(f"❌ Error processing {csv_file}: {e}")
                predictions.append({
                    'file': Path(csv_file).name,
                    'final_spread': None,
                    'final_total': None,
                    'live_bm_total_line': None,
                    'live_bm_spread_line': None,
                    'error': str(e)
                })
        return predictions

def main():
    parser = argparse.ArgumentParser(description='Make predictions using trained model')
    parser.add_argument('--model', default='models/main/best_model.pth', 
                       help='Path to the trained model file')
    parser.add_argument('--input', required=True, 
                       help='Input CSV file or directory containing CSV files')
    parser.add_argument('--output', default='predictions.csv',
                       help='Output CSV file for predictions')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = GamePredictor(args.model)
    
    # Determine input files
    input_path = Path(args.input)
    if input_path.is_file():
        csv_files = [str(input_path)]
    elif input_path.is_dir():
        csv_files = list(input_path.glob('*.csv'))
        csv_files = [str(f) for f in csv_files]
    else:
        raise ValueError(f"Input path does not exist: {input_path}")
    
    print(f"📊 Found {len(csv_files)} CSV files to process")
    
    # Make predictions
    predictions = predictor.predict_batch(csv_files)
    
    # Save results
    df = pd.DataFrame(predictions)
    df.to_csv(args.output, index=False)
    
    print(f"\n📊 PREDICTION SUMMARY")
    print(f"={'='*50}")
    print(f"📄 Processed files: {len(csv_files)}")
    print(f"✅ Successful predictions: {len([p for p in predictions if 'error' not in p])}")
    print(f"❌ Failed predictions: {len([p for p in predictions if 'error' in p])}")
    print(f"💾 Results saved to: {args.output}")
    
    # Show sample predictions
    successful_preds = [p for p in predictions if 'error' not in p]
    if successful_preds:
        print(f"\n📈 SAMPLE PREDICTIONS")
        print(f"{'-'*50}")
        for i, pred in enumerate(successful_preds[:5]):  # Show first 5
            print(f"{pred['file']:<30} | Spread: {pred['final_spread']:>7.2f} | Total: {pred['final_total']:>7.2f}")
        
        if len(successful_preds) > 5:
            print(f"... and {len(successful_preds) - 5} more")

if __name__ == "__main__":
    main()
