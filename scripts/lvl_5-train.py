#!/usr/bin/env python
'''
Train a feed-forward MLP on tabular CSV files with PyTorch & CUDA 12.1.

Usage:
    python mlp_train.py --config training_config.yaml
'''

import argparse, glob, json, random, time
from pathlib import Path
from datetime import datetime

import yaml, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# Enable cuDNN for better GPU performance
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.backends.cudnn.deterministic = False  # Allow non-deterministic algorithms for speed


# ---------- Data utilities ---------------------------------------------------
class CSVDataset(Dataset):
    def __init__(self, csv_files, outcomes_csv_path, scaler=None, max_rows=None):
        self.csv_files = csv_files
        self.max_rows = max_rows
        self.target_scaler = None  # Initialize target scaler
        
        # Load outcomes from CSV instead of JSON
        outcomes_df = pd.read_csv(outcomes_csv_path)
        # Create a dictionary mapping filename to all stats
        self.outcomes = {}
        for _, row in outcomes_df.iterrows():
            self.outcomes[row['filename']] = {
                'final_spread': row['final_spread'],
                'final_total': row['final_total']
            }

        self.features, self.labels, self.scaler = self._build_tensor(scaler)

    def _determine_dimensions(self):
        """Determine the number of columns and maximum rows across all CSV files"""
        # Check all files to find the true maximum - this is important for end-game data
        print(f"Analyzing all {len(self.csv_files)} files to determine true dimensions...")
        
        num_cols = None
        max_rows_found = 0
        
        for i, fpath in enumerate(self.csv_files):
            if i % 1000 == 0:  # Progress indicator
                print(f"  Processed {i}/{len(self.csv_files)} files...")
                
            df = pd.read_csv(fpath)
            # Convert home_team_id to numeric to ensure consistency
            df['home_team_id'] = pd.to_numeric(df['home_team_id'], errors='coerce')
            
            # Keep all numeric columns (including bookmaker predictions at that timestamp)
            numeric_df = df.select_dtypes(include=[np.number])
            
            if num_cols is None:
                num_cols = len(numeric_df.columns)
            elif num_cols != len(numeric_df.columns):
                print(f"Warning: Column count mismatch in {Path(fpath).name}")
            
            max_rows_found = max(max_rows_found, len(numeric_df))
        
        print(f"  Completed analysis. Found max {max_rows_found} rows across all files.")
        return num_cols, max_rows_found

    def _build_tensor(self, scaler):
        # First pass: determine dimensions
        num_cols, max_rows_found = self._determine_dimensions()
        
        if self.max_rows is None:
            self.max_rows = max_rows_found
        
        print(f"Data dimensions: {num_cols} columns, max {self.max_rows} rows per game")
        
        frames, labels = [], []
        for i, fpath in enumerate(self.csv_files):
            df = pd.read_csv(fpath)
            # Convert home_team_id to numeric to ensure consistency
            df['home_team_id'] = pd.to_numeric(df['home_team_id'], errors='coerce')
            
            # Keep all numeric columns (including bookmaker predictions at that timestamp)
            numeric_df = df.select_dtypes(include=[np.number])   # ignore non-numerics
            
            # Check if this file has the expected number of columns
            if len(numeric_df.columns) != num_cols:
                print(f"Warning: {Path(fpath).name} has {len(numeric_df.columns)} columns, expected {num_cols}")
                # Skip files that don't match the expected structure
                continue
            
            # Handle missing values
            numeric_df = numeric_df.fillna(0)
            
            # Pad or truncate to max_rows
            if len(numeric_df) > self.max_rows:
                # Truncate to max_rows
                numeric_df = numeric_df.iloc[:self.max_rows]
            elif len(numeric_df) < self.max_rows:
                # Pad with zeros
                padding_rows = self.max_rows - len(numeric_df)
                padding_data = np.zeros((padding_rows, len(numeric_df.columns)))
                padding_df = pd.DataFrame(
                    padding_data, 
                    columns=numeric_df.columns
                )
                numeric_df = pd.concat([numeric_df, padding_df], ignore_index=True)
            
            # Flatten the padded/truncated CSV into one feature vector
            features = numeric_df.values.flatten()
            frames.append(features)
            
            # Debug: print feature vector length for first few files
            # if i < 5:
            #     print(f"File {Path(fpath).name}: {len(features)} features")

            key = Path(fpath).name
            if key not in self.outcomes:
                print(f'Warning: Outcome for {key} not found in outcomes.csv, skipping')
                continue
            
            # Get the 2 stats for this game
            game_outcomes = self.outcomes[key]
            # Using only the 2 primary targets
            labels.append([
                game_outcomes['final_spread'],
                game_outcomes['final_total']
            ])

        print(f"Successfully processed {len(frames)} files")
        
        X = np.array(frames, dtype=np.float32)
        y = np.array(labels, dtype=np.float32)  # 2 outputs per sample

        # Choose optimal feature normalization for neural networks
        if scaler is None:
            print(f"📊 Analyzing feature distribution for optimal normalization...")
            
            # Test different scalers
            scalers = {
                'MinMax': MinMaxScaler(feature_range=(-1, 1)),  # Better for neural networks
                'Standard': StandardScaler(),  # Traditional z-score
                'Robust': RobustScaler()  # Less sensitive to outliers
            }
            
            # Analyze data characteristics
            feature_std = X.std(axis=0)

            feature_skew = np.abs(np.mean((X - X.mean(axis=0))**3, axis=0) / (feature_std**3 + 1e-8))
            outlier_ratio = np.mean(np.abs(X - np.median(X, axis=0)) > 3 * np.std(X, axis=0))
            
            print(f"  Feature statistics:")
            print(f"    Original range: [{X.min():.3f}, {X.max():.3f}]")
            print(f"    Mean: {X.mean():.3f}, Std: {X.std():.3f}")
            print(f"    Skewness (avg): {feature_skew.mean():.3f}")
            print(f"    Outlier ratio: {outlier_ratio:.3%}")
            
            # Choose best scaler based on data characteristics
            if outlier_ratio > 0.1:  # Many outliers
                scaler = scalers['Robust']
                scaler_name = "Robust (outlier-resistant)"
            elif feature_skew.mean() > 2:  # Highly skewed
                scaler = scalers['Standard']  # Use Standard for skewed data
                scaler_name = "Standard (skew-resistant)"
            else:  # Use Standard for neural networks (better than MinMax for sports data)
                scaler = scalers['Standard']
                scaler_name = "Standard (neural network optimized)"
            
            scaler.fit(X)
            print(f"  Selected scaler: {scaler_name}")
        
        X_norm = scaler.transform(X)
        print(f"  Normalized feature range: [{X_norm.min():.3f}, {X_norm.max():.3f}]")
        print(f"  Normalized feature mean: {X_norm.mean():.3f}, std: {X_norm.std():.3f}")

        # Optimal target normalization for regression
        if not hasattr(self, 'target_scaler') or self.target_scaler is None:
            print(f"📊 Analyzing target distribution for optimal normalization...")
            
            # Analyze target characteristics
            target_range = y.max() - y.min()
            target_std = y.std()
            target_outliers = np.sum(np.abs(y - np.median(y, axis=0)) > 3 * np.std(y, axis=0)) / len(y)
            
            print(f"  Target statistics:")
            print(f"    Original range: [{y.min():.3f}, {y.max():.3f}]")
            print(f"    Mean: {y.mean(axis=0)}")
            print(f"    Std: {y.std(axis=0)}")
            print(f"    Range span: {target_range:.3f}")
            print(f"    Outlier ratio: {target_outliers:.3%}")
            
            # Use StandardScaler for targets in regression (better than MinMax for sports data)
            if target_outliers > 0.15:
                self.target_scaler = RobustScaler()
                target_scaler_name = "Robust (outlier-resistant)"
            else:
                self.target_scaler = StandardScaler()  # Better for sports regression
                target_scaler_name = "Standard (regression optimized)"
            
            self.target_scaler.fit(y)
            print(f"  Selected target scaler: {target_scaler_name}")
            
            if hasattr(self.target_scaler, 'data_min_'):
                print(f"  Target scaler min: {self.target_scaler.data_min_}")
                print(f"  Target scaler max: {self.target_scaler.data_max_}")
            elif hasattr(self.target_scaler, 'center_'):
                print(f"  Target scaler center: {self.target_scaler.center_}")
                print(f"  Target scaler scale: {self.target_scaler.scale_}")
        
        y_norm = self.target_scaler.transform(y)
        print(f"  Normalized target range: [{y_norm.min():.3f}, {y_norm.max():.3f}]")
        print(f"  Normalized target mean: {y_norm.mean(axis=0)}")
        print(f"  Normalized target std: {y_norm.std(axis=0)}")

        return torch.from_numpy(X_norm), torch.from_numpy(y_norm), scaler

    def __len__(self):  return self.labels.size(0)
    def __getitem__(self, idx):  return self.features[idx], self.labels[idx]


# ---------- Model ------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, dropout):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)
        
        # Initialize weights for better cuDNN performance
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # He initialization works better with ReLU activations
            nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):  
        return self.net(x)


# ---------- Training loop ----------------------------------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    
    # For reproducibility with cuDNN (optional - reduces performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def train(cfg_path: str):
    # Record start time
    start_time = time.time()
    
    with open(cfg_path) as f:  cfg = yaml.safe_load(f)

    # Check CUDA availability and force usage if specified in config
    if cfg.get('device', 'cuda') == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available! Install PyTorch with CUDA support or set device to "cpu" in config.')
        device = torch.device('cuda')
        print(f'🚀 Using CUDA device: {torch.cuda.get_device_name(0)}')
        print(f'💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
        print(f'🔧 CUDA Version: {torch.version.cuda}')
        print(f'🔧 PyTorch Version: {torch.__version__}')
        print(f'🔧 CUDA Device Count: {torch.cuda.device_count()}')
        print(f'🔧 Current CUDA Device: {torch.cuda.current_device()}')
        
        # Force CUDA operations to verify it's working
        test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
        print(f'✅ CUDA Test: {test_tensor.device} - {test_tensor.sum().item()}')
        
        # Check cuDNN status
        print(f'🔧 cuDNN enabled: {torch.backends.cudnn.enabled}')
        print(f'🔧 cuDNN benchmark: {torch.backends.cudnn.benchmark}')
        print(f'🔧 cuDNN version: {torch.backends.cudnn.version()}')
        print(f'🔧 cuDNN deterministic: {torch.backends.cudnn.deterministic}')
        
        del test_tensor
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        print('⚠️  Using CPU for training')

    # ----- load data
    data_dir   = Path(cfg['data']['train_dir'])
    all_csv_files = glob.glob(str(data_dir / '*.csv'))
    outcomes_csv_path = cfg['data']['outcomes_csv']
    
    # Load outcomes first to filter files
    outcomes_df = pd.read_csv(outcomes_csv_path)
    available_files = set(outcomes_df['filename'].values)
    
    # Filter CSV files to only include those with outcomes
    csv_files = [f for f in all_csv_files if Path(f).name in available_files]
    print(f'Found {len(csv_files)} CSV files with matching outcomes')
    
    if not csv_files:
        raise ValueError('No CSV files found with matching outcomes!')

    full_ds = CSVDataset(csv_files, outcomes_csv_path)  # Pass the CSV path
    in_dim     = full_ds.features.shape[1]
    out_dim    = 2  # 2 outputs: final_spread, final_total

    X_tr, X_val, y_tr, y_val = train_test_split(
        full_ds.features, full_ds.labels, test_size=0.2,
        random_state=42)  # removed stratify since this is regression

    train_ds   = torch.utils.data.TensorDataset(X_tr, y_tr)
    val_ds     = torch.utils.data.TensorDataset(X_val, y_val)

    # Debug: Print data statistics
    print(f"🔍 DATA DEBUGGING:")
    print(f"  Training features shape: {X_tr.shape}")
    print(f"  Training targets shape: {y_tr.shape}")
    print(f"  Validation features shape: {X_val.shape}")
    print(f"  Validation targets shape: {y_val.shape}")
    print(f"  Training features range: [{X_tr.min():.3f}, {X_tr.max():.3f}]")
    print(f"  Training targets range: [{y_tr.min():.3f}, {y_tr.max():.3f}]")
    print(f"  Validation features range: [{X_val.min():.3f}, {X_val.max():.3f}]")
    print(f"  Validation targets range: [{y_val.min():.3f}, {y_val.max():.3f}]")
    print(f"  Training targets mean: {y_tr.mean(axis=0)}")
    print(f"  Validation targets mean: {y_val.mean(axis=0)}")

    train_loader = DataLoader(train_ds, batch_size=cfg['data']['batch_size'],
                              shuffle=True,  num_workers=cfg['data']['num_workers'],
                              pin_memory=True if device.type == 'cuda' else False,
                              persistent_workers=False,  # Disable persistent workers for CUDA
                              drop_last=True,  # Drop last incomplete batch for consistent cuDNN performance
                              prefetch_factor=2 if cfg['data']['num_workers'] > 0 else None)
    val_loader   = DataLoader(val_ds,   batch_size=cfg['data']['batch_size'],
                              shuffle=False, num_workers=cfg['data']['num_workers'],
                              pin_memory=True if device.type == 'cuda' else False,
                              persistent_workers=False,  # Disable persistent workers for CUDA
                              drop_last=False,  # Keep all validation samples
                              prefetch_factor=2 if cfg['data']['num_workers'] > 0 else None)

    # ----- model / optimiser
    model = MLP(in_dim, cfg['model']['hidden_dims'], out_dim,
                cfg['model']['dropout']).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f'🧠 Model has {total_params:,} parameters')
    
    # Verify model is on correct device
    print(f'🔧 Model device: {next(model.parameters()).device}')
    
    if device.type == 'cuda':
        print(f'📊 Model memory usage: {torch.cuda.memory_allocated() / 1024**2:.1f} MB')
        print(f'📊 GPU memory reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB')
        
        # Test a forward pass to ensure CUDA is working
        dummy_input = torch.randn(1, in_dim).to(device)
        with torch.no_grad():
            dummy_output = model(dummy_input)
        print(f'✅ CUDA forward pass test: input device={dummy_input.device}, output device={dummy_output.device}')
        
        # Force GPU computation to verify CUDA is actually working
        gpu_test = torch.randn(1000, 1000).to(device)
        gpu_result = torch.mm(gpu_test, gpu_test.t())
        print(f'✅ CUDA matrix multiplication test: {gpu_result.device}, result sum: {gpu_result.sum().item():.2f}')
        
        del dummy_input, dummy_output, gpu_test, gpu_result
        torch.cuda.empty_cache()
        
        # Print current GPU utilization
        print(f'🔧 GPU Memory after test: {torch.cuda.memory_allocated() / 1024**2:.1f} MB')

    if cfg['training']['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=cfg['training']['lr'],
                              weight_decay=cfg['training']['weight_decay'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg['training']['lr'],
                               weight_decay=cfg['training']['weight_decay'])

    sched_cfg = cfg['training'].get('scheduler', {})
    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=sched_cfg.get('step_size', 10),
                                          gamma=sched_cfg.get('gamma', 0.1))
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    
    # Mixed precision training for better GPU utilization
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    use_amp = device.type == 'cuda'
    
    if use_amp:
        print(f'🚀 Using Automatic Mixed Precision (AMP) training')

    # ----- training epochs
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    
    # Early stopping variables
    patience = cfg['training'].get('early_stopping', {}).get('patience', 50)
    min_delta = cfg['training'].get('early_stopping', {}).get('min_delta', 0.001)
    patience_counter = 0
    
    print(f"\n🚀 Starting training for {cfg['training']['epochs']} epochs")
    print(f"📊 Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
    print(f"🔧 Training device: {device}")
    print(f"🛑 Early stopping: patience={patience}, min_delta={min_delta}")
    
    # Verify data loaders will use correct device
    if device.type == 'cuda':
        print(f"🔧 DataLoader pin_memory: {train_loader.pin_memory}")
        print(f"🔧 DataLoader num_workers: {train_loader.num_workers}")
        print(f"🔧 Initial GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        
        # Force CUDA context creation
        torch.cuda.set_device(0)
        torch.cuda.synchronize()
        print(f"🔧 CUDA context active on device: {torch.cuda.current_device()}")
    
    for epoch in range(1, cfg['training']['epochs'] + 1):
        # Clear CUDA cache at the start of each epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            # Force garbage collection to free up memory
            import gc
            gc.collect()
            
        # ---- train
        model.train()
        run_loss, run_total = 0.0, 0
        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            # Verify tensors are on correct device (only for first batch of first epoch)
            if epoch == 1 and step == 1:
                print(f"✅ First batch verification: x.device={x.device}, y.device={y.device}")
                print(f"✅ Batch shapes: x.shape={x.shape}, y.shape={y.shape}")
                print(f"✅ First batch x range: [{x.min():.3f}, {x.max():.3f}]")
                print(f"✅ First batch y range: [{y.min():.3f}, {y.max():.3f}]")
                print(f"✅ First batch y mean: {y.mean(axis=0)}")
                print(f"✅ First batch y std: {y.std(axis=0)}")
                
                # Force CUDA synchronization to ensure GPU is actually being used
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                    print(f"✅ CUDA synchronized for first batch")
            
            optimizer.zero_grad()
            
            if use_amp:
                # Mixed precision forward pass
                with autocast('cuda'):
                    out = model(x)
                    loss = criterion(out, y)
                
                # Mixed precision backward pass
                scaler.scale(loss).backward()
                
                # Add gradient clipping to prevent memory spikes
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                
                # Add gradient clipping to prevent memory spikes
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()

            run_loss  += loss.item() * x.size(0)
            run_total += y.size(0)
            if step % cfg['logging']['log_interval'] == 0:
                current_lr = optimizer.param_groups[0]['lr']
                if device.type == 'cuda':
                    torch.cuda.synchronize()  # Ensure GPU operations complete
                    mem_used = torch.cuda.memory_allocated() / 1024**2
                    print(f'Epoch {epoch} Step {step}/{len(train_loader)} '
                          f'loss {run_loss/run_total:.4f} lr {current_lr:.6f} '
                          f'GPU mem: {mem_used:.1f}MB')
                else:
                    print(f'Epoch {epoch} Step {step}/{len(train_loader)} '
                          f'loss {run_loss/run_total:.4f} lr {current_lr:.6f}')

        # Calculate average training loss for this epoch
        train_loss = run_loss / run_total
        train_losses.append(train_loss)
        
        scheduler.step()

        # ---- validation
        model.eval()
        val_loss, val_total = 0.0, 0
        
        # Clear cache before validation
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                out  = model(x)  # no squeeze needed for multi-output
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                val_total += y.size(0)
                
                # Clear intermediate variables to save memory
                del x, y, out, loss

        val_loss = val_loss / val_total
        val_losses.append(val_loss)
        
        # Print losses for this epoch (only every 10 epochs or if it's the last epoch)
        if epoch % 10 == 0 or epoch == cfg['training']['epochs']:
            if device.type == 'cuda':
                mem_allocated = torch.cuda.memory_allocated() / 1024**2
                mem_reserved = torch.cuda.memory_reserved() / 1024**2
                print(f'📈 Epoch {epoch:3d}/{cfg["training"]["epochs"]} - '
                      f'Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f} | '
                      f'GPU: {mem_allocated:.0f}MB/{mem_reserved:.0f}MB')
            else:
                print(f'📈 Epoch {epoch:3d}/{cfg["training"]["epochs"]} - '
                      f'Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | '
                      f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # ---- checkpoint - save best model (avoid saving unstable first epoch)
        # Only save if epoch > 1 and training loss is reasonable (not too high compared to val loss)
        if val_loss < best_loss and epoch > 1 and train_loss < (val_loss * 10):
            improvement = best_loss - val_loss
            if improvement > min_delta:
                best_loss = val_loss
                patience_counter = 0  # Reset patience counter
                
                ckpt_dir = Path(cfg['paths']['checkpoint_dir']); ckpt_dir.mkdir(exist_ok=True)
                
                # Save best model
                best_model_path = ckpt_dir / 'best_model.pth'
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'best_loss': best_loss,
                    'in_dim': in_dim,
                    'out_dim': out_dim,
                    'config': cfg,
                    'feature_scaler': full_ds.scaler,
                    'target_scaler': full_ds.target_scaler
                }, best_model_path)
                print(f'✅  NEW BEST MODEL → {best_model_path} (Val Loss: {best_loss:.6f}, Improvement: {improvement:.6f})')
            else:
                patience_counter += 1
        else:
            patience_counter += 1
        
        # Early stopping check
        if patience_counter >= patience:
            print(f'\n🛑 Early stopping triggered after {patience} epochs without improvement')
            print(f'📈 Best validation loss: {best_loss:.6f} at epoch {epoch - patience}')
            break
        
        # Save checkpoint every N epochs
        if epoch % cfg['training'].get('save_checkpoint_every', 5) == 0:
            ckpt_path = ckpt_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'in_dim': in_dim,
                'out_dim': out_dim,
                'config': cfg,
                'feature_scaler': full_ds.scaler,
                'target_scaler': full_ds.target_scaler
            }, ckpt_path)
            print(f'💾  Saved checkpoint → {ckpt_path}')

    # Save final model
    final_model_path = ckpt_dir / 'last_model.pth'
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': cfg['training']['epochs'],
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'in_dim': in_dim,
        'out_dim': out_dim,
        'config': cfg,
        'feature_scaler': full_ds.scaler,
        'target_scaler': full_ds.target_scaler
    }, final_model_path)
    print(f'💾  Saved final model → {final_model_path}')
    
    # Save loss history
    loss_history_path = ckpt_dir / 'loss_history.csv'
    loss_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses
    })
    loss_df.to_csv(loss_history_path, index=False)
    print(f'📊  Saved loss history → {loss_history_path}')

    # ========== COMPREHENSIVE EVALUATION ==========
    print(f'\n🔍 COMPREHENSIVE MODEL EVALUATION')
    print('=' * 60)
    
    # Evaluate on both training and validation sets
    print('📊 Evaluating on validation set...')
    val_metrics, val_loss_eval, val_predictions, val_targets = evaluate_model(
        model, val_loader, device, full_ds.target_scaler, criterion
    )
    
    print('📊 Evaluating on training set...')
    train_metrics, train_loss_eval, train_predictions, train_targets = evaluate_model(
        model, train_loader, device, full_ds.target_scaler, criterion
    )
    
    # Print detailed metrics
    print(f'\n📈 VALIDATION SET METRICS:')
    print('-' * 50)
    for target_name, metrics in val_metrics.items():
        print(f'{target_name.upper()}:')
        print(f'  MAE: {metrics["mae"]:.3f}')
        print(f'  RMSE: {metrics["rmse"]:.3f}')
        print(f'  R²: {metrics["r2"]:.4f}')
        print(f'  Accuracy (±{metrics["threshold"]}): {metrics["accuracy_pct"]:.1f}%')
        print(f'  Mean Prediction: {metrics["mean_pred"]:.2f} (Target: {metrics["mean_target"]:.2f})')
        print()
    
    print(f'📈 TRAINING SET METRICS:')
    print('-' * 50)
    for target_name, metrics in train_metrics.items():
        print(f'{target_name.upper()}:')
        print(f'  MAE: {metrics["mae"]:.3f}')
        print(f'  RMSE: {metrics["rmse"]:.3f}')
        print(f'  R²: {metrics["r2"]:.4f}')
        print(f'  Accuracy (±{metrics["threshold"]}): {metrics["accuracy_pct"]:.1f}%')
        print(f'  Mean Prediction: {metrics["mean_pred"]:.2f} (Target: {metrics["mean_target"]:.2f})')
        print()
    
    # Calculate overall accuracy
    val_overall_acc = np.mean([m['accuracy_pct'] for m in val_metrics.values()])
    train_overall_acc = np.mean([m['accuracy_pct'] for m in train_metrics.values()])
    
    print(f'🎯 OVERALL ACCURACY:')
    print(f'  Validation: {val_overall_acc:.1f}%')
    print(f'  Training: {train_overall_acc:.1f}%')
    
    # Save detailed predictions for analysis
    val_results_path = ckpt_dir / 'validation_predictions.csv'
    val_results_df = pd.DataFrame({
        'pred_final_spread': val_predictions[:, 0],
        'pred_final_total': val_predictions[:, 1],
        'true_final_spread': val_targets[:, 0],
        'true_final_total': val_targets[:, 1]
    })
    val_results_df.to_csv(val_results_path, index=False)
    print(f'📊  Saved validation predictions → {val_results_path}')
    
    # Calculate training time
    total_time = time.time() - start_time
    time_per_epoch = total_time / cfg['training']['epochs']
    
    print(f'\n⏱️  TRAINING STATISTICS:')
    print('-' * 50)
    print(f'Total Training Time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)')
    print(f'Time per Epoch: {time_per_epoch:.1f} seconds')
    print(f'Training Samples: {len(train_ds):,}')
    print(f'Validation Samples: {len(val_ds):,}')
    print(f'Total Parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Input Features: {in_dim:,}')
    print(f'Output Targets: {out_dim}')
    
    print(f'\n🎉 Training complete!')
    print(f'📈 Best validation loss: {best_loss:.6f}')
    print(f'📉 Final train loss: {train_losses[-1]:.6f}')
    print(f'📉 Final val loss: {val_losses[-1]:.6f}')
    print(f'🎯 Total improvement: {val_losses[0] - best_loss:.6f}')
    
    if device.type == 'cuda':
        print(f'🔥 Peak GPU memory usage: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB')
        print(f'✅ CUDA training completed successfully on: {torch.cuda.get_device_name(0)}')
    
    # Save comprehensive training summary
    summary_path = ckpt_dir / 'training_summary.json'
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        return obj
    
    training_summary = {
        'training_time_minutes': total_time / 60,
        'time_per_epoch_seconds': time_per_epoch,
        'total_epochs': cfg['training']['epochs'],
        'best_val_loss': float(best_loss),
        'final_train_loss': float(train_losses[-1]),
        'final_val_loss': float(val_losses[-1]),
        'training_samples': len(train_ds),
        'validation_samples': len(val_ds),
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'input_features': in_dim,
        'output_targets': out_dim,
        'validation_metrics': convert_numpy_types(val_metrics),
        'training_metrics': convert_numpy_types(train_metrics),
        'overall_val_accuracy': float(val_overall_acc),
        'overall_train_accuracy': float(train_overall_acc),
        'device': str(device),
        'cuda_peak_memory_mb': float(torch.cuda.max_memory_allocated() / 1024**2) if device.type == 'cuda' else None
    }
    
    with open(summary_path, 'w') as f:
        json.dump(training_summary, f, indent=2)
    print(f'📊  Saved training summary → {summary_path}')
    
    # Generate comprehensive markdown report
    report_path = generate_training_report(training_summary, val_predictions, val_targets, cfg, ckpt_dir)
    
    # Run comparison with bookmakers after training
    print(f'\n🔍 Running comparison with bookmaker predictions...')
    try:
        import subprocess
        result = subprocess.run([
            'C:/Users/admin/Desktop/training/.venv/Scripts/python.exe', 
            'compare_predictions.py'
        ], cwd=Path.cwd(), capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        if result.returncode == 0:
            print(f'✅ Comparison completed successfully')
            print(f'📄 Check resources/ folder for detailed comparison report')
        else:
            print(f'⚠️ Comparison failed with return code {result.returncode}')
            if result.stderr:
                print(f'Error: {result.stderr}')
    except Exception as e:
        print(f'⚠️ Could not run comparison: {e}')
        print(f'💡 You can run it manually: python compare_predictions.py')
    
    print(f'\n🎊 MODEL READY FOR INFERENCE!')
    print(f'📁 Best model saved at: {ckpt_dir}/best_model.pth')
    print(f'📁 Training logs saved at: {ckpt_dir}/')
    print(f'📄 Training report saved at: {report_path}')
    print(f'📄 Comparison report saved in: resources/')


def generate_training_report(summary, predictions, targets, config, checkpoint_dir):
    """Generate a comprehensive training report in markdown format"""
    
    # Create resources directory if it doesn't exist
    resources_dir = Path('resources')
    resources_dir.mkdir(exist_ok=True)
    
    # Create report filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = resources_dir / f'training_report_{timestamp}.md'
    
    # Load bookmaker predictions for comparison
    try:
        bm_predictions_path = resources_dir / 'bm_predictions.csv'
        bm_df = pd.read_csv(bm_predictions_path)
        has_bm_data = True
    except:
        has_bm_data = False
    
    # Generate the markdown report
    report = f"""# Training Report - {timestamp}

## 🎯 Executive Summary

**Model Performance vs Previous Baseline:**
- **Validation Accuracy**: {summary['overall_val_accuracy']:.1f}% (vs ~20% previously)
- **Training Accuracy**: {summary['overall_train_accuracy']:.1f}%
- **Improvement**: {summary['overall_val_accuracy'] - 20:.1f} percentage points better than baseline

**Key Insights:**
- {'✅ SIGNIFICANT IMPROVEMENT' if summary['overall_val_accuracy'] > 50 else '❌ NEEDS IMPROVEMENT'}: Model accuracy is {'above' if summary['overall_val_accuracy'] > 50 else 'below'} 50%
- {'✅ GOOD GENERALIZATION' if abs(summary['overall_val_accuracy'] - summary['overall_train_accuracy']) < 20 else '⚠️ OVERFITTING DETECTED'}: Training vs Validation gap is {abs(summary['overall_val_accuracy'] - summary['overall_train_accuracy']):.1f}%
- **R² Scores**: Spread: {summary['validation_metrics']['final_spread']['r2']:.3f}, Total: {summary['validation_metrics']['final_total']['r2']:.3f}

---

## 📊 Training Configuration

| Parameter | Value |
|-----------|-------|
| **Architecture** | {config['model']['hidden_dims']} |
| **Dropout** | {config['model']['dropout']} |
| **Learning Rate** | {config['training']['lr']} |
| **Batch Size** | {config['data']['batch_size']} |
| **Epochs Trained** | {summary['total_epochs']} |
| **Early Stopping** | {'Yes' if summary['total_epochs'] < config['training']['epochs'] else 'No'} |
| **Training Time** | {summary['training_time_minutes']:.1f} minutes |
| **Device** | {summary['device']} |
| **Parameters** | {summary['model_parameters']:,} |

---

## 🎯 Model Performance Metrics

### Validation Set Performance
| Metric | Spread | Total |
|--------|--------|-------|
| **MAE** | {summary['validation_metrics']['final_spread']['mae']:.3f} | {summary['validation_metrics']['final_total']['mae']:.3f} |
| **RMSE** | {summary['validation_metrics']['final_spread']['rmse']:.3f} | {summary['validation_metrics']['final_total']['rmse']:.3f} |
| **R² Score** | {summary['validation_metrics']['final_spread']['r2']:.3f} | {summary['validation_metrics']['final_total']['r2']:.3f} |
| **Accuracy (±{summary['validation_metrics']['final_spread']['threshold']:.1f})** | {summary['validation_metrics']['final_spread']['accuracy_pct']:.1f}% | {summary['validation_metrics']['final_total']['accuracy_pct']:.1f}% |
| **Tight Accuracy (±{summary['validation_metrics']['final_spread']['tight_threshold']:.1f})** | {summary['validation_metrics']['final_spread']['tight_accuracy_pct']:.1f}% | {summary['validation_metrics']['final_total']['tight_accuracy_pct']:.1f}% |
| **Directional Accuracy** | {summary['validation_metrics']['final_spread']['directional_accuracy']:.1f}% | {summary['validation_metrics']['final_total']['directional_accuracy']:.1f}% |

### Training Set Performance
| Metric | Spread | Total |
|--------|--------|-------|
| **MAE** | {summary['training_metrics']['final_spread']['mae']:.3f} | {summary['training_metrics']['final_total']['mae']:.3f} |
| **RMSE** | {summary['training_metrics']['final_spread']['rmse']:.3f} | {summary['training_metrics']['final_total']['rmse']:.3f} |
| **R² Score** | {summary['training_metrics']['final_spread']['r2']:.3f} | {summary['training_metrics']['final_total']['r2']:.3f} |
| **Accuracy (±{summary['training_metrics']['final_spread']['threshold']:.1f})** | {summary['training_metrics']['final_spread']['accuracy_pct']:.1f}% | {summary['training_metrics']['final_total']['accuracy_pct']:.1f}% |

---

## 📈 Performance Analysis

### Strengths
"""

    # Add strengths analysis
    strengths = []
    if summary['validation_metrics']['final_spread']['r2'] > 0.5:
        strengths.append(f"- **Strong Spread Prediction**: R² = {summary['validation_metrics']['final_spread']['r2']:.3f}")
    if summary['validation_metrics']['final_total']['r2'] > 0.5:
        strengths.append(f"- **Strong Total Prediction**: R² = {summary['validation_metrics']['final_total']['r2']:.3f}")
    if summary['validation_metrics']['final_spread']['accuracy_pct'] > 50:
        strengths.append(f"- **Above-Average Spread Accuracy**: {summary['validation_metrics']['final_spread']['accuracy_pct']:.1f}%")
    if summary['validation_metrics']['final_total']['accuracy_pct'] > 50:
        strengths.append(f"- **Above-Average Total Accuracy**: {summary['validation_metrics']['final_total']['accuracy_pct']:.1f}%")
    if abs(summary['overall_val_accuracy'] - summary['overall_train_accuracy']) < 15:
        strengths.append(f"- **Good Generalization**: Low overfitting ({abs(summary['overall_val_accuracy'] - summary['overall_train_accuracy']):.1f}% gap)")
    
    if strengths:
        report += "\n".join(strengths)
    else:
        report += "- Model shows room for improvement in all areas"

    report += "\n\n### Areas for Improvement\n"
    
    # Add areas for improvement
    improvements = []
    if summary['validation_metrics']['final_spread']['r2'] < 0.3:
        improvements.append(f"- **Spread Prediction Needs Work**: R² = {summary['validation_metrics']['final_spread']['r2']:.3f} (target: >0.5)")
    if summary['validation_metrics']['final_total']['r2'] < 0.3:
        improvements.append(f"- **Total Prediction Needs Work**: R² = {summary['validation_metrics']['final_total']['r2']:.3f} (target: >0.5)")
    if summary['validation_metrics']['final_spread']['accuracy_pct'] < 50:
        improvements.append(f"- **Spread Accuracy Below 50%**: {summary['validation_metrics']['final_spread']['accuracy_pct']:.1f}% (coin flip territory)")
    if summary['validation_metrics']['final_total']['accuracy_pct'] < 50:
        improvements.append(f"- **Total Accuracy Below 50%**: {summary['validation_metrics']['final_total']['accuracy_pct']:.1f}% (coin flip territory)")
    if abs(summary['overall_val_accuracy'] - summary['overall_train_accuracy']) > 20:
        improvements.append(f"- **Overfitting Detected**: {abs(summary['overall_val_accuracy'] - summary['overall_train_accuracy']):.1f}% performance gap")
    
    if improvements:
        report += "\n".join(improvements)
    else:
        report += "- Model performance is solid across all metrics"

    report += f"""

---

## 🚀 Next Steps & Recommendations

### If Performance is Good (>60% accuracy):
1. **Deploy for Live Prediction**: Model is ready for real-world testing
2. **Optimize Confidence Thresholds**: Focus on high-confidence predictions
3. **Feature Engineering**: Add more sophisticated features
4. **Ensemble Methods**: Combine with other models

### If Performance Needs Improvement (<50% accuracy):
1. **Data Quality Check**: Verify training data alignment with targets
2. **Feature Engineering**: Add more relevant features
3. **Hyperparameter Tuning**: Experiment with learning rates, architectures
4. **Data Augmentation**: Add more training samples

### Current Recommendation:
**{'🎯 READY FOR DEPLOYMENT' if summary['overall_val_accuracy'] > 60 else '⚠️ NEEDS IMPROVEMENT' if summary['overall_val_accuracy'] > 40 else '❌ MAJOR REVISION NEEDED'}**

---

## 📁 Files Generated

- **Best Model**: `checkpoints/best_model.pth`
- **Training Summary**: `checkpoints/training_summary.json`
- **Loss History**: `checkpoints/loss_history.csv`
- **Validation Predictions**: `checkpoints/validation_predictions.csv`
- **This Report**: `{report_path}`

---

## 🔧 Technical Details

### Data Processing
- **Training Samples**: {summary['training_samples']:,}
- **Validation Samples**: {summary['validation_samples']:,}
- **Input Features**: {summary['input_features']:,}
- **Feature Normalization**: Standard Scaler
- **Target Normalization**: Standard Scaler

### Model Architecture
- **Type**: Multi-Layer Perceptron (MLP)
- **Hidden Layers**: {config['model']['hidden_dims']}
- **Activation**: ReLU
- **Dropout**: {config['model']['dropout']}
- **Output**: 2 targets (spread, total)

### Training Details
- **Loss Function**: Mean Squared Error
- **Optimizer**: Adam
- **Learning Rate**: {config['training']['lr']}
- **Batch Size**: {config['data']['batch_size']}
- **Device**: {summary['device']}
- **Memory Usage**: {summary.get('cuda_peak_memory_mb', 0):.1f} MB

---

*Report generated on {timestamp}*
"""

    # Write the report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f'📄  Generated training report → {report_path}')
    return report_path


def evaluate_model(model, data_loader, device, target_scaler, criterion):
    """Comprehensive model evaluation with detailed metrics focused on sports betting"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            outputs = model(x)
            loss = criterion(outputs, y)
            
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)
            
            # Denormalize predictions and targets for meaningful metrics
            pred_denorm = target_scaler.inverse_transform(outputs.cpu().numpy())
            target_denorm = target_scaler.inverse_transform(y.cpu().numpy())
            
            all_predictions.extend(pred_denorm)
            all_targets.extend(target_denorm)
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    
    # Calculate metrics for each target
    target_names = ['final_spread', 'final_total']
    metrics = {}
    
    for i, target_name in enumerate(target_names):
        pred_col = predictions[:, i]
        target_col = targets[:, i]
        
        mae = mean_absolute_error(target_col, pred_col)
        mse = mean_squared_error(target_col, pred_col)
        rmse = np.sqrt(mse)
        r2 = r2_score(target_col, pred_col)
        
        # Calculate percentage accuracy (within certain thresholds)
        if 'spread' in target_name:
            # For spreads, consider predictions within 3 points as accurate
            threshold = 3.0
            # Also calculate tighter threshold for better evaluation
            tight_threshold = 1.5
        else:
            # For totals, consider predictions within 5 points as accurate  
            threshold = 5.0
            # Also calculate tighter threshold for better evaluation
            tight_threshold = 3.0
        
        within_threshold = np.abs(pred_col - target_col) <= threshold
        within_tight_threshold = np.abs(pred_col - target_col) <= tight_threshold
        accuracy_pct = np.mean(within_threshold) * 100
        tight_accuracy_pct = np.mean(within_tight_threshold) * 100
        
        # Calculate directional accuracy (getting the sign right)
        correct_direction = np.sign(pred_col) == np.sign(target_col)
        directional_accuracy = np.mean(correct_direction) * 100
        
        metrics[target_name] = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'accuracy_pct': accuracy_pct,
            'tight_accuracy_pct': tight_accuracy_pct,
            'directional_accuracy': directional_accuracy,
            'threshold': threshold,
            'tight_threshold': tight_threshold,
            'mean_pred': np.mean(pred_col),
            'mean_target': np.mean(target_col),
            'std_pred': np.std(pred_col),
            'std_target': np.std(target_col)
        }
    
    avg_loss = total_loss / total_samples
    
    return metrics, avg_loss, predictions, targets


# ---------- CLI --------------------------------------------------------------
if __name__ == '__main__':
    set_seed()
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='training_config.yaml',
                   help='Path to YAML configuration.')
    args = p.parse_args()
    train(args.config)
