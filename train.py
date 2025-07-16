#!/usr/bin/env python
import yaml
import torch.optim as optim
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import time
import subprocess
import random
import gc
import json
import glob
import argparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from datetime import datetime
from pathlib import Path
_k = 'tight_threshold'
_j = 'cuda_peak_memory_mb'
_i = 'input_features'
_h = 'model_parameters'
_g = 'validation_samples'
_f = 'training_samples'
_e = 'training_time_minutes'
_d = 'file_name'
_c = 'directional_accuracy'
_b = 'tight_accuracy_pct'
_a = 'total_epochs'
_Z = 'mean_target'
_Y = 'mean_pred'
_X = 'dropout'
_W = 'hidden_dims'
_V = 'batch_size'
_U = 'device'
_T = 'target_scaler'
_S = 'threshold'
_R = 'rmse'
_Q = 'mae'
_P = 'lr'
_O = 'overall_train_accuracy'
_N = 'epochs'
_M = 'model'
_L = 'training_metrics'
_K = 'data'
_J = False
_I = 'overall_val_accuracy'
_H = None
_G = 'accuracy_pct'
_F = 'r2'
_E = True
_D = 'training'
_C = 'final_total'
_B = 'final_spread'
_A = 'validation_metrics'
torch.backends.cudnn.enabled = _E
torch.backends.cudnn.benchmark = _E
torch.backends.cudnn.deterministic = _J


class CSVDataset(Dataset):
    def __init__(self, csv_files, outcomes_csv_path, scaler=None, max_rows=None, bm_pred_path=None):
        self.csv_files = csv_files
        self.max_rows = max_rows
        self.target_scaler = None
        outcomes_df = pd.read_csv(outcomes_csv_path)
        self.outcomes = {row[_d]: {_B: row[_B], _C: row[_C]}
                         for _, row in outcomes_df.iterrows()}
        self.bm_predictions = {}
        if bm_pred_path and Path(bm_pred_path).exists():
            bm_data = pd.read_csv(bm_pred_path)
            for _, row in bm_data.iterrows():
                self.bm_predictions[row["filename"]] = {
                    "bm_total_prediction_at_halftime": row["bm_total_prediction_at_halftime"],
                    "bm_spread_prediction_at_halftime": row["bm_spread_prediction_at_halftime"]
                }
        self.features, self.labels, self.scaler = self._build_tensor(scaler)

    def _determine_dimensions(self):
        cols = None
        max_rows = 0
        for path in self.csv_files:
            df = pd.read_csv(path)
            numeric = df.select_dtypes(include=[np.number])
            if cols is None:
                cols = len(numeric.columns)
            max_rows = max(max_rows, len(numeric))
        return cols, max_rows

    def _build_tensor(self, scaler):
        in_dim, max_rows = self._determine_dimensions()
        if self.max_rows is None:
            self.max_rows = max_rows
        feature_list = []
        label_list = []
        file_names = []
        for csv_path in self.csv_files:
            df = pd.read_csv(csv_path)
            numeric = df.select_dtypes(include=[np.number])
            numeric = numeric.fillna(0)
            if len(numeric) > self.max_rows:
                numeric = numeric.iloc[:self.max_rows]
            elif len(numeric) < self.max_rows:
                pad = pd.DataFrame(np.zeros(
                    (self.max_rows - len(numeric), len(numeric.columns))), columns=numeric.columns)
                numeric = pd.concat([numeric, pad], ignore_index=True)
            feature_list.append(numeric.values.flatten())
            name = Path(csv_path).name
            file_names.append(name)
            if name in self.outcomes and name in self.bm_predictions:
                outcome = self.outcomes[name]
                bm = self.bm_predictions[name]
                spread_label = 1.0 if outcome[_B] > bm["bm_spread_prediction_at_halftime"] else 0.0
                total_label = 1.0 if outcome[_C] > bm["bm_total_prediction_at_halftime"] else 0.0
                label_list.append([spread_label, total_label])
        features = np.array(feature_list, dtype=np.float32)
        labels = np.array(label_list, dtype=np.float32)
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(features)
        features = scaler.transform(features)
        self.scaler = scaler
        self.file_names = file_names
        self.target_scaler = None
        return torch.from_numpy(features), torch.from_numpy(labels), scaler

    def __len__(self):
        return self.labels.size(0)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class MLP(nn.Module):
    """Simple feed-forward network with optional batch normalization."""

    def __init__(self, in_dim, hidden_dims, out_dim, dropout, batch_norm=False):
        super().__init__()
        layers = []
        dim = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(dim, h))
            if batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.extend([nn.ReLU(), nn.Dropout(dropout)])
            dim = h
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(
                module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.net(x)


def set_seed(seed=42): A = seed; random.seed(A); np.random.seed(
    A); torch.manual_seed(A); torch.cuda.manual_seed_all(A)


def temporal_train_test_split(features, labels, file_names, test_size=0.2):
    """
    Perform temporal train-test split to prevent data leakage.
    Training data comes from earlier time periods than validation data.
    """
    # Create list of (filename, index) pairs and sort by timestamp
    file_index_pairs = [(file_names[i], i) for i in range(len(file_names))]

    # Sort by timestamp extracted from filename (format: game_YYYY-MM-DD_HH-MM.csv)
    def extract_timestamp(filename):
        try:
            # Extract timestamp from filename like "game_2025-01-01_04-28.csv"
            timestamp_str = filename.replace('game_', '').replace('.csv', '')
            return timestamp_str
        except:
            return filename

    file_index_pairs.sort(key=lambda x: extract_timestamp(x[0]))

    # Calculate split point for temporal split
    split_point = int(len(file_index_pairs) * (1 - test_size))

    # Get indices for train and test sets
    train_indices = [pair[1] for pair in file_index_pairs[:split_point]]
    test_indices = [pair[1] for pair in file_index_pairs[split_point:]]

    # Create temporal split
    X_train = features[train_indices]
    X_test = features[test_indices]
    y_train = labels[train_indices]
    y_test = labels[test_indices]

    # Get temporal boundary info
    train_files = [file_index_pairs[i][0] for i in range(split_point)]
    test_files = [file_index_pairs[i][0]
                  for i in range(split_point, len(file_index_pairs))]

    print(f"🕐 TEMPORAL SPLIT APPLIED:")
    print(
        f"  Training period: {extract_timestamp(train_files[0])} to {extract_timestamp(train_files[-1])}")
    print(
        f"  Validation period: {extract_timestamp(test_files[0])} to {extract_timestamp(test_files[-1])}")
    print(f"  ✅ NO TEMPORAL LEAKAGE: All training data comes before validation data")

    return X_train, X_test, y_train, y_test


def train(cfg_path):
    AQ = 'final_val_loss'
    AP = 'final_train_loss'
    AO = 'val_loss'
    AN = 'train_loss'
    AM = 'early_stopping'
    AL = 'weight_decay'
    A3 = 'feature_scaler'
    A2 = 'config'
    A1 = 'out_dim'
    A0 = 'in_dim'
    z = 1.
    l = 'epoch'
    k = 'scheduler'
    j = 'optimizer'
    i = 'num_workers'
    E = 'cuda'
    AR = time.time()
    with open(cfg_path)as m:
        A = yaml.safe_load(m)
    device_cfg = A.get(_U, 'auto')
    if device_cfg == 'auto':
        if torch.cuda.is_available():
            B = torch.device(E)
            print(f"🚀 Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            B = torch.device('cpu')
            print('⚠️  CUDA not available, falling back to CPU')
    elif device_cfg == E:
        if not torch.cuda.is_available():
            print('⚠️  CUDA not available, falling back to CPU')
            B = torch.device('cpu')
        else:
            B = torch.device(E)
            print(f"🚀 Using CUDA device: {torch.cuda.get_device_name(0)}")
            print(
                f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
            print(f"🔧 CUDA Version: {torch.version.cuda}")
            print(f"🔧 PyTorch Version: {torch.__version__}")
            print(f"🔧 CUDA Device Count: {torch.cuda.device_count()}")
            print(f"🔧 Current CUDA Device: {torch.cuda.current_device()}")
            n = torch.tensor([z, 2., 3.]).cuda()
            print(f"✅ CUDA Test: {n.device} - {n.sum().item()}")
            print(f"🔧 cuDNN enabled: {torch.backends.cudnn.enabled}")
            print(f"🔧 cuDNN benchmark: {torch.backends.cudnn.benchmark}")
            print(f"🔧 cuDNN version: {torch.backends.cudnn.version()}")
            print(f"🔧 cuDNN deterministic: {torch.backends.cudnn.deterministic}")
            del n
            torch.cuda.empty_cache()
    else:
        B = torch.device('cpu')
        print('⚠️  Using CPU for training')
    AS = Path(A[_K]['train_dir'])
    AT = glob.glob(str(AS/'*.csv'))
    A4 = A[_K]['outcomes_csv']
    AU = pd.read_csv(A4)
    AV = set(AU[_d].values)
    o = [A for A in AT if Path(A).name in AV]

    # Sort files chronologically to prevent temporal leakage
    def extract_timestamp_for_sort(filepath):
        try:
            filename = Path(filepath).name
            timestamp_str = filename.replace('game_', '').replace('.csv', '')
            return timestamp_str
        except:
            return filename

    o.sort(key=extract_timestamp_for_sort)
    print(
        f"Found {len(o)} CSV files with matching outcomes (sorted chronologically)")
    print(f"📅 Data spans from {Path(o[0]).name} to {Path(o[-1]).name}")

    if not o:
        raise ValueError('No CSV files found with matching outcomes!')

    # Load dataset with bookmaker predictions
    bm_pred_path = "bm_pred_plus_outcomes.csv"
    J = CSVDataset(o, A4, bm_pred_path=bm_pred_path if Path(
        bm_pred_path).exists() else None)
    N = J.features.shape[1]
    R = 2

    # Use temporal split instead of random split to prevent data leakage
    X, Y, S, T = temporal_train_test_split(
        J.features, J.labels, J.file_names, test_size=0.2)
    p = torch.utils.data.TensorDataset(X, S)
    q = torch.utils.data.TensorDataset(Y, T)
    print(f"🔍 DATA DEBUGGING:")
    print(f"  Training features shape: {X.shape}")
    print(f"  Training targets shape: {S.shape}")
    print(f"  Validation features shape: {Y.shape}")
    print(f"  Validation targets shape: {T.shape}")
    print(f"  Training features range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"  Training targets range: [{S.min():.3f}, {S.max():.3f}]")
    print(f"  Validation features range: [{Y.min():.3f}, {Y.max():.3f}]")
    print(f"  Validation targets range: [{T.min():.3f}, {T.max():.3f}]")
    print(f"  Training targets mean: {S.mean(axis=0)}")
    print(f"  Validation targets mean: {T.mean(axis=0)}")
    O = DataLoader(p, batch_size=A[_K][_V], shuffle=_E, num_workers=A[_K][i], pin_memory=_E if B.type ==
                   E else _J, persistent_workers=_J, drop_last=_E, prefetch_factor=2 if A[_K][i] > 0 else _H)
    A5 = DataLoader(q, batch_size=A[_K][_V], shuffle=_J, num_workers=A[_K][i], pin_memory=_E if B.type ==
                    E else _J, persistent_workers=_J, drop_last=_J, prefetch_factor=2 if A[_K][i] > 0 else _H)
    C = MLP(N, A[_M][_W], R, A[_M][_X], batch_norm=A[_M].get('batch_norm', False)).to(B)

    # Compute positive class weights to handle imbalance
    cfg_weights = A[_D].get('class_weights')
    if cfg_weights is not None:
        pos_weight = torch.tensor(cfg_weights, dtype=torch.float32)
    else:
        label_sum = S.sum(dim=0)
        total_labels = S.size(0)
        pos_weight = (total_labels - label_sum) / (label_sum + 1e-6)
        pos_weight = torch.clamp(pos_weight, min=1.0)
    AW = sum(A.numel()for A in C.parameters())
    print(f"🧠 Model has {AW:,} parameters")
    print(f"🔧 Model device: {next(C.parameters()).device}")
    if B.type == E:
        print(
            f"📊 Model memory usage: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        print(
            f"📊 GPU memory reserved: {torch.cuda.memory_reserved()/1024**2:.1f} MB")
        r = torch.randn(1, N).to(B)
        with torch.no_grad():
            A6 = C(r)
        print(
            f"✅ CUDA forward pass test: input device={r.device}, output device={A6.device}")
        s = torch.randn(1000, 1000).to(B)
        t = torch.mm(s, s.t())
        print(
            f"✅ CUDA matrix multiplication test: {t.device}, result sum: {t.sum().item():.2f}")
        del r, A6, s, t
        torch.cuda.empty_cache()
        print(
            f"🔧 GPU Memory after test: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    if A[_D][j].lower() == 'sgd':
        I = optim.SGD(C.parameters(), lr=A[_D][_P], weight_decay=A[_D][AL])
    else:
        I = optim.Adam(C.parameters(), lr=A[_D][_P], weight_decay=A[_D][AL])
        A7 = A[_D].get(k, {})
        Z = optim.lr_scheduler.StepLR(I, step_size=A7.get('step_size', 10),
                                       gamma=A7.get('gamma', .1))
        U = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(B))
        a = GradScaler(E) if B.type == E else _H
        A8 = B.type == E
    if A8:
        print(f"🚀 Using Automatic Mixed Precision (AMP) training")
    P = float('inf')
    V = []
    b = []
    c = A[_D].get(AM, {}).get('patience', 50)
    A9 = A[_D].get(AM, {}).get('min_delta', .001)
    d = 0
    print(f"\n🚀 Starting training for {A[_D][_N]} epochs")
    print(f"📊 Training samples: {len(p)}, Validation samples: {len(q)}")
    print(f"🔧 Training device: {B}")
    print(f"🛑 Early stopping: patience={c}, min_delta={A9}")
    if B.type == E:
        print(f"🔧 DataLoader pin_memory: {O.pin_memory}")
        print(f"🔧 DataLoader num_workers: {O.num_workers}")
        print(
            f"🔧 Initial GPU memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
        torch.cuda.set_device(0)
        torch.cuda.synchronize()
        print(
            f"🔧 CUDA context active on device: {torch.cuda.current_device()}")

    # Create versioned model directory
    versioned_dir = create_versioned_model_dir()
    print(f"📁 Created versioned model directory: {versioned_dir}")

    # Update checkpoint directory to use versioned directory
    A['paths']['checkpoint_dir'] = str(versioned_dir)
    L = versioned_dir  # Set L to versioned directory for all file saving

    # Also create the original checkpoints directory for compatibility
    original_checkpoints = Path('./checkpoints')
    original_checkpoints.mkdir(exist_ok=True)

    for H in range(1, A[_D][_N]+1):
        if B.type == E:
            torch.cuda.empty_cache()
            import gc
            gc.collect()
        C.train()
        e, f = .0, 0
        for (g, (F, D)) in enumerate(O, 1):
            F, D = F.to(B, non_blocking=_E), D.to(B, non_blocking=_E)
            if H == 1 and g == 1:
                print(
                    f"✅ First batch verification: x.device={F.device}, y.device={D.device}")
                print(f"✅ Batch shapes: x.shape={F.shape}, y.shape={D.shape}")
                print(f"✅ First batch x range: [{F.min():.3f}, {F.max():.3f}]")
                print(f"✅ First batch y range: [{D.min():.3f}, {D.max():.3f}]")
                print(f"✅ First batch y mean: {D.mean(axis=0)}")
                print(f"✅ First batch y std: {D.std(axis=0)}")
                if B.type == E:
                    torch.cuda.synchronize()
                    print(f"✅ CUDA synchronized for first batch")
            I.zero_grad()
            if A8:
                with autocast(E):
                    Q = C(F)
                    M = U(Q, D)
                a.scale(M).backward()
                a.unscale_(I)
                torch.nn.utils.clip_grad_norm_(C.parameters(), max_norm=z)
                a.step(I)
                a.update()
            else:
                Q = C(F)
                M = U(Q, D)
                M.backward()
                torch.nn.utils.clip_grad_norm_(C.parameters(), max_norm=z)
                I.step()
            e += M.item()*F.size(0)
            f += D.size(0)
            if g % A['logging']['log_interval'] == 0:
                AA = I.param_groups[0][_P]
                # Evaluate on validation set for current val loss
                C.eval()
                val_loss = None
                with torch.no_grad():
                    val_loss_sum = 0.0
                    val_count = 0
                    for valF, valD in A5:
                        valF, valD = valF.to(B, non_blocking=_E), valD.to(
                            B, non_blocking=_E)
                        valQ = C(valF)
                        valM = U(valQ, valD)
                        val_loss_sum += valM.item() * valF.size(0)
                        val_count += valD.size(0)
                    val_loss = val_loss_sum / \
                        val_count if val_count > 0 else float('nan')
                C.train()
                if B.type == E:
                    AX = torch.cuda.memory_allocated()/1024**2
                    print(
                        f"Epoch {H} Step {g}/{len(O)} loss {e/f:.4f} val_loss {val_loss:.4f} lr {AA:.6f} GPU mem: {AX:.1f}MB")
                else:
                    print(
                        f"Epoch {H} Step {g}/{len(O)} loss {e/f:.4f} val_loss {val_loss:.4f} lr {AA:.6f}")
        W = e/f
        V.append(W)
        Z.step()
        C.eval()
        K, AB = .0, 0
        if B.type == E:
            torch.cuda.empty_cache()
        with torch.no_grad():
            for (F, D) in A5:
                F, D = F.to(B, non_blocking=_E), D.to(B, non_blocking=_E)
                Q = C(F)
                M = U(Q, D)
                K += M.item()*F.size(0)
                AB += D.size(0)
                del F, D, Q, M
        K = K/AB
        b.append(K)
        if H % 10 == 0 or H == A[_D][_N]:
            if B.type == E:
                AY = torch.cuda.memory_allocated()/1024**2
                AZ = torch.cuda.memory_reserved()/1024**2
                print(
                    f"📈 Epoch {H:3d}/{A[_D][_N]} - Train Loss: {W:.6f} | Val Loss: {K:.6f} | LR: {I.param_groups[0][_P]:.6f} | GPU: {AY:.0f}MB/{AZ:.0f}MB")
            else:
                print(
                    f"📈 Epoch {H:3d}/{A[_D][_N]} - Train Loss: {W:.6f} | Val Loss: {K:.6f} | LR: {I.param_groups[0][_P]:.6f}")
        if K < P and H > 1 and W < K*10:
            Aa = P-K
            if Aa > A9:
                P = K
                d = 0
                L = Path(A['paths']['checkpoint_dir'])
                L.mkdir(exist_ok=_E)
                Ab = L/'best_model.pth'
                torch.save({_M: C.state_dict(), j: I.state_dict(), k: Z.state_dict(
                ), l: H, 'best_loss': P, A0: N, A1: R, A2: A, A3: J.scaler, _T: J.target_scaler}, Ab)
            else:
                d += 1
        else:
            d += 1
        if d >= c:
            print(
                f"\n🛑 Early stopping triggered after {c} epochs without improvement")
            print(f"📈 Best validation loss: {P:.6f} at epoch {H-c}")
            break
        if H % A[_D].get('save_checkpoint_every', 5) == 0:
            AC = L/f"checkpoint_epoch_{H}.pth"
            torch.save({_M: C.state_dict(), j: I.state_dict(), k: Z.state_dict(
            ), l: H, AN: W, AO: K, A0: N, A1: R, A2: A, A3: J.scaler, _T: J.target_scaler}, AC)
            print(f"💾  Saved checkpoint → {AC}")
        AD = L/'last_model.pth'
        torch.save({_M: C.state_dict(), j: I.state_dict(), k: Z.state_dict(
        ), l: A[_D][_N], AP: V[-1], AQ: b[-1], A0: N, A1: R, A2: A, A3: J.scaler, _T: J.target_scaler}, AD)
        print(f"💾  Saved final model → {AD}")
        AE = L/'loss_history.csv'
        Ac = pd.DataFrame({l: range(1, len(V)+1), AN: V, AO: b})
        Ac.to_csv(AE, index=_J)
        print(f"📊  Saved loss history → {AE}")
        print(f"\n🔍 COMPREHENSIVE MODEL EVALUATION")
        print('='*60)
        print('📊 Evaluating on validation set...')
        u, Ah, AF, AG = evaluate_model(C, A5, B, U)
        print('📊 Evaluating on training set...')
        v, Ai, Aj, Ak = evaluate_model(C, O, B, U)
        print(f"\n📈 VALIDATION SET METRICS:")
        print('-'*50)
        for (w, G) in u.items():
            print(f"{w.upper()}:")
            print(f"  Accuracy: {G[_G]:.1f}%")
            print(f"  Precision: {G['precision']:.3f}")
            print(f"  Recall: {G['recall']:.3f}")
            print(f"  F1: {G['f1']:.3f}")
            print(f"  Mean Prob: {G['mean_prob']:.3f}")
            print()
        print(f"📈 TRAINING SET METRICS:")
        print('-'*50)
        for (w, G) in v.items():
            print(f"{w.upper()}:")
            print(f"  Accuracy: {G[_G]:.1f}%")
            print(f"  Precision: {G['precision']:.3f}")
            print(f"  Recall: {G['recall']:.3f}")
            print(f"  F1: {G['f1']:.3f}")
            print(f"  Mean Prob: {G['mean_prob']:.3f}")
            print()
    AH = np.mean([A[_G]for A in u.values()])
    AI = np.mean([A[_G]for A in v.values()])
    print(f"🎯 OVERALL ACCURACY:")
    print(f"  Validation: {AH:.1f}%")
    print(f"  Training: {AI:.1f}%")
    AJ = L/'validation_predictions.csv'
    Ad = pd.DataFrame({'pred_spread_win': AF[:, 0], 'pred_total_win': AF[:, 1],
                      'true_spread_win': AG[:, 0], 'true_total_win': AG[:, 1]})
    Ad.to_csv(AJ, index=_J)
    print(f"📊  Saved validation predictions → {AJ}")
    x = L/'training_summary.json'

    def y(obj):
        A = obj
        if isinstance(A, np.floating):
            return float(A)
        elif isinstance(A, np.integer):
            return int(A)
        elif isinstance(A, np.ndarray):
            return A.tolist()
        elif isinstance(A, dict):
            return {A: y(B)for (A, B) in A.items()}
        return A
    AK = time.time()-AR
    Ae = AK/A[_D][_N]
    Af = {_e: AK/60, 'time_per_epoch_seconds': Ae, _a: A[_D][_N], 'best_val_loss': float(P), AP: float(V[-1]), AQ: float(b[-1]), _f: len(p), _g: len(q), _h: sum(A.numel()for A in C.parameters(
    )), _i: N, 'output_targets': R, _A: y(u), _L: y(v), _I: float(AH), _O: float(AI), _U: str(B), _j: float(torch.cuda.max_memory_allocated()/1024**2)if B.type == E else _H}
    with open(x, 'w')as m:
        json.dump(Af, m, indent=2)
    print(f"📊  Saved training summary → {x}")
    print(f"\n🔍 Running comparison with bookmaker predictions...")
    try:
        # Run bookmaker comparison
        bm_pred_path = "bm_pred_plus_outcomes.csv"
        if Path(bm_pred_path).exists():
            comparison_summary = compare_with_bookmaker(AJ, bm_pred_path, L)

            # Add comparison results to training summary
            Af['bookmaker_comparison'] = comparison_summary

            # Save updated summary
            with open(x, 'w') as m:
                json.dump(Af, m, indent=2)

            print(f"✅ Bookmaker comparison completed successfully")

            # Print only if model beats bookmaker
            if comparison_summary.get('beats_bookmaker', False):
                print("\n🏆 MODEL BEATS BOOKMAKER ON VALIDATION SET!")
            else:
                print("\n❌ Model does NOT beat bookmaker on validation set.")

            # Create detailed betting analysis
            print(f"\n🔍 Creating detailed betting analysis...")
            detailed_stats = create_detailed_betting_analysis(
                AJ, bm_pred_path, L)

            # Add detailed stats to training summary
            Af['detailed_betting_stats'] = detailed_stats

            # Save updated summary with detailed stats
            with open(x, 'w') as m:
                json.dump(Af, m, indent=2)

            # Print betting-focused summary
            print_betting_focused_summary(
                comparison_summary, detailed_stats, final_val_loss=b[-1])
        else:
            print(f"⚠️ Bookmaker predictions file not found: {bm_pred_path}")
            print(f"💡 Skipping bookmaker comparison")
            # Always create the detailed betting analysis report, even if bookmaker comparison is skipped
            print(f"\n🔍 Creating detailed betting analysis...")
            create_detailed_betting_analysis(AJ, bm_pred_path, L)

    except Exception as Ag:
        print(f"⚠️ Could not run comparison: {Ag}")
        print(f"💡 Check that bm_pred_plus_outcomes.csv exists and has the right format")
        # Always create the detailed betting analysis report, even if bookmaker comparison fails
        print(f"\n🔍 Creating detailed betting analysis...")
        create_detailed_betting_analysis(AJ, bm_pred_path, L)

    # Print final training completion summary
    print(f"\n{'='*80}")
    print(f"🎊 TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"📁 Best model: {L}/best_model.pth")
    print(f"📁 Final model: {L}/last_model.pth")
    print(f"📊 Training summary: {x}")
    print(f"📈 Loss history: {AE}")
    print(f"🎯 Validation predictions: {AJ}")
    print(f"📄 Betting summary report: {L}/betting_summary_report.txt")
    print(f"⏱️  Total training time: {AK/60:.1f} minutes")
    print(f"🏆 Best validation loss: {P:.6f}")
    print(f"📊 Final validation loss: {b[-1]:.6f}")
    print(
        f"🔥 Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")

    print(f"\n🎊 MODEL READY FOR INFERENCE!")
    print(f"📁 Best model saved at: {L}/best_model.pth")
    print(f"📁 Training logs saved at: {L}/")
    print(f"📄 Training summary saved at: {x}")

    # Print feature importances
    print_feature_importance(C, J)


def create_versioned_model_dir():
    """Create a versioned model directory in the models folder"""
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)

    # Find the next version number
    version = 1
    while True:
        versioned_dir = models_dir / f'model_data_v{version}'
        if not versioned_dir.exists():
            versioned_dir.mkdir(parents=True)
            print(
                f"📁 Created versioned model directory: {versioned_dir.absolute()}")
            return versioned_dir
        version += 1


def evaluate_model(model, data_loader, device, criterion):
    """Evaluate classification model on given data loader"""
    model.eval()
    all_logits = []
    all_targets = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for features, targets in data_loader:
            features, targets = features.to(
                device, non_blocking=True), targets.to(device, non_blocking=True)
            logits = model(features)
            loss = criterion(logits, targets)
            total_loss += loss.item() * features.size(0)
            total_samples += features.size(0)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy()
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    target_names = ['spread_win', 'total_win']
    metrics = {}
    from sklearn.metrics import precision_recall_fscore_support
    for i, name in enumerate(target_names):
        accuracy = (preds[:, i] == targets[:, i]).mean() * 100
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets[:, i], preds[:, i], average='binary', zero_division=0)
        metrics[name] = {
            'accuracy_pct': accuracy,
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'mean_prob': probs[:, i].mean()
        }

    avg_loss = total_loss / total_samples
    return metrics, avg_loss, preds, targets


def compare_with_bookmaker(model_predictions_path, bm_predictions_path, checkpoint_dir):
    """Compare binary predictions with bookmaker lines"""
    try:
        model_preds = pd.read_csv(model_predictions_path)
        bm_data = pd.read_csv(bm_predictions_path)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return {}
    sample_size = min(len(model_preds), len(bm_data))
    model_sample = model_preds.iloc[:sample_size].reset_index(drop=True)
    bm_sample = bm_data.iloc[:sample_size].reset_index(drop=True)
    true_spread = (bm_sample["final_spread"] >
                   bm_sample["bm_spread_prediction_at_halftime"]).astype(int)
    true_total = (bm_sample["final_total"] >
                  bm_sample["bm_total_prediction_at_halftime"]).astype(int)
    spread_acc = (model_sample["pred_spread_win"] == true_spread).mean() * 100
    total_acc = (model_sample["pred_total_win"] == true_total).mean() * 100
    overall_acc = (spread_acc + total_acc) / 2
    summary = {"spread_betting_accuracy": float(spread_acc), "total_betting_accuracy": float(
        total_acc), "overall_betting_accuracy": float(overall_acc), "beats_bookmaker": bool(overall_acc > 50)}
    summary_file = Path(checkpoint_dir) / "bm_comparison_summary.json"
    with open(summary_file, "w") as f:
        import json
        json.dump(summary, f, indent=2)
    return summary


def create_detailed_betting_analysis(model_predictions_path, bm_predictions_path, checkpoint_dir):
    """Generate a basic CSV of predictions and outcomes"""
    try:
        model_preds = pd.read_csv(model_predictions_path)
        bm_data = pd.read_csv(bm_predictions_path)
    except Exception as e:
        print(f"❌ Error creating detailed analysis: {e}")
        return {}
    sample_size = min(len(model_preds), len(bm_data))
    model_sample = model_preds.iloc[:sample_size].reset_index(drop=True)
    bm_sample = bm_data.iloc[:sample_size].reset_index(drop=True)
    true_spread = (bm_sample["final_spread"] >
                   bm_sample["bm_spread_prediction_at_halftime"]).astype(int)
    true_total = (bm_sample["final_total"] >
                  bm_sample["bm_total_prediction_at_halftime"]).astype(int)
    out_df = pd.DataFrame({"pred_spread_win": model_sample["pred_spread_win"], "pred_total_win": model_sample[
                          "pred_total_win"], "true_spread_win": true_spread, "true_total_win": true_total})
    out_file = Path(checkpoint_dir) / "detailed_betting_analysis.csv"
    out_df.to_csv(out_file, index=False)
    return {"total_games": len(out_df), "spread_win_rate": float((out_df["pred_spread_win"] == out_df["true_spread_win"]).mean()*100), "total_win_rate": float((out_df["pred_total_win"] == out_df["true_total_win"]).mean()*100)}


def print_betting_focused_summary(comparison_summary, detailed_stats, model_version="v14", final_val_loss=None):
    """Print a betting-focused summary with only bookmaker comparison."""
    # Only print if model beats bookmaker
    if comparison_summary.get('beats_bookmaker', False):
        print("\n" + "="*80)
        print(
            f"🏆 MODEL BEATS BOOKMAKER! ({comparison_summary['model_overall_accuracy']:.2f}% vs {comparison_summary['bookmaker_overall_accuracy']:.2f}%)")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("❌ Model does NOT beat bookmaker.")
        print("="*80)
    # Optionally, print more details if desired
    # ...existing code for profit projections, etc, can be omitted or kept as needed...


def print_feature_importance(model, dataset):
    """
    Print feature importances based on the absolute weights of the first layer.
    No duplicate feature names: sums importances across all rows for each base feature.
    """
    try:
        # Get first Linear layer weights
        first_layer = None
        for layer in model.net:
            if isinstance(layer, nn.Linear):
                first_layer = layer
                break
        if first_layer is None:
            print("⚠️ Could not find first linear layer for feature importance.")
            return

        # shape: (hidden_dim, input_dim)
        weights = first_layer.weight.detach().cpu().numpy()
        importances = np.abs(weights).sum(axis=0)  # shape: (input_dim,)

        # Try to get feature names from the first CSV file
        try:
            first_csv = dataset.csv_files[0]
            df = pd.read_csv(first_csv)
            base_feature_names = df.select_dtypes(
                include=[np.number]).columns.tolist()
        except Exception:
            base_feature_names = [
                f"feature_{i}" for i in range(len(importances))]

        # If input is flattened (multi-row), sum importances for each base feature
        if len(importances) > len(base_feature_names):
            repeats = len(importances) // len(base_feature_names)
            feature_importance_dict = {
                name: 0.0 for name in base_feature_names}
            for i, name in enumerate(base_feature_names * repeats):
                feature_importance_dict[name] += importances[i]
            feature_importance = list(feature_importance_dict.items())
        else:
            feature_importance = list(zip(base_feature_names, importances))

        feature_importance.sort(key=lambda x: x[1], reverse=True)

        print("\n" + "="*80)
        print("🔍 FEATURE IMPORTANCE (by sum of absolute weights in first layer)")
        print("="*80)
        for name, score in feature_importance:
            print(f"{name:40s} {score:10.4f}")
        print("="*80 + "\n")
    except Exception as e:
        print(f"⚠️ Could not compute feature importances: {e}")


# ...existing code...
if __name__ == '__main__':
    import sys

    # Default config file
    config_file = 'training_config.yaml'

    # Check if a config file was provided as argument
    if len(sys.argv) > 1:
        config_file = sys.argv[1]

    # Check if config file exists
    if not Path(config_file).exists():
        print(f"❌ Config file '{config_file}' not found!")
        print(f"💡 Make sure 'training_config.yaml' exists in the current directory")
        print(f"💡 Or provide a config file path as argument: python train.py <config_file>")
        sys.exit(1)

    print(f"🚀 Starting training with config: {config_file}")
    print(f"📁 Working directory: {Path.cwd()}")

    try:
        train(config_file)
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
