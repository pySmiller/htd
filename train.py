#!/usr/bin/env python
_k='tight_threshold'
_j='cuda_peak_memory_mb'
_i='input_features'
_h='model_parameters'
_g='validation_samples'
_f='training_samples'
_e='training_time_minutes'
_d='file_name'
_c='directional_accuracy'
_b='tight_accuracy_pct'
_a='total_epochs'
_Z='mean_target'
_Y='mean_pred'
_X='dropout'
_W='hidden_dims'
_V='batch_size'
_U='device'
_T='target_scaler'
_S='threshold'
_R='rmse'
_Q='mae'
_P='lr'
_O='overall_train_accuracy'
_N='epochs'
_M='model'
_L='training_metrics'
_K='data'
_J=False
_I='overall_val_accuracy'
_H=None
_G='accuracy_pct'
_F='r2'
_E=True
_D='training'
_C='final_total'
_B='final_spread'
_A='validation_metrics'
import argparse,glob,json,gc,random,subprocess,time,numpy as np,pandas as pd,torch,torch.nn as nn,torch.optim as optim,yaml
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from torch.utils.data import Dataset,DataLoader
from torch.amp import autocast,GradScaler
torch.backends.cudnn.enabled=_E
torch.backends.cudnn.benchmark=_E
torch.backends.cudnn.deterministic=_J
class CSVDataset(Dataset):
	def __init__(A,csv_files,outcomes_csv_path,scaler=_H,max_rows=_H,bm_pred_path=_H):
		A.csv_files=csv_files;A.max_rows=max_rows;A.target_scaler=_H;C=pd.read_csv(outcomes_csv_path);A.outcomes={}
		for(D,B)in C.iterrows():A.outcomes[B[_d]]={_B:B[_B],_C:B[_C]}
		
		# Load bookmaker predictions for comparison
		A.bm_predictions={}
		if bm_pred_path and Path(bm_pred_path).exists():
			bm_data=pd.read_csv(bm_pred_path)
			for(D,B)in bm_data.iterrows():
				A.bm_predictions[B['filename']]={
					'bm_total_prediction_at_halftime':B['bm_total_prediction_at_halftime'],
					'bm_spread_prediction_at_halftime':B['bm_spread_prediction_at_halftime']
				}
		
		A.features,A.labels,A.scaler=A._build_tensor(scaler)
	def _determine_dimensions(C):
		print(f"Analyzing all {len(C.csv_files)} files to determine true dimensions...");A=_H;B=0
		for(E,F)in enumerate(C.csv_files):
			if E%1000==0:print(f"  Processed {E}/{len(C.csv_files)} files...")
			G=pd.read_csv(F);D=G.select_dtypes(include=[np.number])
			if A is _H:A=len(D.columns)
			elif A!=len(D.columns):print(f"Warning: Column count mismatch in {Path(F).name}")
			B=max(B,len(D))
		print(f"  Completed analysis. Found max {B} rows across all files.");return A,B
	def _build_tensor(A,scaler):
		V='Robust (outlier-resistant)';U='Robust';N='Standard';E=scaler;H,W=A._determine_dimensions()
		if A.max_rows is _H:A.max_rows=W
		print(f"Data dimensions: {H} columns, max {A.max_rows} rows per game");I,O,file_names=[],[],[]
		for(e,J)in enumerate(A.csv_files):
			X=pd.read_csv(J);B=X.select_dtypes(include=[np.number])
			if len(B.columns)!=H:print(f"Warning: {Path(J).name} has {len(B.columns)} columns, expected {H}");continue
			B=B.fillna(0)
			if len(B)>A.max_rows:B=B.iloc[:A.max_rows]
			elif len(B)<A.max_rows:Y=A.max_rows-len(B);Z=np.zeros((Y,len(B.columns)));a=pd.DataFrame(Z,columns=B.columns);B=pd.concat([B,a],ignore_index=_E)
			b=B.values.flatten();I.append(b);K=Path(J).name;file_names.append(K)
			if K not in A.outcomes:print(f"Warning: Outcome for {K} not found in outcomes.csv, skipping");continue
			P=A.outcomes[K];O.append([P[_B],P[_C]])
		print(f"Successfully processed {len(I)} files");D=np.array(I,dtype=np.float32);C=np.array(O,dtype=np.float32);A.file_names=file_names
		if E is _H:
			print(f"📊 Analyzing feature distribution for optimal normalization...");L={'MinMax':MinMaxScaler(feature_range=(-1,1)),N:StandardScaler(),U:RobustScaler()};c=D.std(axis=0);Q=np.abs(np.mean((D-D.mean(axis=0))**3,axis=0)/(c**3+1e-08));R=np.mean(np.abs(D-np.median(D,axis=0))>3*np.std(D,axis=0));print(f"  Feature statistics:");print(f"    Original range: [{D.min():.3f}, {D.max():.3f}]");print(f"    Mean: {D.mean():.3f}, Std: {D.std():.3f}");print(f"    Skewness (avg): {Q.mean():.3f}");print(f"    Outlier ratio: {R:.3%}")
			if R>.1:E=L[U];M=V
			elif Q.mean()>2:E=L[N];M='Standard (skew-resistant)'
			else:E=L[N];M='Standard (neural network optimized)'
			E.fit(D);print(f"  Selected scaler: {M}")
		F=E.transform(D);print(f"  Normalized feature range: [{F.min():.3f}, {F.max():.3f}]");print(f"  Normalized feature mean: {F.mean():.3f}, std: {F.std():.3f}")
		if not hasattr(A,_T)or A.target_scaler is _H:
			print(f"📊 Analyzing target distribution for optimal normalization...");d=C.max()-C.min();f=C.std();S=np.sum(np.abs(C-np.median(C,axis=0))>3*np.std(C,axis=0))/len(C);print(f"  Target statistics:");print(f"    Original range: [{C.min():.3f}, {C.max():.3f}]");print(f"    Mean: {C.mean(axis=0)}");print(f"    Std: {C.std(axis=0)}");print(f"    Range span: {d:.3f}");print(f"    Outlier ratio: {S:.3%}")
			if S>.15:A.target_scaler=RobustScaler();T=V
			else:A.target_scaler=StandardScaler();T='Standard (regression optimized)'
			A.target_scaler.fit(C);print(f"  Selected target scaler: {T}")
			if hasattr(A.target_scaler,'data_min_'):print(f"  Target scaler min: {A.target_scaler.data_min_}");print(f"  Target scaler max: {A.target_scaler.data_max_}")
			elif hasattr(A.target_scaler,'center_'):print(f"  Target scaler center: {A.target_scaler.center_}");print(f"  Target scaler scale: {A.target_scaler.scale_}")
		G=A.target_scaler.transform(C);print(f"  Normalized target range: [{G.min():.3f}, {G.max():.3f}]");print(f"  Normalized target mean: {G.mean(axis=0)}");print(f"  Normalized target std: {G.std(axis=0)}");return torch.from_numpy(F),torch.from_numpy(G),E
	def __len__(A):return A.labels.size(0)
	def __getitem__(A,idx):return A.features[idx],A.labels[idx]
class MLP(nn.Module):
	def __init__(A,in_dim,hidden_dims,out_dim,dropout):
		super().__init__();B,C=[],in_dim
		for D in hidden_dims:B+=[nn.Linear(C,D),nn.ReLU(),nn.Dropout(dropout)];C=D
		B.append(nn.Linear(C,out_dim));A.net=nn.Sequential(*B);A.apply(A._init_weights)
	def _init_weights(B,module):
		A=module
		if isinstance(A,nn.Linear):
			nn.init.kaiming_uniform_(A.weight,mode='fan_in',nonlinearity='relu')
			if A.bias is not _H:nn.init.constant_(A.bias,0)
	def forward(A,x):return A.net(x)
def set_seed(seed=42):A=seed;random.seed(A);np.random.seed(A);torch.manual_seed(A);torch.cuda.manual_seed_all(A)

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
	test_files = [file_index_pairs[i][0] for i in range(split_point, len(file_index_pairs))]
	
	print(f"🕐 TEMPORAL SPLIT APPLIED:")
	print(f"  Training period: {extract_timestamp(train_files[0])} to {extract_timestamp(train_files[-1])}")
	print(f"  Validation period: {extract_timestamp(test_files[0])} to {extract_timestamp(test_files[-1])}")
	print(f"  ✅ NO TEMPORAL LEAKAGE: All training data comes before validation data")
	
	return X_train, X_test, y_train, y_test

def train(cfg_path):
	AQ='final_val_loss';AP='final_train_loss';AO='val_loss';AN='train_loss';AM='early_stopping';AL='weight_decay';A3='feature_scaler';A2='config';A1='out_dim';A0='in_dim';z=1.;l='epoch';k='scheduler';j='optimizer';i='num_workers';E='cuda';AR=time.time()
	with open(cfg_path)as m:A=yaml.safe_load(m)
	if A.get(_U,E)==E:
		if not torch.cuda.is_available():raise RuntimeError('CUDA is not available! Install PyTorch with CUDA support or set device to "cpu" in config.')
		B=torch.device(E);print(f"🚀 Using CUDA device: {torch.cuda.get_device_name(0)}");print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB");print(f"🔧 CUDA Version: {torch.version.cuda}");print(f"🔧 PyTorch Version: {torch.__version__}");print(f"🔧 CUDA Device Count: {torch.cuda.device_count()}");print(f"🔧 Current CUDA Device: {torch.cuda.current_device()}");n=torch.tensor([z,2.,3.]).cuda();print(f"✅ CUDA Test: {n.device} - {n.sum().item()}");print(f"🔧 cuDNN enabled: {torch.backends.cudnn.enabled}");print(f"🔧 cuDNN benchmark: {torch.backends.cudnn.benchmark}");print(f"🔧 cuDNN version: {torch.backends.cudnn.version()}");print(f"🔧 cuDNN deterministic: {torch.backends.cudnn.deterministic}");del n;torch.cuda.empty_cache()
	else:B=torch.device('cpu');print('⚠️  Using CPU for training')
	AS=Path(A[_K]['train_dir']);AT=glob.glob(str(AS/'*.csv'));A4=A[_K]['outcomes_csv'];AU=pd.read_csv(A4);AV=set(AU[_d].values);o=[A for A in AT if Path(A).name in AV]
	
	# Sort files chronologically to prevent temporal leakage
	def extract_timestamp_for_sort(filepath):
		try:
			filename = Path(filepath).name
			timestamp_str = filename.replace('game_', '').replace('.csv', '')
			return timestamp_str
		except:
			return filename
	
	o.sort(key=extract_timestamp_for_sort)
	print(f"Found {len(o)} CSV files with matching outcomes (sorted chronologically)")
	print(f"📅 Data spans from {Path(o[0]).name} to {Path(o[-1]).name}")
	
	if not o:raise ValueError('No CSV files found with matching outcomes!')
	
	# Load dataset with bookmaker predictions
	bm_pred_path = "bm_pred_plus_outcomes.csv"
	J=CSVDataset(o,A4,bm_pred_path=bm_pred_path if Path(bm_pred_path).exists() else None);N=J.features.shape[1];R=2
	
	# Use temporal split instead of random split to prevent data leakage
	X,Y,S,T = temporal_train_test_split(J.features, J.labels, J.file_names, test_size=0.2);p=torch.utils.data.TensorDataset(X,S);q=torch.utils.data.TensorDataset(Y,T);print(f"🔍 DATA DEBUGGING:");print(f"  Training features shape: {X.shape}");print(f"  Training targets shape: {S.shape}");print(f"  Validation features shape: {Y.shape}");print(f"  Validation targets shape: {T.shape}");print(f"  Training features range: [{X.min():.3f}, {X.max():.3f}]");print(f"  Training targets range: [{S.min():.3f}, {S.max():.3f}]");print(f"  Validation features range: [{Y.min():.3f}, {Y.max():.3f}]");print(f"  Validation targets range: [{T.min():.3f}, {T.max():.3f}]");print(f"  Training targets mean: {S.mean(axis=0)}");print(f"  Validation targets mean: {T.mean(axis=0)}");O=DataLoader(p,batch_size=A[_K][_V],shuffle=_E,num_workers=A[_K][i],pin_memory=_E if B.type==E else _J,persistent_workers=_J,drop_last=_E,prefetch_factor=2 if A[_K][i]>0 else _H);A5=DataLoader(q,batch_size=A[_K][_V],shuffle=_J,num_workers=A[_K][i],pin_memory=_E if B.type==E else _J,persistent_workers=_J,drop_last=_J,prefetch_factor=2 if A[_K][i]>0 else _H);C=MLP(N,A[_M][_W],R,A[_M][_X]).to(B);AW=sum(A.numel()for A in C.parameters());print(f"🧠 Model has {AW:,} parameters");print(f"🔧 Model device: {next(C.parameters()).device}")
	if B.type==E:
		print(f"📊 Model memory usage: {torch.cuda.memory_allocated()/1024**2:.1f} MB");print(f"📊 GPU memory reserved: {torch.cuda.memory_reserved()/1024**2:.1f} MB");r=torch.randn(1,N).to(B)
		with torch.no_grad():A6=C(r)
		print(f"✅ CUDA forward pass test: input device={r.device}, output device={A6.device}");s=torch.randn(1000,1000).to(B);t=torch.mm(s,s.t());print(f"✅ CUDA matrix multiplication test: {t.device}, result sum: {t.sum().item():.2f}");del r,A6,s,t;torch.cuda.empty_cache();print(f"🔧 GPU Memory after test: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
	if A[_D][j].lower()=='sgd':I=optim.SGD(C.parameters(),lr=A[_D][_P],weight_decay=A[_D][AL])
	else:I=optim.Adam(C.parameters(),lr=A[_D][_P],weight_decay=A[_D][AL])
	A7=A[_D].get(k,{});Z=optim.lr_scheduler.StepLR(I,step_size=A7.get('step_size',10),gamma=A7.get('gamma',.1));U=nn.MSELoss();a=GradScaler(E)if B.type==E else _H;A8=B.type==E
	if A8:print(f"🚀 Using Automatic Mixed Precision (AMP) training")
	P=float('inf');V=[];b=[];c=A[_D].get(AM,{}).get('patience',50);A9=A[_D].get(AM,{}).get('min_delta',.001);d=0;print(f"\n🚀 Starting training for {A[_D][_N]} epochs");print(f"📊 Training samples: {len(p)}, Validation samples: {len(q)}");print(f"🔧 Training device: {B}");print(f"🛑 Early stopping: patience={c}, min_delta={A9}")
	if B.type==E:print(f"🔧 DataLoader pin_memory: {O.pin_memory}");print(f"🔧 DataLoader num_workers: {O.num_workers}");print(f"🔧 Initial GPU memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB");torch.cuda.set_device(0);torch.cuda.synchronize();print(f"🔧 CUDA context active on device: {torch.cuda.current_device()}")
	
	# Create versioned model directory
	versioned_dir = create_versioned_model_dir()
	print(f"📁 Created versioned model directory: {versioned_dir}")
	
	# Update checkpoint directory to use versioned directory
	A['paths']['checkpoint_dir'] = str(versioned_dir)
	L = versioned_dir  # Set L to versioned directory for all file saving
	
	# Also create the original checkpoints directory for compatibility
	original_checkpoints = Path('./checkpoints')
	original_checkpoints.mkdir(exist_ok=True)
	
	for H in range(1,A[_D][_N]+1):
		if B.type==E:torch.cuda.empty_cache();import gc;gc.collect()
		C.train();e,f=.0,0
		for(g,(F,D))in enumerate(O,1):
			F,D=F.to(B,non_blocking=_E),D.to(B,non_blocking=_E)
			if H==1 and g==1:
				print(f"✅ First batch verification: x.device={F.device}, y.device={D.device}");print(f"✅ Batch shapes: x.shape={F.shape}, y.shape={D.shape}");print(f"✅ First batch x range: [{F.min():.3f}, {F.max():.3f}]");print(f"✅ First batch y range: [{D.min():.3f}, {D.max():.3f}]");print(f"✅ First batch y mean: {D.mean(axis=0)}");print(f"✅ First batch y std: {D.std(axis=0)}")
				if B.type==E:torch.cuda.synchronize();print(f"✅ CUDA synchronized for first batch")
			I.zero_grad()
			if A8:
				with autocast(E):Q=C(F);M=U(Q,D)
				a.scale(M).backward();a.unscale_(I);torch.nn.utils.clip_grad_norm_(C.parameters(),max_norm=z);a.step(I);a.update()
			else:Q=C(F);M=U(Q,D);M.backward();torch.nn.utils.clip_grad_norm_(C.parameters(),max_norm=z);I.step()
			e+=M.item()*F.size(0);f+=D.size(0)
			if g%A['logging']['log_interval']==0:
				AA=I.param_groups[0][_P]
				# Evaluate on validation set for current val loss
				C.eval()
				val_loss = None
				with torch.no_grad():
					val_loss_sum = 0.0
					val_count = 0
					for valF, valD in A5:
						valF, valD = valF.to(B, non_blocking=_E), valD.to(B, non_blocking=_E)
						valQ = C(valF)
						valM = U(valQ, valD)
						val_loss_sum += valM.item() * valF.size(0)
						val_count += valD.size(0)
					val_loss = val_loss_sum / val_count if val_count > 0 else float('nan')
				C.train()
				if B.type==E:
					AX=torch.cuda.memory_allocated()/1024**2
					print(f"Epoch {H} Step {g}/{len(O)} loss {e/f:.4f} val_loss {val_loss:.4f} lr {AA:.6f} GPU mem: {AX:.1f}MB")
				else:
					print(f"Epoch {H} Step {g}/{len(O)} loss {e/f:.4f} val_loss {val_loss:.4f} lr {AA:.6f}")
		W=e/f;V.append(W);Z.step();C.eval();K,AB=.0,0
		if B.type==E:torch.cuda.empty_cache()
		with torch.no_grad():
			for(F,D)in A5:F,D=F.to(B,non_blocking=_E),D.to(B,non_blocking=_E);Q=C(F);M=U(Q,D);K+=M.item()*F.size(0);AB+=D.size(0);del F,D,Q,M
		K=K/AB;b.append(K)
		if H%10==0 or H==A[_D][_N]:
			if B.type==E:AY=torch.cuda.memory_allocated()/1024**2;AZ=torch.cuda.memory_reserved()/1024**2;print(f"📈 Epoch {H:3d}/{A[_D][_N]} - Train Loss: {W:.6f} | Val Loss: {K:.6f} | LR: {I.param_groups[0][_P]:.6f} | GPU: {AY:.0f}MB/{AZ:.0f}MB")
			else:print(f"📈 Epoch {H:3d}/{A[_D][_N]} - Train Loss: {W:.6f} | Val Loss: {K:.6f} | LR: {I.param_groups[0][_P]:.6f}")
		if K<P and H>1 and W<K*10:
			Aa=P-K
			if Aa>A9:P=K;d=0;L=Path(A['paths']['checkpoint_dir']);L.mkdir(exist_ok=_E);Ab=L/'best_model.pth';torch.save({_M:C.state_dict(),j:I.state_dict(),k:Z.state_dict(),l:H,'best_loss':P,A0:N,A1:R,A2:A,A3:J.scaler,_T:J.target_scaler},Ab)
			else:d+=1
		else:d+=1
		if d>=c:print(f"\n🛑 Early stopping triggered after {c} epochs without improvement");print(f"📈 Best validation loss: {P:.6f} at epoch {H-c}");break
		if H%A[_D].get('save_checkpoint_every',5)==0:AC=L/f"checkpoint_epoch_{H}.pth";torch.save({_M:C.state_dict(),j:I.state_dict(),k:Z.state_dict(),l:H,AN:W,AO:K,A0:N,A1:R,A2:A,A3:J.scaler,_T:J.target_scaler},AC);print(f"💾  Saved checkpoint → {AC}")
	AD=L/'last_model.pth';torch.save({_M:C.state_dict(),j:I.state_dict(),k:Z.state_dict(),l:A[_D][_N],AP:V[-1],AQ:b[-1],A0:N,A1:R,A2:A,A3:J.scaler,_T:J.target_scaler},AD);print(f"💾  Saved final model → {AD}");AE=L/'loss_history.csv';Ac=pd.DataFrame({l:range(1,len(V)+1),AN:V,AO:b});Ac.to_csv(AE,index=_J);print(f"📊  Saved loss history → {AE}");print(f"\n🔍 COMPREHENSIVE MODEL EVALUATION");print('='*60);print('📊 Evaluating on validation set...');u,Ah,AF,AG=evaluate_model(C,A5,B,J.target_scaler,U);print('📊 Evaluating on training set...');v,Ai,Aj,Ak=evaluate_model(C,O,B,J.target_scaler,U);print(f"\n📈 VALIDATION SET METRICS:");print('-'*50)
	for(w,G)in u.items():print(f"{w.upper()}:");print(f"  MAE: {G[_Q]:.3f}");print(f"  RMSE: {G[_R]:.3f}");print(f"  R²: {G[_F]:.4f}");print(f"  Accuracy (±{G[_S]}): {G[_G]:.1f}%");print(f"  Mean Prediction: {G[_Y]:.2f} (Target: {G[_Z]:.2f})");print()
	print(f"📈 TRAINING SET METRICS:");print('-'*50)
	for(w,G)in v.items():print(f"{w.upper()}:");print(f"  MAE: {G[_Q]:.3f}");print(f"  RMSE: {G[_R]:.3f}");print(f"  R²: {G[_F]:.4f}");print(f"  Accuracy (±{G[_S]}): {G[_G]:.1f}%");print(f"  Mean Prediction: {G[_Y]:.2f} (Target: {G[_Z]:.2f})");print()
	AH=np.mean([A[_G]for A in u.values()]);AI=np.mean([A[_G]for A in v.values()]);print(f"🎯 OVERALL ACCURACY:");print(f"  Validation: {AH:.1f}%");print(f"  Training: {AI:.1f}%");AJ=L/'validation_predictions.csv';Ad=pd.DataFrame({'pred_final_spread':AF[:,0],'pred_final_total':AF[:,1],'true_final_spread':AG[:,0],'true_final_total':AG[:,1]});Ad.to_csv(AJ,index=_J);print(f"📊  Saved validation predictions → {AJ}");x=L/'training_summary.json'
	def y(obj):
		A=obj
		if isinstance(A,np.floating):return float(A)
		elif isinstance(A,np.integer):return int(A)
		elif isinstance(A,np.ndarray):return A.tolist()
		elif isinstance(A,dict):return{A:y(B)for(A,B)in A.items()}
		return A
	AK=time.time()-AR;Ae=AK/A[_D][_N];Af={_e:AK/60,'time_per_epoch_seconds':Ae,_a:A[_D][_N],'best_val_loss':float(P),AP:float(V[-1]),AQ:float(b[-1]),_f:len(p),_g:len(q),_h:sum(A.numel()for A in C.parameters()),_i:N,'output_targets':R,_A:y(u),_L:y(v),_I:float(AH),_O:float(AI),_U:str(B),_j:float(torch.cuda.max_memory_allocated()/1024**2)if B.type==E else _H}
	with open(x,'w')as m:json.dump(Af,m,indent=2)
	print(f"📊  Saved training summary → {x}");print(f"\n🔍 Running comparison with bookmaker predictions...")
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
			detailed_stats = create_detailed_betting_analysis(AJ, bm_pred_path, L)
			
			# Add detailed stats to training summary
			Af['detailed_betting_stats'] = detailed_stats
			
			# Save updated summary with detailed stats
			with open(x, 'w') as m:
				json.dump(Af, m, indent=2)
			
			# Print betting-focused summary
			print_betting_focused_summary(comparison_summary, detailed_stats, final_val_loss=b[-1])
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
	print(f"🔥 Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
	print(f"{'='*80}")
	
	print(f"\n🎊 MODEL READY FOR INFERENCE!");print(f"📁 Best model saved at: {L}/best_model.pth");print(f"📁 Training logs saved at: {L}/");print(f"📄 Training summary saved at: {x}")
	
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
			print(f"📁 Created versioned model directory: {versioned_dir.absolute()}")
			return versioned_dir
		version += 1
def evaluate_model(model, data_loader, device, target_scaler, criterion):
	"""Evaluate model on given data loader"""
	model.eval()
	all_predictions = []
	all_targets = []
	total_loss = 0.0
	total_samples = 0
	
	with torch.no_grad():
		for features, targets in data_loader:
			features, targets = features.to(device, non_blocking=True), targets.to(device, non_blocking=True)
			outputs = model(features)
			loss = criterion(outputs, targets)
			total_loss += loss.item() * features.size(0)
			total_samples += features.size(0)
			
			# Convert back to original scale
			pred_denorm = target_scaler.inverse_transform(outputs.cpu().numpy())
			target_denorm = target_scaler.inverse_transform(targets.cpu().numpy())
			
			all_predictions.extend(pred_denorm)
			all_targets.extend(target_denorm)
	
	predictions = np.array(all_predictions)
	targets = np.array(all_targets)
	
	# Calculate metrics for each target
	target_names = ['final_spread', 'final_total']
	metrics = {}
	
	for i, target_name in enumerate(target_names):
		pred = predictions[:, i]
		true = targets[:, i]
		
		mae = mean_absolute_error(true, pred)
		mse = mean_squared_error(true, pred)
		rmse = np.sqrt(mse)
		r2 = r2_score(true, pred)
		
		# Accuracy thresholds
		if 'spread' in target_name:
			threshold = 3.0
			tight_threshold = 1.5
		else:
			threshold = 3.0
			tight_threshold = 1.5
		
		within_threshold = np.abs(pred - true) <= threshold
		within_tight = np.abs(pred - true) <= tight_threshold
		
		accuracy = np.mean(within_threshold) * 100
		tight_accuracy = np.mean(within_tight) * 100
		
		# Directional accuracy
		directional_correct = np.sign(pred) == np.sign(true)
		directional_accuracy = np.mean(directional_correct) * 100
		
		metrics[target_name] = {
			'mae': mae,
			'mse': mse,
			'rmse': rmse,
			'r2': r2,
			'accuracy_pct': accuracy,
			'tight_accuracy_pct': tight_accuracy,
			'directional_accuracy': directional_accuracy,
			'threshold': threshold,
			'tight_threshold': tight_threshold,
			'mean_pred': np.mean(pred),
			'mean_target': np.mean(true),
			'std_pred': np.std(pred),
			'std_target': np.std(true)
		}
	
	avg_loss = total_loss / total_samples
	return metrics, avg_loss, predictions, targets
def compare_with_bookmaker(model_predictions_path, bm_predictions_path, checkpoint_dir):
	"""
	Compare model predictions with bookmaker predictions - streamlined for betting focus
	"""
	
	# Load data
	try:
		model_preds = pd.read_csv(model_predictions_path)
		bm_data = pd.read_csv(bm_predictions_path)
		
	except Exception as e:
		print(f"❌ Error loading data: {e}")
		return {}
	
	# Match sample sizes
	if len(model_preds) > len(bm_data):
		model_sample = model_preds.sample(n=len(bm_data), random_state=42).reset_index(drop=True)
		bm_sample = bm_data.copy()
	else:
		bm_sample = bm_data.sample(n=len(model_preds), random_state=42).reset_index(drop=True)
		model_sample = model_preds.copy()
	
	# Calculate betting accuracy - most important metric
	def calculate_betting_accuracy(model_pred_col, bm_pred_col, actual_col):
		"""Calculate accuracy of betting decisions (which side to bet on)"""
		model_preds = model_sample[model_pred_col].values
		bm_preds = bm_sample[bm_pred_col].values
		actual_vals = bm_sample[actual_col].values
		
		# Which side should we bet on?
		model_over_bm = model_preds > bm_preds
		actual_over_bm = actual_vals > bm_preds
		
		# How often did we pick the right side?
		correct_bets = (model_over_bm == actual_over_bm).sum()
		return (correct_bets / len(model_sample)) * 100
	
	# Calculate prediction accuracy for comparison
	def calculate_accuracy(pred_col, actual_col, threshold=3.0, is_model=True):
		"""Calculate accuracy within threshold"""
		if is_model:
			preds = model_sample[pred_col].values
		else:
			preds = bm_sample[pred_col].values
		actual = bm_sample[actual_col].values
		
		diff = np.abs(preds - actual)
		return (diff <= threshold).mean() * 100
	
	# Key metrics
	spread_betting_acc = calculate_betting_accuracy('pred_final_spread', 'bm_spread_prediction_at_halftime', 'final_spread')
	total_betting_acc = calculate_betting_accuracy('pred_final_total', 'bm_total_prediction_at_halftime', 'final_total')
	
	model_spread_acc = calculate_accuracy('pred_final_spread', 'final_spread', threshold=3.0, is_model=True)
	bm_spread_acc = calculate_accuracy('bm_spread_prediction_at_halftime', 'final_spread', threshold=3.0, is_model=False)
	
	model_total_acc = calculate_accuracy('pred_final_total', 'final_total', threshold=5.0, is_model=True)
	bm_total_acc = calculate_accuracy('bm_total_prediction_at_halftime', 'final_total', threshold=5.0, is_model=False)
	
	# Overall performance
	model_overall = (model_spread_acc + model_total_acc) / 2
	bm_overall = (bm_spread_acc + bm_total_acc) / 2
	overall_betting_acc = (spread_betting_acc + total_betting_acc) / 2
	
	# High confidence opportunities
	spread_diff = np.abs(model_sample['pred_final_spread'] - bm_sample['bm_spread_prediction_at_halftime'])
	total_diff = np.abs(model_sample['pred_final_total'] - bm_sample['bm_total_prediction_at_halftime'])
	
	high_conf_spread = spread_diff > 5.0
	high_conf_total = total_diff > 8.0
	
	spread_opportunities = high_conf_spread.sum()
	total_opportunities = high_conf_total.sum()
	
	# Save detailed comparison
	comparison_file = Path(checkpoint_dir) / 'bookmaker_comparison.csv'
	comparison_data = pd.DataFrame({
		'model_pred_spread': model_sample['pred_final_spread'],
		'model_pred_total': model_sample['pred_final_total'],
		'bm_pred_spread': bm_sample['bm_spread_prediction_at_halftime'],
		'bm_pred_total': bm_sample['bm_total_prediction_at_halftime'],
		'actual_spread': bm_sample['final_spread'],
		'actual_total': bm_sample['final_total']
	})
	comparison_data.to_csv(comparison_file, index=False)
	
	# Create summary report
	summary = {
		'model_spread_accuracy': float(model_spread_acc),
		'bookmaker_spread_accuracy': float(bm_spread_acc),
		'model_total_accuracy': float(model_total_acc),
		'bookmaker_total_accuracy': float(bm_total_acc),
		'model_overall_accuracy': float(model_overall),
		'bookmaker_overall_accuracy': float(bm_overall),
		'spread_betting_accuracy': float(spread_betting_acc),
		'total_betting_accuracy': float(total_betting_acc),
		'overall_betting_accuracy': float(overall_betting_acc),
		'spread_advantage': float(model_spread_acc - bm_spread_acc),
		'total_advantage': float(model_total_acc - bm_total_acc),
		'overall_advantage': float(model_overall - bm_overall),
		'high_confidence_spread_opportunities': int(spread_opportunities),
		'high_confidence_total_opportunities': int(total_opportunities),
		'games_analyzed': int(len(model_sample)),
		'above_coinflip': bool(overall_betting_acc > 50),
		'beats_bookmaker': bool(model_overall > bm_overall),
		'betting_viable': bool(overall_betting_acc > 52.5)
	}
	
	summary_file = Path(checkpoint_dir) / 'bm_comparison_summary.json'
	with open(summary_file, 'w') as f:
		import json
		json.dump(summary, f, indent=2)
	
	return summary
def create_detailed_betting_analysis(model_predictions_path, bm_predictions_path, checkpoint_dir):
	"""
	Create a detailed betting analysis showing each individual bet with comprehensive details
	"""
	print(f"\n📊 Creating detailed betting analysis...")
	
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
		
		# Create summary statistics with enhanced betting metrics
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
			'model_beats_bookmaker_rate': detailed_df['model_better_than_bookmaker'].mean() * 100,
			
			# Enhanced profitability metrics
			'spread_roi_183x': ((detailed_df['spread_bet_won'].mean() * 1.83) - 1) * 100,
			'total_roi_183x': ((detailed_df['total_bet_won'].mean() * 1.83) - 1) * 100,
			'breakeven_rate_183x': (1 / 1.83) * 100,
			'spread_edge': detailed_df['spread_bet_won'].mean() * 100 - (1 / 1.83) * 100,
			'total_edge': detailed_df['total_bet_won'].mean() * 100 - (1 / 1.83) * 100,
			
			# Streak analysis
			'max_spread_win_streak': 0,
			'max_spread_loss_streak': 0,
			'max_total_win_streak': 0,
			'max_total_loss_streak': 0,
			
			# Confidence level analysis
			'very_high_conf_spread_bets': (detailed_df['spread_confidence_level'] > 10.0).sum(),
			'very_high_conf_total_bets': (detailed_df['total_confidence_level'] > 15.0).sum(),
			'low_conf_spread_bets': (detailed_df['spread_confidence_level'] < 2.0).sum(),
			'low_conf_total_bets': (detailed_df['total_confidence_level'] < 3.0).sum(),
		}
		
		# Calculate winning streaks
		spread_wins = detailed_df['spread_bet_won'].values
		total_wins = detailed_df['total_bet_won'].values
		
		def calculate_streaks(wins):
			max_win_streak = max_loss_streak = current_win_streak = current_loss_streak = 0
			for win in wins:
				if win:
					current_win_streak += 1
					current_loss_streak = 0
					max_win_streak = max(max_win_streak, current_win_streak)
				else:
					current_loss_streak += 1
					current_win_streak = 0
					max_loss_streak = max(max_loss_streak, current_loss_streak)
			return max_win_streak, max_loss_streak
		
		summary_stats['max_spread_win_streak'], summary_stats['max_spread_loss_streak'] = calculate_streaks(spread_wins)
		summary_stats['max_total_win_streak'], summary_stats['max_total_loss_streak'] = calculate_streaks(total_wins)
		
		# Save detailed analysis
		detailed_file = Path(checkpoint_dir) / 'detailed_betting_analysis.csv'
		detailed_df.to_csv(detailed_file, index=False)
		
		# Create a comprehensive summary report
		summary_file = Path(checkpoint_dir) / 'betting_summary_report.txt'
		with open(summary_file, 'w', encoding='utf-8') as f:
			f.write("="*80 + "\n")
			f.write("COMPREHENSIVE BETTING ANALYSIS REPORT\n")
			f.write("="*80 + "\n\n")
			
			f.write(f"PROFITABILITY ANALYSIS (1.83x Odds)\n")
			f.write(f"{'─'*50}\n")
			f.write(f"Break-even rate needed: {summary_stats['breakeven_rate_183x']:.1f}%\n")
			f.write(f"Spread betting edge: {summary_stats['spread_edge']:+.1f}%\n")
			f.write(f"Total betting edge: {summary_stats['total_edge']:+.1f}%\n")
			f.write(f"Spread ROI: {summary_stats['spread_roi_183x']:+.1f}%\n")
			f.write(f"Total ROI: {summary_stats['total_roi_183x']:+.1f}%\n\n")
			
			f.write(f"OVERALL STATISTICS:\n")
			f.write(f"{'─'*50}\n")
			f.write(f"Total Games Analyzed: {summary_stats['total_games']}\n")
			f.write(f"Model vs Bookmaker Win Rate: {summary_stats['model_beats_bookmaker_rate']:.1f}%\n\n")
			
			f.write(f"SPREAD BETTING PERFORMANCE:\n")
			f.write(f"{'─'*50}\n")
			f.write(f"Spread Bets Won: {summary_stats['spread_bets_won']}/{summary_stats['total_games']}\n")
			f.write(f"Spread Win Rate: {summary_stats['spread_bet_win_rate']:.1f}%\n")
			f.write(f"Average Confidence: {summary_stats['avg_spread_confidence']:.2f}\n")
			f.write(f"High Confidence Bets (>5.0): {summary_stats['high_confidence_spread_bets']}\n")
			f.write(f"Very High Confidence Bets (>10.0): {summary_stats['very_high_conf_spread_bets']}\n")
			f.write(f"Low Confidence Bets (<2.0): {summary_stats['low_conf_spread_bets']}\n")
			f.write(f"Max Win Streak: {summary_stats['max_spread_win_streak']}\n")
			f.write(f"Max Loss Streak: {summary_stats['max_spread_loss_streak']}\n\n")
			
			f.write(f"TOTALS (O/U) BETTING PERFORMANCE:\n")
			f.write(f"{'─'*50}\n")
			f.write(f"Total Bets Won: {summary_stats['total_bets_won']}/{summary_stats['total_games']}\n")
			f.write(f"Total Win Rate: {summary_stats['total_bet_win_rate']:.1f}%\n")
			f.write(f"Average Confidence: {summary_stats['avg_total_confidence']:.2f}\n")
			f.write(f"High Confidence Bets (>8.0): {summary_stats['high_confidence_total_bets']}\n")
			f.write(f"Very High Confidence Bets (>15.0): {summary_stats['very_high_conf_total_bets']}\n")
			f.write(f"Low Confidence Bets (<3.0): {summary_stats['low_conf_total_bets']}\n")
			f.write(f"Max Win Streak: {summary_stats['max_total_win_streak']}\n")
			f.write(f"Max Loss Streak: {summary_stats['max_total_loss_streak']}\n\n")
			
			f.write(f"COMBINED BETTING PERFORMANCE:\n")
			f.write(f"{'─'*50}\n")
			f.write(f"Both Bets Won: {summary_stats['both_bets_won']}/{summary_stats['total_games']}\n")
			f.write(f"Both Bets Win Rate: {summary_stats['both_bets_win_rate']:.1f}%\n")
			f.write(f"Either Bet Won: {summary_stats['either_bet_won']}/{summary_stats['total_games']}\n")
			f.write(f"Either Bet Win Rate: {summary_stats['either_bet_win_rate']:.1f}%\n\n")
			
			# Profitability recommendations
			f.write(f"PROFITABILITY RECOMMENDATIONS:\n")
			f.write(f"{'─'*50}\n")
			if summary_stats['spread_roi_183x'] > 5:
				f.write(f"✅ SPREAD BETTING: PROFITABLE ({summary_stats['spread_roi_183x']:+.1f}% ROI)\n")
				f.write(f"   → Focus on high-confidence bets for best returns\n")
			else:
				f.write(f"❌ SPREAD BETTING: NOT PROFITABLE ({summary_stats['spread_roi_183x']:+.1f}% ROI)\n")
			
			if summary_stats['total_roi_183x'] > 5:
				f.write(f"✅ TOTALS BETTING: PROFITABLE ({summary_stats['total_roi_183x']:+.1f}% ROI)\n")
			else:
				f.write(f"❌ TOTALS BETTING: NOT PROFITABLE ({summary_stats['total_roi_183x']:+.1f}% ROI)\n")
			
			f.write(f"\n")
			
			# Top performing bets
			f.write(f"TOP 10 HIGHEST CONFIDENCE SPREAD BETS:\n")
			f.write(f"{'─'*50}\n")
			top_spread = detailed_df.nlargest(10, 'spread_confidence_level')
			for _, row in top_spread.iterrows():
				f.write(f"Game {row['game_id']}: {row['spread_model_bet']} (Conf: {row['spread_confidence_level']:.1f}) - {'WON' if row['spread_bet_won'] else 'LOST'}\n")
			
			f.write(f"\nTOP 10 HIGHEST CONFIDENCE TOTAL BETS:\n")
			f.write(f"{'─'*50}\n")
			top_total = detailed_df.nlargest(10, 'total_confidence_level')
			for _, row in top_total.iterrows():
				f.write(f"Game {row['game_id']}: {row['total_model_bet']} (Conf: {row['total_confidence_level']:.1f}) - {'WON' if row['total_bet_won'] else 'LOST'}\n")
		
		print(f"✅ Detailed betting analysis saved to: {detailed_file}")
		print(f"✅ Summary report saved to: {summary_file}")
		
		return summary_stats
		
	except Exception as e:
		print(f"❌ Error creating detailed analysis: {e}")
		return {}

def print_betting_focused_summary(comparison_summary, detailed_stats, model_version="v14", final_val_loss=None):
	"""Print a betting-focused summary with only bookmaker comparison."""
	# Only print if model beats bookmaker
	if comparison_summary.get('beats_bookmaker', False):
		print("\n" + "="*80)
		print(f"🏆 MODEL BEATS BOOKMAKER! ({comparison_summary['model_overall_accuracy']:.2f}% vs {comparison_summary['bookmaker_overall_accuracy']:.2f}%)")
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

		weights = first_layer.weight.detach().cpu().numpy()  # shape: (hidden_dim, input_dim)
		importances = np.abs(weights).sum(axis=0)  # shape: (input_dim,)

		# Try to get feature names from the first CSV file
		try:
			first_csv = dataset.csv_files[0]
			df = pd.read_csv(first_csv)
			base_feature_names = df.select_dtypes(include=[np.number]).columns.tolist()
		except Exception:
			base_feature_names = [f"feature_{i}" for i in range(len(importances))]

		# If input is flattened (multi-row), sum importances for each base feature
		if len(importances) > len(base_feature_names):
			repeats = len(importances) // len(base_feature_names)
			feature_importance_dict = {name: 0.0 for name in base_feature_names}
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