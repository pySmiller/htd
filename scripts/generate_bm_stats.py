_I11,_H10,_E35,_B99,_A82='final_spread','file_name','final_total','bm_spread_prediction_at_halftime','bm_total_prediction_at_halftime'
import os,csv,pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
def process_single_file(file_info):
	N='elapsed_time_perc';A,S=file_info;T=S/Path(A)
	try:
		D=pd.read_csv(T)
		if D.empty:print(f"Warning: {A} is empty");return{_H10:A,_E35:'',_I11:'',_A82:'',_B99:''}
		O=D.iloc[-1];I=O['home_score'];J=O['away_score']
		if pd.isna(I)or pd.isna(J):print(f"Warning: {A} has missing scores in final row");return{_H10:A,_E35:'',_I11:'',_A82:'',_B99:''}
		E=float(I)+float(J);F=float(I)-float(J);K=D[D[N]<51.]
		if K.empty:print(f"Warning: {A} has no rows with elapsed_time_perc < 51");B='';C=''
		else:
			P=K.loc[K[N].idxmax()];B=P['live_bm_total_line'];C=P['live_bm_spread_line']
			if pd.isna(B):B=''
			if pd.isna(C):C=''
		return{_H10:A,_E35:E,_I11:F,_A82:B,_B99:C}
	except Exception as U:print(f"Error processing {A}: {str(U)}");return{_H10:A,_E35:'',_I11:'',_A82:'',_B99:''}
def generate_bm_stats():
	I=False;D='coerce';J=Path('C:/Users/admin/Desktop/training/data');K=Path('C:/Users/admin/Desktop/training/outcomes.csv');B=[A for A in os.listdir(J)if A.endswith('.csv')and A!='validation.csv'];B.sort();print(f"Processing {len(B)} CSV files using 16 workers for BM stats...");Q=[(A,J)for A in B];E=[]
	with ProcessPoolExecutor(max_workers=16)as R:
		L={R.submit(process_single_file,A):A[0]for A in Q};F=0
		for M in L:
			try:
				S=M.result();E.append(S);F+=1
				if F%100==0:print(f"Processed {F}/{len(B)} files...")
			except Exception as T:N=L[M];print(f"Error processing {N}: {str(T)}");E.append({_H10:N,_E35:'',_I11:'',_A82:'',_B99:''})
	E.sort(key=lambda x:x[_H10]);A=pd.DataFrame(E);A.to_csv(K,index=I);print(f"\nBM Stats generation complete!");print(f"Processed {len(B)} files");print(f"Results saved to: {K}");print('\nSample results:');print(A.head(10).to_string(index=I));C=A[A[_E35]!=''];print(f"\nFiles with valid data: {len(C)}");print(f"Files with missing data: {len(A)-len(C)}")
	if len(C)>0:
		print(f"Average final total: {C[_E35].mean():.1f}");print(f"Average final spread: {C[_I11].mean():.1f}");G=A[A[_A82]!=''];H=A[A[_B99]!=''];print(f"Files with valid live BM total lines: {len(G)}");print(f"Files with valid live BM spread lines: {len(H)}")
		if len(G)>0:
			print(f"Average live BM total line: {pd.to_numeric(G[_A82],errors=D).mean():.1f}")
		if len(H)>0:
			print(f"Average live BM spread line: {pd.to_numeric(H[_B99],errors=D).mean():.1f}")
if __name__=='__main__':generate_bm_stats()