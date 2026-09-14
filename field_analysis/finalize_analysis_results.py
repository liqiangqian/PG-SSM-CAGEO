import json, hashlib, math, os, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT=Path(__file__).resolve().parents[1]
if '--strict' in sys.argv:
    print('WARNING: --strict is a superseded partition check and is not the manuscript protocol.', file=sys.stderr)
RESULT_ROOT=Path(os.environ.get('PGSSM_OUTPUT_DIR', ROOT/'private_field_results'))
OUT=RESULT_ROOT/'analysis_results_locked'
DATA=Path(os.environ.get('PGSSM_FIELD_DATA_DIR', ROOT/'private_field_data'))
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def erf(x): return np.vectorize(math.erf)(x)
def scores(y,mu,sig,den):
 sig=np.maximum(sig,1e-8); z=(y-mu)/sig; phi=np.exp(-.5*z*z)/np.sqrt(2*np.pi); Phi=.5*(1+erf(z/np.sqrt(2)))
 lo=mu-1.645*sig; hi=mu+1.645*sig; hit=(y>=lo)&(y<=hi); width=np.mean(hi-lo)
 wink=np.where(y<lo,(hi-lo)+20*(lo-y),np.where(y>hi,(hi-lo)+20*(y-hi),hi-lo))
 crps=np.mean(sig*(z*(2*Phi-1)+2*phi-1/np.sqrt(np.pi))); nll=np.mean(.5*np.log(2*np.pi*sig**2)+.5*z*z)
 return {'n':len(y),'RMSE':float(np.sqrt(np.mean((y-mu)**2))),'MAE':float(np.mean(np.abs(y-mu))),'R2':float(r2_score(y,mu)),'MASE':float(np.mean(np.abs(y-mu))/den),'CRPS':float(crps),'NLL':float(nll),'PI90_hits':int(hit.sum()),'PI90_coverage':float(hit.mean()),'PI90_width':float(width),'Winkler90':float(np.mean(wink)),'negative_support_mass':float(np.mean(.5*(1+erf((-mu/sig)/np.sqrt(2))))) }
def main():
 p= pd.read_csv(OUT/'test_predictions_seed11.csv'); p.Date=pd.to_datetime(p.Date)
 # The raw workbook has no target observation on these two dates; remove them from every score.
 missing=pd.to_datetime(['2024-10-12','2024-10-13']); valid=p[~p.Date.isin(missing)].copy()
 raw=pd.read_parquet(DATA/'five_wells_timeseries_clean.parquet'); center=pd.read_csv(DATA/'five_wells_info.csv')['well_id'].astype(str).iloc[0]; yall=raw[center]['U/mg/l']
 den=float(np.mean(np.abs(np.diff(yall.iloc[:347].to_numpy()))))
 rows=[]
 for m,g in valid.groupby('model'):
  g=g.sort_values('Date'); sc=scores(g.Observed.to_numpy(),g.Predicted.to_numpy(),((g.Upper90-g.Lower90)/3.29).to_numpy(),den); rows.append({'model':m,**sc})
 # persistence at the same dated endpoints, using the concentration known at t = target date - H.
 persist=[]
 for d in valid[valid.model=='full'].Date:
  origin=d-pd.Timedelta(days=7); persist.append(float(yall.loc[origin]))
 g=valid[valid.model=='full'].sort_values('Date'); py=np.asarray(persist); sy=np.full(len(py),float(np.std(yall.iloc[:347].to_numpy())))
 rows.append({'model':'persistence',**scores(g.Observed.to_numpy(),py,sy,den)})
 pd.DataFrame(rows).to_csv(OUT/'final_metrics_valid.csv',index=False)
 # Paired block bootstrap, full PG-SSM versus persistence; negative values favour PG-SSM only for squared-error reduction.
 fy=g.Observed.to_numpy()-g.Predicted.to_numpy(); pyerr=g.Observed.to_numpy()-py; n=len(fy); rng=np.random.default_rng(20260912); draws=[]
 for _ in range(5000):
  ids=[]
  while len(ids)<n:
   s=int(rng.integers(0,n)); ids.extend([(s+j)%n for j in range(7)])
  ids=np.asarray(ids[:n]); draws.append([float(np.mean(pyerr[ids]**2-fy[ids]**2)),float(np.mean(np.abs(pyerr[ids])-np.abs(fy[ids])))])
 draws=np.asarray(draws); est=[float(np.mean(pyerr**2-fy**2)),float(np.mean(np.abs(pyerr)-np.abs(fy)))]
 boot=pd.DataFrame({'comparison':['persistence_minus_full']*2,'metric':['RMSE_squared_difference','MAE_difference'],'estimate':est,'low':[np.quantile(draws[:,0],.025),np.quantile(draws[:,1],.025)],'high':[np.quantile(draws[:,0],.975),np.quantile(draws[:,1],.975)],'block_length_days':[7,7],'repetitions':[5000,5000]})
 boot.to_csv(OUT/'final_paired_block_bootstrap.csv',index=False)
 # Calibration at several nominal levels from the same Gaussian distribution.
 fg=valid[valid.model=='full'].sort_values('Date'); yy=fg.Observed.to_numpy(); mm=fg.Predicted.to_numpy(); ss=((fg.Upper90-fg.Lower90)/3.29).to_numpy(); cal=[]
 for nominal in [.50,.80,.90,.95]:
  z={.50:.67448975,.80:1.28155157,.90:1.64485363,.95:1.95996398}[nominal]; lo=mm-z*ss; hi=mm+z*ss; cal.append({'nominal':nominal,'hits':int(((yy>=lo)&(yy<=hi)).sum()),'n':len(yy),'coverage':float(((yy>=lo)&(yy<=hi)).mean()),'mean_width':float(np.mean(hi-lo))})
 pd.DataFrame(cal).to_csv(OUT/'final_calibration_valid.csv',index=False)
 # Stage audit: mutually exclusive target-history slope classes, assigned causally at each origin.
 stage=[]
 for _,r in fg.iterrows():
  origin=r.Date-pd.Timedelta(days=7); hist=yall.loc[:origin].tail(7).to_numpy(); slope=float(hist[-1]-hist[0]) if len(hist)==7 else np.nan
  lab='rising' if slope>=.10 else ('falling' if slope<=-.10 else 'stable')
  stage.append((lab,r.Observed,r.Predicted,(r.Upper90-r.Lower90)/3.29))
 st=[]
 for lab in ['rising','stable','falling']:
  z=[x for x in stage if x[0]==lab];
  if z:
   yy=np.array([x[1] for x in z]); mm=np.array([x[2] for x in z]); ss=np.array([x[3] for x in z]); st.append({'stage':lab,'n':len(z),'coverage':float(np.mean((yy>=mm-1.645*ss)&(yy<=mm+1.645*ss))),'mean_width':float(np.mean(3.29*ss)),'MAE':float(np.mean(np.abs(yy-mm)))})
 pd.DataFrame(st).to_csv(OUT/'final_stage_metrics.csv',index=False)
 # Residual diagnostics and one-to-one peak audit.
 res=fg.Observed.to_numpy()-fg.Predicted.to_numpy(); acf={}
 for lag in range(1,8): acf[str(lag)]=float(np.corrcoef(res[:-lag],res[lag:])[0,1]) if len(res)>lag else None
 yv=fg.Observed.to_numpy(); mv=fg.Predicted.to_numpy(); thr=float(np.quantile(yv,.90)); pobs=[i for i in range(1,len(yv)-1) if yv[i]>=yv[i-1] and yv[i]>yv[i+1] and yv[i]>=thr]; thrm=float(np.quantile(mv,.90)); ppred=[i for i in range(1,len(mv)-1) if mv[i]>=mv[i-1] and mv[i]>mv[i+1] and mv[i]>=thrm]; matches=[]; used=set()
 for i in pobs:
  cand=[j for j in ppred if j not in used and abs(j-i)<=7]
  if cand: j=min(cand,key=lambda j:abs(j-i)); used.add(j); matches.append({'observed_date':str(fg.iloc[i].Date.date()),'predicted_date':str(fg.iloc[j].Date.date()),'offset_days':int(j-i),'magnitude_error':float(mv[j]-yv[i])})
 diag={'n':len(res),'residual_mean':float(res.mean()),'residual_sd':float(res.std(ddof=1)),'residual_acf_lag1_7':acf,'standardized_mean':float((res/np.maximum(((fg.Upper90-fg.Lower90)/3.29).to_numpy(),1e-8)).mean()),'observed_peaks':len(pobs),'predicted_peaks':len(ppred),'matches':matches,'peak_rule':'local maximum, >=90th percentile, minimum separation 7 d, matching tolerance 7 d'}
 (OUT/'final_diagnostics.json').write_text(json.dumps(diag,ensure_ascii=False,indent=2),encoding='utf-8')
 # Final provenance manifest.
 man=json.loads((OUT/'analysis_manifest.json').read_text(encoding='utf-8')); man.update({'raw_target_observed_days':494,'raw_target_missing_dates':['2024-10-12','2024-10-13'],'valid_scored_target_days':int(len(valid[valid.model=='full'])),'mase_denominator_training_abs_diff':den,'final_scored_file':'final_metrics_valid.csv','bootstrap_file':'final_paired_block_bootstrap.csv','calibration_file':'final_calibration_valid.csv','stage_file':'final_stage_metrics.csv','diagnostics_file':'final_diagnostics.json','evaluation':'one train+validation refit after selection; test parameters remain fixed','evaluation_partition':'locked_73_endpoints'})
 for name in ['metrics_by_seed.csv','metrics_summary.csv','test_predictions_seed11.csv','inference_parameter_sweep.csv','inference_parameter_sweep_note.txt','rolling_origin_metrics.csv','rolling_origin_predictions.csv','final_metrics_valid.csv','final_paired_block_bootstrap.csv','final_calibration_valid.csv','final_stage_metrics.csv','final_diagnostics.json']:
  if (OUT/name).exists():
   man.setdefault('output_sha256',{})[name]=sha(OUT/name)
 man['code_sha256']={
  'execute_r3_experiments.py':sha(ROOT/'field_analysis'/'execute_r3_experiments.py'),
  'inference_parameter_sweep.py':sha(ROOT/'field_analysis'/'inference_parameter_sweep.py'),
  'rolling_origin_audit.py':sha(ROOT/'field_analysis'/'rolling_origin_audit.py'),
  'finalize_analysis_results.py':sha(ROOT/'field_analysis'/'finalize_analysis_results.py')}
 (OUT/'analysis_manifest_final.json').write_text(json.dumps(man,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'valid_n':len(g),'metrics':rows,'bootstrap':boot.to_dict('records'),'calibration':cal,'stage':st,'diagnostics':diag},ensure_ascii=False))
if __name__=='__main__': main()
