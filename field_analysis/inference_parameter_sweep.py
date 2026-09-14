import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0,str(Path(__file__).parent)); import execute_r3_experiments as ex

def main():
    root=Path(__file__).resolve().parents[1]
    out=Path(os.environ.get('PGSSM_OUTPUT_DIR', root/'private_field_results'))/'analysis_results_locked'; out.mkdir(parents=True,exist_ok=True)
    df,arr,y,dist=ex.read_data(); n=len(df); ntr=347; nv=74
    mean=arr[:ntr].mean(0); std=np.where(arr[:ntr].std(0)<1e-6,1,arr[:ntr].std(0)); dates=df.index.to_numpy()
    Xtr,Ytr,_,_=ex.windows_endpoint(arr,y,dates,0,ntr,mean,std); Xv,Yv,_,_=ex.windows_endpoint(arr,y,dates,ntr,ntr+nv,mean,std); Xte,Yte,Dte,_=ex.windows_endpoint(arr,y,dates,ntr+nv,n,mean,std)
    m0,ep,_,delta_rate=ex.fit_model(Xtr,Ytr,Xv,Yv,'full',11,mean,std); m=ex.fit_fixed(np.concatenate([Xtr,Xv]),np.concatenate([Ytr,Yv]),'full',11,ep,mean,std,delta_rate)
    ym,ys=mean[0,0],std[0,0]; yraw=Yte*ys+ym; valid=~pd.to_datetime(Dte).isin(pd.to_datetime(['2024-10-12','2024-10-13']))
    den=float(np.mean(np.abs(np.diff(y[:ntr])))); rows=[]
    for alpha in [0.0,0.3,0.6,0.9]:
      for beta in [0.0,0.25,0.5]:
       for sd in [60.,120.,240.]:
        m.alpha=alpha;m.beta=beta;m.sd=sd
        with torch.no_grad(): mz,lv=m(torch.tensor(Xte))
        mu=ym+ys*mz.numpy(); sig=ys*np.exp(.5*lv.numpy());
        with torch.no_grad():
         _,_,weights=m.graph(torch.tensor(Xte)); mean_incoming=float(weights[valid].sum(-1).mean())
        rows.append({'alpha':alpha,'beta':beta,'sd_m':sd,'mean_normalized_incoming_weight':mean_incoming,**ex.deterministic(yraw[valid],(mu.numpy() if hasattr(mu,'numpy') else mu)[valid],den)})
    pd.DataFrame(rows).to_csv(out/'inference_parameter_sweep.csv',index=False)
    (out/'inference_parameter_sweep_note.txt').write_text('Post-fit inference-stage perturbation of the seed-11 checkpoint; no retraining and no test-based selection. The grid tests computational coefficient dependence. It is not full retraining sensitivity and not hydraulic-parameter calibration.',encoding='utf-8')
    print(pd.DataFrame(rows).sort_values('RMSE').head(8).to_string(index=False))
if __name__=='__main__':main()
