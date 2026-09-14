"""Locked authorized field evaluation for CAGEO-D-26-00782R1.

Sample membership is determined by target date. After validation selection the
model is refit once on training+validation endpoint samples. Test parameters
remain fixed. This script is not a test-period expanding update.
"""
import json, hashlib, math, os, random, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('PGSSM_FIELD_DATA_DIR', ROOT / 'private_field_data'))
RESULT_ROOT = Path(os.environ.get('PGSSM_OUTPUT_DIR', ROOT / 'private_field_results'))
# Deprecated flags are accepted only so older wrappers do not crash.
# The locked protocol is always target-date assignment + one train+validation refit.
if '--strict' in sys.argv:
    print('WARNING: --strict is a superseded partition check and is not the manuscript protocol.', file=sys.stderr)
if '--expanding' in sys.argv:
    print('WARNING: --expanding is a deprecated alias. The locked protocol does not update parameters during test scoring.', file=sys.stderr)
OUT = RESULT_ROOT / 'analysis_results_locked'
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [11]
VARS = ['U/mg/l','Q日抽','Q瞬时','Q累计','U/㎏','工作频率']
L, H = 28, 7
TRAIN_DAYS, VALID_DAYS = 347, 74


def well_roles():
    """Read protected well identifiers without publishing them in the repository."""
    info = pd.read_csv(DATA / 'five_wells_info.csv')
    wells = info['well_id'].astype(str).tolist()[:5]
    if len(wells) < 5:
        raise ValueError('five_wells_info.csv must list the extraction well first, then four injectors.')
    return wells, wells[0], wells[1:]
DEVICE = torch.device('cpu')
torch.set_num_threads(2)

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def read_data():
    wells, center, _injectors = well_roles()
    df = pd.read_parquet(DATA/'five_wells_timeseries_clean.parquet')
    df.index = pd.to_datetime(df.index)
    arr = np.stack([df[w][VARS].to_numpy(float) for w in wells], axis=1) # T,N,F
    y = df[center]['U/mg/l'].to_numpy(float)
    info = pd.read_csv(DATA/'five_wells_info.csv').set_index('well_id')
    xy = info.loc[wells,['X','Y']].to_numpy(float)
    dist = np.sqrt(((xy-xy[0])**2).sum(1)); return df, arr, y, dist

def windows(arr, y, dates, start, end, mean, std):
    # strict partition: every history and endpoint is in [start,end)
    xs, ys, ds, origins = [], [], [], []
    for i in range(start, end-L-H+1):
        xs.append((arr[i:i+L]-mean)/std); ys.append((y[i+L+H-1]-mean[0,0])/std[0,0])
        origins.append(i+L-1); ds.append(dates[i+L+H-1])
    return np.asarray(xs,np.float32), np.asarray(ys,np.float32), np.asarray(ds), np.asarray(origins)

def windows_endpoint(arr, y, dates, endpoint_start, endpoint_end, mean, std):
    # Forecast endpoints lie in [endpoint_start, endpoint_end); histories may use only earlier data.
    xs, ys, ds, origins = [], [], [], []
    first_i=max(0, endpoint_start-L-H+1); last_i=endpoint_end-L-H+1
    for i in range(first_i, last_i):
        xs.append((arr[i:i+L]-mean)/std); ys.append((y[i+L+H-1]-mean[0,0])/std[0,0])
        origins.append(i+L-1); ds.append(dates[i+L+H-1])
    return np.asarray(xs,np.float32), np.asarray(ys,np.float32), np.asarray(ds), np.asarray(origins)

class PGSSM(nn.Module):
    def __init__(self, mode='full', hidden=32, sd=120.0, alpha=0.6, beta=0.25, inj_idx=1, ext_idx=1):
        super().__init__(); self.mode=mode; self.sd=sd; self.alpha=alpha; self.beta=beta
        self.g = nn.Sequential(nn.Linear(12,32), nn.Tanh())
        self.slow = nn.GRU(32, hidden, batch_first=True)
        self.fast = nn.GRU(6, hidden//2, batch_first=True)
        if mode=='single': self.single=nn.GRU(12,hidden,batch_first=True)
        outdim = hidden if mode=='single' else hidden + hidden//2
        self.head=nn.Sequential(nn.Linear(outdim,32),nn.Tanh(),nn.Linear(32,2))
        self.register_buffer('dist', torch.tensor([0.0,102.35,78.00,114.47,300.00]))
    def graph(self,x):
        # x: B,L,N,F; receiving-row convention: center receives injector messages
        center=x[:,:,0,:]
        inj=x[:,:,1:,:]
        if self.mode=='center':
            w=torch.zeros(4,device=x.device)
        elif self.mode=='equal':
            # Four equal injector edges plus a unit self-loop. The self-loop is
            # represented by the separately supplied centre features.
            w=torch.ones(4,device=x.device)/5
        elif self.mode=='distance':
            ww=torch.exp(-(self.dist[1:]**2)/(2*self.sd**2)); w=ww/(1+ww.sum())
        else:
            ww=torch.exp(-(self.dist[1:]**2)/(2*self.sd**2))
            injflow=inj[:,:,:,1].clamp_min(0)
            extflow=center[:,:,1].clamp_min(0).unsqueeze(-1)
            # Standardized flow proxies are bounded by sigmoid. Incoming edges
            # are normalized together with a unit centre self-loop, so the
            # common extraction-flow factor does not cancel algebraically.
            inorm=torch.sigmoid(injflow); enorm=torch.sigmoid(extflow)
            ww=ww.view(1,1,4)*(1+self.alpha*inorm)*(1+self.beta*enorm)
            w=ww/(1+ww.sum(-1,keepdim=True)+1e-6)
            return center, (inj*w.unsqueeze(-1)).sum(2), w
        return center, (inj*w.view(1,1,4,1)).sum(2), w
    def forward(self,x):
        c, m, _ = self.graph(x)
        if self.mode in ('single','single_nog'):
            z=torch.cat([c,m],-1); h,_=self.single(z); q=h[:,-1]
        else:
            z=self.g(torch.cat([c,m],-1)); hs,_=self.slow(z)
            op=torch.cat([c[:,:,:1], c[:,:,1:3], m[:,:,:3]],-1)
            hf,_=self.fast(op); q=torch.cat([hs[:,-1],hf[:,-1]],-1)
        out=self.head(q); mu=out[:,0]; logvar=out[:,1].clamp(-6.0,3.0); return mu, logvar

def loss_fn(mu, lv, y, last, slope7_raw, target_mean, target_std, delta_rate, phys=True):
    nll=0.5*(lv+(y-mu)**2/torch.exp(lv)+math.log(2*math.pi)).mean()
    if not phys: return nll
    # Soft original-scale plausibility terms, including rising-stage guidance.
    mu_raw=mu*target_std+target_mean
    last_raw=last*target_std+target_mean
    delta_raw=mu_raw-last_raw
    neg=torch.relu(-mu_raw).pow(2).mean()
    rate=torch.relu(torch.abs(delta_raw)/H-delta_rate).pow(2).mean()
    rising=slope7_raw>=0.10
    stage=torch.relu(-delta_raw[rising]).pow(2).mean() if torch.any(rising) else mu.new_tensor(0.0)
    return nll + 0.08*neg + 0.05*rate + 0.03*stage

def fit_model(Xtr,ytr,Xv,yv,mode,seed,mean,std,epochs=300):
    seed_all(seed); model=PGSSM(mode=mode).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=2e-3,weight_decay=1e-4)
    xt=torch.tensor(Xtr); yt=torch.tensor(ytr); xv=torch.tensor(Xv); yvt=torch.tensor(yv)
    last_t=torch.tensor(Xtr[:,-1,0,0]); last_v=torch.tensor(Xv[:,-1,0,0])
    slope_t=torch.tensor((Xtr[:,-1,0,0]-Xtr[:,-7,0,0])*std[0,0]); slope_v=torch.tensor((Xv[:,-1,0,0]-Xv[:,-7,0,0])*std[0,0])
    target_mean=float(mean[0,0]); target_std=float(std[0,0])
    delta_rate=float(np.quantile(np.abs((ytr-Xtr[:,-1,0,0])*target_std)/H,0.95))
    best=None; patience=0
    for ep in range(epochs):
        model.train(); opt.zero_grad(); mu,lv=model(xt); loss=loss_fn(mu,lv,yt,last_t,slope_t,target_mean,target_std,delta_rate,phys=(mode!='nophys')); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2); opt.step()
        model.eval();
        with torch.no_grad():
            muv,lvv=model(xv); vl=loss_fn(muv,lvv,yvt,last_v,slope_v,target_mean,target_std,delta_rate,phys=(mode!='nophys')).item()
        if best is None or vl<best[0]-1e-5:
            best=(vl,{k:v.detach().clone() for k,v in model.state_dict().items()}); patience=0
        else: patience+=1
        if patience>=40: break
    model.load_state_dict(best[1]); model.eval(); return model, ep+1, best[0], delta_rate

def fit_fixed(Xtr,ytr,mode,seed,epochs,mean,std,delta_rate):
    seed_all(seed); model=PGSSM(mode=mode).to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=2e-3,weight_decay=1e-4)
    xt=torch.tensor(Xtr); yt=torch.tensor(ytr); last_t=xt[:,-1,0,0]; slope_t=(xt[:,-1,0,0]-xt[:,-7,0,0])*float(std[0,0])
    for _ in range(max(1,int(epochs))):
        model.train(); opt.zero_grad(); mu,lv=model(xt); loss=loss_fn(mu,lv,yt,last_t,slope_t,float(mean[0,0]),float(std[0,0]),delta_rate,phys=(mode!='nophys')); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),2); opt.step()
    model.eval(); return model

def gaussian_scores(y,mu,sig):
    z=(y-mu)/sig; phi=np.exp(-0.5*z*z)/np.sqrt(2*np.pi); Phi=0.5*(1+np.vectorize(math.erf)(z/np.sqrt(2)))
    crps=np.mean(sig*(z*(2*Phi-1)+2*phi-1/np.sqrt(np.pi)))
    nll=np.mean(0.5*np.log(2*np.pi*sig**2)+0.5*((y-mu)/sig)**2)
    lo=mu-1.645*sig; hi=mu+1.645*sig; hit=(y>=lo)&(y<=hi)
    width=np.mean(hi-lo); wink=np.where(y<lo,(hi-lo)+2/0.1*(lo-y),np.where(y>hi,(hi-lo)+2/0.1*(y-hi),hi-lo))
    return {'CRPS':float(crps),'NLL':float(nll),'PI90_coverage':float(hit.mean()),'PI90_hits':int(hit.sum()),'PI90_n':int(len(y)),'PI90_width':float(width),'Winkler90':float(np.mean(wink)),'neg_mass':float(np.mean(0.5*(1+np.vectorize(math.erf)((-mu/sig)/np.sqrt(2)))))}

def deterministic(y,mu,den):
    return {'RMSE':float(mean_squared_error(y,mu)**0.5),'MAE':float(mean_absolute_error(y,mu)),'R2':float(r2_score(y,mu)),'MASE':float(mean_absolute_error(y,mu)/den) if den else np.nan}

def peak_metrics(dates,y,mu):
    # fixed, predeclared event rule: local maxima with >=90th percentile and 7-day separation
    thr=np.quantile(y,0.90); cand=[i for i in range(1,len(y)-1) if y[i]>=y[i-1] and y[i]>y[i+1] and y[i]>=thr]
    peaks=[]
    for i in cand:
        if not peaks or i-peaks[-1]>=7: peaks.append(i)
        elif y[i]>y[peaks[-1]]: peaks[-1]=i
    pt=np.quantile(mu,0.90); pc=[i for i in range(1,len(mu)-1) if mu[i]>=mu[i-1] and mu[i]>mu[i+1] and mu[i]>=pt]
    pp=[]
    for i in pc:
        if not pp or i-pp[-1]>=7: pp.append(i)
        elif mu[i]>mu[pp[-1]]: pp[-1]=i
    matches=[]; used=set()
    for i in peaks:
        cand2=[j for j in pp if j not in used and abs(j-i)<=7]
        if cand2:
            j=min(cand2,key=lambda k:abs(k-i)); used.add(j); matches.append((j-i,mu[j]-y[i],i,j))
    return {'observed_peaks':len(peaks),'predicted_peaks':len(pp),'matched_peaks':len(matches),'PTE_days':float(np.mean([a for a,_,_,_ in matches])) if matches else None,'PME_mg_L':float(np.mean([b for _,b,_,_ in matches])) if matches else None,'peak_threshold_observed':float(thr),'peak_threshold_predicted':float(pt)}

def run():
    df,arr,y,dist=read_data(); dates=df.index.to_numpy()
    n=len(df); ntr=TRAIN_DAYS; nv=VALID_DAYS; bounds={'train':[0,ntr],'validation':[ntr,ntr+nv],'test':[ntr+nv,n]}
    # training-only normalization; target stats equal extraction-well U stats from training rows
    mean=arr[:ntr].mean((0,)); std=arr[:ntr].std((0)); std=np.where(std<1e-6,1,std)
    X={}; Y={}; D={}; O={}
    for split,(a,b) in bounds.items():
        # Sample membership is determined by target date. Histories may cross an
        # earlier calendar partition, but no observation after the origin is used.
        X[split],Y[split],D[split],O[split]=windows_endpoint(arr,y,dates,a,b,mean,std)
    # persistence and within-framework component/graph variants
    modes=['full','nophys','center','equal','distance','single']
    results=[]; pred_records=[]; histories=[]
    all_modes={m:[] for m in modes}
    for mode in modes:
        for seed in SEEDS:
            mode2=mode; Xtr,Xv,Xte=X['train'],X['validation'],X['test']
            model,ep,vl,delta_rate=fit_model(Xtr,Y['train'],Xv,Y['validation'],mode2,seed,mean,std)
            # ONE post-validation train+validation refit. Test parameters stay fixed.
            model=fit_fixed(np.concatenate([Xtr,Xv]),np.concatenate([Y['train'],Y['validation']]),mode2,seed,ep,mean,std,delta_rate)
            with torch.no_grad(): mu_z,lv=model(torch.tensor(Xte)); mu_z=mu_z.numpy(); sig_z=np.exp(0.5*lv.numpy())
            ym=mean[0,0]; ys=std[0,0]; mu=ym+ys*mu_z; sig=ys*sig_z; yt=y[O['test']+H-1] if False else y[(bounds['test'][0]+L+H-1):bounds['test'][1]]
            # Y[test] is standardized endpoint; recover from original dates
            yt=Y['test']*ys+ym
            den=float(np.mean(np.abs(np.diff(y[:ntr]))))
            dm=deterministic(yt,mu,den); gs=gaussian_scores(yt,mu,sig); pk=peak_metrics(D['test'],yt,mu)
            row={'model':mode,'seed':seed,'epochs':ep,'val_loss':vl,'delta_rate_mg_L_day':delta_rate,'parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),**dm,**gs,**pk}
            results.append(row); all_modes[mode].append((yt,mu,sig,D['test']))
            if seed==SEEDS[0]:
                for i,(d,obs,pr,s) in enumerate(zip(D['test'],yt,mu,sig)): pred_records.append({'model':mode,'seed':seed,'Date':str(pd.Timestamp(d).date()),'Observed':float(obs),'Predicted':float(pr),'Lower90':float(pr-1.645*s),'Upper90':float(pr+1.645*s),'Residual':float(obs-pr),'Sigma':float(s)})
    # persistence baseline on the same targets: last concentration in each history
    yp=Y['test']*std[0,0]+mean[0,0]
    last=np.array([arr[i,0,0] for i in O['test']])
    den=float(np.mean(np.abs(np.diff(y[:ntr]))))
    row={'model':'persistence','seed':0,'epochs':0,'val_loss':np.nan,'delta_rate_mg_L_day':np.nan,'parameters':0,**deterministic(yp,last,den),**gaussian_scores(yp,last,np.full(len(last),np.std(Y['train'])*std[0,0])) ,**peak_metrics(D['test'],yp,last)}
    results.append(row)
    rdf=pd.DataFrame(results); rdf.to_csv(OUT/'metrics_by_seed.csv',index=False)
    # Aggregate mean/sd over seeds for model controls
    agg=rdf[rdf.model!='persistence'].groupby('model').agg({c:['mean','std'] for c in ['RMSE','MAE','R2','MASE','CRPS','NLL','PI90_coverage','PI90_width','Winkler90','PTE_days','PME_mg_L']}).reset_index()
    agg.to_csv(OUT/'metrics_summary.csv',index=False)
    pd.DataFrame(pred_records).to_csv(OUT/'test_predictions_seed11.csv',index=False)
    # bootstrap paired RMSE/MAE differences on seed 11 full vs controls
    p11=pd.DataFrame(pred_records); full=p11[p11.model=='full'].sort_values('Date')
    boot=[]; rng=np.random.default_rng(20260912); ntest=len(full)
    for m in modes:
        if m=='full': continue
        oth=p11[p11.model==m].sort_values('Date'); d_rm=(full.Predicted.to_numpy()-full.Observed.to_numpy())**2-(oth.Predicted.to_numpy()-oth.Observed.to_numpy())**2; d_ma=np.abs(full.Predicted-full.Observed).to_numpy()-np.abs(oth.Predicted-oth.Observed).to_numpy()
        for b in range(2000):
            # moving blocks preserve short serial dependence
            inds=[]
            while len(inds)<ntest:
                st=int(rng.integers(0,ntest)); inds.extend([(st+j)%ntest for j in range(7)])
            inds=np.asarray(inds[:ntest]); boot.append({'comparison':'full_minus_'+m,'metric':'RMSE2','estimate':float(d_rm.mean()),'low':float(d_rm[inds].mean()),'high':float(d_rm[inds].mean())})
        # quantiles from a second compact array
        arrs=[]
        for b in range(2000):
            inds=[]
            while len(inds)<ntest:
                st=int(rng.integers(0,ntest)); inds.extend([(st+j)%ntest for j in range(7)])
            arrs.append(d_rm[np.asarray(inds[:ntest])].mean())
        boot[-2000:]=[{'comparison':'full_minus_'+m,'metric':'RMSE2','estimate':float(d_rm.mean()),'low':float(np.quantile(arrs,.025)),'high':float(np.quantile(arrs,.975))}]
    pd.DataFrame(boot).drop_duplicates().to_csv(OUT/'paired_block_bootstrap.csv',index=False)
    # chronology and manifest
    manifest={'source':'five_wells_timeseries_clean.parquet (confidential; path withheld)','source_sha256':sha256(DATA/'five_wells_timeseries_clean.parquet'),'coordinates_sha256':sha256(DATA/'five_wells_info.csv'),'receiving_roles':['extraction','injector_1','injector_2','injector_3','injector_4'],'variables':VARS,'date_start':str(df.index.min().date()),'date_end':str(df.index.max().date()),'n_days':n,'L':L,'H':H,'split_bounds':bounds,'split_dates':{k:[str(df.index[a].date()),str(df.index[b-1].date())] for k,(a,b) in bounds.items()},'window_counts':{k:int(len(Y[k])) for k in Y},'normalization':'training partition only','sample_assignment':'target-date based','forecast_origin_rule':'target date minus 7 days','locked_seed':11,'distance_m':[float(x) for x in dist.tolist()],'graph_formula':'Gaussian distance affinity times bounded standardized-flow modulation; incoming edges normalized jointly with a unit centre self-loop','loss_weights':{'lambda_negative':0.08,'lambda_rate':0.05,'lambda_rising_stage':0.03},'stage_rule':'rising if seven-day historical concentration change >= 0.10 mg/L; stable between +/-0.10; falling <= -0.10','rate_threshold_rule':'training-only 95th percentile of absolute H-day endpoint change divided by H','mase_denominator_training_abs_diff':den,'evaluation':'one train+validation refit after selection; test parameters remain fixed','note':'raw clean parquet; no smoothed or interpolated derivatives; endpoint target at t+H; receiving-row graph convention'}
    (OUT/'analysis_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    # simple diagnostic figure from seed 11
    try:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        p=pd.DataFrame(pred_records); p['Date']=pd.to_datetime(p.Date)
        fig,axs=plt.subplots(2,1,figsize=(10,6),sharex=True,gridspec_kw={'height_ratios':[2,1]})
        f=p[p.model=='full']; axs[0].plot(f.Date,f.Observed,'k-',lw=1,label='Observed'); axs[0].plot(f.Date,f.Predicted,color='#1f77b4',label='PG-SSM'); axs[0].fill_between(f.Date,f.Lower90,f.Upper90,color='#1f77b4',alpha=.2,label='PI90'); axs[0].legend(frameon=False,ncol=3); axs[0].set_ylabel('U (mg L$^{-1}$)'); axs[0].set_title('Locked chronological test set (seed 11)')
        axs[1].plot(f.Date,f.Residual,'#555'); axs[1].axhline(0,color='k',lw=.7); axs[1].set_ylabel('Residual'); axs[1].set_xlabel('Target date'); fig.tight_layout(); fig.savefig(OUT/'Fig_R3_test_forecast.png',dpi=220); plt.close(fig)
    except Exception as e: (OUT/'plot_error.txt').write_text(str(e),encoding='utf-8')
    print(json.dumps({'out':str(OUT),'n_days':n,'window_counts':manifest['window_counts'],'summary':str(OUT/'metrics_summary.csv')},ensure_ascii=False))

if __name__=='__main__': run()
