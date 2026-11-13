import os, sys, pandas as pd
_AS=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ABL=os.path.join(_AS,'exp_2','ablation')
sys.path.insert(0,_ABL); sys.path.insert(0,os.path.join(_ABL,'preprocessing'))
import run_ablation as R
from abl_config import LOO_CONFIGS, BEST_DOWNTIME, BEST_TH
from build_streams import clean_streams
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data')
os.makedirs(OUT,exist_ok=True)
mfv1=pd.read_csv(R.MFV1); wmap=R._weight_map()
print('MFV1 rows', len(mfv1), 'weights', len(wmap), flush=True)
for cfg,tog in LOO_CONFIGS.items():
    out=os.path.join(OUT,f'pressure_loo_{cfg}.csv')
    if os.path.exists(out):
        print('exists', out, flush=True); continue
    p=clean_streams(mfv1, downtime=BEST_DOWNTIME, Th=BEST_TH, weight_map=wmap, **tog)
    p.to_csv(out,index=False)
    print(f'loo_{cfg}: rows={len(p)} -> {out}', flush=True)
print('DONE', flush=True)
