"""Regenerate no_priming/no_downtime streams with FLAT downtime (so the LOO baseline all3 == the
headline current_denoised) and build their base-133 machine tables for 1day+2day."""
import os, sys, pandas as pd
_AS=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ABL=os.path.join(_AS,'exp_2','ablation')
sys.path.insert(0,_ABL); sys.path.insert(0,os.path.join(_ABL,'preprocessing')); sys.path.insert(0,os.path.join(_ABL,'models'))
import run_ablation as R
from build_streams import clean_streams
from build_features import pressure_to_day_features, write_splits
from abl_config import feature_dir, CASES
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data'); os.makedirs(OUT,exist_ok=True)
mfv1=pd.read_csv(R.MFV1); wmap=R._weight_map()
# FLAT downtime; toggle priming / downtime; peak removed
FLAT={'no_priming':  dict(remove_priming=False, remove_downtime=True,  remove_peak=True),
      'no_downtime': dict(remove_priming=True,  remove_downtime=False, remove_peak=True)}
for cfg,tog in FLAT.items():
    csv=os.path.join(OUT,f'pressure_flat_{cfg}.csv')
    if os.path.exists(csv):
        press=pd.read_csv(csv); print('exists',csv,len(press),flush=True)
    else:
        press=clean_streams(mfv1, downtime='flat', weight_map=wmap, **tog)
        press.to_csv(csv,index=False); print(f'flat_{cfg}: rows={len(press)} -> {csv}',flush=True)
    # build base-133 tables for both cases under a flat tag
    for case in CASES:
        mdir=feature_dir(f'flat_{cfg}','x','machine_only',False,case)
        if os.path.exists(os.path.join(mdir,'test.csv')):
            print('  base exists',mdir,flush=True); continue
        feats=pressure_to_day_features(press, case=case)
        write_splits(feats, mdir)
        print(f'  built base-133 {cfg}/{case} -> {mdir}',flush=True)
print('DONE',flush=True)
