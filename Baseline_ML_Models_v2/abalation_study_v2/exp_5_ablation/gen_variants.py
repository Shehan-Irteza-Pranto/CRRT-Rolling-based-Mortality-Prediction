"""
gen_variants.py — generate the incremental-preprocessing pressure variants + base-133 tables (rows 1-4).
Configs (FLAT downtime, mirrors the headline recipe): raw (nothing removed), prim (priming only),
primdt (priming+downtime). Peak is NOT removed in any of these (rows 1-4 have peak-removal ✗). all3
(full) already exists as the headline. Reuses exp_2/ablation clean_streams + build_features.
"""
import os
import sys
import pandas as pd

_AS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ABL = os.path.join(_AS, "exp_2", "ablation")
for p in [_ABL, os.path.join(_ABL, "preprocessing"), os.path.join(_ABL, "models")]:
    sys.path.insert(0, p)
import run_ablation as R                                      # noqa: E402
from build_streams import clean_streams                      # noqa: E402
from build_features import pressure_to_day_features, write_splits  # noqa: E402
from abl_config import feature_dir, CASES                    # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"); os.makedirs(OUT, exist_ok=True)
CFGS = {"raw":    dict(remove_priming=False, remove_downtime=False, remove_peak=False),
        "prim":   dict(remove_priming=True,  remove_downtime=False, remove_peak=False),
        "primdt": dict(remove_priming=True,  remove_downtime=True,  remove_peak=False)}


def main():
    mfv1 = pd.read_csv(R.MFV1); wmap = R._weight_map()
    for cfg, tog in CFGS.items():
        csv = os.path.join(OUT, f"pressure_{cfg}.csv")
        if os.path.exists(csv):
            press = pd.read_csv(csv); print(f"exists {csv} ({len(press)})", flush=True)
        else:
            press = clean_streams(mfv1, downtime="flat", weight_map=wmap, **tog)
            press.to_csv(csv, index=False); print(f"{cfg}: rows={len(press)} -> {csv}", flush=True)
        for case in CASES:
            mdir = feature_dir(f"abl5_{cfg}", "x", "machine_only", False, case)
            if os.path.exists(os.path.join(mdir, "test.csv")):
                print(f"  base exists {mdir}", flush=True); continue
            feats = pressure_to_day_features(press, case=case)
            write_splits(feats, mdir)
            print(f"  built base-133 {cfg}/{case} -> {mdir}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
