"""
preprocessing/build_combined.py
===============================
Build the combined (machine + clinical) per-day tables by joining the freshly
regenerated machine 52-stat features with the (denoising-independent) clinical
lab+PTI features, on (PID, Visit_no, Day).

Clinical features come from the existing clinical_only tables — they do not change
with the denoising variant, so they are treated as a fixed covariate source.
"""

import os
import sys
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from abl_config import CLINICAL_ONLY                                # noqa: E402

META = ["PID", "Visit_no", "Day", "Mortality", "days_until_death", "label"]
KEYS = ["PID", "Visit_no", "Day"]
CLINICAL_DIR = CLINICAL_ONLY                                        # v2-resplit clinical (70:10:20)


def _clinical_feats(df):
    return [c for c in df.columns if c not in set(META) | {"Date", "Death_Date"}]


def build_combined_split(machine_df, clinical_path):
    """machine per-day table (meta+52 stats) + clinical features -> combined table."""
    clin = pd.read_csv(clinical_path)
    clin_feats = _clinical_feats(clin)
    mach_feats = [c for c in machine_df.columns if c not in set(META) | {"Date", "Death_Date"}]
    merged = machine_df.merge(clin[KEYS + clin_feats], on=KEYS, how="inner")
    meta = [c for c in META if c in merged.columns]
    return merged[meta + mach_feats + clin_feats]


def build_combined_dir(machine_dir, case, out_dir, clinical_dir=CLINICAL_DIR):
    """Read machine {train,val,test}.csv from machine_dir, join clinical, write to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    counts = {}
    for split in ["train", "val", "test"]:
        mpath = os.path.join(machine_dir, f"{split}.csv")
        cpath = os.path.join(clinical_dir, case, f"{split}.csv")
        if not (os.path.exists(mpath) and os.path.exists(cpath)):
            continue
        out = build_combined_split(pd.read_csv(mpath), cpath)
        out.to_csv(os.path.join(out_dir, f"{split}.csv"), index=False)
        counts[split] = (len(out), out.shape[1])
    return counts


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine-dir", required=True, help="dir with machine {train,val,test}.csv")
    ap.add_argument("--case", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    print(build_combined_dir(a.machine_dir, a.case, a.out_dir))
