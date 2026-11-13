"""
signal.py — gap-aware per-minute pressure signals in PHYSICAL units (exp_3)
==========================================================================
Same gap-aware Stage-0 rule as exp_2/frag_cnn/preprocessing/build_days.py, but it returns the
signal in RAW physical units (NOT z-scored) with NaN at masked minutes — because the richer
physiology features (alarm/excursion thresholds, band-power, end-of-day deltas) are only meaningful
on the real pressure scale. Rows come out in the SAME ORDER as the machine_only per-day table, so
new features can be concatenated to the existing 133 with no join.

  * small gap  (run <= GAP_MAX_MIN, bracketed) -> linear interpolate, mask=1
  * large gap / leading / trailing             -> left NaN, mask=0

6 channels = [access, filter, effluent, return, TMP, PFD]; ARD = return - access is derived on
the fly by feature code that wants it.
"""
import os
import sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from fcfg import (RAW_PRESSURE, machine_table, GAP_MAX_MIN, DAY_LEN,          # noqa: E402
                  N_CHANNELS, CHANNEL_NAMES)

RAW4 = ["ACCESS_PRESSURE", "FILTER_PRESSURE", "EFFLUENT_PRESSURE", "RETURN_PRESSURE"]


def _gap_fill_day(grid4, present, gap_max=GAP_MAX_MIN):
    """grid4 [1440,4] (NaN absent), present[1440] bool -> (filled4, mask[1440])."""
    mask = present.copy()
    obs = np.where(present)[0]
    if len(obs) == 0:
        return grid4, mask
    first, last = obs[0], obs[-1]
    for a, b in zip(obs[:-1], obs[1:]):
        run = b - a - 1
        if run <= 0:
            continue
        if run <= gap_max:
            for c in range(grid4.shape[1]):
                grid4[a:b + 1, c] = np.linspace(grid4[a, c], grid4[b, c], run + 2)
            mask[a + 1:b] = True
    mask[:first] = False; mask[last + 1:] = False
    return grid4, mask


def _raw_index(pids):
    """Read raw pressure once -> {(pid,vis,day): (minute_of_day[n], vals[n,4])} for `pids`."""
    raw = pd.read_csv(RAW_PRESSURE, usecols=["PID", "Visit_no", "Day", "Time"] + RAW4)
    raw = raw[raw["PID"].isin(pids)]
    t = pd.to_datetime(raw["Time"])
    raw = raw.assign(_min=(t.dt.hour * 60 + t.dt.minute).to_numpy())
    idx = {}
    for key, gdf in raw.groupby(["PID", "Visit_no", "Day"], sort=False):
        gdf = gdf.drop_duplicates("_min")
        idx[tuple(int(k) for k in key)] = (gdf["_min"].to_numpy(), gdf[RAW4].to_numpy(np.float32))
    return idx


def load_raw_days(split, case):
    """Return (X[N,6,1440] physical, NaN-masked; mask[N,1440]; keys DataFrame PID/Visit_no/Day/label).

    Row order == machine_table(split, case) order.
    """
    keys = pd.read_csv(machine_table(split, case), usecols=["PID", "Visit_no", "Day", "label"])
    idx = _raw_index(set(keys["PID"]))
    N = len(keys)
    X = np.full((N, N_CHANNELS, DAY_LEN), np.nan, np.float32)
    M = np.zeros((N, DAY_LEN), bool)
    for i, r in enumerate(keys.itertuples()):
        g = idx.get((int(r.PID), int(r.Visit_no), int(r.Day)))
        grid4 = np.full((DAY_LEN, 4), np.nan, np.float32)
        if g is not None:
            mins, vals = g
            grid4[mins] = vals
        present = ~np.isnan(grid4[:, 0])
        grid4, mask = _gap_fill_day(grid4, present)
        a, f, e, rr = grid4[:, 0], grid4[:, 1], grid4[:, 2], grid4[:, 3]
        tmp = (f + rr) / 2.0 - e; pfd = f - rr
        six = np.stack([a, f, e, rr, tmp, pfd], axis=0)   # [6,1440]
        six[:, ~mask] = np.nan
        X[i] = six; M[i] = mask
    return X, M, keys


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--case", default="1day")
    a = ap.parse_args()
    for s in ("train", "val", "test"):
        X, M, keys = load_raw_days(s, a.case)
        print(f"[signal] {a.case}/{s}: days={len(keys)} pos={int(keys['label'].sum())} "
              f"X={X.shape} valid-frac={M.mean():.3f} channels={CHANNEL_NAMES}")
