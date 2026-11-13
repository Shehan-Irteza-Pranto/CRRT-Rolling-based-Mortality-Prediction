"""
prepare_downsampled.py
======================
Build DOWNSAMPLED [T', 4] pressure-stream caches for the sequence rungs, to mitigate the
two failure modes the full-resolution (1440-min) streams showed: the transformer overfits
and its AUROC collapses on length-1440 windows, while cnn1d/rnn underfit. Aggregating the
per-minute stream into short window summaries both smooths noise and shrinks the sequence
length 1440 -> ~96-288 steps, which is the documented remedy (see models/transformer.py).

Each circuit-day raw stream [T, 4] (cropped to MAX_LEN minutes) is split into windows of
W minutes with stride S, and each window is reduced to one timestep by mean or median:

  W (window) in {10, 15} minutes
  overlap    in {none (S = W), 50% (S = W // 2)}
  agg        in {mean, median}
  => 8 variants. Variant name encodes them: w{W}_s{S}_{agg}, e.g. w10_s05_median.

Everything else mirrors prepare_sequences.py EXACTLY (same split membership/labels, train-only
normalization, pad+mask) so the downsampled rungs stay on the identical patient-level protocol.

Output layout (one self-contained cache root per variant, same internal structure as the
full-resolution cache, so the existing SequenceDataset reads it unchanged):

  data/sequence_cache_downsampled/<variant>/<stream>/<case>_{train,val,test}.npz
  data/sequence_cache_downsampled/<variant>/<stream>/<case>_meta.json

Usage
-----
    python prepare_downsampled.py                          # all 8 variants, all cases
    python prepare_downsampled.py --variants w10_s10_mean w15_s15_median
    python prepare_downsampled.py --cases 1day --stream denoised
"""

import os
import sys
import json
import math
import argparse
import warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    STREAM_FILES, DEFAULT_STREAM, CHANNELS, KEY_COLS, TARGET_COL, CASES,
    MAX_LEN, PAD_VALUE, CROP_SIDE, _HERE,
)
# Reuse the framework-free helpers that do NOT depend on MAX_LEN.
from prepare_sequences import build_stream_index, load_split_keys, fit_channel_stats, SPLITS

DOWNSAMPLED_ROOT = os.path.join(_HERE, "data", "sequence_cache_downsampled")

WINDOWS = [10, 15]            # minutes
AGGS    = ["mean", "median"]
OVERLAPS = ["none", "half"]   # half == 50% overlap


def build_variants():
    """The 8 (window x overlap x agg) combinations, each as a spec dict."""
    out = []
    for W in WINDOWS:
        for ov in OVERLAPS:
            S = W if ov == "none" else max(1, W // 2)
            for agg in AGGS:
                out.append({
                    "name": f"w{W:02d}_s{S:02d}_{agg}",
                    "window": W, "stride": S, "agg": agg, "overlap": ov,
                    "max_len": math.ceil(MAX_LEN / S),     # max #windows over MAX_LEN minutes
                })
    return out


VARIANTS = build_variants()
VARIANT_BY_NAME = {v["name"]: v for v in VARIANTS}


# --- Core downsampling ---

def downsample_array(arr, window, stride, agg, max_raw=MAX_LEN):
    """[T, 4] per-minute stream -> [T', 4] window summaries (mean/median, NaN-aware)."""
    arr = arr[:max_raw]                                    # match full-res coverage
    T = arr.shape[0]
    if T == 0:
        return np.zeros((0, len(CHANNELS)), dtype=np.float32)
    aggfn = np.nanmean if agg == "mean" else np.nanmedian
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)   # all-NaN window -> NaN
        for start in range(0, T, stride):
            w = arr[start:start + window]
            if w.shape[0] == 0:
                break
            rows.append(aggfn(w, axis=0))
    return np.asarray(rows, dtype=np.float32)


def build_downsampled_index(stream_index, window, stride, agg):
    return {k: downsample_array(v, window, stride, agg) for k, v in stream_index.items()}


def pad_crop(arr, max_len):
    """Pad/crop a [T', 4] array to max_len. Returns (padded [max_len, 4], mask [max_len])."""
    out = np.full((max_len, len(CHANNELS)), PAD_VALUE, dtype=np.float32)
    mask = np.zeros(max_len, dtype=bool)
    t = arr.shape[0]
    if t == 0:
        return out, mask
    if t >= max_len:
        arr = arr[:max_len] if CROP_SIDE == "right" else arr[-max_len:]
        out[:] = arr; mask[:] = True
    else:
        out[:t] = arr; mask[:t] = True
    return out, mask


def assemble_split(keys_df, ds_index, stats, max_len):
    mean = np.array(stats["mean"], dtype=np.float32)
    std = np.array(stats["std"], dtype=np.float32)
    X, M, Y, K, missing = [], [], [], [], 0
    for _, r in keys_df.iterrows():
        key = (int(r.PID), int(r.Visit_no), int(r.Day))
        arr = ds_index.get(key)
        if arr is None or len(arr) == 0:
            arr = np.zeros((0, len(CHANNELS)), dtype=np.float32); missing += 1
        if arr.size:
            inds = np.where(np.isnan(arr))
            arr = arr.copy(); arr[inds] = np.take(mean, inds[1])   # impute to channel mean
            arr = (arr - mean) / std                                # then standardize
        padded, mask = pad_crop(arr, max_len)
        X.append(padded); M.append(mask); Y.append(int(r[TARGET_COL])); K.append(key)
    return (np.stack(X), np.stack(M),
            np.array(Y, dtype=np.int64), np.array(K, dtype=np.int64), missing)


def prepare_variant_case(variant, case, stream_index, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    keys = {s: load_split_keys(case, s) for s in SPLITS}
    pid = {s: set(keys[s].PID.unique()) for s in SPLITS}
    assert not (pid["train"] & pid["test"]) and not (pid["train"] & pid["val"]) \
        and not (pid["val"] & pid["test"]), "PID leakage across splits!"

    ds_index = build_downsampled_index(stream_index, variant["window"],
                                        variant["stride"], variant["agg"])
    stats = fit_channel_stats(keys["train"], ds_index)         # train-only normalization
    max_len = variant["max_len"]

    summary = {"case": case, "variant": variant, "channel_stats": stats, "splits": {}}
    for s in SPLITS:
        X, M, Y, K, missing = assemble_split(keys[s], ds_index, stats, max_len)
        np.savez_compressed(os.path.join(out_dir, f"{case}_{s}.npz"),
                            X=X, mask=M, y=Y, keys=K, channels=np.array(CHANNELS))
        summary["splits"][s] = {"n": int(len(Y)), "n_pos": int(Y.sum()),
                                "pos_rate": float(Y.mean()) if len(Y) else 0.0,
                                "missing_streams": int(missing)}
        print(f"  [{variant['name']}/{case}/{s}] N={len(Y)} pos={int(Y.sum())} "
              f"({100*Y.mean():.1f}%) X={X.shape}")
    with open(os.path.join(out_dir, f"{case}_meta.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def build_all(variant_names=None, cases=None, stream=DEFAULT_STREAM):
    """Build every requested variant. The expensive raw-stream index is built once."""
    cases = cases or CASES
    variants = [VARIANT_BY_NAME[n] for n in variant_names] if variant_names else VARIANTS
    print(f"Indexing raw stream once: {STREAM_FILES[stream]}")
    stream_index = build_stream_index(STREAM_FILES[stream])
    print(f"  indexed {len(stream_index)} circuit-day streams\n")
    for v in variants:
        print(f"=== variant {v['name']}  (W={v['window']} S={v['stride']} "
              f"agg={v['agg']} overlap={v['overlap']} -> max_len={v['max_len']}) ===")
        out_dir = os.path.join(DOWNSAMPLED_ROOT, v["name"], stream)
        for case in cases:
            prepare_variant_case(v, case, stream_index, out_dir)
        print()
    print(f"Done. Downsampled caches -> {DOWNSAMPLED_ROOT}/<variant>/{stream}/")


def main():
    ap = argparse.ArgumentParser(description="Build downsampled sequence caches (8 variants)")
    ap.add_argument("--variants", nargs="+", default=None,
                    help=f"subset of {[v['name'] for v in VARIANTS]}")
    ap.add_argument("--cases", nargs="+", default=CASES)
    ap.add_argument("--stream", choices=list(STREAM_FILES), default=DEFAULT_STREAM)
    a = ap.parse_args()
    build_all(a.variants, a.cases, a.stream)


if __name__ == "__main__":
    main()
