"""
roc_pathA_probs.py — test probabilities for the 6 Path A DL models (1-day) by re-inference.
Rebuilds each best-val-seed checkpoint via the benchmark's own build+dataset+predict and scores TEST.
Runs in the parent 'models/' import context (models/dl_data, NOT exp_3/dl).
"""
import os
import sys
import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_AS = os.path.dirname(_HERE)
for p in [os.path.join(os.path.dirname(_AS), "Deep_Learning_Models", "models"),
          os.path.join(os.path.dirname(_AS), "Machine_Clinical_Models"),
          _AS, os.path.join(_AS, "models")]:
    sys.path.insert(0, p)
import dl_train as DT                                  # noqa: E402 (models/) -> _build, _predict, AggDataset
from sklearn.metrics import roc_auc_score              # noqa: E402

SAVED = os.path.join(_AS, "results", "saved_models", "dl")
OUT = os.path.join(_HERE, "results", "probs"); os.makedirs(OUT, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CASE = "1day"
CELLS = [("machine_only", "cnn1d"), ("machine_only", "rnn"), ("machine_only", "transformer"),
         ("combined", "cnn1d"), ("combined", "rnn"), ("combined", "transformer")]


def _seed_map():
    a = pd.read_csv(os.path.join(_HERE, "results", "pathA_bestseed.csv"))
    name2m = {"CNN1D": "cnn1d", "RNN (BiLSTM/GRU)": "rnn", "Transformer": "transformer"}
    pipe2fs = {"Machine-only": "machine_only", "Combined": "combined"}
    m = {}
    for _, r in a[a.case == CASE].iterrows():
        if r.model in name2m and r.pipeline in pipe2fs:
            m[(pipe2fs[r.pipeline], name2m[r.model])] = int(r.sel_seed)
    return m


def main():
    seeds = _seed_map()
    reduce_enabled = getattr(DT, "DL_REDUCE_ENABLED", True)
    out = {}
    yref = None
    for pipeline, enc in CELLS:
        seed = seeds[(pipeline, enc)]
        DT._set_seed(seed)
        reduce_spec = DT.DL_REDUCE.get(enc) if reduce_enabled else None
        want_machine = False if pipeline == "combined" else None
        tr = DT.AggDataset(CASE, "train", pipeline, "per_day", "current", "denoised",
                           reduce=reduce_spec, want_machine=want_machine)
        te = DT.AggDataset(CASE, "test", pipeline, "per_day", "current", "denoised",
                           reduce=reduce_spec, want_machine=want_machine,
                           mach_scaler=tr.mach_scaler, clin_scaler=tr.clin_scaler)
        pt = f"current_denoised_{pipeline}_per_day_{enc}_{CASE}_s{seed}.pt"
        ck = torch.load(os.path.join(SAVED, pt), map_location=DEV, weights_only=False)
        model = DT._build(pipeline, "per_day", enc, tr.machine_dim, tr.clinical_dim,
                          ck["best_hp"], tr.seq_feat_dim, DEV)
        model.load_state_dict(ck["state_dict"])
        tp, ty = DT._predict(model, te, DEV, tr.machine_dim, tr.clinical_dim)
        out[f"{pipeline}_{enc}"] = tp
        yref = ty if yref is None else yref
        assert np.array_equal(ty, yref), "test label order mismatch across Path A models"
        print(f"  {pipeline:12s} {enc:11s} seed{seed} AUROC {roc_auc_score(ty, tp):.4f}", flush=True)
    out["yte"] = yref
    np.savez(os.path.join(OUT, "roc_pathA_1day.npz"), **out)
    print(f"saved -> {os.path.join(OUT, 'roc_pathA_1day.npz')}")


if __name__ == "__main__":
    main()
