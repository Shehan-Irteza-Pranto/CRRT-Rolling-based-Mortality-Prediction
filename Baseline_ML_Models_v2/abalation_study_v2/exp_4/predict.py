"""
predict.py — load an exp_4 model bundle and score a cohort (external-validation loader + round-trip check)
=========================================================================================================
Each bundle in exp_4/models/ is self-contained (weights/estimators + fitted preprocessors + threshold).
This module reconstructs the model and scores the held-out TEST set; `--verify` checks the reproduced
AUROC matches the reported metrics (proof the bundles are functional). For an EXTERNAL cohort, feed the
same-format tables / reduced sequences instead of the internal test split (see LOAD_AND_PREDICT.md).
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib

_HERE = os.path.dirname(os.path.abspath(__file__))
_AS = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_AS, "exp_3", "results"))
import combined_fusion_ensemble as CF                # noqa: E402 (sets up paths; gives FusionModel, tables)
sys.path.insert(0, os.path.join(_AS, "exp_3", "dl"))
import train_eval as TE                              # noqa: E402 (reduced-sequence test loader)
from model import AttnCNN1D                          # noqa: E402
import torch                                         # noqa: E402
from sklearn.metrics import roc_auc_score            # noqa: E402
MODELS_DIR = os.path.join(_HERE, "models")
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---- preprocessing replay (same order as training) --------------------------------------------------
def _classical_prob(estimator, prep, table):
    X = table[prep["feats"]].to_numpy(float)
    X = prep["imputer"].transform(X)
    X = CF.transform_with_preprocessor(prep["scale_prep"], X)
    if prep.get("cidx") is not None:
        X = X[:, prep["cidx"]]
    return estimator.predict_proba(X)[:, 1]


def _attn_prob(state_dict, in_ch, arch, Xte, mte):
    m = AttnCNN1D(in_ch, channels=arch["channels"], n_blocks=arch["n_blocks"],
                  kernel=arch["kernel"], dropout=arch["dropout"], attention=arch["attention"]).to(DEV)
    m.load_state_dict(state_dict); m.eval()
    return TE._infer(m, Xte, mte, DEV)


def _fusion_prob(b, Xte, mte, comb_table):
    fm = CF.FusionModel(b["in_ch"], b["clin_dim"], pooling=b.get("pooling", "attention")).to(DEV)
    fm.load_state_dict(b["state_dict"]); fm.eval()
    # clinical branch: impute -> scale -> select clin_idx (fit on train, stored in bundle)
    Xc = comb_table[b["clin_cols"]].to_numpy(float)
    Xc = b["clin_scaler"].transform(b["clin_imputer"].transform(Xc))[:, b["clin_idx"]]
    c = torch.from_numpy(Xc.astype(np.float32))
    out = []
    with torch.no_grad():
        for i in range(0, len(Xte), 256):
            out.append(torch.sigmoid(fm(Xte[i:i+256].to(DEV), mte[i:i+256].to(DEV),
                                        c[i:i+256].to(DEV))).cpu().numpy())
    return np.concatenate(out)


def score(bundle_path, ctx):
    """ctx supplies test data: reduced (Xte,mte), machine tables {base,rich}, combined table, clinical."""
    b = joblib.load(bundle_path); kind = b["kind"]
    if kind == "dl_seq_attn":
        p = _attn_prob(b["state_dict"], b["in_ch"], b["arch"], ctx["Xte"], ctx["mte"])
    elif kind == "ensemble_mean":
        cols = []
        for mem in b["classical_members"]:
            cols.append(_classical_prob(mem["estimator"], mem["prep"], ctx["machine"][mem["featset"]]))
        a = b["attn_cnn1d"]
        cols.append(_attn_prob(a["state_dict"], a["in_ch"], a["arch"], ctx["Xte"], ctx["mte"]))
        p = np.column_stack(cols).mean(1)
    elif kind == "dl_fusion":
        p = _fusion_prob(b, ctx["Xte"], ctx["mte"], ctx["combined"])
    elif kind == "logistic_stack":
        cols = []
        for mem in b["members"]:
            if mem["kind"] == "dl_fusion":
                cols.append(_fusion_prob(mem, ctx["Xte"], ctx["mte"], ctx["combined"]))
            else:
                tbl = ctx["combined"] if mem["group"] == "combined" else ctx["clinical"]
                cols.append(_classical_prob(mem["estimator"], mem["prep"], tbl))
        p = b["meta_learner"].predict_proba(np.column_stack(cols))[:, 1]
    else:
        raise ValueError(kind)
    return p, b["threshold"]


def _test_ctx(case):
    Xte, mte, yte_t, _, _ = TE._load("test", case)
    machine = {fs: CF.CR._load(fs, case)[2] for fs in ("base", "rich")}     # test tables
    combined = CF._combined_table(case, "test")
    clinical = pd.read_csv(os.path.join(CF.CLINICAL_ONLY, case, "test.csv"))
    return {"Xte": Xte, "mte": mte, "machine": machine, "combined": combined,
            "clinical": clinical}, yte_t.numpy().astype(int)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--verify", action="store_true"); a = ap.parse_args()
    ref = pd.concat([pd.read_csv(os.path.join(_HERE, "results", f))
                     for f in ("pathB_machine_bestseed.csv", "pathB_combined_bestseed.csv")], ignore_index=True)
    files = {("Attention-CNN1D", "1day"): "machine_attention_cnn1d_1day.joblib",
             ("Attention-CNN1D", "2day"): "machine_attention_cnn1d_2day.joblib",
             ("Ensemble", "1day"): "machine_ensemble_1day.joblib",
             ("Ensemble", "2day"): "machine_ensemble_2day.joblib",
             ("Attn-CNN fusion", "1day"): "combined_attn_cnn_fusion_1day.joblib",
             ("Attn-CNN fusion", "2day"): "combined_attn_cnn_fusion_2day.joblib",
             ("Ensemble/Stack", "1day"): "combined_ensemble_stack_1day.joblib",
             ("Ensemble/Stack", "2day"): "combined_ensemble_stack_2day.joblib"}
    print(f"{'model':18s} {'case':5s} {'bundle_AUROC':>12} {'reported':>9} {'Δ':>7}")
    for case in ("1day", "2day"):
        ctx, yte = _test_ctx(case)
        for (model, cs), fn in files.items():
            if cs != case:
                continue
            p, thr = score(os.path.join(MODELS_DIR, fn), ctx)
            au = roc_auc_score(yte, p)
            rep = float(ref[(ref.model == model) & (ref.case == case)]["full_AUROC"].iloc[0])
            flag = "OK" if abs(au - rep) < 0.01 else "**MISMATCH**"
            print(f"{model:18s} {case:5s} {au:12.4f} {rep:9.3f} {au-rep:+7.4f}  {flag}")


if __name__ == "__main__":
    main()
