"""
export_machine_bundles.py — save externally-runnable bundles for the MACHINE 'ours' models
==========================================================================================
Re-runs machine Attention-CNN1D + Ensemble per seed, saves the BEST-val seed as a self-contained bundle
(weights/estimators + fitted preprocessors + threshold + recipe) under exp_4/models/, and regenerates
the pathB machine metrics FROM the saved artifacts so table == saved model. GPU.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib

_HERE = os.path.dirname(os.path.abspath(__file__))
_AS = os.path.dirname(_HERE)
for p in [os.path.join(os.path.dirname(_AS), "Deep_Learning_Models", "models"),
          os.path.join(os.path.dirname(_AS), "Machine_Clinical_Models"),
          _AS, os.path.join(_AS, "models"),
          os.path.join(_AS, "exp_3", "features"), os.path.join(_AS, "exp_3", "results"),
          os.path.join(_AS, "exp_3", "dl")]:
    sys.path.insert(0, p)
import train_eval as TE                              # noqa: E402  (exp_3/dl)
import dlcfg                                         # noqa: E402
import classical_rich as CR                          # noqa: E402
import make_resampled_variants as MR                 # noqa: E402
from sklearn.base import clone                       # noqa: E402
from sklearn.metrics import roc_auc_score            # noqa: E402
import torch                                         # noqa: E402

MODELS_DIR = os.path.join(_HERE, "models"); os.makedirs(MODELS_DIR, exist_ok=True)
OUT = os.path.join(_HERE, "results")
TOP_K = 3


def _thr(yva, vp):
    from sklearn.metrics import precision_recall_curve
    pr, rc, th = precision_recall_curve(yva, vp)
    f1 = np.where((pr[:-1] + rc[:-1]) > 0, 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1]), 0)
    return float(th[int(np.argmax(f1))]) if len(th) else 0.5


def _norm_stats(case):
    z = np.load(os.path.join(dlcfg.DATA_DIR, f"norm_{case}.npz"))
    return z["mean"], z["std"]


def _attn_perseed(case, device):
    Xva, mva, yva_t, _, _ = TE._load("val", case)
    Xte, mte, yte_t, _, _ = TE._load("test", case)
    seeds = []
    for sd in range(TE.N_SEEDS):
        model, vbest = TE.train_one(case, sd, device)
        vp = TE._infer(model, Xva, mva, device); tp = TE._infer(model, Xte, mte, device)
        state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        seeds.append({"state": state, "vp": vp, "tp": tp, "vbest": vbest,
                      "in_ch": Xva.shape[-1]})
        print(f"  [attn {case}] seed {sd}: val {vbest:.4f}", flush=True)
    return seeds, yva_t.numpy().astype(int), yte_t.numpy().astype(int)


def _machine_classical(case):
    """Returns fs -> {'prep':..., models: {m: {'est':[per seed], 'vp':[S,n], 'tp':[S,n]}}}, yva, yte."""
    store = {}; yva = yte = None
    for fs in ("base", "rich"):
        tr, va, te, feats = CR._load(fs, case)
        Xtr, ytr = CR._xy(tr, feats); Xva, yva = CR._xy(va, feats); Xte, yte = CR._xy(te, feats)
        imp = CR.SimpleImputer(strategy="median").fit(Xtr)
        Xtr_i, Xva_i, Xte_i = imp.transform(Xtr), imp.transform(Xva), imp.transform(Xte)
        Xtr_sc, Xte_sc, prep = CR.scale_features(Xtr_i, Xte_i)
        Xva_sc = CR.transform_with_preprocessor(prep, Xva_i)
        chosen = CR.mrmr_select(Xtr_sc, ytr, list(feats), min(CR.N_MRMR, Xtr_sc.shape[1]))
        cidx = [feats.index(c) for c in chosen]
        Xtr_s, Xva_s, Xte_s = Xtr_sc[:, cidx], Xva_sc[:, cidx], Xte_sc[:, cidx]
        models = {}
        for m in CR.MODELS:
            best = CR._tune(m, Xtr_s, ytr)
            if best is None:
                continue
            ests, vp, tp = [], [], []
            for sd in range(CR.N_SEEDS):
                est = clone(best)
                try:
                    est.set_params(random_state=sd)
                except ValueError:
                    pass
                est.fit(Xtr_s, ytr)
                ests.append(est); vp.append(est.predict_proba(Xva_s)[:, 1]); tp.append(est.predict_proba(Xte_s)[:, 1])
            models[m] = {"est": ests, "vp": np.array(vp), "tp": np.array(tp)}
        store[fs] = {"prep": {"feats": feats, "imputer": imp, "scale_prep": prep, "cidx": cidx,
                              "sel_names": chosen}, "models": models}
    return store, yva.astype(int), yte.astype(int)


def _ens_for_seed(s, store, attn_seeds, yva):
    cands = {}
    for fs in ("base", "rich"):
        for m, d in store[fs]["models"].items():
            cands[f"cls:{fs}:{m}"] = (d["vp"][s], d["tp"][s])
    cands["dl:attn_cnn1d"] = (attn_seeds[s]["vp"], attn_seeds[s]["tp"])
    vauc = {n: roc_auc_score(yva, v) for n, (v, _) in cands.items()}
    keep, seen = [], set()
    for n in sorted(vauc, key=vauc.get, reverse=True):
        fam = n.split(":")[-1]
        if n.startswith("dl:") or (fam not in seen and len([k for k in keep if k.startswith("cls:")]) < TOP_K):
            keep.append(n); seen.add(fam)
    if "dl:attn_cnn1d" not in keep:
        keep.append("dl:attn_cnn1d")
    Vv = np.column_stack([cands[n][0] for n in keep]); Vt = np.column_stack([cands[n][1] for n in keep])
    return Vv.mean(1), Vt.mean(1), keep


def _row(model, case, vp, tp, yva, yte, seed):
    by = {r["protocol"]: r for r in MR._rows_for_variant(model, case, vp, tp, yva, yte)}
    g = lambda pr, k: round(by[pr][k], 3)
    return {"pipeline": "Machine-only", "model": model, "case": case, "sel_seed": seed, "source": "pathB",
            "full_AUROC": g("full", "AUROC"), "full_AUPRC": g("full", "AUPRC"), "full_F1": g("full", "F1"),
            "full_Precision": g("full", "Precision"), "full_Recall": g("full", "Recall"),
            "r1to1_AUROC": g("1:1", "AUROC"), "r1to1_AUPRC": g("1:1", "AUPRC"),
            "r1to2_AUROC": g("1:2", "AUROC"), "r1to2_AUPRC": g("1:2", "AUPRC")}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    arch = {"class": "AttnCNN1D", "channels": dlcfg.CNN_CHANNELS, "n_blocks": dlcfg.CNN_BLOCKS,
            "kernel": dlcfg.CNN_KERNEL, "dropout": dlcfg.CNN_DROPOUT, "attention": dlcfg.ATTENTION,
            "reduce": dlcfg.REDUCE, "n_channels": dlcfg.N_CHANNELS}
    rows = []
    for case in ("1day", "2day"):
        print(f"==== machine {case} ====", flush=True)
        attn_seeds, yva, yte = _attn_perseed(case, device)
        store, yva2, yte2 = _machine_classical(case)
        assert np.array_equal(yte, yte2) and np.array_equal(yva, yva2)
        mean, std = _norm_stats(case)

        # --- Attention-CNN1D: best-val seed ---
        s_a = int(np.argmax([d["vbest"] for d in attn_seeds]))
        a = attn_seeds[s_a]; thr_a = _thr(yva, a["vp"])
        joblib.dump({"kind": "dl_seq_attn", "case": case, "sel_seed": s_a, "in_ch": a["in_ch"],
                     "arch": arch, "state_dict": a["state"], "norm_mean": mean, "norm_std": std,
                     "threshold": thr_a, "predict": "reduce raw [N,6,1440] via window_agg(k=10)+z-score(norm) "
                     "-> [N,144,30]; AttnCNN1D(in_ch).load_state_dict; sigmoid(logit); >=threshold"},
                    os.path.join(MODELS_DIR, f"machine_attention_cnn1d_{case}.joblib"))
        rows.append(_row("Attention-CNN1D", case, a["vp"], a["tp"], yva, yte, s_a))

        # --- Ensemble: best-val seed ---
        ens = [_ens_for_seed(s, store, attn_seeds, yva) for s in range(TE.N_SEEDS)]
        s_e = int(np.argmax([roc_auc_score(yva, v) for v, _, _ in ens]))
        ev, et, keep = ens[s_e]; thr_e = _thr(yva, ev)
        members = []
        for n in keep:
            if n.startswith("cls:"):
                _, fs, m = n.split(":")
                members.append({"name": n, "featset": fs, "model": m,
                                "estimator": store[fs]["models"][m]["est"][s_e], "prep": store[fs]["prep"]})
        a_e = attn_seeds[s_e]
        joblib.dump({"kind": "ensemble_mean", "case": case, "sel_seed": s_e, "keep": keep, "threshold": thr_e,
                     "classical_members": members,
                     "attn_cnn1d": {"in_ch": a_e["in_ch"], "arch": arch, "state_dict": a_e["state"],
                                    "norm_mean": mean, "norm_std": std},
                     "recipe": "mean of member test-probs (each classical: prep->select->predict_proba; "
                     "attn: reduce+norm->sigmoid); threshold on the mean"},
                    os.path.join(MODELS_DIR, f"machine_ensemble_{case}.joblib"))
        rows.append(_row("Ensemble", case, ev, et, yva, yte, s_e))
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "pathB_machine_bestseed.csv"), index=False)
        print(f"  saved attn(seed{s_a}) + ensemble(seed{s_e}) bundles for {case}", flush=True)
    print(f"\nbundles -> {MODELS_DIR}\nmetrics -> {os.path.join(OUT,'pathB_machine_bestseed.csv')}")


if __name__ == "__main__":
    main()
