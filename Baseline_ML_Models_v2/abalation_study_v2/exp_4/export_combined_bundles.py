"""
export_combined_bundles.py — save externally-runnable bundles for the COMBINED 'ours' models
============================================================================================
Re-runs the combined CNN-fusion + logistic top-4 Stack per seed, saves the BEST-val seed of Attn-CNN
fusion and Ensemble/Stack as self-contained bundles (fusion weights + clinical preprocessor; stack =
member estimators/weights + fitted meta-learner + threshold), and regenerates the pathB combined
metrics FROM the saved run. GPU.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib

_HERE = os.path.dirname(os.path.abspath(__file__))
_AS = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_AS, "exp_3", "results"))
import combined_fusion_ensemble as CF                # noqa: E402
from sklearn.base import clone                       # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score, precision_recall_curve  # noqa: E402
MR = CF.MR
MODELS_DIR = os.path.join(_HERE, "models"); os.makedirs(MODELS_DIR, exist_ok=True)
OUT = os.path.join(_HERE, "results")


def _thr(yva, vp):
    pr, rc, th = precision_recall_curve(yva, vp)
    f1 = np.where((pr[:-1] + rc[:-1]) > 0, 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1]), 0)
    return float(th[int(np.argmax(f1))]) if len(th) else 0.5


def _classical_keep(tables, feature_set):
    """Mirror CF._classical_probs but retain per-seed estimators + the fitted preprocessor."""
    tr, va, te = tables
    feats = [x for x in tr.columns if x not in CF.META]
    Xtr, ytr = tr[feats].to_numpy(float), tr["label"].to_numpy(int)
    Xva, Xte = va[feats].to_numpy(float), te[feats].to_numpy(float)
    imp = CF.SimpleImputer(strategy="median").fit(Xtr)
    Xtr, Xva, Xte = imp.transform(Xtr), imp.transform(Xva), imp.transform(Xte)
    Xtr, Xte, prep = CF.scale_features(Xtr, Xte); Xva = CF.transform_with_preprocessor(prep, Xva)
    cidx = None
    if feature_set == "combined":
        cidx, _ = CF.select_columns(feats, Xtr, ytr, "combined", n_machine=CF.N_MACHINE, n_clinical=CF.N_CLINICAL)
        Xtr, Xva, Xte = Xtr[:, cidx], Xva[:, cidx], Xte[:, cidx]
    out = {}
    for m in CF.MODELS:
        best = CF.CR._tune(m, Xtr, ytr)
        if best is None:
            continue
        ests, vp, tp = [], [], []
        for sd in range(CF.CR.N_SEEDS):
            est = clone(best)
            try:
                est.set_params(random_state=sd)
            except ValueError:
                pass
            est.fit(Xtr, ytr)
            ests.append(est); vp.append(est.predict_proba(Xva)[:, 1]); tp.append(est.predict_proba(Xte)[:, 1])
        out[m] = {"est": ests, "vp": np.array(vp), "tp": np.array(tp)}
    prep_meta = {"feats": feats, "imputer": imp, "scale_prep": prep, "cidx": cidx, "feature_set": feature_set}
    return out, prep_meta


def _row(model, case, vp, tp, yva, yte, seed):
    by = {r["protocol"]: r for r in MR._rows_for_variant(model, case, vp, tp, yva, yte)}
    g = lambda pr, k: round(by[pr][k], 3)
    return {"pipeline": "Combined", "model": model, "case": case, "sel_seed": seed, "source": "pathB",
            "full_AUROC": g("full", "AUROC"), "full_AUPRC": g("full", "AUPRC"), "full_F1": g("full", "F1"),
            "full_Precision": g("full", "Precision"), "full_Recall": g("full", "Recall"),
            "r1to1_AUROC": g("1:1", "AUROC"), "r1to1_AUPRC": g("1:1", "AUPRC"),
            "r1to2_AUROC": g("1:2", "AUROC"), "r1to2_AUPRC": g("1:2", "AUPRC")}


def main():
    farch = {"class": "FusionModel", "CH": CF.CH, "BLOCKS": CF.BLOCKS, "KERNEL": CF.KERNEL, "DROP": CF.DROP,
             "N_CLIN_DL": CF.N_CLIN_DL}
    rows = []
    for case in ("1day", "2day"):
        print(f"==== combined {case} ====", flush=True)
        fv, ft, yva, yte, pid_te, fb = CF._fusion_probs(case, per_seed=True, return_bundle=True)
        comb, comb_prep = _classical_keep([CF._combined_table(case, s) for s in ("train", "val", "test")], "combined")
        clin, clin_prep = _classical_keep([pd.read_csv(os.path.join(CF.CLINICAL_ONLY, case, f"{s}.csv"))
                                           for s in ("train", "val", "test")], "clinical_only")
        S = fv.shape[0]

        def cands_seed(s):
            c = {"dl:cnn_fusion": (fv[s], ft[s])}
            for m, d in comb.items():
                c[f"cls:combined:{m}"] = (d["vp"][s], d["tp"][s])
            for m, d in clin.items():
                c[f"cls:clinical:{m}"] = (d["vp"][s], d["tp"][s])
            return c

        # --- Attn-CNN fusion: best-val seed ---
        s_f = int(np.argmax([roc_auc_score(yva, fv[s]) for s in range(S)]))
        thr_f = _thr(yva, fv[s_f])
        joblib.dump({"kind": "dl_fusion", "case": case, "sel_seed": s_f, "arch": farch,
                     "in_ch": fb["in_ch"], "clin_dim": fb["clin_dim"], "pooling": fb["pooling"],
                     "state_dict": fb["states"][s_f], "clin_imputer": fb["clin_imputer"],
                     "clin_scaler": fb["clin_scaler"], "clin_idx": fb["clin_idx"], "clin_cols": fb["clin_cols"],
                     "threshold": thr_f, "predict": "machine reduced [N,144,30] + clinical(mRMR-40 branch: "
                     "impute->scale->select clin_idx) -> FusionModel; sigmoid; >=threshold"},
                    os.path.join(MODELS_DIR, f"combined_attn_cnn_fusion_{case}.joblib"))
        rows.append(_row("Attn-CNN fusion", case, fv[s_f], ft[s_f], yva, yte, s_f))

        # --- Ensemble/Stack (top-4 by val): best-val seed ---
        stack_seeds = []
        for s in range(S):
            c = cands_seed(s); names = list(c)
            vauc = {n: roc_auc_score(yva, c[n][0]) for n in names}
            top = sorted(vauc, key=vauc.get, reverse=True)[:4]
            Lv = np.column_stack([c[n][0] for n in top]); Lt = np.column_stack([c[n][1] for n in top])
            meta = LogisticRegression(max_iter=5000).fit(Lv, yva)
            stack_seeds.append((meta.predict_proba(Lv)[:, 1], meta.predict_proba(Lt)[:, 1], top, meta))
        s_s = int(np.argmax([roc_auc_score(yva, v) for v, _, _, _ in stack_seeds]))
        sv, st, top, meta = stack_seeds[s_s]; thr_s = _thr(yva, sv)
        members = []
        for n in top:
            if n == "dl:cnn_fusion":
                members.append({"name": n, "kind": "dl_fusion", "state_dict": fb["states"][s_s],
                                "in_ch": fb["in_ch"], "clin_dim": fb["clin_dim"], "arch": farch,
                                "clin_imputer": fb["clin_imputer"], "clin_scaler": fb["clin_scaler"],
                                "clin_idx": fb["clin_idx"], "clin_cols": fb["clin_cols"]})
            else:
                _, grp, m = n.split(":")
                src = comb if grp == "combined" else clin
                prep = comb_prep if grp == "combined" else clin_prep
                members.append({"name": n, "kind": "classical", "group": grp, "model": m,
                                "estimator": src[m]["est"][s_s], "prep": prep})
        joblib.dump({"kind": "logistic_stack", "case": case, "sel_seed": s_s, "top": top,
                     "meta_learner": meta, "members": members, "threshold": thr_s,
                     "recipe": "each member -> test prob; column-stack in `top` order; "
                     "meta_learner.predict_proba; >=threshold"},
                    os.path.join(MODELS_DIR, f"combined_ensemble_stack_{case}.joblib"))
        rows.append(_row("Ensemble/Stack", case, sv, st, yva, yte, s_s))
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "pathB_combined_bestseed.csv"), index=False)
        print(f"  saved fusion(seed{s_f}) + stack(seed{s_s}) bundles for {case}", flush=True)
    print(f"\nbundles -> {MODELS_DIR}\nmetrics -> {os.path.join(OUT,'pathB_combined_bestseed.csv')}")


if __name__ == "__main__":
    main()
