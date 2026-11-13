"""
rerun_machine_bestseed.py — best-seed (by val AUROC) for the MACHINE-only 'ours' DL rows
========================================================================================
Re-runs the machine attention-CNN1D and the machine Ensemble PER SEED (exp_3 averages them) and picks,
per model, the seed with the highest VALIDATION AUROC — then reports that seed's full / 1:1 / 1:2 metrics
at the max-F1-on-val threshold. Produces the two 'ours' machine rows:
  * Attention-CNN1D = exp_3/dl AttnCNN1D (masked attention pooling over the window-reduced raw sequence)
  * Ensemble        = mean of {top-3 classical families by val} + attn-CNN1D  (ensemble:mean, per exp_3)
Reuses exp_3/dl/train_eval.py (attn-CNN per seed) + exp_3/features/classical_rich.py (per-seed classical
via _multiseed_probs, which already returns [N_SEEDS, n]). GPU.  Writes pathB_machine_bestseed.csv.
"""
import os
import sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_AS = os.path.dirname(_HERE)
# path order: exp_3/dl must be FRONT when train_eval imports dl_data (models/ also has a dl_data.py)
for p in [os.path.join(os.path.dirname(_AS), "Deep_Learning_Models", "models"),
          os.path.join(os.path.dirname(_AS), "Machine_Clinical_Models"),
          _AS, os.path.join(_AS, "models"),
          os.path.join(_AS, "exp_3", "features"), os.path.join(_AS, "exp_3", "results"),
          os.path.join(_AS, "exp_3", "dl")]:
    sys.path.insert(0, p)
import train_eval as TE                              # noqa: E402  (exp_3/dl -> caches exp_3/dl/dl_data)
import classical_rich as CR                          # noqa: E402  (exp_3/features)
import make_resampled_variants as MR                 # noqa: E402
from sklearn.metrics import roc_auc_score            # noqa: E402
import torch                                         # noqa: E402

OUT = os.path.join(_HERE, "results"); os.makedirs(os.path.join(OUT, "probs"), exist_ok=True)
TOP_K_CLASSICAL = 3                                  # matches exp_3/ensemble/ensemble.py


def _machine_classical_perseed(case):
    """Per-seed val/test probs for every base_ and rich_ classical model (mirrors classical_rich prep)."""
    out = {}
    yva = yte = None
    for fs in ("base", "rich"):
        tr, va, te, feats = CR._load(fs, case)
        Xtr, ytr = CR._xy(tr, feats); Xva, yva = CR._xy(va, feats); Xte, yte = CR._xy(te, feats)
        imp = CR.SimpleImputer(strategy="median").fit(Xtr)
        Xtr, Xva, Xte = imp.transform(Xtr), imp.transform(Xva), imp.transform(Xte)
        Xtr_sc, Xte_sc, prep = CR.scale_features(Xtr, Xte)
        Xva_sc = CR.transform_with_preprocessor(prep, Xva)
        chosen = CR.mrmr_select(Xtr_sc, ytr, list(feats), min(CR.N_MRMR, Xtr_sc.shape[1]))
        cidx = [feats.index(c) for c in chosen]
        Xtr_s, Xva_s, Xte_s = Xtr_sc[:, cidx], Xva_sc[:, cidx], Xte_sc[:, cidx]
        for m in CR.MODELS:
            best = CR._tune(m, Xtr_s, ytr)
            if best is None:
                continue
            vp, tp = CR._multiseed_probs(best, Xtr_s, ytr, Xva_s, Xte_s)     # [N_SEEDS, n]
            out[f"cls:{fs}:{m}"] = (vp, tp)
    return out, yva.astype(int), yte.astype(int)


def _attn_perseed(case, device):
    """Per-seed attn-CNN1D val/test probs via exp_3/dl train_one (no averaging)."""
    Xva, mva, yva_t, _, _ = TE._load("val", case)
    Xte, mte, yte_t, _, _ = TE._load("test", case)
    vps, tps = [], []
    for sd in range(TE.N_SEEDS):
        model, vbest = TE.train_one(case, sd, device)
        vps.append(TE._infer(model, Xva, mva, device)); tps.append(TE._infer(model, Xte, mte, device))
        print(f"  [attn-cnn {case}] seed {sd}: val {vbest:.4f}", flush=True)
    return np.array(vps), np.array(tps), yva_t.numpy().astype(int), yte_t.numpy().astype(int)


def _ensemble_seed(s, cls, vp_dl, tp_dl, yva):
    """ensemble:mean for seed s = mean of {DL + top-3 classical families by val AUROC}."""
    cands = {n: (vp[s], tp[s]) for n, (vp, tp) in cls.items()}
    cands["dl:attn_cnn1d"] = (vp_dl[s], tp_dl[s])
    vauc = {n: roc_auc_score(yva, v) for n, (v, _) in cands.items()}
    keep, seen = [], set()
    for n in sorted(vauc, key=vauc.get, reverse=True):
        fam = n.split(":")[-1]
        if n.startswith("dl:") or (fam not in seen and len([k for k in keep if k.startswith("cls:")]) < TOP_K_CLASSICAL):
            keep.append(n); seen.add(fam)
    if "dl:attn_cnn1d" not in keep:
        keep.append("dl:attn_cnn1d")
    Vv = np.column_stack([cands[n][0] for n in keep]); Vt = np.column_stack([cands[n][1] for n in keep])
    return Vv.mean(1), Vt.mean(1)


def _pick_and_score(label, pipe, case, seed_probs, yva, yte):
    vaucs = [roc_auc_score(yva, vp) for vp, _ in seed_probs]
    s = int(np.argmax(vaucs)); vp, tp = seed_probs[s]
    by = {r["protocol"]: r for r in MR._rows_for_variant(label, case, vp, tp, yva, yte)}
    out = {"pipeline": pipe, "model": label, "case": case, "sel_seed": s,
           "best_val_auroc": round(float(vaucs[s]), 4), "source": "pathB",
           "seedmean_AUROC": round(float(np.mean([roc_auc_score(yte, t) for _, t in seed_probs])), 3),
           "full_AUROC": round(by["full"]["AUROC"], 3), "full_AUPRC": round(by["full"]["AUPRC"], 3),
           "full_F1": round(by["full"]["F1"], 3), "full_Precision": round(by["full"]["Precision"], 3),
           "full_Recall": round(by["full"]["Recall"], 3),
           "r1to1_AUROC": round(by["1:1"]["AUROC"], 3), "r1to1_AUPRC": round(by["1:1"]["AUPRC"], 3),
           "r1to2_AUROC": round(by["1:2"]["AUROC"], 3), "r1to2_AUPRC": round(by["1:2"]["AUPRC"], 3)}
    print(f"[ok] {case} {label:18s} best-val seed {s} (val {vaucs[s]:.3f}) | full AUROC {out['full_AUROC']} "
          f"P {out['full_Precision']} R {out['full_Recall']} (seed-mean {out['seedmean_AUROC']})", flush=True)
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    rows = []
    for case in ("1day", "2day"):
        print(f"==== machine {case} ====", flush=True)
        vp_dl, tp_dl, yva_d, yte_d = _attn_perseed(case, device)
        cls, yva_c, yte_c = _machine_classical_perseed(case)
        assert np.array_equal(yte_d, yte_c) and np.array_equal(yva_d, yva_c), "attn/classical order mismatch"
        yva, yte = yva_c, yte_c
        np.savez(os.path.join(OUT, "probs", f"machine_attn_perseed_{case}.npz"),
                 vp_dl=vp_dl, tp_dl=tp_dl, yva=yva, yte=yte)
        attn_seeds = [(vp_dl[s], tp_dl[s]) for s in range(len(vp_dl))]
        ens_seeds = [_ensemble_seed(s, cls, vp_dl, tp_dl, yva) for s in range(len(vp_dl))]
        rows.append(_pick_and_score("Attention-CNN1D", "Machine-only", case, attn_seeds, yva, yte))
        rows.append(_pick_and_score("Ensemble", "Machine-only", case, ens_seeds, yva, yte))
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "pathB_machine_bestseed.csv"), index=False)
    print(f"\nsaved -> {os.path.join(OUT, 'pathB_machine_bestseed.csv')}")


if __name__ == "__main__":
    main()
