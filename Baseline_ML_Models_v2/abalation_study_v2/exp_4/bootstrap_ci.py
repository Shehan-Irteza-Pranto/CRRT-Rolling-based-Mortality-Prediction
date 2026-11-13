"""
bootstrap_ci.py — patient-clustered bootstrap CIs and paired differences on the held-out TEST set (1-day).
=========================================================================================================
Reviewer request (R2/R3/R4): resample whole PATIENTS (not days) B=2000 times; report 95% percentile CIs for
AUROC / AUPRC of the headline models and PAIRED deltas computed on identical draws:
  full stack − EHR-only stack, full − classical-only combined stack, full − clinical MLP,
  machine-only ensemble − 0.5 (AUROC CI + P(AUROC<=0.5)).
Inputs: per-day probability CSVs (PID,Visit_no,Day,label,prob) written by regenerate_stack_test_probs.py,
ehr_only_stack.py, component_ablation.py, rerun_machine_ensemble_40.py. Joined on (PID,Visit_no,Day).
Outputs: results/bootstrap_ci_1day.csv, results/bootstrap_paired_1day.csv, results/BOOTSTRAP_CI.md. CPU, seconds.
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

_HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(_HERE, "results")
CASE, B, SEED = "1day", 2000, 0
KEY = ["PID", "Visit_no", "Day"]

MODELS = {   # label -> test-probability file
    "Full stack (Transformer fusion + top-3 combined classical)": f"combined_stack_test_probs_40x60_{CASE}.csv",
    "EHR-only stack (clinical MLP-60 + top-3 clinical classical)":
        f"ehr_only_stack_ehr_only_stack_mlp_60_top_3_clinical_classical_mrmr_60_test_probs_{CASE}.csv",
    "Classical-only combined stack (top-3, no Transformer)": f"component_ablation_M0C0K1_test_probs_{CASE}.csv",
    "Clinical MLP-60 (Table III clinical-only)": f"ehr_only_stack_dl_table_iii_clinical_mlp_mrmr_60_reloaded_test_probs_{CASE}.csv",
    "Machine-only Windowed-Transformer Ensemble (mRMR-40)": f"machine_ensemble_40feat_test_probs_{CASE}.csv",
}
NPZ_MODELS = {   # Table III rows whose best-val-seed test probs live in results/probs/*.npz (re-inference archives)
    "Machine-only CNN1D": ("roc_pathA_1day.npz", "machine_only_cnn1d"),
    "Machine-only RNN (BiLSTM/GRU)": ("roc_pathA_1day.npz", "machine_only_rnn"),
    "Machine-only Patch-Transformer": ("roc_pathA_1day.npz", "machine_only_transformer"),
    "Machine-only Attention-CNN1D": ("machine_attn_perseed_1day.npz", "best_val_seed"),
    "Combined CNN1D fusion": ("roc_pathA_1day.npz", "combined_cnn1d"),
    "Combined RNN (BiLSTM/GRU) fusion": ("roc_pathA_1day.npz", "combined_rnn"),
    "Combined Patch-Transformer fusion": ("roc_pathA_1day.npz", "combined_transformer"),
    "Combined Attention-CNN1D fusion": ("roc_pathB_1day.npz", "fusion"),
}
TABLE3_ORDER = [  # (Table III label, model key)
    ("Clinical-only: Clinical MLP", "Clinical MLP-60 (Table III clinical-only)"),
    ("Clinical-only: Clinical Fusion-free Stack (ours)", "EHR-only stack (clinical MLP-60 + top-3 clinical classical)"),
    ("Machine-only: CNN1D", "Machine-only CNN1D"),
    ("Machine-only: RNN (BiLSTM/GRU)", "Machine-only RNN (BiLSTM/GRU)"),
    ("Machine-only: Patch-Transformer", "Machine-only Patch-Transformer"),
    ("Machine-only: Attention-CNN1D", "Machine-only Attention-CNN1D"),
    ("Machine-only: Windowed-Transformer Ensemble (ours)", "Machine-only Windowed-Transformer Ensemble (mRMR-40)"),
    ("Combined: CNN1D fusion", "Combined CNN1D fusion"),
    ("Combined: RNN (BiLSTM/GRU) fusion", "Combined RNN (BiLSTM/GRU) fusion"),
    ("Combined: Patch-Transformer fusion", "Combined Patch-Transformer fusion"),
    ("Combined: Attention-CNN1D fusion", "Combined Attention-CNN1D fusion"),
    ("Combined: Windowed-Transformer Fusion Stack (ours)", "Full stack (Transformer fusion + top-3 combined classical)"),
]
PAIRS = [  # (A, B) -> A − B
    ("Full stack (Transformer fusion + top-3 combined classical)", "EHR-only stack (clinical MLP-60 + top-3 clinical classical)"),
    ("Full stack (Transformer fusion + top-3 combined classical)", "Classical-only combined stack (top-3, no Transformer)"),
    ("Full stack (Transformer fusion + top-3 combined classical)", "Clinical MLP-60 (Table III clinical-only)"),
    ("EHR-only stack (clinical MLP-60 + top-3 clinical classical)", "Clinical MLP-60 (Table III clinical-only)"),
]


def load():
    base = None
    for name, f in MODELS.items():
        d = pd.read_csv(os.path.join(R, f))[KEY + ["label", "prob"]].rename(columns={"prob": name})
        if base is None:
            base = d
        else:
            d = d.rename(columns={"label": "label_" + name})
            base = base.merge(d, on=KEY, how="inner")
            assert (base["label"] == base["label_" + name]).all(), f"label mismatch: {name}"
            base = base.drop(columns=["label_" + name])
    for name, (f, key) in NPZ_MODELS.items():
        z = np.load(os.path.join(R, "probs", f))
        if key == "best_val_seed":
            sd = int(np.argmax([roc_auc_score(z["yva"], v) for v in z["vp_dl"]])); pr = z["tp_dl"][sd]
        else:
            pr = z[key]
        assert len(pr) == len(base) and (z["yte"].astype(int) == base["label"].to_numpy(int)).all(), f"label mismatch: {name}"
        base[name] = pr                       # archives are in the same sorted (PID,Visit_no,Day) test order
    assert len(base) == 887 and base["label"].sum() == 62 and base["PID"].nunique() == 196, \
        f"unexpected join: {len(base)} rows, {base['label'].sum()} pos, {base['PID'].nunique()} pids"
    return base


def main():
    df = load(); y = df["label"].to_numpy(int); pid = df["PID"].to_numpy()
    ALL = list(MODELS) + list(NPZ_MODELS)
    P = {n: df[n].to_numpy(float) for n in ALL}
    pats = np.unique(pid); by = {p: np.where(pid == p)[0] for p in pats}
    rng = np.random.default_rng(SEED)
    auroc = {n: [] for n in ALL}; auprc = {n: [] for n in ALL}; used = 0
    for b in range(B):
        idx = np.concatenate([by[p] for p in rng.choice(pats, len(pats), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        used += 1
        for n in ALL:
            auroc[n].append(roc_auc_score(y[idx], P[n][idx])); auprc[n].append(average_precision_score(y[idx], P[n][idx]))
    auroc = {n: np.array(v) for n, v in auroc.items()}; auprc = {n: np.array(v) for n, v in auprc.items()}
    ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))

    rows = []
    for n in ALL:
        for metric, point, arr in (("AUROC", roc_auc_score(y, P[n]), auroc[n]),
                                   ("AUPRC", average_precision_score(y, P[n]), auprc[n])):
            lo, hi = ci(arr)
            rows.append({"model": n, "metric": metric, "point": round(point, 3), "ci_lo": round(lo, 3),
                         "ci_hi": round(hi, 3), "half_width": round((hi - lo) / 2, 3), "boot_sd": round(float(arr.std()), 3)})
    m = "Machine-only Windowed-Transformer Ensemble (mRMR-40)"
    pairs = []
    for a, b_ in PAIRS:
        for metric, pa, pb, arrA, arrB in (("AUROC", roc_auc_score(y, P[a]), roc_auc_score(y, P[b_]), auroc[a], auroc[b_]),
                                           ("AUPRC", average_precision_score(y, P[a]), average_precision_score(y, P[b_]), auprc[a], auprc[b_])):
            d = arrA - arrB; lo, hi = ci(d)
            pairs.append({"comparison": f"{a}  −  {b_}", "metric": metric, "delta": round(pa - pb, 3),
                          "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
                          "p_boot_one_sided": round(float((d <= 0).mean()), 4)})
    d = auroc[m] - 0.5; lo, hi = ci(d)
    pairs.append({"comparison": f"{m}  −  0.5 (chance)", "metric": "AUROC", "delta": round(roc_auc_score(y, P[m]) - 0.5, 3),
                  "ci_lo": round(lo, 3), "ci_hi": round(hi, 3), "p_boot_one_sided": round(float((d <= 0).mean()), 4)})

    ci_df, pr_df = pd.DataFrame(rows), pd.DataFrame(pairs)
    ci_df.to_csv(os.path.join(R, f"bootstrap_ci_{CASE}.csv"), index=False)
    pr_df.to_csv(os.path.join(R, f"bootstrap_paired_{CASE}.csv"), index=False)
    md = [f"# Patient-clustered bootstrap (test: 887 days, 196 patients, 62 deaths; B={B}, {used} valid draws, seed {SEED})\n",
          "## 95% percentile CIs\n", "| model | metric | point | 95% CI | half-width |", "|---|---|--:|--:|--:|"]
    md += [f"| {r.model} | {r.metric} | {r.point:.3f} | [{r.ci_lo:.3f}, {r.ci_hi:.3f}] | ±{r.half_width:.3f} |" for r in ci_df.itertuples()]
    md += ["\n## Paired differences (identical patient draws)\n", "| comparison | metric | Δ | 95% CI | one-sided bootstrap p (Δ≤0) |", "|---|---|--:|--:|--:|"]
    md += [f"| {r.comparison} | {r.metric} | {r.delta:+.3f} | [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {r.p_boot_one_sided:.4f} |" for r in pr_df.itertuples()]
    t3 = []
    for label, key in TABLE3_ORDER:
        a = ci_df[(ci_df.model == key) & (ci_df.metric == "AUROC")].iloc[0]; b_ = ci_df[(ci_df.model == key) & (ci_df.metric == "AUPRC")].iloc[0]
        t3.append({"Table III row": label, "AUROC": f"{a.point:.3f}", "AUROC 95% CI": f"[{a.ci_lo:.3f}, {a.ci_hi:.3f}]",
                   "AUPRC": f"{b_.point:.3f}", "AUPRC 95% CI": f"[{b_.ci_lo:.3f}, {b_.ci_hi:.3f}]"})
    t3 = pd.DataFrame(t3); t3.to_csv(os.path.join(R, f"bootstrap_table3_{CASE}.csv"), index=False)
    md += ["\n## Table III rows with patient-clustered 95% CIs\n", "| row | AUROC | 95% CI | AUPRC | 95% CI |", "|---|--:|--:|--:|--:|"]
    md += [f"| {r['Table III row']} | {r['AUROC']} | {r['AUROC 95% CI']} | {r['AUPRC']} | {r['AUPRC 95% CI']} |" for _, r in t3.iterrows()]
    open(os.path.join(R, "BOOTSTRAP_CI.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md)); print(f"\nsaved -> {R}/bootstrap_ci_{CASE}.csv, bootstrap_paired_{CASE}.csv, BOOTSTRAP_CI.md")


if __name__ == "__main__":
    main()
