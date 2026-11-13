"""
make_cohort_tables.py
=========================================================================================
Builds the cohort-description tables for the paper:
  (1) cohort counts per split (patients / circuits / circuit-days / mortality prevalence),
  (2) Table 1 — baseline characteristics at CRRT initiation, stratified by mortality, with
      median [IQR] + Mann-Whitney p (continuous) and n (%) + chi-square p (binary),
  (3) the held-out TEST set under three evaluation protocols — full (imbalanced), balanced
      1:1, and 1:2 — both composition and the best machine-only ensemble's metrics.
Prints all tables as Markdown and writes CSVs to paper_materials/cohort/.
=========================================================================================
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_AS_DIR = os.path.dirname(_HERE)
sys.path.insert(0, _AS_DIR)
sys.path.insert(0, _HERE)
from abl_config import CLINICAL_ONLY                                            # noqa: E402
from subpopulation_analysis import _ensemble_probs                             # noqa: E402

from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             fbeta_score, precision_score, recall_score, precision_recall_curve)

OUT = os.path.join(_HERE, "cohort"); os.makedirs(OUT, exist_ok=True)
CONT = [("PTI_Age", "Age, years"), ("PTI_CrrtInitiationWeight_kg", "Weight, kg"),
        ("PTI_IcuBmi_kg_m2", "BMI, kg/m^2"), ("PTI_CharlsonCoMorbidityScore", "Charlson score"),
        ("PTI_Apache2_total_partial", "APACHE-II (partial)"), ("LAB_LactateUs", "Lactate"),
        ("LAB_AlbuminUs", "Albumin"), ("LAB_INRUs", "INR"), ("LAB_PlateletsUs", "Platelets"),
        ("LAB_pHUs", "pH"), ("LAB_BicarbonateUs", "Bicarbonate"), ("LAB_CreatinineUs", "Creatinine"),
        ("LAB_UreaUs", "Urea"), ("LAB_HemoglobinUs", "Hemoglobin")]
BIN = [("PTI_Gender_Male", "Male sex"), ("PTI_DiabetesMellitus", "Diabetes mellitus"),
       ("PTI_CongestiveHeartFailure", "Congestive heart failure"),
       ("PTI_MildLiverDisease", "Mild liver disease"),
       ("PTI_ModerateOrSevereLiverDisease", "Moderate/severe liver disease"),
       ("PTI_AnyTumour", "Any tumour"), ("PTI_MetastaticTumor", "Metastatic tumour")]


def _full_cohort():
    """All splits' clinical per-day rows, with a 'split' tag (1-day case)."""
    parts = []
    for s in ("train", "val", "test"):
        d = pd.read_csv(os.path.join(CLINICAL_ONLY, "1day", f"{s}.csv")); d["split"] = s
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def cohort_counts():
    rows = []
    for case in ("1day", "2day"):
        for s in ("train", "val", "test"):
            d = pd.read_csv(os.path.join(CLINICAL_ONLY, case, f"{s}.csv"))
            rows.append({"case": case, "split": s, "patients": d["PID"].nunique(),
                         "circuits": d.groupby(["PID", "Visit_no"]).ngroups, "circuit_days": len(d),
                         "deaths": int(d["label"].sum()), "prevalence_%": round(100 * d["label"].mean(), 2)})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "cohort_counts.csv"), index=False)
    return df


def table1():
    """Patient-level baseline (one row per patient = earliest circuit-day), by mortality."""
    d = _full_cohort().sort_values(["PID", "Visit_no", "Day"]).drop_duplicates("PID", keep="first")
    grp = d["Mortality"].astype(int)
    died, surv = d[grp == 1], d[grp == 0]
    rows = [{"variable": "N (patients)", "survivors": f"{len(surv)}", "non-survivors": f"{len(died)}",
             "p": ""}]
    for col, label in CONT:
        if col not in d.columns:
            continue
        a, b = surv[col].dropna(), died[col].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        rows.append({"variable": label,
                     "survivors": f"{a.median():.1f} [{a.quantile(.25):.1f}-{a.quantile(.75):.1f}]",
                     "non-survivors": f"{b.median():.1f} [{b.quantile(.25):.1f}-{b.quantile(.75):.1f}]",
                     "p": f"{p:.3f}" if p >= 0.001 else "<0.001"})
    for col, label in BIN:
        if col not in d.columns:
            continue
        a, b = surv[col].dropna(), died[col].dropna()
        ca, cb = int((a == 1).sum()), int((b == 1).sum())
        tbl = [[ca, len(a) - ca], [cb, len(b) - cb]]
        try:
            p = stats.chi2_contingency(tbl)[1]
            ps = f"{p:.3f}" if p >= 0.001 else "<0.001"
        except ValueError:
            ps = "n/a"
        rows.append({"variable": label, "survivors": f"{ca} ({100*ca/max(len(a),1):.1f}%)",
                     "non-survivors": f"{cb} ({100*cb/max(len(b),1):.1f}%)", "p": ps})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "table1_baseline.csv"), index=False)
    return df


def _metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    return {"AUROC": roc_auc_score(y, p), "AUPRC": average_precision_score(y, p),
            "F1": f1_score(y, pred, zero_division=0), "F2": fbeta_score(y, pred, beta=2, zero_division=0),
            "F3": fbeta_score(y, pred, beta=3, zero_division=0),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0)}


def test_protocols(case="1day", n_iter=100, seed0=0):
    """Composition + best machine-only ensemble metrics under full / 1:1 / 1:2."""
    vprob, tprob, yva, yte, _ = _ensemble_probs(case)
    pr, rc, th = precision_recall_curve(yva, vprob)
    f1 = np.where((pr[:-1] + rc[:-1]) > 0, 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1]), 0)
    thr = float(th[int(np.argmax(f1))])
    pos = np.where(yte == 1)[0]; neg = np.where(yte == 0)[0]
    npos, nneg = len(pos), len(neg)

    comp, perf = [], []
    # full
    comp.append({"protocol": "full (imbalanced)", "N": len(yte), "positives": npos, "negatives": nneg,
                 "prevalence_%": round(100 * npos / len(yte), 2), "iterations": 1})
    mf = _metrics(yte, tprob, thr); mf = {k: round(v, 4) for k, v in mf.items()}
    perf.append({"protocol": "full", **mf})
    # resampled 1:1 and 1:2
    for ratio in (1, 2):
        comp.append({"protocol": f"balanced 1:{ratio}", "N": npos * (1 + ratio), "positives": npos,
                     "negatives": npos * ratio, "prevalence_%": round(100 / (1 + ratio), 2),
                     "iterations": n_iter})
        accs = {m: [] for m in mf}
        for i in range(n_iter):
            rng = np.random.RandomState(seed0 + i)
            sel = np.concatenate([pos, rng.choice(neg, size=min(npos * ratio, nneg), replace=False)])
            mm = _metrics(yte[sel], tprob[sel], thr)
            for m in accs:
                accs[m].append(mm[m])
        perf.append({"protocol": f"1:{ratio}", **{m: round(float(np.mean(v)), 4) for m, v in accs.items()}})
    cdf, pdf = pd.DataFrame(comp), pd.DataFrame(perf)
    cdf.to_csv(os.path.join(OUT, f"test_composition_{case}.csv"), index=False)
    pdf.to_csv(os.path.join(OUT, f"ensemble_by_protocol_{case}.csv"), index=False)
    return cdf, pdf, thr


def _md(df):
    return df.to_markdown(index=False)


if __name__ == "__main__":
    print("### Cohort counts\n"); print(_md(cohort_counts())); print()
    print("### Table 1 — baseline characteristics by mortality\n"); print(_md(table1())); print()
    for case in ("1day", "2day"):
        c, p, thr = test_protocols(case)
        print(f"### Test-set composition ({case})\n"); print(_md(c)); print()
        print(f"### Machine-only ensemble metrics by protocol ({case}, val-F1 thr={thr:.3f})\n")
        print(_md(p)); print()
    print(f"saved CSVs -> {OUT}/")
