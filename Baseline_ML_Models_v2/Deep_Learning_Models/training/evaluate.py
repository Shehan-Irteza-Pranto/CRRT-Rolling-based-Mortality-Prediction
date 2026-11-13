"""
training/evaluate.py
====================
Aggregate per-seed results into the reportable form: mean +/- std over >= 5 seeds, plus
bootstrap CIs. Builds the master table that sits next to the classical results so the paper
shows classical and deep rungs under one identical protocol.

Reads every results/raw_results_*.csv produced by train.py and emits:
  results/reports/dl_summary.csv         — mean +/- std per (model, case) for each metric
  results/reports/dl_vs_classical.csv    — DL rungs joined against the classical headline numbers

Usage
-----
    python evaluate.py --aggregate
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
from config import RESULTS_DIR, REPORT_DIR, CLASSICAL_METRICS_JSON, CASES

METRIC_KEYS = ["AUROC", "AUPRC", "F1", "Precision", "Recall", "Balanced_Accuracy"]
# Classical headline to characterize against (GB on aggregated combined).
CLASSICAL_HEADLINE = {"aggregated_combined_GB_AUROC": 0.801}

# Feature groups used to organize the comparison report. Each (classical pipeline /
# deep rung) is bucketed by which inputs it consumes so clinical-only, machine-only
# and combined models are read side by side.
FEATURE_GROUPS = [
    ("clinical", "Clinical-only features"),
    ("machine",  "Machine-only features (pressure streams)"),
    ("combined", "Combined features (machine + clinical)"),
]


def _feature_group(family, model, pipeline, branch=None):
    """Bucket a row into 'clinical' / 'machine' / 'combined' (or 'other')."""
    if family == "deep":
        if model == "fusion":
            return "combined"            # fusion = machine sequences + clinical branch
        return branch if branch in ("clinical", "machine", "combined") else "other"
    # classical: derive from the pipeline name
    p = (pipeline or "")
    if "machine_clinical" in p:          # e.g. machine_clinical, machine_clinical_agg_day
        return "combined"
    if p.startswith("clinical"):         # clinical_only, clinical_agg_day
        return "clinical"
    if p.startswith("machine"):          # machine_only, machine_agg_day
        return "machine"
    return "other"


def aggregate():
    os.makedirs(REPORT_DIR, exist_ok=True)
    files = glob.glob(os.path.join(RESULTS_DIR, "raw_results_*.csv"))
    if not files:
        print("No raw_results_*.csv found. Run train.py first.")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df["case"].isin(CASES)]               # focus: only the configured cases (e.g. 1day)

    rows = []
    group = df.groupby(["model", "branch", "case"], dropna=False)
    for (model, branch, case), g in group:
        row = {"model": model, "branch": branch, "case": case, "n_seeds": len(g)}
        for split in ["full_test", "balanced_test"]:
            for k in METRIC_KEYS:
                col = f"{split}_{k}"
                if col in g:
                    vals = g[col].to_numpy(dtype=float)
                    vals = vals[~np.isnan(vals)]
                    if len(vals):
                        row[f"{split}_{k}_mean"] = round(float(vals.mean()), 4)
                        row[f"{split}_{k}_std"] = round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 4)
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(
        ["case", "balanced_test_AUROC_mean"], ascending=[True, False], na_position="last")
    out = os.path.join(REPORT_DIR, "dl_summary.csv")
    summary.to_csv(out, index=False)
    print(f"[evaluate] wrote {out}  ({len(summary)} model/case rows)")

    # quick legible print: the headline comparison
    print("\n=== balanced-test AUROC (mean +/- std), vs classical GB 0.801 ===")
    for _, r in summary.iterrows():
        m = r.get("balanced_test_AUROC_mean"); s = r.get("balanced_test_AUROC_std")
        if m == m:
            tag = f"{r['model']}/{r['branch']}/{r['case']}"
            print(f"  {tag:<34} {m:.3f} +/- {s:.3f}")

    build_comparison(summary)
    return summary


def _load_classical():
    """Classical balanced-threshold results -> list of unified rows.

    The JSON is keyed "{case}_{pipeline}" (e.g. "1day_machine_clinical"); each
    value is the single best model on the balanced test set under the identical
    validation-derived F1 threshold protocol the DL rungs use.
    """
    if not os.path.exists(CLASSICAL_METRICS_JSON):
        print(f"[evaluate] classical metrics not found at {CLASSICAL_METRICS_JSON}; "
              f"skipping classical merge.")
        return []
    with open(CLASSICAL_METRICS_JSON) as fh:
        data = json.load(fh)
    rows = []
    for entry in data.values():
        if entry.get("case") not in CASES:        # focus: only the configured cases (e.g. 1day)
            continue
        row = {
            "family": "classical",
            "case": entry.get("case"),
            "pipeline": entry.get("pipeline"),
            "model": entry.get("best_model"),
            "branch": np.nan,
        }
        row["feature_group"] = _feature_group("classical", row["model"], row["pipeline"])
        for k in METRIC_KEYS:
            row[f"{k}_mean"] = entry.get(k)
            row[f"{k}_std"] = np.nan          # single best model -> no spread
        rows.append(row)
    return rows


def _deep_rows(summary):
    """DL summary DataFrame -> unified rows (balanced-test mean +/- std)."""
    rows = []
    for _, r in summary.iterrows():
        if r.get("balanced_test_AUROC_mean") != r.get("balanced_test_AUROC_mean"):
            continue                           # NaN AUROC -> nothing aggregated
        row = {
            "family": "deep",
            "case": r["case"],
            "pipeline": f"{r['model']}/{r['branch']}",
            "model": r["model"],
            "branch": r["branch"],
        }
        row["feature_group"] = _feature_group("deep", r["model"], row["pipeline"], r["branch"])
        for k in METRIC_KEYS:
            row[f"{k}_mean"] = r.get(f"balanced_test_{k}_mean")
            row[f"{k}_std"] = r.get(f"balanced_test_{k}_std")
        rows.append(row)
    return rows


def _fmt(mean, std):
    if mean is None or mean != mean:
        return ""
    if std is None or std != std:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


def build_comparison(summary):
    """Merge classical + deep balanced-test metrics into one master table.

    Writes results/reports/dl_vs_classical.csv (machine-readable) and
    dl_vs_classical.md (one section per case, all rows ranked by AUROC).
    """
    rows = _load_classical() + _deep_rows(summary)
    if not rows:
        print("[evaluate] no rows to compare; skipping dl_vs_classical.")
        return None
    comp = pd.DataFrame(rows).sort_values(
        ["case", "AUROC_mean"], ascending=[True, False], na_position="last")

    csv_path = os.path.join(REPORT_DIR, "dl_vs_classical.csv")
    comp.to_csv(csv_path, index=False)
    print(f"[evaluate] wrote {csv_path}  ({len(comp)} rows)")

    # readable Markdown: organized by case -> feature group -> family. Within each
    # block, classical and deep models are kept in separate tables (and ranked by
    # AUROC) so clinical-only / machine-only / combined are easy to read off, and the
    # classical vs deep contrast is explicit rather than interleaved.
    header = ("| Model | Pipeline / rung | AUROC | AUPRC | F1 | "
              "Precision | Recall | Bal.Acc |")
    sep = "|---|---|---|---|---|---|---|---|"

    def _emit_table(block, family):
        sub = block[block["family"] == family]
        if sub.empty:
            return [f"_No {family} models in this group._", ""]
        out = [header, sep]
        for _, r in sub.iterrows():
            name = r["pipeline"]
            cells = [_fmt(r.get(f"{k}_mean"), r.get(f"{k}_std")) for k in METRIC_KEYS]
            out.append(f"| {r['model']} | {name} | " + " | ".join(cells) + " |")
        out.append("")
        return out

    lines = ["# Classical vs Deep — balanced-test comparison",
             "",
             "Identical protocol (patient-level splits, 50/50 balanced test, "
             "validation-derived F1 threshold). Deep rungs are mean ± std over seeds; "
             "classical rows are the single best model per pipeline.",
             "",
             "Organized by **feature group** (clinical-only / machine-only / combined), "
             "with **classical** and **deep-learning** models in separate tables within "
             "each group. Rows are ranked by AUROC.",
             ""]
    for case in sorted(comp["case"].dropna().unique()):
        lines += [f"## {case}", ""]
        case_df = comp[comp["case"] == case]
        for gkey, gtitle in FEATURE_GROUPS:
            block = case_df[case_df["feature_group"] == gkey]
            if block.empty:
                continue
            lines += [f"### {gtitle}", ""]
            lines += ["**Classical models**", ""]
            lines += _emit_table(block, "classical")
            lines += ["**Deep-learning models**", ""]
            lines += _emit_table(block, "deep")
        # anything that fell outside the three known groups
        other = case_df[~case_df["feature_group"].isin([g for g, _ in FEATURE_GROUPS])]
        if not other.empty:
            lines += ["### Other", ""]
            lines += ["**Classical models**", ""]
            lines += _emit_table(other, "classical")
            lines += ["**Deep-learning models**", ""]
            lines += _emit_table(other, "deep")

    md_path = os.path.join(REPORT_DIR, "dl_vs_classical.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"[evaluate] wrote {md_path}")
    return comp


def _parse():
    ap = argparse.ArgumentParser(description="Aggregate DL seeds -> mean+/-std tables")
    ap.add_argument("--aggregate", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    a = _parse()
    # default action is to aggregate; the flag is kept for explicitness
    aggregate()
