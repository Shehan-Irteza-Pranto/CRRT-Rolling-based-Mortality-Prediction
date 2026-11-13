"""
training/eval_balanced_resample.py
==================================
Repeated balanced-subsampling evaluation — robustness check on top of the single fixed
balanced test set used by the main report.

The main protocol (common/metrics.py::create_balanced_test) keeps ALL positives and draws
the negatives ONCE with a fixed seed. The balanced-test number therefore depends on which
negatives happened to be drawn. This script instead repeats the balanced draw K times
(all positives + a fresh random negative subsample each time), recomputes the metrics on
every draw, and reports the spread — separating two sources of variance:

  * model variance       — std across the 5 training seeds (of each seed's mean-over-draws)
  * negative-draw luck    — std across the K draws within a seed (averaged over seeds)

It reads ONLY the saved test predictions (results/predictions/*.npz: y_true, y_score,
validation-derived threshold), so no retraining is needed and nothing is overwritten.
Outputs go to NEW files in the report dir:

  results/reports/balanced_resampled_metrics.csv   (full breakdown, all metrics)
  results/reports/balanced_resampled.md            (readable per-case tables)

Env-overridable (CRRT_RESULTS_DIR) like the rest of the pipeline, so it also works on any
downsampled variant's predictions.

Usage
-----
    python eval_balanced_resample.py                 # K=100 over results/predictions/
    python eval_balanced_resample.py --k 200
    python eval_balanced_resample.py --threshold half # use 0.5 instead of the val threshold
"""

import os
import sys
import glob
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "common"))
from config import PRED_DIR, REPORT_DIR, RANDOM_STATE
from metrics import compute_metrics

METRIC_KEYS = ["AUROC", "AUPRC", "F1", "Precision", "Recall", "Balanced_Accuracy"]


def _one_seed_draws(y_true, y_score, thr, k):
    """K balanced draws (all positives + fresh negative subsample) -> list of metric dicts.
    Draw j uses seed RANDOM_STATE+j, so the same K negative-subsets are reused across model
    seeds (a controlled comparison of draw vs model variance)."""
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    n_pos = len(pos)
    if n_pos == 0 or len(neg) == 0:
        return []
    size = min(n_pos, len(neg))
    out = []
    for j in range(k):
        rng = np.random.RandomState(RANDOM_STATE + j)
        idx = np.concatenate([pos, rng.choice(neg, size=size, replace=False)])
        yt, ys = y_true[idx], y_score[idx]
        yp = (ys >= thr).astype(int)
        out.append(compute_metrics(yt, yp, ys))
    return out


def evaluate(k, threshold_mode):
    files = sorted(glob.glob(os.path.join(PRED_DIR, "*.npz")))
    if not files:
        print(f"No predictions in {PRED_DIR}. Run training first.")
        return None

    # group prediction files by (tag, case)
    groups = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        groups.setdefault((str(d["tag"]), str(d["case"])), []).append(f)

    rows = []
    for (tag, case), gfiles in sorted(groups.items()):
        # per metric: list of per-seed (mean_over_draws, std_over_draws), and all raw values
        per_seed_mean = {m: [] for m in METRIC_KEYS}
        per_seed_dstd = {m: [] for m in METRIC_KEYS}
        all_vals = {m: [] for m in METRIC_KEYS}
        n_pos = n_neg = None
        for f in gfiles:
            d = np.load(f, allow_pickle=True)
            y_true, y_score = d["y_true"], d["y_score"]
            thr = 0.5 if threshold_mode == "half" else float(d["threshold"])
            n_pos = int(y_true.sum()); n_neg = int((1 - y_true).sum())
            draws = _one_seed_draws(y_true, y_score, thr, k)
            for m in METRIC_KEYS:
                vals = np.array([dr[m] for dr in draws], dtype=float)
                vals = vals[~np.isnan(vals)]
                if len(vals):
                    per_seed_mean[m].append(vals.mean())
                    per_seed_dstd[m].append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
                    all_vals[m].extend(vals.tolist())

        for m in METRIC_KEYS:
            seed_means = np.array(per_seed_mean[m], dtype=float)
            draw_stds = np.array(per_seed_dstd[m], dtype=float)
            allv = np.array(all_vals[m], dtype=float)
            if not len(allv):
                continue
            rows.append({
                "model": tag, "case": case, "metric": m,
                "mean": round(float(seed_means.mean()), 4),
                "std_total": round(float(allv.std(ddof=1)), 4),         # all K x seeds draws
                "std_model_seed": round(float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0, 4),
                "std_negdraw": round(float(draw_stds.mean()), 4),       # negative-draw luck
                "K": k, "n_seeds": len(seed_means),
                "n_pos": n_pos, "n_neg_pool": n_neg, "threshold": threshold_mode,
            })
    return rows


def write_reports(rows, k, threshold_mode):
    import pandas as pd
    os.makedirs(REPORT_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    csv_path = os.path.join(REPORT_DIR, "balanced_resampled_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"[resample] wrote {csv_path}  ({len(df)} rows)")

    def cell(r):
        return f"{r['mean']:.3f} ± {r['std_total']:.3f}"

    thr_note = "validation-derived F1 threshold" if threshold_mode != "half" else "threshold 0.5"
    lines = [f"# Repeated balanced-subsampling evaluation (K={k} draws)",
             "",
             f"All positives kept; negatives re-drawn **{k}** times (fresh random subsample each "
             f"draw). Metrics at the {thr_note}, recomputed per draw. `mean ± std` below is over "
             f"all K×seeds draws. Deep rungs only (classical predictions are not saved here).",
             "",
             "Variance is decomposed in the CSV: `std_model_seed` (across the 5 training seeds) "
             "vs `std_negdraw` (across the K negative draws within a seed = negative-draw luck).",
             ""]
    for case in sorted({r["case"] for r in rows}):
        sub = [r for r in rows if r["case"] == case]
        lines += [f"## {case}", "",
                  "| model | " + " | ".join(METRIC_KEYS) + " |",
                  "|---|" + "---|" * len(METRIC_KEYS)]
        for model in sorted({r["model"] for r in sub}):
            cells = []
            for m in METRIC_KEYS:
                r = next((x for x in sub if x["model"] == model and x["metric"] == m), None)
                cells.append(cell(r) if r else "")
            lines.append(f"| {model} | " + " | ".join(cells) + " |")
        lines.append("")
        # AUROC variance decomposition (the headline question)
        lines += ["**AUROC variance decomposition** — model (seed) vs negative-draw luck:", "",
                  "| model | AUROC mean | std (model/seed) | std (neg-draw) |",
                  "|---|---|---|---|"]
        for model in sorted({r["model"] for r in sub}):
            r = next((x for x in sub if x["model"] == model and x["metric"] == "AUROC"), None)
            if r:
                lines.append(f"| {model} | {r['mean']:.3f} | {r['std_model_seed']:.3f} | {r['std_negdraw']:.3f} |")
        lines.append("")
    md_path = os.path.join(REPORT_DIR, "balanced_resampled.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"[resample] wrote {md_path}")


def main():
    ap = argparse.ArgumentParser(description="Repeated balanced-subsampling evaluation")
    ap.add_argument("--k", type=int, default=100, help="number of balanced draws (default 100)")
    ap.add_argument("--threshold", choices=["val", "half"], default="val",
                    help="'val' = saved validation-derived F1 threshold; 'half' = 0.5")
    a = ap.parse_args()
    print(f"=== repeated balanced subsampling: K={a.k}, threshold={a.threshold}, "
          f"predictions={PRED_DIR} ===")
    rows = evaluate(a.k, a.threshold)
    if rows:
        write_reports(rows, a.k, a.threshold)


if __name__ == "__main__":
    main()
