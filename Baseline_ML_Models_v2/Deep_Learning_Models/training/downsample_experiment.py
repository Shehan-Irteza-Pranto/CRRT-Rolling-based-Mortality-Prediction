"""
training/downsample_experiment.py
=================================
Run the full downsampling study: build the 8 downsampled sequence caches, train the
sequence + fusion rungs on each, and compare across variants. Targets the failure modes
the 1440-min streams showed (transformer overfit/collapse, cnn1d/rnn underfit) by feeding
much shorter, smoothed windows.

For each variant it reuses run_all.py UNCHANGED, pointed via env vars at:
  * the variant's downsampled cache   (CRRT_SEQ_CACHE_DIR)
  * an isolated results folder        (CRRT_RESULTS_DIR = results_downsampled/<variant>)
so each variant gets its own raw_results / best_models / curves / report, fully separated
from the full-resolution results/ folder. run_all handles 2-GPU dispatch + aggregate + plots.

Only the sequence-based rungs are run (cnn1d, rnn, transformer, fusion); the MLP rungs use
tabular features and are unaffected by downsampling.

Usage
-----
    python downsample_experiment.py --gpus 0,1                 # all 8 variants
    python downsample_experiment.py --gpus 0,1 --seeds 3       # quicker
    python downsample_experiment.py --gpus 0,1 --variants w10_s10_mean w15_s15_median
    python downsample_experiment.py --skip-build               # caches already built
    python downsample_experiment.py --compare-only             # just rebuild the comparison
"""

import os
import sys
import glob
import argparse
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "data"))

from config import _HERE as PKG_DIR, CASES
import prepare_downsampled as PD

RESULTS_DOWNSAMPLED_ROOT = os.path.join(PKG_DIR, "results_downsampled")
SEQUENCE_RUNGS = ["cnn1d", "rnn", "transformer", "fusion"]   # run_all --only filter (fusion = both encoders)


def train_variant(variant_name, gpus, seeds, cases, stream, n_search=None):
    """Run run_all.py for one variant, pointed at its cache + isolated results dir.
    If n_search is set, first run search_all.py on the variant (HP search), then train
    the winners via run_all --use-search."""
    cache_dir = os.path.join(PD.DOWNSAMPLED_ROOT, variant_name)
    results_dir = os.path.join(RESULTS_DOWNSAMPLED_ROOT, variant_name)
    os.makedirs(results_dir, exist_ok=True)
    env = dict(os.environ, CRRT_SEQ_CACHE_DIR=cache_dir, CRRT_RESULTS_DIR=results_dir)

    print(f"\n############ variant {variant_name}  ->  {results_dir} ############", flush=True)

    if n_search:
        search_argv = [sys.executable, "-u", os.path.join(_HERE, "search_all.py"),
                       "--only", *SEQUENCE_RUNGS, "--cases", *cases, "--stream", stream,
                       "--n-iter", str(n_search)]
        if gpus:
            search_argv += ["--gpus", gpus]
        if subprocess.run(search_argv, env=env).returncode != 0:
            print(f"[downsample] variant {variant_name} SEARCH FAILED")
            return False

    argv = [sys.executable, "-u", os.path.join(_HERE, "run_all.py"),
            "--only", *SEQUENCE_RUNGS, "--cases", *cases, "--stream", stream]
    if gpus:
        argv += ["--gpus", gpus]
    if seeds:
        argv += ["--seeds", str(seeds)]
    if n_search:
        argv += ["--use-search"]

    rc = subprocess.run(argv, env=env).returncode
    if rc != 0:
        print(f"[downsample] variant {variant_name} FAILED (rc={rc})")
    return rc == 0


def build_comparison(variant_names, cases):
    """Cross-variant table of balanced-test AUROC per (rung, variant), incl. the
    full-resolution baseline from results/reports/dl_summary.csv for reference."""
    import pandas as pd

    def _summary(results_dir):
        p = os.path.join(results_dir, "reports", "dl_summary.csv")
        return pd.read_csv(p) if os.path.exists(p) else None

    # column order: full-res baseline first, then the variants
    sources = [("full_res", os.path.join(PKG_DIR, "results"))]
    sources += [(v, os.path.join(RESULTS_DOWNSAMPLED_ROOT, v)) for v in variant_names]

    rows = {}   # (model, branch, case) -> {source: "mean ± std"}
    for src, rdir in sources:
        df = _summary(rdir)
        if df is None:
            continue
        for _, r in df.iterrows():
            m = r.get("balanced_test_AUROC_mean"); s = r.get("balanced_test_AUROC_std")
            if m != m:        # NaN
                continue
            key = (r["model"], r["branch"], r["case"])
            rows.setdefault(key, {})[src] = f"{m:.3f} ± {s:.3f}" if s == s else f"{m:.3f}"

    os.makedirs(RESULTS_DOWNSAMPLED_ROOT, exist_ok=True)
    cols = [s for s, _ in sources]
    # CSV
    import csv
    csv_path = os.path.join(RESULTS_DOWNSAMPLED_ROOT, "comparison_balanced_auroc.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "branch", "case", *cols])
        for (model, branch, case), d in sorted(rows.items()):
            w.writerow([model, branch, case, *[d.get(c, "") for c in cols]])
    # Markdown, one table per case, rungs as rows, variants as columns
    lines = ["# Downsampling study — balanced-test AUROC (mean ± std over seeds)",
             "",
             "Sequence/fusion rungs trained on each downsampled variant vs the full-resolution "
             "(1440-min) baseline. Variant `w{W}_s{S}_{agg}` = W-min windows, stride S "
             "(S=W non-overlap, S=W/2 ~50% overlap), aggregated by mean/median.",
             ""]
    for case in cases:
        lines += [f"## {case}", "", "| rung | " + " | ".join(cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for (model, branch, c), d in sorted(rows.items()):
            if c != case:
                continue
            rung = f"{model}/{branch}"
            lines.append(f"| {rung} | " + " | ".join(d.get(col, "") for col in cols) + " |")
        lines.append("")
    md_path = os.path.join(RESULTS_DOWNSAMPLED_ROOT, "COMPARISON.md")
    with open(md_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"[downsample] wrote {csv_path}\n[downsample] wrote {md_path}")


def main():
    ap = argparse.ArgumentParser(description="Downsampling study across 8 variants")
    ap.add_argument("--gpus", default=None, help="comma-separated GPU ids, e.g. '0,1'")
    ap.add_argument("--seeds", type=int, default=None, help="first N config SEEDS")
    ap.add_argument("--cases", nargs="+", default=CASES)
    ap.add_argument("--variants", nargs="+", default=None,
                    help=f"subset of {[v['name'] for v in PD.VARIANTS]}")
    ap.add_argument("--stream", default="denoised")
    ap.add_argument("--skip-build", action="store_true", help="caches already built")
    ap.add_argument("--compare-only", action="store_true", help="only (re)build the comparison")
    ap.add_argument("--search", type=int, default=None, metavar="N_ITER",
                    help="per-variant HP search with N_ITER configs before training (--use-search)")
    a = ap.parse_args()

    variant_names = a.variants or [v["name"] for v in PD.VARIANTS]

    if a.compare_only:
        build_comparison(variant_names, a.cases)
        return

    if not a.skip_build:
        print("=== Stage 1/3: building downsampled caches ===")
        PD.build_all(variant_names, a.cases, a.stream)

    stage2 = "search + train" if a.search else "training"
    print(f"\n=== Stage 2/3: {stage2} sequence rungs per variant ===")
    for v in variant_names:
        train_variant(v, a.gpus, a.seeds, a.cases, a.stream, n_search=a.search)

    print("\n=== Stage 3/3: cross-variant comparison ===")
    build_comparison(variant_names, a.cases)
    print("\n=== downsampling study complete ===")


if __name__ == "__main__":
    main()
