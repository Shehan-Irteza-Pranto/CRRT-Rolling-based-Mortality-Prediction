"""
run_dl_benchmark.py
===================
Deep-learning benchmark across BOTH GPUs (job-level parallelism).

Enumerates DL configs (pipeline x scope x encoder x case x seed) and runs them as
subprocesses of models/dl_train.py, keeping ONE job per GPU running at a time
(2 concurrent on a 2-GPU box) by pinning CUDA_VISIBLE_DEVICES per process. Small
models -> distributing whole jobs across GPUs beats DataParallel within a model.

Each job writes results/dl_rows/<job>.csv; on completion they are concatenated
into results/dl_benchmark.csv.

Usage
-----
    python run_dl_benchmark.py --gpus 0 1 --seeds 5 --scopes per_day agg
    python run_dl_benchmark.py --gpus 0 1 --seeds 1 --encoders cnn1d --epochs 3   # quick
"""

import os
import sys
import time
import argparse
import subprocess
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from abl_config import RESULTS_DIR                                   # noqa: E402

DL_TRAIN = os.path.join(_HERE, "models", "dl_train.py")
DL_CLINICAL = os.path.join(_HERE, "models", "dl_clinical_mlp.py")
PY = sys.executable


def enumerate_jobs(pipelines, scopes, encoders, cases, seeds, variant, peak, epochs,
                   search_iter, rows_dir, clinical_mlp=True):
    tag = f"{variant}_{peak}"
    jobs = []
    for pl in pipelines:
        for sc in scopes:
            for enc in encoders:
                for case in cases:
                    for seed in seeds:
                        name = f"{pl}_{sc}_{enc}_{case}_s{seed}_{tag}"
                        out = os.path.join(rows_dir, name + ".csv")
                        cmd = [PY, DL_TRAIN, "--pipeline", pl, "--scope", sc,
                               "--encoder", enc, "--case", case, "--seed", str(seed),
                               "--variant", variant, "--peak", peak,
                               "--search-iter", str(search_iter), "--out", out]
                        if epochs is not None:
                            cmd += ["--epochs", str(epochs)]
                        jobs.append({"name": name, "cmd": cmd, "out": out})
    # clinical_only DL = MLP on the mRMR clinical features (tabular, no encoder)
    if clinical_mlp:
        for sc in scopes:
            for case in cases:
                for seed in seeds:
                    name = f"clinical_only_{sc}_mlp_{case}_s{seed}_{tag}"
                    out = os.path.join(rows_dir, name + ".csv")
                    cmd = [PY, DL_CLINICAL, "--scope", sc, "--case", case, "--seed", str(seed),
                           "--variant", variant, "--peak", peak, "--search-iter", str(search_iter),
                           "--out", out]
                    if epochs is not None:
                        cmd += ["--epochs", str(epochs)]
                    jobs.append({"name": name, "cmd": cmd, "out": out})
    return jobs


def run(jobs, gpus, resume=True):
    """Schedule jobs across GPUs: at most one running per GPU.

    RESUME: jobs whose per-job CSV already exists (non-empty) are skipped, so an
    interrupted benchmark continues where it stopped.
    """
    done, failed = [], []
    pending = []
    for j in jobs:
        if resume and os.path.exists(j["out"]) and os.path.getsize(j["out"]) > 0:
            done.append(j["name"])
            print(f"[skip-done] {j['name']}")
        else:
            pending.append(j)
    running = {}          # gpu -> (proc, job)
    while pending or running:
        for g in gpus:
            if g not in running and pending:
                job = pending.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
                log = open(job["out"] + ".log", "w")
                proc = subprocess.Popen(job["cmd"], env=env, stdout=log, stderr=subprocess.STDOUT)
                running[g] = (proc, job, log)
                print(f"[gpu{g}] start {job['name']}")
        time.sleep(2)
        for g in list(running):
            proc, job, log = running[g]
            if proc.poll() is not None:
                log.close()
                ok = (proc.returncode == 0 and os.path.exists(job["out"]))
                (done if ok else failed).append(job["name"])
                print(f"[gpu{g}] {'done' if ok else 'FAIL'} {job['name']}"
                      f"{'' if ok else ' (see ' + job['out'] + '.log)'}")
                del running[g]
    return done, failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--pipelines", nargs="+", default=["machine_only", "combined"])
    ap.add_argument("--scopes", nargs="+", default=["per_day", "agg"])
    ap.add_argument("--encoders", nargs="+", default=["cnn1d", "rnn", "transformer"])
    ap.add_argument("--cases", nargs="+", default=["1day", "2day"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--variant", default="current")
    ap.add_argument("--peak", default="denoised", choices=["denoised", "noised"])
    ap.add_argument("--search-iter", type=int, default=12,
                    help="randomized HP sets per DL config (0 = fixed defaults)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--fresh", action="store_true", help="ignore existing per-job CSVs (no resume)")
    ap.add_argument("--out", default=None,
                    help="defaults to results/dl_<variant>_<peak>.csv (per-peak, no collision)")
    a = ap.parse_args()
    if a.out is None:
        a.out = os.path.join(RESULTS_DIR, f"dl_{a.variant}_{a.peak}.csv")

    rows_dir = os.path.join(RESULTS_DIR, "dl_rows")
    os.makedirs(rows_dir, exist_ok=True)
    jobs = enumerate_jobs(a.pipelines, a.scopes, a.encoders, a.cases,
                          list(range(a.seeds)), a.variant, a.peak, a.epochs, a.search_iter, rows_dir)
    print(f"[run_dl_benchmark] {len(jobs)} jobs across GPUs {a.gpus}")
    done, failed = run(jobs, a.gpus, resume=not a.fresh)

    rows = [pd.read_csv(os.path.join(rows_dir, n + ".csv"))
            for n in done if os.path.exists(os.path.join(rows_dir, n + ".csv"))]
    if rows:
        df = pd.concat(rows, ignore_index=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        df.to_csv(a.out, index=False)
        print(f"\n[run_dl_benchmark] wrote {a.out}  ({len(df)} rows; {len(failed)} failed)")
    else:
        print(f"\n[run_dl_benchmark] no rows produced ({len(failed)} failed)")


if __name__ == "__main__":
    main()
