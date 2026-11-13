"""
training/search_all.py
======================
Run the per-model hyperparameter search for every rung, dispatched across GPUs. Mirrors
run_all.py's rung list + GPU dispatch, but invokes search.py (random search over
config.SEARCH_SPACE, scored on validation AUROC) instead of train.py. Writes one
results/search_<tag>_<case>.json per rung, whose `best.hp` run_all.py --use-search then
trains with over all seeds.

Env-overridable (CRRT_SEQ_CACHE_DIR / CRRT_RESULTS_DIR), so the SAME search runs against
full-resolution or any downsampled variant and lands in that variant's results dir.

Usage
-----
    python search_all.py --gpus 0,1                       # all rungs, default n-iter
    python search_all.py --gpus 0,1 --n-iter 20           # cheaper
    python search_all.py --gpus 0,1 --only transformer fusion
"""

import os
import sys
import queue
import argparse
import threading
import subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from config import CASES, N_SEARCH_ITER, RESULTS_DIR, DEFAULT_STREAM
import train as T

# (model, branch, seq_encoder, weight). Same 8 rungs as run_all; heaviest first so both
# GPUs finish around the same time. seq_encoder is set only for fusion.
RUNGS = [
    ("transformer", "machine",  None,    5),
    ("fusion",      "machine",  "rnn",   4),
    ("fusion",      "machine",  "cnn1d", 4),
    ("rnn",         "machine",  None,    3),
    ("cnn1d",       "machine",  None,    2),
    ("mlp",         "combined", None,    1),
    ("mlp",         "machine",  None,    1),
    ("mlp",         "clinical", None,    1),
]


def _search_argv(model, branch, seq_encoder, case, n_iter, stream):
    argv = [sys.executable, "-u", os.path.join(_HERE, "search.py"),
            "--model", model, "--branch", branch, "--case", case,
            "--n-iter", str(n_iter), "--stream", stream]
    if seq_encoder:
        argv += ["--seq-encoder", seq_encoder]
    return argv


def run_parallel(rungs, gpus, cases, n_iter, stream):
    log_dir = os.path.join(RESULTS_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    gpu_q = queue.Queue()
    for g in gpus:
        gpu_q.put(g)
    lock = threading.Lock()
    failures = []

    def worker(model, branch, seq_encoder, case):
        tag = T.make_tag(model, seq_encoder if model == "fusion" else branch)
        gpu = gpu_q.get()
        try:
            log_path = os.path.join(log_dir, f"search_{tag}_{case}.log")
            with lock:
                print(f"[search] GPU {gpu} <- {tag}/{case:<6} (log: {log_path})", flush=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            with open(log_path, "w") as lf:
                rc = subprocess.run(_search_argv(model, branch, seq_encoder, case, n_iter, stream),
                                    env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
            with lock:
                print(f"[search] GPU {gpu} -> {tag}/{case:<6} "
                      f"{'ok' if rc == 0 else f'FAILED rc={rc}'}", flush=True)
                if rc != 0:
                    failures.append((tag, case, rc, log_path))
        finally:
            gpu_q.put(gpu)

    jobs = [(m, b, enc, c) for (m, b, enc, _w) in rungs for c in cases]
    threads = [threading.Thread(target=worker, args=j) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failures:
        print("\n[search] some searches failed:")
        for tag, case, rc, lp in failures:
            print(f"  {tag}/{case}: rc={rc}  see {lp}")
    return not failures


def main():
    ap = argparse.ArgumentParser(description="Per-rung HP search across GPUs")
    ap.add_argument("--gpus", default=None, help="comma-separated GPU ids, e.g. '0,1'")
    ap.add_argument("--n-iter", type=int, default=N_SEARCH_ITER)
    ap.add_argument("--only", nargs="+", default=None, help="restrict to these model names")
    ap.add_argument("--cases", nargs="+", default=CASES)
    ap.add_argument("--stream", default=DEFAULT_STREAM)
    a = ap.parse_args()

    rungs = [r for r in sorted(RUNGS, key=lambda r: -r[3])
             if (a.only is None or r[0] in a.only)]
    if not a.gpus:
        print("[search] --gpus not set; running sequentially on the default device")
    gpus = [g.strip() for g in (a.gpus or "0").split(",") if g.strip() != ""]
    print(f"=== search_all: {len(rungs)} rungs x {len(a.cases)} cases, "
          f"n_iter={a.n_iter}, GPUs {gpus} -> {RESULTS_DIR} ===")
    run_parallel(rungs, gpus, a.cases, a.n_iter, a.stream)
    print("\n=== search complete. Train the winners with: run_all.py --use-search ===")


if __name__ == "__main__":
    main()
