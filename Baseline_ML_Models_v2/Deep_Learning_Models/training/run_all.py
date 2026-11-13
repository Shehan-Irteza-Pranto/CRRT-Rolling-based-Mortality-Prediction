"""
training/run_all.py
===================
Train every deep rung from scratch over all config SEEDS, then aggregate and plot.
This is the single "start from the beginning" entry point: it (re)writes
results/raw_results_*.csv, saves the best checkpoint per seed under
results/best_models/, logs per-epoch train/val loss + AUROC under
results/training_history/, then builds the comparison tables and the
loss / AUROC / precision-recall curves.

Rungs (8): mlp{clinical,machine,combined}, cnn1d, rnn, transformer,
           fusion{cnn1d,rnn}.

Multi-GPU
---------
Each individual rung is small (few-hundred-sample dataset, small models), so
splitting ONE model across GPUs (DataParallel/DDP) would only add overhead. The
right parallelism is across *rungs*: with --gpus, each rung trains in its own
process pinned to one GPU (CUDA_VISIBLE_DEVICES), scheduled dynamically so both
GPUs stay busy. Every rung writes a unique result tag, so there are no file
collisions. Per-rung logs land in results/logs/.

Usage
-----
    python run_all.py                          # all rungs, sequential, 1 GPU (in-process)
    python run_all.py --gpus 0,1               # spread rungs across GPU 0 and 1
    python run_all.py --gpus 0,1 --cases 1day  # restrict cases
    python run_all.py --gpus 0,1 --only cnn1d fusion transformer
"""

import os
import sys
import json
import queue
import argparse
import threading
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CASES, SEEDS, DEFAULT_STREAM, RESULTS_DIR

import train as T
import evaluate as E
import plots as P

# (model, branch, hp, weight) per rung. branch is the MLP feature source; hp carries
# the fusion sequence-encoder choice. `weight` is a rough cost used to start the
# heaviest rungs first so both GPUs finish around the same time.
RUNGS = [
    ("transformer", "machine",  {"seq_encoder": None},      5),
    ("fusion",      "machine",  {"seq_encoder": "rnn"},     4),
    ("fusion",      "machine",  {"seq_encoder": "cnn1d"},   4),
    ("rnn",         "machine",  {},                          3),
    ("cnn1d",       "machine",  {},                          2),
    ("mlp",         "combined", {},                          1),
    ("mlp",         "machine",  {},                          1),
    ("mlp",         "clinical", {},                          1),
]


def _clean_hp(hp):
    """Drop placeholder keys (e.g. seq_encoder=None for non-fusion)."""
    return {k: v for k, v in hp.items() if v is not None}


def _rung_label(model, branch, hp):
    return f"{model}/{hp['seq_encoder']}" if model == "fusion" else f"{model}/{branch}"


def _load_best_hp(tag, case):
    """Best hp from results/search_<tag>_<case>.json, or None if absent/empty."""
    path = os.path.join(RESULTS_DIR, f"search_{tag}_{case}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        best = json.load(f).get("best")
    return best.get("hp") if best else None


def _resolve_hp(model, branch, base_hp, cases, use_search):
    """When --use-search, replace the default hp with the rung's searched winner.
    Search is per-case; training uses one hp across cases, so we take the first case's
    winner (1-day runs make this exact). Falls back to defaults if no search json."""
    if not use_search:
        return base_hp
    tag = T.make_tag(model, base_hp.get("seq_encoder", branch))
    best = _load_best_hp(tag, cases[0])
    if best is None:
        print(f"[run_all] --use-search: no search json for {tag}/{cases[0]}; using defaults")
        return base_hp
    merged = {**base_hp, **best}                 # keep seq_encoder; searched keys win
    print(f"[run_all] {tag}: searched hp -> {merged}")
    if len(cases) > 1:
        print(f"[run_all] note: applying {tag}'s {cases[0]}-searched hp to all cases {cases}")
    return merged


# ---------------------------------------------------------------- multi-GPU dispatch

def _train_argv(model, branch, hp, cases, seeds, stream):
    py = sys.executable
    train_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
    argv = [py, "-u", train_py, "--model", model, "--branch", branch,
            "--cases", *cases, "--stream", stream]
    if seeds:                                   # pass the exact seed ids, not a count
        argv += ["--seed-list", *[str(s) for s in seeds]]
    if hp:
        argv += ["--hp", json.dumps(hp)]
    return argv


def run_parallel(rungs, gpus, cases, seeds, stream):
    """Dispatch one subprocess per rung across `gpus`, max len(gpus) concurrent.
    A rung blocks until a GPU frees up (the gpu queue caps concurrency)."""
    log_dir = os.path.join(RESULTS_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    gpu_q = queue.Queue()
    for g in gpus:
        gpu_q.put(g)

    lock = threading.Lock()
    failures = []

    def worker(model, branch, hp):
        tag = T.make_tag(model, hp.get("seq_encoder", branch))
        gpu = gpu_q.get()                       # blocks until a GPU is free
        try:
            log_path = os.path.join(log_dir, f"train_{tag}.log")
            with lock:
                print(f"[dispatch] GPU {gpu} <- {tag:<16} (log: {log_path})", flush=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            with open(log_path, "w") as lf:
                rc = subprocess.run(
                    _train_argv(model, branch, hp, cases, seeds, stream),
                    env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
            with lock:
                status = "ok" if rc == 0 else f"FAILED rc={rc}"
                print(f"[dispatch] GPU {gpu} -> {tag:<16} {status}", flush=True)
                if rc != 0:
                    failures.append((tag, rc, log_path))
        finally:
            gpu_q.put(gpu)                       # release GPU for the next rung

    threads = [threading.Thread(target=worker, args=(m, b, hp)) for (m, b, hp) in rungs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if failures:
        print("\n[dispatch] some rungs failed:")
        for tag, rc, lp in failures:
            print(f"  {tag}: rc={rc}  see {lp}")
    return not failures


# ---------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser(description="Train all DL rungs from scratch, aggregate, plot")
    ap.add_argument("--cases", nargs="+", default=CASES)
    ap.add_argument("--seeds", type=int, default=None, help="use the first N config SEEDS")
    ap.add_argument("--only", nargs="+", default=None, help="restrict to these model names")
    ap.add_argument("--stream", default=DEFAULT_STREAM)
    ap.add_argument("--gpus", default=None,
                    help="comma-separated GPU ids, e.g. '0,1'. Omit to run in-process on the default device.")
    ap.add_argument("--use-search", action="store_true",
                    help="train each rung with its searched winner from results/search_<tag>_<case>.json "
                         "(run search_all.py first); falls back to defaults if absent")
    a = ap.parse_args()

    seeds = SEEDS[: a.seeds] if a.seeds else SEEDS
    rungs = [(m, b, _resolve_hp(m, b, _clean_hp(hp), a.cases, a.use_search))
             for (m, b, hp, _w) in sorted(RUNGS, key=lambda r: -r[3])
             if (a.only is None or m in a.only)]
    n_seeds = len(seeds)

    if a.gpus:
        gpus = [g.strip() for g in a.gpus.split(",") if g.strip() != ""]
        print(f"=== run_all (parallel): {len(rungs)} rungs x {len(a.cases)} cases x {n_seeds} seeds "
              f"across GPUs {gpus} ===")
        ok = run_parallel(rungs, gpus, a.cases, seeds, a.stream)
        if not ok:
            print("\n[run_all] aborting aggregate/plots — fix the failed rungs and re-run.")
            sys.exit(1)
    else:
        print(f"=== run_all (in-process): {len(rungs)} rungs x {len(a.cases)} cases x {n_seeds} seeds ===")
        for i, (model, branch, hp) in enumerate(rungs, 1):
            print(f"\n########## [{i}/{len(rungs)}] {_rung_label(model, branch, hp)} ##########")
            T.run(model, a.cases, branch, seeds, hp, a.stream)

    print("\n=== aggregating results -> comparison tables ===")
    E.aggregate()
    print("\n=== generating curves (loss / AUROC / precision-recall) ===")
    P.generate()
    print("\n=== run_all complete ===")


if __name__ == "__main__":
    main()
