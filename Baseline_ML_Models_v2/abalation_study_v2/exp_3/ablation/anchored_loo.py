"""
anchored_loo.py — leave-one-out ablation of the BEST ensemble, ANCHORED to the headline
=========================================================================================
Unlike ensemble_ablation.py (light recipe, dose-downtime baseline), this anchors the LOO to the
headline machine-only ensemble:
  * all3  = the headline current_denoised result (flat downtime + peak smoothing) -> reuse the
            already-computed exp_3 ensemble (1day 0.649 / 2day 0.643).
  * no_peak / no_priming / no_downtime = run the FULL headline pipeline (build_rich_features ->
            classical_rich tuned, base+rich, 5 models x 5 seeds -> attention-CNN1D 5 seeds ->
            val-ranked ensemble:mean) on the corresponding FLAT-downtime stream.
So all3 reproduces the headline and the leave-one-out deltas are interpretable against it.

Per config the exp_3 modules are pointed at the config's data via runtime overrides of their
data-binding names, with isolated per-config data dirs (no cache cross-talk). Output:
results/anchored_loo.csv.
"""
import os
import sys
import shutil
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP3 = os.path.dirname(_HERE)
_AS_DIR = os.path.dirname(_EXP3)
# insert so that exp_3/dl and exp_3/features take priority over the parent models/ dir
# (both have a dl_data.py / similar names — exp_3's must win). Last inserted = highest priority.
for p in [_AS_DIR, os.path.join(_AS_DIR, "models"),
          os.path.join(_EXP3, "features"), os.path.join(_EXP3, "dl")]:
    sys.path.insert(0, p)
from abl_config import DENOISE_VARIANTS                                       # noqa: E402
from sklearn.metrics import roc_auc_score                                     # noqa: E402
from sklearn.linear_model import LogisticRegression                          # noqa: E402

# Import the DL modules FIRST (while exp_3/dl is path-front) so dl/dl_data.py is cached before
# classical_rich re-inserts models/ at the front (which else shadows dl_data with the parent's).
import day_signal as DS                                                       # noqa: E402
import dl_data as DD                                                          # noqa: E402
import train_eval as TE                                                       # noqa: E402
import build_rich_features as BRF                                             # noqa: E402
import classical_rich as CR                                                   # noqa: E402

ABL_DATA = os.path.join(_AS_DIR, "exp_2", "ablation", "data", "abalation")    # base-133 tables
WORK = os.path.join(_HERE, "anchored_data"); os.makedirs(WORK, exist_ok=True)
RESULTS = os.path.join(_HERE, "results"); os.makedirs(RESULTS, exist_ok=True)
CASES = ("1day", "2day")
CLASSICAL_MODELS = ["GradientBoosting", "RF", "LightGBM", "XGBoost", "LogisticRegression"]
TOP_K = 3

# config -> (pressure path, base-133 table dir tag under ABL_DATA)
def _press_loo(name):
    return os.path.join(_HERE, "data", f"pressure_flat_{name}.csv")
CONFIGS = {
    "all3":        ("REUSE_HEADLINE", None),
    "no_peak":     (DENOISE_VARIANTS["current"]["noised"], "current_noised"),
    "no_priming":  (_press_loo("no_priming"),  "flat_no_priming_x"),
    "no_downtime": (_press_loo("no_downtime"), "flat_no_downtime_x"),
}


def _ensemble_mean(class_npz, dl_npz):
    """Reproduce the headline ensemble:mean (val-ranked top-3 classical families + DL)."""
    cz = np.load(class_npz); dz = np.load(dl_npz)
    yva, yte, pid_te = cz["yva"].astype(int), cz["yte"].astype(int), cz["pid_te"]
    cands = {}
    for fs in ("base", "rich"):
        for m in CLASSICAL_MODELS:
            if f"val_{fs}_{m}" in cz.files:
                cands[f"cls:{fs}:{m}"] = (cz[f"val_{fs}_{m}"], cz[f"test_{fs}_{m}"])
    cands["dl:attn_cnn1d"] = (dz["val"], dz["test"])
    vauc = {n: roc_auc_score(yva, v) for n, (v, _) in cands.items()}
    keep, seen = [], set()
    for n in sorted(vauc, key=vauc.get, reverse=True):
        fam = n.split(":")[-1]
        if n.startswith("dl:") or (fam not in seen and sum(k.startswith("cls:") for k in keep) < TOP_K):
            keep.append(n); seen.add(fam)
    if "dl:attn_cnn1d" not in keep:
        keep.append("dl:attn_cnn1d")
    Vt = np.column_stack([cands[n][1] for n in keep])
    ens = Vt.mean(1)
    pats = np.unique(pid_te); rng = np.random.default_rng(0)
    by = {p: np.where(pid_te == p)[0] for p in pats}; aucs = []
    for _ in range(1000):
        idx = np.concatenate([by[p] for p in rng.choice(pats, len(pats), replace=True)])
        if len(np.unique(yte[idx])) > 1:
            aucs.append(roc_auc_score(yte[idx], ens[idx]))
    lo, hi = round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)
    return round(roc_auc_score(yte, ens), 4), f"[{lo}, {hi}]", keep


def _run_pipeline(name, pressure, base_tag, case):
    """Run the full headline pipeline for one (config, case) in an isolated data dir; write probs."""
    cfgdir = os.path.join(WORK, name); rich = os.path.join(cfgdir, "rich", case)
    dld = os.path.join(cfgdir, "dldata"); res = os.path.join(cfgdir, "results")
    for d in (rich, dld, res):
        os.makedirs(d, exist_ok=True)

    def base_tbl(split, c):
        return os.path.join(ABL_DATA, base_tag, "machine_only", c, f"{split}.csv")

    def rich_tbl(split, c):
        return os.path.join(cfgdir, "rich", c, f"{split}.csv")

    ds, brf, cr, dd, te = DS, BRF, CR, DD, TE
    ds.RAW_PRESSURE = pressure; ds.machine_table = base_tbl
    brf.machine_table = base_tbl; brf.rich_table = rich_tbl
    cr.machine_table = base_tbl; cr.rich_table = rich_tbl; cr.RESULTS_DIR = res
    dd.DATA_DIR = dld
    te.RESULTS_DIR = res

    for split in ("train", "val", "test"):
        brf.build_split(split, case)
    cr.evaluate(case)                                  # -> res/classical_probs_<case>.npz
    te.run(case)                                       # -> res/dl_probs_<case>.npz
    return (os.path.join(res, f"classical_probs_{case}.npz"),
            os.path.join(res, f"dl_probs_{case}.npz"))


def run():
    rows = []
    for name, (pressure, base_tag) in CONFIGS.items():
        for case in CASES:
            if pressure == "REUSE_HEADLINE":
                cnpz = os.path.join(_EXP3, "features", "results", f"classical_probs_{case}.npz")
                dnpz = os.path.join(_EXP3, "dl", "results", f"dl_probs_{case}.npz")
            else:
                base0 = os.path.join(ABL_DATA, base_tag, "machine_only", case, "test.csv")
                if not (os.path.exists(pressure) and os.path.exists(base0)):
                    print(f"[skip] {name}/{case}: missing pressure or base table", flush=True); continue
                cnpz, dnpz = _run_pipeline(name, pressure, base_tag, case)
            auroc, ci, keep = _ensemble_mean(cnpz, dnpz)
            rows.append({"config": name, "case": case, "ens_AUROC": auroc, "ens_CI95": ci,
                         "n_base_models": len(keep)})
            pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "anchored_loo.csv"), index=False)
            print(f"[ok] {name}/{case}: ENSEMBLE {auroc} {ci}  (members {keep})", flush=True)
    print("\n================ anchored leave-one-out (full headline recipe) ================")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    run()
