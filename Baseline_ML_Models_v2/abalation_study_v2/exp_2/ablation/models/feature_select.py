"""
models/feature_select.py
========================
mRMR feature selection (minimum Redundancy Maximum Relevance), matching the
baseline pipelines. Fit on TRAIN only; caller reuses the chosen columns for
val/test (no leakage).

  clinical_only : top-N clinical
  machine_only  : top-N machine
  combined      : grouped — top-N machine + top-N clinical (union)

Machine vs clinical columns are split by the pressure-channel name prefix
(abl_config.is_machine_col).
"""

import os
import sys
import json
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
from abl_config import (N_MRMR, N_MRMR_MACHINE, N_MRMR_CLINICAL, is_machine_col,  # noqa: E402
                        N_MRMR_MACHINE_MAX, RESULTS_DIR)

SELECTION_DIR = os.path.join(RESULTS_DIR, "feature_selection")


def mrmr_select(X, y, feat_names, k):
    """Return the mRMR top-k feature names from columns `feat_names` of array X."""
    from mrmr import mrmr_classif
    k = min(k, X.shape[1])
    if k <= 0 or X.shape[1] == 0:
        return list(feat_names)
    Xdf = pd.DataFrame(X, columns=feat_names)
    chosen = mrmr_classif(X=Xdf, y=pd.Series(np.asarray(y)), K=k, show_progress=False)
    return list(chosen)


def select_columns(feat_names, X_tr, y_tr, feature_set,
                   n_machine=N_MRMR_MACHINE, n_clinical=N_MRMR_CLINICAL, n_single=N_MRMR):
    """Pick feature columns by mRMR for a feature_set. Returns (indices, names).

    Indices are into the original `feat_names` order (sorted, stable).
    """
    name_to_idx = {f: i for i, f in enumerate(feat_names)}

    if feature_set == "combined":
        mach = [f for f in feat_names if is_machine_col(f)]
        clin = [f for f in feat_names if not is_machine_col(f)]
        m_idx = [name_to_idx[f] for f in mach]
        c_idx = [name_to_idx[f] for f in clin]
        chosen_m = mrmr_select(X_tr[:, m_idx], y_tr, mach, n_machine) if mach else []
        chosen_c = mrmr_select(X_tr[:, c_idx], y_tr, clin, n_clinical) if clin else []
        chosen = chosen_m + chosen_c
    elif feature_set == "machine_only":
        chosen = mrmr_select(X_tr, y_tr, list(feat_names), n_single)
    elif feature_set == "clinical_only":
        chosen = mrmr_select(X_tr, y_tr, list(feat_names), n_clinical)
    else:
        chosen = list(feat_names)

    sel_idx = sorted(name_to_idx[f] for f in chosen)
    sel_names = [feat_names[i] for i in sel_idx]
    return sel_idx, sel_names


def get_or_compute_combined_ranking(key, feat_names, X_tr, y_tr,
                                    n_clinical=N_MRMR_CLINICAL, max_machine=N_MRMR_MACHINE_MAX):
    """For the DYNAMIC combined: cache the fixed clinical top-N + the FULL mRMR machine
    ranking (up to max_machine). The per-model k-search then sweeps a prefix of the
    machine ranking. Cached ONCE per cell (shared across all 8 models).

    Returns (clinical_top_names, machine_ranked_names).
    """
    os.makedirs(SELECTION_DIR, exist_ok=True)
    path = os.path.join(SELECTION_DIR, key + "_ranking.json")
    if os.path.exists(path):
        with open(path) as fh:
            d = json.load(fh)
        return d["clinical_top"], d["machine_ranked"]

    name_to_idx = {f: i for i, f in enumerate(feat_names)}
    mach = [f for f in feat_names if is_machine_col(f)]
    clin = [f for f in feat_names if not is_machine_col(f)]
    m_idx = [name_to_idx[f] for f in mach]; c_idx = [name_to_idx[f] for f in clin]

    # clinical_top = EXACTLY the clinical_only selection ("all the best clinical features"),
    # so combined with machine_k=0 reproduces the clinical_only model. Reuse its cache if the
    # clinical_only config already ran (it does — clinical_only precedes combined); else compute.
    clin_key = key.replace("_combined_", "_clinical_only_")
    clin_path = os.path.join(SELECTION_DIR, clin_key + ".json")
    if clin and os.path.exists(clin_path):
        with open(clin_path) as fh:
            clin_sel = json.load(fh).get("selected", [])
        clin_set = set(clin)
        clinical_top = [f for f in clin_sel if f in clin_set]
    else:
        clinical_top = mrmr_select(X_tr[:, c_idx], y_tr, clin, n_clinical) if clin else []
    machine_ranked = mrmr_select(X_tr[:, m_idx], y_tr, mach,
                                 min(len(mach), max_machine)) if mach else []
    with open(path, "w") as fh:
        json.dump({"key": key, "n_clinical": len(clinical_top),
                   "n_machine_ranked": len(machine_ranked),
                   "clinical_top": clinical_top, "machine_ranked": machine_ranked}, fh, indent=2)
    return clinical_top, machine_ranked


def get_or_compute_selection(key, feat_names, X_tr, y_tr, feature_set):
    """Compute the mRMR selection for a cell ONCE and cache it (shared across models).

    `key` identifies the cell (e.g. "current_denoised_combined_agg_1day"). The first
    caller computes + saves results/feature_selection/<key>.json; later callers load
    the same list -> every model in the cell uses an IDENTICAL feature set.
    Returns (indices into feat_names, names).
    """
    os.makedirs(SELECTION_DIR, exist_ok=True)
    path = os.path.join(SELECTION_DIR, key + ".json")
    name_to_idx = {f: i for i, f in enumerate(feat_names)}
    if os.path.exists(path):
        with open(path) as fh:
            names = json.load(fh)["selected"]
        names = [f for f in names if f in name_to_idx]          # keep order via re-sort below
        sel_idx = sorted(name_to_idx[f] for f in names)
        return sel_idx, [feat_names[i] for i in sel_idx]

    sel_idx, sel_names = select_columns(feat_names, X_tr, y_tr, feature_set)
    with open(path, "w") as fh:
        json.dump({"key": key, "feature_set": feature_set,
                   "n_machine": N_MRMR_MACHINE, "n_clinical": N_MRMR_CLINICAL, "n_single": N_MRMR,
                   "n_selected": len(sel_names), "selected": sel_names}, fh, indent=2)
    return sel_idx, sel_names
