"""
run_preproc_ablation.py — rows 1-4: the tuned-transformer combined stack under incremental preprocessing.
Redirects the exp_4 combined model's data bindings (day_signal.RAW_PRESSURE, build_rich_features tables,
dl_data cache, CF._machine_keys, CF._combined_table) at each cleaned-signal variant, builds the enriched
table when needed, then runs the tuned patch-Transformer fusion + top-3 classical logistic stack, best-val
seed. Metrics via MR._rows_for_variant. Writes exp_5_ablation/results/rows1_4.csv. GPU.
"""
import os
import sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_AS = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_AS, "exp_4"))
import stack_with_tuned_transformer as ST                    # noqa: E402 (imports ABL, HPT, CF)
ABL = ST.ABL; CF = ST.CF; MR = CF.MR
import build_rich_features as BRF                            # noqa: E402 (exp_3/features on path via CF)
import day_signal as DS                                      # noqa: E402
from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.metrics import roc_auc_score                    # noqa: E402
import torch                                                 # noqa: E402

CASE = "1day"
ABL_DATA = os.path.join(_AS, "exp_2", "ablation", "data", "abalation")
WORK = os.path.join(_HERE, "work"); os.makedirs(WORK, exist_ok=True)
OUT = os.path.join(_HERE, "results"); os.makedirs(OUT, exist_ok=True)
KEYS = CF.KEY_COLS
META = CF.META
# rows 1-4: (name, cleaning tag, enriched?)
CONFIGS = [("row1_raw_base", "raw", False), ("row2_raw_rich", "raw", True),
           ("row3_prim_rich", "prim", True), ("row4_primdt_rich", "primdt", True)]


def _base_tbl(tag):
    return lambda split, c: os.path.join(ABL_DATA, f"abl5_{tag}_x", "machine_only", c, f"{split}.csv")


def _bind(name, tag, enriched):
    cfgdir = os.path.join(WORK, name)
    rich = os.path.join(cfgdir, "rich"); dld = os.path.join(cfgdir, "dldata")
    for d in (os.path.join(rich, CASE), dld):
        os.makedirs(d, exist_ok=True)
    base_tbl = _base_tbl(tag)
    rich_tbl = lambda split, c: os.path.join(rich, c, f"{split}.csv")
    press = os.path.join(_HERE, "data", f"pressure_{tag}.csv")
    # --- redirect the raw signal + tabular tables + DL cache ---
    DS.RAW_PRESSURE = press; DS.machine_table = base_tbl
    BRF.machine_table = base_tbl; BRF.rich_table = rich_tbl
    CF.DD.DATA_DIR = dld
    CF._machine_keys = lambda c, split: pd.read_csv(base_tbl(split, c), usecols=KEYS + ["label"])
    mach_tbl = rich_tbl if enriched else base_tbl
    if enriched:
        for split in ("train", "val", "test"):
            if not os.path.exists(rich_tbl(split, CASE)):
                BRF.build_split(split, CASE)

    def combined_table(c, split):                            # machine (base/rich) + clinical
        r = pd.read_csv(mach_tbl(split, c))
        cl = pd.read_csv(os.path.join(CF.CLINICAL_ONLY, c, f"{split}.csv"))
        clc = [x for x in cl.columns if x not in META]
        return r.merge(cl[KEYS + clc], on=KEYS, how="left")
    CF._combined_table = combined_table


def _forced_stack(fv, ft, comb, clin, yva, S):
    stack = []
    for s in range(S):
        c = {"dl:tf": (fv[s], ft[s])}
        for m, d in comb.items():
            c[f"cls:combined:{m}"] = (d[0][s], d[1][s])
        for m, d in clin.items():
            c[f"cls:clinical:{m}"] = (d[0][s], d[1][s])
        vauc = {n: roc_auc_score(yva, c[n][0]) for n in c}
        cls3 = [n for n in sorted(vauc, key=vauc.get, reverse=True) if n.startswith("cls:")][:3]
        names = ["dl:tf"] + cls3
        Lv = np.column_stack([c[n][0] for n in names]); Lt = np.column_stack([c[n][1] for n in names])
        mt = LogisticRegression(max_iter=5000).fit(Lv, yva)
        stack.append((mt.predict_proba(Lv)[:, 1], mt.predict_proba(Lt)[:, 1]))
    return stack


def run_config(name, tag, enriched):
    print(f"==== {name} (clean={tag}, enriched={enriched}) ====", flush=True)
    _bind(name, tag, enriched)
    data, clin_t = ABL._fusion_data(CASE)                    # sequence + clinical tensors from the variant
    tf_vp, tf_tp = ST._train_tf(data, clin_t)                # tuned patch-Transformer fusion, 5 seeds
    yva = data["val"][4].astype(int); yte = data["test"][4].astype(int); S = tf_vp.shape[0]
    comb = CF._classical_probs([CF._combined_table(CASE, s) for s in ("train", "val", "test")], "combined", per_seed=True)
    clin = CF._classical_probs([pd.read_csv(os.path.join(CF.CLINICAL_ONLY, CASE, f"{s}.csv"))
                                for s in ("train", "val", "test")], "clinical_only", per_seed=True)
    stack = _forced_stack(tf_vp, tf_tp, comb, clin, yva, S)
    s_s = int(np.argmax([roc_auc_score(yva, v) for v, _ in stack]))
    by = {r["protocol"]: r for r in MR._rows_for_variant(name, CASE, stack[s_s][0], stack[s_s][1], yva, yte)}
    g = lambda k: round(by["full"][k], 3)
    row = {"config": name, "clean": tag, "enriched": enriched, "sel_seed": s_s,
           "AUROC": g("AUROC"), "AUPRC": g("AUPRC"), "Precision": g("Precision"), "Recall": g("Recall"), "F1": g("F1")}
    print(f"  -> AUROC {row['AUROC']}  AUPRC {row['AUPRC']}  P {row['Precision']} R {row['Recall']} F1 {row['F1']}", flush=True)
    return row


def main():
    rows = []
    for name, tag, enriched in CONFIGS:
        rows.append(run_config(name, tag, enriched))
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "rows1_4.csv"), index=False)
    print(f"\nsaved -> {os.path.join(OUT, 'rows1_4.csv')}")


if __name__ == "__main__":
    main()
