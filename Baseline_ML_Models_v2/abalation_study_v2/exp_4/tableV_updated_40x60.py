"""
tableV_updated_40x60.py — Table V (ablation of the deployed combined model), updated to the 40/60
feature list (classical base learners = 40 machine + 60 clinical).

Bottom block (rows 5-8) = sequence-encoder swap under full preprocessing: CNN1D / RNN (BiGRU) /
Patch-Transformer / Windowed-Transformer (ours), each as a fusion (encoder + clinical MLP branch) forced
into a logistic stack with the top-3 classical (40+60). Best-of-5-seeds by validation AUROC.
Top block (rows 1-4, incremental preprocessing) is produced separately by run_preproc_ablation.py with
the same 40/60 patch (see tableV_rows1_4_40x60.py). Writes results/tableV_rows5_8_40x60.csv.  GPU.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import ablation_ensemble_branch as ABL                    # noqa: E402 (fusion_probs_encoder, _fusion_data)
import stack_with_tuned_transformer as ST                 # noqa: E402 (_train_tf = tuned windowed transformer)
import sweep_mrmr_stack as SW                              # noqa: E402 (_forced_stack_bestval)
CF = ABL.CF
CASE = "1day"
CF.N_MACHINE, CF.N_CLINICAL = 40, 60                       # classical base learners at best features
OUT = os.path.join(_HERE, "results", "tableV_rows5_8_40x60.csv")


def main():
    print("classical base learners (combined 40x60 + clinical-only) ...", flush=True)
    comb = CF._classical_probs([CF._combined_table(CASE, s) for s in ("train", "val", "test")],
                               "combined", per_seed=True)
    clin = CF._classical_probs([pd.read_csv(os.path.join(CF.CLINICAL_ONLY, CASE, f"{s}.csv"))
                                for s in ("train", "val", "test")], "clinical_only", per_seed=True)

    rows = []
    # default-HP encoders (GeneralFusion): cnn1d / rnn / patch-transformer
    for enc, label in [("cnn1d", "CNN1D"), ("rnn", "RNN (BiGRU)"), ("transformer", "Patch-Transformer")]:
        print(f"==== encoder = {label} ====", flush=True)
        fv, ft, yva, yte = ABL.fusion_probs_encoder(CASE, enc)
        b = SW._forced_stack_bestval(fv, ft, comb, clin, yva, yte)
        rows.append({"encoder": label, "AUROC": round(b["test"], 3), "val_AUROC": round(b["val"], 3),
                     "sel_seed": b["seed"]})
        print(f"  {label}: test {b['test']:.4f}  val {b['val']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)

    # tuned Windowed-Transformer (ours)
    print("==== encoder = Windowed-Transformer (ours) ====", flush=True)
    data, clin_t = ABL._fusion_data(CASE)
    yva, yte = data["val"][4].astype(int), data["test"][4].astype(int)
    tf_vp, tf_tp = ST._train_tf(data, clin_t)
    b = SW._forced_stack_bestval(tf_vp, tf_tp, comb, clin, yva, yte)
    rows.append({"encoder": "Windowed-Transformer (ours)", "AUROC": round(b["test"], 3),
                 "val_AUROC": round(b["val"], 3), "sel_seed": b["seed"]})
    print(f"  Windowed-Transformer (ours): test {b['test']:.4f}  val {b['val']:.4f}", flush=True)

    df = pd.DataFrame(rows); df.to_csv(OUT, index=False)
    print("\n=== Table V bottom block (encoder swap, full preprocessing, 40/60) ===")
    print(df.to_string(index=False))
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
