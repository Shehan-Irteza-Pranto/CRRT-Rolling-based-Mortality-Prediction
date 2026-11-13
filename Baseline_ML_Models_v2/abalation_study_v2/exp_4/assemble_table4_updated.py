"""assemble_table4_updated.py — build the updated Table IV (DL benchmark, 1-day) into one CSV.
Feature-using rows updated (Clinical MLP clinical=60, machine Windowed-Transformer Ensemble machine=40,
combined Windowed-Transformer Fusion Stack 40+60); pure-sequence encoders carried over unchanged from
results/table_vi_bestseed.csv. -> results/table4_updated_1day.csv"""
import os
import sys
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "exp_3", "results"))
import make_resampled_variants as MR                       # noqa: E402
R = os.path.join(_HERE, "results")
COLS = ["AUROC", "AUPRC", "F1", "Precision", "Recall", "AUROC_1to1", "AUPRC_1to1", "AUROC_1to2", "AUPRC_1to2"]

vi = pd.read_csv(os.path.join(R, "table_vi_bestseed.csv"))
vi = vi[vi["case"] == "1day"].set_index(["pipeline", "model"])


def frm_vi(pipeline, model):
    r = vi.loc[(pipeline, model)]
    return [r.full_AUROC, r.full_AUPRC, r.full_F1, r.full_Precision, r.full_Recall,
            r.r1to1_AUROC, r.r1to1_AUPRC, r.r1to2_AUROC, r.r1to2_AUPRC]


# --- updated rows ---
cm = pd.read_csv(os.path.join(R, "clinical_mlp_60feat_1day.csv")).iloc[0]
clinical_mlp = [cm.AUROC, cm.AUPRC, cm.F1, cm.Precision, cm.Recall, cm.AUROC_1to1, cm.AUPRC_1to1, cm.AUROC_1to2, cm.AUPRC_1to2]
me = pd.read_csv(os.path.join(R, "machine_tuned_transformer_40feat_1day.csv")).iloc[0]
mach_ens = [me.full_AUROC, me.full_AUPRC, me.full_F1, me.full_Precision, me.full_Recall,
            me.r1to1_AUROC, me.r1to1_AUPRC, me.r1to2_AUROC, me.r1to2_AUPRC]
# combined stack 40x60 from saved probs
te = pd.read_csv(os.path.join(R, "combined_stack_test_probs_40x60_1day.csv"))
va = pd.read_csv(os.path.join(R, "combined_stack_val_probs_40x60_1day.csv"))
by = {r["protocol"]: r for r in MR._rows_for_variant("stack", "1day",
      va["prob"].to_numpy(float), te["prob"].to_numpy(float), va["label"].to_numpy(int), te["label"].to_numpy(int))}
stack = [by["full"]["AUROC"], by["full"]["AUPRC"], by["full"]["F1"], by["full"]["Precision"], by["full"]["Recall"],
         by["1:1"]["AUROC"], by["1:1"]["AUPRC"], by["1:2"]["AUROC"], by["1:2"]["AUPRC"]]

# --- assemble in Table IV order ---
ROWS = [
    ("Clinical-only", "Clinical MLP", clinical_mlp, "updated: clinical mRMR 60"),
    ("Machine-only", "CNN1D", frm_vi("Machine-only", "CNN1D"), "unchanged"),
    ("Machine-only", "RNN (BiLSTM/GRU)", frm_vi("Machine-only", "RNN (BiLSTM/GRU)"), "unchanged"),
    ("Machine-only", "Patch-Transformer", frm_vi("Machine-only", "Transformer"), "unchanged"),
    ("Machine-only", "Attention-CNN1D", frm_vi("Machine-only", "Attention-CNN1D"), "unchanged"),
    ("Machine-only", "Windowed-Transformer Ensemble (ours)", mach_ens, "updated: machine mRMR 40"),
    ("Combined", "CNN1D fusion", frm_vi("Combined", "CNN1D"), "unchanged"),
    ("Combined", "RNN (BiLSTM/GRU) fusion", frm_vi("Combined", "RNN (BiLSTM/GRU)"), "unchanged"),
    ("Combined", "Patch-Transformer fusion", frm_vi("Combined", "Transformer"), "unchanged"),
    ("Combined", "Attention-CNN1D fusion", frm_vi("Combined", "Attn-CNN fusion"), "unchanged"),
    ("Combined", "Windowed-Transformer Fusion Stack (ours)", stack, "updated: 40 machine + 60 clinical"),
]
out = pd.DataFrame([{"pipeline": p, "model": m, **{c: round(float(v), 3) for c, v in zip(COLS, vals)}, "note": note}
                    for p, m, vals, note in ROWS])
out.to_csv(os.path.join(R, "table4_updated_1day.csv"), index=False)
print(out.to_string(index=False))
print(f"\nsaved -> {os.path.join(R, 'table4_updated_1day.csv')}")
