"""roc_pathB_probs.py — test probabilities for the 4 Path B 'ours' models (1-day) from their bundles."""
import os
import sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import predict                                        # noqa: E402 (exp_3 context: CF + TE)

OUT = os.path.join(_HERE, "results", "probs"); os.makedirs(OUT, exist_ok=True)
FILES = {"attn": "machine_attention_cnn1d_1day.joblib", "ensemble": "machine_ensemble_1day.joblib",
         "fusion": "combined_attn_cnn_fusion_1day.joblib", "stack": "combined_ensemble_stack_1day.joblib"}


def main():
    ctx, yte = predict._test_ctx("1day")
    out = {"yte": yte}
    for key, fn in FILES.items():
        p, thr = predict.score(os.path.join(predict.MODELS_DIR, fn), ctx)
        out[key] = p
        from sklearn.metrics import roc_auc_score
        print(f"  {key:9s} AUROC {roc_auc_score(yte, p):.4f}", flush=True)
    np.savez(os.path.join(OUT, "roc_pathB_1day.npz"), **out)
    print(f"saved -> {os.path.join(OUT, 'roc_pathB_1day.npz')}")


if __name__ == "__main__":
    main()
