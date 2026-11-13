"""
models/dl_search.py
===================
Documented hyperparameter search space for the aggregated-DL models, plus a
seeded sampler. Mirrors the spirit of the classical RandomizedSearch (a fixed,
documented budget) so the DL models are tuned on the same footing.

`sample_hp(encoder, n_iter, seed)` returns a deterministic list of HP dicts:
    {"lr", "weight_decay", "dropout", "batch_size", "fusion_dim", "seq_kwargs": {...}}
"""

import numpy as np

# --- shared optimization / regularization knobs ---
COMMON_SPACE = {
    "lr":           [3e-4, 1e-3, 1e-4],
    "weight_decay": [1e-3, 1e-4],
    "dropout":      [0.2, 0.3, 0.5],
    "batch_size":   [32, 64],
    "fusion_dim":   [64, 128],
}

# --- per-encoder architecture knobs (-> seq_kwargs) ---
ENCODER_SPACE = {
    "cnn1d":       {"channels": [32, 64, 128], "n_blocks": [2, 3], "kernel": [3, 5, 7]},
    "rnn":         {"hidden": [64, 128], "layers": [1, 2], "cell": ["lstm", "gru"]},
    "transformer": {"d_model": [64, 128], "n_heads": [4, 8], "n_layers": [2, 4], "ff": [128, 256]},
}


def _sample_from(space, rng):
    return {k: (v[rng.randint(len(v))]) for k, v in space.items()}


def sample_hp(encoder, n_iter=12, seed=0):
    """Deterministic list of n_iter HP dicts for `encoder` (RandomState(seed))."""
    rng = np.random.RandomState(seed)
    enc_space = ENCODER_SPACE.get(encoder, {})
    out = []
    for _ in range(max(1, n_iter)):
        hp = _sample_from(COMMON_SPACE, rng)
        # dropout drives both the head and the encoder; pass encoder knobs via seq_kwargs.
        seq_kwargs = _sample_from(enc_space, rng)
        seq_kwargs["dropout"] = hp["dropout"]
        # transformer needs d_model divisible by n_heads
        if encoder == "transformer" and seq_kwargs["d_model"] % seq_kwargs["n_heads"] != 0:
            seq_kwargs["n_heads"] = 4 if seq_kwargs["d_model"] % 4 == 0 else 2
        hp["seq_kwargs"] = seq_kwargs
        out.append(hp)
    return out


if __name__ == "__main__":
    for enc in ENCODER_SPACE:
        hps = sample_hp(enc, n_iter=3, seed=0)
        hps2 = sample_hp(enc, n_iter=3, seed=0)
        assert hps == hps2, "sampler must be deterministic for a fixed seed"
        print(f"\n{enc}: {len(hps)} sampled HP sets")
        for h in hps:
            print("  ", h)
    print("\nOK: deterministic per (encoder, seed).")
