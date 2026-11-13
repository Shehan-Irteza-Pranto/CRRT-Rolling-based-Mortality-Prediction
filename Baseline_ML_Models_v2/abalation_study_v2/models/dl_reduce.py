"""
models/dl_reduce.py
===================
Reduce the long current-day stream [N, T, C] before the sequence encoder.

Two methods (both -> [N, T', F], plus a window/patch mask [N, T']):
  * window_aggregate : non-overlapping windows of size k, each -> [mean,min,max,std,slope]
                       per channel. F = C*5.  (cnn1d / rnn / bilstm)
  * patchify         : non-overlapping patches of length P, flattened. F = C*P.
                       The encoder's input Linear does the patch embedding. (transformer)

All statistics are computed over the VALID (unmasked) timesteps within each
window; empty windows -> zero features + mask False.
"""

import numpy as np

WINDOW_STATS = ["mean", "min", "max", "std", "slope"]


def window_aggregate(X, mask, k=10):
    """[N,T,C], [N,T] -> Xr [N, T//k, C*5], maskr [N, T//k]."""
    X = np.asarray(X, dtype=np.float64)
    M = np.asarray(mask, dtype=bool)
    N, T, C = X.shape
    W = T // k
    T2 = W * k
    Xw = X[:, :T2].reshape(N, W, k, C)
    Mw = M[:, :T2].reshape(N, W, k)
    m = Mw.astype(np.float64)[..., None]                 # [N,W,k,1]
    n = m.sum(axis=2)                                     # [N,W,1]
    safe = np.clip(n, 1.0, None)

    mean = (Xw * m).sum(axis=2) / safe                   # [N,W,C]
    big = np.where(Mw[..., None], Xw, np.inf)
    sml = np.where(Mw[..., None], Xw, -np.inf)
    vmin = big.min(axis=2); vmax = sml.max(axis=2)
    var = (m * (Xw - mean[:, :, None, :]) ** 2).sum(axis=2) / safe
    std = np.sqrt(np.clip(var, 0, None))

    t = np.arange(k, dtype=np.float64)                   # in-window time index
    tm = (m[..., 0] * t[None, None, :]).sum(axis=2) / safe[..., 0]      # [N,W]
    tdev = t[None, None, :] - tm[:, :, None]                            # [N,W,k]
    num = (m[..., 0][..., None] * tdev[..., None] * (Xw - mean[:, :, None, :])).sum(axis=2)  # [N,W,C]
    den = (m[..., 0] * tdev ** 2).sum(axis=2)                           # [N,W]
    slope = num / np.clip(den, 1e-9, None)[:, :, None]

    maskr = (n[..., 0] > 0)                               # [N,W]
    empty = ~maskr
    for arr in (mean, vmin, vmax, std, slope):
        arr[empty] = 0.0
    out = np.stack([mean, vmin, vmax, std, slope], axis=-1)   # [N,W,C,5]
    Xr = out.reshape(N, W, C * len(WINDOW_STATS)).astype(np.float32)
    return Xr, maskr


def patchify(X, mask, P=60):
    """[N,T,C], [N,T] -> Xr [N, T//P, C*P], maskr [N, T//P]."""
    X = np.asarray(X, dtype=np.float32)
    M = np.asarray(mask, dtype=bool)
    N, T, C = X.shape
    W = T // P
    T2 = W * P
    Xr = X[:, :T2].reshape(N, W, P * C)                   # time-major within patch, then channel
    maskr = M[:, :T2].reshape(N, W, P).any(axis=2)
    return Xr, maskr


def reduce_stream(X, mask, spec):
    """spec = (method, params). Returns (Xr float32, maskr bool, feat_dim)."""
    if spec is None:
        Xr = np.asarray(X, dtype=np.float32)
        return Xr, np.asarray(mask, dtype=bool), Xr.shape[-1]
    method, params = spec
    if method == "window_agg":
        Xr, mr = window_aggregate(X, mask, k=params.get("k", 10))
    elif method == "patch":
        Xr, mr = patchify(X, mask, P=params.get("P", 60))
    else:
        raise ValueError(f"unknown reduce method: {method}")
    return Xr, mr, Xr.shape[-1]


# --- smoke test ---
if __name__ == "__main__":
    N, T, C = 8, 1440, 4
    # channel c = global ramp t (so in-window slope == 1, min/max = window ends)
    X = np.tile(np.arange(T)[None, :, None], (N, 1, C)).astype(float)
    mask = np.ones((N, T), bool)
    mask[:, 1400:] = False                                 # pad the tail

    Xr, mr, F = reduce_stream(X, mask, ("window_agg", {"k": 10}))
    print("window_agg:", Xr.shape, "feat_dim", F, "mask", mr.shape)
    assert Xr.shape == (N, 144, 20) and F == 20
    # window 0: values 0..9 -> mean 4.5, min 0, max 9, slope 1
    w0 = Xr[0, 0].reshape(C, 5)[0]   # channel 0: [mean,min,max,std,slope]
    assert abs(w0[0] - 4.5) < 1e-5 and w0[1] == 0 and w0[2] == 9 and abs(w0[4] - 1.0) < 1e-6, w0
    # last windows fully padded (1400-1439 -> windows 140..143) -> mask False
    assert (~mr[0, 140:]).all(), mr[0, 138:]

    Xp, mp, Fp = reduce_stream(X, mask, ("patch", {"P": 60}))
    print("patch:", Xp.shape, "feat_dim", Fp)
    assert Xp.shape == (N, 24, 240) and Fp == 240

    Xraw, mraw, Fraw = reduce_stream(X, mask, None)
    assert Xraw.shape == (N, 1440, 4) and Fraw == 4
    print("OK: window-agg (slope/min/max correct, padded->mask False), patch, and raw passthrough.")
