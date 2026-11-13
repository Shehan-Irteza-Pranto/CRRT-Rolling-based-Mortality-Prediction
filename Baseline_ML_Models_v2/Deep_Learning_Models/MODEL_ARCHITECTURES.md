# Sequence Model Architectures — Input → Output

How the three sequence rungs — **1D-CNN**, **RNN (BiLSTM/GRU)**, and
**Transformer** — actually work in this codebase, traced end to end. Every claim
cites the source line so it stays honest against the code.

The three models are deliberately interchangeable: each is an **encoder** that
turns a variable-length, 4-channel pressure window into a fixed-size embedding,
followed by a single linear head producing one logit. They share the exact same
input pipeline, loss, and evaluation protocol — only the encoder differs. That
is what makes the baseline-ladder comparison fair.

---

## 1. The shared input — one sequence = one circuit-day

All three models consume the **raw per-minute CRRT pressure streams**, never the
52 engineered summary features. The signal has **4 channels in a fixed order**:

```
ACCESS_PRESSURE, FILTER_PRESSURE, EFFLUENT_PRESSURE, RETURN_PRESSURE
```

(`config.py:35`). That channel order is the meaning of the last axis everywhere
downstream.

### From raw CSV to tensor (`data/prepare_sequences.py`)
For each circuit-day keyed by `(PID, Visit_no, Day)`:

1. **Normalize** — per-channel mean/std are fit **on the train split only**;
   residual NaNs are imputed to the channel mean *before* standardizing, then the
   window is standardized `(x - mean) / std` (`prepare_sequences.py:116-121`).
   Fitting stats on train alone is what prevents leakage into val/test.
2. **Pad / crop + mask** — each window is forced to a fixed length
   `MAX_LEN = 1440` (minutes in a day; `config.py:66`). Shorter windows are
   right-padded; longer ones are right-cropped (`CROP_SIDE="right"`). A boolean
   **mask** marks real vs. padded timesteps — `True = real`
   (`pad_crop`, `prepare_sequences.py:90-101`).
3. **Save** — `X [N, 1440, 4]`, `mask [N, 1440]`, `y [N]`, and the join `keys`
   are written to a compressed `.npz` per split (`prepare_sequences.py:147-150`).

### From tensor to batch (`data/sequence_dataset.py`)
`SequenceDataset` yields one sample as `(x [T,4] float32, mask [T] bool,
y float32)` (`sequence_dataset.py:28-43`). The DataLoader stacks these into a
batch:

```
x    : [B, T, 4]   float   standardized pressures (T = MAX_LEN = 1440)
mask : [B, T]      bool    True where the timestep is real, False where padded
y    : [B]         float   mortality label (0/1)
```

### The uniform contract
`build_model(name, ...)` dispatches to the right class (`train.py:61-77`), and
**every** model implements the same call signature:

```python
forward(x, mask) -> logit   # shape [B], raw pre-sigmoid score
```

So swapping CNN ↔ RNN ↔ Transformer changes nothing outside the encoder. The
mask is passed into all three so padded timesteps never contaminate pooling,
recurrence, or attention.

---

## 2. 1D-CNN — `models/cnn1d.py`

**Rung 2. Captures *local* waveform morphology** — the short-range dynamics that
hand-crafted stats like skewness / range / IQR only crudely approximate.

### Trace
- **Reshape to channels-first.** Convolutions slide over *time*, so `x [B,T,4]`
  is transposed to `[B,4,T]` (the 4 pressure channels become the conv input
  channels) (`cnn1d.py:33-34`).
- **Stacked temporal conv blocks.** `n_blocks` (default 3) of:
  `Conv1d(kernel=5, padding=kernel//2)` → `BatchNorm1d` → `ReLU` → `Dropout`.
  The first block maps `4 → channels` (default 64), the rest `channels → channels`.
  Padding = `kernel//2` is "same" padding, so the time length **T is preserved**
  through the stack (`cnn1d.py:18-29`). Output: `[B, channels, T]`.
- **Masked global mean pooling.** The mask `[B,1,T]` zeroes padded positions,
  then the encoder sums over time and divides by the count of valid steps —
  a true mean over *real* timesteps only (`cnn1d.py:35-37`). Output: `[B, channels]`.
- **Head.** `Linear(channels → 1)` → squeeze → `logit [B]` (`cnn1d.py:45-48`).

```
x [B,T,4]
  └─ transpose ─────────────► [B,4,T]
  └─ {Conv1d k=5 (same) → BN → ReLU → Dropout} × n_blocks ► [B,channels,T]
  └─ masked mean over time ─► [B,channels]
  └─ Linear(channels→1) ────► logit [B]
```

**Intuition.** A width-5 kernel sees a 5-minute neighborhood; stacking blocks
widens the receptive field. The CNN learns morphological motifs (slopes, dips,
oscillations) directly from the waveform.

**HP search space:** `channels {32,64,128}`, `n_blocks {2,3,4}`,
`kernel {3,5,7}` (`config.py:90`).

---

## 3. RNN (BiLSTM / GRU) — `models/rnn.py`

**Rung 3. Captures *longer* temporal dependencies** across the window via
recurrence.

### Trace
- **Bidirectional recurrence.** An LSTM (or GRU) with `input_size=4`,
  `hidden` (default 128), `bidirectional=True`. It is bidirectional because we
  classify a **completed** circuit-day window — not online forecasting — so the
  model is allowed to read the whole sequence in both directions
  (`rnn.py:16-25`).
- **Packing to skip padding.** `lengths = mask.sum(1)` gives each sample's real
  length; `pack_padded_sequence(..., enforce_sorted=False)` lets the RNN process
  only real timesteps, then `pad_packed_sequence(..., total_length=T)` restores
  the `[B, T, 2·hidden]` shape (`2·hidden` because forward+backward states are
  concatenated) (`rnn.py:27-31`).
- **Masked mean pooling.** Mean of the per-timestep hidden states over the valid
  steps → `[B, 2·hidden]` (`rnn.py:32-35`).
- **Head.** `Linear(2·hidden → 1)` → `logit [B]` (`rnn.py:38-45`).

```
x [B,T,4]
  └─ pack_padded_sequence (lengths from mask)
  └─ BiLSTM / BiGRU ───────► [B,T,2·hidden]   (pad_packed back to T)
  └─ masked mean over time ► [B,2·hidden]
  └─ Linear(2·hidden→1) ───► logit [B]
```

**Intuition.** The recurrent state carries information across the whole window,
so the RNN can model dependencies longer than a single conv kernel — e.g. a
drift that only matters relative to where the window started.

**HP search space:** `hidden {64,128,256}`, `layers {1,2}`,
`cell {lstm,gru}` (`config.py:91`).

---

## 4. Transformer encoder — `models/transformer.py`

**Rung 4. The strongest sequence rung**, and the source of native
attention-based interpretability (`interpretability/attention_rollout.py`).

### Trace
- **Input projection.** `Linear(4 → d_model)` lifts each timestep into the model
  dimension (`transformer.py:40`, `:53`).
- **`[CLS]` token.** A single learnable vector is prepended to the sequence; its
  final state will be the pooled window representation — the same trick BERT uses
  (`transformer.py:41`, `:54-55`).
- **Positional encoding.** Sinusoidal PE is added so the otherwise
  order-agnostic attention knows *when* each timestep occurred
  (`SinusoidalPE`, `transformer.py:23-34`, applied at `:56`).
- **Padding-aware attention.** `src_key_padding_mask` marks padded timesteps as
  "ignore"; the CLS position is always attended (a `False` is prepended for it)
  (`transformer.py:57-60`).
- **Encoder stack.** `nn.TransformerEncoder` with `n_layers` (default 4) layers,
  each = multi-head self-attention (`n_heads`, default 8) + GELU feed-forward
  (`dim_feedforward = ff`) + dropout → `[B, T+1, d]` (`transformer.py:43-61`).
- **Pool + head.** Take the **CLS state** `h[:, 0]` as the window embedding
  `[B, d]`, then `Linear(d → 1)` → `logit [B]` (`transformer.py:63-68`). The
  separate `encode()` method returns that CLS embedding for reuse by the fusion
  rung.

```
x [B,T,4]
  └─ Linear(4→d) ───────────────────────────► [B,T,d]
  └─ prepend [CLS], add sinusoidal PE ───────► [B,T+1,d]
  └─ TransformerEncoder × n_layers ──────────► [B,T+1,d]   (key_padding_mask)
  └─ take CLS state h[:,0] ──────────────────► [B,d]
  └─ Linear(d→1) ────────────────────────────► logit [B]
```

**Intuition.** Self-attention lets any timestep attend directly to any other, so
long-range structure is one hop away (no recurrence to carry it). The attention
weights are inspectable, which feeds the interpretability thread.

**Cost note.** Full self-attention is O(T²) for T up to ~1440; if memory-bound,
the stream can be downsampled in `prepare_sequences.py` or a linear-attention
layer swapped in (documented at `transformer.py:13-15`).

**HP search space:** `d_model {64,128}`, `n_heads {4,8}`, `n_layers {2,4}`,
`ff {128,256}` (`config.py:92`).

---

## 5. Shared head, loss, and output

Once the encoder produces a logit, the rest is identical across all three (and
matches the classical protocol so the comparison is fair):

- **Loss.** Trained on the full imbalanced set with **weighted BCE or focal
  loss**; `pos_weight = n_neg / n_pos` (`sequence_dataset.py:45-48`;
  `training/losses.py`). Training **early-stops on validation AUROC**.
- **Probabilities.** At evaluation the logit passes through a sigmoid to get a
  probability, scoring threshold-independent **AUROC / AUPRC**.
- **Class decision.** The hard 0/1 prediction uses an **F1-optimal threshold
  derived on the balanced validation set**, then applied to the balanced test
  set — the same validation-derived threshold rule the classical models use.

---

## 6. Side-by-side

| | **1D-CNN** | **RNN (BiLSTM/GRU)** | **Transformer** |
|---|---|---|---|
| Rung | 2 | 3 | 4 |
| Input | `[B,T,4]` | `[B,T,4]` | `[B,T,4]` |
| Core op | temporal convolutions | bidirectional recurrence | multi-head self-attention |
| Captures | local waveform morphology | longer temporal dependencies | global, long-range structure |
| Handles padding via | mask in pooling | `pack_padded_sequence` | `src_key_padding_mask` |
| Pooling → embedding | masked mean → `[B,channels]` | masked mean → `[B,2·hidden]` | `[CLS]` state → `[B,d]` |
| Head | `Linear(→1)` | `Linear(→1)` | `Linear(→1)` |
| Interpretability hook | Integrated Gradients / DeepSHAP | Integrated Gradients / DeepSHAP | attention rollout |
| Key HPs | channels, n_blocks, kernel | hidden, layers, cell | d_model, n_heads, n_layers, ff |

All three feed the **fusion rung** (rung 5) by exposing their pooled embedding,
which is late-fused with the clinical MLP branch.

---

*Sources: `models/cnn1d.py`, `models/rnn.py`, `models/transformer.py`,
`data/prepare_sequences.py`, `data/sequence_dataset.py`, `training/train.py`,
`config.py`.*
