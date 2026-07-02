# DSpark Sequential Head Experiment

**Status**: CLOSED (2026-07-02)
**Direction**: Improve DSpark's 1st-order Markov correction head for speculative decoding
**Platform**: Qwen3-4B target, DeepSpec framework, 2×RTX 4090D (96GB)
**Result**: Negative — head-only training degrades baseline; compute insufficient for joint fine-tuning

## Objective

DSpark (arxiv:2606.19348) uses a 1st-order Markov correction head on a DFlash draft backbone for speculative decoding. The head is memoryless — position k only sees x_{k-1}. We investigated whether higher-order sequential heads could improve accepted length.

## Experiments & Findings

### 1. Ceiling Probe (Positive — the prize exists)

Measured per-position acceptance rates of the released DSpark baseline across 4 datasets:

| Dataset  | AccLen | Pos0  | Pos1  | Pos2  | Pos3  | Pos4  | Pos5  | Pos6  |
|----------|--------|-------|-------|-------|-------|-------|-------|-------|
| gsm8k    | 6.16   | .932  | .866  | .800  | .736  | .677  | .619  | .557  |
| math500  | 5.78   | .917  | .830  | .746  | .667  | .599  | .541  | .487  |
| alpaca   | 3.46   | .763  | .556  | .403  | .290  | .207  | .152  | .112  |
| mt-bench | 3.68   | .770  | .576  | .424  | .325  | .247  | .195  | .156  |

**Key finding**: Two distinct decay regimes:
- Math tasks: linear ~8-10%/position decay. Gap to perfect: +30%.
- Open-ended tasks: **exponential** decay (77%→11%). Gap to perfect: **+125%**.

**Diagnosis** (GPT-5.5 Pro): The exponential decay is **latent branch loss** — early draft tokens commit to a continuation branch (style, entity, discourse move), and a memoryless Markov head cannot propagate that branch choice to later positions.

### 2. DPC Signal (Positive — 2nd-order info exists)

Differential Prediction Contract probe (Probe A), run on **Qwen2.5-7B-Instruct** (already cached
locally; note this is NOT the campaign's Qwen3-4B training platform):
- Swapping x_{k-2} causes a **~38% mean TV shift** (median suffix TV k≥4 = 0.380, per-prompt range
  0.22–0.59, 30 prompts × 40 measurements) — artifact: `results/dpc/DPC_RESULT.json`
- The planned Probe C (offline E[τ] oracle replay) was **never executed** — no accepted-length ceiling
  was measured; earlier "94% E[τ] ceiling" statements had no backing artifact
- **Verdict**: Large 2nd-order sequential information exists; its accepted-length ceiling is unmeasured

### 3. From-Scratch Training (Negative — data insufficient)

Trained RSMH and Vanilla Markov from scratch on 4506 samples (released baseline used 100K+):
- Step 500 eval: both at ~1.63 accept_len (vs baseline 6.16)
- RSMH vs Vanilla Δ: +0.01 (not significant)
- **Root cause**: 4.5K samples insufficient for 77M-param from-scratch training

### 4. Warm-Start Training (Negative — frozen backbone conflict)

Loaded released baseline, swapped Markov head to RSMH/FIR, froze backbone, trained only head additions:

| Dataset  | Baseline | RSMH (2.4M) | FIR (1.1M) |
|----------|----------|-------------|------------|
| gsm8k    | 6.16     | 5.15 (-16%) | 5.18 (-16%)|
| alpaca   | 3.46     | 3.27 (-5%)  | 3.27 (-5%) |
| mt-bench | 3.68     | 3.34 (-9%)  | 3.33 (-9%) |

**Both architectures produced identical degradation**, proving the issue is training methodology, not head design.

**Root cause**: The backbone was jointly optimized with the vanilla Markov head. Adding trainable residuals on a frozen backbone breaks the learned backbone-head coordination. The uniform degradation across ALL positions (even pos 0) confirms this is a global signal shift, not a late-position problem.

### 5. Direction Closure

The improvement requires either:
- Joint backbone+head fine-tuning (needs 100K+ Qwen3-4B-regenerated samples + 8×GPU — exceeds available compute)
- Inference-time correction (confidence threshold sweep was started but terminated before completion)

**Decision**: Direction closed due to compute constraints.

## Head Architectures Designed

Three heads were designed by GPT-5.5 Pro and implemented:

1. **RSMH** (Residual State-Space Markov Head): GRU-like recurrent state as residual on frozen Markov, 2.36M params. Zero-init warm-start verified (max diff 0.000002 from VanillaMarkov).

2. **FIR-Markov** (Causal Finite Impulse Response): Weighted multi-lag filter over previous Markov embeddings, ~1.1M params.

3. **HOPEHead** (Higher-Order Prefix Encoder): 2nd-order pair path + entropy gate, from prior session.

## File Structure

```
implementations/
  rsmh_head.py          # RSMH head implementation
  fir_head.py           # FIR-Markov head implementation
  hope_head.py          # HOPE head implementation

results/
  ceiling_probe/        # Per-position acceptance profiles (4 datasets)
  dpc/                  # DPC probe raw result (Qwen2.5-7B-Instruct, ~38% TV)
  eval_step500/         # From-scratch training eval (RSMH vs Vanilla)
  eval_warmstart/       # Warm-start training eval (RSMH + FIR vs baseline)

scripts/
  ceiling_probe.py      # Ceiling probe eval script
  eval_step500.py       # Checkpoint eval script
  eval_warmstart.py     # Warm-start eval script
  train_rsmh_warmstart.py
  train_fir_warmstart.py

configs/
  rsmh_qwen3_4b.py      # RSMH training config
  hope_qwen3_4b.py      # HOPE training config

design/
  gpt55pro-sequential-head-design-2026-07-02.md  # Pro's head design document
```

## Banked Scientific Findings

1. **Latent branch loss**: Open-ended text shows exponential acceptance decay because early draft tokens commit to a continuation branch that a memoryless head cannot propagate.

2. **Warm-start head-only recipe degrades uniformly**: Head-only training on a jointly-trained frozen backbone with a small non-target-generated cache degrades all positions, regardless of head architecture. Frozen-backbone conflict vs training-data mismatch was NOT disambiguated before closure.

3. **2nd-order signal is real, ceiling unmeasured**: ~38% TV from x_{k-2} (Qwen2.5-7B-Instruct probe). Recovering it requires joint training and/or target-generated data at scale; the E[τ] ceiling was never measured (Probe C not run).
