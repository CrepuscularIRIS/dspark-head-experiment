# GPT-5.5 Pro: Sequential Correction Head Design (2026-07-02)

**Source**: ChatGPT Pro 扩展, chat "Sequential Correction Head Design"
**Input**: Ceiling probe results (4 datasets, per-position acceptance) + DPC signal + constraints

## Key Diagnosis: Latent Branch Loss

The open-ended exponential decay is NOT just "later positions are harder" — it's **latent branch loss**.

- **Math**: target hidden state + previous token already constrain the next tokens strongly (operators, digits, equation formatting). Missing dependencies behave like roughly additive local error → **linear decay**.
- **Open-ended (Alpaca/MT-Bench)**: first 1-2 sampled draft tokens choose a continuation **branch** (style, entity, wording, discourse move, sentence plan). Later tokens depend on that branch, not merely x_{k-1}. Once the branch variable is dropped, every later position pays another overlap penalty → **multiplicative/exponential decay**.
- DPC's ~30% TV when swapping x_{k-2} is exactly the signature that x_{k-1} alone is insufficient.

## Parameter Budget Clarification

VanillaMarkov's W1 (Embedding(vocab, 256)) + W2 (Linear(256, vocab)) = ~77.8M params at rank 256. The <5M budget should count only NEW trainable parameters. Strategy: freeze W1/W2, train only the sequential residual.

## Warm-Start Formula (shared by all candidates)

```
logits_k = base_logits_k + W2(e_k + δ_k)
```
where e_k = W1[x_{k-1}] (frozen Markov), δ_k = learned residual (zero at init).

---

## Rank 1: RSMH (Residual State-Space Markov Head) — RECOMMENDED

**Architecture**: Keeps a 256-d recurrent state in the same latent space as Markov embeddings.

For each position k:
- e_k = W1[x_{k-1}]  (frozen)
- u_k = e_k + U_e·e_k + U_s·s_{k-1} + U_h·h_k  (update)
- g_k = σ(G_e·e_k + G_s·s_{k-1} + G_h·h_k)  (gate)
- s_k = g_k ⊙ s_{k-1} + (1-g_k) ⊙ u_k  (state)
- δ_k = R_e·e_k + R_s·s_k + R_h·h_k + p_k  (residual)
- logits_k = base_logits_k + W2(e_k + δ_k)

**Key difference from previous RNN attempt**: starts as EXACT VanillaMarkov at init (all new matrices zero). State is a RESIDUAL correction, not a replacement. State exists at init (EMA of prev embeddings) but has zero effect on logits until R_s learns.

**Parameters**: ~2.36M new trainable (+ optional LoRA-8: ~3.58M total)
- U branches: (2r + d) × r = 3072 × 256 ≈ 0.79M × 3 = 2.37M
- Position residuals: 7 × 256 ≈ 0.002M

**Why it beats Markov**: s_k carries compressed path x_0,...,x_{k-1}. Critical for open-ended text where early tokens choose a continuation mode (list, caveat, direct answer, definition, quoted phrase, syntactic construction).

**Kill experiment**: Falsify if (1) open-ended Pos3-6 gain < +3 absolute points, (2) forcing s_k=0 at eval changes TV by <5% at positions 2-6, (3) improves training TV but not acceptance length.

---

## Rank 2: CLR (Causal Latent Retrieval Head)

**Architecture**: Instead of compressing all history into one EMA state, keeps last 6 Markov latents separately. Uses tiny causal attention over the within-block buffer.

- q_k = RMSNorm(e_k + Q_h·h_k)
- K_j = RMSNorm(e_j + K_h·h_j), V_j = e_j + V_h·h_j
- c_k = Σ_{j<k} softmax(q_k^T K_j / √r + b_{k-j}) V_j
- δ_k = R_c·c_k + R_e·e_k + R_h·h_k

Still O(1) per position because block is capped at 7 tokens.

**Parameters**: ~2.76M (+ LoRA-8: ~3.98M)

**Why it beats Markov**: Direct access to x_{k-2}, x_{k-3}, etc. without blurring into one recurrent vector. DPC probe specifically says x_{k-2} matters. Useful when the relevant earlier token is NOT the immediately previous one ("The main reason is..." → later token depends on "reason", not "is").

**Kill experiment**: Falsify if attention over lags remains uniform, or if replacing cache with only x_{k-1} gives same acceptance.

---

## Rank 3: FIR-Markov (Causal Finite Impulse Response)

**Architecture**: Simplest. Ring buffer of previous Markov embeddings + weighted sum.

- δ_k = R_h·h_k + Σ_{i=1}^6 α_{k,i} T_i m_k^i
- α_{k,i} = σ(a_i^T h_k + b_i)  (scalar lag gates)
- m_k^i = e_{k-i-1}  (previous Markov embeddings)

No recurrence, no attention. Just a causal filter over previous draft tokens.

**Parameters**: ~1.13M

**Why it beats Markov**: Direct "DPC probe says x_{k-2} matters" architecture. Should help phrase-local dependencies. Less likely to capture abstract branch state. Almost impossible to catastrophically fail.

**Kill experiment**: Falsify if lag-2 + lag-3 gives < +1 absolute point on Pos2-6.

---

## Recommended Experiment Order

1. **FIR-Markov K=2/K=6**: fast sanity check that longer in-block memory helps under DeepSpec's native loop
2. **RSMH**: main candidate (highest expected impact)
3. **CLR**: if RSMH improves but leaves large Pos4-6 gap on open-ended tasks

**Success signature**: MT/Alpaca Pos3-6 gain >> Math Pos3-6 gain (confirms latent branch loss hypothesis)

Track: ΔTV@k, Δaccept@k, ΔAccLen by dataset
