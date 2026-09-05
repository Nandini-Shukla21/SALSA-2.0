# SALSA 2.0 — Lightweight Neural Cryptanalysis of LWE/RLWE

**Final technical report**

All figures in this report are read from recorded artifacts under `results/`.
No measurement, comparison or citation has been invented.

---

## 1. Abstract

SALSA (Wenger, Chen, Charton, Lauter, NeurIPS 2022) attacks Learning With Errors
by training a transformer on public samples `(A, b)` to predict `b`, then reading
the secret back out through chosen inputs the model never saw in training. Its
published configuration uses roughly 51M parameters.

SALSA 2.0 asks whether the attack survives a ~12× parameter reduction. It
reimplements the pipeline from scratch, CPU-first, under a hard 4–5M parameter
budget, and adds two things the released source lacks: a secret-free recovery
decision rule, and the paper's independent residual verification step.

The final model, **NACT-F**, has **4,238,208 trainable parameters**. Its single
architectural change from the baseline is the input representation: **one token
per coordinate** instead of two digit tokens, which halves the encoder sequence
and makes absolute coordinate identity expressible. A controlled ablation showed
this representation, not the numerical features originally proposed alongside it,
carries the improvement.

On the diagnostic instances **n=12, h=2** and **n=20, h=2** with q=251 and
sigma=3, NACT-F trained on 100,032 samples achieves exact secret recovery at both
dimensions, with coordinate accuracy 1.0000, Hamming distance 0, 8 of 10 probe
multipliers succeeding individually, and independent residual verification
passing at both. The 4,131,200-parameter baseline recovers nothing at n=12.

**This is not a practical break of LWE or RLWE.** The demonstrated instances have
C(12,2) = 66 and C(20,2) = 190 possible secrets.

---

## 2. Problem statement

Given public LWE samples `(A, b)` with `b = A s + e mod q`, recover the binary
secret `s` — using only public data, and then confirm the recovered candidate
independently.

Three questions drive the project:

1. Can a model an order of magnitude smaller than SALSA's retain secret-recovery
   capability?
2. If it fails, *why* — capacity, training budget, or representation?
3. Can the whole attack be made to run reproducibly on a CPU?

The scientific hazard throughout is self-deception. A sparse secret makes an
all-zeros guess score well; a model can predict `b` accurately in-distribution
and still fail completely on the probes recovery depends on; and the released
reference resolves candidate polarity against the true secret, which cannot be
part of an attack. Each of these is addressed explicitly below.

---

## 3. Background: LWE and RLWE

For a prime modulus `q`, dimension `n`, secret `s ∈ {0,1}^n` of Hamming weight
`h`, and small errors `e` drawn from a discrete Gaussian of standard deviation
`sigma`, each sample is a row `a ∈ Z_q^n` together with

```
b = a · s + e   (mod q)
```

`h` is **the number of non-zero secret coordinates**. Recovering `s` from many
such samples is the LWE problem, and its hardness underpins several
post-quantum schemes.

In **RLWE**, rows of `A` are not independent: each block of `n` rows is generated
by rotating a single vector, giving circulant (or negacyclic) structure. This is
the setting used throughout this project and in the SALSA paper.

Parameters used here: **q = 251, sigma = 3, h = 2, n ∈ {12, 20}**, RLWE
circulant.

Two consequences matter for interpretation:

- **A sparse secret gives accuracy away.** An all-zeros candidate scores
  `(n − h)/n` coordinate accuracy — 0.8333 at n=12 and 0.9000 at n=20. Coordinate
  accuracy alone is therefore never evidence of recovery; only exact recovery is.
- **The error bounds what any attacker can achieve.** Even an attacker who knew
  `s` exactly would still pay for the noise in `b`, which sets an irreducible
  floor on prediction loss.

---

## 4. Original SALSA

Established by executing and inspecting the released source (full audit in
`results/original_fidelity_audit/`).

**Representation.** Integers are written as base-81 digits, least-significant
first, fixed width, no separator — `2n + 2` tokens per row. SALSA 2.0's encoder
reproduces this token-for-token; the equivalence was confirmed by running the
original `encoders.py` against ours across bases 2, 7 and 81.

**Model.** A Universal Transformer — one shared layer applied repeatedly, plus a
learned copy gate. The paper credits sharing and gating with a substantial
sample-efficiency gain.

**Direct recovery.** Feed the model `a = K·e_i`. Since `a · s = K·s_i mod q`, the
answer should sit near `0` when `s_i = 0` and near `K` when `s_i = 1`. Reading all
`n` coordinates gives a candidate secret.

**Limitations found in the audit:**

| finding | consequence |
|---|---|
| `evaluator.py` resolves candidate polarity **against the true secret**, reporting the best of six variants scored that way | Cannot be part of an attack. SALSA 2.0 replaced it with an anchored rule needing no ground truth, which is a strictly harder criterion. |
| The paper's §4.4 residual verification is **not implemented** | SALSA 2.0 implements it. |
| The supplied `transformer.py` had been modernised to **disable** the UT looping and copy gate | The supplied model reference is not the paper's architecture. SALSA 2.0 keeps both. |
| `K` is reduced modulo `B^int_len` (6561), not modulo `q` | Several published `K` values land outside `Z_q` entirely. SALSA 2.0 reduces mod `q`. |
| Each training instance is reused ~10× by default | Sample-count comparisons against the paper are not like-for-like; SALSA 2.0 used no reuse. |

**Scaling concerns.** Roughly 51M parameters and ~3.9M distinct training samples.
Probe inputs are never generated during training — the measured probability that
a training row resembles a `K·e_i` probe is about 4.8 × 10⁻²⁶ — so the original
also requires its transformer to extrapolate to inputs its training distribution
does not produce.

---

## 5. Motivation for SALSA 2.0

The V1 baseline, **Salsa2-GatedUT** (4,131,200 parameters), reproduced the
architecture faithfully at 1/12 the scale. At n=12, h=2 on 100,032 samples it
learned in-distribution — validation loss 1.7413, acc_tau 0.3550 — but **direct
recovery failed completely**: 0 of 10 probe multipliers recovered the secret,
58% of probe outputs were undecodable, and the model answered a single constant
value on 8 of 10 multipliers.

A generalization study then measured *why*. Feeding the frozen model inputs of
decreasing density, and scoring against the best input-blind constant predictor
on the same targets, gave a monotone collapse:

| non-zero coordinates | 12 | 10 | 8 | 6 | 4 | 2 | 1 | `K·e_i` probes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 lift over best constant | +0.1309 | +0.0991 | +0.0718 | −0.0527 | −0.1787 | −0.3730 | −0.4800 | **−0.4417** |

Spearman ρ = 1.000 against sparsity, exact permutation p = 0.0004. The lift
crosses zero between 8 and 6 non-zero coordinates: below that, the model's output
carries no usable information about its input. Probes sit at the far end of the
same curve, not on a separate cliff.

The diagnosis was therefore **representation, not capacity**: a probe is a
1-in-10²⁵ input under the training distribution, and in base-81 lsb-first
"this coordinate is zero" is a conjunction across two token positions that the
encoder must spend attention to compute.

---

## 6. SALSA 2.0 system design

Design constraints held fixed throughout:

- **CPU-first.** Pure PyTorch, no CUDA dependency, no custom kernels, no
  sequential scans. Every result was produced on a laptop CPU.
- **Hard parameter budget.** 4,000,000 ≤ trainable ≤ 5,000,000, with no padding
  parameters. Counts verified three ways: actual `numel`, a component breakdown
  in which every named tensor maps to exactly one component, and an analytical
  formula computed without touching the model.
- **Reproducibility.** Every experiment runs from a config file plus a fixed
  seed. Checkpoints carry a config fingerprint and the pipeline refuses to load
  one whose recorded identity disagrees with the config.
- **Structural secret isolation.** The public sample type carries no secret
  field, and tests scan the recovery package's source text for any reference to
  ground truth.

---

## 7. NACT and NACT-F

### The representation change

| | V1 / original | NACT and NACT-F |
|---|---|---|
| tokens per coordinate | 2 digit tokens | **1 coordinate token** |
| encoder sequence | `2n + 2` | **`n + 2`** |
| n=12 | 26 | **14** |
| n=128 | 258 | **130** |
| coordinate identity | RoPE (relative only) | learned **absolute** embedding |

`b = Σ aᵢ sᵢ` binds coordinate `i` to bit `sᵢ`, so the task needs absolute
coordinate identity; rotary embeddings supply only relative offsets. With one
token per coordinate, position *is* coordinate index. The coordinate table is a
fixed `128 × 512` matrix, so **the parameter count does not vary with `n`**.

### Full NACT versus NACT-F

**Full NACT** (4,241,288) adds a centered residue feature, a `cos`/`sin` Fourier
pair (the characters of `Z_q`, chosen because addition mod `q` is addition of
angles), a zero/non-zero indicator, a learned zero-coordinate vector, and a
per-head attention bias on zero-coordinate keys.

**NACT-F** (4,238,208) removes all six, keeping the coordinate token, the base-81
digit embeddings it is built from, and the absolute coordinate embedding.

The ablation was decisive. Against full NACT at matched depth, seed and budget:

| metric | full NACT | NACT-F | difference | 3σ seed band | resolvable? |
|---|---:|---:|---:|---:|:---:|
| validation loss | 0.9388 | 0.9891 | +0.0503 | 0.0696 | no |
| acc_tau | 0.9839 | 0.9839 | +0.0000 | 0.0064 | no |
| exact integer accuracy | 0.1011 | 0.0913 | −0.0098 | 0.0275 | no |

Every difference falls inside full NACT's own seed-to-seed noise. NACT-F retains
100% of the V1→NACT acc_tau gap while removing six components and 3,080
parameters (0.073%). **The one-token representation is the important
simplification.**

One caveat is recorded rather than smoothed over: NACT-F's sparse-input lift at
the extreme end is roughly half full NACT's (+0.0500 versus +0.1167 on probes).
The margin narrowed; the recovery outcome did not change.

### Model dimensions

Encoder 512 wide, 8 heads, 1 shared parameter set, `T_e = 2` passes, FFN
multiplier 4. Decoder 128 wide, 4 heads, 1 shared set, `T_d = 2`, FFN hidden 512.
RMSNorm, GELU, no biases, RoPE on self-attention, copy gate on every layer,
vocabulary 85. Loops reuse the shared set, so effective depth costs no parameters.

---

## 8. Data generation and encoding

Samples are generated on demand from a fixed problem instance, index-seeded so
any batch is reproducible on its own and a run can resume mid-stream. `A` rows
come from RLWE circulant blocks; errors are drawn from an exact truncated
discrete Gaussian.

**Public/private separation is structural.** The public sample type holds only
`(A, b, q, structure)` — it has no secret field, so a training loop physically
cannot receive one. Ground truth lives in a separate type used only by evaluation.

`LatticeCodec` encodes at base 81, lsb-first, fixed width, no separator,
producing `2n + 2` tokens framed by `<bos>`/`<eos>` over an 85-token vocabulary.
NACT-F consumes this same token layout and folds it into `n + 2` coordinate
positions, reconstructing each integer losslessly — asserted by test for every
dimension from 12 to 128 and for both representations.

---

## 9. Training and evaluation methodology

Identical protocol for every model compared: AdamW, fixed learning-rate schedule
with 200 warmup steps, batch size 64, 100,032 samples seen, 2,048 validation
sequences, monitored on validation loss, seed fixed and shared across arms.

**Baselines are computed exactly, not approximated**, because several of them are
high enough to masquerade as success:

| baseline | value | why it matters |
|---|---:|---|
| chance token accuracy | 0.44622 | base-81 lsb-first makes token accuracy near 45% free |
| chance acc_tau | 0.19287 | |
| chance exact integer accuracy | 0.00398 | |
| **marginal-only loss** | **1.86498 nats** | a model that learned only the output marginals scores this — the real zero point |
| uniform loss | 4.44265 nats | far too generous a reference |
| irreducible floor | 0.83920 nats | what a perfect attacker still pays for the noise |

`acc_tau` uses the paper's plain absolute difference `|b − b̂| ≤ 0.1·q`. A
circular-distance variant is retained but explicitly labelled a diagnostic and
never substituted for the paper's metric.

---

## 10. Secret recovery method

For each coordinate `i` and multiplier `K`, build the probe `a = K·e_i` reduced
mod `q`. Since `a · s = K·s_i mod q`, the noiseless answer is `0` when `s_i = 0`
and `K` when `s_i = 1`.

**Decision rule (anchored).** Assign each prediction to whichever of `0` or
`K mod q` it is nearer on the ring `Z_q`, giving a signed margin. This needs no
ground truth and has no polarity ambiguity — deliberately unlike the released
code, which resolves polarity against the true secret.

**K sweep.** `[1, 31, 63, 94, 125, 126, 157, 188, 220, 250]`. A probe's entire
information budget is its ring separation `min(K mod q, q − K mod q)`, maximal at
`K ≈ q/2`. `K = 1` and `K = 250` have separation 1 and are included as
**negative controls that should fail**.

**Aggregation.** Two secret-free rules:

- **`aggregate`** (default) — separation-weighted vote across all K.
- **`best_margin`** (diagnostic) — highest mean margin, guarded (§14).

**What recovery may not do.** It may not inspect the secret, use it to choose K
or polarity, rank candidates by it, or select a checkpoint by recovery
performance. `DirectRecovery` takes `(model, codec, method, batch_size)` and
`recover(k_values, …)` — there is no parameter through which a secret could pass.

---

## 11. Independent verification

Implements the paper's §4.4 residual test, which the released source omits.

Given a candidate `c` and fresh public samples, compute `r = (b − A c) mod q` and
map into centered representatives `[−q/2, q/2)` — without centering, a residue of
`q − 1` reads as a large positive error when it is really `−1`.

If `c = s`, then `r = e`: tight around zero at the configured sigma. If `c` is
wrong by any non-zero delta, `r = A·delta + e`, close to uniform on `Z_q` with
standard deviation `sqrt((q² − 1)/12) ≈ 72.5`. **That ~24× gap is the evidence**,
not any goodness-of-fit statistic.

Acceptance criteria, fixed before any statistic was computed:

1. centered residual std ≤ 1.5·sigma
2. mean |centered residual| ≤ 1.5·sigma·√(2/π)
3. candidate std < 0.25 × smallest baseline std
4. fraction within 3·sigma ≥ 0.99

The verifier's signature is the guarantee: `(A, b, candidate, q, sigma)`. No
secret, model output or training label reaches it.

---

## 12. Experimental results

### Prediction — how well the model predicts `b`

| n | h | model | parameters | samples | loss | acc_tau | exact integer acc | token acc | decode failures |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 12 | 2 | **NACT-F** | 4,238,208 | 100,032 | **0.9891** | **0.9839** | **0.0913** | 0.6934 | 0.0000 |
| 20 | 2 | **NACT-F** | 4,238,208 | 100,032 | **0.9764** | **0.9863** | **0.1035** | 0.6995 | 0.0000 |
| 12 | 2 | V1 GatedUT | 4,131,200 | 100,032 | 1.7413 | 0.3550 | 0.0078 | 0.5324 | 0.0000 |
| 20 | 2 | V1 GatedUT | 4,131,200 | 200,000 | 1.8233 | 0.2344 | 0.0034 | 0.4590 | 0.0000 |

V1's loss sits essentially at the marginals-only baseline of 1.86498 — at this
budget it has largely not learned secret-related structure. Its exact integer
accuracy at n=20 (0.0034) is **below** its own chance level of 0.00398. NACT-F's
loss sits closer to the 0.83920 irreducible floor. Note V1's n=20 row used
**twice** the sample budget.

### Secret recovery — did the attack recover `s`?

| n | h | exact secret recovery | coordinate accuracy | Hamming distance | successful K | probe decode validity | verification |
|---:|---:|:---:|---:|---:|---:|---:|:---:|
| 12 | 2 | **YES** | 1.0000 | 0 | 8/10 | 1.0000 | **PASS** |
| 20 | 2 | **YES** | 1.0000 | 0 | 8/10 | 1.0000 | **PASS** |
| 12 | 2 (V1) | NO | 0.1667 | 10 | 0/10 | 0.4167 | — |

The two failing K are exactly the negative controls. Free all-zeros baselines:
0.8333 at n=12, 0.9000 at n=20 — both beaten only by an exact recovery.

### Verification residuals

| | n=12 | n=20 |
|---|---:|---:|
| recovered candidate std | **2.998** | **2.941** |
| all-zeros baseline std | 71.803 | 72.477 |
| all-ones baseline std | 72.676 | 72.756 |
| configured sigma | 3.0 | 3.0 |

### Reproducibility across secrets

At n=12, three checkpoints trained under different seeds — three different
secrets — were attacked: **3/3 exact recovery, 3/3 verification PASS**, mean
coordinate accuracy 1.000. Only one secret has been tested at n=20.

### Prediction quality is not recovery success

Exact integer accuracy is about 0.10 while secret recovery is exact. These are
different claims: recovery does not need per-sample perfection, it needs probe
responses to fall on the correct side of the anchor. Conversely V1 at n=12
reached acc_tau 0.3550 — clearly above its 0.19287 chance level — and recovered
nothing. **Prediction quality is necessary but not sufficient.**

---

## 13. End-to-end pipeline

```text
public LWE/RLWE data → encoding → NACT-F → predict b
   → direct secret recovery → candidate secret
   → independent residual verification → evaluation
```

One entry point:

```python
from salsa import run_salsa2_pipeline
result = run_salsa2_pipeline("configs/pipeline_n12.yaml")
```

The three stages are separate fields of the result, never merged:

| stage | permitted inputs |
|---|---|
| recovery | public probes and model predictions only |
| verification | `(A, b, candidate, q, sigma)` only |
| evaluation | the only stage that loads the secret, and it runs **last** |

Ordering is enforced by construction: the secret is not read until a candidate
*and* a verdict exist. `evaluate=False` runs the whole attack with no ground
truth available, which is the real adversarial setting.

Both established results replay through the integrated pipeline with **12/12
checks matching**, in about one second per configuration.

---

## 14. Recovery rule hardening

The n=20 experiment exposed a real defect in the single-K selection rule, and it
is documented rather than quietly patched.

**The degeneracy.** At `K = 1` the two hypotheses `b ≈ 0` and `b ≈ 1` are
adjacent. A model answering a constant `0` is then at distance 0 from one anchor
and 1 from the other, so `|score| = 1` for every coordinate — the **maximum
possible margin**. The rule rewards the least informative probes precisely
because they are least informative. At n=20 it selected `K = 1`, a negative
control, whose candidate was not an exact recovery.

**The fix.** `min_separation` excludes such probes from *selection*, defaulting
to `floor(sigma) + 1` — derived from the problem, since a probe can only
discriminate if its hypotheses are further apart than the error scale. At
sigma = 3 this is 4, which excludes `K ∈ {1, 250}`. Excluded probes are still
measured and still vote with their small separation weight; they are excluded
from selection, not from the evidence.

**The default changed** to the separation-weighted `aggregate` vote, which had
already recovered correctly at n=20 because the degenerate probes carry weight
0.008 each. Both rules remain secret-free: separation is a function of `K` and
`q` only.

Measured effect: the guarded rule selects `K = 188` at n=20 instead of `K = 1`.

---

## 15. Limitations

- **Only n=12, h=2 and n=20, h=2 were experimentally demonstrated.**
- **n=30, n=50 and n=128 were never trained or tested.** The architecture accepts
  them and forward cost has been benchmarked, but no model exists at those
  dimensions.
- **One trained secret and checkpoint per demonstrated dimension** — three
  secrets at n=12 via three seeds, but only one at n=20. Seed-to-seed variance at
  n=20 is unmeasured.
- **Diagnostic low-dimensional instances.** C(12,2) = 66 and C(20,2) = 190
  possible secrets. Recovering a secret from a 190-element space is not an attack
  on LWE.
- **The results do not establish a practical LWE/RLWE security break**, and
  nothing here should be cited as evidence about deployed parameter sets.
- **Two dimensions is not a scaling law.**
- **Component attribution is partial.** Ablation F removed six components at
  once; variants C, D and E were designed but not run, so no individual feature
  has been attributed.
- **No comparison to the original at matched conditions.** Model size, sample
  budget and sample-reuse policy all differ, as the fidelity audit documents.
- **The `min_separation` guard is validated at sigma = 3 only.**
- **The distinguisher recovery mode is not implemented.**

---

## 16. Conclusion

A 4,238,208-parameter CPU-trained transformer performs the complete SALSA direct
recovery pipeline — learn, recover, verify — at n=12 and n=20 with h=2, on
100,032 training samples per dimension, where a 4,131,200-parameter faithful
reimplementation of the original architecture recovers nothing.

The gap is not capacity: the two models differ by 2.6% in parameters. It is
**representation**. Spending one token per coordinate instead of two digit tokens
halves the encoder sequence and makes absolute coordinate identity expressible,
and a controlled ablation showed this change alone accounts for the improvement —
the numerical, zero-aware and sparse-attention features proposed alongside it
were removable without measurable loss at one seed.

The diagnostic path that led there is reusable independently of the result: V1's
failure was traced to sparse-input generalization by measuring lift over the best
input-blind constant across a sparsity sweep, which showed a monotone collapse
with the recovery probes at the far end of the same curve rather than on a
separate cliff.

**What this does not show** is anything cryptographic. The instances are
diagnostic by construction, chosen so the whole pipeline could be exercised
end to end on a CPU. Whether the result survives at cryptographically meaningful
dimensions is untested and is the obvious next question.

---

## 17. Future work

None of the following has been performed.

1. **Dimension scaling** — n=30, then n=50, holding the protocol fixed. The
   binding constraint is expected to be the sample requirement rather than the
   architecture, since the parameter count does not vary with `n`.
2. **Multi-seed validation** — several secrets per dimension, so recovery is
   reported as a rate with an uncertainty estimate rather than a single outcome.
3. **Complete the ablation matrix** — variants C, D and E, to attribute the
   residual sparse-input margin that NACT-F narrowed without changing the
   recovery outcome.
4. **Scalability analysis** — how the sample requirement grows with `n` and `h`,
   and whether the one-token advantage holds as sequences lengthen toward n=128.
5. **Larger Hamming weights** — only h=2 has been demonstrated end to end.
6. **Validate the `min_separation` guard across sigma.**

---

## Provenance

| claim area | artifact |
|---|---|
| Original SALSA comparison | `results/original_fidelity_audit/` |
| V1 sparse-input collapse | `results/recovery_generalization/` |
| V1 vs NACT at equal depth | `results/equal_depth_ablation/` |
| Ablation F | `results/nact_ablation_F/` |
| n=12 recovery and verification | `results/nact_ablation_F_recovery/`, `results/v2_verification/` |
| Multi-secret reproduction | `results/v2_recovery_robustness/` |
| n=20 training and recovery | `results/n20_nact_f_pilot/`, `results/n20_nact_f_recovery/` |
| Pipeline replay | `results/final_pipeline/pipeline_report.md` |

Test suite: **687 passed, 2 skipped**.

**No training was performed in producing this report.**
