# SALSA 2.0 — Lightweight Neural Cryptanalysis of LWE/RLWE

Experimental neural cryptanalysis on controlled low-dimensional LWE/RLWE-style
instances. This project investigates whether a **compact coordinate-level
Transformer** can learn secret-dependent structure from public samples and
support **direct secret recovery**, at roughly one twelfth the scale of the
published SALSA model, running entirely on a CPU.

The final model, **NACT-F**, has **4,238,208 trainable parameters** and completes
the full attack — learn, recover, verify — at n=12 and n=20 with h=2.

> ### ⚠ Important scientific boundary
>
> **SALSA 2.0 demonstrates neural secret recovery on controlled low-dimensional
> LWE/RLWE-style instances. It does not demonstrate a practical break of modern
> LWE/RLWE cryptographic security.**
>
> The demonstrated instances have C(12,2) = 66 and C(20,2) = 190 possible
> secrets. These are diagnostic scales, chosen so the whole pipeline could be
> exercised end to end on a laptop. Nothing here should be read as evidence about
> deployed parameter sets.

---

## Table of contents

[1. Overview](#1-overview) · [2. Motivation](#2-project-motivation) ·
[3. Cryptographic background](#3-cryptographic-background) ·
[4. The learning problem](#4-the-lwerlwe-learning-problem) ·
[5. Original SALSA](#5-original-salsa) ·
[6. Limitations identified](#6-limitations-identified-in-original-salsa) ·
[7. SALSA 2.0](#7-salsa-20) · [8. Architecture evolution](#8-architecture-evolution) ·
[9. NACT-F](#9-final-nact-f-architecture) · [10. Data generation](#10-data-generation) ·
[11. Secret representation](#11-secret-representation) ·
[12. Integer encoding](#12-integer-encoding) · [13. Training](#13-training-pipeline) ·
[14. Metrics](#14-prediction-and-evaluation-metrics) ·
[15. Direct secret recovery](#15-direct-secret-recovery) ·
[16. Rule hardening](#16-recovery-rule-hardening) ·
[17. Verification](#17-independent-residual-verification) ·
[18. Pipeline](#18-end-to-end-pipeline) ·
[19. Ground-truth isolation](#19-ground-truth-isolation) ·
[20. Configuration](#20-experimental-configuration) ·
[21. Results](#21-experimental-results) · [22. V1 vs NACT-F](#22-v1-vs-nact-f) ·
[23. Ablation](#23-nact-ablation-findings) ·
[24. Reproducibility](#24-reproducibility-and-replay-validation) ·
[25. Structure](#25-project-structure) · [26. Installation](#26-installation) ·
[27. Usage](#27-usage) · [28. Tests](#28-tests) ·
[29. Final findings](#29-final-findings) · [30. Limitations](#30-limitations) ·
[31. Future work](#31-future-work) ·
[32. Interpretation](#32-scientific-interpretation) ·
[33. References](#33-references) · [34. Status](#34-license--project-status)

---

## 1. Overview

**SALSA** is a machine-learning attack on Learning With Errors. It trains a
sequence-to-sequence Transformer on public samples `(A, b)` to predict `b` from
`A`. If the model learns that map well enough, it has implicitly learned the
secret, and the secret can be read back out by feeding the model chosen inputs it
never saw during training.

**SALSA 2.0** is an independent, from-scratch reimplementation and extension. It
reproduces the mathematical setup and the released encoding exactly, then asks a
different question: *how small can the model be and still recover the secret?*

Three things distinguish it:

1. **Scale.** The published configuration uses approximately 51M parameters. This
   project caps the budget at 4–5M — a roughly 12× reduction — and runs on a CPU.
2. **Representation.** Instead of spending two digit tokens per coordinate, it
   spends **one coordinate token**, halving the encoder sequence and making
   absolute coordinate identity expressible.
3. **Methodology.** Every stage is isolated so the attack cannot consult the
   answer, and every recovered secret is independently verified against fresh
   samples using a residual test the released source does not implement.

---

## 2. Project motivation

The published SALSA result raises a question it does not answer: **is the model
size necessary?** A 51M-parameter model is expensive to train and to study. If
the attack survives at 4M, experimentation becomes cheap enough to iterate on,
and the architecture must earn its capacity rather than buy results with
parameters.

A second motivation is methodological. During this project's audit of the
released source, two issues emerged that affect how results should be read: the
reference recovery code resolves candidate polarity **against the true secret**,
and the paper's residual verification step is absent from the released code.
SALSA 2.0 addresses both.

---

## 3. Cryptographic background

Learning With Errors is a lattice problem whose hardness underpins several
post-quantum cryptographic schemes. An adversary is given many noisy linear
equations over `Z_q` and must recover the hidden solution vector. The noise is
what makes it hard: without it, Gaussian elimination solves the system.

**RLWE** (Ring-LWE) is the structured variant used by deployed schemes. Rows of
the public matrix are not independent — each block of `n` rows is generated by
rotating a single vector — which makes keys smaller and operations faster, at the
cost of extra algebraic structure.

---

## 4. The LWE/RLWE learning problem

Every public sample satisfies

```text
b = A s + e   (mod q)
```

| symbol | meaning | value in demonstrated experiments |
|---|---|---|
| `A` | public matrix / public linear structure, rows in `Z_q^n` | RLWE circulant rows |
| `s` | **the secret**, a binary vector in `{0,1}^n` — unknown to the attack | recovered, not given |
| `e` | noise vector, small, drawn independently per sample | discrete Gaussian |
| `b` | public observation, one integer in `Z_q` per row | computed as above |
| `q` | modulus, prime | **251** |
| `n` | lattice / secret dimension | **12, 20** |
| `h` | secret **Hamming weight** — the number of non-zero secret coordinates | **2** |
| `sigma` | configured noise scale | **3** |

The secret contains a controlled number of non-zero coordinates: exactly `h` of
the `n` entries are 1 and the rest are 0.

### Why these are diagnostic instances

The number of possible secrets is the number of ways to choose `h` positions
from `n`:

```text
C(12,2) = 66
C(20,2) = 190
```

**Recovering a secret from a 190-element space is not an attack on LWE.** These
dimensions were chosen so that training, recovery and verification could all be
exercised end to end on a CPU in minutes rather than GPU-weeks. They do not
represent realistic deployed cryptographic security parameters.

A second consequence of sparsity matters for reading every result below: an
all-zeros guess already scores `(n − h)/n` coordinate accuracy for free —
**0.8333 at n=12 and 0.9000 at n=20**. Coordinate accuracy alone is therefore
never evidence of recovery. Only **exact** recovery is.

---

## 5. Original SALSA

The following is established by direct inspection and execution of the released
source during this project's fidelity audit (`results/original_fidelity_audit/`).

**Public samples.** The attack consumes only `(A, b)` pairs, which a real
adversary could observe.

**Integer encoding and tokenization.** Each integer is written as fixed-width
base-81 digits, least-significant digit first, with no separator. An
`n`-dimensional row therefore becomes `2n + 2` tokens including boundary markers.

**Transformer processing.** A Universal Transformer: a single shared layer
applied repeatedly across loops, combined with a learned copy gate that lets the
layer leave some positions untouched on some iterations. The paper credits
sharing plus gating with a substantial sample-efficiency gain.

**Secret-dependent learning.** Training on `(A, b)` forces the model to compute
`a · s mod q`. Succeeding at that requires internalising `s`.

**Direct recovery.** Feed the model the chosen input `a = K · e_i`, where `e_i`
is the `i`-th standard basis vector. Because

```text
a · s = K · s_i   (mod q)
```

the noiseless answer is near `0` when `s_i = 0` and near `K` when `s_i = 1`.
Reading all `n` coordinates reconstructs a candidate secret.

**Scale.** The project records identify the original SALSA scale as approximately
**51M parameters**, trained on roughly 3.9M distinct samples with each reused
about ten times.

**Why digit-level representation lengthens sequences.** Two tokens per
coordinate means an `n`-coordinate row occupies `2n` positions. At n=128 that is
258 encoder positions rather than 130 — and coordinate identity becomes indirect,
since "which coordinate am I looking at" is `position // 2` rather than
`position`.

---

## 6. Limitations identified in original SALSA

Each item below is supported by this project's records.

### Model size

Approximately 51M parameters is far above the 4–5M lightweight target, making
each experiment expensive to run and to repeat.

### Representation

Digit-level encoding produces longer sequences and makes coordinate identity
indirect. Since `b = Σ aᵢ sᵢ` binds coordinate `i` to secret bit `sᵢ`, the model
needs to know *which* coordinate it is reading, and rotary embeddings encode only
relative offsets.

### Computational cost

Large models combined with substantial sample requirements raise the cost of
every experiment, which limits how many controlled variations can be explored.

### Recovery distribution shift

Recovery probes can differ sharply from anything seen in training. This project
measured the gap: under the training distribution, the probability that a
training row resembles a `K · e_i` probe is approximately **4.8 × 10⁻²⁶**. The
original therefore also requires its Transformer to extrapolate to inputs its
training distribution does not produce.

*This is an observation about the input distribution, not a proof that any
particular SALSA variant fails.* The original was not run at scale in this
project, and no claim is made about its behaviour.

### Reproducibility and fidelity issues found during audit

| finding | consequence |
|---|---|
| Encoding order and layout | Reproduced exactly. The audit executed the original `encoders.py` and confirmed token-for-token agreement across bases 2, 7 and 81. |
| Noise generation | The original samples `round(N(0, sigma))`; SALSA 2.0 samples an exact truncated discrete Gaussian. Measured total-variation distance 0.00042. |
| Ring construction | The original's generator negates the strict upper triangle of the circulant matrix, which makes it **negacyclic**. SALSA 2.0's demonstrated runs use the **circulant** variant. |
| Training instance reuse | The original reuses each instance about ten times by default. SALSA 2.0's runs use no reuse, so sample-count comparisons are not like-for-like. |
| Recovery polarity logic | The released `evaluator.py` produces candidate vectors and their inverses, then keeps whichever matches the **true secret**. That cannot be part of an attack. SALSA 2.0 replaced it with an anchored rule requiring no ground truth — a strictly harder criterion. |
| Residual verification | The paper's residual test is **not implemented** in the released source. SALSA 2.0 implements it. |
| Probe magnitude | The original reduces `K` modulo `B^int_len` (6561) rather than modulo `q`, which for several published `K` values leaves the probe outside `Z_q`. SALSA 2.0 reduces modulo `q`. |

---

## 7. SALSA 2.0

Project goals, all of which the repository addresses:

- **Reproduce the mathematical setup** — LWE and RLWE generation, exact-weight
  binary secrets, discrete Gaussian noise.
- **Reproduce encoding behaviour** — verified token-for-token against the
  released encoder.
- **Reduce model size** — a hard 4–5M trainable parameter budget with no padding
  parameters, verified three independent ways.
- **Reduce representation length** — one coordinate token instead of two digit
  tokens.
- **Make coordinate identity explicit** — a learned absolute coordinate
  embedding.
- **Investigate recovery** — direct secret recovery with a secret-free decision
  rule.
- **Isolate evaluation from ground truth** — structurally, not by convention.
- **Provide independent verification** — the residual test the released code
  omits.
- **Create a reproducible pipeline** — one entry point, config-driven, seeded.

---

## 8. Architecture evolution

| Version | Architecture | Parameters | Purpose |
|---|---|---:|---|
| **V1** | Salsa2-GatedUT | **4,131,200** | compact baseline — a faithful reduction of the original architecture |
| **V2** | Salsa2-NACT | **4,241,288** | coordinate/numerical-aware architecture |
| **Final** | **NACT-F** | **4,238,208** | simplified final architecture |

### V1 — Salsa2-GatedUT

The compact baseline. It reproduces the original architecture's essential
features at 1/12 the scale:

- **Shared Universal Transformer encoder** — one parameter set applied across
  configurable loops, so effective depth costs no parameters.
- **Configurable loops** — `T_e` and `T_d` are runtime knobs that do not change
  the parameter count.
- **Copy gate** — a learned per-position, per-channel gate blending each layer's
  input and output, letting a looped layer leave positions untouched.
- **RoPE self-attention** — rotary positional embeddings, which contribute zero
  positional parameters.
- **Compact decoder** — 128 wide against the encoder's 512, because the output is
  a single integer.
- **CPU-oriented design** — pure PyTorch, no CUDA dependency.

Parameter count: **4,131,200**. It consumes the digit-token representation
described in §5, giving `2n + 2` encoder positions.

### V2 — Salsa2-NACT

The Numerical-Aware Coordinate Transformer. It keeps V1's encoder body, decoder
and gating unchanged and replaces only the **input front end**, changing the
model from digit-centric to coordinate-centric processing.

Each coordinate becomes one token, built from:

- **digit-derived information** — embeddings of the same base-81 digits the codec
  already produces, so codec compatibility is preserved;
- **centered residue** — the signed integer value, symmetric about zero;
- **Fourier features** — `cos(2πx/q)` and `sin(2πx/q)`, the characters of `Z_q`,
  chosen because addition modulo `q` becomes addition of angles;
- **zero-aware information** — a zero/non-zero indicator and a learned
  zero-coordinate vector;
- **coordinate identity** — a learned absolute coordinate embedding;
- **sparse-aware attention bias** — one learned scalar per encoder head, added to
  the attention logit of keys at zero coordinates;
- **shared Transformer computation** — the V1 encoder body and decoder, unchanged.

Parameter count: **4,241,288**.

---

## 9. Final NACT-F architecture

**NACT-F** is V2 with the additional numerical, zero-aware and sparse-attention
features removed, retaining only the coordinate-level representation:

**Retained** — one coordinate token per `aᵢ`; the base-81 digit embeddings it is
built from; the absolute coordinate embedding; the full Transformer backbone,
decoder, copy gate, RMSNorm and RoPE.

**Removed** — centered residue, both Fourier features, the zero indicator, the
learned zero-coordinate vector, the sparse attention bias.

Parameter count: **4,238,208**.

### Why NACT-F became the final model

A controlled ablation at matched depth, seed and sample budget found that the
**one-token coordinate representation and coordinate identity accounted for the
dominant observed improvement**. NACT-F matched full NACT within one-seed noise
on every primary metric, and completed secret recovery just as well.

> **Careful statement.** The removed features were removable **without measurable
> loss in the tested ablation setting** — one seed, n=12, h=2, 100,032 samples.
> This is not a claim that they are universally useless. NACT-F's sparse-input
> margin at the extreme end was roughly half full NACT's, which is recorded in
> §23 rather than smoothed over.

### Model dimensions

| | encoder | decoder |
|---|---|---|
| width | 512 | 128 |
| heads | 8 | 4 |
| parameter sets | 1 (shared) | 1 (shared) |
| passes | `T_e = 2` | `T_d = 2` |
| FFN hidden | 2048 (multiplier 4) | 512 |

RMSNorm, GELU, no biases, RoPE on self-attention only, copy gate on every layer,
vocabulary 85.

The coordinate embedding is a fixed `128 × 512` table, so **the parameter count
does not vary with `n`** — one architecture covers n=12 through n=128, and only
the sequence length changes.

---

## 10. Data generation

Public samples are generated on demand from a fixed problem instance:

```text
b = A s + e   (mod q)
```

- `A` rows are drawn from RLWE rotation blocks (see §16 on the ring variant).
- `s` is a binary vector of **exact** Hamming weight `h`, constructed by choosing
  `h` distinct coordinate positions without replacement — not by independent
  coin flips, which would only give `h` on average.
- `e` is drawn from an exact truncated discrete Gaussian of scale `sigma`.

**Data generation and secret handling are separated structurally.** The public
sample type carries only `(A, b, q, structure)` and has **no secret field**, so a
training loop cannot receive one even by accident. Ground truth lives in a
separate type used only by evaluation.

**Determinism.** Each batch is seeded from a derivation of the master seed and
its index, so any batch is reproducible on its own and a run can resume
mid-stream. The secret is derived from a separate branch of the same seed, so
changing the data stream never changes which secret is attacked.

---

## 11. Secret representation

The secret is a binary vector `s ∈ {0,1}^n` with exactly `h` non-zero
coordinates. In the demonstrated experiments `h = 2`, so:

- at n=12 the secret has 2 ones and 10 zeros — 66 possibilities;
- at n=20 the secret has 2 ones and 18 zeros — 190 possibilities.

The recovery target is the full vector. Partial credit is reported (coordinate
accuracy, Hamming distance) but **only exact recovery counts as success**, for
the free-baseline reason given in §4.

---

## 12. Integer encoding

The released-code-compatible representation was deliberately reproduced, and
verified token-for-token by executing the original encoder.

- **Fixed width** — every integer occupies the same number of digit tokens.
- **Base 81** — two digits suffice for `Z_251`.
- **Least-significant digit first** — `digit0 = x mod 81` is emitted before
  `digit1 = x // 81`.
- **No separator** in the representation used for all demonstrated results.

### Special tokens

| Token | ID |
|---|--:|
| `<pad>` | 0 |
| `<bos>` | 1 |
| `<eos>` | 2 |
| `<sep>` | 3 |

Followed by 81 digit tokens, giving a **vocabulary of 85**.

### Representations R and P

Two layouts exist in the codebase. **R** (base 81, no separator) is the
released-code-compatible layout and produced **every demonstrated result**.
**P** inserts a separator between coordinates, giving `3n + 1` tokens instead of
`2n + 2`; it is implemented and tested but was not used for the final results.

### Sequence lengths

| | R, digit tokens | NACT-F, coordinate tokens |
|---|---:|---:|
| n=12 | 26 | **14** |
| n=20 | 42 | **22** |
| n=128 | 258 | **130** |

```text
Original-style digit representation

coordinate 1 → digit 1 + digit 2
coordinate 2 → digit 1 + digit 2
...
             ↓
        longer sequence


NACT / NACT-F

coordinate 1 → one coordinate token
coordinate 2 → one coordinate token
...
             ↓
        shorter sequence
        + explicit coordinate identity
```

With the fixed-width base-81 representation used here, the coordinate-level
representation **approximately halves** the sequence relative to the two-digit
layout. Exactly: `2n + 2` becomes `n + 2`, so the reduction approaches one half
as `n` grows and is slightly less at small `n` because both layouts carry the
same two boundary tokens.

NACT-F consumes the **same token layout the codec already produces** and folds it
into coordinate positions internally, reconstructing each integer losslessly.
This is asserted by test for every dimension from 12 to 128 and for both
representations, so no change to the data pipeline was required.

---

## 13. Training pipeline

```text
generated samples
      ↓
   encoding
      ↓
  Transformer
      ↓
  prediction
      ↓
     loss
      ↓
   metrics
      ↓
  checkpoint
```

- **Loss** — token-level cross-entropy over the target sequence, teacher-forced.
- **Padding masking** — padding positions are excluded from the loss via
  `ignore_index`, so they never contribute a gradient.
- **Validation** — a held-out stream drawn from the same instance but a disjoint,
  independently seeded split.
- **Checkpoint selection** — driven by the configured monitor metric
  (`valid_loss`, minimised). The best and last checkpoints are saved, and every
  checkpoint carries a configuration fingerprint that later stages verify.
- **Seed handling** — one master seed; every other seed is derived from it, so a
  run is reproducible from its config file alone.
- **CPU-oriented operation** — thread counts are resolved explicitly; no CUDA is
  required at any point.

The trainer receives **only public samples**. An assertion checks the type of
every batch it consumes.

> Training commands are deliberately omitted from §27. Reproducing the final
> results does not require training — the checkpoints exist and the pipeline
> replays against them in about a second.

---

## 14. Prediction and evaluation metrics

### `acc_tau`

SALSA's primary compatibility metric: the fraction of predictions satisfying

```text
|b - b_hat| <= tau * q          with tau = 0.1
```

This uses the **plain absolute difference**, matching the released
implementation. A circular-distance variant is computed and retained in the
codebase but is explicitly labelled a **diagnostic** and is never substituted for
the paper's metric.

### Exact integer accuracy

The fraction of samples where the decoded prediction `b_hat` equals `b`
**exactly**. Chance level: `1/q = 0.00398`.

### Token accuracy

The fraction of individual output tokens predicted correctly under teacher
forcing. **Chance level is 0.44622** for this representation — base-81 lsb-first
makes a token accuracy near 45% free — so this metric is reported as secondary
and never read as evidence of learning on its own.

### Exact secret recovery

Whether the recovered candidate equals the true secret in **every** coordinate.
This is the cryptanalytic outcome and is a different claim from any prediction
metric.

### Coordinate accuracy

The fraction of secret coordinates the candidate gets right. Reported alongside
the free all-zeros baseline (0.8333 at n=12, 0.9000 at n=20), because on a sparse
secret it is high by default.

### Hamming distance

The number of coordinates where candidate and true secret differ. Zero means
exact recovery.

### Decode validity

The fraction of model outputs that decode to a valid integer at all. A model can
emit a token sequence that decodes to nothing; that is a different failure from
emitting a wrong number, and the two are counted separately.

### Loss reference points

| reference | value (nats) | meaning |
|---|---:|---|
| uniform | 4.44265 | a model predicting uniformly at random |
| **marginals-only** | **1.86498** | a model that learned only the output marginals — **the real zero point** |
| irreducible floor | 0.83920 | what a perfect attacker still pays for the noise in `b` |

> **High prediction accuracy does not automatically imply exact secret recovery.**
> V1 at n=12 reached `acc_tau` 0.3550 — clearly above its 0.19287 chance level —
> and recovered nothing. Prediction quality is necessary but not sufficient.

---

## 15. Direct secret recovery

For each secret coordinate `i` and each multiplier `K`, construct the probe

```text
a = K * e_i     (reduced modulo q)
```

where `e_i` is the `i`-th standard basis vector. Because

```text
a · s = K * s_i   (mod q)
```

the noiseless answer is `0` when `s_i = 0` and `K` when `s_i = 1`. The model is
driven from `<bos>` with no target supplied, so nothing derived from `b` or `s`
reaches it.

**Hypothesis comparison.** Each decoded prediction is assigned to whichever of the
two mathematically predicted values — `0` or `K mod q` — it is nearer to on the
ring `Z_q`, producing a signed margin. This *anchored* rule needs no ground truth
and has no polarity ambiguity, unlike the released code's approach (§6).

**Coordinate inference and candidate construction.** Reading all `n` coordinates
gives one candidate secret per `K`; the sweep is then aggregated (§16).

**Probe separation.** A probe's entire information budget is the ring distance
between its two hypotheses, `min(K mod q, q − K mod q)` — maximal at `K ≈ q/2`
and vanishing as `K` approaches 0 or `q`.

**K sweep used throughout:** `[1, 31, 63, 94, 125, 126, 157, 188, 220, 250]`.
`K = 1` and `K = 250` have separation 1 and are included as **negative controls
that should fail**. If they were to succeed, the measurement would be suspect
rather than the model impressive.

---

## 16. Recovery-rule hardening

### The problem

The unguarded `best_margin` rule could select a low-separation negative control
such as `K = 1`. At `K = 1` the two hypotheses `b ≈ 0` and `b ≈ 1` are adjacent,
so a model answering a constant `0` sits at distance 0 from one anchor and 1 from
the other — giving `|score| = 1` for every coordinate, **the maximum possible
margin**. A small hypothesis gap produces a deceptively large margin, and the
rule rewards the least informative probes precisely because they are least
informative. At n=20 it selected `K = 1`, whose candidate was not an exact
recovery.

### The fix

**Default rule: `aggregate`** — a separation-weighted vote across the whole
sweep, where each `K`'s contribution is weighted by its ring separation. The
degenerate probes carry weight 0.008 each and are effectively ignored.

**Diagnostic rule: `best_margin`** — retained, now guarded by

```text
min_separation = floor(sigma) + 1
```

which at `sigma = 3` gives `min_separation = 4`, excluding `K ∈ {1, 250}`.

**The threshold is derived from the configured noise scale**, not chosen by hand:
a probe can only discriminate if its two hypotheses are further apart than the
error scale.

**Excluded `K` values remain measured and still vote** with the small weight
their separation earns them — they are excluded from *selection*, not from the
evidence. Both rules are secret-free: separation is a function of `K` and `q`
only.

Measured effect: the guarded rule selects `K = 188` at n=20 instead of `K = 1`.

### Ring variant note

The demonstrated pipeline uses the **circulant** RLWE construction
(`lwe.rlwe_variant: circulant`). The fidelity audit separately established that
the supplied original generator's construction behaves as **negacyclic** — it
negates the strict upper triangle of the circulant matrix. These are different
constructions and are not treated as interchangeable anywhere in this project.
**All demonstrated results in §21 were produced with the circulant
configuration.**

---

## 17. Independent residual verification

Implements the residual test the released source does not provide.

Given a candidate `s_hat` and a set of **fresh** public samples:

```text
r = b - A s_hat   (mod q)
```

Residues are mapped to centered representatives in `[-q/2, q/2)` before any
statistic is computed — without this, a residue of `q − 1` reads as a large
positive error when it is really `−1`.

**Why the test discriminates.** If `s_hat = s`, then `r = e`: tight around zero
at the configured `sigma`. If `s_hat` is wrong by any non-zero delta, then
`r = A·delta + e`, close to uniform on `Z_q` with standard deviation
`sqrt((q² − 1)/12) ≈ 72.5`. The gap between ≈3 and ≈72.5 is the evidence.

**Why fresh samples matter.** Verifying on the training or validation stream
would test whether the candidate explains data the model was fitted to. Fresh
samples from an independently seeded split test whether it explains the
*instance*.

**The verifier receives only:**

```text
(A, b, candidate, q, sigma)
```

The true secret is **not required and not passed**. That signature is the
guarantee.

**Acceptance criteria**, fixed before any statistic was computed:

1. centered residual std ≤ 1.5 · sigma
2. mean absolute centered residual ≤ 1.5 · sigma · √(2/π)
3. candidate std < 0.25 × smallest baseline std
4. fraction within 3 · sigma ≥ 0.99

### Recovery vs verification vs evaluation

| | question it answers | inputs |
|---|---|---|
| **recovery** | what does the model say the secret is? | public probes, model predictions |
| **verification** | does that candidate explain fresh public data? | `(A, b, candidate, q, sigma)` |
| **evaluation** | was it right? | the true secret — loaded last |

Verification is not evaluation. It confirms a candidate against the mathematics
without ever knowing the answer, which is exactly what a real adversary could do.

---

## 18. End-to-end pipeline

```text
public LWE/RLWE data
        ↓
     encoding
        ↓
      NACT-F
        ↓
     predict b
        ↓
direct secret recovery
        ↓
   candidate secret
        ↓
residual verification
        ↓
  final evaluation
```

### Public API

```python
from salsa import run_salsa2_pipeline

result = run_salsa2_pipeline(
    "configs/pipeline_n12.yaml"
)
print(result.exact_recovery, result.verification_passed)
```

### CLI

```bash
python scripts/run_pipeline.py configs/pipeline_n12.yaml
python scripts/run_pipeline.py configs/pipeline_n20.yaml
```

Replay currently completes in **approximately one second** per configuration with
the existing checkpoints and configuration. **This is replay time, not training
time** — the pipeline loads a completed checkpoint and performs inference,
recovery and verification only.

Results are written to the directory named by `pipeline.output_dir` (default
`results/final_pipeline/`) as `<label>_result.json`.

---

## 19. Ground-truth isolation

The pipeline enforces this ordering:

```text
Prediction
    ↓
Recovery
    ↓
Candidate
    ↓
Independent Verification
    ↓
Ground-truth evaluation
```

- **Recovery does not load the true secret.** `DirectRecovery` is constructed
  with `(model, codec, method, batch_size)` and invoked as `recover(k_values, …)`
  — there is **no parameter through which a secret could be passed**. A test
  additionally scans the recovery package's source text for any reference to
  ground truth.
- **Verification does not load the true secret.** Its signature is
  `(A, b, candidate, q, sigma)`.
- **Evaluation loads the true secret only after** both a candidate and a
  verification verdict already exist.

The ordering is enforced by construction, not by convention. And:

```python
result = run_salsa2_pipeline("configs/pipeline_n12.yaml", evaluate=False)
```

runs the complete attack pipeline **without ground truth at any point** — the
real adversarial setting. This is an important reproducibility and anti-leakage
feature, and it is covered by a test asserting the secret field stays empty.

---

## 20. Experimental configuration

| Parameter | Value |
|---|---:|
| `q` | 251 |
| `sigma` | 3 |
| `h` | 2 |
| `n` | 12, 20 |
| structure | RLWE, circulant |
| representation | R (base 81, lsb-first, no separator, fixed width) |
| vocabulary | 85 |
| `tau` | 0.1 |
| Training samples | 100,032 each |
| Batch size | 64 |
| Optimizer | AdamW |
| Warmup steps | 200 |
| Validation sequences | 2,048 |
| Monitor metric | `valid_loss` (min) |
| Encoder / decoder loops | `T_e = 2`, `T_d = 2` |
| Final model | NACT-F |
| Parameters | 4,238,208 |
| Seeds | 0 (n=12 also validated at 42 and 123) |

---

## 21. Experimental results

### Main results

| Metric | n=12 | n=20 |
|---|---:|---:|
| `acc_tau` | 0.9839 | 0.9863 |
| Exact integer accuracy | 0.0913 | 0.1035 |
| **Exact secret recovery** | **YES** | **YES** |
| Coordinate accuracy | 1.0000 | 1.0000 |
| Hamming distance | 0 | 0 |
| Successful K | 8/10 | 8/10 |
| Probe decode validity | 1.0000 | 1.0000 |
| Independent verification | **PASS** | **PASS** |
| Parameters | 4,238,208 | 4,238,208 |

The two K values that do not succeed are exactly `K = 1` and `K = 250`, the
deliberate negative controls (§15).

### Supporting prediction figures

| | n=12 | n=20 |
|---|---:|---:|
| validation loss | 0.9891 | 0.9764 |
| token accuracy | 0.6934 | 0.6995 |
| greedy token accuracy | 0.6771 | 0.6851 |
| decode failure rate | 0.0000 | 0.0000 |
| mean \|b − b_hat\| | 7.15 | 6.51 |
| training samples | 100,032 | 100,032 |

Both losses sit well below the 1.86498 marginals-only baseline and closer to the
0.83920 irreducible floor.

### Verification residuals

| | n=12 | n=20 |
|---|---:|---:|
| recovered candidate std | **2.998** | **2.941** |
| all-zeros baseline std | 71.803 | 72.477 |
| all-ones baseline std | 72.676 | 72.756 |
| configured `sigma` | 3.0 | 3.0 |

### Reproduction across secrets

At n=12, three checkpoints trained under different seeds — and therefore three
**different secrets** — were attacked: **3/3 exact recovery, 3/3 verification
PASS**. Only one secret has been tested at n=20.

---

## 22. V1 vs NACT-F

Both at n=12, h=2, 100,032 samples, matched protocol and seed:

| | V1 Salsa2-GatedUT | NACT-F |
|---|---:|---:|
| Parameters | 4,131,200 | 4,238,208 |
| Validation loss | 1.7413 | **0.9891** |
| `acc_tau` | 0.3550 | **0.9839** |
| Exact integer accuracy | 0.0078 | **0.0913** |
| Probe decode validity | 0.4167 | **1.0000** |
| Successful K | 0/10 | **8/10** |
| **Exact secret recovery** | **NO** | **YES** |

V1's loss sits essentially at the marginals-only baseline of 1.86498 — at this
budget it has largely not learned secret-related structure. On the probes, 58% of
its outputs were undecodable and it answered a single constant value on 8 of 10
multipliers.

> **Conclusion, stated carefully.** The parameter counts are similar — a 2.6%
> difference — while recovery behaviour differs substantially. The project
> evidence therefore points to **representation as the principal observed source
> of improvement**. This is not a proof that representation is the only cause;
> the two models also differ in effective sequence length and in the presence of
> absolute coordinate identity, which is precisely what the representation change
> consists of.

---

## 23. NACT ablation findings

The one-token representation was isolated directly by constructing **NACT-F**:
full NACT with all six additional numerical, zero-aware and sparse-attention
components removed, keeping only the coordinate-level representation.

At matched depth, seed and sample budget:

| metric | full NACT | NACT-F | difference | 3σ seed band | resolvable at one seed? |
|---|---:|---:|---:|---:|:---:|
| validation loss | 0.9388 | 0.9891 | +0.0503 | 0.0696 | no |
| `acc_tau` | 0.9839 | 0.9839 | +0.0000 | 0.0064 | no |
| exact integer accuracy | 0.1011 | 0.0913 | −0.0098 | 0.0275 | no |

Every difference falls inside full NACT's own measured seed-to-seed noise band.
**NACT-F retained approximately all of the observed `acc_tau` gap relative to V1
in the tested seed** — the recorded retention figure is 1.000 for `acc_tau` and
0.937 for loss.

Summary of the ablation:

- **full NACT → strong performance**
- **NACT-F → strong performance**
- **the additional numerical and sparse-aware features were not necessary for the
  demonstrated recovery**

One difference is recorded rather than smoothed over: NACT-F's sparse-input lift
at the extreme end is roughly half full NACT's (+0.0500 versus +0.1167 on
probes). The margin narrowed; the recovery outcome did not change.

> **Remaining C/D/E ablations were designed but not executed.** No individual
> component has been attributed; ablation F removed six at once.

---

## 24. Reproducibility and replay validation

Both established results were replayed through the integrated pipeline and
checked against the values earlier phases recorded. **12/12 replay checks
matched.**

| check | n=12 expected | n=12 replayed | n=20 expected | n=20 replayed |
|---|---:|---:|---:|---:|
| exact recovery | True | **True** | True | **True** |
| coordinate accuracy | 1.0 | **1.0** | 1.0 | **1.0** |
| Hamming distance | 0 | **0** | 0 | **0** |
| successful K | 8 | **8** | 8 | **8** |
| verification passed | True | **True** | True | **True** |
| parameter count | 4,238,208 | **4,238,208** | 4,238,208 | **4,238,208** |

Reproducibility mechanisms in the codebase:

- every experiment runs from a **config file plus a fixed seed**;
- configs compose via `extends:` with strict validation — an unknown key is an
  error, not a silent ignore;
- each config has a **SHA-256 fingerprint** recorded with the run;
- checkpoints carry that fingerprint, and the pipeline **refuses to load** a
  checkpoint whose recorded identity disagrees with the config;
- checkpoints are located from run metadata, never by an assumed filename.

---

## 25. Project structure

```text
SALSA 2.0/
├── salsa/                       library
│   ├── data/                    LWE/RLWE generation, secrets, encoding
│   │   ├── lwe.py               problem instances, public/private separation
│   │   ├── rlwe.py              circulant and negacyclic constructions
│   │   ├── secrets.py           exact-Hamming-weight binary secrets
│   │   └── encoding.py          LatticeCodec, vocabulary, integer encoder
│   ├── models/                  architectures and parameter accounting
│   │   ├── transformer.py       V1 Salsa2-GatedUT, shared blocks
│   │   ├── nact.py              NACT / NACT-F front end and model
│   │   ├── attention.py         multi-head attention, RoPE
│   │   ├── embeddings.py        token and rotary embeddings
│   │   └── parameter_count.py   three-way budget verification
│   ├── training/                trainer, losses, metrics, seeding
│   ├── recovery/                direct.py — direct secret recovery
│   ├── verification/            residual.py — independent residual test
│   ├── evaluation/              reserved
│   ├── utils/                   config, device, logging
│   └── pipeline.py              run_salsa2_pipeline — the entry point
├── configs/                     31 YAML experiment definitions
├── scripts/                     26 entry points and analysis tools
├── tests/                       17 test modules
├── results/                     recorded artifacts for every phase
│   ├── final_pipeline/          final report, results, status, replay
│   ├── original_fidelity_audit/ comparison against the released SALSA source
│   ├── recovery_generalization/ sparse-input study
│   ├── nact_ablation_F/         the ablation that produced NACT-F
│   ├── n20_nact_f_pilot/        n=20 training
│   └── n20_nact_f_recovery/     n=20 recovery
├── pyproject.toml
├── requirements.txt
└── README.md
```

| directory | purpose |
|---|---|
| `salsa/` | the library — everything importable |
| `configs/` | every experiment is defined by a config; results are reproducible from config + seed |
| `scripts/` | CLI entry points (`run_pipeline.py`, `train.py`) and per-phase analysis tools |
| `tests/` | 687 tests covering mathematics, budgets, secret isolation and the pipeline |
| `results/` | recorded artifacts. Nothing here is regenerated by documentation tasks |

Key documents: `results/final_pipeline/SALSA2_Final_Report.md` (full technical
report), `results/final_pipeline/project_status.md` (completion state),
`results/original_fidelity_audit/` (comparison against the released source).

---

## 26. Installation

Verified environment for the recorded results:

| component | version |
|---|---|
| Python | 3.10.11 (`pyproject.toml` requires ≥ 3.9) |
| PyTorch | 2.13.0+cpu |
| NumPy | 2.2.6 |

Declared dependencies (`requirements.txt`): `numpy>=1.23,<3`, `PyYAML>=6.0`,
`torch>=2.0`, `psutil>=5.9`, plus `pytest>=7.4` for tests and `matplotlib>=3.7`
for plots only.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# CPU-only PyTorch (recommended; no CUDA needed anywhere in this project)
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

python -m pip install -r requirements.txt
```

`matplotlib` is required only for regenerating plots; data generation, training,
evaluation, recovery and verification do not need it.

---

## 27. Usage

All commands below are lightweight. **None of them trains a model.**

### Pipeline replay — the main entry point

```bash
python scripts/run_pipeline.py configs/pipeline_n12.yaml
python scripts/run_pipeline.py configs/pipeline_n20.yaml
```

Approximately one second each. Prints the three stages separately and writes a
full result object to `results/final_pipeline/`.

### Attack setting — no ground truth consulted

```bash
python scripts/run_pipeline.py configs/pipeline_n12.yaml --no-evaluate
```

### From Python

```python
from salsa import run_salsa2_pipeline

result = run_salsa2_pipeline("configs/pipeline_n12.yaml")
print(result.summary_line())
```

### Inspect a configuration

```bash
python scripts/show_config.py configs/pipeline_n12.yaml
```

### Regenerate the final documentation artifacts

```bash
python scripts/final_documentation.py     # final_results.csv, final_summary.json, project_status.md
python scripts/pipeline_report.py         # replays both dimensions and writes pipeline_report.*
```

### Tests

```bash
pytest -q
```

> **Historical / research-only.** `scripts/train.py` reproduces a training run
> from a config (roughly 20 minutes per model on a CPU). It is **not required** to
> reproduce the final results, since the checkpoints already exist and the
> pipeline replays against them.

---

## 28. Tests

```text
687 passed, 2 skipped
```

This status was unchanged by the final integration and documentation work.
**No validated experiment was modified to make tests pass**, and no test was
weakened. The two skips are dimension guards, not failures.

Coverage, in the order the project built it:

- **mathematics before neural code** — LWE/RLWE generation, exact-weight secrets,
  error distributions, the codec's encode/decode round trip verified against the
  original SALSA encoder;
- **parameter budgets** — every model's count checked three independent ways
  (actual `numel`, component breakdown, analytical formula) with nothing
  unclassified;
- **secret isolation** — assertions that the recovery package's source contains
  no reference to ground truth, and that no API parameter could carry one;
- **the integrated pipeline** — construction, checkpoint identity gating,
  degenerate-K handling, separation-weighted aggregation, the verifier interface,
  config parsing, result serialisation, and stage ordering;
- **regression behaviour** for every earlier phase.

---

## 29. Final findings

1. A **4,238,208-parameter** CPU-trained Transformer completes the full SALSA
   direct-recovery pipeline — learn, recover, verify — at **n=12 and n=20** with
   h=2, on 100,032 training samples per dimension.
2. The **4,131,200-parameter** V1 baseline, a faithful compact reduction of the
   original architecture, **recovers nothing** at n=12 under the identical
   protocol.
3. The parameter counts differ by 2.6%, so the project evidence points to the
   **coordinate-level representation**, not capacity, as the principal observed
   source of improvement.
4. **Exact secret recovery** was achieved at both demonstrated dimensions, with
   coordinate accuracy 1.0000 and Hamming distance 0, and 8 of 10 probe
   multipliers succeeding individually — the two failures being the deliberate
   negative controls.
5. **Independent residual verification passed** at both dimensions: residual
   standard deviation 2.998 and 2.941 against a configured `sigma` of 3.0, while
   every incorrect candidate sits near 72.5.
6. Recovery **reproduced across three different secrets** at n=12 (3/3 exact, 3/3
   verified) using three independently seeded checkpoints.
7. **Replay is reproducible**: 12/12 checks matched through the integrated
   pipeline, in about one second per configuration.
8. The test suite stands at **687 passed, 2 skipped**, with secret isolation
   enforced structurally rather than by convention.

---

## 30. Limitations

**These results do not constitute a practical break of deployed LWE/RLWE
cryptography.**

- **Only n=12 and n=20 were experimentally demonstrated.**
- **n=30 was NOT trained or tested.**
- **n=50 was NOT trained or tested.**
- **n=128 was NOT trained or tested.** The architecture accepts it and forward
  cost has been benchmarked, but no model has been fitted at that dimension.
- **Only limited multi-seed validation exists** — three secrets at n=12, one at
  n=20. Seed-to-seed variance at n=20 is unmeasured.
- **n=12 and n=20 have small diagnostic secret spaces** — C(12,2) = 66 and
  C(20,2) = 190.
- **Ablations C, D and E were not run.** Ablation F removed six components at
  once, so no individual feature has been attributed.
- **`min_separation` validation is currently at `sigma = 3` only.**
- **Distinguisher recovery was not implemented** — only the direct recovery mode.
- **Only `h = 2` has been demonstrated end to end.**
- **No comparison to the original at matched conditions.** Model size, sample
  budget and sample-reuse policy all differ, as the fidelity audit documents.
- **Two dimensions is not a scaling law.**

---

## 31. Future work

None of the following has been performed.

1. **Dimension scaling** to n=30, n=50 and n=128. The parameter count does not
   vary with `n`, so the binding constraint is expected to be the sample
   requirement rather than the architecture.
2. **Multi-seed validation**, so recovery is reported as a rate with an
   uncertainty estimate rather than a single outcome.
3. **Remaining NACT ablations** (variants C, D and E), to attribute the residual
   sparse-input margin that NACT-F narrowed without changing the outcome.
4. **Different `h` values**, since only `h = 2` has been demonstrated.
5. **Different `q` and `sigma` configurations**, including validating the
   `min_separation` guard away from `sigma = 3`.
6. **Generalization to unseen secrets** — the current setting trains one model per
   secret, as SALSA does; whether a model can generalize across secrets is
   untested.
7. **Distinguisher-based experiments** — the second SALSA recovery mode.
8. **Computational scaling analysis** — how the sample requirement grows with `n`
   and `h`, and whether the one-token advantage holds as sequences lengthen.

---

## 32. Scientific interpretation

> **The dominant observed improvement is associated with changing from
> digit-centric to coordinate-centric representation while explicitly exposing
> coordinate identity.**

The supporting evidence:

- **Similar parameter scale.** V1 has 4,131,200 parameters and NACT-F has
  4,238,208 — a 2.6% difference. Capacity does not distinguish them.
- **V1 failed recovery.** 0 of 10 multipliers, 58% of probe outputs undecodable,
  a single constant answer on 8 of 10 multipliers.
- **NACT-F succeeded.** Exact recovery at both demonstrated dimensions, probe
  decode validity 1.0000, verification PASS.
- **NACT-F retained nearly all of the observed primary-metric improvement.** The
  recorded `acc_tau` gap retention relative to V1 is 1.000.
- **The full numerical and sparse-aware feature set was not required** for the
  demonstrated recovery — removing all six components changed nothing resolvable
  at one seed.

A supporting diagnostic explains *why* the representation matters. Feeding the
frozen V1 model inputs of decreasing density and scoring against the best
input-blind constant predictor gave a monotone collapse, with lift crossing zero
between 8 and 6 non-zero coordinates and reaching −0.4417 on the recovery probes.
The probes sat at the far end of that same curve rather than on a separate cliff,
which located the failure in how sparse inputs are represented rather than in
model capacity or training budget alone.

> **Do not overgeneralize beyond the tested setting.** This interpretation is
> supported at n=12 and n=20 with h=2, q=251, sigma=3, at one training budget,
> with limited seed replication. It is not established at other dimensions,
> weights, or noise scales.

---

## 33. References

**Primary reference**, as recorded in this project's fidelity audit:

> Wenger, Chen, Charton, Lauter. *SALSA: Attacking Lattice Cryptography with
> Transformers.* NeurIPS 2022.

The audit in `results/original_fidelity_audit/` was conducted against a supplied
source checkout of the original implementation, which is **not redistributed in
this repository**. That audit records which files in the supplied checkout were
unmodified and which had been altered.

Architectural influences named in the codebase:

> Csordás et al. *The Neural Data Router* — the copy-gate mechanism used in the
> shared/looped layers.

> **No DOIs, URLs or bibliographic identifiers are given**, because none are
> recorded in the repository and this README does not invent them. Readers should
> resolve the citations above from the venue.

---

## 34. License / project status

### License

`pyproject.toml` declares `license = { text = "MIT" }`. **No `LICENSE` file is
present in the repository.** This discrepancy is documented rather than resolved
here — adding a license file is the repository owner's decision.

### Project status

```text
Core implementation:        COMPLETE
Demonstrated validation:    n=12 and n=20
Larger-scale validation:    FUTURE WORK
```

| area | state |
|---|---|
| Data generation, encoding, models, training | complete and tested |
| Direct secret recovery | complete, secret-free, hardened |
| Independent residual verification | complete |
| End-to-end pipeline | complete, replay validated 12/12 |
| Test suite | 687 passed, 2 skipped |
| n=12, h=2 | demonstrated: learning, exact recovery, verification |
| n=20, h=2 | demonstrated: learning, exact recovery, verification |
| n=30, n=50, n=128 | **not attempted** |
| Multi-seed validation | partial (3 secrets at n=12, 1 at n=20) |
| Ablations C, D, E | designed, **not executed** |
| Distinguisher recovery | **not implemented** |
