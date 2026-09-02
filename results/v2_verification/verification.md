# Phase 20 — independent mathematical verification of the phase-19 candidate

**Read-only mathematics.** No model was loaded, no recovery was run, nothing was
trained and no checkpoint was touched.

## What was verified, and how the test is kept honest

Candidate, taken verbatim from phase 19 and never altered here: `[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]` — length 12, binary, Hamming weight **2**.

Produced by `results/equal_depth_ablation/v2_te2/nact_n12_h2_te2/seed_123/checkpoints/best.pt`.

The verifier is a single function taking **only** `(A, b, candidate, q)`. It has no
parameter through which a secret, a training label or a model output could reach
it. The data generator does use the true secret to build `b` — that is what makes
these real LWE samples — but only `A` and `b` leave that scope.

**Why the test discriminates.** If the candidate equals the secret then
`b - Ac = e (mod q)` and the centered residual *is* the error, with standard
deviation ≈ sigma = 3. If the candidate is wrong by any
nonzero delta the residual is `A·delta + e (mod q)`, close to uniform on Z_q with
standard deviation ≈ **72.5**. That gap carries the evidence.

Residues are mapped to centered representatives in `[-q/2, q/2)` before any
statistic is computed; without that a residue of `q-1` would read as a large
positive error when it is really `-1`.

## Fresh samples

| set | data seed | samples |
|---|---:|---:|
| `phase20_verify_a` | 3689010707 | 2,048 |
| `phase20_verify_b` | 1763909695 | 2,048 |
| *training stream, for contrast* | 720569232 | — |

Both verification streams are derived from split labels the training run never
used, so the samples are fresh and independent of training and of each other.
The same samples are used for every candidate within a set, so the comparison is paired.

## Acceptance criteria — fixed before the numbers

- **c1_std_close_to_sigma** — centered residual std <= 1.5 * sigma
- **c2_mean_abs_small** — mean |centered residual| <= 1.5 * sigma * sqrt(2/pi)
- **c3_beats_baselines** — candidate std < 0.25 * (smallest baseline std)
- **c4_within_three_sigma** — fraction of |centered residual| <= 3 sigma is >= 0.99
- **decision** — A VERIFIED when all four hold on BOTH fresh sample sets; B PARTIALLY VERIFIED when c3 holds but some other criterion fails; C NOT VERIFIED when c3 fails on either set.

## Residual statistics

### `phase20_verify_a`

| candidate | std | mean abs | median abs | p95 | p99 | max | ≤1σ | ≤2σ | ≤3σ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **recovered candidate** | **2.988** | 2.358 | 2.0 | 6.0 | 7.0 | 10 | 0.7651 | 0.9707 | 0.9995 |
| all zeros | 72.214 | 62.721 | 61.0 | 119.0 | 124.0 | 125 | 0.0249 | 0.0503 | 0.0713 |
| all ones | 72.805 | 63.034 | 62.0 | 120.0 | 124.5 | 125 | 0.0278 | 0.0425 | 0.0693 |
| random weight 2 | 73.366 | 63.896 | 65.0 | 119.0 | 124.0 | 125 | 0.0303 | 0.0474 | 0.0684 |

### `phase20_verify_b`

| candidate | std | mean abs | median abs | p95 | p99 | max | ≤1σ | ≤2σ | ≤3σ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **recovered candidate** | **2.966** | 2.355 | 2.0 | 6.0 | 7.0 | 11 | 0.7617 | 0.9688 | 0.9976 |
| all zeros | 72.190 | 62.470 | 62.0 | 120.0 | 124.0 | 125 | 0.0264 | 0.0542 | 0.0786 |
| all ones | 71.221 | 61.595 | 61.0 | 119.0 | 124.0 | 125 | 0.0269 | 0.0483 | 0.0767 |
| random weight 2 | 71.511 | 61.846 | 62.0 | 118.0 | 124.0 | 125 | 0.0229 | 0.0425 | 0.0605 |

## Criteria outcome

| set | c1 std ≈ σ | c2 mean abs | c3 beats baselines | c4 within 3σ |
|---|:---:|:---:|:---:|:---:|
| `phase20_verify_a` | PASS | PASS | PASS | PASS |
| `phase20_verify_b` | PASS | PASS | PASS | PASS |

## Error-distribution check

| set | test | statistic | dof | p | rejects at 0.05 |
|---|---|---:|---:|---:|:---:|
| `phase20_verify_a` | chi-square vs discrete Gaussian σ=3 | 9.273 | 17 | 0.9313 | no |
| `phase20_verify_b` | chi-square vs discrete Gaussian σ=3 | 30.885 | 17 | 0.0206 | yes |

Assumptions stated plainly: residuals treated as i.i.d.; bins pooled from the
tails until every expected count reaches 5; degrees of freedom `bins - 1` because
sigma is the **configured** value, not fitted from the data.

*Weak instrument on 2048 samples; failing to reject is not proof of the distribution. The discriminating evidence is the baseline comparison, not this test.*

**One of the two sets rejects at 0.05 and this is reported rather than buried.**
Set A gives p = 0.93, set B gives p = 0.02. Three things are worth stating and
none of them is a reinterpretation after the fact:

1. The chi-square was **deliberately excluded from the acceptance criteria**, which
   were fixed before any statistic was computed. The classification does not depend
   on it and was not adjusted because of it.
2. Running two tests at alpha = 0.05 gives roughly a 10% chance of at least one
   rejection even when the candidate is exactly correct. A single rejection out of
   two is ordinary sampling variation, not a signal.
3. The quantity that actually discriminates is unaffected: the candidate's residual
   standard deviation is 2.99 and 2.97 against a configured sigma of 3.0, while every
   incorrect candidate sits at roughly 72 on both sets. A wrong candidate cannot
   produce a residual spread of 3 by chance.

If the decision had rested on the goodness-of-fit test, this result would be
reported as **B, partially verified**. It does not, and the pre-registered criteria
pass 8 out of 8.

## Original SALSA comparison

The supplied original checkout contains no LWE residual verifier; the only 'residual' matches are neural residual connections in layers/. evaluator.py implements no verification. This confirms the phase-11 finding, so there was nothing to reuse and the direct residual test was used.

## Verification decision

### **A. VERIFIED**

Reached from residual evidence alone. The true secret was not loaded until after
this decision was made and influenced no threshold, no sample and no candidate.

## Ground-truth comparison — evaluation only

*EVALUATION ONLY. Loaded after the verification decision was made; it influenced no threshold, no sample, and no candidate.*

| field | value |
|---|---|
| candidate | `[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]` |
| true secret | `[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]` |
| candidate == true secret | **True** |
| Hamming distance | 0 |
| coordinate accuracy | 1.0000 |
| candidate Hamming weight | 2 |
| true Hamming weight | 2 |

## Statement of what this shows

The recovered candidate independently satisfies the LWE/RLWE residual consistency
check on two fresh sample sets and substantially outperforms incorrect candidate
baselines.

**This is an n=12, h=2 diagnostic verification and nothing more.** It is not a
cryptographic result, not a claim about n=30 or n=128, not a general attack
success, and not a security break of practical LWE. The instance has C(12,2) = 66
possible secrets.

## Figures

- `01_residual_distribution.png`
- `02_candidate_vs_baselines.png`
- `03_absolute_residuals.png`

## Artifacts

- `verification.md` (this file)
- `verification.json`
- `verification.csv`
