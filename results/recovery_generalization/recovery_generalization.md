# Phase 12 - Recovery-distribution generalization analysis

**Read-only.** No training was performed. The model, the data generator and the
recovery implementation are unmodified; every model parameter was frozen
(`requires_grad=False`, 0 trainable).

Every statement is tagged **MEASURED** (a number this run produced),
**INFERENCE** (a reading of those numbers) or **HYPOTHESIS** (a proposal this
experiment does not test). Causation is not claimed from this experiment.

## Setup

- Checkpoint `results/control_a_n12_h2/control/checkpoints/best.pt`, epoch 19, 100,032 samples, 4,131,200 parameters, fingerprint `98afa3f729f8607e`. All architectural and cryptographic values were verified against the configuration before loading.
- Instance n=12, h=2, q=251, sigma=3.0, rlwe/circulant, representation R, vocabulary 85, tau=0.1.
- 2,048 evaluation vectors per sparsity level. Probes are exhaustive: 10 K values x 12 coordinates = 120.
- Seeds: config 0, analysis 1325168527. The instrumented decoder is asserted identical to `salsa.training.metrics.greedy_decode` before any measurement is taken.

### How the measurement is kept clean

| stage | what it sees |
|---|---|
| 1. model response | `a` only. Decode validity, entropy, uniqueness and variance need no target at all. |
| 2. evaluation | `b = a.s + e mod q`, built by the data layer *after* the model answered, with fresh errors. Never shown to the model, never used to select an input, never fed back into stage 1. |

For `K*e_i` inputs the stage-2 numbers measure **prediction fidelity on probe
inputs**. They are not secret recovery and are not reported as such. Secret
recovery was measured in phase 10; the answer there was NO and that stands.

## 1. Why raw acc_tau is not comparable across sparsity levels

**MEASURED.** Making `a` sparser also moves the *target* distribution. With a weight-2
secret over 12 coordinates a single-nonzero `a` misses the support 10 times out of 12,
so `a.s = 0` and `b` is just the error, tight around zero:

| nnz | target entropy (nats) | fraction of b within tau of 0 | acc_tau of the best input-blind constant |
|---:|---:|---:|---:|
| 12 | 5.465 | 0.1982 | 0.2236 (constant 47) |
| 10 | 5.451 | 0.2305 | 0.2354 (constant 225) |
| 8 | 5.427 | 0.2559 | 0.2207 (constant 225) |
| 6 | 5.206 | 0.3779 | 0.2817 (constant 25) |
| 4 | 4.749 | 0.5391 | 0.3530 (constant 25) |
| 2 | 3.913 | 0.7617 | 0.4546 (constant 25) |
| 1 | 3.298 | 0.8711 | 0.5083 (constant 23) |

**INFERENCE.** A constant predictor gets *better* as the input gets sparser, so a
falling acc_tau on its own understates the collapse. The honest quantity is the
**lift**: acc_tau minus what the best input-blind constant scores on the very same
targets. Lift <= 0 means the model's output carries no usable information about
its input.

## 2. The table

| input_type | nnz | samples | decode_validity | acc_tau | best-constant | **lift** | exact_accuracy | unique_outputs | prediction_entropy | mean_error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A. RLWE rows from the generator | 12 | 2048 | 1.0000 | 0.3623 | 0.2217 | **+0.1406** | 0.0122 | 9 | 1.173 | 59.3 |
| B. sparse random, nnz=12 | 12 | 2048 | 1.0000 | 0.3545 | 0.2236 | **+0.1309** | 0.0083 | 11 | 1.195 | 61.4 |
| B. sparse random, nnz=10 | 10 | 2048 | 0.9985 | 0.3345 | 0.2354 | **+0.0991** | 0.0083 | 12 | 1.280 | 57.2 |
| B. sparse random, nnz=8 | 8 | 2048 | 0.9980 | 0.2925 | 0.2207 | **+0.0718** | 0.0054 | 12 | 1.441 | 57.2 |
| B. sparse random, nnz=6 | 6 | 2048 | 0.9878 | 0.2290 | 0.2817 | **-0.0527** | 0.0059 | 11 | 1.698 | 67.3 |
| B. sparse random, nnz=4 | 4 | 2048 | 0.9512 | 0.1743 | 0.3530 | **-0.1787** | 0.0044 | 12 | 1.902 | 77.2 |
| B. sparse random, nnz=2 | 2 | 2048 | 0.7793 | 0.0815 | 0.4546 | **-0.3730** | 0.0020 | 11 | 1.155 | 95.4 |
| B. sparse random, nnz=1 | 1 | 2048 | 0.5303 | 0.0283 | 0.5083 | **-0.4800** | 0.0010 | 6 | 0.581 | 104.0 |
| C. probes K*e_i (all K, all i) | 1 | 120 | 0.4167 | 0.0417 | 0.4833 | **-0.4417** | 0.0000 | 3 | 0.443 | 104.5 |

Chance from the representation alone: acc_tau 0.19287, exact 0.00398. `prediction_entropy` is the empirical
entropy in nats of the decoded output distribution; uniform over Z_q would be 5.525. `mean_error` is mean |b - b_hat| over decodable predictions.

**MEASURED.** Regime A (real generator rows, acc_tau 0.3623) and regime B at
nnz=12 (synthetic dense vectors, 0.3545) agree to 0.0078. The synthetic input generator is
therefore not a confound: the only thing varying down the table is sparsity.

## 3. Hypothesis test: is the degradation monotone?

Spearman rho against nnz over the seven sparsity levels, with an **exact**
permutation p-value -- all 5,040 orderings enumerated, no normal approximation.

| metric | rho | exact p | monotone as a sequence? |
|---|---:|---:|:---:|
| acc_tau | +1.000 | 0.0004 | yes |
| exact_accuracy | +0.955 | 0.0040 | no |
| decode_validity | +1.000 | 0.0004 | yes |
| output_entropy_nats | +0.286 | 0.5560 | no |
| mean_abs_error_valid | -0.857 | 0.0238 | no |
| mean_token_predictive_entropy_nats | -0.321 | 0.4976 | no |
| acc_tau_minus_best_constant | +1.000 | 0.0004 | yes |

**MEASURED.** `acc_tau`, `decode_validity` and the lift over the best constant are
all perfectly monotone in sparsity (rho +1.000, exact p 0.0004). The lift crosses zero
between nnz=8 and nnz=6: at nnz=8 the model still beats an input-blind
constant, at nnz=6 and below it does not, and by nnz=1 it is 0.4800 *worse*
than ignoring its input entirely.

**MEASURED.** Output entropy is **not** monotone (rho +0.286, exact p 0.5560). It rises from 1.195 nats at nnz=12
to 1.902 at nnz=4, then falls to 0.581 at nnz=1.

**INFERENCE.** Two different failures happen in sequence, not one. Moderate
sparsity makes the model *diffuse*: it spreads its answers over more values while
getting them less right. Extreme sparsity makes it *collapse*: it stops producing
varied answers at all, and increasingly emits token sequences that do not decode
to a number.

## 4. Do the probes sit on the same curve?

Regime C has the same nnz as regime B's last level, so the two are directly
comparable. A gap would mean something beyond sparsity is acting.

| quantity | B, nnz=1 (random position and value) | C, K*e_i probes |
|---|---:|---:|
| acc_tau | 0.0283 | 0.0417 |
| decode_validity | 0.5303 | 0.4167 |
| unique outputs | 6 | 3 |
| output entropy (nats) | 0.581 | 0.443 |

**MEASURED.** The probes land at or just past the endpoint of the sparsity curve:
acc_tau differs by 0.0133, and the probes are slightly *more*
collapsed (fewer unique outputs, lower validity, lower entropy).

**INFERENCE.** Regime C is a continuation of regime B, not a separate cliff.
Sparsity alone reproduces almost all of the probe behaviour, which is what the
distribution-shift account predicts. The small residual is consistent with the
probes being more extreme still -- one fixed value on a fixed diagonal, rather
than a random position and magnitude -- but this experiment cannot separate that
from sampling noise at this size.

## 5. Probe response in detail

| K | K mod q | separation | decode_validity | unique outputs across i | variance across i | modal value | modal fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | 0.00 | 0 | - | - | - |
| 31 | 31 | 31 | 1.00 | 1 | 0.0 | 81 | 1.00 |
| 63 | 63 | 63 | 1.00 | 1 | 0.0 | 81 | 1.00 |
| 94 | 94 | 94 | 0.08 | 1 | 0.0 | 162 | 0.08 |
| 125 | 125 | 125 | 0.00 | 0 | - | - | - |
| 126 | 126 | 125 | 0.00 | 0 | - | - | - |
| 157 | 157 | 94 | 0.08 | 1 | 0.0 | 162 | 0.08 |
| 188 | 188 | 63 | 1.00 | 2 | 3645.0 | 81 | 0.83 |
| 220 | 220 | 31 | 1.00 | 2 | 3645.0 | 81 | 0.83 |
| 250 | 250 | 1 | 0.00 | 0 | - | - | - |

**MEASURED.** Averaged over the 10 K values, the number of distinct decoded
answers across all 12 coordinates is 0.8 -- fewer than one
distinct answer per K. Four K values produce nothing decodable at all, including
K=125 and K=126, which carry the *largest possible* separation and should be the
most informative probes there are.

**MEASURED.** Every decodable probe answer in this run was a multiple of 81: only
81, 162 and 243 ever appeared. In base-81 lsb-first that means the low output
digit was always 0.

**INFERENCE.** The probe response is not a noisy version of the right answer. The
model emits a near-constant token pattern whose low digit mirrors the
overwhelmingly-zero input digits. That is what an extrapolation failure looks
like, not what a precision failure looks like. And that the best-separation probes
are the *least* decodable is the opposite of what an information-limited readout
would predict.

## 6. A residual signal in the probe response

**MEASURED, stage 1 (no target involved).** Across all K values the model gives a
non-modal answer at exactly 2 of the 12 coordinates: [2, 10]. Every other
coordinate always receives the modal answer.

**MEASURED, stage 2 (the secret enters here, strictly after the fact).** The true
support is [2, 10]. These are the same set. At K=[188, 220] the released
code's `bin2` mode rule, inverted, equals the true secret on all 12 coordinates.

**This is not a recovery result and must not be read as one.** Four things stand
between it and one:

1. The pattern was found by inspecting these results. The quoted coincidence
   probability 0.0152 = 1/C(12,2) assumes a uniform random
   support and does **not** correct for that search. It is a lead, not a
   significance test.
2. The released code resolves the inverted-versus-uninverted polarity by comparing
   to the true secret. That step cannot be part of an attack -- which is precisely
   why phase 10 replaced it with the secret-free anchored rule.
3. It holds for 2 of 10 K values. Four give nothing decodable, two give a flat
   constant, and two single out only half the support -- and inverting *those*
   gives the wrong answer.
4. One secret, one checkpoint, one dimension. n=1.

**INFERENCE.** The phase-10 collapse was not total. Some coordinate-selective
information about the secret survives into the probe response; it is simply not
expressed in the *numeric value* the anchored rule reads, because the model
answers 81/162/243 rather than anything near 0 or near K.

**HYPOTHESIS (untested here).** A readout keyed to *which coordinates differ from
the modal answer*, with polarity fixed by a secret-free tie-break such as
"choose the candidate whose Hamming weight is closest to the known h", might
extract that signal. Testing it on this data would be circular, because this data
generated the hypothesis. It needs fresh secrets, fresh K values and ideally a
fresh checkpoint, with the rule fixed in advance. That is a proposal for a future
phase, not a result of this one.

## 7. Bearing on the phase-11 causes

**MEASURED.** Degradation is monotone in sparsity across acc_tau, decode validity
and lift-over-best-constant, and the probes sit on the same curve.

**INFERENCE.** This is consistent with **distribution shift / out-of-distribution
generalization** as a mechanism of the phase-10 failure, and it is the outcome the
phase-11 hypothesis predicted in advance. The prediction was falsifiable -- a flat
curve, or an intact curve with a cliff only at the probes, would have contradicted
it -- and it was not falsified.

**This experiment does not establish causation.** It varies sparsity on one frozen
model and observes covariation. It cannot separate distribution shift from the
other phase-11 causes, because it holds those fixed rather than manipulating them:

| phase-11 cause | what phase 12 can say about it |
|---|---|
| 2. Distribution shift | Consistent with, and predicted in advance. Not proven causal: one model, one budget. |
| 4. Insufficient training | **Untouched.** One checkpoint at one training budget. A better-trained model might degrade more gracefully; nothing here tests that. |
| 3. Reduced capacity | **Untouched.** One model size. Nothing here compares 4.13M against 51M. |
| 1, 5, 6 | Unaffected by this experiment. |

**HYPOTHESIS.** Separating cause 2 from cause 4 needs this same sparsity curve
measured on checkpoints at several training budgets. If the zero-lift crossing
point moves toward sparser inputs with more training, sparsity tolerance is
something the model learns and cause 4 dominates; if it does not move, the
sensitivity is intrinsic and cause 2 dominates. Not run here.

## 8. Comparison with original SALSA

**MEASURED in phase 11, from the original source.** The original's `gen_expr` only
ever emits rows of the (nega)circulant matrix built from a uniform `a`. Probe-like
inputs are never generated during training: the measured probability that a
training row looks like a probe is 4.799e-26, about 1 in 2.1e25. The original
therefore also requires its transformer to extrapolate to inputs its own training
distribution does not produce.

**This does not mean the original collapses to the same degree.** That has not been
measured. The original uses roughly 51M parameters against our 4,131,200, roughly
3.9M distinct training samples against our 100,032, and reuses each sample about
ten times. Any of those could change how far a model extrapolates. Establishing
whether the original degrades along the same sparsity curve would require running
this same measurement against a trained original checkpoint. This phase did not do
that and does not claim it.

## Figures

- `01_accuracy_vs_sparsity.png`
- `02_acc_tau_vs_sparsity.png`
- `03_decode_validity_vs_sparsity.png`
- `04_output_entropy_vs_sparsity.png`

x-axis is the number of nonzero coordinates in `a`, dense on the left, with the
`K*e_i` probes marked as the extreme sparse endpoint.

## Artifacts

- `recovery_generalization.md` (this file)
- `recovery_generalization.json`
- `recovery_generalization.csv`
