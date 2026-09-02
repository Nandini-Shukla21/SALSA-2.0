# Phase 21 — does V2's recovery generalise beyond one secret?

**Inference and recovery only.** Nothing was trained or fine-tuned, no checkpoint
was modified, and the architecture, data generator, codec and recovery algorithm
are all unchanged.

## A problem with the specified design, and what was done about it

SALSA trains one model per secret. A frozen checkpoint encodes the secret it was trained on, so probing it always returns that secret. Experiment 1, the specified design, therefore cannot test generalisation across secrets and is reported as a secret-specificity control. Experiment 2 uses the three existing checkpoints, which were trained under different seeds and therefore on different secrets, and is the genuine test. It covers three secrets, not ten, because only three checkpoints exist and training more is out of scope.

Both experiments are reported. Experiment 1 is the design as specified; experiment
2 is the one that can answer the question.

## Experiment 1 — secret-specificity control (the specified design)

One frozen model (`results/equal_depth_ablation/v2_te2/nact_n12_h2_te2/seed_123/checkpoints/best.pt`, trained under seed 123), ten fresh secrets generated from a dedicated
seed unrelated to training, the phase-19 secret excluded, all unique, fixed
before the model was run.

- Training secret: `[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]`
- Recovered candidate: `[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]`
- Candidate equals the training secret: **True**
- **Candidate identical for all ten nominated secrets: True** — as it must be, since
  recovery never sees a secret.

| idx | generation seed | fresh secret | coord. accuracy | hamming | exact |
|---:|---:|---|---:|---:|:---:|
| 0 | 101429978 | `[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]` | 0.667 | 4 | no |
| 1 | 11005877 | `[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]` | 0.667 | 4 | no |
| 2 | 1736850619 | `[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]` | 0.667 | 4 | no |
| 3 | 368315112 | `[0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0]` | 0.833 | 2 | no |
| 4 | 3269322274 | `[1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]` | 0.833 | 2 | no |
| 5 | 1566455064 | `[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0]` | 0.833 | 2 | no |
| 6 | 2384904417 | `[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]` | 0.667 | 4 | no |
| 7 | 2894236140 | `[0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]` | 0.833 | 2 | no |
| 8 | 3249827391 | `[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]` | 0.833 | 2 | no |
| 9 | 808163997 | `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]` | 0.667 | 4 | no |

**Exact recoveries: 0/10.** This is the expected outcome
and it is not a failure of the model. A frozen per-secret model returns the secret
it was trained on; the probability that a nominated random weight-2 vector happens
to equal it is 1 in C(12,2) = 66. **This table measures secret-specificity,
not recovery robustness, and must not be read as the latter.**

## Experiment 2 — three checkpoints, three different secrets

The phase-17 runs under seeds 0, 42 and 123 were each trained against a different
secret. Attacking each with its own secret is a genuine multi-secret test and needs
no training.

| training seed | true support | recovered candidate | coord. acc | hamming | exact | #K exact | probe validity | verified |
|---:|---|---|---:|---:|:---:|---:|---:|:---:|
| 0 | [2, 10] | `[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]` | 1.000 | 0 | **YES** | 7/10 | 1.000 | PASS |
| 42 | [6, 7] | `[0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]` | 1.000 | 0 | **YES** | 8/10 | 1.000 | PASS |
| 123 | [4, 5] | `[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]` | 1.000 | 0 | **YES** | 8/10 | 1.000 | PASS |

Baselines, on the same secrets:

| training seed | recovered | all-zeros | all-ones | random weight-2 |
|---:|---:|---:|---:|---:|
| 0 | **1.000** | 0.833 | 0.167 | 0.667 |
| 42 | **1.000** | 0.833 | 0.167 | 0.667 |
| 123 | **1.000** | 0.833 | 0.167 | 0.833 |

Every recovered candidate beats the free all-zeros score of 0.833, and the
negative controls K=1 and K=250 (ring separation 1) failed on every secret, as
they must.

### Independent verification

Each claimed exact recovery was re-checked with the phase-20 residual verifier on
2,048 fresh samples from a dedicated stream. The verifier receives only
`(A, b, candidate, q)`.

| training seed | verification data seed | residual std | mean abs | ≤3σ | passes |
|---:|---:|---:|---:|---:|:---:|
| 0 | 3053420901 | 2.968 | 2.349 | 1.0000 | **PASS** |
| 42 | 2645249236 | 3.024 | 2.394 | 0.9976 | **PASS** |
| 123 | 2928911206 | 2.981 | 2.366 | 0.9976 | **PASS** |

An incorrect candidate would sit near the uniform reference of 72.5.

## Summary metrics (experiment 2)

| metric | value |
|---|---:|
| secrets attacked | 3 |
| exact recoveries | 3 |
| exact recovery rate | 1.0000 |
| partial | 0 |
| failed | 0 |
| mean coordinate accuracy | 1.0000 |
| median coordinate accuracy | 1.0000 |
| mean hamming distance | 0.0000 |
| median hamming distance | 0 |
| mean successful K | 7.6667 |
| fraction all informative K successful | 0.6667 |
| informative K count | 8 |
| mean probe decode validity | 1.0000 |
| verification success rate among claimed exact | 1.0000 |
| verified count | 3 |

## Failure analysis

No failures to analyse: every secret tested recovered exactly and verified.

The only non-recovering K values were the two negative controls, whose ring
separation of 1 makes the two hypotheses nearly indistinguishable. That is
failure mode **D, K-specific**, and it is the predicted and desired behaviour —
the algorithm was not changed to make them succeed.

## Interpretation

**MEASURED.** Across the three secrets for which a trained checkpoint exists,
direct recovery succeeded exactly 3/3 times, every claimed recovery passed independent
residual verification, mean coordinate accuracy was 1.000, mean Hamming distance 0.0, and a mean of 7.7 of 10 K values recovered
the secret on their own.

**MEASURED.** A frozen checkpoint returns only the secret it was trained on, for
every one of ten nominated alternatives.

**INFERRED.** The phase-19 result was not a peculiarity of one secret: the
mechanism reproduces on the two other secrets for which models exist. Three is a
small number and the secrets were not chosen adversarially, so this is suggestive
rather than settled.

**NOT ESTABLISHED.** Nothing here speaks to n=20, n=30 or n=128 recovery, to
practical LWE cryptanalysis, or to any general security break. Ten secrets were
requested; three were testable without training, so the robustness evidence is
thinner than the phase intended. This remains an n=12, h=2 diagnostic instance
with C(12,2) = 66 possible secrets.

The original SALSA is not compared against here: its setup, model size, sample
budget and reuse policy all differ, as phase 11 documented.

## Classification

### **A — exact recovery with independent verification on every secret tested**

Qualified: this is **3/3 on the secrets that could be tested**, not 10/10.
Reaching a ten-secret result requires training a model per secret, which this
phase excludes.

## Figures

- `01_recovery_rate.png`
- `02_coordinate_accuracy.png`
- `03_hamming_distance.png`
- `04_successful_K_distribution.png`

## Artifacts

- `recovery_robustness.md` (this file)
- `recovery_robustness.json`
- `recovery_robustness.csv`
