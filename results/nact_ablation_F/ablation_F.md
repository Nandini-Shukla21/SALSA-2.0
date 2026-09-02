# Phase 23 — Ablation F: one-token representation only

**One training run.** NACT-F at seed 0, plus the existing phase-12 sparsity
evaluation. No other training, no additional seeds, no secret recovery, no
architecture change; V1 and the full-NACT checkpoint are untouched.

## The variant

**Removed:** `centered residue`, `cos feature`, `sin feature`, `zero-indicator numerical feature`, `learned zero-coordinate vector`, `sparse attention key bias`.

**Retained:** `one coordinate token per a_i`, `base-81 digit embeddings`, `absolute coordinate embedding`, `transformer backbone`, `T_e=2, T_d=2`, `copy gate`, `RMSNorm`, `RoPE`, `decoder`.

The absolute coordinate embedding is retained deliberately: it is the identity
mechanism the one-token layout *requires* — position must mean coordinate index —
and it is not a numerical feature. Removing it is variant E, which was not approved.

**Parameters: 4,238,208**, exactly the 4,238,208 phase 22 predicted (True). That is -3,080 against full NACT, -0.073% — far too small for capacity to explain
any outcome. No padding parameters were added.

## Control parity

A field-by-field diff found **one** configuration difference: `model.nact_variant: full -> one_token_only`. Seed, n, h, q, sigma,
RLWE structure, encoding, optimizer, learning rate, scheduler, batch size,
validation size, sample budget, evaluation cadence, tau, T_e and T_d are all
inherited unchanged from the full-NACT config.

## Primary comparison

| Metric | V1 GatedUT | Full NACT | NACT-F |
|---|---:|---:|---:|
| Validation loss (nats) | 1.7413 | 0.9388 | **0.9891** |
| SALSA acc_tau | 0.3550 | 0.9839 | **0.9839** |
| Exact integer accuracy | 0.0078 | 0.1011 | **0.0913** |
| Token accuracy | 0.5324 | 0.6969 | **0.6934** |
| Greedy token accuracy | 0.4531 | 0.6825 | **0.6771** |
| Perfect accuracy | 0.0078 | 0.1011 | **0.0913** |
| Decode failure rate | 0.0000 | 0.0000 | **0.0000** |
| Mean |b - b_hat| | 63.4736 | 6.6875 | **7.1465** |
| Median |b - b_hat| | 38.0000 | 2.0000 | **3.0000** |
| Trainable parameters | 4,131,200 | 4,241,288 | 4,238,208 |
| Samples/sec | 369.1 | 96.1 | 108.9 |
| Tokens/sec | 1107.2 | 288.3 | 326.7 |
| Wall-clock (s) | 345 | 1294 | 1146 |
| CPU RSS (MiB) | 766.0 | 370.6 | 396.0 |

Baselines: uniform loss 4.4427, **marginal-only loss 1.8650** (the real zero point), chance token accuracy 0.44622, chance exact 0.00398, chance acc_tau 0.19287.

Throughput and wall-clock for V1 come from a different session and are not
comparable across arms; NACT-F and full NACT were run on the same machine.

## Full NACT vs NACT-F, and what one seed can resolve

| Metric | Full NACT | NACT-F | Change | Relative | 3-sigma seed band | Resolvable at one seed? |
|---|---:|---:|---:|---:|---:|:---:|
| Validation loss (nats) | 0.9388 | 0.9891 | +0.0503 | +5.35% | 0.0696 | no |
| SALSA acc_tau | 0.9839 | 0.9839 | +0.0000 | +0.00% | 0.0063 | no |
| Exact integer accuracy | 0.1011 | 0.0913 | -0.0098 | -9.66% | 0.0276 | no |
| Token accuracy | 0.6969 | 0.6934 | -0.0036 | -0.51% | - | - |
| Greedy token accuracy | 0.6825 | 0.6771 | -0.0054 | -0.79% | - | - |
| Perfect accuracy | 0.1011 | 0.0913 | -0.0098 | -9.66% | - | - |
| Decode failure rate | 0.0000 | 0.0000 | +0.0000 | - | - | - |
| Mean |b - b_hat| | 6.6875 | 7.1465 | +0.4590 | +6.86% | - | - |
| Median |b - b_hat| | 2.0000 | 3.0000 | +1.0000 | +50.00% | - | - |

The 3σ band is full NACT's own seed-to-seed spread, measured over the three
phase-17 runs. **Every primary difference falls inside it**, so at one seed the
two models are not distinguishable on these metrics.

### Fraction of the V1 → Full NACT gap that NACT-F retains

| metric | retained |
|---|---:|
| valid_loss | 93.7% |
| valid_acc_tau | 100.0% |
| valid_exact_accuracy | 89.5% |
| valid_token_accuracy | 97.8% |

## Sparsity

Same phase-12 procedure, same K sweep, same seed, same test vectors, unmodified.

| nnz | V1 lift | Full NACT lift | NACT-F lift |
|---:|---:|---:|---:|
| 12 | +0.1309 | +0.7642 | **+0.7651** |
| 10 | +0.0991 | +0.7402 | **+0.7393** |
| 8 | +0.0718 | +0.7231 | **+0.7222** |
| 6 | -0.0527 | +0.5879 | **+0.5981** |
| 4 | -0.1787 | +0.4243 | **+0.4346** |
| 2 | -0.3730 | +0.2158 | **+0.1895** |
| 1 | -0.4800 | +0.1196 | **+0.0562** |
| **K·e_i** | -0.4417 | +0.1167 | **+0.0500** |

Decode validity: V1 falls to 0.417 on the probes, while both NACT variants hold 1.000.

**NACT-F never crosses zero lift either**, so it keeps the property that
distinguishes V2 from V1. But its margin at the sparse end is thinner: at nnz=1 it
holds +0.0562 against full NACT's +0.1196, and on the probes +0.0500 against +0.1167 — roughly half.

## Interpretation

**MEASURED.** NACT-F reaches validation loss 0.9891, acc_tau 0.9839 and exact accuracy 0.0913, against full NACT's 0.9388 / 0.9839 / 0.1011 and V1's 1.7413 / 0.3550 / 0.0078. Every full-vs-F difference sits inside full
NACT's own one-seed noise band. Sparsity lift stays positive at every level for
NACT-F, with a thinner margin at nnz≤2 and on the probes.

**INFERRED.** On this evidence the bulk of NACT's advantage over V1 comes from
the **one-token coordinate representation plus the absolute coordinate identity
it makes expressible**, not from the centered residue, the Fourier pair, the zero
indicator, the zero vector or the sparse attention bias. The removed features are
not doing the heavy lifting on in-distribution learning at n=12, h=2. The thinner
sparse-end margin is the one place they may still be contributing, and this run
cannot tell whether that difference is real or seed noise.

**NOT ESTABLISHED.**

- **That the removed features contribute nothing.** One seed cannot establish a
  null. Phase 22 pre-registered exactly this case: a variant landing within 3σ of
  full NACT needs more seeds before 'no effect' can be claimed. Only one seed was
  authorised, so the honest statement is *indistinguishable at one seed*, not
  *equivalent*.
- **Which individual component matters.** F removes six things at once. Variants
  C, D and E were not run.
- Behaviour at n=20, n=30 or n=128.
- Secret-recovery capability for NACT-F — not run, and out of scope for this phase.
- General secret-recovery robustness.

**NACT-F is not 'better' for having fewer parameters.** It is 3,080 parameters
smaller, 0.073%, which is scientifically meaningless. The interesting result is
that removing those components changed so little, not that the model shrank.

## A note on the artifacts

The phase-12 sparsity script writes its own `recovery_generalization.md` last and
raised a `TypeError` while formatting it, because NACT-F's non-modal probe
coordinates are not equal to the secret support and an optional probability is
`None`. Its JSON, CSV and figures — the measurement — were written before that
point and are complete. The evaluation implementation was not modified, as
instructed. Separately, passing `--run-dir` as a relative path triggers a
`relative_to()` failure in the same reporting tail; passing an absolute path
avoids it and is what produced these artifacts. That is also why phase 17's
sparsity JSON was missing.

## Figures

- `01_loss_comparison.png`
- `02_acc_tau_comparison.png`
- `03_exact_accuracy_comparison.png`
- `04_sparsity_lift_comparison.png`

## Artifacts

- `ablation_F.md` (this file)
- `ablation_F.json`
- `ablation_F.csv`

PHASE 23 COMPLETE — ABLATION F ONLY. NO OTHER TRAINING PERFORMED.
