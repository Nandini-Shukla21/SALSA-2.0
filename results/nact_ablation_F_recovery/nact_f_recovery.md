# Phase 24 — NACT-F direct secret recovery

**Recovery only.** Nothing was trained or resumed, no checkpoint was modified,
and the model, codec, data generator and recovery algorithm are all unchanged.
The phase-20 residual verifier was imported from its own file unmodified.

## Result

```
EXACT SECRET RECOVERY: YES

recovered candidate  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
true secret          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]
Hamming distance     0
coordinate accuracy  1.0000
K values exact       8/10  [31, 63, 94, 125, 126, 157, 188, 220]
```

## Checkpoint

`C:\Users\Nandini Shukla\OneDrive\Desktop\SALSA 2.0\results\nact_ablation_F\nact_n12_h2_te2_F\seed_0\checkpoints\best.pt`

Located via run metadata (artifacts/summary.json best_epoch, monitor valid_loss); filename not assumed. Loaded with `strict=True`.
Verified: **4,238,208 parameters**, variant
`one_token_only` with all three ablation switches off ({'use_numerical_features': False, 'use_zero_vector': False, 'use_sparse_attention_bias': False}), n=12, h=2, q=251, sigma=3.0, representation R, T_e=2, T_d=2, seed 0, fingerprint `f148ac749a6db162`.

## Secret-free guarantees

- K selected by *highest mean decision margin (uses predictions only)* — a function of the model's own predictions. Selected K = **188**.
- `DirectRecovery` exposes no parameter through which a secret could be passed.
- The candidate is produced and printed before the secret is loaded.

K sweep: `[1, 31, 63, 94, 125, 126, 157, 188, 220, 250]` — the phase-19 sweep, unchanged.

## Per-K behaviour

| K | sep | decode validity | margin | unique preds | modal | candidate | weight | accuracy | exact |
|---:|---:|---:|---:|---:|---:|---|---:|---:|:---:|
| 1 *(neg. control)* | 1 | 1.000 | 0.044 | 2 | 241 | `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` | 0 | 0.833 | no |
| 31 | 31 | 1.000 | 0.609 | 4 | 238 | `[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]` | 2 | 1.000 | **YES** |
| 63 | 63 | 1.000 | 0.725 | 3 | 238 | `[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]` | 2 | 1.000 | **YES** |
| 94 | 94 | 1.000 | 0.871 | 6 | 240 | `[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]` | 2 | 1.000 | **YES** |
| 125 | 125 | 1.000 | 0.804 | 2 | 237 | `[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]` | 2 | 1.000 | **YES** |
| 126 | 125 | 1.000 | 0.801 | 4 | 237 | `[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]` | 2 | 1.000 | **YES** |
| 157 | 94 | 1.000 | 0.730 | 4 | 237 | `[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]` | 2 | 1.000 | **YES** |
| 188 | 63 | 1.000 | 0.943 | 5 | 0 | `[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]` | 2 | 1.000 | **YES** |
| 220 | 31 | 1.000 | 0.896 | 4 | 0 | `[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]` | 2 | 1.000 | **YES** |
| 250 *(neg. control)* | 1 | 1.000 | 0.049 | 2 | 241 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]` | 12 | 0.167 | no |

**Negative controls behaved as required: True.** K=1 and K=250 have ring
separation 1, so the two hypotheses are almost the same target and they must fail.

**K=1 is worth pausing on.** It produced the all-zeros candidate, which scores
0.833 coordinate accuracy for free on a weight-2 secret over 12 coordinates. That
is exactly the trap a sparse instance sets: it is **not** a recovery, it is what
you get for guessing nothing, and it is scored as a failure here. K=250 produced
all-ones at 0.167. Neither is counted.

## Baselines

| candidate | coordinate accuracy | hamming | exact |
|---|---:|---:|:---:|
| **recovered candidate** `[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]` | **1.0000** | 0 | YES |
| all zeros `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` | 0.8333 | 2 | no |
| all ones `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]` | 0.1667 | 10 | no |
| random weight 2 `[0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]` | 0.6667 | 4 | no |

## Independent verification

*scripts/v2_verification.py residual_statistics, imported unmodified.* Fresh split `phase24_verify_nact_f`, data seed 1554056842, 2,048 samples. The
verifier receives only `(A, b, candidate, q)`.

| candidate | residual std | mean abs | ≤3σ | max abs |
|---|---:|---:|---:|---:|
| **recovered candidate** | **2.967** | 2.355 | 0.9985 | 10 |
| all zeros | 72.816 | 62.952 | 0.0776 | 125 |
| all ones | 72.471 | 62.672 | 0.0767 | 125 |
| random weight 2 | 72.280 | 62.725 | 0.0669 | 125 |

Configured sigma is 3; any wrong candidate sits near the uniform reference of 72.5.

| criterion | result |
|---|:---:|
| `c1_std_close_to_sigma` | PASS |
| `c2_mean_abs_small` | PASS |
| `c3_beats_baselines` | PASS |
| `c4_within_three_sigma` | PASS |

### **VERIFICATION: PASS**

## V1 vs Full NACT vs NACT-F (all seed 0)

| Metric | V1 GatedUT | Full NACT | NACT-F |
|---|---:|---:|---:|
| Trainable parameters | 4,131,200 | 4,241,288 | **4,238,208** |
| Validation loss | 1.7413 | 0.9388 | **0.9891** |
| SALSA acc_tau | 0.3550 | 0.9839 | **0.9839** |
| Exact integer accuracy | 0.0078 | 0.1011 | **0.0913** |
| Probe decode validity | 0.4167 | 1.0000 | **1.0000** |
| Probe lift | -0.4417 | 0.1167 | **0.0500** |
| Exact secret recovery | **NO** | **YES** | ****YES**** |
| K values with exact recovery | 0 | 7 | **8** |

All three rows are the seed-0 checkpoints, so the comparison is like-for-like.
Full NACT's seed-0 recovery figures come from the phase-21 robustness run, not
from phase 19 (which attacked seed 123).

## Decision

### **A — NACT-F recovers and independently verifies the secret**

## Answers to the six questions

1. **Did NACT-F recover the secret?** Yes, exactly.
2. **How many K values succeeded?** 8 of 10 — every informative K, with both negative controls
   failing as they should.
3. **Coordinate accuracy:** 1.0000 (Hamming distance 0), against 0.8333 for a free
   all-zeros guess.
4. **Verification:** PASS — residual std 2.967 against a configured sigma of 3, while every wrong candidate sits near 72.5.
5. **Comparison with Full NACT:** both recover exactly and both verify. NACT-F
   used more successful K values on this seed, but that is a single-seed
   difference and not a meaningful ranking.
6. **Are the extra numerical and sparse-aware features necessary for recovery?**
   **On this instance, no.** NACT-F removed the centered residue, both Fourier
   features, the zero indicator, the zero vector and the sparse attention bias,
   and still completed the pipeline end to end. That is a stronger statement than
   phase 23 could make: phase 23 showed the features were not needed for
   *prediction*; this shows they were not needed for *recovery* either — despite
   NACT-F's probe lift being roughly half of full NACT's. The margin narrowed
   without the outcome changing.

## Statement

NACT-F reproduced the complete direct-recovery pipeline on the n=12, h=2
diagnostic instance.

**This is not a practical LWE break**, and says nothing about n=20, n=30 or
n=128, or about general cryptanalytic success. The instance has C(12,2) = 66
possible secrets. One secret, one checkpoint, one dimension.

## Artifacts

- `nact_f_recovery.md` (this file)
- `nact_f_recovery.json`
- `nact_f_recovery.csv`
- `direct_recovery_n12_h2.json` — raw output of the phase-19 recovery script
