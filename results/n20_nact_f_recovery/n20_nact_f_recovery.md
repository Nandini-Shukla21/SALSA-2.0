# Phase 26 — NACT-F direct secret recovery at n=20, h=2

**Inference, recovery and verification only.** Nothing trained or fine-tuned, no
checkpoint modified, and the architecture, codec, data generator and recovery
rules are all unchanged. The phase-20 verifier was imported unmodified.

## Result

```
EXACT SECRET RECOVERY: YES

recovered (aggregate vote)  [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
true secret                 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
Hamming distance            0
coordinate accuracy         1.0000
K values exact              8/10  [31, 63, 94, 125, 126, 157, 188, 220]
```

## A defect this experiment exposed — reported, not patched

**The secret-free single-K selection rule chose K=1, a
negative control, and that candidate is NOT an exact recovery** (0.9000 accuracy, Hamming 2).

At K=1 and K=250 the ring separation is 1, so the two anchors are adjacent. The model answers a constant 0, giving distance 0 to one anchor and 1 to the other, so |score| = 1 for every coordinate and the mean margin is the maximum possible. The margin rule therefore rewards the LEAST informative probes precisely because they are least informative. This did not surface at n=12, where the margins at K=1 were 0.044 (NACT-F) and 0.333 (full NACT). The rule was NOT changed.

The exact recovery above comes from the **separation-weighted vote over all K (also secret-free)**, which
is equally secret-free and weights each K by its ring separation — so the two
degenerate probes contribute 0.008 each and are effectively ignored. Both rules
ship in the existing implementation; one failed here and one did not.

**I did not change the rule, re-run with different rules, or select a different
checkpoint.** The failure is a real limitation of the margin heuristic at low
separation and is more useful reported than hidden.

## Checkpoint

`C:\Users\Nandini Shukla\OneDrive\Desktop\SALSA 2.0\results\n20_nact_f_pilot\nact_n20_h2_te2_F\seed_0\checkpoints\best.pt`

Located via run metadata best_epoch, monitor valid_loss (min); filename not assumed; loaded `strict=True`. All 18
checks passed: 4,238,208 parameters, variant `one_token_only` with switches {'use_numerical_features': False, 'use_zero_vector': False, 'use_sparse_attention_bias': False}, n=20, h=2, q=251, sigma=3.0, RLWE/circulant, representation R, T_e=2, T_d=2, seed 0, best epoch 19, fingerprint `bbf51d2d5d8182c5`.

## Per-K results

| K | sep | weight | decode validity | margin | unique preds | modal | candidate weight | accuracy | exact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 1 *(neg. control)* | 1 | 0.008 | 1.000 | 1.0000 | 1 | 0 | 0 | 0.900 | no |
| 31 | 31 | 0.247 | 1.000 | 0.9742 | 2 | 0 | 2 | 1.000 | **YES** |
| 63 | 63 | 0.502 | 1.000 | 0.9815 | 3 | 0 | 2 | 1.000 | **YES** |
| 94 | 94 | 0.749 | 1.000 | 0.9819 | 4 | 0 | 2 | 1.000 | **YES** |
| 125 | 125 | 0.996 | 1.000 | 0.9576 | 4 | 3 | 2 | 1.000 | **YES** |
| 126 | 125 | 0.996 | 1.000 | 0.9682 | 4 | 3 | 2 | 1.000 | **YES** |
| 157 | 94 | 0.749 | 1.000 | 0.9467 | 5 | 3 | 2 | 1.000 | **YES** |
| 188 | 63 | 0.502 | 1.000 | 0.9968 | 2 | 0 | 2 | 1.000 | **YES** |
| 220 | 31 | 0.247 | 1.000 | 0.9882 | 4 | 0 | 2 | 1.000 | **YES** |
| 250 *(neg. control)* | 1 | 0.008 | 1.000 | 1.0000 | 1 | 0 | 0 | 0.900 | no |

**Negative controls failed as required: True.** Both produced the all-zeros
candidate — which at n=20 scores 0.900 for free — and neither is counted.

## Baselines

*A weight-2 secret over 20 coordinates gives an all-zeros guess 0.9000 coordinate accuracy for free -- higher than the 0.8333 trap at n=12. Coordinate accuracy alone is NOT evidence of recovery; only exact recovery is.*

| candidate | coordinate accuracy | hamming | exact |
|---|---:|---:|:---:|
| **recovered aggregate** | **1.0000** | 0 | YES |
| all zeros | 0.9000 | 2 | no |
| all ones | 0.1000 | 18 | no |
| random weight 2 | 0.8000 | 4 | no |

## Independent verification

*scripts/v2_verification.py residual_statistics, imported unmodified.* Fresh split `phase26_verify_n20`, data
seed 4222496425, 2,048 samples. The
verifier receives only `(A, b, candidate, q)` — the true secret never reaches it.

| candidate | residual std | mean abs | median abs | p95 | ≤3σ |
|---|---:|---:|---:|---:|---:|
| **recovered aggregate** | **2.982** | 2.335 | 2.0 | 6.0 | 0.9995 |
| all zeros | 71.879 | 62.088 | 64.0 | 119.0 | 0.0762 |
| all ones | 72.100 | 62.355 | 63.0 | 119.0 | 0.0786 |
| random weight 2 | 72.634 | 62.672 | 64.0 | 119.0 | 0.0820 |

Configured sigma is 3; any wrong candidate
sits near the uniform reference of 72.5.

| criterion | result |
|---|:---:|
| `c1_std_close_to_sigma` | PASS |
| `c2_mean_abs_small` | PASS |
| `c3_beats_baselines` | PASS |
| `c4_within_three_sigma` | PASS |

### **VERIFICATION: PASS**

## n=12 vs n=20 (both NACT-F)

| | n=12, h=2 | n=20, h=2 |
|---|---:|---:|
| Search space C(n,2) | 66 | 190 |
| Exact recovery | YES | **YES** |
| K values exact | 8/10 | **8/10** |
| Coordinate accuracy | 1.0000 | **1.0000** |
| Verification | PASS | **PASS** |

The same eight informative K values succeeded at both dimensions, and both
negative controls failed at both. The pipeline behaved identically.

## Comparison with V1 at n=20

*V1 values are from phase 7 (control_b_n20_h2). V1 was NOT retrained and direct recovery was NEVER run against a V1 n=20 checkpoint, so the recovery row is 'not measured', not 'failed'.*

| | V1 GatedUT (phase 7) | NACT-F (phases 25–26) |
|---|---:|---:|
| Parameters | 4,131,200 | 4,238,208 |
| Training samples | 200,000 | 100,032 |
| Validation loss | 1.8233 | 0.9764 |
| SALSA acc_tau | 0.2344 | 0.9863 |
| Exact integer accuracy | 0.0034 | 0.1035 |
| Probe decode validity | not measured at n=20 | 1.0000 |
| Direct secret recovery | **not measured** | **exact, verified** |

**V1's recovery row is 'not measured', not 'failed'.** Direct recovery was never
run against a V1 n=20 checkpoint. What is known is that at n=12 V1 failed
completely (58% of probes undecodable, single constant answer on 8 of 10 K (phase 10/12)), but that is a different
dimension and is not evidence about n=20.

## Classification

### **A — Exact recovery + independent verification**

NACT-F demonstrates end-to-end direct secret recovery and independent residual
verification at n=20, h=2.

## The six answers

1. **Exact recovery:** YES — via the separation-weighted
   aggregate vote. The single-K margin rule failed and picked a negative control.
2. **Coordinate accuracy:** 1.0000, against 0.9000
   for a free all-zeros guess — which is why only exact recovery counts here.
3. **Hamming distance:** 0.
4. **Successful K values:** 8 of 10 — every informative K, both negative controls failing.
5. **Verification:** PASS — residual std 2.982 against sigma 3, wrong candidates near 72.5.
6. **Does NACT-F generalize from n=12 to n=20?** On this evidence, yes for both
   learning and recovery: same architecture, same parameter count, same sample
   budget, same eight successful K values, exact recovery and verification at
   both dimensions. Two dimensions is not a trend, and one seed and one secret
   per dimension is a narrow base.

## Limitations

- **Not a practical LWE break.** n=20, h=2 has C(20,2) = 190 possible secrets. This is an
  experimental diagnostic scale.
- **No general cryptanalytic success**, and nothing here about n=30 or n=128.
- **One secret, one seed, one checkpoint** at n=20.
- **The single-K selection rule is not reliable at low separation** and would need
  revision before being used unsupervised — a finding, not a fix applied here.
- V1's n=20 recovery was never measured, so the recovery comparison is one-sided.

## Artifacts

- `n20_nact_f_recovery.md` (this file)
- `n20_nact_f_recovery.json`
- `n20_nact_f_recovery.csv`
- `direct_recovery_n12_h2.json` — raw output of the phase-19 recovery script
