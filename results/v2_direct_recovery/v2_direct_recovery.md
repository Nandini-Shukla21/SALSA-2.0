# Phase 19 — V2 NACT direct secret recovery

**Recovery only.** No training, no resume, no checkpoint modified, no model, data
generator or codec change. `salsa/recovery/direct.py` — the recovery algorithm
itself — is byte-identical to phase 10.

## Result

```
EXACT SECRET RECOVERY: YES

recovered candidate  [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
true secret          [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
Hamming distance     0
correct coordinates  12 / 12
coordinate accuracy  1.0000
```

## Step 1 — checkpoint

`results\equal_depth_ablation\v2_te2\nact_n12_h2_te2\seed_123\checkpoints\best.pt`

Loaded with `strict=True`. Verified: 4,241,288 parameters, arch `nact`, n=12, h=2, q=251, sigma=3.0, representation R, vocabulary 85, fingerprint `a4cb9d0c870aec5e`.

## Steps 2–4 — the procedure was secret-free

This is the load-bearing claim, so it is evidenced rather than asserted:

- `DirectRecovery.__init__` takes `['model', 'codec', 'method', 'batch_size']` and `recover` takes `['k_values']`. **There is no parameter through which a secret could be passed.**
- K was selected by *highest mean decision margin (uses predictions only)* — a function of the model's own predictions. Selected K = **125**.
- The candidate is printed before the secret is loaded; the true secret enters only
  in the evaluation stage that follows.
- `tests/test_direct_recovery.py - 30 passed`, including tests asserting the
  recovery package cannot reference ground truth.

K sweep: `[1, 31, 63, 94, 125, 126, 157, 188, 220, 250]` — the phase-10 sweep, unchanged.

## Steps 3 & 5 — per-K results

| K | K mod q | separation | decode validity | margin (confidence) | unique preds | modal | recovered candidate | accuracy | hamming | exact |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|:---:|
| 1 | 1 | 1 | 1.000 | 0.333 | 1 | 2 | `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]` | 0.167 | 10 | no |
| 31 | 31 | 31 | 1.000 | 0.858 | 2 | 2 | `[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]` | 1.000 | 0 | **YES** |
| 63 | 63 | 63 | 1.000 | 0.915 | 3 | 2 | `[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]` | 1.000 | 0 | **YES** |
| 94 | 94 | 94 | 1.000 | 0.954 | 2 | 2 | `[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]` | 1.000 | 0 | **YES** |
| 125 | 125 | 125 | 1.000 | 0.968 | 2 | 2 | `[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]` | 1.000 | 0 | **YES** |
| 126 | 126 | 125 | 1.000 | 0.963 | 3 | 2 | `[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]` | 1.000 | 0 | **YES** |
| 157 | 157 | 94 | 1.000 | 0.959 | 3 | 2 | `[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]` | 1.000 | 0 | **YES** |
| 188 | 188 | 63 | 1.000 | 0.950 | 2 | 2 | `[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]` | 1.000 | 0 | **YES** |
| 220 | 220 | 31 | 1.000 | 0.886 | 2 | 2 | `[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]` | 1.000 | 0 | **YES** |
| 250 | 250 | 1 | 1.000 | 0.200 | 1 | 2 | `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` | 0.833 | 2 | no |

**8 of 10 K values recover the secret exactly.** The aggregate
separation-weighted vote also recovers it exactly.

### Negative controls behaved as required

K=1 and K=250 have ring separation 1, so the two hypotheses are nearly the same target and they SHOULD fail. They do. Had they succeeded, the measurement would be suspect rather than the model good. Both failed as required: **True**.

### Per-coordinate detail at the selected K

K = 125, separation 125. The
arithmetic the model is doing is visible directly:

| i | true s_i | recovered | decoded b | d(b,0) | d(b,K) | score |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 2 | 2 | 123 | -0.968 |
| 1 | 0 | 0 | 2 | 2 | 123 | -0.968 |
| 2 | 0 | 0 | 2 | 2 | 123 | -0.968 |
| 3 | 0 | 0 | 2 | 2 | 123 | -0.968 |
| 4 | 1 | 1 | 127 | 124 | 2 | +0.968 |
| 5 | 1 | 1 | 127 | 124 | 2 | +0.968 |
| 6 | 0 | 0 | 2 | 2 | 123 | -0.968 |
| 7 | 0 | 0 | 2 | 2 | 123 | -0.968 |
| 8 | 0 | 0 | 2 | 2 | 123 | -0.968 |
| 9 | 0 | 0 | 2 | 2 | 123 | -0.968 |
| 10 | 0 | 0 | 2 | 2 | 123 | -0.968 |
| 11 | 0 | 0 | 2 | 2 | 123 | -0.968 |

For every zero coordinate the model answers ≈0; for the two support coordinates it
answers ≈125. That is exactly `b = K·s_i + e mod q`. **The model
is computing the modular arithmetic, not pattern-matching a memorised output.**

## Step 6 — baselines (so sparsity cannot fake a recovery)

| candidate | coordinate accuracy | hamming | exact |
|---|---:|---:|:---:|
| all-zeros `[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]` | 0.8333 | 2 | no |
| all-ones `[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]` | 0.1667 | 10 | no |
| **recovered** `[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]` | **1.0000** | 0 | **YES** |

A weight-2 secret over 12 coordinates gives an all-zeros guess 0.833 for free. The recovered candidate
reaches 1.000 and is exact, so it is not the sparsity artefact that would flatter a
weak result. The all-ones complement scores 0.167, confirming neither polarity is
privileged.

## Step 7 — V1 vs V2

| Metric | V1 GatedUT | V2 NACT | Change |
|---|---|---|---|
| Architecture | Salsa2-GatedUT | Salsa2-NACT | front end replaced |
| Trainable parameters | 4,131,200 | 4,241,288 | 110,088 |
| Probe decode validity | 0.4167 | 1.0000 | 0.5833 |
| Mean unique predictions across i (per K) | 0.8000 | 2.1000 | 1.3000 |
| K values giving one constant answer | 8 | 2 | -6 |
| K values with exact recovery | 0 | 8 | 8 |
| Coordinate accuracy (selected K) | 0.1667 | 1.0000 | 0.8333 |
| EXACT SECRET RECOVERY | NO | YES | NO -> YES |
| Probe lift (phase-12 procedure) | -0.4417 | 0.1167 | measured on the phase-17 SEED-0 checkpoint, not this seed-123 one |

The probe-lift row compares V1's phase-12 measurement against the phase-17
measurement, which was taken on the **seed-0** NACT checkpoint. The recovery above
runs against **seed 123**. Same configuration, different run — the lift figure is
context, not a measurement of this checkpoint.

V1's failure mode is gone entirely: it left 58% of probes undecodable and answered
with a single constant on most K values. V2 decodes every probe and varies its
answer by coordinate.

## Step 8 — classification

**A. Exact secret recovery achieved.**

The full secret was recovered by the secret-free procedure, at the K chosen without
reference to the answer, and by the aggregate vote. It beats the all-zeros baseline
and the negative controls failed as they should.

### How the candidate was produced, and why no ground truth influenced it

1. `build_probe_matrix(n, K, q)` builds `K·e_i` from public parameters only.
2. The model is driven from `<bos>`; no target sequence is supplied, so nothing
   derived from `b` or `s` reaches it.
3. Each prediction is assigned to whichever of `0` or `K mod q` it is nearer on the
   ring — the anchored rule, which needs no polarity resolution and therefore no
   ground truth. (The released SALSA code resolves polarity *against the true
   secret*; phase 10 replaced that precisely because it cannot be part of an attack.)
4. K is chosen by highest mean absolute margin — a property of the predictions.
5. Only then is the secret loaded, for scoring.

## What this does and does not establish

- **Establishes:** a 4,241,288-parameter CPU-trained model, at equal encoder depth
  to V1 and on the identical 100,032-sample budget, performs SALSA Algorithm 1
  end-to-end on this instance where the 4,131,200-parameter V1 failed completely.
- **Does not establish anything cryptographic.** n=12, h=2 has C(12,2) = 66 possible
  secrets — a diagnostic positive control, as it has been since phase 7. Recovering
  a secret from a 66-element space is not an attack on LWE.
- **One secret, one checkpoint, one dimension.** Seed 123 only. Whether this holds
  across secrets, seeds, or at n=20/30/128 is untested here.
- **No verification step.** The paper's residual test (§4.4) is not implemented; the
  candidate is reported, not independently verified against fresh samples.

## Artifacts

- `v2_direct_recovery.md` (this file)
- `v2_direct_recovery.json`
- `v2_direct_recovery.csv`
- `direct_recovery_n12_h2.json` — raw output of the phase-10 recovery script
