# Phase 27 — Salsa 2.0 end-to-end pipeline

**Integration and read-only replay.** No training, no new experiment, no
checkpoint created or modified.

## The workflow, in one call

```
  public LWE/RLWE data
          |
      encoding  (LatticeCodec, base 81, lsb-first, no separator)
          |
      NACT-F    (one coordinate token per a_i, encoder sequence n+2)
          |
      predict b (greedy decode)
          |
  direct secret recovery   <- STAGE 1: no secret in scope
          |
     candidate secret
          |
  residual verification    <- STAGE 2: only (A, b, candidate, q, sigma)
          |
    structured result      <- STAGE 3: the secret enters here, last
```

```python
from salsa import run_salsa2_pipeline

result = run_salsa2_pipeline('configs/pipeline_n12.yaml')
print(result.exact_recovery, result.verification_passed)
```

or `python scripts/run_pipeline.py configs/pipeline_n20.yaml`.

## Why the three stages are separate fields

They have different epistemic status, and merging them is how a recovery
experiment fools itself:

| stage | what it may see |
|---|---|
| **recovery** | public probes and model predictions only |
| **verification** | (A, b, candidate, q, sigma) only |
| **evaluation** | the only stage that loads the true secret, and it runs last |

The ordering is enforced by construction — the secret is not read until a
candidate *and* a verification verdict already exist — and `evaluate=False`
runs the whole thing with no ground truth at all, which is the real attack
setting.

## Recovery rule hardening

**The defect.** the unguarded best_margin rule selected K=1 at n=20 because a constant answer at separation 1 yields the maximum possible margin.

**What changed.** Two rules are now named and documented:

- **`aggregate (separation-weighted vote)`** — the default.
- **`best_margin (single K, guarded by min_separation)`** — retained as a diagnostic.

**The guard.** min_separation defaults to floor(sigma) + 1 = 4 at sigma=3, which excludes K=1 and K=250 (separation 1) from selection. It is derived from the problem rather than
picked: a probe can only discriminate if its two hypotheses are further apart
than the error scale. Excluded K are still measured and still vote, with the
negligible weight their separation earns them — they are excluded from
*selection*, not from the evidence.

**Still secret-free: True.** Separation is a function of
K and q only. Nothing in the rule consults the secret.

Measured effect: at n=20 the unguarded rule selected K=1; the guarded rule
selects K=188.

## Replay of the established results

| check | n=12 expected | n=12 replayed | n=20 expected | n=20 replayed |
|---|---:|---:|---:|---:|
| exact_recovery | True | True | True | True |
| coordinate_accuracy | 1.0 | 1.0 | 1.0 | 1.0 |
| hamming_distance | 0 | 0 | 0 | 0 |
| successful_k | 8 | 8 | 8 | 8 |
| verification_passed | True | True | True | True |
| parameter_count | 4238208 | 4238208 | 4238208 | 4238208 |

n=12 source: phase 24 (results/nact_ablation_F_recovery). n=20 source: phase 26 (results/n20_nact_f_recovery).

**Both replays reproduce the established results exactly.**

Verification residuals on fresh samples:

| | n=12 | n=20 |
|---|---:|---:|
| verification split | pipeline_verify_n12 | pipeline_verify_n20 |
| data seed | 2156377967 | 276041130 |
| residual std | 2.998 | 2.941 |
| all-zeros baseline std | 71.803 | 72.477 |
| runtime (s) | 0.87 | 1.41 |

The whole pipeline runs in about a second per configuration, because nothing
in it trains.

## Final scientific summary

| Stage | n=12, h=2 | n=20, h=2 |
|---|---|---|
| learning | measured | measured |
| direct recovery | **YES** | **YES** |
| coordinate accuracy | **100%** | **100%** |
| verification | **PASS** | **PASS** |
| search space C(n,2) | 66 | 190 |
| successful K | 8/10 | 8/10 |

### How the project got here

- **V1 Salsa2-GatedUT, 4,131,200 parameters.** Learned in-distribution at
  n=12 but direct recovery failed completely: 58% of probes were undecodable
  and the model answered a single constant on 8 of 10 K values.
- **The sparse-input problem.** Phase 12 measured V1's lift over the best
  input-blind constant falling monotonically with sparsity and crossing zero
  between nnz=8 and nnz=6, reaching −0.44 on the `K·e_i` probes. Phase 11 had
  measured why: a probe is roughly a 1-in-10²⁵ input under the training
  distribution, and detecting a zero coordinate is a conjunction across two
  token positions.
- **NACT was introduced** to attack that: one token per coordinate, absolute
  coordinate identity, and numerical features chosen for the modular structure.
- **NACT-F, 4,238,208 parameters,** is the ablation that removed every extra
  numerical, zero-aware and sparse-bias component and kept only the one-token
  representation. It matched full NACT within one-seed noise, which is why the
  **one-token representation is the important simplification** — the features
  were not doing the heavy lifting.
- **Recovery improvement.** V1: no exact recovery, probe decode validity 0.417.
  NACT-F: exact recovery at both dimensions, probe decode validity 1.000 and 1.000.
- **Verification.** Residual std ~3.0 against a configured sigma of 3.0, while
  every incorrect candidate sits near 72.5 — a separation of roughly 24x.

### Currently demonstrated

**n=12, h=2 and n=20, h=2** — learning, exact direct secret recovery, and
independent residual verification, end to end.

### Not demonstrated

**n=30. n=50. n=128. A practical cryptanalytic attack. A general security
break.** These are diagnostic instances: C(12,2) = 66 and C(20,2) = 190
possible secrets. One secret, one checkpoint and one seed per dimension. Two
dimensions is not a scaling law.

## Artifacts

- `pipeline_report.md` (this file)
- `pipeline_report.json`
- `pipeline_report.csv`
- `nact_f_n12_result.json`, `nact_f_n20_result.json` — full result objects

**PHASE 27 COMPLETE — END-TO-END PIPELINE INTEGRATION ONLY. NO NEW TRAINING PERFORMED.**
