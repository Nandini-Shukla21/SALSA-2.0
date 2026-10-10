# SALSA 2.0 — Project Report

**Lightweight Neural Cryptanalysis of LWE/RLWE**

Final model **Modified NACT** · **4,238,208 parameters** · CPU-only
Status: **COMPLETE / FROZEN**

> **Scope boundary.** SALSA 2.0 achieves exact secret recovery on controlled
> low-dimensional LWE/RLWE-style instances at n=12 and n=20 with h=2. **It does
> not establish a practical break of deployed LWE/RLWE cryptography.** The
> demonstrated instances have C(12,2) = 66 and C(20,2) = 190 possible secrets.

*Every figure below is read from recorded artifacts under `results/`. Nothing is
estimated or invented. The deeper technical treatment is in
`results/final_pipeline/SALSA2_Final_Report.md`.*

---

## 1. What the project asked

SALSA (Wenger, Chen, Charton, Lauter, NeurIPS 2022) attacks Learning With Errors
by training a Transformer on public samples `(A, b)` to predict `b`. If the
model learns the map, it has implicitly learned the secret, which can then be
read back out through chosen inputs the model never saw in training.

The published configuration uses roughly **51M parameters**. This project asked:

1. Does the attack survive a ~12× parameter reduction?
2. If it fails, is the cause capacity, training budget, or representation?
3. Can the whole thing — learn, recover, verify — run reproducibly on a CPU?

## 2. The problem

```
b = A s + e   (mod q)
```

| symbol | meaning | value used |
|---|---|---|
| `A` | public matrix, RLWE circulant rows | — |
| `s` | the secret, binary, **recovered not given** | unknown to the attack |
| `e` | noise, discrete Gaussian | — |
| `q` | modulus (prime) | 251 |
| `n` | dimension | 12, 20 |
| `h` | Hamming weight — non-zero secret coordinates | 2 |
| `sigma` | noise scale | 3 |

**A sparse secret gives accuracy away.** An all-zeros guess already scores
`(n−h)/n` — **0.8333 at n=12, 0.9000 at n=20**. Coordinate accuracy alone is
therefore never evidence of recovery; only *exact* recovery is.

## 3. Headline results

### Secret recovery — the cryptanalytic outcome

| | n=12, h=2 | n=20, h=2 |
|---|---:|---:|
| **Exact secret recovery** | **YES** | **YES** |
| Coordinate accuracy | 1.0000 | 1.0000 |
| Hamming distance | 0 | 0 |
| Successful probe multipliers | 8/10 | 8/10 |
| Probe decode validity | 1.0000 | 1.0000 |
| **Independent residual verification** | **PASS** | **PASS** |
| Search space C(n,2) | 66 | 190 |

The two multipliers that fail are `K = 1` and `K = 250`, the deliberate
**negative controls** — their ring separation is 1, so they *should* fail.

### Verification residuals

| | n=12 | n=20 |
|---|---:|---:|
| recovered candidate, residual std | **2.998** | **2.941** |
| all-zeros baseline, residual std | 71.803 | 72.477 |
| configured `sigma` | 3.0 | 3.0 |

A correct candidate leaves `b − As = e`, tight at sigma. A wrong one leaves
something near-uniform on `Z_q`, std ≈ 72.5. **That ~24× separation is the
evidence**, and the verifier never sees the secret.

### Prediction quality — a separate claim

| | n=12 | n=20 | chance |
|---|---:|---:|---:|
| validation loss (nats) | 0.9891 | 0.9764 | 1.86498 (marginals-only) |
| `acc_tau` | 0.9839 | 0.9863 | 0.19287 |
| exact integer accuracy | 0.0913 | 0.1035 | 0.00398 |
| token accuracy | 0.6934 | 0.6995 | 0.44622 |
| decode failure rate | 0.0000 | 0.0000 | — |

**Exact integer accuracy ≈ 0.10 and exact secret recovery are different
claims.** Recovery does not need per-sample perfection; it needs probe responses
to fall on the correct side of the anchor.

## 4. The central finding

| model | parameters | n=12 acc_tau | n=12 exact recovery |
|---|---:|---:|:---:|
| V1 Salsa2-GatedUT | 4,131,200 | 0.3550 | **NO** (0/10 K) |
| **Modified NACT** | **4,238,208** | **0.9839** | **YES** (8/10 K) |

The parameter counts differ by **2.6%**, and V1's loss (1.7413) sits essentially
at the marginals-only baseline of 1.86498 — at this budget it has largely not
learned secret-related structure. On the probes, V1 left **58% of outputs
undecodable** and answered a single constant on 8 of 10 multipliers.

> **The evidence points to representation, not capacity, as the principal
> observed source of improvement.**

### Why: the sparse-input diagnosis

Feeding the frozen V1 model inputs of decreasing density, scored against the
best input-blind constant predictor:

| non-zero coordinates | 12 | 10 | 8 | 6 | 4 | 2 | 1 | `K·e_i` probes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **V1** lift | +0.1309 | +0.0991 | +0.0718 | −0.0527 | −0.1787 | −0.3730 | −0.4800 | **−0.4417** |
| **Modified NACT** lift | +0.7651 | +0.7393 | +0.7222 | +0.5981 | +0.4346 | +0.1895 | +0.0562 | **+0.0500** |

Spearman ρ = 1.000 against sparsity, exact permutation p = 0.0004. V1 crosses
zero between 8 and 6 non-zero coordinates — below that its output carries no
usable information about its input. The recovery probes sit at the far end of
the *same curve*, not on a separate cliff. **Modified NACT never crosses zero**,
and holds probe decode validity 1.000 where V1 collapses to 0.417.

### The architectural change

| | V1 / original | Modified NACT |
|---|---|---|
| tokens per coordinate | 2 digit tokens | **1 coordinate token** |
| encoder sequence | `2n + 2` | **`n + 2`** |
| coordinate identity | RoPE (relative only) | learned **absolute** embedding |

`b = Σ aᵢ sᵢ` binds coordinate `i` to bit `sᵢ`, so the model needs to know
*which* coordinate it is reading. With one token per coordinate, position *is*
coordinate index. The coordinate table is a fixed `128 × 512` matrix, so **the
parameter count does not change with `n`**.

### The ablation that produced the final model

Modified NACT is full NACT with six components removed — centered residue, both
Fourier features, zero indicator, zero vector, sparse attention bias. At matched
depth, seed and budget, every difference fell **inside NACT's own seed noise**:

| metric | NACT | Modified NACT | difference | 3σ seed band | resolvable? |
|---|---:|---:|---:|---:|:---:|
| validation loss | 0.9388 | 0.9891 | +0.0503 | 0.0696 | no |
| `acc_tau` | 0.9839 | 0.9839 | +0.0000 | 0.0064 | no |
| exact integer accuracy | 0.1011 | 0.0913 | −0.0098 | 0.0275 | no |

Modified NACT retained **100%** of the V1→NACT `acc_tau` gap. The removed
features were removable **without measurable loss in the tested setting** — one
seed, n=12 — which is not a claim that they are universally useless.

## 5. Methodology — how self-deception was avoided

This mattered more than any single result, because three failure modes were live
throughout: sparse secrets make bad candidates look good, in-distribution
accuracy does not imply recovery, and the released reference resolves candidate
polarity *against the true secret*.

**Three stages, structurally isolated:**

| stage | what it may see |
|---|---|
| recovery | public probes and model predictions only — `DirectRecovery` has **no parameter** through which a secret could pass |
| verification | `(A, b, candidate, q, sigma)` only |
| evaluation | the only stage that loads the secret, and it runs **last** |

`evaluate=False` runs the whole attack with no ground truth at all. Tests scan
the recovery package's source text for any reference to ground truth.

**Baselines computed exactly, not assumed.** Chance token accuracy is 0.44622
for this representation — near-45% token accuracy is free, and was never read as
learning.

**A defect found and reported, not patched away.** At n=20 the single-K
`best_margin` rule selected `K = 1`, a negative control: when the two hypotheses
are adjacent, a model answering a constant scores the *maximum* possible margin.
The default is now the separation-weighted `aggregate` vote, and `best_margin`
is guarded by `min_separation = floor(sigma) + 1`, derived from the noise scale
rather than chosen by hand. Both rules remain secret-free.

## 6. Reproducibility

```bash
python scripts/run_pipeline.py configs/pipeline_n12.yaml
python scripts/run_pipeline.py configs/pipeline_n20.yaml
pytest -q
```

Neither pipeline command trains; each takes about **one second** against the
stored checkpoints.

- **Replay agreement: 12/12 checks** match the independently recorded results.
- **Tests: 690 passed, 2 skipped.**
- **Multi-secret:** at n=12, three checkpoints trained under different seeds —
  three different secrets — gave **3/3 exact recovery, 3/3 verification PASS**.
- Every experiment runs from a config file plus a fixed seed; checkpoints carry
  a config fingerprint and the pipeline refuses to load a mismatched one.
- Training cost, for reference: ~1,146 s (n=12) and ~1,424 s (n=20) on a laptop
  CPU, 100,032 samples each.

## 7. What is demonstrated — and what is not

### Demonstrated

- Neural prediction of `b` well above every chance baseline, at n=12 and n=20.
- **Exact secret recovery** at both dimensions.
- **Independent residual verification** at both dimensions.
- A ~12× smaller model than published SALSA completing the full pipeline on CPU.
- Recovery reproduced across three secrets at n=12.

### Not demonstrated

- **n=30, n=50, n=128** — never trained or tested. No model exists at those
  dimensions; `results/archive/n30_prediction/` holds an *extrapolation*, not an
  experiment.
- **A practical LWE or RLWE break.** Nothing here speaks to deployed parameters.
- **A scaling law** — two dimensions is not a trend.
- **Component attribution** — ablation F removed six features at once; variants
  C, D and E were designed but never run.
- **Multi-seed validation at n=20** — one secret, one seed.
- Only `h = 2` was demonstrated end to end; the distinguisher recovery mode was
  never implemented; the `min_separation` guard is validated at `sigma = 3` only.

## 8. Next steps

None of these has been performed.

1. **Dimension scaling** to n=30, n=50, n=128 — the parameter count is invariant
   in `n`, so the sample requirement, not the architecture, is the expected
   binding constraint.
2. **Multi-seed validation**, so recovery becomes a rate with an uncertainty
   estimate rather than a single outcome.
3. **Remaining ablations** (C, D, E) to attribute the residual sparse-input
   margin that Modified NACT narrowed without changing the outcome.
4. **Larger Hamming weights** and other `q`/`sigma` settings.

---

## Appendix — where the evidence lives

| claim | artifact |
|---|---|
| Comparison against released SALSA source | `results/original_fidelity_audit/` |
| V1 sparse-input collapse | `results/recovery_generalization/` |
| V1 vs NACT at equal depth | `results/equal_depth_ablation/` |
| The ablation producing Modified NACT | `results/nact_ablation_F/` |
| n=12 recovery and verification | `results/nact_ablation_F_recovery/`, `results/v2_verification/` |
| Multi-secret reproduction | `results/v2_recovery_robustness/` |
| n=20 training and recovery | `results/n20_nact_f_pilot/`, `results/n20_nact_f_recovery/` |
| Pipeline replay | `results/final_pipeline/pipeline_report.md` |
| Full technical report | `results/final_pipeline/SALSA2_Final_Report.md` |
| Index of what is authoritative | `results/README.md` |

**Naming.** Directories and files containing `nact_f` / `NACT-F` are the
historical internal identifiers for **Modified NACT**, retained to preserve
provenance.

**Model lineage.** Original SALSA → Salsa2-GatedUT (V1, 4,131,200) → NACT
(4,241,288) → **Modified NACT (4,238,208, final)**.

**No training was performed in producing this report.**
