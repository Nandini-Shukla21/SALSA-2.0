# SALSA 2.0 — project status

A reviewer should be able to read this page and know exactly what has been
done, what has been measured, and what has not been attempted.

---

## COMPLETED

Engineering and methodology that is finished and tested.

- **Data layer.** LWE and RLWE generation (circulant and negacyclic), exact
  Hamming-weight binary secrets, discrete Gaussian errors, index-seeded
  reproducible batch streams. Public and private data are separated by type:
  `LWESample` has no secret field.
- **Encoding.** `LatticeCodec` at base 81, lsb-first, fixed width, with and
  without separators. Verified token-for-token against the original SALSA
  `encoders.py` by executing it.
- **Models.** V1 Salsa2-GatedUT (4,131,200), V2 Salsa2-NACT (4,241,288) and
  NACT-F (4,238,208). Every count verified three independent ways — actual,
  component breakdown, analytical formula — with nothing unclassified and no
  padding parameters.
- **Training pipeline.** CPU-first trainer with alignment verification,
  checkpointing with strict config-fingerprint checks, and resume.
- **Direct secret recovery.** SALSA Algorithm 1 with a secret-free anchored
  decision rule, replacing the released code's polarity resolution against
  ground truth.
- **Recovery rule hardening.** Two documented secret-free rules; the default
  `aggregate` vote and a guarded `best_margin` diagnostic. The guard fixes a
  measured degeneracy at low separation.
- **Independent residual verification.** Implements the paper's §4.4 test,
  which is absent from the released source. Receives only
  `(A, b, candidate, q, sigma)`.
- **End-to-end pipeline.** One entry point, `run_salsa2_pipeline`, with the
  three stages held apart and the ground truth loaded last.
- **Test suite.** 687 passed, 2 skipped.
- **Fidelity audit** against the released SALSA source, documenting every
  difference and classifying it.

---

## DEMONSTRATED

Experimentally measured, with artifacts in `results/`.

| | n=12, h=2 | n=20, h=2 |
|---|---:|---:|
| search space C(n,h) | 66 | 190 |
| training samples | 100,032 | 100,032 |
| validation loss | 0.9891 | 0.9764 |
| acc_tau | 0.9839 | 0.9863 |
| exact integer accuracy | 0.0913 | 0.1035 |
| probe decode validity | 1.0000 | 1.0000 |
| coordinate accuracy | 1.0000 | 1.0000 |
| Hamming distance | 0 | 0 |
| successful K | 8/10 | 8/10 |
| residual std (verification) | 2.998 | 2.941 |
| **exact secret recovery** | **YES** | **YES** |
| **residual verification** | **PASS** | **PASS** |

Additional demonstrated findings:

- **V1 fails where NACT-F succeeds at n=12.** V1: acc_tau 0.3550, exact secret recovery NO, 0/10 successful K, probe decode validity 0.417.
- **Sparse-input generalization.** V1's lift over the best input-blind
  constant crosses zero between nnz=8 and nnz=6 and reaches -0.4417 on the probes; NACT-F stays positive throughout, at +0.0500.
- **Recovery reproduces across secrets at n=12.** 3/3 secrets recovered exactly and verified, using three checkpoints trained under
  different seeds.
- **Component attribution.** The one-token representation carries the
  improvement; removing all six numerical/zero-aware features changed nothing
  measurable at one seed.

---

## NOT YET TESTED

Nothing below has been run. No partial evidence exists for any of it.

- **n=30** — never trained. Estimated cost is known; no checkpoint exists.
- **n=50** — never trained.
- **n=128** — never trained. The architecture accepts it (parameter count is
  invariant in `n`) and forward cost has been benchmarked, but no model has
  been fitted at that dimension.
- **Multi-seed validation at n=20** — one seed, one secret only. Seed-to-seed
  variance at n=20 is unmeasured.
- **Ablations C, D and E** — designed and costed in the component-attribution
  study, never executed. Only ablation F was run.
- **Distinguisher recovery** — the second SALSA recovery mode; not implemented.
- **Knowledge distillation and quantization** — considered, never attempted.
- **Larger Hamming weights** — only h=2 has been demonstrated end to end.

---

## FUTURE WORK

Reasonable next experiments, none performed.

1. **Dimension scaling.** Train NACT-F at n=30, then n=50, holding the
   protocol fixed. The binding constraint is expected to be the sample
   requirement rather than the architecture, since the parameter count does
   not change with `n`.
2. **Multi-seed recovery rates.** Several secrets per dimension, so recovery
   can be reported as a rate with an uncertainty estimate instead of a single
   outcome.
3. **Complete the ablation matrix.** Variants C, D and E would attribute the
   residual sparse-input margin, which NACT-F narrowed to roughly half of full
   NACT's without changing the recovery outcome.
4. **Scalability analysis.** Measure how the sample requirement grows with `n`
   and `h`, and whether the one-token representation's advantage holds as the
   sequence lengthens toward n=128.
5. **Harden the guard across sigma.** `min_separation = floor(sigma) + 1` has
   been validated at sigma=3 only.

---

## What this project does NOT claim

- Not a practical break of LWE or RLWE.
- No result at any cryptographically meaningful dimension.
- No claim about deployed parameter sets.
- No general cryptanalytic success.

The demonstrated instances have C(12,2) = 66 and
C(20,2) = 190 possible secrets. These are diagnostic
scales chosen so the whole pipeline could be exercised end to end on a CPU.
