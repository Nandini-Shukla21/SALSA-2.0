# Modified SALSA — Evidence Audit

Every substantive numerical claim in `Modified_SALSA_Research_Paper.pdf` mapped
to the file, log or published source that supports it.

Paths are relative to the repository root. "Replayed" means the figure was
reproduced during preparation of the manuscript by loading a stored checkpoint
and re-running inference; no training was performed for this paper.

Prepared: 14 October 2026.

---

## 0. Provenance classes used in this audit

| class | meaning |
|---|---|
| **M** | Measured in this project and read from a recorded artifact |
| **R** | Replayed during manuscript preparation (artifact re-confirmed) |
| **N** | Newly measured during manuscript preparation (no training) |
| **C** | Computed from the code, a configuration, or closed-form arithmetic |
| **L** | Reported in published literature, cited, not reproduced here |
| **X** | Explicitly declined — marked *not reliably estimable* in the paper |

---

## 1. Model and parameter counts

| Claim in the paper | Value | Class | Source |
|---|---|:--:|---|
| Modified NACT trainable parameters | 4,238,208 | M | `results/final_pipeline/final_summary.json` → `parameter_count`; also both training summaries |
| Full NACT trainable parameters | 4,241,288 | M | `results/nact_v2/architecture_report.json` → `parameter_verification.v2` |
| Baseline (Salsa2-GatedUT) parameters | 4,131,200 | M | same file → `parameter_verification.v1` |
| Plain transformer control | 4,251,520 | M | `results/archive/pilot_control_n30_r/pilot/artifacts/summary.json` |
| Counts verified three independent ways, in agreement | `all_three_agree: true`, `unclassified_parameters: []` | M | `results/nact_v2/architecture_report.json` |
| Component breakdown (Table II) | encoder body 3,672,064; decoder 404,736; output 10,880; front end 150,528 | C | summed from `parameter_verification.v2.breakdown`; the figure script asserts the sum equals 4,238,208 |
| Ablation removes 3,080 parameters (0.073 %) | 2,560 + 512 + 8 | C | same breakdown |
| Parameter count invariant in *n* | identical at n = 12, 20, 30, 50, 70, 90, 128 | M | `architecture_report.json` → `n_invariance`, `n_does_not_change_count: true` |
| Invariant in loop counts | encoder loops {1,2,4,8,16}, decoder loops {1,2,4} | M | same file → `encoder_loop_invariance`, `decoder_loop_invariance` |
| Model configuration (d_e 512 / d_d 128, 8/4 heads, T_e = T_d = 2, FFN ×4, gated, RoPE, RMSNorm) | — | C | `configs/target_4_5m.yaml`, `configs/nact_n12_h2_te2.yaml`, and the recorded `metadata.json` of each run |
| Coordinate table is 128 × 512; hard ceiling at n = 128 | `max_coordinates: int = 128`; `if n > spec.max_coordinates: raise ValueError` | C | `salsa/models/nact.py` |
| Only rows 0…n−1 receive gradient | `embedded + self.coordinate_embedding[:n]` | C | `salsa/models/nact.py`, `NactFrontEnd.forward` |

---

## 2. Instance and training configuration

| Claim | Value | Class | Source |
|---|---|:--:|---|
| q, σ, h, τ | 251, 3.0, 2, 0.1 | M | `metadata.json` → `config.lwe`, `config.evaluation` in both run dirs |
| Structure | RLWE, circulant | M | same |
| Encoding | base 81, width 2, LSB-first, no separator, bos/eos | M | same → `config.encoding` |
| Vocabulary 85 | 4 specials + 81 digits | C | `salsa/data/encoding.py`; `SPECIAL_TOKENS` plus base |
| Encoder sequence length n+2 | 14 at n = 12, 22 at n = 20 | M | `results/final_pipeline/pipeline_report.json` → `instance.encoder_sequence_length` |
| Optimiser AdamW, lr 5e-4, wd 0.01, β (0.9, 0.98), ε 1e-9 | — | M | `metadata.json` → `config.training` |
| Warmup 200, cosine to 1e-6, grad clip 1.0 | — | M | same |
| Batch 64 × 79 steps × 20 epochs | — | M | `artifacts/summary.json` → `plan` |
| Sample budget 100,000 configured → 100,032 seen | — | M | `summary.json` → `plan.sample_budget`, `state.samples_seen` |
| Sample reuse = 1 | `sample_reuse: 1` | M | `metadata.json` → `config.lwe.sample_reuse` |
| Validation set 2,048 | — | M | `summary.json` → `plan.valid_samples` |
| Seed 0, deterministic | — | M | `metadata.json` → `config.experiment` |
| Target is 3 tokens regardless of n | `tokens_seen = 300096 = 3 × 100032` at both dimensions | M | both `summary.json` → `state.tokens_seen` |
| Hardware: AMD64 12 logical cores, 6 torch threads, 15.33 GB RAM, no CUDA | — | M | `metadata.json` → `device_report` |
| Software: Python 3.10.11, PyTorch 2.13.0+cpu, NumPy 2.2.6 | — | M | `metadata.json` → `packages` |
| Training commit for the n = 12 run | `018a8a8` | M | `metadata.json` → `git_commit`; confirmed present in local history |

---

## 3. Chance baselines

All are n-invariant because *b* is uniform on **Z**_q for every instance tested.

| Quantity | Value | Class | Source |
|---|---|:--:|---|
| Uniform-prediction loss | 4.442651 nats | C | `summary.json` → `chance_baselines.uniform_loss_nats` |
| Marginals-only loss | 1.864984 nats | C | same → `marginal_loss_nats` |
| Chance acc_τ | 0.192870 | C | same → `chance_acc_tau` |
| Chance exact integer accuracy | 0.003984 | C | same → `chance_exact_accuracy` |
| Chance token accuracy | 0.446215 | C | same → `chance_token_accuracy` |
| 3σ bars on 2,048 sequences (acc_τ > 0.21903, exact > 0.00816) | — | C | `results/dimension_scaling/dimension_scaling.json` → `shared_thresholds.three_sigma_bar`; pre-registered in `configs/nact_n20_h2_te2_F.yaml` before the n = 20 run |

---

## 4. Modified NACT observed results (Table V)

Source for n = 12: `results/nact_ablation_F/nact_n12_h2_te2_F/seed_0/artifacts/summary.json`
Source for n = 20: `results/n20_nact_f_pilot/nact_n20_h2_te2_F/seed_0/artifacts/summary.json`

| Quantity | n = 12 | n = 20 | Class |
|---|---:|---:|:--:|
| Validation loss | 0.989069 | 0.976393 | M |
| acc_τ | 0.983887 | 0.986328 | M |
| Exact integer accuracy | 0.091309 | 0.103516 | M |
| Token accuracy | 0.693359 | 0.699544 | M |
| Mean \|b − b̂\| | 7.1465 | 6.5093 | M |
| Decode failure rate | 0.0 | 0.0 | M |
| Best epoch | 19 | 19 | M |
| Wall-clock (s) | 1,146.30 | 1,424.24 | M |
| Throughput (samples/s) | 108.902 | 87.428 | M |
| Peak resident memory (MB) | 446.83 | 503.66 | M |

Recovery and verification, source `results/final_pipeline/pipeline_report.json`,
**re-executed during manuscript preparation** (`python scripts/run_pipeline.py
configs/pipeline_n12.yaml` and `…_n20.yaml`):

| Quantity | n = 12 | n = 20 | Class |
|---|---:|---:|:--:|
| Exact recovery | yes | yes | R |
| Coordinate accuracy / Hamming distance | 1.0 / 0 | 1.0 / 0 | R |
| Multipliers recovering alone | 8 of 10 probed = 8 of 8 eligible | same | R |
| Excluded by the σ-derived guard | K = 1, K = 250 | same | R |
| Probe decode validity | 1.000 | 1.000 | R |
| Residual std, candidate | 2.998 | 2.941 | R |
| Residual std, all-zeros | 71.803 | 72.477 | R |
| Fraction within 3σ | 0.9980 | 0.9985 | R |
| Criteria C1–C4 | all pass | all pass | R |
| Replay wall-clock | 1.09 s | 1.39 s | R |
| Verification split / data seed | `pipeline_verify_n12` / 2156377967 | `pipeline_verify_n20` / 276041130 | R |

> **Note on multiple verification runs.** Phase 24 and phase 26 recorded
> verification on different data seeds (`phase24_verify_nact_f`,
> `phase26_verify_n20`) and obtained residual std 2.967 and 2.982. The paper
> quotes the pipeline-replay figures (2.998, 2.941) consistently, because
> those are the ones reproduced during manuscript preparation. All runs agree
> to within 0.06 and all pass the same four criteria.

---

## 5. Comparison with the baseline (Table VI)

| Quantity | Value | Class | Source |
|---|---|:--:|---|
| V1 at n = 12, seed 0: loss 1.741287, acc_τ 0.35498, exact 0.007812, token 0.5324 | — | M | `results/nact_ablation_F/ablation_F.json` → `comparison`; originally from the phase-16 audit snapshot in `results/v1_v2_audit/` |
| V1 probe decode validity 0.4167; probe lift −0.4417; 0 of 10 K exact | — | M | same `comparison` block |
| V1 answered a single constant on 8 of 10 multipliers; 58 % of probe outputs undecodable | — | M | `results/n20_nact_f_recovery/n20_nact_f_recovery.json` → `v1_n20_comparison.v1_probe_behaviour_at_n12` |
| Equal-depth multi-seed means (V1 5 seeds / NACT 3 seeds) | loss 1.8096 ± 0.0469 vs 0.9136 ± 0.0232; acc_τ 0.2232 ± 0.1197 vs 0.9863 ± 0.0021; exact 0.0060 ± 0.0035 vs 0.1110 ± 0.0092; token 0.4784 ± 0.0408 vs 0.7014 ± 0.0039; mean \|b−b̂\| 87.30 ± 28.36 vs 6.00 ± 0.59 | M | `results/equal_depth_ablation/equal_depth_ablation_interim.{json,md}` |
| Only `model.arch` differs between the arms | field-by-field parity check | M | same file, §1 |
| Sign test floors at p = 0.25 with 3 pairs | — | C | same file, §3; exact two-sided sign test |
| 2 of 5 planned NACT seeds never completed | seeds 456 (interrupted), 789 (never started) | M | same file, §2 |

> **Correction recorded.** An earlier project document attributed the
> three-seed block to Modified NACT. The artifact records
> `parameter_count = 4241288` for those checkpoints, i.e. **full NACT**. The
> paper attributes it correctly and the table carries an explicit note.

---

## 6. Ablation F (Table VII)

Source: `results/nact_ablation_F/ablation_F.{json,md}`

| Quantity | Value | Class |
|---|---|:--:|
| Full NACT vs Modified NACT: loss 0.9388 → 0.9891 (+0.0503), 3σ band 0.0696 | — | M |
| acc_τ 0.9839 → 0.9839 (+0.0000), band 0.0063 | — | M |
| Exact 0.1011 → 0.0913 (−0.0098), band 0.0276 | — | M |
| Every difference inside the band | — | C |
| Gap retention vs V1: loss 93.7 %, acc_τ 100.0 %, exact 89.5 %, token 97.8 % | — | M |
| One configuration difference: `nact_variant: full → one_token_only` | — | M | field-by-field diff recorded in `ablation_F.json` → `control_parity` |
| Six components removed simultaneously; variants C, D, E never run | — | M | `ablation_F.md`, "NOT ESTABLISHED" |

The 3σ band is full NACT's own three-seed spread from the phase-17 runs.

---

## 7. Sparse-input lift (Figure 4b)

Source: `results/nact_ablation_F/ablation_F.json` → `sparsity`, keyed `v1`,
`full`, `f`.

The artifact carries nine rows per model: a `training_distribution` row at
nnz = 12, seven `sparse_random` levels, and one `probe_K_times_e_i` row. The
figure and the paper use the eight non-`training_distribution` rows. Values
(lift = acc_τ minus best input-blind constant):

| nnz | V1 | full NACT | Modified NACT |
|---:|---:|---:|---:|
| 12 | +0.1309 | +0.7642 | +0.7651 |
| 10 | +0.0991 | +0.7402 | +0.7393 |
| 8 | +0.0718 | +0.7231 | +0.7222 |
| 6 | −0.0527 | +0.5879 | +0.5981 |
| 4 | −0.1787 | +0.4243 | +0.4346 |
| 2 | −0.3730 | +0.2158 | +0.1895 |
| 1 | −0.4800 | +0.1196 | +0.0562 |
| probes | −0.4417 | +0.1167 | +0.0500 |

Decode validity on the probes: V1 0.417, both NACT variants 1.000. **M**

Probe-vs-training distribution statistics (M, from
`results/original_fidelity_audit/fidelity_audit.json` →
`training_distribution_vs_probes`): mean zero coordinates per training row
0.04808; maximum seen in 200,000 rows 3; token-level zero-digit fraction 0.1558
training vs 0.8462 probe; probability a training row looks like a probe
4.80 × 10⁻²⁶.

**Limitation recorded in the paper:** single checkpoint per arm; seed
variability of the lift curve is unmeasured.

---

## 8. The classical enumeration baseline (Table VIII) — newly measured

**Class N.** Measured during manuscript preparation by a standalone NumPy
script; no model is loaded and nothing is trained. The scoring rule is the
project's own acceptance criterion C1 (centred residual std ≤ 1.5σ) applied to
2,048 fresh samples per instance, with data generated by
`salsa.data.rlwe.generate_rlwe_matrix`, `generate_binary_secret` and
`generate_error` at q = 251, σ = 3, circulant.

| n | h | candidates | wall-clock | instances | uniquely correct | false accepts |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 2 | 66 | 0.012 s | 1 | 1/1 | 0 |
| 20 | 2 | 190 | 0.022 s | 1 | 1/1 | 0 |
| 30 | 2 | 435 | 0.050 s | 1 | 1/1 | 0 |
| 50 | 2 | 1,225 | 0.144 s | 20 | 20/20 | 0 of 24,480 |
| 128 | 2 | 8,128 | 0.988 s | 20 | 20/20 | 0 of 162,540 |
| 128 | 3 | 341,376 | 193.6 s | 1 | 1/1 | 0 |

Observed margin: the correct candidate's residual std was 2.93–3.03 and the
second-smallest was 68.6–71.2 in every sweep.

Stated caveats, reproduced in the paper: enumeration assumes *h* is known; the
false-acceptance rate is a property of the residual test at 2,048 samples and
q = 251 and would need re-measuring at much larger candidate counts or smaller
sample sizes.

Reproduction scripts are in the session scratchpad rather than the repository;
they are 40 lines of NumPy and are described in full in the paper. **They
should be committed to the repository before submission** — see the review
checklist.

---

## 9. Negative results reported

| Claim | Value | Class | Source |
|---|---|:--:|---|
| GatedUT at n = 30, h = 3, 100,032 samples: loss 1.842615, acc_τ 0.134277, exact 0.002930 | at/below chance | M | `results/pilot_gatedut_n30_r/pilot/artifacts/summary.json` |
| Plain transformer at n = 30, h = 3: loss 1.843147, acc_τ 0.147949, exact 0.002441 | at/below chance | M | `results/archive/pilot_control_n30_r/pilot/artifacts/summary.json` |
| V1 at n = 20, h = 2 on 200,000 samples: loss 1.823288, acc_τ 0.234375, exact 0.003418 | exact below its 0.003984 chance level | M | `results/control_b_n20_h2/control/artifacts/summary.json` |
| V1 recovery at n = 20 was never run | "not measured", not "failed" | M | `results/n20_nact_f_recovery/n20_nact_f_recovery.json` → `v1_n20_comparison.note` |

> **Correction recorded.** `PROJECT_REPORT.md` §7 stated that n = 30 was
> "never trained or tested. No model exists at those dimensions." Two models
> of comparable size *were* trained at n = 30 and both failed. The paper
> reports them as negative results and states that neither used the
> coordinate-token representation and both used h = 3.

---

## 10. Cost and scalability

| Claim | Value | Class | Source |
|---|---|:--:|---|
| Forward benchmark, same session, batch 32, 5 repeats, representation R | n = 30: V1 91.7 vs NACT T_e=2 150.1 samples/s (1.64×); n = 128: 20.0 vs 38.9 (1.95×) | M | `results/nact_v2/architecture_report.json` → `forward_benchmark` |
| FFN term dominates attention across the range | d_e² = 262,144 vs L² = 16,900 at n = 128 | C | arithmetic from d_e = 512, L = 130 |
| Cost model ms/sample ≈ 5.24 + 0.282 L | fitted through the two measured Modified NACT points | C | `scripts/scalability_forecast.py`, `empirical_cost_model` |
| Relative compute per 100k samples: 1.25× / 1.74× / 2.23× / 2.73× / 3.22× / 3.66× at n = 30…128 | extrapolation | C | same script; labelled a floor, not an estimate |
| V1 series visibly super-linear in L | 0.30 ms/token L = 26→42; 0.56 ms/token L = 42→62 | C | derived from `results/dimension_scaling/dimension_scaling.csv` throughputs |
| Peak memory: 447 MB (n = 12), 504 MB (n = 20), 777 MB (V1 at n = 30) | — | M | respective `summary.json` → `cpu_memory_mb.peak_rss_mb` |
| Independent generator polynomials fall from 8,336 (n = 12) to 782 (n = 128) at a 100,032-sample budget | ⌈m/n⌉ | C | `salsa/data/rlwe.py`, `generate_rlwe_matrix` |
| Published sample requirement at n = 30, h = 3: 2²¹·⁹ to 2²⁴·⁸ | 39–292× this project's per-run budget | L | SALSA Table 2 @ Wenger et al. 2022 |
| Loss / acc_τ / recovery at n ≥ 30 | **not reliably estimable** | X | declined; no basis in the evidence |

---

## 11. Fidelity audit (Table IV)

Source: `results/original_fidelity_audit/fidelity_audit.{json,md}`, a
stage-by-stage comparison against the released reference implementation.
Tally: 5 exact, 4 intentional improvements, 3 harmless differences, 3
potentially behaviour-changing, 2 major incompatibilities.

| Difference reported in the paper | Class | Notes |
|---|:--:|---|
| Sample reuse ~10× vs 1× | M | Audit stage 1b |
| Reference matrices are negacyclic despite the paper's "circulant" wording | M | Audit stage 4 |
| Reference reduces K mod B^w, not mod q | M | Audit stage 13 |
| Reference resolves candidate polarity against the true secret | M | Audit stage 14 |
| ~51 M vs 4.24 M parameters | M/L | Audit stage 9; 51 M is the published figure |
| Residual verification absent from the released evaluator | M | Audit stage 16 |
| Error distribution TVD 0.00219, std 3.0139 vs 3.0000 | M | Audit stage 3 |

> **Caveat on the audit's own currency.** The audit was written at project
> phase 10, before residual verification was implemented here; its stage-16
> row says verification is absent from *both*. Verification was implemented
> subsequently (`salsa/verification/residual.py`) and the paper states the
> current position. The audit also notes that the supplied reference checkout
> had the Universal Transformer looping and the copy gate **disabled**, so it
> is not the paper's architecture either.

---

## 12. Repository claims

| Claim | Verified how | Class |
|---|---|:--:|
| Repository is public at `github.com/Nandini-Shukla21/SALSA-2.0` | fetched 14 Oct 2026; page renders, labelled Public | N |
| 236 commits on the default branch | shown on the repository page; matches `git log --oneline \| wc -l` locally | N/M |
| 328 tracked files | `git ls-files \| wc -l` | M |
| Checkpoints are **not** published | `.gitignore` lists `*.pt`; `git ls-files \| grep -c '\.pt$'` returns 0 | M |
| No LICENSE file although `pyproject.toml` declares MIT | `git ls-files \| grep -i licen` returns nothing; `pyproject.toml` line `license = { text = "MIT" }` | M |
| Test suite: 690 passed, 2 skipped | `pytest -q` re-run during manuscript preparation, exit code 0, 692 results with 2 skips | R |
| Retraining time quoted (≈19 min at n = 12, ≈24 min at n = 20) | from recorded `state.elapsed_seconds` 1,146.3 and 1,424.2 | C |

---

## 13. Citations

All twenty-two bibliography entries were checked against a primary source
during manuscript preparation. No DOI, author list, venue or year was supplied
from memory.

| Key | How verified |
|---|---|
| `wenger2022salsa` | The paper PDF supplied with the task; NeurIPS 2022 stated on page 1 |
| `regev2005` | Cited as [56] in the SALSA bibliography; STOC 2005 |
| `lpr2010` | IACR ePrint 2012/230 fetched; states "Full version of paper appearing in Eurocrypt 2010" |
| `li2023picante` | Author list, CCS 2023 venue, pages 2606–2620 and DOI 10.1145/3576915.3623076 confirmed by search of arXiv 2303.04178, ePrint 2023/340 and the CWI record |
| `li2023verde` | Author list and NeurIPS 2023 venue confirmed; arXiv 2306.11641, ePrint 2023/968 |
| `vaswani2017`, `dehghani2019`, `csordas2022`, `charton2022` | Cited as [67], [27], [24], [17] in the SALSA bibliography |
| `su2024rope` | arXiv 2104.09864 confirmed; author list and title checked |
| `zhang2019rmsnorm` | arXiv 1910.07467, NeurIPS 2019 poster page confirmed |
| `hendrycks2016gelu` | arXiv 1606.08415 confirmed |
| `loshchilov2019adamw` | arXiv 1711.05101, ICLR 2019 confirmed |
| `albrecht2021hestandard`, `albrecht2017dual`, `chen2020concrete`, `curtis2019sparse`, `chen2011bkz`, `lll1982`, `gohr2019` | Cited as [4], [6], [20], [25], [22], [43], [30] in the SALSA bibliography |
| `fips203` | csrc.nist.gov/pubs/fips/203/final fetched: title, number and date 13 August 2024 confirmed, together with the Module-LWE basis |
| `salsa_code` | URL given in the SALSA paper, §1 |

Two bibliographic details were **deliberately omitted** rather than asserted
without verification: the journal (JACM) version of `lpr2010`, and page or
volume numbers for several conference entries.

---

## 14. Claims the paper declines to make

| Not claimed | Why |
|---|---|
| A recovery *rate* or success probability for Modified NACT | One secret and one seed per dimension |
| Any scaling law | Two points |
| That the removed ablation components contribute nothing | One seed cannot establish a null |
| That the baseline fails at n = 20 | Recovery was never run against a baseline checkpoint there |
| That dimension is irrelevant at fixed h | C(n,2) grows quadratically; the paper says so explicitly |
| Statistical significance for the equal-depth comparison | Three pairs floor the two-sided sign test at p = 0.25 |
| Any statement about deployed LWE parameter security | Out of scope and unsupported |
| DRDO or SAG endorsement | Not held |
| Loss, acc_τ or recovery outcome at any n ≥ 30 | Marked *not reliably estimable* |
