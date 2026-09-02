# Phase 17 Equal-Depth Ablation — Pilot / Interim Result

> **This is NOT a 5-seed final result.** The seed sweep was stopped after three
> of five V2 seeds. Everything below is a pilot reading of partial data.

Read-only: nothing was trained to produce this report, no evaluation was re-run,
no code was modified and no result was deleted.

## What was being asked

Phase 16 showed a large V2 advantage, but V2 ran at `encoder_loops=4` against
V1's 2, confounding architecture with effective depth. This phase holds depth
equal so `model.arch` is the only thing that varies.

## 1. Design and parity

| | V1 GatedUT | V2 NACT |
|---|---|---|
| effective depth | T_e=2, T_d=2 | T_e=2, T_d=2 |
| parameters | 4,131,200 | 4,241,288 (+110,088) |
| samples/seed | 100,032 | 100,032 |

A field-by-field parity check confirmed **`model.arch` is the only configuration
difference** — field-by-field: n, h, q, sigma, structure, encoding, seed, optimizer, learning rate, scheduler, batch size, validation size, sample budget, evaluation cadence and tau all identical. Both arms share the same secret and data stream.

V1 was **not** retrained. Its five seeds come from the preserved phase-16 audit
snapshot (`results/v1_v2_audit/v1_v2_comparison.json`), the surviving record after
the phase-16 run directories were emptied.

## 2. Seed status — the central limitation

| arm | seeds available |
|---|---|
| V1 GatedUT T_e=2 | [0, 42, 123, 456, 789] (5) |
| **V2 NACT T_e=2** | **[0, 42, 123] (3 of 5)** |

Seed 456 was interrupted mid-run and is excluded; seed 789 never started. Neither
was deleted. Paired statistics use only the 3 seeds present in both arms: [0, 42, 123].

## 3. Learning metrics

| Metric | V1 GatedUT T_e=2 (5 seeds) | V2 NACT T_e=2 (3 seeds) | Difference | Winner |
|---|---:|---:|---:|:---:|
| Validation loss (nats) | 1.8096 ± 0.0469 | 0.9136 ± 0.0232 | -0.8960 | **V2** |
| SALSA acc_tau | 0.2232 ± 0.1197 | 0.9863 ± 0.0021 | +0.7631 | **V2** |
| Exact integer accuracy | 0.0060 ± 0.0035 | 0.1110 ± 0.0092 | +0.1050 | **V2** |
| Token accuracy | 0.4784 ± 0.0408 | 0.7014 ± 0.0039 | +0.2230 | **V2** |
| Greedy token accuracy | 0.4517 ± 0.0141 | 0.6876 ± 0.0048 | +0.2359 | **V2** |
| Perfect accuracy | 0.0060 ± 0.0035 | 0.1110 ± 0.0092 | +0.1050 | **V2** |
| Mean |b - b_hat| | 87.3035 ± 28.3630 | 6.0026 ± 0.5938 | -81.3009 | **V2** |
| Decode failure rate | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | +0.0000 | **TIE** |
| Samples/sec | 360.5546 ± 15.3064 | 108.2173 ± 10.7518 | -252.3373 | **V1** |
| Tokens/sec | 1,081.6400 ± 45.9109 | 324.6667 ± 32.2308 | -756.9733 | **V1** |
| Wall-clock (s) | 353.0878 ± 15.5481 | 1,200.8050 ± 113.2378 | +847.7172 | **V1** |
| CPU RSS (MiB) | 712.3594 ± 102.1490 | 375.7604 ± 13.6340 | -336.5990 | **V2** |
| Trainable parameters | 4,131,200.0000 ± 0.0000 | 4,241,288.0000 ± 0.0000 | +110,088.0000 | **V1** |

Chance: acc_tau 0.19287, exact 0.00398, token 0.44621. A model that learned only the output marginals scores 1.8650 nats; the irreducible floor is 0.8392.

**Reading the loss.** V1's mean sits essentially at the marginals-only baseline —
i.e. at this budget V1 largely has not learned secret-related structure. V2's mean
sits well below it and closer to the error floor.

**Token accuracy must not be read alone.** V1's mean is near the 0.4462 chance
level; V2 clears it. That is why acc_tau and exact accuracy are the primary metrics.

### Per-seed detail

| seed | V1 loss | V2 loss | V1 acc_tau | V2 acc_tau | V1 exact | V2 exact |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.7413 | 0.9388 | 0.3550 | 0.9839 | 0.0078 | 0.1011 |
| 42 | 1.8431 | 0.9089 | 0.1895 | 0.9878 | 0.0034 | 0.1128 |
| 123 | 1.8422 | 0.8931 | 0.1240 | 0.9873 | 0.0020 | 0.1191 |
| 456 | 1.7796 | — | 0.3442 | — | 0.0107 | — |
| 789 | 1.8419 | — | 0.1035 | — | 0.0059 | — |

### Paired test, and why it cannot conclude much

| Metric | paired n | sign-test p | unanimous |
|---|---:|---:|:---:|
| Validation loss (nats) | 3 | 0.2500 | yes |
| SALSA acc_tau | 3 | 0.2500 | yes |
| Exact integer accuracy | 3 | 0.2500 | yes |
| Token accuracy | 3 | 0.2500 | yes |
| Greedy token accuracy | 3 | 0.2500 | yes |
| Perfect accuracy | 3 | 0.2500 | yes |
| Mean |b - b_hat| | 3 | 0.2500 | yes |
| Decode failure rate | 0 | nan | no |

The arms share seeds, data stream and secret, so runs pair naturally and an exact
two-sided sign test is the appropriate test; nothing justifies assuming normality.
**With 3 pairs the smallest attainable two-sided p is 0.25.** No statistical
significance is claimed and none is reachable. Unanimity across three seeds is
reported as consistency of direction, nothing more.

## 4. Was phase 16's advantage just the extra depth?

| primary metric | V2 at T_e=4 (5 seeds) | V2 at T_e=2 (3 seeds) | change |
|---|---:|---:|---:|
| Validation loss (nats) | 0.9608 | 0.9136 | -0.0471 |
| SALSA acc_tau | 0.9869 | 0.9863 | -0.0006 |
| Exact integer accuracy | 0.1013 | 0.1110 | +0.0097 |

**Halving NACT's encoder depth barely changes it.** On the seeds available, the
phase-16 advantage does not appear to have come from the extra loops. This is the
single most useful thing the interim data says, and it is what the phase was for.

## 5. Sparsity generalization

*transcribed from the completed run's console output; the run wrote its four figures then failed before writing its JSON/CSV/MD. Evaluation not re-run; phase-12 script not modified.*

Same phase-12 procedure, same K sweep [1, 31, 63, 94, 125, 126, 157, 188, 220, 250], same seed, same test vectors, same
metrics, same decoder. Evaluated on the **seed-0** V2 T_e=2 checkpoint.

| nnz | V1 lift | V2 lift | V1 acc_tau | V2 acc_tau | V1 decode validity | V2 decode validity |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | +0.1309 | **+0.7642** | 0.3545 | 0.9878 | 1.000 | 1.000 |
| 10 | +0.0991 | **+0.7402** | 0.3345 | 0.9756 | 0.999 | 1.000 |
| 8 | +0.0718 | **+0.7231** | 0.2925 | 0.9438 | 0.998 | 1.000 |
| 6 | -0.0527 | **+0.5879** | 0.2290 | 0.8696 | 0.988 | 1.000 |
| 4 | -0.1787 | **+0.4243** | 0.1743 | 0.7773 | 0.951 | 1.000 |
| 2 | -0.3730 | **+0.2158** | 0.0815 | 0.6704 | 0.779 | 1.000 |
| 1 | -0.4800 | **+0.1196** | 0.0283 | 0.6279 | 0.530 | 1.000 |
| **K·e_i probes** | -0.4417 | **+0.1167** | 0.0417 | 0.6000 | 0.417 | 1.000 |

**Lift** is acc_tau minus what the best input-blind constant scores on the same
targets. Lift ≤ 0 means the model's output carries no usable information about its
input. This correction matters because sparser inputs make a constant predictor
*better*, so raw acc_tau alone would understate the collapse.

### The headline

- **V1** crosses zero nnz=8 and nnz=6, falling to **-0.4800** at nnz=1 and **-0.4417** on the probes, with probe
  decode validity 0.417 — most probe outputs were unreadable.
- **V2 never crosses zero in the measured range.** Lift stays positive at every
  level: **+0.1196** at nnz=1 and **+0.1167** on the probes, with probe
  decode validity **1.000** — every output readable.

**This strongly suggests V2 removes the severe sparse-input collapse seen in V1.**
The boundary did not move to nnz=6, 4, 2 or 1 — on this evidence it moved off the
measured range entirely, and V1's unreadable-output failure on probes is absent.

**But it is not yet a multi-seed statistical conclusion.** It rests on **one**
checkpoint (V2 seed 0) against **one** V1 checkpoint. The sparsity evaluation was
not repeated across seeds for either arm, so seed-to-seed variability of the lift
curve is entirely unmeasured — and V1's *learning* metrics vary a lot by seed
(acc_tau 0.10–0.36), which is reason to expect its sparsity curve might vary too.

**It is also not a secret-recovery result.** Recovery was not run and is out of
scope. What is measured is prediction fidelity on sparse and probe-shaped inputs.

## 6. CPU efficiency

V1 throughput was recorded in the phase-16 session and V2 T_e=2 in the phase-17 session, on a machine under different load. They are NOT comparable. The controlled same-session speed measurement is the phase-15 forward benchmark in results/nact_v2/architecture_report.json.

| | V1 GatedUT T_e=2 | V2 NACT T_e=2 |
|---|---:|---:|
| Samples/sec | 360.6 | 108.2 |
| Tokens/sec | 1,081.6 | 324.7 |
| Wall-clock (s) | 353.1 | 1,200.8 |
| CPU RSS (MiB) | 712.4 | 375.8 |

**Do not read the throughput rows as an architecture comparison.** The controlled,
same-session measurement is the phase-15 forward benchmark: V2 at T_e=2 runs at
**~1.95× V1's throughput at n=128** and ~1.64× at n=30, because its encoder
sequence is `n+2` rather than `2n+2`. Memory is measured within each run and is
the more trustworthy column here.

## 7. Parameter counts

| | parameters | fp32 |
|---|---:|---:|
| V1 Salsa2-GatedUT | 4,131,200 | 15.76 MiB |
| V2 Salsa2-NACT | 4,241,288 | 16.18 MiB |
| difference | +110,088 (+2.7%) | |

The +110,088 is entirely the input front end (digit embeddings 82,944; coordinate
embedding 65,536; numerical projection 2,560; special embeddings 2,048; zero vector
512; sparse attention bias 8; minus V1's 43,520 token embedding). Verified three
ways — actual, component breakdown, analytical — with nothing unclassified. Both
models sit inside the 4–5M budget with no padding parameters.

## 8. Limitations

1. **Three of five V2 seeds.** Not a 5-seed result and must not be cited as one.
2. **No reachable significance.** 3 pairs floor the two-sided sign test at p=0.25.
3. **Sparsity is single-checkpoint on both sides.** One V2 seed vs one V1 seed;
   seed variability of the lift curve is unmeasured.
4. **The V2 sparsity artifact is incomplete.** The evaluation ran and printed its
   full table and wrote its four figures, but crashed before writing its JSON/CSV/MD.
   The numbers here are transcribed from that run's output. Re-running it would
   restore the artifacts; that was not done because no further runs were authorised.
5. **Throughput is not comparable across arms** (different sessions).
6. **Component attribution is impossible.** The zero indicator, centered residue,
   Fourier pair, coordinate embedding, sparse attention bias and one-token-per-
   coordinate change were introduced together.
7. **Nothing cryptographic.** n=12, h=2 has C(12,2)=66 secrets and is a diagnostic
   positive control, as in phases 7 and 16.

## Interim classification

On the available evidence the result points to **A — NACT improves learning at
equal depth AND improves sparse generalization** — but with 3 of 5 seeds and a
single-checkpoint sparsity evaluation this is a **provisional** classification, not
a confirmed one. Completing seeds 456 and 789, and evaluating sparsity on more than
one checkpoint per arm, are what would settle it.

## Artifacts

- `equal_depth_ablation_interim.md` (this file)
- `equal_depth_ablation_interim.json`
- `equal_depth_ablation_interim.csv`
- `sparsity_v2_te2/0{1,2,3,4}_*.png` — the four figures the V2 sparsity run wrote
