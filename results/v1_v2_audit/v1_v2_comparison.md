# V1 GatedUT vs V2 NACT - audit

**Read-only.** No training, no checkpoints written, no evaluation rerun, no code
changed. Every number is read from the `artifacts/summary.json` each run wrote.

## Two limitations, stated first

**1. No checkpoints exist for either arm.** All ten Phase-16 runs wrote metrics but
no `checkpoints/` directory, so the best checkpoint could not be located for either
model and nothing can be re-evaluated. The audit is therefore metrics-only, from
each run's own recorded validation pass. Best epoch is read from metadata:
V1 epoch 19, V2 epoch 19, monitored on `valid_loss (min)`.

**2. The two arms differ in TWO settings, not one.**

| setting | V1 | V2 |
|---|---|---|
| `model.arch` | gated_universal_transformer | nact |
| `model.encoder_loops` | 2 | 4 |

(4 further differences are run labels -- `experiment.name`, `output_root`, `notes`, `tags` -- which change no data, no optimisation and no evaluation.)

`encoder_loops` 2 -> 4 doubles V2's effective encoder depth at no parameter cost.
It is part of the approved architecture, but it means **this comparison is between
two configurations, not a clean isolation of the input front end.** Any statement
that the numerical front end alone caused the gain is not supported by these runs.

Everything else is identical: n=12, h=2, q=251, sigma=3, RLWE circulant, base 81
lsb-first no separator, same secret index, same seed, same AdamW settings, schedule,
batch size 64, 100,032 samples, 2,048 validation sequences, tau=0.1, greedy decode.
Both arms of each pair share the same validation stream.

## Head-to-head (seed 0, the exactly matched pair)

| Metric | V1 GatedUT | V2 NACT | Winner | Difference | % |
|---|---:|---:|:---:|---:|---:|
| Trainable parameters | 4,131,200.0000 | 4,241,288.0000 | **V1** | +110,088.0000 | +2.7% |
| Best epoch | 19.0000 | 19.0000 | **--** | +0.0000 | +0.0% |
| Samples trained | 100,032.0000 | 100,032.0000 | **--** | +0.0000 | +0.0% |
| Validation loss (nats) | 1.7413 | 0.9552 | **V2** | -0.7861 | -45.1% |
| Teacher-forced token accuracy | 0.5324 | 0.6989 | **V2** | +0.1665 | +31.3% |
| Teacher-forced perfect accuracy | 0.0078 | 0.1050 | **V2** | +0.0972 | +1243.8% |
| Greedy token accuracy | 0.4531 | 0.6828 | **V2** | +0.2297 | +50.7% |
| Exact integer accuracy | 0.0078 | 0.1050 | **V2** | +0.0972 | +1243.8% |
| SALSA acc_tau | 0.3550 | 0.9849 | **V2** | +0.6299 | +177.4% |
| Decode failure rate | 0.0000 | 0.0000 | **TIE** | +0.0000 | -- |
| Mean |b - b_hat| | 63.4736 | 6.5244 | **V2** | -56.9492 | -89.7% |
| Median |b - b_hat| | 38.0000 | 2.0000 | **V2** | -36.0000 | -94.7% |
| Samples/sec | 369.0790 | 259.8220 | **V1** | -109.2570 | -29.6% |
| Tokens/sec | 1,107.2000 | 779.5000 | **V1** | -327.7000 | -29.6% |
| Wall-clock training (s) | 344.6953 | 470.6023 | **V1** | +125.9070 | +36.5% |
| CPU RSS (MiB) | 766.0469 | 609.6406 | **V2** | -156.4062 | -20.4% |

Chance levels: token 0.4462, exact 0.00398, acc_tau 0.1929. Loss reference points: 1.8650 nats for a model that
learned only the output marginals (the real zero point), 0.8392 nats irreducible.

## Across all 5 seeds (descriptive spread, not a significance test)

| metric | V1 mean | V1 range | V2 mean | V2 range | ranges overlap |
|---|---:|---|---:|---|:---:|
| valid_loss | 1.8096 | [1.7413, 1.8431] | 0.9608 | [0.8924, 1.0827] | **no** |
| valid_acc_tau | 0.2232 | [0.1035, 0.3550] | 0.9869 | [0.9844, 0.9922] | **no** |
| valid_exact_accuracy | 0.0060 | [0.0020, 0.0107] | 0.1013 | [0.0728, 0.1152] | **no** |
| valid_token_accuracy | 0.4784 | [0.4451, 0.5324] | 0.6989 | [0.6890, 0.7043] | **no** |
| valid_perfect_accuracy | 0.0060 | [0.0020, 0.0107] | 0.1013 | [0.0728, 0.1152] | **no** |
| valid_greedy_token_accuracy | 0.4517 | [0.4422, 0.4757] | 0.6838 | [0.6693, 0.6921] | **no** |
| valid_mean_distance | 87.3035 | [63.4736, 119.7432] | 6.1366 | [4.8418, 7.6665] | **no** |
| samples_per_second | 360.5546 | [344.0010, 375.7060] | 263.2898 | [248.6530, 274.6320] | **no** |
| elapsed_seconds | 353.0878 | [338.2629, 370.4971] | 463.9723 | [443.0712, 491.4379] | **no** |
| rss_mb | 712.3594 | [601.1719, 836.6719] | 625.8250 | [573.8125, 701.9688] | yes |

## What the evidence supports

- **V2 has higher acc_tau by 0.6299** at seed 0 (0.3550 -> 0.9849), and by 0.7637 on the 5-seed mean.
- **V2 has lower validation loss by 0.7861 nats** at seed 0. V2's mean (0.9608) sits below the marginals-only baseline of 1.8650;
  V1's mean (1.8096) sits essentially at it.
- **V2 exact integer accuracy is ~17x V1's** on the 5-seed mean (0.0060 -> 0.1013), against a chance level of 0.00398.
- **V1 is unstable at this budget.** Its acc_tau ranges [0.1035, 0.3550] and 3 of 5 seeds ([42, 123, 789]) land at or below the
  0.1929 chance line - those seeds learned nothing measurable. V2's range is [0.9844, 0.9922] with every seed far above chance.
- **V1 is faster.** V2 runs at 73% of V1's training throughput and takes
  1.31x the wall clock, which is the expected cost of `encoder_loops=4`.
  The phase-15 forward benchmark showed V2 at `T_e=2` is ~1.95x *faster* than V1 at
  n=128, so this is a depth cost, not a front-end cost.
- Loss, acc_tau, exact accuracy and mean error have **non-overlapping ranges across
  all five seeds**. That is a description of the observed spread, not a significance
  test, and five paired runs at one instance size is a narrow base.

## What it does not support

- Not a cryptographic result. n=12, h=2 has only C(12,2)=66 possible secrets and was
  always a diagnostic positive control.
- **Not an attribution to the numerical front end.** `encoder_loops` differs too.
  Separating them needs a V2 run at `T_e=2`, which does not exist.
- **Not a secret-recovery result.** Nothing here was run through recovery, and phase 10
  measured V1's direct recovery as a failure. V2 is untested on that axis.
- **Sparsity generalization: not available in existing artifacts.** `results/recovery_generalization/` holds V1's phase-12 output only; no V2 equivalent
  exists, and no new experiment was run. The primary phase-12 research test - whether
  V2 moves the zero-lift crossing point away from nnz=8/6 - is therefore still open.

## Verdict

```
OVERALL WINNER:                 V2 Salsa2-NACT (accuracy), with caveats above

BEST ACCURACY:                  V2 - lower loss, higher acc_tau, higher exact
                                accuracy, lower error, on every one of 5 seeds
BEST CPU EFFICIENCY:            V1 - ~1.37x the training throughput as configured
                                (V2's deficit is encoder_loops=4, not the front end)
BEST PARAMETER EFFICIENCY:      V1 on raw count (4,131,200 vs 4,241,288, -2.6%);
                                V2 on accuracy per parameter, by a wide margin
BEST SPARSE-INPUT GENERALIZATION: UNDETERMINED - no V2 sparsity artifacts exist
BEST OVERALL:                   V2, for in-distribution learning at n=12,h=2 only.
                                The question the architecture was designed to answer
                                - sparse-input generalization - remains unmeasured.
```

## Artifacts

- `v1_v2_comparison.md` (this file)
- `v1_v2_comparison.json`
- `v1_v2_comparison.csv`
