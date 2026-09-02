# Phase 25 — NACT-F pilot at n=20, h=2

**One bounded training pilot.** No secret recovery, no n=30, no second seed, no
continuation to 200k. The existing n=12 checkpoints were not touched.

## Result

### Outcome **A — strong learning**

All three pre-registered gates, fixed before the run and not revised afterwards:

| gate | threshold | measured | pass |
|---|---:|---:|:---:|
| SALSA acc_tau above chance + 3σ | > 0.21903 | **0.9863** | **yes** |
| Exact integer accuracy above chance + 3σ | > 0.00816 | **0.1035** | **yes** |
| Validation loss below the marginals-only baseline | < 1.86498 | **0.9764** | **yes** |

For scale: chance acc_tau is 0.19287 and chance exact accuracy 0.00398. The measured acc_tau is not marginally
above its bar, it is **4.5× the chance level**. The loss sits closer to the
irreducible floor (0.8392) than to the marginals-only
baseline.

**Token accuracy is deliberately not the headline.** At 0.6995 it clears the 0.44622 chance level, but token accuracy near
chance has misled this project before and is reported as a secondary metric only.

## Setup

- Model: **NACT-F (one_token_only)**, 4,238,208 parameters — **unchanged from n=12** (True), because the coordinate embedding is a fixed 128 x 512 table, so n does not enter the parameter count; only the encoder sequence length changes, 14 -> 22.
- Switches confirmed off: `{'use_numerical_features': False, 'use_zero_vector': False, 'use_sparse_attention_bias': False}`. No removed feature
  was added back.
- Instance: n=20, h=2, q=251, sigma=3, RLWE/circulant, representation R, base 81,
  lsb_first, no separator, fixed width.
- Fresh secret: `[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]`, support [4, 17], weight 2. Derived by derive_seed(experiment.seed=0, 'secret', 0) at n=20 — a different dimension, so it
  shares nothing with the n=12 secret.
- Control parity: **one** config difference, `lwe.n: 12 -> 20`. Optimizer, learning
  rate, scheduler, batch size 64, validation protocol, tau, loss and seed are all
  inherited from the phase-23/24 NACT-F run.
- Budget: 100,032 samples, the approved ceiling. Not exceeded.

## Comparison

| Metric | V1 GatedUT n=20 | NACT-F n=12 | **NACT-F n=20** |
|---|---:|---:|---:|
| Validation loss (nats) | 1.8233 | 0.9891 | **0.9764** |
| SALSA acc_tau | 0.2344 | 0.9839 | **0.9863** |
| Exact integer accuracy | 0.0034 | 0.0913 | **0.1035** |
| Token accuracy | 0.4590 | 0.6934 | **0.6995** |
| Greedy token accuracy | 0.4391 | 0.6771 | **0.6851** |
| Perfect accuracy | 0.0034 | 0.0913 | **0.1035** |
| Decode failure rate | 0.0000 | 0.0000 | **0.0000** |
| Mean |b - b_hat| | 76.9692 | 7.1465 | **6.5093** |
| Median |b - b_hat| | 61.0000 | 3.0000 | **3.0000** |
| Trainable parameters | 4,131,200 | 4,238,208 | **4,238,208** |
| Training samples | **200,000** | 100,032 | **100,032** |
| Samples/sec | 43.8 | 108.9 | 87.4 |
| Tokens/sec | 131.5 | 326.7 | 262.3 |
| Wall-clock (s) | 7636 | 1146 | 1424 |
| CPU RSS (MiB) | 236.8 | 396.0 | 411.3 |

**The pilot used 100,032 samples. The V1 n=20 reference used 200,000 - twice as many - so the comparison understates rather than overstates NACT-F.**

### Separating the dimension change from the architecture change

Two comparisons are available and they answer different questions:

- **Architecture, dimension held at n=20.** V1 vs NACT-F: acc_tau 0.2344 → 0.9863, loss 1.8233 → 0.9764, exact 0.0034 → 0.1035 — and
  NACT-F did it on half the samples. V1's exact accuracy at n=20 was actually
  **below its own chance level** (0.00398).
- **Dimension, architecture held at NACT-F.** n=12 vs n=20: acc_tau 0.9839 → 0.9863, loss 0.9891 → 0.9764, exact 0.0913 → 0.1035. Going
  from 12 to 20 coordinates cost essentially nothing on this budget.

The second comparison is the more surprising one. The secret search space grew
from C(12,2) = 66 to C(20,2) = 190, and the encoder sequence from 14 to 22
positions, yet every primary metric is flat.

## What this does and does not show

**MEASURED.** NACT-F learns the n=20, h=2 problem on a 100,032-sample budget,
clearing all three pre-registered thresholds by wide margins, with decode failure
rate 0.0000 and mean |b − b̂| of 6.51 against V1's 76.97.

**INFERRED.** The simplified one-token architecture does not degrade between n=12
and n=20 at this budget. Whatever V1's difficulty at n=20 was, it is not intrinsic
to the problem at this scale.

**NOT ESTABLISHED — and these are the claims that matter:**

- **No n=20 secret recovery.** Recovery was not run. Learning to predict `b` is
  necessary for recovery but is not the same thing, and phases 10–12 showed a
  model can predict well in-distribution and still fail completely on probes.
- **Nothing about n=30 or n=128.** One dimension step is not a scaling law.
- **No cryptographic success.** n=20, h=2 has C(20,2) = 190 possible secrets.
- **One seed, one secret.** No variance estimate at n=20 exists.
- The sparsity generalization endpoint was not measured at n=20.

## Figures

- `01_valid_loss.png`
- `02_acc_tau.png`
- `03_exact_accuracy.png`
- `04_learning_curve.png`

## Artifacts

- `n20_nact_f_pilot.md` (this file)
- `n20_nact_f_pilot.json`
- `n20_nact_f_pilot.csv`
- checkpoints: `results/n20_nact_f_pilot/nact_n20_h2_te2_F/seed_0/checkpoints/` (best.pt, last.pt)

**PHASE 25 COMPLETE — 100k PILOT ONLY. NO ADDITIONAL TRAINING PERFORMED.**
