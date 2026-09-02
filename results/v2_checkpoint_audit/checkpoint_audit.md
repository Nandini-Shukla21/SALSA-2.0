# Phase 18 — V2 NACT T_e=2 checkpoint audit

**Read-only.** No training, no resume, no recovery, no distinguisher, no
verification, no sparsity rerun, no benchmark. No checkpoint was created or
altered; no model, data or recovery code was modified.

## Available checkpoints

Scanned `results/equal_depth_ablation/v2_te2/nact_n12_h2_te2` — 4 run directories, 3 completed.

| seed | best epoch | samples | valid loss | acc_tau | exact | token | greedy | params | T_e | T_d | fingerprints match |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 19 | 100,032 | 0.9388 | 0.9839 | 0.1011 | 0.6969 | 0.6825 | 4,241,288 | 2 | 2 | yes |
| 123 | 19 | 100,032 | 0.8931 | 0.9873 | 0.1191 | 0.7044 | 0.6921 | 4,241,288 | 2 | 2 | yes |
| 42 | 19 | 100,032 | 0.9089 | 0.9878 | 0.1128 | 0.7028 | 0.6883 | 4,241,288 | 2 | 2 | yes |

| seed | checkpoint path | CPU RSS (MiB) | architecture | config fingerprint |
|---:|---|---:|---|---|
| 0 | `results/equal_depth_ablation/v2_te2/nact_n12_h2_te2/seed_0/checkpoints/best.pt` | 370.6 | nact | `be1c85196c6d267c` |
| 123 | `results/equal_depth_ablation/v2_te2/nact_n12_h2_te2/seed_123/checkpoints/best.pt` | 391.2 | nact | `a4cb9d0c870aec5e` |
| 42 | `results/equal_depth_ablation/v2_te2/nact_n12_h2_te2/seed_42/checkpoints/best.pt` | 365.5 | nact | `431e51ed5d60f02c` |

**Excluded:**

- `seed_456` — no artifacts/summary.json. Checkpoint files present: ['best.pt', 'last.pt']. Not deleted.

The checkpoints store **no separate weight fingerprint**; the only stored
fingerprint is the config fingerprint, so "fingerprints match" compares the one
in the checkpoint against the one in the run summary.

## Selection

Metric **`valid_loss` (min)**, read from each run's own config.yaml training.monitor_metric — not assumed.

Ranking: seed 123 = 0.8931, seed 42 = 0.9089, seed 0 = 0.9388.

**No secret was consulted and no recovery result influenced the choice.** The true secret is never loaded by this script and no recovery outcome influences the choice; selecting an attack target using the answer would invalidate the attack.

```
BEST CHECKPOINT:
path              results/equal_depth_ablation/v2_te2/nact_n12_h2_te2/seed_123/checkpoints/best.pt
seed              123
epoch             19
samples           100,032
validation loss   0.893135
acc_tau           0.987305
exact accuracy    0.119141
fingerprint       a4cb9d0c870aec5e
```

## Config fidelity — selected checkpoint

| check | expected | actual | match |
|---|---|---|:---:|
| `arch` | nact | nact | ✓ |
| `encoder_loops` | 2 | 2 | ✓ |
| `decoder_loops` | 2 | 2 | ✓ |
| `parameter_count` | 4241288 | 4241288 | ✓ |
| `n` | 12 | 12 | ✓ |
| `hamming_weight` | 2 | 2 | ✓ |
| `q` | 251 | 251 | ✓ |
| `sigma` | 3.0 | 3.0 | ✓ |
| `structure` | rlwe | rlwe | ✓ |
| `rlwe_variant` | circulant | circulant | ✓ |
| `base` | 81 | 81 | ✓ |
| `digit_order` | lsb_first | lsb_first | ✓ |
| `separator` | False | False | ✓ |
| `fixed_width` | True | True | ✓ |
| `vocab_size` | 85 | 85 | ✓ |
| `codec_vocabulary` | 85 | 85 | ✓ |
| `nact_encoder_length` | 14 | 14 | ✓ |
| `codec_input_length_v1_layout` | 26 | 26 | ✓ |
| `output_length` | 4 | 4 | ✓ |
| `seed_recorded` | 123 | 123 | ✓ |

All checks match: **True**.

Note on lengths: the codec still emits the **V1 token layout** of 26 tokens, which NACT reads and folds into its own **14-position** encoder sequence (`n+2`). Output length stays 4 and the vocabulary stays 85, so the recovery interface is unchanged.

## Integrity check

Loaded into a **fresh** `SalsaNact`; one tiny synthetic batch of 2 rows (one
random, one all-zero). **No accuracy was computed from it and none would mean
anything.**

| check | result |
|---|:---:|
| checkpoint loads (`strict=True`) | ✓ |
| fresh model is `SalsaNact` | ✓ |
| all expected state_dict keys present | ✓ (36/36 tensors) |
| no unexpected keys | ✓ |
| parameter count after load | ✓ (4,241,288) |
| model config matches checkpoint spec | ✓ |
| all parameters on CPU | ✓ |
| all parameters finite | ✓ |
| logits shape correct | ✓ ((2, 4, 85)) |
| logits finite | ✓ (max abs 16.6786) |
| no NaNs | ✓ |
| encoder length = n+2 | ✓ ((2, 14, 512)) |
| front end reconstructs `a` exactly | ✓ |

Shapes: `src (2, 26)` → `memory (2, 14, 512)` → `logits (2, 4, 85)`.

## V1 baseline, for context

*metadata inspection only; V1 was not reloaded or retrained.*

| field | value |
|---|---|
| run | `results/control_a_n12_h2/control` |
| checkpoint | `results/control_a_n12_h2/control/checkpoints/best.pt` |
| architecture | gated_universal_transformer |
| parameters | 4,131,200 |
| n / h | 12 / 2 |
| valid loss / acc_tau / exact | 1.7297 / 0.3472 / 0.0088 |
| matches phase-10 expectations | ✓ |

This is the checkpoint phase 10 attacked, where direct recovery returned NO.

## Caveats carried forward

- The V2 seed sweep is **incomplete** (3 of 5 seeds). Selecting the best of three
  is a valid secret-free rule, but it is a smaller pool than intended.
- The three completed seeds are nearly indistinguishable on the selection metric,
  so the choice between them is close to arbitrary on the evidence.
- Nothing here says anything about recovery. Phase 17's sparsity result is
  suggestive of better probe behaviour, but recovery has not been run against V2.

---

**READY FOR DIRECT RECOVERY: YES**

**SELECTED CHECKPOINT:**

`results/equal_depth_ablation/v2_te2/nact_n12_h2_te2/seed_123/checkpoints/best.pt`

**REASON:**

It has the lowest recorded `valid_loss` (0.8931) of the three completed equal-depth NACT runs under the run's own monitor metric, loads cleanly into a fresh model with all 4,241,288 parameters and every config value matching, and was selected without consulting the secret.
