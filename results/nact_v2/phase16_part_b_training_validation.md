# PHASE 16 PART B — NACT EXPERIMENTAL / TRAINING VALIDATION

## 1. Objective

Determine whether the approved Salsa2-NACT architecture trains and evaluates under SALSA's established positive-control protocol. This is an experimental feasibility result; it is not a claim of production readiness or multi-seed superiority.

## 2. Repository and Commit

- Repository: `/Users/pjay/Downloads/SALSA-2.0-git`
- HEAD at start and finish: `2f5ee58 MODEL 2 ADDED`
- Source changes: none.
- Results root: `results/nact_v2/phase16_part_b/`

## 3. Experimental Protocol

The repository's existing entry point, `scripts/train.py`, was used for both runs. The trainer streams deterministic public RLWE batches, holds a fixed validation set, checkpoints every validation, and selects `best.pt` by minimum `valid_loss`.

| Setting | Value |
| --- | --- |
| seed | 0, deterministic |
| device | CPU, float32; 4 torch threads |
| optimizer | AdamW (beta1 0.9, beta2 0.98, eps 1e-9) |
| learning rate | 0.0005; cosine decay to 1e-6 after 200-step warmup |
| weight decay / clipping | 0.01 / 1.0 |
| batch / evaluation batch | 64 / 128 |
| training budget | 100,000 configured samples; 1,563 steps; 20 epochs/evaluations |
| validation / test | fixed independent streams, 2,048 samples each |
| selection rule | minimum validation cross-entropy loss |
| decoding | repository greedy decode (`beam_size: 1`) |
| primary task metric | `acc_tau`, tolerance 0.1 |

Only `experiment.output_root` was overridden to retain this phase's artifacts under the requested result directory; no scientific or training hyperparameter was changed.

## 4. Dataset

No materialized dataset file is used. SALSA generates reproducible, disjoint public RLWE streams from seed 0: circulant RLWE, n=12, q=251, sigma=3.0, binary secret of Hamming weight 2. Representation R uses base-81, least-significant-digit-first, fixed-width encoding without separators. The trainer receives public `(A, b)` samples only.

## 5. V1 Control

A matching n=12/h=2 V1 result was not present, so the existing `configs/control_a_n12_h2.yaml` protocol was run with no training-setting changes.

- Model: `Salsa2-GatedUT` / `SalsaTransformer`
- Parameters: 4,131,200
- Best checkpoint: epoch 19, step 1,563, `best.pt`
- Best/final validation loss: 1.741287
- Validation `acc_tau`: 0.354980; greedy exact accuracy: 0.007812
- Wall time: 344.695 s; mean throughput: 369.079 samples/s

## 6. NACT Configuration

- Config: `configs/nact_n12_h2.yaml`
- Model: `Salsa2-NACT` / `SalsaNact`
- Encoder loops: 4
- Parameters: 4,241,288 (verified both directly and through the normal config/build path)
- Vocabulary: 85
- Best checkpoint: epoch 19, step 1,563, `best.pt`

## 7. Training Procedure

The normal trainer completed its alignment guard before optimization: it verified public-only training data, shape agreement, finite logits, and the 85-token vocabulary. NACT then completed all 20 planned epochs; stop reason was `planned epochs completed`.

## 8. Training Results

NACT final-epoch training loss was 0.951123. Its validation loss decreased from 2.013612 after epoch 0 to 0.955208 at epoch 19. No non-finite logits or numerical error occurred.

## 9. Validation Results

Final NACT validation over 2,048 examples:

- loss: 0.955208
- token accuracy: 0.698893
- greedy token accuracy: 0.682780
- greedy exact accuracy: 0.104980
- `acc_tau`: 0.984863
- decode-failure rate: 0.0

For context, the configured chance `acc_tau` baseline is 0.192870.

## 10. Evaluation Results

The saved NACT `best.pt` loaded successfully into `SalsaNact`. The repository's `evaluate` function completed greedy evaluation on the independent 2,048-example test stream; forward logits were finite and had vocabulary size 85.

- test loss: 0.950091
- test token accuracy: 0.700195
- test greedy token accuracy: 0.686686
- test greedy exact accuracy: 0.104980
- test `acc_tau`: 0.989746
- test decode-failure rate: 0.0

Raw output: `results/nact_v2/phase16_part_b/evaluation.log` and `test_metrics.json`.

## 11. V1 vs NACT Comparison

Both entries use the same n=12/h=2 control, seed, data-stream construction, batch sizes, optimiser, schedule, sample budget, validation split, and checkpoint rule. NACT's four encoder loops and its approved input architecture differ as documented by its config.

| Metric | V1 | NACT |
| --- | ---: | ---: |
| Parameters | 4,131,200 | 4,241,288 |
| Final train loss | 1.735678 | 0.951123 |
| Best/final validation loss | 1.741287 | 0.955208 |
| Validation `acc_tau` | 0.354980 | 0.984863 |
| Validation greedy exact accuracy | 0.007812 | 0.104980 |
| Wall time (s) | 344.695 | 470.602 |
| Mean samples/s | 369.079 | 259.822 |
| Final CPU RSS (MB) | 766.05 | 609.64 |

The single matched run shows successful NACT convergence and better values than this particular V1 control run, but it does **not** establish statistical superiority.

## 12. Performance / Memory

NACT was slower in this CPU run: 259.822 versus 369.079 samples/s (about 70% of V1 throughput). RSS is a process snapshot reported by SALSA, not a reliable hardware peak-memory measurement; the final snapshots were 609.64 MB for NACT and 766.05 MB for V1. No OOM occurred.

## 13. Checkpoints

NACT raw checkpoints:

- `results/nact_v2/phase16_part_b/nact_training/nact_n12_h2/phase16_part_b_nact/checkpoints/best.pt`
- `results/nact_v2/phase16_part_b/nact_training/nact_n12_h2/phase16_part_b_nact/checkpoints/last.pt`

V1 control checkpoints are retained under `v1_control/control_a_n12_h2/phase16_part_b_v1/checkpoints/`.

## 14. Failures or Anomalies

None in the training/evaluation protocol. The initial config-path build attempt encountered an environment sandbox write denial before a run directory existed; the subsequent authorized normal trainer run completed successfully. No architecture or V1 code was modified.

## 15. Interpretation

This experiment distinguishes the required questions:

- Architecture correctness was established in Part A and reconfirmed by normal-build parameter/alignment checks here.
- Training feasibility is demonstrated: NACT completed all planned optimization steps without numerical, checkpoint, or decoding failures.
- Task performance is measured: NACT achieved test `acc_tau` 0.989746 on this positive control.
- Superiority is not established: one deterministic seed and one easier control do not provide a statistical or broader-task conclusion.

## 16. Limitations

- One seed only; the repository protocol does not mandate a multi-seed study.
- This is the n=12, h=2 diagnostic positive control, not the n=30 cryptographic setting.
- V1 received matching validation but no separate V1 test decode in this phase; the direct controlled comparison therefore uses validation metrics.
- CPU RSS snapshots are not peak-GPU/peak-system memory measurements.

## 17. Final Verdict

### PASS

NACT successfully executed the approved 100,000-sample / 1,563-step training protocol, saved and reloaded its best checkpoint, and completed the standard independent test evaluation with finite 85-way logits and working greedy decoding. This PASS is for training feasibility and measured task performance under the stated control, not for claims of general superiority.

## 18. Recommendation for Next Phase

If a comparative architecture claim is needed, run the repository's same protocol across independently recorded additional seeds for both V1 and NACT, then report variation. Separately validate on the intended n=30 task before making cryptographic-performance claims.
