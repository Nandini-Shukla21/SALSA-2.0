# Phase 22 — NACT component attribution study (design)

> **DESIGN ONLY. NO TRAINING PERFORMED.** No checkpoint was created or modified,
> no recovery was run, and the architecture was not changed. The only file access
> was reading already-trained weights for a zero-cost diagnostic.

## The question

NACT changed eight things at once. Which of them causes the improvement?

| # | component | parameters | note |
|---:|---|---:|---|
| 1 | one token per coordinate | 0 | a sequence-layout change, not a parameter: L_in goes 2n+2 -> n+2 |
| 2 | base-81 digit embeddings | 82,944 | replaces V1's 85 x 512 token embedding (43,520) |
| 3 | centered numerical residue | 512 | one column of the numerical projection |
| 4 | zero / nonzero indicator | 512 | one column of the numerical projection |
| 5 | Fourier cos/sin features | 1,024 | two columns of the numerical projection |
| 6 | absolute coordinate embedding | 65,536 | the only component with a parameter cost above 0.1% of the model |
| 7 | learned zero-coordinate feature | 512 | a single learned vector added when a_i == 0 |
| 8 | sparse attention key bias | 8 | one learned scalar per encoder head, zero-initialised |

Everything outside the front end — encoder body, decoder, output — is **4,087,680 parameters and is identical in every variant.**

## Proposed variants and their exact parameter counts

| key | variant | features | params | Δ vs full | % | budget | isolates |
|---|---|---:|---:|---:|---:|:---:|---|
| **A** | Full NACT (reference) | 4 | 4,241,288 | +0 | +0.000% | OK | nothing - this is the reference arm |
| **B** | no numerical features (keep zero indicator) | 1 | 4,239,752 | -1,536 | -0.036% | OK | centered residue + Fourier pair, jointly |
| **C** | no zero-aware features | 3 | 4,240,256 | -1,032 | -0.024% | OK | zero indicator + zero vector + sparse bias, jointly |
| **D** | no Fourier features | 2 | 4,240,264 | -1,024 | -0.024% | OK | the k=1 character pair |
| **E** | no absolute coordinate embedding | 4 | 4,175,752 | -65,536 | -1.545% | OK | absolute coordinate identity |
| **F** | one-token coordinate representation only | 0 | 4,238,208 | -3,080 | -0.073% | OK | ALL numerical/zero features at once |
| **G** | no sparse attention bias | 4 | 4,241,280 | -8 | -0.000% | OK | the per-head zero-key bias (8 parameters) |

### Could the parameter differences confound the result?

**No, with one qualification.** Six of the seven variants sit within **3,080
parameters — under 0.08%** — of full NACT, because every numerical feature costs
one 512-wide column and the sparse bias costs 8 scalars. A difference that small
cannot plausibly produce the effects being chased, which are of order 0.76 in
acc_tau.

The qualification is **E**, which drops the 65,536-parameter coordinate table
(1.5% of the model). That is still small, but it is the one variant where a
capacity explanation is not automatically dismissible and it must be stated when
E is reported.

**No padding parameters are proposed.** If a variant were thought too small, the
fair remedy is not to inflate it but to report the count honestly and, if
capacity is genuinely in doubt, add a matched-capacity control that spends the
difference somewhere architecturally neutral — not to manufacture dead weights.
Nothing here needs that: the largest gap is 1.5%.

Every variant stays inside the 4–5M budget.

## Redundancy — which variants can be dropped

**Nested ladder.** F (0 features) < B (1) < D (2) < A (4) — A ladder of single-feature additions: F->B isolates the zero indicator, B->D the centered residue, D->A the Fourier pair. Four arms give three clean single-feature contrasts.

**Leave-one-out.** A minus one of {zero-aware, Fourier, coordinate, sparse bias} = C, D, E, G — Every contrast is against the same reference A, which is what attribution needs. Preferred over the ladder for that reason.

| overlap | verdict |
|---|---|
| B and F: differ only by the zero-indicator scalar | keep both only if F->B is the specific question; otherwise F alone |
| B and D: differ only by the centered-residue scalar | B is redundant once F, D and A are run - the ladder is covered |
| C and G: C removes the sparse bias as part of zero-awareness, G removes it alone | G is subsumed by C for the sparsity question, and is separately predicted a no-op by the weight diagnostic |

**Safe to drop: ['G', 'B'].**

- **G** — predicted from trained weights: the bias never moved off zero, so removing 8 parameters that do nothing cannot change the outcome
- **B** — its two contrasts, F->B and B->D, are both recoverable from the F/D/A arms; it adds a run without adding a distinct question

## A zero-cost diagnostic that shrinks the matrix

*read-only inspection of trained weights; no forward pass, no training.*

| seed | max abs sparse bias | zero-indicator col | centered col | cos col | sin col | zero-vector norm | coord-emb norm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0119 | 1.338 | 1.335 | 1.349 | 1.429 | 0.244 | 5.400 |
| 42 | 0.0201 | 1.303 | 1.297 | 1.413 | 1.482 | 0.252 | 5.357 |
| 123 | 0.0144 | 1.335 | 1.359 | 1.434 | 1.493 | 0.202 | 5.389 |

- **Sparse attention bias.** The per-head bias was initialised at exactly 0 and reached at most 0.0201 after 100,032 samples, on all three seeds. Attention logits are O(1-10), so a bias of this size shifts the softmax negligibly. Variant G is therefore PREDICTED to be indistinguishable from A, and can be dropped from the paid matrix in favour of this zero-cost diagnostic.
- **Numerical features.** All four numerical columns carry comparable weight norm (~1.30-1.49) on every seed, with the Fourier pair marginally highest. No feature is ignored, so none of B, C or D can be predicted away and each needs a run. Weight norm is a weak proxy for importance and is used here only to rule OUT a component, never to rule one in.
- **Coordinate embedding.** The coordinate table reaches a norm comparable to the digit embeddings (~5.4 vs ~5.9), so it is carrying real signal and variant E is worth paying for.

## Controlled variables

Every future ablation run must hold all of the following constant, so that the
component under test is the only thing that varies:

```
  lwe.n = 12
  lwe.hamming_weight = 2
  lwe.q = 251
  lwe.sigma = 3.0
  lwe.structure = rlwe
  lwe.rlwe_variant = circulant
  lwe.secret_index
  lwe.num_train_samples = 100000 (100,032 seen)
  lwe.num_valid_samples = 2048
  lwe.sample_reuse = 1
  encoding.base = 81
  encoding.digit_order = lsb_first
  encoding.separator = False
  encoding.fixed_width = True
  experiment.seed
  training.optimizer = adamw
  training.learning_rate
  training.scheduler
  training.batch_size = 64
  training.warmup_steps = 200
  training.max_epochs = 20
  training.weight_decay
  training.grad_clip
  training.monitor_metric = valid_loss
  evaluation.tolerance = 0.1
  model.encoder_dim = 512
  model.decoder_dim = 128
  model.encoder_heads = 8
  model.decoder_heads = 4
  model.encoder_loops = 2
  model.decoder_loops = 2
  model.gated = True
  model.ffn_multiplier = 4.0
  model.dropout = 0.0
```

The existing `configs/nact_n12_h2_te2.yaml` already fixes all of these, so each
variant should be a config that `extends:` it and changes only its own component.

## Metrics and endpoints

**Primary, in rank order:** 1. SALSA acc_tau, 2. exact integer accuracy, 3. validation loss, 4. sparse-input lift, 5. direct secret recovery.

**Secondary:** token accuracy, decode validity, CPU throughput, memory.

**Sparsity endpoints:** nnz = [12, 8, 6, 4, 2, 1] plus `K*e_i` probes.
The main endpoint is **lift over the best input-blind constant**, because sparser
inputs make a constant predictor better and raw acc_tau alone would understate a
collapse. V1's zero-lift boundary sits between nnz=8 and nnz=6; V2 stays positive
through nnz=1. **The endpoint for every ablation is where its boundary lands
between those two.**

## Staged plan

*Each stage is paid for only by the variants that survived the previous one, and the gates are fixed before any run.*

| stage | name | applies to | cost per variant | gate |
|---:|---|---|---:|---|
| 1 | learning metrics | every variant | 1201 s | advance a variant to stage 2 if its acc_tau is within 0.05 of full NACT, OR if it drops by more than 0.05 - both outcomes are informative. Only variants that land in neither case (impossible by construction) are dropped. |
| 2 | sparsity generalization | every variant that trained successfully | 240 s | advance to stage 3 only if lift stays positive through nnz=1, the property that distinguishes V2 from V1 |
| 3 | direct secret recovery | only variants passing the stage-2 gate | 120 s | terminal |

**Does every ablation need recovery? No.** Recovery is cheap, so the reason to stage it is not cost but meaning. Phase 12 established that probe behaviour is the mechanism; a variant that fails stage 2 will fail recovery for a reason already measured, and reporting it as an independent finding would double-count one cause.

## How many seeds

**seed 0 screening for all variants; extra seeds only where needed.**

The three phase-17 NACT runs give acc_tau sd = 0.0021 and loss sd = 0.0232. A single seed therefore resolves any acc_tau difference larger than about 0.0064 (3 sd). The effect this study is chasing is the V1-vs-V2 gap of roughly 0.76 in acc_tau - two orders of magnitude larger than the seed noise. Spending 5 seeds per variant to resolve an effect that one seed resolves is wasted compute.

Add seeds when:

- a variant lands within 3 sd of full NACT and the claim is 'no effect' - absence of an effect needs more evidence than presence of one
- two variants land within 3 sd of each other and their ORDER matters
- the single decisive contrast in the final write-up, which should carry 3 seeds so the headline is not a one-run claim

**Caveat.** The measured sd is full NACT's. A variant that degrades toward V1-like behaviour may also inherit V1's much larger seed spread - V1's acc_tau ranged 0.10 to 0.36 across seeds. Any variant whose seed-0 result is clearly degraded must therefore get 3 seeds before its magnitude is quoted, even though its DIRECTION is safe from one run.

## Runtime estimates

Basis: three phase-17 NACT T_e=2 runs, same machine, 100,032 samples each.

| item | seconds |
|---|---:|
| training, one variant, one seed, 100,032 samples | 1201 (range 1075–1294) |
| — of which compute | 931 |
| — of which validation and overhead | 270 (22%) |
| sparsity evaluation, one variant | 240 |
| recovery + residual verification, one variant | 120 |

| scenario | experiments | total |
|---|---:|---:|
| **recommended (F, C, E + confirmation seeds)** | 5 training runs | **1.93 h** |
| all six new variants, one seed each | 6 training runs | 2.60 h |
| all six variants at five seeds | 30 training runs | 10.61 h |

Note that **variant A costs nothing** — it is already trained on three seeds from
phase 17.

## Recommendation

| rank | variant | value | distinguishes |
|---:|---|---|---|
| 1 | **F** — one-token only | highest | representation change vs information change |
| 2 | **C** — no zero-aware features | high | zero-awareness vs continuous value information |
| 3 | **E** — no coordinate embedding | high | absolute vs relative position |
| 4 | **D** — no Fourier | medium | ring structure vs plain magnitude |
| 5 | **G** — no sparse bias | predicted, do not pay for it | nothing measurable |

**F (one-token only).** It splits the two competing explanations in one run. If F matches A, every numerical feature is decoration and the gain is the tokenisation plus coordinate identity. If F falls back toward V1, the features are doing the work. No other single experiment separates these.

**C (no zero-aware features).** Phase 14 predicted the zero indicator repairs a measured two-position conjunction, and phase 12 measured the sparse collapse it was meant to fix. C is the direct test of the project's own stated hypothesis H2, and it is the one most likely to move the sparsity endpoint.

**E (no coordinate embedding).** b = sum_i a_i s_i binds coordinate i to bit s_i, so absolute identity is theoretically required and RoPE supplies only relative position. The trained coordinate table has a norm comparable to the digit embeddings, so it is carrying signal. This is also the only variant with a parameter delta worth stating.

**D (no Fourier).** Tests the modular-arithmetic claim specifically. Lower priority only because F and C already bracket it: if F matches A, D is moot.

**G (no sparse bias).** The trained bias never left its zero initialisation on any seed. Run it only if the prediction itself is to be checked.

### Minimum recommended set: **F, C, E**

F, C and E are the three arms that map onto three genuinely different explanations - representation, zero-awareness, positional identity. A is already trained on three seeds and costs nothing. D is held in reserve and only run if F shows the features matter but C does not explain why. G is predicted from weights already on disk.

### If only one experiment can be afforded: **F**

26 minutes, and it answers *is the gain the tokenisation or the numerical features?* —
the single question with the most competing explanations behind it.

## What this design can and cannot settle

- It attributes the improvement among **components of NACT**. It does not revisit
  whether NACT beats V1 — phases 17 and 19 measured that.
- Interactions are not identified. Leave-one-out finds each component's marginal
  contribution *in the presence of the others*; if two features are mutually
  redundant, removing either alone may show nothing while removing both shows a
  large effect. Variant F guards against exactly that by removing all of them.
- Everything stays at n=12, h=2, which is a diagnostic instance with C(12,2) = 66
  secrets. Component attribution there need not transfer to larger n.

## Artifacts

- `ablation_design.md` (this file)
- `ablation_design.json`
- `ablation_design.csv`

**NO TRAINING PERFORMED.**
