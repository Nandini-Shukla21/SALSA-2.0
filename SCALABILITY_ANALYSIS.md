# SALSA 2.0 — Scalability Analysis and Forecast

**Prediction document. No model was trained, retrained, or evaluated to produce it.**
Every measured number is read from an existing artifact under `results/`; every
forecast is labelled as a forecast. Code claims cite file and line.

Date: 2026-10-10 · Model under analysis: **Modified NACT, 4,238,208 parameters**

---

## 0. Discrepancies between the stated context and the repository

Checked against code, configs and recorded artifacts rather than against the
prose reports.

| stated | repository | verdict |
|---|---|---|
| ~4.24M parameters | `parameter_count = 4238208` in both final `summary.json` files | **confirmed** |
| n=12 and n=20 demonstrated | `nact_ablation_F/.../seed_0`, `n20_nact_f_pilot/.../seed_0` | **confirmed** |
| h=2, q=251, sigma=3, circulant | confirmed in both configs and inherited `base.yaml` | **confirmed** |
| ~100,032 samples per run | `samples_seen = 100032` in both | **confirmed** |
| n=30 is untested | **partially wrong — see below** | **discrepancy** |

### 0.1 n=30 is not untested — it is untested *for this architecture*

`PROJECT_REPORT.md` §7 states that n=30, 50 and 128 were "never trained or
tested. No model exists at those dimensions." Two models of comparable size
*were* trained at n=30, and both failed:

| run | model | params | n | h | samples | valid loss | acc_tau | exact |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `results/pilot_gatedut_n30_r/pilot` | Salsa2-GatedUT | 4,131,200 | 30 | 3 | 100,032 | 1.8426 | 0.1343 | 0.0029 |
| `results/archive/pilot_control_n30_r/pilot` | Salsa2-CompactTransformer | 4,251,520 | 30 | 3 | 100,032 | 1.8431 | 0.1479 | 0.0024 |

Chance for these metrics is loss 1.86498 (marginals-only), `acc_tau` 0.19287,
exact 0.00398. Both runs sit **at or below chance on every one** — `acc_tau` is
*below* chance, exact accuracy is *below* chance, and loss is a hair under the
marginal baseline. These are clean negative results, not missing experiments.

The accurate statement is: **no Modified NACT model exists at any n > 20**, and
the only n=30 evidence in the project uses a different architecture *and* a
different Hamming weight (h=3, from `base.yaml:34`, not h=2).

Recommend rewording §7. As written it understates the evidence base and hides
the only n=30 compute anchor the project owns.

### 0.2 The three-secret result belongs to full NACT, not Modified NACT

`PROJECT_REPORT.md` §6 reports "at n=12, three checkpoints trained under
different seeds — three different secrets — gave 3/3 exact recovery" in a
document whose subject is Modified NACT. The artifact
(`results/v2_recovery_robustness/recovery_robustness.json`) records
`parameter_count = 4241288` for all three checkpoints — those are **full NACT**
models, taken from `results/equal_depth_ablation/v2_te2/`.

So the multi-secret evidence for **Modified NACT specifically** is:

| | n=12 | n=20 |
|---|---:|---:|
| independent secrets attacked | 1 | 1 |
| exact recoveries | 1 | 1 |

Two single-seed successes. The 3/3 result is real, valuable and from the
immediately adjacent architecture — but it is not Modified NACT, and it cannot
be used to claim a recovery *rate* for the final model. Recommend attributing it
explicitly wherever it appears.

### 0.3 Smaller inconsistencies

- `results/archive/n30_prediction/` is a sample-requirement extrapolation for
  **Salsa2-GatedUT at h=3**, not for Modified NACT. Its five fitted models span
  **29×** at n=30. It must not be read as a Modified NACT forecast.
- Both final `summary.json` files record `model_name = "Salsa2-NACT"`. The
  display name `Salsa2-Modified-NACT` was introduced in Phase 30, after those
  artifacts were written. Not an error; it matters when reading artifacts.
- The repo annotation "paper recovers h=3 up to n=128" (`configs/base.yaml:34`)
  is the project's own note. It was not re-verified against the paper here.

---

## A. Executive conclusion

**The existing evidence supports extrapolation of *cost*, weakly supports
extrapolation of *trainability* to about n=30, and supports extrapolation of
*cryptanalytic capability* essentially nowhere.**

Three separate reasons, in descending order of importance:

1. **At h=2 the problem is not hard at any dimension in the requested range.**
   The secret space is C(n,2) — 66 at n=12, 8,128 at n=128. The project's own
   residual verifier (`salsa/verification/residual.py`) separates a correct
   candidate from a wrong one at roughly 24× in residual standard deviation
   (2.998 vs 71.803 at n=12). Exhaustive enumeration over C(n,2) candidates,
   scored by that verifier, recovers the secret at **every** dimension listed —
   with no neural network involved. A successful recovery at n=128, h=2 would
   therefore demonstrate nothing about neural cryptanalysis. The load-bearing
   axis of difficulty is **h** and **sample efficiency**, not n.

2. **Two positive dimensions cannot fix a scaling law.** Any two-parameter model
   fits two points exactly and leaves no residual. The project has already
   measured this failure mode: `results/archive/n30_prediction/` fits five
   plausible models through two breakpoints and gets answers spanning 29×.
   Adding n=30 would make it three points, which is the first budget at which
   the *shape* of the curve becomes an empirical question rather than a choice.

3. **The one negative data point is informative and discouraging.** At n=30,
   h=3, a 4.13M model under the identical 100,032-sample budget landed at chance
   on all three metrics. Modified NACT is a better architecture at n=12 by a
   large margin, but nothing in the evidence says how much of a dimension
   increment that advantage buys.

What *is* solidly extrapolable: per-sample compute, memory, sequence length, and
the architectural ceiling. Those are determined by the code and by three
measured throughput points, not by learning dynamics.

---

## B. Scalability table

"Recovery outlook" below means: *would the trained neural attack, retrained at
that dimension on a budget of this order, plausibly recover the secret.* It is
not a claim that recovery there would be scientifically meaningful — see §A.1.

Compute burden is **per 100,000 training samples, relative to the measured n=20
run**, derived in §D.5. It is a floor, not a total; total cost is this figure
multiplied by the number of samples required, which is unknown.

| n | evidence status | recovery outlook | compute/100k vs n=20 | main risk | confidence |
|---:|---|---|---:|---|---|
| 12 | **Observed** — recovered, verified, 3 seeds | demonstrated | 0.80× | none | **High** (measured) |
| 20 | **Observed** — recovered, verified, 1 seed | demonstrated | 1.00× | single seed, single secret | **High** (measured) |
| 30 | **Observed for other models at h=3** (both at chance); not observed for Modified NACT | **uncertain** | ~1.25× | sample budget insufficient; regime change between 20 and 30 | **Medium** |
| 50 | Extrapolated | **uncertain → unlikely** at 100k samples | ~1.74× | sample requirement likely far above budget | **Low** |
| 70 | Extrapolated | **unlikely** at this budget | ~2.2× | as above, plus untested capacity adequacy | **Low** |
| 90 | Extrapolated | **unlikely** at this budget | ~2.7× | as above | **Low** |
| 110 | Extrapolated | **unlikely** at this budget | ~3.2× | as above | **Low** |
| 128 | Extrapolated; **hard architectural boundary** | **unlikely** at this budget | ~3.7× | as above; `max_coordinates` ceiling reached exactly | **Low** |

No dimension above 20 carries a quantitative success probability, because none
can be estimated from the available evidence without inventing one.

---

## C. Evidence versus prediction

### C.1 Observed (measured, in this repository)

| quantity | n=12 | n=20 | n=30 |
|---|---:|---:|---:|
| architecture | Modified NACT | Modified NACT | GatedUT / CompactTransformer |
| parameters | 4,238,208 | 4,238,208 | 4,131,200 / 4,251,520 |
| h | 2 | 2 | **3** |
| samples seen | 100,032 | 100,032 | 100,032 |
| valid loss | 0.9891 | 0.9764 | 1.8426 / 1.8431 |
| `acc_tau` | 0.9839 | 0.9863 | 0.1343 / 0.1479 |
| exact accuracy | 0.0913 | 0.1035 | 0.0029 / 0.0024 |
| **exact secret recovery** | **YES** | **YES** | not attempted |
| residual verification | PASS (2.998) | PASS (2.941) | — |
| training throughput | 108.902 samp/s | 87.428 samp/s | 29.392 / — samp/s |
| peak RSS | 446.8 MB | 503.7 MB | 777.3 MB |

Also observed: V1 GatedUT at n=20, h=2 on a **200,000**-sample budget (twice the
NACT budget) reached loss 1.8233, `acc_tau` 0.2344, exact 0.0034 — exact
accuracy *below* its 0.00398 chance level (`results/control_b_n20_h2/`).

### C.2 Literature-supported (as recorded in this repository)

- SALSA (Wenger, Chen, Charton, Lauter, NeurIPS 2022) uses roughly **51M**
  parameters — about 12× this project's model.
- `results/archive/n30_prediction/n30_prediction.md` records SALSA Table 2 as
  reporting **3,913,424 to 29,210,829 samples (2^21.9–2^24.8)** for n=30,
  q=251, sparse binary secret. That is **39× to 292×** this project's entire
  per-run budget, for the *smallest* dimension in the extrapolation range.
- `results/original_fidelity_audit/` records that the original **reuses each
  distinct LWE instance ~10×** (`sample_reuse=1` here). The audit's own note:
  the paper's "log2 samples" counts distinct instances, so this project's
  sample budgets are not directly comparable and its sample-requirement
  extrapolations "may be pessimistic by up to ~10×".

**Why the original's results do not transfer.** The fidelity audit records
differences in architecture (~51M vs 4.24M; the supplied checkout had looping
and gating disabled), matrix construction (the original is **negacyclic**
despite the paper's "circulant" wording; this project is genuinely circulant),
error distribution (rounded vs truncated discrete Gaussian), sample reuse
(10× vs 1×), probe construction, K reduction, and the candidate decision rule
(the original resolves polarity **against the true secret**; this project uses a
secret-free anchored rule). Any one of these breaks a direct transfer of a
dimension result. SALSA succeeding at n=128 is not evidence that this model
will.

### C.3 Extrapolated (predictions, not results)

Everything in §B for n ≥ 30, the cost model in §D.5, and §E. None of it has been
measured.

---

## D. Architecture analysis — code-level constraints

### D.1 Hard ceiling at n = 128, and a silent breakage above it

```
salsa/models/nact.py:484   if n > spec.max_coordinates: raise ValueError(...)
salsa/models/nact.py:340   max_coordinates=int(max(128, config.lwe.n))
salsa/models/nact.py:192   max_coordinates: int = 128
```

n = 128 fits **exactly**. For n > 128 the table silently grows to size n, adding
512 parameters per coordinate — the model is then no longer 4,238,208
parameters, and `expect_parameters: 4238208` in `configs/pipeline_n12.yaml:27`
and `pipeline_n20.yaml:27` will reject it. This is a boundary worth knowing
about before anyone configures n=256.

### D.2 The decisive constraint: coordinate rows are trained only up to n

```
salsa/models/nact.py   embedded = embedded + self.coordinate_embedding[:n]...
```

Only rows `0..n-1` are ever indexed, so only those receive gradient. **A
checkpoint trained at n=12 has 116 of its 128 coordinate rows still at random
initialisation.** Loading it at n=30 is technically legal — the tensor is there,
the forward pass runs, nothing raises — and 18 of the 30 coordinate identities
will be random vectors the encoder has never seen.

This is the single most important distinction in this whole document:
**input-shape compatibility is not learned capability.** The table is sized for
128 coordinates so that *one architecture* covers every n without a parameter
change. It does not mean one *checkpoint* covers every n. It does not mean
anything at all about n=128 recovery.

Three independent reasons the same checkpoint cannot be reused at a larger n:

1. untrained coordinate rows (above);
2. `config_fingerprint` — the pipeline's `check_identity` refuses a checkpoint
   whose config does not match;
3. SALSA's design trains **one model per secret**, and the secret at n=30 is a
   different object from the secret at n=12. There is nothing to transfer.

### D.3 Sequence length and attention cost

| | V1 / original | Modified NACT |
|---|---|---|
| encoder length | `2n + 2` | **`n + 2`** |
| at n=128 | 258 | **130** |

Per encoder layer-application: attention is O(L²·d_e), FFN is O(L·d_e²). With
`encoder_dim = 512`, d_e² = 262,144 against L² = 16,900 at n=128 — the FFN term
dominates across the entire requested range, so **compute is approximately
linear in L up to n=128**. Attention quadratic blow-up is not the binding
constraint here. The one-token representation also halves L relative to V1,
which is why Modified NACT is both better *and* cheaper.

### D.4 The decoder is dimension-invariant — and that is a problem

`tokens_seen = 300096 = 3 × 100032` in both runs: the target is **3 tokens
regardless of n**, because `b` is a single integer in Z_q. So as n grows:

- input grows linearly;
- the number of terms in `b = Σ aᵢsᵢ` grows linearly;
- supervision per sample stays fixed at **≤ log2(251) = 7.97 bits**.

Signal per sample falls as roughly 1/n while the function to be learned gets
harder. This is a structural argument that sample requirements should grow
**super-linearly** in n — consistent with the published figures in §C.2 — and it
is the reason the compute figures in §D.5 are a floor rather than an estimate.

### D.5 Compute and memory

Measured anchors, from `summary.json` `throughput.compute_seconds`:

| model | n | L | samples/s | ms/sample |
|---|---:|---:|---:|---:|
| Modified NACT | 12 | 14 | 108.902 | 9.183 |
| Modified NACT | 20 | 22 | 87.428 | 11.438 |
| V1 GatedUT | 12 | 26 | 55.569 | 17.996 |
| V1 GatedUT | 20 | 42 | 43.838 | 22.812 |
| V1 GatedUT | 30 | 62 | 29.392 | 34.023 |

**Labelled assumption.** Fitting a straight line through the two Modified NACT
points gives `ms/sample ≈ 5.236 + 0.2819 × L`. Extrapolating it:

| n | L | ms/sample | 100k samples | vs n=20 |
|---:|---:|---:|---:|---:|
| 30 | 32 | 14.3 | ~0.40 h | 1.25× |
| 50 | 52 | 19.9 | ~0.55 h | 1.74× |
| 70 | 72 | 25.5 | ~0.71 h | 2.23× |
| 90 | 92 | 31.2 | ~0.87 h | 2.73× |
| 110 | 112 | 36.8 | ~1.02 h | 3.22× |
| 128 | 130 | 41.9 | ~1.16 h | 3.66× |

**Treat these as a floor.** The V1 series has three points and is visibly
super-linear in L (0.30 ms/token from L=26→42, 0.56 ms/token from L=42→62), so
the true Modified NACT curve is likely above the line. More importantly, these
are costs *per 100k samples*; the total is this × samples-required, and that
second factor is both unknown and the dominant one. The published n=30 figure
of 2^21.9–2^24.8 samples, if it applied here, would turn ~0.40 h into
**16–120 h** at n=30 alone.

**Memory is not a constraint.** Peak RSS was 446.8 MB (n=12) and 503.7 MB
(n=20); V1 at n=30 with L=62 peaked at 777.3 MB. Activation memory scales as
batch × L × d_e; at n=128 with L=130 this stays comfortably inside a laptop's
RAM. CPU memory will not be what stops this.

### D.6 Data generation

`generate_rlwe_matrix` (`salsa/data/rlwe.py:160`) draws `ceil(m / n)` generator
vectors and expands each into an n×n rotation block. At a fixed budget of
100,032 samples the number of **independent generator polynomials** falls from
**8,336 at n=12 to 782 at n=128**. Sample *generation* stays cheap, but the
diversity of the training distribution at a fixed budget thins as n grows — a
second, independent reason to expect the sample requirement to rise faster than
n. Note this is a property of the circulant RLWE structure, not of the model.

### D.7 What does *not* constrain scaling

- `digit_width = 2` is a function of q and base only (q=251, base=81) — n-free.
- RoPE is relative and parameter-free; no positional table to outgrow.
- Parameter count is genuinely invariant in n up to 128
  (`salsa/models/parameter_count.py:211`).
- `lwe.n >= 2` is the only validation on n (`salsa/utils/config.py:510`).
- Chance baselines (`acc_tau` 0.19287, exact 0.00398, marginal loss 1.86498)
  depend on q and tau, not n — the pre-registered thresholds transfer unchanged
  to any dimension, which makes cross-dimension comparison clean.

---

## E. Recommended prediction

**n = 30 is the largest dimension for which a cautious, evidence-based
prediction is reasonable.** Not a prediction of success — a prediction that the
experiment is *informative*.

Reasons:

1. It is one increment beyond measured ground, and increments are where
   extrapolation is least unreliable.
2. It is the only untested dimension where a **same-budget negative control
   already exists** (GatedUT and CompactTransformer, both at chance). A Modified
   NACT run there is therefore a controlled comparison on the first try, not an
   isolated number.
3. It is the **third point**. With n=12, 20, 30 the shape of the
   trainability-versus-dimension curve stops being a modelling choice and starts
   being a measurement. The 29× spread in `results/archive/n30_prediction/`
   is exactly the disease a third point treats.
4. Cost is bounded and known to within a small factor: ~1.25× the n=20 run per
   100k samples, i.e. **well under an hour** at the current budget.

Two design notes if you run it:

- **Match h to the existing n=30 controls (h=3), or re-run a control at h=2.**
  As it stands the n=30 evidence is h=3 and the Modified NACT evidence is h=2;
  running Modified NACT at n=30, h=2 would compare against nothing.
- **Pre-register the thresholds before looking**, as Phase 25 did. The chance
  baselines are n-invariant, so the same three bars (`acc_tau` > 0.21903,
  exact > 0.00816, loss < 1.86498) apply without modification.

**n = 50 and above are not currently predictable.** They are three to six
increments past evidence, past a dimension where two comparable models already
failed, on a budget one to two orders of magnitude below the only published
figure for the easier case.

---

## F. Limitations

1. **Two positive dimensions fix no scaling law.** Every two-parameter model
   fits them exactly. Model choice, not data, would drive any number produced.
2. **Both positive dimensions are single-seed at n=20** and single-h. Recovery
   at n=20 is one secret, one seed — an outcome, not a rate.
3. **The sample requirement at any untested n is unknown**, and it is the
   dominant term in total cost. Nothing here estimates it for Modified NACT.
4. **Capacity adequacy has never been varied.** 4.24M may be sufficient at
   n ≤ 20 and insufficient at n=50, in which case no sample budget succeeds and
   the whole cost extrapolation is moot. No capacity sweep exists.
5. **A regime change between n=20 and n=30 would invalidate every model
   equally** and is not detectable from the present data.
6. **The n=30 negative controls use h=3 and a different architecture**, so they
   bound nothing about Modified NACT directly.
7. **Component attribution is incomplete.** Ablation F removed six components at
   once, at one seed; variants C, D and E were designed and never run. Which
   part of the representation carries the advantage — and therefore whether it
   survives a dimension increase — is not established.
8. **No conclusion about cryptographic relevance can be reached by increasing n
   at h=2**, for the reason in §A.1. That requires increasing h.

### Optional check you can run yourself (not run here)

To confirm §A.1 quantitatively: enumerate all C(n,2) weight-2 binary vectors at
n=128 (8,128 of them), score each with `verify_candidate(A, b, c, q, sigma)` on
the existing validation samples, and confirm that exactly one has residual std
near 3 while the rest sit near 72.5. It is a seconds-to-minutes NumPy job. If it
behaves as the n=12 and n=20 verification already does, it establishes that
h=2 recovery is achievable at n=128 without any model — which is the point.

---

## G. Paragraph for the project report, under "Scalability Analysis and Future Work"

> SALSA 2.0 demonstrates exact secret recovery at n=12 and n=20 with h=2, and
> the architecture's parameter count is invariant in the lattice dimension up to
> a hard ceiling of 128 coordinates. Neither fact supports extrapolation to
> larger dimensions. The coordinate embedding is sized for 128 positions but
> only its first n rows receive gradient during training, so a checkpoint
> trained at one dimension carries untrained coordinate identities at any larger
> one: input-shape compatibility is not demonstrated cryptanalytic capability,
> and every larger dimension requires retraining from scratch. Compute scales
> benignly — the one-token representation gives an encoder sequence of n+2, and
> because the feed-forward term dominates attention at width 512 across this
> range, per-sample cost grows roughly linearly in n, extrapolating to about
> 3.7× the measured n=20 cost per 100,000 samples at n=128, with peak memory
> under 1 GB throughout. The binding constraint is therefore the sample
> requirement, not the architecture, and it is unmeasured: the project's two
> positive dimensions are fitted exactly by any two-parameter scaling model, and
> an earlier internal extrapolation at n=30 produced estimates spanning a factor
> of 29 depending only on which model was assumed. A same-budget pilot at n=30
> with a 4.13M-parameter predecessor architecture reached chance on every metric
> at h=3, and the published SALSA sample counts for n=30 exceed this project's
> entire per-run budget by one to two orders of magnitude. Finally, and most
> importantly, raising n at fixed h=2 does not raise cryptographic difficulty:
> the secret space is C(n,2), only 8,128 candidates at n=128, and the project's
> own residual verifier separates correct from incorrect candidates by roughly a
> factor of 24 in residual standard deviation, so exhaustive enumeration would
> succeed at every dimension considered here without any neural network. The
> scientifically informative next experiments therefore increase the Hamming
> weight and measure sample efficiency, with a single n=30 run — matched to the
> Hamming weight of the existing controls and with pre-registered thresholds —
> as the one dimension increment the current evidence can justify.

---

## H. Forecast scenarios per dimension

Qualitative by construction. Numbers appear only where a measured anchor
supplies them; everything else is marked **not reliably estimable**, which is a
finding rather than a gap to be filled.

The scenarios are not equally likely and no probability is attached to them,
because two positive dimensions cannot support one.

### n = 30 — the only dimension with a same-budget control

| scenario | what it would look like | what would have to be true |
|---|---|---|
| **Optimistic** | loss stays near 0.98, `acc_tau` near 0.98, recovery succeeds on most informative K | the one-token representation's advantage is roughly dimension-free in this range, and 100,032 samples remain sufficient |
| **Central** | loss rises toward but stays clearly below 1.86498; `acc_tau` falls well below 0.98 but clearly above 0.19287; recovery succeeds on fewer K, or not at all | learning degrades smoothly and the budget becomes marginal |
| **Pessimistic** | loss at ~1.84, `acc_tau` at or below chance, probe outputs largely undecodable | Modified NACT lands where GatedUT and CompactTransformer already landed at n=30 |

The pessimistic scenario is **not hypothetical at this dimension** — it is the
measured outcome for two other 4.1–4.25M models at the same budget (at h=3).

### n = 50 to n = 110

| scenario | what it would look like | what would have to be true |
|---|---|---|
| **Optimistic** | learning persists with a much larger sample budget; recovery remains possible | the sample requirement grows polynomially and stays within reach of a CPU |
| **Central** | 100k samples give chance-level performance; a budget 1–2 orders of magnitude larger is needed before anything is learned | the requirement grows roughly as the published SALSA figures suggest |
| **Pessimistic** | 4.24M parameters are insufficient regardless of budget | capacity, not data, becomes binding — a possibility the project has never tested, since capacity was never varied |

### n = 128 — the architectural boundary

Everything above applies, plus: `max_coordinates = 128` is reached **exactly**.
The model fits; 116 of its 128 coordinate rows would need training from scratch
at this dimension, as they would at any other. Nothing about the table's size
makes n=128 reachable.

And the point that outranks all of the above: at h=2, C(128,2) = 8,128. A
successful recovery here would not be a cryptanalytic result, because
enumeration plus the existing verifier gets there without a model.

### What a sudden, non-smooth failure would look like

Performance need not degrade smoothly, and three mechanisms in this codebase
could produce a cliff rather than a slope:

1. **Decode validity collapse.** Recovery needs the model's *output* to be a
   decodable integer on probe inputs far outside the training distribution. V1
   left 58% of probe outputs undecodable at n=12 while still scoring above
   chance in-distribution. Decode validity can fail abruptly while `acc_tau`
   still looks reasonable — a failure invisible in Plot B and fatal in Plot D.
2. **The constant-predictor attractor.** As inputs get sparser, a constant
   output becomes a *better* predictor. The measured V1 lift curve crosses zero
   between 8 and 6 non-zero coordinates. Past that crossing the model's output
   carries no information about its input at all — a threshold, not a slope.
3. **Budget cliff.** If the sample requirement crosses the fixed budget between
   two dimensions, the observed result jumps from "learns" to "chance" with
   nothing in between. That is exactly the shape of the measured n=20 to n=30
   transition for V1.

## I. Figures, data and how to regenerate them

```bash
python scripts/scalability_forecast.py
```

Reads recorded artifacts only. No training, no evaluation, no checkpoint load;
runs in seconds. Outputs to `results/scalability_forecast/`:

| file | what it shows |
|---|---|
| `plot_a_valid_loss.png` | measured loss at n=12, 20; envelope bounded by measured best and measured chance; the two other-architecture n=30 points plotted separately |
| `plot_b_acc_tau.png` | the same construction for `acc_tau`, with the n=30 points **below** chance |
| `plot_c_exact_recovery.png` | recovery as **counts of independent secrets**, never a rate; full NACT's 3/3 drawn as a distinct series |
| `plot_d_probes.png` | informative-K success (8/8) and probe decode validity (1.000), with V1's 0.417 as reference |
| `plot_e_compute.png` | relative cost per 100k samples: measured, extrapolated wall-clock, theoretical FLOP proxy, and V1's measured three-point curve |
| `scalability_forecast.csv` | every value with an `evidence_class` of observed / literature / theoretical / extrapolated / not_estimable |
| `forecast_table.md` | the summary table |

Drawing conventions, applied consistently: filled markers and solid lines are
**measured**; open markers and dashed lines are **extrapolated**; dotted lines
are **theoretical**; grey shading means **no evidence of any kind**. Series from
other architectures are never joined to the Modified NACT line.

Why the envelopes in plots A and B are drawn the way they are: each is bounded
below by the best value this project has measured and above by the chance
baseline recorded in the same artifact. It is not a confidence interval and
nothing is fitted. It spans "nothing degrades" to "complete failure" — and its
width is the result, not a defect of the drawing.

## J. Predicted Scalability of Modified NACT

> Modified NACT has been measured at two lattice dimensions, n=12 and n=20, both
> with Hamming weight h=2, and at both it predicts far above every chance
> baseline (validation loss 0.9891 and 0.9764 against a marginals-only baseline
> of 1.86498; `acc_tau` 0.9839 and 0.9863 against a chance level of 0.19287) and
> recovers the secret exactly, confirmed by an independent residual test that
> never sees the secret. Each of those recoveries is a single secret under a
> single seed, so they are two outcomes and not a success rate; the project's
> 3-of-3 multi-secret result belongs to the adjacent full-NACT architecture
> (4,241,288 parameters), not to the final model. Beyond n=20 the architecture
> imposes no barrier below its hard ceiling: the parameter count is invariant in
> the dimension because the coordinate embedding is a fixed 128x512 table, the
> one-token representation keeps the encoder sequence at n+2, and because the
> feed-forward term dominates attention at width 512 throughout this range, the
> per-sample cost rises roughly linearly — an extrapolation from the two
> measured throughputs gives about 1.25x the n=20 cost per 100,000 samples at
> n=30 and about 3.7x at n=128, with peak memory under 1 GB, against a
> theoretical encoder FLOP ratio of 7.1x that omits the dimension-independent
> overhead making up roughly half of measured wall-clock. What cannot be
> predicted is whether the model would learn anything at those dimensions. Only
> the first n rows of the coordinate table ever receive gradient, so every
> dimension requires training from scratch and no checkpoint transfers; the
> sample requirement at any untested dimension is unmeasured and is the dominant
> term in total cost; and the two measured dimensions are fitted exactly by any
> two-parameter scaling model, an under-determination the project has already
> quantified at a factor of 29 in an earlier n=30 extrapolation. The one
> relevant negative result is measured: at n=30, h=3, on the same
> 100,032-sample budget, a 4.13M-parameter predecessor and a 4.25M-parameter
> control both reached chance on every metric, and the published SALSA sample
> counts for n=30 exceed this project's entire per-run budget by one to two
> orders of magnitude. Degradation need not be smooth — probe decode validity
> can collapse while in-distribution accuracy still looks healthy, and a sample
> requirement crossing a fixed budget produces a cliff rather than a slope. A
> single pilot at n=30, matched to the Hamming weight of the existing controls
> and with its thresholds pre-registered, is therefore the one dimension
> increment the present evidence can justify. Finally, and decisively for how
> these results should be read: at fixed h=2 the secret space is C(n,2), just
> 8,128 candidates at n=128, and the project's own residual verifier separates
> correct from incorrect candidates by roughly a factor of 24 in residual
> standard deviation, so exhaustive enumeration would recover the secret at
> every dimension considered here without any neural network at all. Increasing
> n at fixed h=2 does not increase cryptographic difficulty; the scientifically
> informative axes are the Hamming weight and sample efficiency.
