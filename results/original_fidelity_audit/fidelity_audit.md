# Original SALSA fidelity audit

**Read-only.** Nothing was trained; no implementation file was modified.
Every comparison that could be executed was executed against the reference
source rather than argued from memory.

## Provenance of the reference

| file | status |
|---|---|
| `src/envs/encoders.py` | unmodified Meta release |
| `src/envs/generators.py` | unmodified Meta release |
| `src/envs/__init__.py` | unmodified Meta release |
| `src/envs/lattice.py` | modified in the supplied checkout |
| `src/evaluator.py` | modified in the supplied checkout |
| `src/trainer.py` | modified in the supplied checkout |
| `src/optim.py` | modified in the supplied checkout |
| `src/model/transformer.py` | modified in the supplied checkout |
| `src/model/__init__.py` | modified in the supplied checkout |
| `train.py` | modified in the supplied checkout |

> src/model/transformer.py carries 'Modern SALSA' markers and DISABLES the Universal Transformer looping and the copy gate ('obsolete in Modern SALSA'). Those are the two features the paper credits with a 14x sample-efficiency gain (Table 3). The supplied model reference is therefore NOT the paper's architecture, and Salsa 2.0 -- which keeps sharing and gating -- is architecturally closer to the paper than the supplied fork is.

## Stage-by-stage comparison

| Stage | Original SALSA | Salsa 2.0 | Identical? | Class | Difference | Consequence |
|---|---|---|:---:|:---:|---|---|
| 1a. coefficient range of a | rng.randint(0, percQ_bound*Q) with percQ_bound=1.0, maxQ_prob=0 -> [0, 250] | generate_uniform_matrix, uniform [0, q) -> [0, 250] | yes | **A** | none | Probe and training coefficient ranges agree. |
| 1b. sample reuse | reuse=True, times_reused=10, num_reuse_samples=10000, K=1 BY DEFAULT | sample_reuse=1 in every completed Salsa 2.0 run (each sample fresh) | NO | **D** | The original recycles each distinct LWE instance ~10 times; ours never reuses. | For a fixed number of gradient examples the original needs ~10x FEWER DISTINCT samples. The paper's 'log2 samples' counts distinct instances, so our per-sample budget is not directly comparable to theirs, and our sample-requirement extrapolations may be pessimistic by up to ~10x. |
| 2. secret generation | rejection loop: draw idx until unset, h times -> exactly h ones | rng.choice(n, h, replace=False) -> exactly h ones | yes | **C** | Different sampling mechanism, identical distribution (uniform over size-h supports). | None. Both give exactly h ones with a uniform support. |
| 3. error generation | np.int64(rng.normal(0, sigma).round()) -- ROUNDED Gaussian | exact truncated discrete Gaussian, P(x) ~ exp(-x^2/2s^2) | NO | **B** | Different distributions; total variation distance 0.00219, std 3.0139 vs 3.0000. | Negligible. Both are centred, integer, and of essentially the same width; the LWE relation and every metric are unaffected. |
| 4. matrix construction | circulant(a) with the strict UPPER TRIANGLE NEGATED, then mod q (i.e. NEGACYCLIC, despite the paper saying 'circulant' and the option being named circ_rlwe) | rlwe_variant=circulant (no negation) -- what every completed run used | NO | **D** | The original applies a sign flip on wrap-around; our trained runs did not. The original equals the TRANSPOSE of our negacyclic variant. | The training distribution differs. Row sets are not the same matrices, so the function the model must learn is different, though of the same family and difficulty. Salsa 2.0 implements negacyclic and can select it by config; no completed run used it. |
| 5. integer encoding | write_int_normal: fixed int_len = floor(log(Q, B)) + 1 digits, LSB-first | IntegerEncoder(fixed_width, digit_order='lsb_first'), width = smallest w with B^w >= q | yes | **A** | Width formulas differ only when q is an exact power of B (never for prime q). | None for q=251. |
| 6. input sequence construction | encode(row) then batch_sequences wraps with eos_index at BOTH ends; no_separator defaults to TRUE so '|' is never emitted | <bos> + digits + <eos>, separator=false under Representation R | yes | **A** | Identical token bodies; the two boundary tokens have different names and the original reuses one id for both ends. | None. Sequence length and content agree exactly. |
| 7. output sequence construction | y = output_encoder.write_int(b) (LSB-first digits), wrapped with eos at both ends by batch_sequences | <bos> + digits + <eos> | yes | **A** | Same digits, same order, same length. | None. |
| 8. vocabulary | 86 ids: 3 specials + '|' + '+' + 81 digits; eos_index=0 ('<s>'), pad_index=1 ('</s>'); '<pad>' (id 2) NEVER used; '+' never emitted | 85 ids: <pad>=0 <bos>=1 <eos>=2 <sep>=3 + 81 digits | NO | **C** | 86 vs 85 ids; the original carries two entries it never emits under the default settings and uses one token for both sequence ends. | None functionally. Embedding tables differ in size by one row, which is why the phase-4 counts moved by 768 parameters when V was corrected to 85. |
| 9. model architecture | PAPER: gated Universal Transformer, 1024/512, 16-32/4 heads, shared layer looped 2 (enc) and 8 (dec), copy gate, ~51M parameters. SUPPLIED transformer.py: looping and gating DISABLED ('obsolete in Modern SALSA'), plain block stack, RoPE, RMSNorm | Gated Universal Transformer, 512/128, 8/4 heads, 1 shared layer per side, T_e=T_d=2, copy gate, RoPE, RMSNorm, 4,131,200 parameters | NO | **E** | 12.3x fewer parameters than the paper. The supplied reference has REMOVED the UT looping and copy gate that the paper credits with a 14x sample-efficiency gain (Table 3), so it is not the paper's architecture either. | Capacity is the largest single deviation from the paper. Salsa 2.0 is architecturally CLOSER to the paper than the supplied fork, since it keeps sharing and gating, but is far smaller. |
| 10. training objective | F.cross_entropy(scores, y, reduction='mean') over non-pad target tokens | sequence_cross_entropy, ignore_index=pad, mean over scored tokens | yes | **A** | none | None. |
| 11. decoding for recovery | generate_beam(beam_size=params.beam_size), default beam_size=1 -> greedy in effect; hypotheses REVERSED before decode | greedy_decode: argmax, encoder run once, no reversal needed because encode/decode are true inverses | yes | **C** | Beam-1 equals greedy. The original's reversal compensates for write_int being LSB-first while parse_int reads MSB-first. | None at beam_size=1, the configured default. |
| 12. K probe construction | specialA = np.identity(N) * K; specialB = inner(specialA, TRUE SECRET). K values include 239145, 42899, Q, 3Q+7, random in [Q,10Q), 71/92/101/193/241 | build_probe_matrix = (K mod q) * I_n; NO b is constructed at all | NO | **B** | Identical probe geometry. The original builds a target from the true secret (needed only by its batching); ours decodes from <bos> and never constructs one. | Ours is a strictly cleaner attack: it cannot leak the secret through the target. Probe rows are identical. |
| 13. K reduction modulo q | write_int takes K mod B^int_len (mod 6561 at base 81), NOT mod q, so a large K can be encoded as a value OUTSIDE Z_q | K is reduced mod q, so the probe is always a legal element of Z_q | NO | **D** | The original can feed the model values >= q that never occur in training. | Only K mod q is mathematically meaningful (a.s = K*s_i mod q), so ours is correct. The original's out-of-range probes are an additional, unintended distribution shift on top of the intended one. |
| 14. candidate-bit decision | 3 rules (mean / mode / softmax-mean threshold) each producing a vector AND its inverse; the variant matching the TRUE SECRET best is reported | anchor rule: assign to whichever of 0 or K mod q the prediction is nearer on the ring; no inverse, no ground truth | NO | **B** | The original resolves polarity using the answer it is trying to find. | The original's reported 'secret matching' figure is optimistic: it takes a max over 6 variants scored against the truth. Ours is a genuine attack statistic. This makes our numbers NOT directly comparable to theirs -- ours are stricter. |
| 15. evaluation / success criterion | success iff any of the 6 variants matches the true secret in ALL N coordinates (np.any(match_counts == N)) | exact recovery iff candidate == true secret in all n coordinates, plus an explicit all-zeros baseline so sparsity cannot masquerade as recovery | yes | **B** | Same all-coordinates criterion; we add the trivial baseline the original does not report. | None to the criterion. The added baseline prevents a weight-h secret's (n-h)/n free accuracy being read as partial success. |
| 16. verification | residuals r = a.s_hat - b mod q; accept if std(r) ~ sigma (3) rather than ~ q/sqrt(12) (72.5). Described in paper 4.4; NOT implemented in the supplied evaluator.py | NOT IMPLEMENTED (phase 10 scope explicitly excluded it) | NO | **E** | Absent from both the supplied reference and Salsa 2.0. | No candidate can currently be verified without consulting the true secret. This is the missing final stage of the attack pipeline. |

Class legend: **A** Exact fidelity; **B** Intentional improvement; **C** Harmless implementation difference; **D** Potentially behaviour-changing difference; **E** Major incompatibility

Tally: Exact fidelity x5, Intentional improvement x4, Harmless implementation difference x3, Potentially behaviour-changing difference x3, Major incompatibility x2

## Training distribution versus recovery probes

- **training_distribution**: a ~ Uniform(Z_q^n), every coordinate independent
- **probe_distribution**: a = K * e_i: exactly one non-zero coordinate
- **mean_zero_coordinates_training**: 0.04808
- **zero_coordinates_in_probe**: 11
- **max_zero_coordinates_seen_in_200k_training_rows**: 3
- **probability_a_training_row_looks_like_a_probe**: 4.799338009446765e-26
- **one_in_how_many**: 1 in 2.084e+25
- **token_level_zero_digit_fraction_training**: 0.15576153846153845
- **token_level_zero_digit_fraction_probe**: 0.8461538461538461
- **l1_norm_training_mean**: 1500.66472
- **l1_norm_probe**: 125
- **original_trains_on_probe_like_inputs**: False
- **original_expectation**: gen_expr only ever produces rows of the (nega)circulant matrix built from a uniform a. No probe-like input appears in training. The original therefore REQUIRES the transformer to extrapolate to a region of input space it has never seen, with a ~51M-parameter model.

## Direct-recovery pipelines, side by side

### Original SALSA (reconstructed from evaluator.eval_secret)

```
for K in [239145, 42899, Q, 3Q+7, 42900, 5 random in [Q,10Q), 71,92,101,193,241]:
    specialA = identity(N) * K                     # rows are K*e_i, NOT reduced mod q
    specialB = inner(specialA, TRUE_SECRET)        # <-- uses the secret
    x = [input_encoder.encode(row) for row in specialA]
        write_int: K mod B^int_len  (mod 6561, NOT mod q)  -> can leave Z_q
        no separator (no_separator defaults True)
    y = [output_encoder.write_int(b) for b in specialB]
    x, y = batch_sequences(x), batch_sequences(y)  # eos_index at BOTH ends
    encoded = encoder(x)
    generations = decoder.generate_beam(encoded, beam_size=1)   # beam 1 == greedy
    pred[i] = output_encoder.decode(hyp[::-1])[0]  # REVERSED before decode
                                                   # exception -> pred[i] = -1
    bin1 = 0 if x > mean(pred) else 1               # threshold rules
    bin2 = 1 where pred == mode(pred) else 0
    bin3 = 0 if x > mean(softmax(pred)) else 1
    candidates = [bin1, ~bin1, bin2, ~bin2, bin3, ~bin3]        # each AND its inverse
    match_counts = [sum(c == TRUE_SECRET) for c in candidates]  # <-- uses the secret
    success iff any(match_counts == N)
# NOTE: verification (paper 4.4, residual std test) is NOT implemented here.
```

### Salsa 2.0

```
for K in configured sweep [1,31,63,94,125,126,157,188,220,250]:
    P = (K mod q) * identity(n)                    # rows are K*e_i, reduced mod q
    # no b is constructed at all
    src_ids, _ = codec.encode_batch(P)             # same encoder as training
        fixed width, lsb_first, no separator, <bos>..<eos>
    memory = model.encode(src_ids)                 # encoder runs once
    generated = greedy_decode(from <bos>, output_length-1 steps)
    b_hat, ok = decode_generated_ids(generated)    # undecodable -> ok=False
    d0 = ring_distance(b_hat, 0);  dK = ring_distance(b_hat, K mod q)
    score = (d0 - dK) / (d0 + dK)                  # anchored on 0 and K only
    score = 0 where not ok                         # unreadable = no evidence
    bit[i] = 1 if score > 0 else 0
    candidate_K = bits
aggregate = sign(sum over K of separation_weight(K) * score)
selected_K = argmax mean|score|                    # secret-free selection
# EVALUATION, strictly afterwards: compare against the true secret, and against
# the all-zeros baseline that a weight-h secret gives away for free.
```

### Line-by-line

| step | original | Salsa 2.0 | class |
|---|---|---|:---:|
| probe geometry | identity(N) * K | identity(n) * (K mod q) | **B** |
| probe value range | K mod B^int_len; may exceed q | K mod q; always in Z_q | **D** |
| target b | built from the TRUE SECRET | never constructed | **B** |
| encoding | input_encoder.encode, no separator | identical token body | **A** |
| boundaries | eos_index at both ends | <bos> ... <eos> | **C** |
| inference | generate_beam(beam_size=1) | greedy argmax | **C** |
| output decode | decode(hyp[::-1]); failure -> -1 | decode_generated_ids; failure flagged and counted | **B** |
| bit decision | mean/mode/softmax threshold | ring distance to 0 vs K | **B** |
| polarity | vector AND inverse, best vs TRUE SECRET | anchored on K; no inverse, no ground truth | **B** |
| multiple K | 10 values, any success counts | 10 values + separation-weighted vote + secret-free selection | **B** |
| success | all N coordinates match | all n coordinates match, plus all-zeros baseline reported | **B** |
| verification | NOT implemented in evaluator.py | NOT implemented (out of phase-10 scope) | **E** |

## Cause analysis for the phase-10 recovery failure

### 1. Incomplete reproduction of original SALSA

**Confidence: Low**

*Evidence for:*
- Matrix construction differs: the original negates the upper triangle (negacyclic); our trained runs used plain circulant.
- Sample reuse: the original reuses each instance ~10x by default; our runs used none.
- Verification (paper 4.4) is implemented in neither.

*Evidence against:*
- Integer encoding is token-for-token identical across bases 2/7/81.
- Input and output sequence bodies are identical; only marker names differ.
- Coefficient range, secret distribution and training objective all match.
- Probe geometry is identical (K*I); ours is strictly cleaner.
- None of these differences bears on whether a probe elicits a coordinate-dependent answer.

> Real differences exist, but none of them explains a model that returns one constant value to all twelve probes.

### 2. Distribution shift between training rows and probes

**Confidence: High**

*Evidence for:*
- A probe has 11 zero coordinates; training rows average 0.048 and never exceeded 3 in 200,000 rows.
- A uniform row looks like a probe with probability 4.799e-26 (1 in 2.084e+25).
- At token level 0.846 of probe digit tokens are '0' against 0.156 in training.
- MEASURED phase-10 behaviour: on real rows the model gave 4 distinct readable answers; on probes it gave ONE constant value (K=63) or unreadable output (11/12 at K=94, 12/12 at K=125).
- The model reproduced its recorded validation metrics exactly, so the collapse is specific to the probe inputs, not to loading.

*Evidence against:*
- The original faces the identical shift by construction and still reports recovery, so shift alone is not sufficient to prevent it.

> Directly measured. This is the proximate mechanism of the observed failure; it does not by itself establish that recovery is unreachable.

### 3. Reduced model capacity

**Confidence: Medium**

*Evidence for:*
- 4,131,200 parameters versus the paper's ~51M: 12.3x smaller.
- The paper (5.4, 8) identifies model size, and encoder width in particular, as the key factor for scaling.
- Extrapolating far outside the training distribution is exactly the kind of behaviour capacity buys.

*Evidence against:*
- Capacity was never varied in Salsa 2.0, so there is no measurement.
- The 4.13M model demonstrably learned the n=12 task in-distribution (acc_tau 0.347 against a 0.193 chance level).

> Plausible and consistent with the paper, but untested here.

### 4. Insufficient training

**Confidence: High**

*Evidence for:*
- 100,032 samples against the paper's 2^21.9 = 3.9M for n=30.
- Final validation loss 1.7297 sits only 0.135 nats below the marginals-only baseline of 1.8650 and far above the 0.839 floor set by the LWE error, so the model is early on its learning curve.
- The paper's Figure 3 shows recovery occurring shortly AFTER loss starts falling sharply; our loss had not entered that regime.
- With reuse disabled, our gradient budget per distinct sample is ~10x smaller than the original's default.

*Evidence against:*
- The paper states recovery succeeds 'long before the transformer has been trained to high accuracy'.

> Strongly supported by the loss position relative to both baselines, and by the sample budget.

### 5. Representation differences

**Confidence: Low**

*Evidence for:*
- Vocabulary 85 vs 86; different boundary-token convention.

*Evidence against:*
- Encoding verified token-for-token identical across three bases.
- Input and output bodies identical; lengths identical.
- Probe encodings round-trip exactly (asserted by test).
- Ours reduces K mod q, which is the mathematically correct choice.

> Effectively excluded by direct token-level comparison.

### 6. Algebraic structure differences

**Confidence: Low**

*Evidence for:*
- Our trained runs used plain circulant; the original uses negacyclic.
- The learned function therefore differs.

*Evidence against:*
- The probe relation b = K*s_i mod q holds for ANY row, whatever matrix built it, so the attack is well posed either way.
- Both structures present the same per-row task family and difficulty.
- The n=12 model did learn its own structure successfully.

> A genuine fidelity gap, but not a mechanism for probe collapse.

### 7. Combination

**Confidence: High**

*Evidence for:*
- Causes 2 and 4 are jointly sufficient and both are directly evidenced: the model is early in training AND the probe lies far outside its input distribution.
- Cause 3 plausibly governs how much training would be needed before extrapolation appears.

*Evidence against:*
- No single experiment yet separates 3 from 4.

> BEST SUPPORTED: insufficient training (4) leaves a model with no extrapolation to a probe distribution it has never seen (2); capacity (3) is a plausible but unmeasured modifier. Causes 1, 5 and 6 are real but do not explain the collapse.
