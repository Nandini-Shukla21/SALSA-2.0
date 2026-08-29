# Phase 14 - Salsa 2.0 v2 architecture study

**Design only.** Nothing was trained. No model, data generator, checkpoint or
recovery code was modified, and no model code was written.

## Anchor

Every parameter count in this document comes from formulas that reproduce the
shipped GatedUT **exactly**: 4,131,200 predicted, 4,131,200 measured. The script asserts
this before computing anything else, so counts for models that do not exist yet
rest on formulas known to be correct for a model that does.

## 0. Where the current budget actually goes

| component | parameters | share |
|---|---:|---:|
| encoder_ffn | 2,097,152 | 50.8% |
| encoder_attention | 1,048,576 | 25.4% |
| encoder_copy_gate | 524,800 | 12.7% |
| decoder_cross_attention | 163,840 | 4.0% |
| decoder_ffn | 131,072 | 3.2% |
| decoder_self_attention | 65,536 | 1.6% |
| encoder_embedding | 43,520 | 1.1% |
| decoder_copy_gate | 32,896 | 0.8% |
| decoder_embedding | 10,880 | 0.3% |
| output_projection | 10,880 | 0.3% |
| encoder_normalization | 1,536 | 0.0% |
| decoder_normalization | 512 | 0.0% |

**The encoder is 89.9% of the model and the input representation is 1.1%.**
Three encoder tensors -- FFN, attention, copy gate -- are 88.9% of the budget.
Any redesign that spends parameters on the input representation is spending from
the cheapest part of the model, and any redesign that shortens the sequence is
attacking the expensive part without touching the parameter count at all.

## 1. Candidates

| key | name | mixing | tokens/coord | encoder | decoder | parameters | fp32 | budget |
|---|---|---|---:|---|---|---:|---:|---|
| A | Salsa2-GatedUT (current baseline) | full self-attention | 2 | d=512 H=8 sets=1 T=2 | d=128 H=4 sets=1 T=2 | 4,131,200 | 15.76 MiB | OK |
| B | Salsa2-NACT (numerical-aware compact transformer) | full self-attention + sparse-aware attention bias | 1 | d=512 H=8 sets=1 T=4 | d=128 H=4 sets=1 T=2 | 4,241,288 | 16.18 MiB | OK |
| B-lite | Salsa2-NACT at T_e=2 (identical parameters to B) | full self-attention + sparse-aware attention bias | 1 | d=512 H=8 sets=1 T=2 | d=128 H=4 sets=1 T=2 | 4,241,288 | 16.18 MiB | OK |
| C | Salsa2-Hybrid (depthwise conv + gated MLP + attention) | depthwise conv (local) + full self-attention (global) | 1 | d=512 H=4 sets=1 T=4 | d=128 H=4 sets=1 T=2 | 4,113,796 | 15.69 MiB | OK |
| C-alt | Salsa2-SSM (state-space instead of attention) | diagonal state-space scan, no attention in the encoder | 1 | d=512 H=0 sets=1 T=4 | d=128 H=4 sets=1 T=2 | 4,004,736 | 15.28 MiB | OK |

### Parameter breakdowns

**A - Salsa2-GatedUT (current baseline)** (4,131,200)

| component | parameters | share |
|---|---:|---:|
| encoder_ffn | 2,097,152 | 50.8% |
| encoder_attention | 1,048,576 | 25.4% |
| encoder_copy_gate | 524,800 | 12.7% |
| decoder_cross_attention | 163,840 | 4.0% |
| decoder_ffn | 131,072 | 3.2% |
| decoder_self_attention | 65,536 | 1.6% |
| encoder_embedding | 43,520 | 1.1% |
| decoder_copy_gate | 32,896 | 0.8% |
| decoder_embedding | 10,880 | 0.3% |
| output_projection | 10,880 | 0.3% |
| encoder_normalization | 1,536 | 0.0% |
| decoder_normalization | 512 | 0.0% |

**B - Salsa2-NACT (numerical-aware compact transformer)** (4,241,288)

| component | parameters | share |
|---|---:|---:|
| encoder_ffn | 2,097,152 | 49.4% |
| encoder_attention | 1,048,576 | 24.7% |
| encoder_copy_gate | 524,800 | 12.4% |
| decoder_cross_attention | 163,840 | 3.9% |
| decoder_ffn | 131,072 | 3.1% |
| encoder_digit_embeddings | 82,944 | 2.0% |
| coordinate_embedding | 65,536 | 1.5% |
| decoder_self_attention | 65,536 | 1.5% |
| decoder_copy_gate | 32,896 | 0.8% |
| decoder_embedding | 10,880 | 0.3% |
| output_projection | 10,880 | 0.3% |
| numerical_feature_projection | 2,560 | 0.1% |
| encoder_special_embeddings | 2,048 | 0.0% |
| encoder_normalization | 1,536 | 0.0% |
| zero_coordinate_vector | 512 | 0.0% |
| decoder_normalization | 512 | 0.0% |
| sparse_attention_bias | 8 | 0.0% |

**B-lite - Salsa2-NACT at T_e=2 (identical parameters to B)** (4,241,288)

| component | parameters | share |
|---|---:|---:|
| encoder_ffn | 2,097,152 | 49.4% |
| encoder_attention | 1,048,576 | 24.7% |
| encoder_copy_gate | 524,800 | 12.4% |
| decoder_cross_attention | 163,840 | 3.9% |
| decoder_ffn | 131,072 | 3.1% |
| encoder_digit_embeddings | 82,944 | 2.0% |
| coordinate_embedding | 65,536 | 1.5% |
| decoder_self_attention | 65,536 | 1.5% |
| decoder_copy_gate | 32,896 | 0.8% |
| decoder_embedding | 10,880 | 0.3% |
| output_projection | 10,880 | 0.3% |
| numerical_feature_projection | 2,560 | 0.1% |
| encoder_special_embeddings | 2,048 | 0.0% |
| encoder_normalization | 1,536 | 0.0% |
| zero_coordinate_vector | 512 | 0.0% |
| decoder_normalization | 512 | 0.0% |
| sparse_attention_bias | 8 | 0.0% |

**C - Salsa2-Hybrid (depthwise conv + gated MLP + attention)** (4,113,796)

| component | parameters | share |
|---|---:|---:|
| encoder_gated_mlp | 1,966,080 | 47.8% |
| encoder_attention | 1,048,576 | 25.5% |
| encoder_copy_gate | 524,800 | 12.8% |
| decoder_cross_attention | 163,840 | 4.0% |
| decoder_ffn | 131,072 | 3.2% |
| encoder_digit_embeddings | 82,944 | 2.0% |
| coordinate_embedding | 65,536 | 1.6% |
| decoder_self_attention | 65,536 | 1.6% |
| decoder_copy_gate | 32,896 | 0.8% |
| decoder_embedding | 10,880 | 0.3% |
| output_projection | 10,880 | 0.3% |
| encoder_depthwise_conv | 3,072 | 0.1% |
| numerical_feature_projection | 2,560 | 0.1% |
| encoder_special_embeddings | 2,048 | 0.0% |
| encoder_normalization | 2,048 | 0.0% |
| zero_coordinate_vector | 512 | 0.0% |
| decoder_normalization | 512 | 0.0% |
| sparse_attention_bias | 4 | 0.0% |

**C-alt - Salsa2-SSM (state-space instead of attention)** (4,004,736)

| component | parameters | share |
|---|---:|---:|
| encoder_gated_mlp | 2,359,296 | 58.9% |
| encoder_ssm | 549,376 | 13.7% |
| encoder_copy_gate | 524,800 | 13.1% |
| decoder_cross_attention | 163,840 | 4.1% |
| decoder_ffn | 131,072 | 3.3% |
| encoder_digit_embeddings | 82,944 | 2.1% |
| coordinate_embedding | 65,536 | 1.6% |
| decoder_self_attention | 65,536 | 1.6% |
| decoder_copy_gate | 32,896 | 0.8% |
| decoder_embedding | 10,880 | 0.3% |
| output_projection | 10,880 | 0.3% |
| numerical_feature_projection | 2,560 | 0.1% |
| encoder_special_embeddings | 2,048 | 0.1% |
| encoder_normalization | 2,048 | 0.1% |
| zero_coordinate_vector | 512 | 0.0% |
| decoder_normalization | 512 | 0.0% |

## 2. Hypothesis H4 - is a state-space component motivated?

**Answer: no, and this is arithmetic rather than opinion.**

Per encoder pass the four projections cost `4*L*d^2` MACs and the attention
product costs `2*L^2*d`. They are equal at `L = 2d = 1024`, which is n = 511 under
representation R and n = 1022 with coordinate tokens.
**Every dimension this project cares about, including n=128, sits far below that**
**crossover**, so attention is not the bottleneck and removing it buys almost nothing.

The measured attention share of encoder MACs confirms it: **6.7% at n=128 for
candidate A and 3.5% for the coordinate-token models**. Removing attention
entirely could not recover more than that, and would cost the only content-based
mixing in the encoder.

Candidate C-alt also turns out to be *worse* on the metric it was supposed to win:
at n=128 it needs **2,216 MiB** of activation memory at batch 64 against 1,374 MiB
for B and 1,493 MiB for A, because a diagonal SSM keeps a `d x N` state per
position. The state-space option loses on parameters, on memory, and on inductive
bias, and wins only a compute term that was never the bottleneck.

Two further CPU-specific points argue against a state-space encoder here:

1. A selective/diagonal scan is **sequential in L**. Attention at these lengths is
   one dense `L x L` matmul, which BLAS executes at near-peak on every CPU this
   project targets. A short scan is latency-bound and loses to a small matmul.
2. The task needs **content-based** global mixing: `b = sum_i a_i s_i` requires
   pairing coordinate i with secret bit i regardless of distance. A diagonal SSM
   mixes by recency, not by content, which is the wrong inductive bias for a sum
   over an unordered support.

Candidate C-alt is included and costed precisely so this conclusion is a
measured comparison rather than an assumption.

## 3. Scaling to n = 128

`B-lite` is not a fifth architecture: it is candidate B with `T_e=2` instead of 4.
Loops reuse one parameter set, so B and B-lite have byte-for-byte identical
weights and differ only in a runtime knob. The row is there to separate what the
shorter sequence buys from what the extra depth spends.

### Representation R (base 81, no separator)

| candidate | n | L_in (R) | GMACs/sample | attention share | activation MiB @ batch 64 |
|---|---:|---:|---:|---:|---:|
| A | 12 | 26 | 0.1922 | 0.0072 | 131.9 |
| A | 30 | 62 | 0.4630 | 0.0170 | 315.5 |
| A | 50 | 102 | 0.7700 | 0.0277 | 531.5 |
| A | 70 | 142 | 1.0836 | 0.0381 | 759.9 |
| A | 90 | 182 | 1.4037 | 0.0483 | 1000.8 |
| A | 128 | 258 | 2.0301 | 0.0672 | 1493.1 |
| B | 12 | 14 | 0.2063 | 0.0039 | 140.2 |
| B | 30 | 32 | 0.4740 | 0.0089 | 317.8 |
| B | 50 | 52 | 0.7744 | 0.0143 | 521.1 |
| B | 70 | 72 | 1.0782 | 0.0197 | 730.6 |
| B | 90 | 92 | 1.3852 | 0.0250 | 946.4 |
| B | 128 | 130 | 1.9776 | 0.0350 | 1373.6 |
| B-lite | 12 | 14 | 0.1032 | 0.0039 | 72.9 |
| B-lite | 30 | 32 | 0.2370 | 0.0089 | 161.8 |
| B-lite | 50 | 52 | 0.3872 | 0.0143 | 263.5 |
| B-lite | 70 | 72 | 0.5391 | 0.0197 | 368.3 |
| B-lite | 90 | 92 | 0.6926 | 0.0250 | 476.3 |
| B-lite | 128 | 130 | 0.9888 | 0.0350 | 690.1 |
| C | 12 | 14 | 0.1991 | 0.0040 | 125.4 |
| C | 30 | 32 | 0.4575 | 0.0092 | 281.8 |
| C | 50 | 52 | 0.7477 | 0.0148 | 458.5 |
| C | 70 | 72 | 1.0412 | 0.0204 | 638.3 |
| C | 90 | 92 | 1.3379 | 0.0259 | 821.3 |
| C | 128 | 130 | 1.9108 | 0.0362 | 1177.6 |
| C-alt | 12 | 14 | 0.1922 | 0.0000 | 243.6 |
| C-alt | 30 | 32 | 0.4394 | 0.0000 | 549.8 |
| C-alt | 50 | 52 | 0.7139 | 0.0000 | 889.9 |
| C-alt | 70 | 72 | 0.9885 | 0.0000 | 1230.1 |
| C-alt | 90 | 92 | 1.2631 | 0.0000 | 1570.2 |
| C-alt | 128 | 130 | 1.7849 | 0.0000 | 2216.5 |

### Representation P (base 81, with separator)

| candidate | n | L_in (P) | GMACs/sample | attention share | activation MiB @ batch 64 |
|---|---:|---:|---:|---:|---:|
| A | 12 | 37 | 0.2744 | 0.0102 | 186.9 |
| A | 30 | 91 | 0.6849 | 0.0248 | 470.8 |
| A | 50 | 151 | 1.1550 | 0.0404 | 813.0 |
| A | 70 | 211 | 1.6399 | 0.0556 | 1183.3 |
| A | 90 | 271 | 2.1396 | 0.0703 | 1581.8 |
| A | 128 | 385 | 3.1295 | 0.0970 | 2416.3 |
| B | 12 | 13 | 0.1915 | 0.0036 | 130.5 |
| B | 30 | 31 | 0.4590 | 0.0086 | 307.8 |
| B | 50 | 51 | 0.7593 | 0.0140 | 510.8 |
| B | 70 | 71 | 1.0629 | 0.0194 | 720.0 |
| B | 90 | 91 | 1.3698 | 0.0248 | 935.4 |
| B | 128 | 129 | 1.9619 | 0.0347 | 1362.0 |
| B-lite | 12 | 13 | 0.0958 | 0.0036 | 68.0 |
| B-lite | 30 | 31 | 0.2295 | 0.0086 | 156.8 |
| B-lite | 50 | 51 | 0.3797 | 0.0140 | 258.3 |
| B-lite | 70 | 71 | 0.5315 | 0.0194 | 363.0 |
| B-lite | 90 | 91 | 0.6849 | 0.0248 | 470.8 |
| B-lite | 128 | 129 | 0.9809 | 0.0347 | 684.3 |
| C | 12 | 13 | 0.1849 | 0.0037 | 116.8 |
| C | 30 | 31 | 0.4431 | 0.0089 | 273.0 |
| C | 50 | 51 | 0.7331 | 0.0145 | 449.6 |
| C | 70 | 71 | 1.0264 | 0.0201 | 629.3 |
| C | 90 | 91 | 1.3230 | 0.0256 | 812.1 |
| C | 128 | 129 | 1.8956 | 0.0360 | 1168.0 |
| C-alt | 12 | 13 | 0.1785 | 0.0000 | 226.6 |
| C-alt | 30 | 31 | 0.4256 | 0.0000 | 532.8 |
| C-alt | 50 | 51 | 0.7002 | 0.0000 | 872.9 |
| C-alt | 70 | 71 | 0.9748 | 0.0000 | 1213.1 |
| C-alt | 90 | 91 | 1.2494 | 0.0000 | 1553.2 |
| C-alt | 128 | 129 | 1.7711 | 0.0000 | 2199.5 |

**Representation P costs 50% more positions than R for digit-token models**
(`3n+1` against `2n+2`). At n=128 that is 385 positions against 258, **3.13 GMACs**
**against 2.03, and 2,416 MiB of activations against 1,493 MiB** at batch 64 -- a
54% compute surcharge. It buys nothing this project has measured: the phase-3.5
audit established that the released code itself uses no separator. For a
coordinate-token front end the separator is absorbed entirely -- there is nothing
left to separate -- so P and R converge to the same length. **Recommendation: R.**

## 4. Numerical representation - what is meaningful and what is not

### Measured basis for the decision

- Spearman(value, digit0) = +0.2219. Under `lsb_first` this is the token the encoder reads **first** for every
  coordinate, and it carries little ordinal information about the value it helps
  encode -- it is a sawtooth that resets three times across Z_q.
- Spearman(value, digit1) = +0.9482. All of the ordinal content is in the second token.
- `digit0 == 0` holds for 4 of 251 residues and `digit1 == 0` for 81, but
  only **1** residue satisfies both. Detecting a zero coordinate is a
  **conjunction across two sequence positions** that the encoder must spend
  attention to compute, once per coordinate, on every forward pass.

### Verdicts

| feature | scalars | verdict | reason |
|---|---:|---|---|
| `signed centered residue  ((x + q//2) mod q - q//2) / (q/2)` | 1 | **KEEP** | b = sum_{i in supp} a_i + e mod q is integer addition followed by one reduction, so the integer magnitude of a_i is the quantity the model must actually add. Centering puts it symmetric about zero, which suits RMSNorm and bias-free FFNs, and matches the centered error term. |
| `normalized residue x/q` | 0 | **REJECT - redundant** | An affine function of the signed centered residue. A linear layer converts one to the other, so including both adds parameters and no information. |
| `Fourier pair(s) cos/sin(2*pi*k*x/q), k = 1..K` | 2 | **KEEP (K=1 only)** | The characters of Z_q are the irreducible representations of the group the task actually lives in: addition mod q becomes addition of angles. This is the one genuinely non-affine feature that encodes wraparound, which no polynomial in x can express. K=1 is kept; higher harmonics are left for the model to build, since q=251 is prime and the k=1 pair already separates every residue. |
| `modular distance to zero  min(x, q-x)` | 0 | **REJECT - redundant** | Determined by the k=1 Fourier pair: min(x, q-x) is a monotone function of cos(2*pi*x/q). Adding it separately is duplication. |
| `magnitude feature |centered(x)|` | 0 | **REJECT - duplicate** | It is the modular distance to zero under another name. |
| `zero / nonzero indicator  1[x == 0]` | 1 | **KEEP** | The single most defensible sparsity feature. In base-81 lsb-first, 'this coordinate is zero' is a CONJUNCTION across two token positions (digit0 == 0 AND digit1 == 0), which the encoder must spend attention to compute. Phase 12 measured that the model stops using its input somewhere between 8 and 6 nonzero coordinates; this makes the defining bit of that regime a free input rather than something to be inferred. |
| `absolute coordinate index embedding` | 0 | **KEEP as an embedding table, not a scalar** | b = sum_i a_i s_i binds coordinate i to secret bit s_i, so the task needs ABSOLUTE coordinate identity. RoPE supplies only RELATIVE position. This is a real gap in the current model and it costs n_max * d, once, independent of n. |
| `digit embedding (existing LatticeCodec tokens)` | 0 | **KEEP** | Retained unchanged so the v2 model stays compatible with the existing codec, the existing target format and the existing recovery interface, and so a token-only ablation arm remains available. |

**Kept: 4 scalars per coordinate** -- the zero indicator, the signed centered
residue, and the k=1 Fourier pair -- plus the existing digit embeddings and an
absolute coordinate embedding. Everything else on the candidate list is either an
affine function of what is already there (`x/q`) or the same quantity renamed
(`magnitude`, `modular distance to zero`). Rejecting them is not conservatism; it
is refusing to pay parameters for duplicated information.

**Scientific caveat.** Adding numerical features changes the representation, so v2
results are no longer directly comparable to the token-only setting the original
SALSA uses. A token-only ablation arm must be kept and reported alongside, or the
comparison to phases 6-12 is broken.

## 5. Sparsity

**How zeros are represented today.** Under R a zero coordinate becomes 2 copies
of the same zero-digit token. At n=128 with a weight-3 secret-like sparse probe,
250 of 258 encoder positions carry the identical symbol. Phase 11 measured the
consequence directly: the zero-digit token fraction is 0.156 on training rows
against 0.846 on probes.

| n | nnz | R length | zero-digit tokens | fraction | coordinate-token length | fraction |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 12 | 26 | 0 | 0.000 | 14 | 0.000 |
| 12 | 6 | 26 | 12 | 0.462 | 14 | 0.429 |
| 12 | 1 | 26 | 22 | 0.846 | 14 | 0.786 |
| 12 | 1 | 26 | 22 | 0.846 | 14 | 0.786 |
| 30 | 30 | 62 | 0 | 0.000 | 32 | 0.000 |
| 30 | 15 | 62 | 30 | 0.484 | 32 | 0.469 |
| 30 | 3 | 62 | 54 | 0.871 | 32 | 0.844 |
| 30 | 1 | 62 | 58 | 0.935 | 32 | 0.906 |
| 50 | 50 | 102 | 0 | 0.000 | 52 | 0.000 |
| 50 | 25 | 102 | 50 | 0.490 | 52 | 0.481 |
| 50 | 6 | 102 | 88 | 0.863 | 52 | 0.846 |
| 50 | 1 | 102 | 98 | 0.961 | 52 | 0.942 |
| 70 | 70 | 142 | 0 | 0.000 | 72 | 0.000 |
| 70 | 35 | 142 | 70 | 0.493 | 72 | 0.486 |
| 70 | 8 | 142 | 124 | 0.873 | 72 | 0.861 |
| 70 | 1 | 142 | 138 | 0.972 | 72 | 0.958 |
| 90 | 90 | 182 | 0 | 0.000 | 92 | 0.000 |
| 90 | 45 | 182 | 90 | 0.495 | 92 | 0.489 |
| 90 | 11 | 182 | 158 | 0.868 | 92 | 0.859 |
| 90 | 1 | 182 | 178 | 0.978 | 92 | 0.967 |
| 128 | 128 | 258 | 0 | 0.000 | 130 | 0.000 |
| 128 | 64 | 258 | 128 | 0.496 | 130 | 0.492 |
| 128 | 16 | 258 | 224 | 0.868 | 130 | 0.862 |
| 128 | 1 | 258 | 254 | 0.984 | 130 | 0.977 |

**Are zero and nonzero efficiently distinguishable?** Not today -- it is the
two-position conjunction measured above. With a 1-bit input feature it is free.

**Is attention diluted by zeros?** Yes, structurally. Softmax over L keys assigns
mass to every position; when 97% of positions are an identical uninformative
symbol, the informative keys must win by logit margin alone, and there is nothing
in the architecture that lets the model cheaply say 'ignore these'. The proposed
fix is one learned scalar per head added to the attention logit of any zero
coordinate (8 parameters in total), which lets the encoder learn to suppress or
attend to zeros as the data demands, without hard-coding either.

**Is coordinate-wise processing before global mixing useful?** Yes, and it is
already the main structural change: folding the `width` digit tokens of a
coordinate into one position is exactly per-coordinate processing before mixing.
It halves the sequence, quarters the attention term, and makes position equal
coordinate index so absolute identity becomes expressible.

## 6. Hypotheses

| hypothesis | verdict from this phase | basis |
|---|---|---|
| H1: GatedUT is sufficient, training strategy is the limit | **Not refuted, and partly supported** | Phase 11 rated insufficient training High. The n=12 run saw 100,032 samples against the paper's ~3.9M, and sample reuse was off. This phase cannot settle it, and a v2 architecture does not excuse leaving it untested. |
| H2: numerical-aware encoding improves sparse generalization | **Plausible, mechanism identified, untested** | The zero indicator removes a measured two-position conjunction, and the sparse attention bias gives a mechanism for the dilution that phase 12 measured. Neither is evidence that it works. |
| H3: a hybrid improves CPU efficiency without losing global interactions | **Weakly supported at best** | The attention share of MACs is small at every n <= 128, so there is little CPU efficiency to win. Candidate C costs about the same and adds two new block types. |
| H4: a pure state-space model is not necessarily advantageous | **Supported** | Crossover at L = 2d = 1024, far above n=128. Plus: scans are sequential on CPU, and diagonal SSMs mix by recency where the task needs content-based mixing. |

## 7. Recommendation

### Recommended: candidate B, `Salsa2-NACT`

**4,241,288 parameters** (16.18 MiB fp32), inside the 4-5M budget with no
padding parameters.

**1. Why should it handle sparse inputs better?** Three specific mechanisms, each
tied to something measured rather than to intuition: the zero indicator turns a
two-position conjunction into a 1-bit input; the per-head sparse attention bias
gives the encoder an explicit, learnable way to discount zero coordinates instead
of relying on logit margin; and the coordinate-token front end halves the number
of positions a sparse input floods with identical symbols.

**2. Why does it stay within 4-5M?** Because the encoder body is unchanged and it
is 89.9% of the budget. The entire new front end costs about 110,088 parameters, spent in
the cheapest region of the model. Encoder loops rise from 2 to 4, which increases
effective depth at **zero** parameter cost -- that is what the shared layer is for.

**3. Why is it CPU-friendly?** Halving the sequence halves everything linear in L,
which is where essentially all the compute is. The `B-lite` row makes the size of
that effect explicit -- **identical weights to B, loops held at T_e=2 to match A**:

| n=128, representation R | A | B-lite (T_e=2) | B (T_e=4) |
|---|---:|---:|---:|
| encoder positions | 258 | 130 | 130 |
| GMACs / sample | 2.03 | **0.99** | 1.98 |
| activation MiB @ batch 64 | 1493 | **690** | 1374 |
| effective encoder depth | 2 | 2 | **4** |

So the shorter sequence is worth **51% of the compute and 54% of the activation**
**memory at equal depth**, or -- the setting recommended here -- **twice the**
**effective depth at the same compute as today**. It is one or the other, not both;
T_e is a runtime knob costing no parameters, so the choice can be made per
experiment. No new operator types either: embeddings, dense matmuls and softmax,
all BLAS-friendly, no sequential scans, no custom kernels, no CUDA.

**4. Why is it better suited to n=128?** Sequence length at n=128 falls from 258
to 130, and activation memory at batch 64 from 1,493 MiB to 690 MiB at equal
depth -- the difference between a CPU training run that fits comfortably and one
that does not. The coordinate embedding is a fixed `n_max x d` table, so **the**
**parameter count is identical at every n from 12 to 128**: one model covers the
whole dimension sweep, and n=128 needs no re-architecting. Dimensions beyond 128
would need the table extended, which is the one place n enters the parameter count.

**5. What of the current GatedUT is retained?** The parameter-shared looped
encoder, the copy gate, RMSNorm, the bias-free 2-matrix GELU FFN, RoPE, the
encoder/decoder width asymmetry (512/128), the decoder in its entirety, and the
existing `LatticeCodec` tokens and target format. Phase 11 found the supplied
reference *disabled* looping and gating; keeping them keeps v2 closer to the
published architecture than that fork.

**6. What is replaced?** Only the input front end: `width` digit tokens per
coordinate become one coordinate token; RoPE-only positioning gains an absolute
coordinate embedding; and attention gains a per-head zero-coordinate bias. The
encoder body, the decoder and the output interface are untouched.

**7. What new research contribution?** A named, falsifiable claim: *that the
sparse-input failure measured in phase 12 is a representation defect rather than a
capacity defect, and that per-coordinate tokenisation with a zero indicator and a
learned sparse attention bias moves the zero-lift crossing point toward sparser
inputs at constant parameter count.* Phase 12 already established the measurement
that would confirm or refute it, on the same axis, with the same baselines.

### Exact specification

```
encoder   d = 512, heads = 8, parameter sets = 1, loops T_e = 4, gated, RMSNorm,
          RoPE on self-attention, FFN multiplier 4 (hidden 2048), no biases
decoder   d = 128, heads = 4, parameter sets = 1, loops T_d = 2, gated, RMSNorm,
          RoPE on self-attention only, cross-attention unrotated, FFN hidden 512
sharing   1 encoder set reused 4 times; 1 decoder set reused twice.
          Loops cost no parameters and are runtime knobs.

input     one token per coordinate, i = 0 .. n-1, built as
            e_i = sum_{w<2} DigitEmb_w[digit_w(a_i)]
                + W_num @ [ 1[a_i == 0],
                            centered(a_i) / (q/2),
                            cos(2*pi*a_i/q), sin(2*pi*a_i/q) ] + b_num
                + CoordEmb[i]
            zero coordinates additionally blend in a learned zero vector
          plus <bos> and <eos> from a 4-entry special table -> L_in = n + 2
          attention logits receive beta_head * 1[a_j == 0] on key j

output    UNCHANGED. Same LatticeCodec target sequence <bos> d0 d1 <eos>,
          same 85-token vocabulary, same greedy/beam decode, same
          decode_generated_ids contract, so phase-10 recovery and phase-12
          generalization scripts run against it without modification.
```

**Expected parameter count: 4,241,288** (vs 4,131,200 today, +110,088). Exact PyTorch counting must
confirm this before any training, using the existing three-way check in
`salsa.models.parameter_count` (actual, component breakdown, analytical).

### What this recommendation does not claim

Nothing here has been trained. H2 is a mechanism, not a result. The honest
sequencing is that H1 remains live and cheap to test: the current GatedUT has
never been trained past 100,032 samples with sample reuse enabled, and phase 11
rated insufficient training a High-confidence cause. **A v2 architecture should
not be adopted on the strength of a design document while the training-budget
explanation for the same failure is still untested.** The two are separable and
should be separated.

## Artifacts

- `architecture_comparison.md` (this file)
- `architecture_comparison.json`
- `architecture_comparison.csv`
