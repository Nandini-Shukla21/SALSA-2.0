"""Executable fidelity audit: original SALSA versus Salsa 2.0.

Read-only.  Nothing is trained and no implementation file is touched.  Where a
comparison can be *run* rather than argued, it is run: the original
``encoders.py`` is loaded directly from the reference checkout and its output
compared token for token against ours, and the original matrix and error
constructions are reproduced from their source and compared numerically.

Provenance matters here.  In the reference checkout only ``encoders.py``,
``generators.py`` and ``envs/__init__.py`` still carry the initial-commit
timestamp; ``lattice.py``, ``evaluator.py``, ``trainer.py``, ``optim.py``,
``train.py`` and ``model/transformer.py`` were modified later, and
``transformer.py`` carries "Modern SALSA" markers that *disable* the Universal
Transformer looping and the copy gate.  Data and encoding are therefore audited
against genuine original code; model and training stages are audited against a
fork that no longer matches the paper, and that is recorded as such.
"""

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

ORIGINAL = Path("c:/Users/Nandini Shukla/OneDrive/Desktop/SALSA-main/folder")
OUTPUT_DIR = REPO_ROOT / "results" / "original_fidelity_audit"

from salsa.data import (  # noqa: E402
    IntegerEncoder,
    LatticeCodec,
    circulant_matrix,
    generate_binary_secret,
    generate_error,
    negacyclic_matrix,
)

Q, BASE, N = 251, 81, 12

#: Difference classes required by the audit brief.
CLASSES = {
    "A": "Exact fidelity",
    "B": "Intentional improvement",
    "C": "Harmless implementation difference",
    "D": "Potentially behaviour-changing difference",
    "E": "Major incompatibility",
}


def load_original_encoder():
    """Load the unmodified original ``encoders.py`` straight from the checkout."""
    path = ORIGINAL / "src" / "envs" / "encoders.py"
    spec = importlib.util.spec_from_file_location("orig_encoders", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Params:
    """Minimal stand-in for the original argparse namespace."""

    def __init__(self, base=BASE, q=Q, n=N, balanced=False, no_sep=True):
        self.input_int_base = self.output_int_base = base
        self.balanced_base = balanced
        self.no_separator = no_sep
        self.Q = q
        self.N = n


def original_matrix(a: np.ndarray, q: int) -> np.ndarray:
    """Reproduce ``generators.RLWE.get_sample``'s matrix construction exactly.

    From the unmodified source::

        c = circulant(a)                    # C[i, j] = a[(i - j) mod n]
        tri = np.triu_indices(N, 1)
        c[tri] *= -1                        # negate the strict upper triangle
        c = c % self.Q
    """
    n = len(a)
    matrix = np.empty((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(n):
            matrix[i, j] = a[(i - j) % n]
    matrix[np.triu_indices(n, 1)] *= -1
    return matrix % q


def rounded_gaussian_pmf(sigma: float, bound: int) -> np.ndarray:
    """Exact pmf of ``round(N(0, sigma))`` -- what the original samples."""
    support = np.arange(-bound, bound + 1)
    cdf = lambda x: 0.5 * (1.0 + math.erf(x / (sigma * math.sqrt(2.0))))
    return np.array([cdf(k + 0.5) - cdf(k - 0.5) for k in support])


def discrete_gaussian_pmf(sigma: float, bound: int) -> np.ndarray:
    """Exact pmf of the truncated discrete Gaussian -- what Salsa 2.0 samples."""
    support = np.arange(-bound, bound + 1, dtype=np.float64)
    weights = np.exp(-(support ** 2) / (2.0 * sigma * sigma))
    return weights / weights.sum()


#: Both direct-recovery pipelines, reconstructed from source.
PIPELINES = {
    "original": """for K in [239145, 42899, Q, 3Q+7, 42900, 5 random in [Q,10Q), 71,92,101,193,241]:
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
# NOTE: verification (paper 4.4, residual std test) is NOT implemented here.""",
    "salsa_2_0": """for K in configured sweep [1,31,63,94,125,126,157,188,220,250]:
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
# the all-zeros baseline that a weight-h secret gives away for free.""",
    "comparison": [
        {"step": "probe geometry", "original": "identity(N) * K",
         "salsa_2_0": "identity(n) * (K mod q)", "class": "B"},
        {"step": "probe value range", "original": "K mod B^int_len; may exceed q",
         "salsa_2_0": "K mod q; always in Z_q", "class": "D"},
        {"step": "target b", "original": "built from the TRUE SECRET",
         "salsa_2_0": "never constructed", "class": "B"},
        {"step": "encoding", "original": "input_encoder.encode, no separator",
         "salsa_2_0": "identical token body", "class": "A"},
        {"step": "boundaries", "original": "eos_index at both ends",
         "salsa_2_0": "<bos> ... <eos>", "class": "C"},
        {"step": "inference", "original": "generate_beam(beam_size=1)",
         "salsa_2_0": "greedy argmax", "class": "C"},
        {"step": "output decode", "original": "decode(hyp[::-1]); failure -> -1",
         "salsa_2_0": "decode_generated_ids; failure flagged and counted", "class": "B"},
        {"step": "bit decision", "original": "mean/mode/softmax threshold",
         "salsa_2_0": "ring distance to 0 vs K", "class": "B"},
        {"step": "polarity", "original": "vector AND inverse, best vs TRUE SECRET",
         "salsa_2_0": "anchored on K; no inverse, no ground truth", "class": "B"},
        {"step": "multiple K", "original": "10 values, any success counts",
         "salsa_2_0": "10 values + separation-weighted vote + secret-free selection",
         "class": "B"},
        {"step": "success", "original": "all N coordinates match",
         "salsa_2_0": "all n coordinates match, plus all-zeros baseline reported",
         "class": "B"},
        {"step": "verification", "original": "NOT implemented in evaluator.py",
         "salsa_2_0": "NOT implemented (out of phase-10 scope)", "class": "E"},
    ],
}


def run_checks() -> List[Dict[str, Any]]:
    """Run every executable comparison and return one row per stage."""
    orig = load_original_encoder()
    rows: List[Dict[str, Any]] = []

    def add(stage: str, original: str, salsa2: str, identical: bool,
            klass: str, difference: str, consequence: str, evidence: str = "") -> None:
        rows.append({
            "stage": stage, "original_salsa": original, "salsa_2_0": salsa2,
            "identical": identical, "class": klass,
            "class_name": CLASSES[klass], "difference": difference,
            "consequence": consequence, "evidence": evidence,
        })

    # -- 1. coefficient range -------------------------------------------- #
    # generators.py: maxQ_prob default 0 so the branch never fires; percQ_bound
    # default 1.0 -> maxQ = 1.0 * Q; a = rng.randint(0, maxQ) -> [0, Q-1].
    add("1a. coefficient range of a",
        "rng.randint(0, percQ_bound*Q) with percQ_bound=1.0, maxQ_prob=0 -> [0, 250]",
        "generate_uniform_matrix, uniform [0, q) -> [0, 250]",
        True, "A", "none",
        "Probe and training coefficient ranges agree.",
        "generators.py get_sample; salsa/data/lwe.py")

    # -- 1b. sample reuse -------------------------------------------------- #
    add("1b. sample reuse",
        "reuse=True, times_reused=10, num_reuse_samples=10000, K=1 BY DEFAULT",
        "sample_reuse=1 in every completed Salsa 2.0 run (each sample fresh)",
        False, "D",
        "The original recycles each distinct LWE instance ~10 times; ours never reuses.",
        "For a fixed number of gradient examples the original needs ~10x FEWER "
        "DISTINCT samples. The paper's 'log2 samples' counts distinct instances, so "
        "our per-sample budget is not directly comparable to theirs, and our "
        "sample-requirement extrapolations may be pessimistic by up to ~10x.",
        "generators.py RLWE.__init__ / get_reused_sample; lattice.py register_args")

    # -- 2. secret generation ---------------------------------------------- #
    # Original: repeat h times, draw idx until s[idx] != 1 -> uniform support.
    original_supports = set()
    rng = np.random.default_rng(0)
    for _ in range(20000):
        s = np.zeros(N, dtype=np.int64)
        for _ in range(2):
            while True:
                idx = int(rng.integers(N))
                if s[idx] != 1:
                    s[idx] = 1
                    break
        original_supports.add(tuple(np.flatnonzero(s)))
    ours_supports = {tuple(np.flatnonzero(generate_binary_secret(N, 2, seed=s)))
                     for s in range(20000)}
    same_support_set = original_supports == ours_supports
    add("2. secret generation",
        "rejection loop: draw idx until unset, h times -> exactly h ones",
        "rng.choice(n, h, replace=False) -> exactly h ones",
        True, "C",
        "Different sampling mechanism, identical distribution (uniform over "
        "size-h supports).",
        "None. Both give exactly h ones with a uniform support.",
        f"both reach all {len(original_supports)} = C(12,2) supports: {same_support_set}")

    # -- 3. error generation ----------------------------------------------- #
    bound = 30
    pmf_round = rounded_gaussian_pmf(3.0, bound)
    pmf_exact = discrete_gaussian_pmf(3.0, bound)
    support = np.arange(-bound, bound + 1)
    tv = 0.5 * np.abs(pmf_round - pmf_exact).sum()
    std_round = math.sqrt(float((pmf_round * support ** 2).sum()))
    std_exact = math.sqrt(float((pmf_exact * support ** 2).sum()))
    add("3. error generation",
        "np.int64(rng.normal(0, sigma).round()) -- ROUNDED Gaussian",
        "exact truncated discrete Gaussian, P(x) ~ exp(-x^2/2s^2)",
        False, "B",
        f"Different distributions; total variation distance {tv:.5f}, "
        f"std {std_round:.4f} vs {std_exact:.4f}.",
        "Negligible. Both are centred, integer, and of essentially the same width; "
        "the LWE relation and every metric are unaffected.",
        f"TV={tv:.5f}, std_rounded={std_round:.4f}, std_discrete={std_exact:.4f}")

    # -- 4. matrix construction -------------------------------------------- #
    a = np.array([67, 77, 10, 18, 200, 3])
    original_m = original_matrix(a, Q)
    ours_circ = circulant_matrix(a, q=Q)
    ours_nega = negacyclic_matrix(a, q=Q)
    matches_circ = bool(np.array_equal(original_m, ours_circ))
    matches_nega = bool(np.array_equal(original_m, ours_nega))
    matches_nega_t = bool(np.array_equal(original_m, ours_nega.T))
    add("4. matrix construction",
        "circulant(a) with the strict UPPER TRIANGLE NEGATED, then mod q "
        "(i.e. NEGACYCLIC, despite the paper saying 'circulant' and the option "
        "being named circ_rlwe)",
        "rlwe_variant=circulant (no negation) -- what every completed run used",
        False, "D",
        "The original applies a sign flip on wrap-around; our trained runs did not. "
        "The original equals the TRANSPOSE of our negacyclic variant.",
        "The training distribution differs. Row sets are not the same matrices, so "
        "the function the model must learn is different, though of the same family "
        "and difficulty. Salsa 2.0 implements negacyclic and can select it by "
        "config; no completed run used it.",
        f"orig==circulant:{matches_circ} orig==negacyclic:{matches_nega} "
        f"orig==negacyclic.T:{matches_nega_t}")

    # -- 5. integer encoding ----------------------------------------------- #
    matches = True
    samples = []
    for base in (2, 7, 81):
        encoder_o = orig.Encoder(Params(base=base), single=True, output=True)
        encoder_n = IntegerEncoder(base=base, modulus=Q, digit_order="lsb_first")
        if encoder_o.int_len != encoder_n.width:
            matches = False
        for value in (0, 1, 31, 80, 81, 125, 250):
            a_tok, b_tok = encoder_o.write_int(value), encoder_n.encode(value)
            matches &= a_tok == b_tok
            if base == 81:
                samples.append((value, a_tok, b_tok))
    add("5. integer encoding",
        "write_int_normal: fixed int_len = floor(log(Q, B)) + 1 digits, LSB-first",
        "IntegerEncoder(fixed_width, digit_order='lsb_first'), width = "
        "smallest w with B^w >= q",
        matches, "A" if matches else "E",
        "Width formulas differ only when q is an exact power of B (never for prime q).",
        "None for q=251.",
        f"token-for-token identical across bases 2/7/81: {matches}; "
        f"e.g. 250 -> {samples[-1][1]}")

    # -- 6. input sequence construction ------------------------------------ #
    encoder_in = orig.Encoder(Params(no_sep=True), single=False, output=False)
    original_body = encoder_in.encode([67, 77, 10, 18])
    codec_r = LatticeCodec(n=4, q=Q, base=BASE, separator=False,
                           digit_order="lsb_first")
    ours_seq = codec_r.encode_input([67, 77, 10, 18])
    body_same = original_body == ours_seq[1:-1]
    add("6. input sequence construction",
        "encode(row) then batch_sequences wraps with eos_index at BOTH ends; "
        "no_separator defaults to TRUE so '|' is never emitted",
        "<bos> + digits + <eos>, separator=false under Representation R",
        body_same, "A" if body_same else "E",
        "Identical token bodies; the two boundary tokens have different names and "
        "the original reuses one id for both ends.",
        "None. Sequence length and content agree exactly.",
        f"original body == ours[1:-1]: {body_same}; length {len(ours_seq)} both")

    # -- 7. output sequence construction ------------------------------------ #
    encoder_out = orig.Encoder(Params(), single=True, output=True)
    out_same = encoder_out.write_int(31) == codec_r.encode_output(31)[1:-1]
    add("7. output sequence construction",
        "y = output_encoder.write_int(b) (LSB-first digits), wrapped with "
        "eos at both ends by batch_sequences",
        "<bos> + digits + <eos>",
        out_same, "A" if out_same else "E",
        "Same digits, same order, same length.",
        "None.",
        f"identical: {out_same}; both 2 digits + 2 markers = 4 tokens")

    # -- 8. vocabulary ------------------------------------------------------ #
    words = ["<s>", "</s>", "<pad>"] + ["|", "+"] + sorted({str(i) for i in range(BASE)})
    add("8. vocabulary",
        f"{len(words)} ids: 3 specials + '|' + '+' + {BASE} digits; eos_index=0 "
        f"('<s>'), pad_index=1 ('</s>'); '<pad>' (id 2) NEVER used; '+' never emitted",
        f"{codec_r.vocabulary.size} ids: <pad>=0 <bos>=1 <eos>=2 <sep>=3 + {BASE} digits",
        False, "C",
        "86 vs 85 ids; the original carries two entries it never emits under the "
        "default settings and uses one token for both sequence ends.",
        "None functionally. Embedding tables differ in size by one row, which is why "
        "the phase-4 counts moved by 768 parameters when V was corrected to 85.",
        "lattice.py words/eos_index/pad_index")

    # -- 9. model architecture ---------------------------------------------- #
    add("9. model architecture",
        "PAPER: gated Universal Transformer, 1024/512, 16-32/4 heads, shared layer "
        "looped 2 (enc) and 8 (dec), copy gate, ~51M parameters. "
        "SUPPLIED transformer.py: looping and gating DISABLED ('obsolete in Modern "
        "SALSA'), plain block stack, RoPE, RMSNorm",
        "Gated Universal Transformer, 512/128, 8/4 heads, 1 shared layer per side, "
        "T_e=T_d=2, copy gate, RoPE, RMSNorm, 4,131,200 parameters",
        False, "E",
        "12.3x fewer parameters than the paper. The supplied reference has REMOVED "
        "the UT looping and copy gate that the paper credits with a 14x sample-"
        "efficiency gain (Table 3), so it is not the paper's architecture either.",
        "Capacity is the largest single deviation from the paper. Salsa 2.0 is "
        "architecturally CLOSER to the paper than the supplied fork, since it keeps "
        "sharing and gating, but is far smaller.",
        "paper 4.2/5.3; transformer.py TransformerLayer + self.loops warnings")

    # -- 10. training objective --------------------------------------------- #
    add("10. training objective",
        "F.cross_entropy(scores, y, reduction='mean') over non-pad target tokens",
        "sequence_cross_entropy, ignore_index=pad, mean over scored tokens",
        True, "A", "none", "None.",
        "transformer.py predict(); salsa/training/losses.py")

    # -- 11. decoding ------------------------------------------------------- #
    add("11. decoding for recovery",
        "generate_beam(beam_size=params.beam_size), default beam_size=1 -> "
        "greedy in effect; hypotheses REVERSED before decode",
        "greedy_decode: argmax, encoder run once, no reversal needed because "
        "encode/decode are true inverses",
        True, "C",
        "Beam-1 equals greedy. The original's reversal compensates for write_int "
        "being LSB-first while parse_int reads MSB-first.",
        "None at beam_size=1, the configured default.",
        "evaluator.run_beam_generation; lattice.check_prediction hyp[::-1]")

    # -- 12. K probe construction ------------------------------------------- #
    add("12. K probe construction",
        "specialA = np.identity(N) * K; specialB = inner(specialA, TRUE SECRET). "
        "K values include 239145, 42899, Q, 3Q+7, random in [Q,10Q), 71/92/101/193/241",
        "build_probe_matrix = (K mod q) * I_n; NO b is constructed at all",
        False, "B",
        "Identical probe geometry. The original builds a target from the true "
        "secret (needed only by its batching); ours decodes from <bos> and never "
        "constructs one.",
        "Ours is a strictly cleaner attack: it cannot leak the secret through the "
        "target. Probe rows are identical.",
        "evaluator.eval_secret; salsa/recovery/direct.py")

    # -- 13. K reduction ----------------------------------------------------- #
    probes = {}
    encoder_probe = orig.Encoder(Params(), single=True, output=True)
    for K in (239145, 42899, 251, 760, 241):
        tokens = encoder_probe.write_int(K)
        effective = sum(int(d) * BASE ** i for i, d in enumerate(tokens))
        probes[K] = {"original_effective_value": effective,
                     "original_in_Zq": bool(effective < Q),
                     "salsa2_effective_value": K % Q}
    add("13. K reduction modulo q",
        "write_int takes K mod B^int_len (mod 6561 at base 81), NOT mod q, so a "
        "large K can be encoded as a value OUTSIDE Z_q",
        "K is reduced mod q, so the probe is always a legal element of Z_q",
        False, "D",
        "The original can feed the model values >= q that never occur in training.",
        "Only K mod q is mathematically meaningful (a.s = K*s_i mod q), so ours is "
        "correct. The original's out-of-range probes are an additional, unintended "
        "distribution shift on top of the intended one.",
        json.dumps(probes))

    # -- 14. candidate-bit decision ------------------------------------------ #
    add("14. candidate-bit decision",
        "3 rules (mean / mode / softmax-mean threshold) each producing a vector AND "
        "its inverse; the variant matching the TRUE SECRET best is reported",
        "anchor rule: assign to whichever of 0 or K mod q the prediction is nearer "
        "on the ring; no inverse, no ground truth",
        False, "B",
        "The original resolves polarity using the answer it is trying to find.",
        "The original's reported 'secret matching' figure is optimistic: it takes a "
        "max over 6 variants scored against the truth. Ours is a genuine attack "
        "statistic. This makes our numbers NOT directly comparable to theirs -- ours "
        "are stricter.",
        "evaluator.eval_secret match_counts/match_vecs argmax")

    # -- 15. success criterion ------------------------------------------------ #
    add("15. evaluation / success criterion",
        "success iff any of the 6 variants matches the true secret in ALL N "
        "coordinates (np.any(match_counts == N))",
        "exact recovery iff candidate == true secret in all n coordinates, plus an "
        "explicit all-zeros baseline so sparsity cannot masquerade as recovery",
        True, "B",
        "Same all-coordinates criterion; we add the trivial baseline the original "
        "does not report.",
        "None to the criterion. The added baseline prevents a weight-h secret's "
        "(n-h)/n free accuracy being read as partial success.",
        "evaluator.eval_secret; scripts/recover_secret.py")

    # -- 16. verification ------------------------------------------------------ #
    add("16. verification",
        "residuals r = a.s_hat - b mod q; accept if std(r) ~ sigma (3) rather than "
        "~ q/sqrt(12) (72.5). Described in paper 4.4; NOT implemented in the "
        "supplied evaluator.py",
        "NOT IMPLEMENTED (phase 10 scope explicitly excluded it)",
        False, "E",
        "Absent from both the supplied reference and Salsa 2.0.",
        "No candidate can currently be verified without consulting the true secret. "
        "This is the missing final stage of the attack pipeline.",
        "paper 4.4; grep finds no residual test in evaluator.py")

    return rows


def distribution_shift() -> Dict[str, Any]:
    """Quantify how far a K*e_i probe sits from the training distribution."""
    codec = LatticeCodec(n=N, q=Q, base=BASE, separator=False, digit_order="lsb_first")
    rng = np.random.default_rng(0)
    training = rng.integers(0, Q, size=(200_000, N), dtype=np.int64)

    training_zeros = (training == 0).sum(axis=1)
    probe_zeros = N - 1

    # P(a uniform training row has >= n-1 zero coordinates)
    p_zero = 1.0 / Q
    p_at_least = (
        math.comb(N, N - 1) * p_zero ** (N - 1) * (1 - p_zero) + p_zero ** N
    )

    src_train, _ = codec.encode_batch(training[:20_000])
    zero_id = codec.vocabulary.token_id("0")
    train_zero_frac = float((src_train == zero_id).mean())
    probe_src, _ = codec.encode_batch((np.eye(N, dtype=np.int64) * 125))
    probe_zero_frac = float((probe_src == zero_id).mean())

    return {
        "training_distribution": "a ~ Uniform(Z_q^n), every coordinate independent",
        "probe_distribution": "a = K * e_i: exactly one non-zero coordinate",
        "mean_zero_coordinates_training": float(training_zeros.mean()),
        "zero_coordinates_in_probe": probe_zeros,
        "max_zero_coordinates_seen_in_200k_training_rows": int(training_zeros.max()),
        "probability_a_training_row_looks_like_a_probe": p_at_least,
        "one_in_how_many": f"1 in {1/p_at_least:.3e}",
        "token_level_zero_digit_fraction_training": train_zero_frac,
        "token_level_zero_digit_fraction_probe": probe_zero_frac,
        "l1_norm_training_mean": float(np.abs(training).sum(axis=1).mean()),
        "l1_norm_probe": 125,
        "original_trains_on_probe_like_inputs": False,
        "original_expectation": (
            "gen_expr only ever produces rows of the (nega)circulant matrix built "
            "from a uniform a. No probe-like input appears in training. The original "
            "therefore REQUIRES the transformer to extrapolate to a region of input "
            "space it has never seen, with a ~51M-parameter model."
        ),
    }


def causes(shift: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Assess each candidate explanation for the phase-10 recovery failure."""
    return [
        {
            "cause": "1. Incomplete reproduction of original SALSA",
            "evidence_for": [
                "Matrix construction differs: the original negates the upper "
                "triangle (negacyclic); our trained runs used plain circulant.",
                "Sample reuse: the original reuses each instance ~10x by default; "
                "our runs used none.",
                "Verification (paper 4.4) is implemented in neither.",
            ],
            "evidence_against": [
                "Integer encoding is token-for-token identical across bases 2/7/81.",
                "Input and output sequence bodies are identical; only marker names "
                "differ.",
                "Coefficient range, secret distribution and training objective all "
                "match.",
                "Probe geometry is identical (K*I); ours is strictly cleaner.",
                "None of these differences bears on whether a probe elicits a "
                "coordinate-dependent answer.",
            ],
            "confidence": "Low",
            "verdict": "Real differences exist, but none of them explains a model "
                       "that returns one constant value to all twelve probes.",
        },
        {
            "cause": "2. Distribution shift between training rows and probes",
            "evidence_for": [
                f"A probe has {shift['zero_coordinates_in_probe']} zero coordinates; "
                f"training rows average "
                f"{shift['mean_zero_coordinates_training']:.3f} and never exceeded "
                f"{shift['max_zero_coordinates_seen_in_200k_training_rows']} in "
                "200,000 rows.",
                f"A uniform row looks like a probe with probability "
                f"{shift['probability_a_training_row_looks_like_a_probe']:.3e} "
                f"({shift['one_in_how_many']}).",
                f"At token level {shift['token_level_zero_digit_fraction_probe']:.3f} "
                "of probe digit tokens are '0' against "
                f"{shift['token_level_zero_digit_fraction_training']:.3f} in training.",
                "MEASURED phase-10 behaviour: on real rows the model gave 4 distinct "
                "readable answers; on probes it gave ONE constant value (K=63) or "
                "unreadable output (11/12 at K=94, 12/12 at K=125).",
                "The model reproduced its recorded validation metrics exactly, so the "
                "collapse is specific to the probe inputs, not to loading.",
            ],
            "evidence_against": [
                "The original faces the identical shift by construction and still "
                "reports recovery, so shift alone is not sufficient to prevent it.",
            ],
            "confidence": "High",
            "verdict": "Directly measured. This is the proximate mechanism of the "
                       "observed failure; it does not by itself establish that "
                       "recovery is unreachable.",
        },
        {
            "cause": "3. Reduced model capacity",
            "evidence_for": [
                "4,131,200 parameters versus the paper's ~51M: 12.3x smaller.",
                "The paper (5.4, 8) identifies model size, and encoder width in "
                "particular, as the key factor for scaling.",
                "Extrapolating far outside the training distribution is exactly the "
                "kind of behaviour capacity buys.",
            ],
            "evidence_against": [
                "Capacity was never varied in Salsa 2.0, so there is no measurement.",
                "The 4.13M model demonstrably learned the n=12 task in-distribution "
                "(acc_tau 0.347 against a 0.193 chance level).",
            ],
            "confidence": "Medium",
            "verdict": "Plausible and consistent with the paper, but untested here.",
        },
        {
            "cause": "4. Insufficient training",
            "evidence_for": [
                "100,032 samples against the paper's 2^21.9 = 3.9M for n=30.",
                "Final validation loss 1.7297 sits only 0.135 nats below the "
                "marginals-only baseline of 1.8650 and far above the 0.839 floor set "
                "by the LWE error, so the model is early on its learning curve.",
                "The paper's Figure 3 shows recovery occurring shortly AFTER loss "
                "starts falling sharply; our loss had not entered that regime.",
                "With reuse disabled, our gradient budget per distinct sample is ~10x "
                "smaller than the original's default.",
            ],
            "evidence_against": [
                "The paper states recovery succeeds 'long before the transformer has "
                "been trained to high accuracy'.",
            ],
            "confidence": "High",
            "verdict": "Strongly supported by the loss position relative to both "
                       "baselines, and by the sample budget.",
        },
        {
            "cause": "5. Representation differences",
            "evidence_for": [
                "Vocabulary 85 vs 86; different boundary-token convention.",
            ],
            "evidence_against": [
                "Encoding verified token-for-token identical across three bases.",
                "Input and output bodies identical; lengths identical.",
                "Probe encodings round-trip exactly (asserted by test).",
                "Ours reduces K mod q, which is the mathematically correct choice.",
            ],
            "confidence": "Low",
            "verdict": "Effectively excluded by direct token-level comparison.",
        },
        {
            "cause": "6. Algebraic structure differences",
            "evidence_for": [
                "Our trained runs used plain circulant; the original uses negacyclic.",
                "The learned function therefore differs.",
            ],
            "evidence_against": [
                "The probe relation b = K*s_i mod q holds for ANY row, whatever "
                "matrix built it, so the attack is well posed either way.",
                "Both structures present the same per-row task family and difficulty.",
                "The n=12 model did learn its own structure successfully.",
            ],
            "confidence": "Low",
            "verdict": "A genuine fidelity gap, but not a mechanism for probe "
                       "collapse.",
        },
        {
            "cause": "7. Combination",
            "evidence_for": [
                "Causes 2 and 4 are jointly sufficient and both are directly "
                "evidenced: the model is early in training AND the probe lies far "
                "outside its input distribution.",
                "Cause 3 plausibly governs how much training would be needed before "
                "extrapolation appears.",
            ],
            "evidence_against": [
                "No single experiment yet separates 3 from 4.",
            ],
            "confidence": "High",
            "verdict": "BEST SUPPORTED: insufficient training (4) leaves a model with "
                       "no extrapolation to a probe distribution it has never seen "
                       "(2); capacity (3) is a plausible but unmeasured modifier. "
                       "Causes 1, 5 and 6 are real but do not explain the collapse.",
        },
    ]


def main() -> int:
    """Produce the audit artefacts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = run_checks()
    shift = distribution_shift()
    cause_rows = causes(shift)

    tally: Dict[str, int] = {}
    for row in rows:
        tally[row["class"]] = tally.get(row["class"], 0) + 1

    payload = {
        "status": "READ-ONLY FIDELITY AUDIT - nothing trained, no implementation "
                  "file modified",
        "provenance": {
            "unmodified_meta_original": ["src/envs/encoders.py",
                                         "src/envs/generators.py",
                                         "src/envs/__init__.py"],
            "modified_in_the_supplied_checkout": ["src/envs/lattice.py",
                                                  "src/evaluator.py",
                                                  "src/trainer.py", "src/optim.py",
                                                  "src/model/transformer.py",
                                                  "src/model/__init__.py", "train.py"],
            "critical_note": (
                "src/model/transformer.py carries 'Modern SALSA' markers and "
                "DISABLES the Universal Transformer looping and the copy gate "
                "('obsolete in Modern SALSA'). Those are the two features the paper "
                "credits with a 14x sample-efficiency gain (Table 3). The supplied "
                "model reference is therefore NOT the paper's architecture, and "
                "Salsa 2.0 -- which keeps sharing and gating -- is architecturally "
                "closer to the paper than the supplied fork is."
            ),
        },
        "class_legend": CLASSES,
        "class_tally": tally,
        "stages": rows,
        "training_distribution_vs_probes": shift,
        "direct_recovery_pipelines": PIPELINES,
        "cause_analysis": cause_rows,
    }
    (OUTPUT_DIR / "fidelity_audit.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    fields = ["stage", "original_salsa", "salsa_2_0", "identical", "class",
              "class_name", "difference", "consequence", "evidence"]
    with (OUTPUT_DIR / "fidelity_audit.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    write_markdown(payload)
    print(f"wrote {OUTPUT_DIR / 'fidelity_audit.json'}")
    print(f"wrote {OUTPUT_DIR / 'fidelity_audit.csv'}")
    print(f"wrote {OUTPUT_DIR / 'fidelity_audit.md'}")
    print()
    print("class tally:", {CLASSES[k]: v for k, v in sorted(tally.items())})
    for row in rows:
        if row["class"] in ("D", "E"):
            print(f"  [{row['class']}] {row['stage']}")
    return 0


def write_markdown(payload: Dict[str, Any]) -> None:
    """Render the human-readable audit."""
    lines = [
        "# Original SALSA fidelity audit",
        "",
        "**Read-only.** Nothing was trained; no implementation file was modified.",
        "Every comparison that could be executed was executed against the reference",
        "source rather than argued from memory.",
        "",
        "## Provenance of the reference",
        "",
        "| file | status |",
        "|---|---|",
    ]
    for path in payload["provenance"]["unmodified_meta_original"]:
        lines.append(f"| `{path}` | unmodified Meta release |")
    for path in payload["provenance"]["modified_in_the_supplied_checkout"]:
        lines.append(f"| `{path}` | modified in the supplied checkout |")
    lines += ["", f"> {payload['provenance']['critical_note']}", "",
              "## Stage-by-stage comparison", "",
              "| Stage | Original SALSA | Salsa 2.0 | Identical? | Class | Difference | Consequence |",
              "|---|---|---|:---:|:---:|---|---|"]
    for row in payload["stages"]:
        lines.append(
            f"| {row['stage']} | {row['original_salsa']} | {row['salsa_2_0']} | "
            f"{'yes' if row['identical'] else 'NO'} | **{row['class']}** | "
            f"{row['difference']} | {row['consequence']} |"
        )
    lines += ["", "Class legend: "
              + "; ".join(f"**{k}** {v}" for k, v in payload["class_legend"].items()),
              "", "Tally: "
              + ", ".join(f"{payload['class_legend'][k]} x{v}"
                          for k, v in sorted(payload["class_tally"].items())),
              "", "## Training distribution versus recovery probes", ""]
    shift = payload["training_distribution_vs_probes"]
    for key, value in shift.items():
        lines.append(f"- **{key}**: {value}")
    lines += ["", "## Direct-recovery pipelines, side by side", "",
              "### Original SALSA (reconstructed from evaluator.eval_secret)", "",
              "```", payload["direct_recovery_pipelines"]["original"], "```", "",
              "### Salsa 2.0", "",
              "```", payload["direct_recovery_pipelines"]["salsa_2_0"], "```", "",
              "### Line-by-line", "",
              "| step | original | Salsa 2.0 | class |", "|---|---|---|:---:|"]
    for step in payload["direct_recovery_pipelines"]["comparison"]:
        lines.append(f"| {step['step']} | {step['original']} | {step['salsa_2_0']} | "
                     f"**{step['class']}** |")
    lines += ["", "## Cause analysis for the phase-10 recovery failure", ""]
    for entry in payload["cause_analysis"]:
        lines += [f"### {entry['cause']}", "",
                  f"**Confidence: {entry['confidence']}**", "", "*Evidence for:*"]
        lines += [f"- {e}" for e in entry["evidence_for"]]
        lines += ["", "*Evidence against:*"]
        lines += [f"- {e}" for e in entry["evidence_against"]]
        lines += ["", f"> {entry['verdict']}", ""]
    (OUTPUT_DIR / "fidelity_audit.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
