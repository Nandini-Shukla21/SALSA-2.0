"""Tests for the integer <-> token encoding layer.

The central property is exact round-tripping: for every value in ``Z_q``, in
every supported base, ``decode(encode(x)) == x``.  A representation that loses
information would corrupt every downstream experiment while still "training"
perfectly happily, so these tests are exhaustive over ``Z_q`` rather than
sampled.
"""

from pathlib import Path

import numpy as np
import pytest

from salsa.data.encoding import (
    BOS,
    EOS,
    PAD,
    SEP,
    EncodingError,
    IntegerEncoder,
    LatticeCodec,
    Vocabulary,
    benchmark_codec,
    decode_integer,
    decode_lattice_input,
    decode_lattice_output,
    decode_vector,
    digit_width,
    encode_integer,
    encode_lattice_input,
    encode_lattice_output,
    encode_vector,
)
from salsa.data.lwe import LWEParams, generate_lwe_sample
from salsa.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

# Official experiment configuration.
Q = 251
BASES = [2, 7, 81]
DIMENSIONS = [4, 12, 30, 50, 128]


# --------------------------------------------------------------------------- #
# Digit width
# --------------------------------------------------------------------------- #
def test_digit_width_matches_the_modulus() -> None:
    """width = smallest w with base**w > max_value."""
    assert digit_width(81, Q - 1) == 2
    assert digit_width(7, Q - 1) == 3
    assert digit_width(2, Q - 1) == 8
    assert digit_width(10, 0) == 1
    assert digit_width(10, 9) == 1
    assert digit_width(10, 10) == 2


def test_digit_width_rejects_bad_arguments() -> None:
    """A base below 2 or a negative maximum is an error."""
    with pytest.raises(EncodingError):
        digit_width(1, 10)
    with pytest.raises(EncodingError):
        digit_width(10, -1)


@pytest.mark.parametrize("base", BASES)
def test_encoder_width_covers_the_whole_field(base: int) -> None:
    """The chosen width can express every element of Z_q."""
    encoder = IntegerEncoder(base=base, modulus=Q)
    assert encoder.capacity >= Q
    assert base ** (encoder.width - 1) < Q


# --------------------------------------------------------------------------- #
# Round-trip over the whole field
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("base", BASES)
def test_round_trip_over_every_value_in_zq(base: int) -> None:
    """decode(encode(x)) == x for all x in [0, q), exhaustively."""
    encoder = IntegerEncoder(base=base, modulus=Q)
    for value in range(Q):
        tokens = encoder.encode(value)
        assert len(tokens) == encoder.width
        assert encoder.decode(tokens) == value


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("value", [0, 1, Q - 1])
def test_boundary_values(base: int, value: int) -> None:
    """The values the requirements call out explicitly: 0, 1 and q-1."""
    encoder = IntegerEncoder(base=base, modulus=Q)
    assert encoder.decode(encoder.encode(value)) == value


@pytest.mark.parametrize("base", BASES)
def test_value_q_wraps_to_zero(base: int) -> None:
    """q itself is not an element of Z_q: it is reduced to 0."""
    encoder = IntegerEncoder(base=base, modulus=Q)
    assert encoder.prepare(Q) == 0
    assert encoder.decode(encoder.encode(Q)) == 0
    assert encoder.encode(Q) == encoder.encode(0)
    assert encoder.decode(encoder.encode(Q + 7)) == 7


@pytest.mark.parametrize("base", BASES)
def test_large_values_from_matrix_operations_round_trip(base: int) -> None:
    """Raw dot products (far above q) reduce and round-trip correctly."""
    rng = np.random.default_rng(0)
    encoder = IntegerEncoder(base=base, modulus=Q)
    a = rng.integers(0, Q, size=(64, 30), dtype=np.int64)
    s = np.zeros(30, dtype=np.int64)
    s[[3, 11, 27]] = 1
    raw = a @ s  # up to 3 * 250 = 750, well above q
    assert int(raw.max()) > Q
    for value in raw.tolist():
        assert encoder.decode(encoder.encode(value)) == value % Q


def test_known_encodings_are_what_we_expect() -> None:
    """Spot-check the digit layout, most significant first."""
    base81 = IntegerEncoder(base=81, modulus=Q)
    assert base81.encode(0) == ["0", "0"]
    assert base81.encode(1) == ["0", "1"]
    assert base81.encode(80) == ["0", "80"]
    assert base81.encode(81) == ["1", "0"]
    assert base81.encode(250) == ["3", "7"]  # 3*81 + 7 = 250

    base2 = IntegerEncoder(base=2, modulus=Q)
    assert base2.encode(0) == ["0"] * 8
    assert base2.encode(1) == ["0"] * 7 + ["1"]
    assert base2.encode(250) == list("11111010")

    base7 = IntegerEncoder(base=7, modulus=Q)
    assert base7.encode(250) == ["5", "0", "5"]  # 5*49 + 0*7 + 5 = 250


# --------------------------------------------------------------------------- #
# Representation variants
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("base", [7, 81])
def test_lsb_first_order_round_trips_and_differs(base: int) -> None:
    """Digit order is a representation choice, and an invertible one."""
    msb = IntegerEncoder(base=base, modulus=Q)
    lsb = IntegerEncoder(base=base, modulus=Q, digit_order="lsb_first")
    for value in range(Q):
        assert lsb.decode(lsb.encode(value)) == value
    assert lsb.encode(1) == list(reversed(msb.encode(1)))


@pytest.mark.parametrize("base", [7, 81])
def test_balanced_representation_round_trips(base: int) -> None:
    """Balanced digits carry the same information with a different alphabet."""
    encoder = IntegerEncoder(base=base, modulus=Q, balanced=True)
    half = (base - 1) // 2
    for value in range(Q):
        tokens = encoder.encode(value)
        assert len(tokens) == encoder.width
        assert all(-half <= int(t) <= half for t in tokens)
        assert encoder.decode(tokens) == value


def test_balanced_uses_negative_digits() -> None:
    """A value just below q is written with negative digits, not large ones."""
    encoder = IntegerEncoder(base=81, modulus=Q, balanced=True)
    assert any(int(t) < 0 for t in encoder.encode(Q - 1))
    assert encoder.decode(encoder.encode(Q - 1)) == Q - 1


def test_balanced_requires_an_odd_base() -> None:
    """Base 2 has no symmetric digit set."""
    with pytest.raises(EncodingError, match="odd base"):
        IntegerEncoder(base=2, modulus=Q, balanced=True)


def test_signed_encoder_round_trips_negative_values() -> None:
    """Signed mode handles genuinely signed data, e.g. an error vector."""
    encoder = IntegerEncoder(base=81, max_value=100, signed=True)
    for value in range(-100, 101):
        tokens = encoder.encode(value)
        assert tokens[0] in ("<+>", "<->")
        assert encoder.decode(tokens) == value


def test_unsigned_encoder_rejects_negative_values() -> None:
    """Without a modulus or a sign token, a negative value is an error."""
    encoder = IntegerEncoder(base=81, max_value=100)
    with pytest.raises(EncodingError, match="signed"):
        encoder.encode(-1)


def test_variable_width_round_trips() -> None:
    """Variable-width encoding is available but not the default."""
    encoder = IntegerEncoder(base=81, modulus=Q, fixed_width=False)
    assert encoder.encode(5) == ["5"]
    assert encoder.decode(["5"]) == 5
    for value in range(Q):
        assert encoder.decode(encoder.encode(value)) == value


def test_encoder_construction_is_validated() -> None:
    """Bad encoder configurations are rejected at construction."""
    with pytest.raises(EncodingError):
        IntegerEncoder(base=1, modulus=Q)
    with pytest.raises(EncodingError):
        IntegerEncoder(base=81, modulus=Q, digit_order="middle_out")
    with pytest.raises(EncodingError):
        IntegerEncoder(base=81)  # no modulus, max_value or width
    with pytest.raises(EncodingError):
        IntegerEncoder(base=81, modulus=Q, balanced=True, signed=True)
    with pytest.raises(EncodingError):
        IntegerEncoder(base=81, modulus=Q, width=0)
    with pytest.raises(EncodingError, match="cannot represent"):
        IntegerEncoder(base=2, modulus=Q, width=4)


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
def test_special_tokens_have_fixed_ids() -> None:
    """A trained checkpoint's embedding rows must keep their meaning."""
    vocab = Vocabulary([str(i) for i in range(81)])
    assert (vocab.pad_id, vocab.bos_id, vocab.eos_id, vocab.sep_id) == (0, 1, 2, 3)
    assert vocab.tokens[:4] == [PAD, BOS, EOS, SEP]
    assert vocab.size == 4 + 81


def test_vocabulary_round_trips_tokens_and_ids() -> None:
    """encode/decode are inverse on the id level."""
    vocab = Vocabulary([str(i) for i in range(7)])
    tokens = [BOS, "3", "0", SEP, "6", EOS]
    assert vocab.decode(vocab.encode(tokens)) == tokens


def test_vocabulary_rejects_unknown_tokens_and_ids() -> None:
    """Unknown tokens are an error, never mapped to a fallback id."""
    vocab = Vocabulary([str(i) for i in range(7)])
    with pytest.raises(EncodingError, match="not in the vocabulary"):
        vocab.token_id("9")
    with pytest.raises(EncodingError, match="outside the vocabulary"):
        vocab.token_at(999)


def test_vocabulary_rejects_bad_digit_sets() -> None:
    """Empty, duplicated or colliding digit alphabets are rejected."""
    with pytest.raises(EncodingError):
        Vocabulary([])
    with pytest.raises(EncodingError, match="duplicates"):
        Vocabulary(["0", "1", "1"])
    with pytest.raises(EncodingError, match="collide"):
        Vocabulary(["0", SEP])


# --------------------------------------------------------------------------- #
# Vectors
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("separator", [True, False])
def test_vector_round_trip(base: int, separator: bool) -> None:
    """Vectors round-trip with and without separator tokens."""
    encoder = IntegerEncoder(base=base, modulus=Q)
    values = np.array([0, 1, 80, 250, 123])
    tokens = encode_vector(values, encoder, separator=separator)
    decoded = decode_vector(tokens, encoder, separator=separator, length=len(values))
    assert np.array_equal(decoded, values)


def test_vector_separators_sit_between_coordinates() -> None:
    """n coordinates carry n-1 separators, never a leading or trailing one."""
    encoder = IntegerEncoder(base=81, modulus=Q)
    tokens = encode_vector([1, 2, 3], encoder, separator=True)
    assert tokens.count(SEP) == 2
    assert tokens[0] != SEP and tokens[-1] != SEP
    assert tokens == ["0", "1", SEP, "0", "2", SEP, "0", "3"]


def test_empty_vector_is_rejected() -> None:
    """An empty vector has no valid representation."""
    encoder = IntegerEncoder(base=81, modulus=Q)
    with pytest.raises(EncodingError, match="empty"):
        encode_vector([], encoder)


def test_vector_length_mismatch_is_detected() -> None:
    """Decoding checks the coordinate count when one is expected."""
    encoder = IntegerEncoder(base=81, modulus=Q)
    tokens = encode_vector([1, 2, 3], encoder)
    with pytest.raises(EncodingError, match="Expected 4 coordinates"):
        decode_vector(tokens, encoder, length=4)


# --------------------------------------------------------------------------- #
# Malformed sequences
# --------------------------------------------------------------------------- #
def test_invalid_digit_is_rejected() -> None:
    """A digit outside the base is an error naming its position."""
    encoder = IntegerEncoder(base=7, modulus=Q)
    with pytest.raises(EncodingError, match="Invalid digit '9' at position 1"):
        encoder.decode(["1", "9", "2"])
    with pytest.raises(EncodingError, match="Invalid digit"):
        encoder.decode(["1", "x", "2"])


def test_invalid_separator_placement_is_rejected() -> None:
    """Leading, trailing and doubled separators are all errors."""
    encoder = IntegerEncoder(base=81, modulus=Q)
    good = encode_vector([1, 2], encoder)
    with pytest.raises(EncodingError, match="no preceding value"):
        decode_vector([SEP] + good, encoder)
    with pytest.raises(EncodingError, match="ends with a separator"):
        decode_vector(good + [SEP], encoder)
    with pytest.raises(EncodingError, match="no preceding value"):
        decode_vector(["0", "1", SEP, SEP, "0", "2"], encoder)


def test_missing_separator_is_detected() -> None:
    """Two values run together are caught by the width check."""
    encoder = IntegerEncoder(base=81, modulus=Q)
    with pytest.raises(EncodingError, match="Expected exactly 2 digit tokens"):
        decode_vector(["0", "1", "0", "2"], encoder, separator=True)


def test_empty_sequence_is_rejected() -> None:
    """Empty input is never a valid encoding of anything."""
    encoder = IntegerEncoder(base=81, modulus=Q)
    with pytest.raises(EncodingError, match="empty"):
        encoder.decode([])
    with pytest.raises(EncodingError, match="empty"):
        decode_vector([], encoder)


def test_truncated_representation_is_rejected() -> None:
    """A value missing digits must not decode to a plausible wrong number."""
    encoder = IntegerEncoder(base=2, modulus=Q)
    full = encoder.encode(250)
    with pytest.raises(EncodingError, match="truncated"):
        encoder.decode(full[:-1])
    with pytest.raises(EncodingError, match="too many"):
        encoder.decode(full + ["1"])


def test_special_token_where_a_digit_belongs_is_rejected() -> None:
    """Special tokens never silently act as digits."""
    encoder = IntegerEncoder(base=81, modulus=Q)
    with pytest.raises(EncodingError, match="Special token"):
        encoder.decode([BOS, "1"])
    with pytest.raises(EncodingError, match="Special token"):
        encoder.decode(["1", PAD])


def test_out_of_range_value_is_rejected_unless_lenient() -> None:
    """81*81 > 251, so the digits can express values Z_q cannot hold."""
    encoder = IntegerEncoder(base=81, modulus=Q)
    too_big = ["80", "80"]  # 6560
    with pytest.raises(EncodingError, match="outside"):
        encoder.decode(too_big)
    assert encoder.decode(too_big, strict=False) == 6560 % Q


def test_sign_token_on_an_unsigned_encoder_is_rejected() -> None:
    """A stray sign token is an error, not silently ignored."""
    encoder = IntegerEncoder(base=81, modulus=Q)
    with pytest.raises(EncodingError, match="unsigned"):
        encoder.decode(["<+>", "0", "1"])


# --------------------------------------------------------------------------- #
# LatticeCodec: the SALSA a -> b representation
# --------------------------------------------------------------------------- #
def test_codec_input_layout_is_exactly_as_documented() -> None:
    """<bos> a0.. <sep> a1.. <sep> ... <eos>"""
    codec = LatticeCodec(n=3, q=Q, base=81)
    tokens = codec.encode_input([1, 80, 250])
    assert tokens == [BOS, "0", "1", SEP, "0", "80", SEP, "3", "7", EOS]
    assert codec.encode_output(250) == [BOS, "3", "7", EOS]


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("n", DIMENSIONS)
def test_codec_round_trips_lattice_instances(base: int, n: int) -> None:
    """A full (a, b) pair survives encoding and decoding unchanged."""
    rng = np.random.default_rng(0)
    codec = LatticeCodec(n=n, q=Q, base=base)
    a = rng.integers(0, Q, size=n, dtype=np.int64)
    b = int(rng.integers(0, Q))

    input_tokens, output_tokens = codec.encode_pair(a, b)
    assert len(input_tokens) == codec.input_length
    assert len(output_tokens) == codec.output_length
    assert np.array_equal(codec.decode_input(input_tokens), a)
    assert codec.decode_output(output_tokens) == b


@pytest.mark.parametrize("base", BASES)
def test_codec_round_trips_real_lwe_samples(base: int) -> None:
    """The representation is exercised on genuine phase-2 LWE data."""
    params = LWEParams(n=30, q=Q, sigma=3.0, hamming_weight=3, structure="rlwe")
    sample = generate_lwe_sample(params, num_instances=16, seed=0)
    codec = LatticeCodec(n=30, q=Q, base=base)
    for row, target in zip(sample.A, sample.b):
        tokens_in, tokens_out = codec.encode_pair(row, int(target))
        assert np.array_equal(codec.decode_input(tokens_in), row)
        assert codec.decode_output(tokens_out) == int(target)


def test_codec_id_level_round_trip() -> None:
    """Token ids round-trip as well as token strings."""
    codec = LatticeCodec(n=8, q=Q, base=81)
    a = np.arange(8) * 31 % Q
    ids = codec.encode_input_ids(a)
    assert all(isinstance(i, int) for i in ids)
    assert max(ids) < codec.vocabulary.size
    assert np.array_equal(codec.decode_input_ids(ids), a)
    assert codec.decode_output_ids(codec.encode_output_ids(77)) == 77


def test_codec_without_separator_or_markers() -> None:
    """Both structural options are honoured and still round-trip."""
    codec = LatticeCodec(n=4, q=Q, base=81, separator=False, include_bos_eos=False)
    a = np.array([1, 2, 3, 250])
    tokens = codec.encode_input(a)
    assert SEP not in tokens and BOS not in tokens and EOS not in tokens
    assert len(tokens) == 4 * 2 == codec.input_length
    assert np.array_equal(codec.decode_input(tokens), a)


def test_codec_supports_different_input_and_output_bases() -> None:
    """Input and output representations can be varied independently."""
    codec = LatticeCodec(n=4, q=Q, base=81, output_base=2)
    assert codec.input_encoder.width == 2
    assert codec.output_encoder.width == 8
    assert codec.output_length == 8 + 2
    assert codec.decode_output(codec.encode_output(250)) == 250
    assert np.array_equal(codec.decode_input(codec.encode_input([1, 2, 3, 4])), [1, 2, 3, 4])


def test_codec_rejects_wrong_length_input() -> None:
    """A vector of the wrong dimension cannot be encoded."""
    codec = LatticeCodec(n=4, q=Q, base=81)
    with pytest.raises(EncodingError, match="length 4"):
        codec.encode_input([1, 2, 3])


def test_codec_rejects_malformed_frames() -> None:
    """Misplaced <bos>/<eos> inside the body is an error."""
    codec = LatticeCodec(n=2, q=Q, base=81)
    tokens = codec.encode_input([1, 2])
    broken = tokens[:3] + [EOS] + tokens[4:]
    with pytest.raises(EncodingError, match="Misplaced special token"):
        codec.decode_input(broken)


def test_codec_decodes_model_output_without_markers() -> None:
    """A model may emit bare digits; decoding must still work."""
    codec = LatticeCodec(n=4, q=Q, base=81)
    assert codec.decode_output(["3", "7"]) == 250
    assert codec.decode_output([BOS, "3", "7"]) == 250


def test_codec_construction_is_validated() -> None:
    """Bad codec dimensions are rejected."""
    with pytest.raises(EncodingError):
        LatticeCodec(n=0, q=Q)
    with pytest.raises(EncodingError):
        LatticeCodec(n=4, q=1)


# --------------------------------------------------------------------------- #
# Batches
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("n", [4, 30, 128])
def test_batch_encoding_matches_the_scalar_path(base: int, n: int) -> None:
    """The fast vectorised path and the readable path must never diverge."""
    rng = np.random.default_rng(1)
    codec = LatticeCodec(n=n, q=Q, base=base)
    A = rng.integers(0, Q, size=(32, n), dtype=np.int64)
    b = rng.integers(0, Q, size=32, dtype=np.int64)

    input_ids, output_ids = codec.encode_batch(A, b)
    assert input_ids.shape == (32, codec.input_length)
    assert output_ids.shape == (32, codec.output_length)
    assert input_ids.dtype == np.int64

    for row_index in range(32):
        assert input_ids[row_index].tolist() == codec.encode_input_ids(A[row_index])
        assert output_ids[row_index].tolist() == codec.encode_output_ids(int(b[row_index]))


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("balanced", [False, True])
def test_batch_round_trip(base: int, balanced: bool) -> None:
    """A whole batch decodes back to the original integers."""
    if balanced and base % 2 == 0:
        pytest.skip("balanced digits need an odd base")
    rng = np.random.default_rng(2)
    codec = LatticeCodec(n=16, q=Q, base=base, balanced=balanced)
    A = rng.integers(0, Q, size=(64, 16), dtype=np.int64)
    b = rng.integers(0, Q, size=64, dtype=np.int64)

    input_ids, output_ids = codec.encode_batch(A, b)
    assert np.array_equal(codec.decode_batch_inputs(input_ids), A)
    assert np.array_equal(codec.decode_batch_outputs(output_ids), b)


def test_batch_encoding_without_targets() -> None:
    """Encoding inputs alone is supported (inference time)."""
    codec = LatticeCodec(n=8, q=Q, base=81)
    input_ids, output_ids = codec.encode_batch(np.zeros((4, 8), dtype=np.int64))
    assert output_ids is None
    assert input_ids.shape == (4, codec.input_length)


def test_batch_shape_mismatches_are_rejected() -> None:
    """Wrong shapes raise instead of broadcasting into nonsense."""
    codec = LatticeCodec(n=8, q=Q, base=81)
    with pytest.raises(EncodingError, match=r"shape \(m, 8\)"):
        codec.encode_batch(np.zeros((4, 7), dtype=np.int64))
    with pytest.raises(EncodingError, match="to match A"):
        codec.encode_batch(np.zeros((4, 8), dtype=np.int64), np.zeros(3, dtype=np.int64))
    with pytest.raises(EncodingError):
        codec.decode_batch_inputs(np.zeros((4, 3), dtype=np.int64))


def test_batch_decoding_validates_the_frame() -> None:
    """A batch whose sequences lack <bos>/<eos> is rejected."""
    codec = LatticeCodec(n=4, q=Q, base=81)
    input_ids, _ = codec.encode_batch(np.zeros((2, 4), dtype=np.int64))
    broken = input_ids.copy()
    broken[0, 0] = codec.vocabulary.sep_id
    with pytest.raises(EncodingError, match="<bos>"):
        codec.decode_batch_inputs(broken)


def test_batch_decoding_rejects_out_of_range_sequences() -> None:
    """Strict batch decoding catches values Z_q cannot hold."""
    codec = LatticeCodec(n=2, q=Q, base=81)
    ids, _ = codec.encode_batch(np.array([[1, 2]], dtype=np.int64))
    ids[0, 1] = codec.vocabulary.token_id("80")
    ids[0, 2] = codec.vocabulary.token_id("80")
    with pytest.raises(EncodingError, match="outside"):
        codec.decode_batch_inputs(ids)
    assert codec.decode_batch_inputs(ids, strict=False)[0, 0] == 6560 % Q


# --------------------------------------------------------------------------- #
# Sequence lengths (an efficiency metric)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "n,base,expected_input",
    [
        (30, 81, 30 * 2 + 29 + 2),
        (30, 7, 30 * 3 + 29 + 2),
        (30, 2, 30 * 8 + 29 + 2),
        (128, 81, 128 * 2 + 127 + 2),
        (128, 7, 128 * 3 + 127 + 2),
        (128, 2, 128 * 8 + 127 + 2),
    ],
)
def test_sequence_length_accounting(n: int, base: int, expected_input: int) -> None:
    """Lengths are computed, reported, and match the actual token count."""
    codec = LatticeCodec(n=n, q=Q, base=base)
    assert codec.input_length == expected_input
    assert len(codec.encode_input(np.zeros(n, dtype=np.int64))) == expected_input

    lengths = codec.lengths()
    assert lengths.input_tokens == expected_input
    assert lengths.output_tokens == codec.output_length
    assert lengths.total_tokens == expected_input + codec.output_length
    assert lengths.separators == n - 1
    assert lengths.specials == 4
    assert set(lengths.to_dict()) >= {"input_tokens", "output_tokens", "total_tokens"}


def test_describe_is_serialisable_for_run_metadata() -> None:
    """The codec description can be written into metadata.json."""
    import json

    codec = LatticeCodec(n=30, q=Q, base=81)
    description = codec.describe()
    json.dumps(description)
    assert description["vocab_size"] == codec.vocabulary.size
    assert description["input_tokens"] == codec.input_length
    assert description["input_base"] == 81


# --------------------------------------------------------------------------- #
# Config integration
# --------------------------------------------------------------------------- #
def test_codec_from_config_matches_the_config_file() -> None:
    """The codec reads its representation from the experiment config."""
    cfg = load_config(CONFIG_DIR / "base.yaml")
    codec = LatticeCodec.from_config(cfg)
    assert codec.n == cfg.lwe.n
    assert codec.q == cfg.lwe.q
    assert codec.input_encoder.base == cfg.encoding.resolved_input_base
    assert codec.output_encoder.base == cfg.encoding.resolved_output_base
    assert codec.separator == cfg.encoding.separator
    assert codec.include_bos_eos == cfg.encoding.include_bos_eos


def test_all_shipped_configs_produce_a_working_codec() -> None:
    """Every shipped config yields a codec that round-trips."""
    rng = np.random.default_rng(0)
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        cfg = load_config(path)
        codec = LatticeCodec.from_config(cfg)
        a = rng.integers(0, cfg.lwe.q, size=cfg.lwe.n, dtype=np.int64)
        assert np.array_equal(codec.decode_input(codec.encode_input(a)), a), path.name


def test_config_base_change_changes_the_representation() -> None:
    """Base is an experimental variable, not a constant in the code."""
    cfg81 = load_config(CONFIG_DIR / "base.yaml")
    cfg7 = load_config(CONFIG_DIR / "base.yaml", overrides=["encoding.base=7"])
    assert LatticeCodec.from_config(cfg81).input_length == 91
    assert LatticeCodec.from_config(cfg7).input_length == 121


# --------------------------------------------------------------------------- #
# Free-function API
# --------------------------------------------------------------------------- #
def test_free_functions_match_the_methods() -> None:
    """The documented free functions are thin, faithful wrappers."""
    encoder = IntegerEncoder(base=81, modulus=Q)
    codec = LatticeCodec(n=4, q=Q, base=81)
    a = [1, 2, 3, 4]

    assert encode_integer(7, encoder) == encoder.encode(7)
    assert decode_integer(encoder.encode(7), encoder) == 7
    assert encode_lattice_input(a, codec) == codec.encode_input(a)
    assert np.array_equal(
        decode_lattice_input(encode_lattice_input(a, codec), codec), a
    )
    assert decode_lattice_output(encode_lattice_output(9, codec), codec) == 9


# --------------------------------------------------------------------------- #
# Performance and platform constraints
# --------------------------------------------------------------------------- #
def test_benchmark_reports_both_paths() -> None:
    """The benchmark helper returns usable throughput numbers."""
    codec = LatticeCodec(n=30, q=Q, base=81)
    result = benchmark_codec(codec, num_instances=200, repeats=1)
    for key in (
        "batch_us_per_instance",
        "batch_instances_per_second",
        "scalar_us_per_instance",
        "speedup_batch_over_scalar",
    ):
        assert key in result and result[key] > 0


def test_encoding_module_has_no_torch_or_cuda_dependency() -> None:
    """Requirement 12/13: the encoder is pure NumPy and CPU-only."""
    source = (REPO_ROOT / "salsa" / "data" / "encoding.py").read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "cuda" not in source.lower()


# --------------------------------------------------------------------------- #
# Representations R and P (phase 4: representation as a controlled variable)
# --------------------------------------------------------------------------- #
def test_representation_r_matches_the_audited_original_layout() -> None:
    """Representation R reproduces the released SALSA code token-for-token.

    The expected sequence is the output of the original
    ``src/envs/encoders.py`` recorded during the phase-3.5 audit (executed,
    not inferred).  Only the marker *names* differ: the original writes its
    ``<s>`` token at both ends, SALSA 2.0 writes ``<bos>`` / ``<eos>``.
    """
    cfg = load_config(CONFIG_DIR / "representation_r.yaml", overrides=["lwe.n=4"])
    cfg.validate()
    codec = LatticeCodec.from_config(cfg)

    tokens = codec.encode_input([67, 77, 10, 18])
    assert tokens == [BOS, "67", "0", "77", "0", "10", "0", "18", "0", EOS]
    assert len(tokens) == 10

    # Per-value digit strings, exactly as the original write_int produced them.
    expected = {0: ["0", "0"], 1: ["1", "0"], 31: ["31", "0"],
                67: ["67", "0"], 77: ["77", "0"], 250: ["7", "3"]}
    for value, digits in expected.items():
        assert codec.input_encoder.encode(value) == digits


def test_representation_p_adds_separators_and_nothing_else() -> None:
    """P differs from R in exactly one setting, so the comparison is controlled."""
    r = load_config(CONFIG_DIR / "representation_r.yaml")
    p = load_config(CONFIG_DIR / "representation_p.yaml")
    differences = {
        field: (getattr(r.encoding, field), getattr(p.encoding, field))
        for field in vars(r.encoding)
        if getattr(r.encoding, field) != getattr(p.encoding, field)
    }
    assert differences == {"separator": (False, True)}


@pytest.mark.parametrize("n,r_length,p_length", [(30, 62, 91), (128, 258, 385)])
def test_representation_sequence_lengths(n: int, r_length: int, p_length: int) -> None:
    """The two representations have the documented token lengths."""
    for name, expected in (("representation_r", r_length), ("representation_p", p_length)):
        cfg = load_config(CONFIG_DIR / f"{name}.yaml", overrides=[f"lwe.n={n}"])
        codec = LatticeCodec.from_config(cfg)
        assert codec.input_length == expected
        assert len(codec.encode_input(np.zeros(n, dtype=np.int64))) == expected


@pytest.mark.parametrize("name", ["representation_r", "representation_p"])
@pytest.mark.parametrize("n", [4, 30, 128])
def test_both_representations_round_trip_exactly(name: str, n: int) -> None:
    """SALSA 2.0 keeps exact round-trips the released decoder does not have."""
    rng = np.random.default_rng(0)
    cfg = load_config(CONFIG_DIR / f"{name}.yaml", overrides=[f"lwe.n={n}"])
    codec = LatticeCodec.from_config(cfg)
    A = rng.integers(0, cfg.lwe.q, size=(16, n), dtype=np.int64)
    b = rng.integers(0, cfg.lwe.q, size=16, dtype=np.int64)

    for row, target in zip(A, b):
        assert np.array_equal(codec.decode_input(codec.encode_input(row)), row)
        assert codec.decode_output(codec.encode_output(int(target))) == int(target)

    input_ids, output_ids = codec.encode_batch(A, b)
    assert np.array_equal(codec.decode_batch_inputs(input_ids), A)
    assert np.array_equal(codec.decode_batch_outputs(output_ids), b)


def test_base_config_is_representation_p() -> None:
    """The shared base config records P, so an unmodified run is P."""
    base = load_config(CONFIG_DIR / "base.yaml")
    p = load_config(CONFIG_DIR / "representation_p.yaml")
    assert base.encoding.digit_order == p.encoding.digit_order == "lsb_first"
    assert base.encoding.separator == p.encoding.separator is True
    assert base.encoding.fixed_width == p.encoding.fixed_width is True


def test_representations_compose_with_model_configs() -> None:
    """A model config can select a representation via multi-parent extends."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "composed.yaml"
        base = (CONFIG_DIR / "base.yaml").as_posix()
        repr_r = (CONFIG_DIR / "representation_r.yaml").as_posix()
        path.write_text(
            f"extends: ['{base}', '{repr_r}']\nexperiment:\n  name: composed\n",
            encoding="utf-8",
        )
        cfg = load_config(path)
        cfg.validate()
        assert cfg.lwe.q == 251 and cfg.lwe.n == 30  # crypto block preserved
        assert cfg.encoding.separator is False       # representation applied
        assert LatticeCodec.from_config(cfg).input_length == 62


def test_digit_order_and_fixed_width_are_validated() -> None:
    """The two new keys are checked, not silently accepted."""
    from salsa.utils.config import ConfigError

    bad_order = load_config(CONFIG_DIR / "base.yaml",
                            overrides=["encoding.digit_order=middle_out"])
    with pytest.raises(ConfigError, match="digit_order"):
        bad_order.validate()

    ambiguous = load_config(
        CONFIG_DIR / "base.yaml",
        overrides=["encoding.fixed_width=false", "encoding.separator=false"],
    )
    with pytest.raises(ConfigError, match="fixed_width"):
        ambiguous.validate()


def test_variable_width_codec_guards_the_fixed_stride_paths() -> None:
    """Variable width is usable with separators, but has no single length."""
    codec = LatticeCodec(n=4, q=Q, base=81, separator=True, fixed_width=False)
    a = np.array([1, 2, 3, 250])
    assert np.array_equal(codec.decode_input(codec.encode_input(a)), a)
    with pytest.raises(EncodingError, match="fixed_width=True"):
        _ = codec.input_length
    with pytest.raises(EncodingError, match="fixed_width=True"):
        codec.encode_batch(np.zeros((2, 4), dtype=np.int64))
    with pytest.raises(EncodingError, match="requires separator=True"):
        LatticeCodec(n=4, q=Q, base=81, separator=False, fixed_width=False)
