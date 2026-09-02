"""Tests for Salsa2-NACT V2.

Three things have to hold, and they are tested separately because they fail
separately:

1. **The budget is real.**  Actual, component-breakdown and analytical counts
   must all agree at 4,241,288, with nothing unclassified.
2. **The front end is lossless.**  NACT reads the codec's own token layout, so
   the integer it reconstructs must equal the integer the codec encoded -- for
   every dimension, and for both representations.
3. **V1 is untouched.**  The GatedUT count, spec and forward path must be exactly
   what they were before NACT existed.
"""

import numpy as np
import pytest
import torch

from salsa.data import LatticeCodec
from salsa.models import (
    NactSpec,
    SalsaNact,
    SalsaTransformer,
    ModelSpec,
    analytical_breakdown,
    build_model,
    count_trainable_parameters,
    parameter_breakdown,
    unclassified_parameters,
)
from salsa.utils import load_config

NACT_PARAMETERS = 4_241_288
V1_PARAMETERS = 4_131_200
DIMENSIONS = (12, 20, 30, 50, 70, 90, 128)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def codec_for(n: int, separator: bool) -> LatticeCodec:
    """Build a codec matching the phase-6 encoding at dimension ``n``."""
    return LatticeCodec(
        n=n, q=251, base=81, separator=separator,
        digit_order="lsb_first", fixed_width=True,
    )


def spec_for(codec: LatticeCodec) -> NactSpec:
    """Build a NACT spec consistent with a codec."""
    return NactSpec(
        vocab_size=codec.vocabulary.size,
        q=codec.q,
        base=codec.input_encoder.base,
        digit_width=codec.input_encoder.width,
        separator=codec.separator,
        max_coordinates=max(128, codec.n),
    )


# --------------------------------------------------------------------------- #
# 1. parameter budget
# --------------------------------------------------------------------------- #
def test_nact_parameter_count_is_exactly_the_approved_target():
    model = SalsaNact(NactSpec())
    assert count_trainable_parameters(model) == NACT_PARAMETERS


def test_nact_parameter_count_matches_the_required_formula():
    model = SalsaNact(NactSpec())
    assert (sum(p.numel() for p in model.parameters() if p.requires_grad)
            == NACT_PARAMETERS)


def test_nact_three_counting_methods_agree():
    spec = NactSpec()
    model = SalsaNact(spec)
    actual = count_trainable_parameters(model)
    measured = parameter_breakdown(model)
    analytical = analytical_breakdown(spec)
    assert actual == sum(measured.values()) == sum(analytical.values())
    for component in analytical:
        assert measured[component] == analytical[component], component


def test_nact_has_no_unclassified_parameters():
    assert unclassified_parameters(SalsaNact(NactSpec())) == []


def test_nact_is_inside_the_hard_budget():
    count = count_trainable_parameters(SalsaNact(NactSpec()))
    assert 4_000_000 <= count <= 5_000_000


def test_nact_front_end_components_are_exactly_as_designed():
    breakdown = parameter_breakdown(SalsaNact(NactSpec()))
    assert breakdown["encoder_digit_embeddings"] == 2 * 81 * 512
    assert breakdown["encoder_special_embeddings"] == 4 * 512
    assert breakdown["numerical_projection"] == 4 * 512 + 512
    assert breakdown["coordinate_embedding"] == 128 * 512
    assert breakdown["zero_coordinate_vector"] == 512
    assert breakdown["sparse_attention_bias"] == 8


def test_parameter_count_does_not_depend_on_n_or_on_loops():
    base = count_trainable_parameters(SalsaNact(NactSpec()))
    for loops in (1, 2, 4, 8):
        assert count_trainable_parameters(
            SalsaNact(NactSpec(encoder_loops=loops))) == base
    # max_coordinates is what n costs, and it is fixed at 128 by the design.
    assert count_trainable_parameters(
        SalsaNact(NactSpec(max_coordinates=128))) == base


def test_no_padding_parameters_every_tensor_is_used_by_the_forward_pass():
    """A gradient must reach every trainable tensor, or it is dead weight."""
    torch.manual_seed(0)
    model = SalsaNact(NactSpec())
    codec = codec_for(12, separator=False)
    # A batch containing both zero and nonzero coordinates, so the zero vector
    # and the sparse bias are both exercised.
    matrix = np.zeros((4, 12), dtype=np.int64)
    matrix[:, :6] = np.arange(1, 7)
    src, _ = codec.encode_batch(matrix)
    tgt = np.tile(np.array([1, 4, 5, 2]), (4, 1))
    logits = model(torch.from_numpy(src), torch.from_numpy(tgt))
    logits.sum().backward()
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert missing == [], f"parameters with no finite gradient: {missing}"


# --------------------------------------------------------------------------- #
# 2. front end correctness
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", DIMENSIONS)
@pytest.mark.parametrize("separator", [False, True], ids=["representation_R", "representation_P"])
def test_front_end_reconstructs_the_encoded_integers_exactly(n, separator):
    rng = np.random.default_rng(n + int(separator))
    codec = codec_for(n, separator)
    model = SalsaNact(spec_for(codec))
    matrix = rng.integers(0, 251, size=(8, n), dtype=np.int64)
    matrix[0, :] = 0                       # all-zero row
    matrix[1, :] = 250                     # all-max row
    src, _ = codec.encode_batch(matrix)
    _, values = model.front_end.decode_values(torch.from_numpy(src))
    assert np.array_equal(values.numpy(), matrix)


@pytest.mark.parametrize("n", DIMENSIONS)
@pytest.mark.parametrize("separator", [False, True], ids=["representation_R", "representation_P"])
def test_encoder_sequence_length_is_n_plus_two(n, separator):
    rng = np.random.default_rng(1000 + n)
    codec = codec_for(n, separator)
    spec = spec_for(codec)
    model = SalsaNact(spec)
    src, _ = codec.encode_batch(rng.integers(0, 251, size=(2, n), dtype=np.int64))
    assert src.shape[1] == codec.input_length            # V1 layout in
    memory = model.encode(torch.from_numpy(src))
    assert memory.shape == (2, n + 2, spec.encoder_dim)  # coordinate frame out
    assert spec.coordinates_for(codec.input_length) == n


def test_representation_r_and_p_give_the_same_encoder_length():
    for n in DIMENSIONS:
        r, p = codec_for(n, False), codec_for(n, True)
        assert r.input_length == 2 * n + 2
        assert p.input_length == 3 * n + 1
        assert spec_for(r).coordinates_for(r.input_length) == n
        assert spec_for(p).coordinates_for(p.input_length) == n


def test_numerical_features_are_correct_and_bounded():
    spec = NactSpec()
    model = SalsaNact(spec)
    values = torch.arange(251).view(1, -1)
    features = model.front_end.numerical_features(values)
    zero, centered, cosine, sine = (features[0, :, i] for i in range(4))

    assert zero[0] == 1.0 and zero[1:].sum() == 0.0
    assert torch.allclose(centered[0], torch.tensor(0.0))
    assert centered.abs().max() <= 1.0 + 1e-6
    assert torch.allclose(cosine ** 2 + sine ** 2, torch.ones(251), atol=1e-5)
    # The k=1 character separates every residue of a prime modulus.
    angles = torch.stack((cosine, sine), dim=-1)
    assert torch.unique(angles.round(decimals=4), dim=0).shape[0] == 251


def test_zero_indicator_is_zero_on_bos_and_eos():
    codec = codec_for(12, separator=False)
    model = SalsaNact(spec_for(codec))
    matrix = np.zeros((2, 12), dtype=np.int64)
    src, _ = codec.encode_batch(matrix)
    _, zero = model.front_end(torch.from_numpy(src))
    assert zero.shape == (2, 14)
    assert zero[:, 0].sum() == 0.0 and zero[:, -1].sum() == 0.0
    assert zero[:, 1:-1].sum() == 2 * 12          # every coordinate is zero


def test_sparse_attention_bias_starts_at_zero_and_is_trainable():
    model = SalsaNact(NactSpec())
    for layer in model.encoder_layers:
        assert torch.equal(layer.sparse_attention_bias,
                           torch.zeros(model.spec.encoder_heads))
        assert layer.sparse_attention_bias.requires_grad


def test_sparse_attention_bias_changes_the_output_only_once_it_is_nonzero():
    """Zero-initialised bias must be a no-op; a nonzero one must have an effect."""
    torch.manual_seed(0)
    model = SalsaNact(NactSpec()).eval()
    codec = codec_for(12, separator=False)
    matrix = np.zeros((1, 12), dtype=np.int64)
    matrix[0, :4] = [7, 19, 200, 88]
    src = torch.from_numpy(codec.encode_batch(matrix)[0])

    with torch.no_grad():
        before = model.encode(src)
        model.encoder_layers[0].sparse_attention_bias.fill_(-4.0)
        after = model.encode(src)
    assert not torch.allclose(before, after), "the bias had no effect when set"


def test_rejects_a_dimension_beyond_the_coordinate_table():
    codec = codec_for(130, separator=False)
    model = SalsaNact(NactSpec(max_coordinates=128))
    src = torch.from_numpy(codec.encode_batch(
        np.zeros((1, 130), dtype=np.int64))[0])
    with pytest.raises(ValueError, match="exceeds max_coordinates"):
        model.encode(src)


def test_rejects_a_source_length_that_is_not_a_valid_layout():
    spec = NactSpec()
    with pytest.raises(ValueError, match="not a valid representation-R layout"):
        spec.coordinates_for(27)


# --------------------------------------------------------------------------- #
# 3. interface compatibility
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", DIMENSIONS)
def test_output_shape_and_vocabulary_are_unchanged(n):
    rng = np.random.default_rng(n)
    codec = codec_for(n, separator=False)
    spec = spec_for(codec)
    model = SalsaNact(spec)
    assert codec.vocabulary.size == 85 and spec.vocab_size == 85
    src, target = codec.encode_batch(
        rng.integers(0, 251, size=(3, n), dtype=np.int64),
        rng.integers(0, 251, size=3, dtype=np.int64))
    logits = model(torch.from_numpy(src), torch.from_numpy(target))
    assert logits.shape == (3, codec.output_length, 85)
    assert torch.isfinite(logits).all()


def test_exposes_the_same_high_level_interface_as_v1():
    for name in ("encode", "decode", "forward", "name", "vocab_size", "device"):
        assert hasattr(SalsaNact(NactSpec()), name), name
    import inspect

    for method in ("encode", "decode", "forward"):
        v1 = inspect.signature(getattr(SalsaTransformer, method))
        v2 = inspect.signature(getattr(SalsaNact, method))
        assert list(v1.parameters) == list(v2.parameters), method


def test_greedy_decode_runs_against_nact_unmodified():
    """The phase-12 and phase-10 scripts drive the model through this function."""
    from salsa.training.metrics import decode_generated_ids, greedy_decode

    torch.manual_seed(0)
    codec = codec_for(12, separator=False)
    model = SalsaNact(spec_for(codec)).eval()
    matrix = np.zeros((5, 12), dtype=np.int64)
    matrix[:, 0] = 125
    src = torch.from_numpy(codec.encode_batch(matrix)[0])
    generated = greedy_decode(model, src, codec.vocabulary.bos_id,
                              codec.output_length - 1)
    assert generated.shape == (5, codec.output_length)
    assert int(generated.min()) >= 0 and int(generated.max()) < 85
    values, valid = decode_generated_ids(codec, generated.numpy())
    assert values.shape == (5,) and valid.shape == (5,)


def test_build_model_dispatches_on_arch():
    config = load_config("configs/nact_n12_h2.yaml")
    config.validate()
    model = build_model(config)
    assert isinstance(model, SalsaNact)
    assert count_trainable_parameters(model) == NACT_PARAMETERS
    assert model.name == "Salsa2-NACT"
    assert model.spec.encoder_loops == 4


def test_config_front_end_fields_come_from_the_codec():
    config = load_config("configs/nact_n12_h2.yaml")
    config.validate()
    codec = LatticeCodec.from_config(config)
    spec = NactSpec.from_config(config)
    assert spec.q == codec.q == 251
    assert spec.base == codec.input_encoder.base == 81
    assert spec.digit_width == codec.input_encoder.width == 2
    assert spec.separator == codec.separator is False
    assert spec.lsb_first is True
    # digit value d must live at vocabulary id digit_id_offset + d
    for value in (0, 1, 40, 80):
        assert codec.vocabulary.token_to_id[str(value)] == spec.digit_id_offset + value


def test_runs_on_cpu_without_cuda():
    model = SalsaNact(NactSpec())
    assert model.device.type == "cpu"
    assert all(p.device.type == "cpu" for p in model.parameters())


# --------------------------------------------------------------------------- #
# 4. V1 must be untouched
# --------------------------------------------------------------------------- #
def test_v1_parameter_count_is_unchanged():
    assert count_trainable_parameters(SalsaTransformer(ModelSpec())) == V1_PARAMETERS
    assert sum(analytical_breakdown(ModelSpec()).values()) == V1_PARAMETERS


def test_v1_forward_path_is_unchanged_by_the_optional_key_bias():
    """The new attention argument must default to a no-op."""
    torch.manual_seed(0)
    model = SalsaTransformer(ModelSpec()).eval()
    src = torch.randint(4, 85, (2, 26))
    tgt = torch.randint(4, 85, (2, 4))
    with torch.no_grad():
        reference = model(src, tgt)

    attention = model.encoder_layers[0].self_attn
    x = torch.randn(2, 26, 512)
    with torch.no_grad():
        assert torch.equal(attention(x), attention(x, key_bias=None))
    with torch.no_grad():
        assert torch.equal(model(src, tgt), reference)


def test_v1_has_no_nact_components():
    breakdown = parameter_breakdown(SalsaTransformer(ModelSpec()))
    for component in ("encoder_digit_embeddings", "numerical_projection",
                      "coordinate_embedding", "sparse_attention_bias",
                      "zero_coordinate_vector", "encoder_special_embeddings"):
        assert breakdown[component] == 0, component
    assert breakdown["encoder_embedding"] == 85 * 512


def test_v1_config_still_builds_v1():
    config = load_config("configs/control_a_n12_h2.yaml")
    config.validate()
    model = build_model(config)
    assert isinstance(model, SalsaTransformer)
    assert count_trainable_parameters(model) == V1_PARAMETERS


# --------------------------------------------------------------------------- #
# 5. Phase-22 ablation variants
# --------------------------------------------------------------------------- #
NACT_F_PARAMETERS = 4_238_208


def _flags(name):
    from salsa.models.nact import _variant_flags

    return _variant_flags(name)


def test_variant_f_parameter_count_matches_the_phase22_prediction():
    from salsa.models.nact import SalsaNact as _Nact

    model = _Nact(NactSpec(encoder_loops=2, **_flags("one_token_only")))
    assert count_trainable_parameters(model) == NACT_F_PARAMETERS


def test_variant_f_three_counting_methods_agree():
    spec = NactSpec(encoder_loops=2, **_flags("one_token_only"))
    model = SalsaNact(spec)
    measured, analytical = parameter_breakdown(model), analytical_breakdown(spec)
    assert (count_trainable_parameters(model) == sum(measured.values())
            == sum(analytical.values()) == NACT_F_PARAMETERS)
    assert unclassified_parameters(model) == []


def test_variant_f_removes_exactly_the_intended_components():
    breakdown = parameter_breakdown(SalsaNact(NactSpec(encoder_loops=2,
                                                       **_flags("one_token_only"))))
    # removed
    assert breakdown["numerical_projection"] == 0
    assert breakdown["zero_coordinate_vector"] == 0
    assert breakdown["sparse_attention_bias"] == 0
    # retained: the one-token representation and coordinate identity
    assert breakdown["encoder_digit_embeddings"] == 2 * 81 * 512
    assert breakdown["coordinate_embedding"] == 128 * 512
    assert breakdown["encoder_special_embeddings"] == 4 * 512


def test_variant_f_has_no_numerical_modules_at_all():
    """No stray parameter may sneak a numerical feature back in."""
    model = SalsaNact(NactSpec(encoder_loops=2, **_flags("one_token_only")))
    assert model.front_end.numerical_projection is None
    assert model.front_end.zero_vector is None
    for layer in model.encoder_layers:
        assert layer.sparse_attention_bias is None
    names = [n for n, _ in model.named_parameters()]
    for forbidden in ("numerical_projection", "zero_vector", "sparse_attention_bias"):
        assert not any(forbidden in n for n in names), forbidden


def test_variant_f_still_encodes_and_decodes():
    torch.manual_seed(0)
    codec = codec_for(12, separator=False)
    model = SalsaNact(NactSpec(vocab_size=codec.vocabulary.size, q=codec.q,
                               base=codec.input_encoder.base,
                               digit_width=codec.input_encoder.width,
                               separator=codec.separator, encoder_loops=2,
                               **_flags("one_token_only"))).eval()
    matrix = np.zeros((4, 12), dtype=np.int64)
    matrix[:, :5] = np.arange(1, 6)
    src, target = codec.encode_batch(matrix, np.arange(4, dtype=np.int64))
    logits = model(torch.from_numpy(src), torch.from_numpy(target))
    assert logits.shape == (4, codec.output_length, 85)
    assert torch.isfinite(logits).all()
    assert model.encode(torch.from_numpy(src)).shape == (4, 14, 512)


def test_full_nact_is_unchanged_by_the_ablation_switches():
    """The defaults must reproduce the shipped full NACT exactly."""
    spec = NactSpec(encoder_loops=2)
    assert spec.variant == "full"
    assert spec.use_numerical_features and spec.use_zero_vector
    assert spec.use_sparse_attention_bias
    model = SalsaNact(spec)
    assert count_trainable_parameters(model) == NACT_PARAMETERS
    assert model.front_end.numerical_projection is not None
    assert model.front_end.zero_vector is not None
    assert model.encoder_layers[0].sparse_attention_bias is not None


def test_existing_full_nact_checkpoints_still_load_strictly():
    """The switches must not have changed the full-NACT state_dict shape."""
    from pathlib import Path as _Path

    root = _Path("results/equal_depth_ablation/v2_te2/nact_n12_h2_te2")
    checkpoint_path = root / "seed_123" / "checkpoints" / "best.pt"
    if not checkpoint_path.is_file():
        pytest.skip("phase-17 checkpoint not present")
    config = load_config("configs/nact_n12_h2_te2_recovery.yaml")
    config.validate()
    model = build_model(config)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    assert count_trainable_parameters(model) == NACT_PARAMETERS


def test_unknown_variant_is_rejected():
    from salsa.models.nact import _variant_flags as flags

    with pytest.raises(ValueError, match="unknown nact_variant"):
        flags("nonsense")
