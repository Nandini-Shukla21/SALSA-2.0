"""Tests for the compact SALSA 2.0 models.

These check the things that would silently corrupt every later phase: that the
shared layer really is one module, that raising the loop count does not grow
the model, that RoPE is length-independent, that the causal mask actually
hides the future, and that the forward pass produces finite logits of the right
shape at every ``n`` and under both representations.
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from salsa.data import LatticeCodec, LWEParams, generate_lwe_sample  # noqa: E402
from salsa.models import (  # noqa: E402
    CopyGate,
    ModelSpec,
    MultiHeadAttention,
    RMSNorm,
    RotaryPositionalEmbedding,
    SalsaTransformer,
    TokenEmbedding,
    apply_rope,
    build_model,
    count_trainable_parameters,
)
from salsa.training.seed import set_seed  # noqa: E402
from salsa.utils.config import load_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

VOCAB = 85
DIMENSIONS = [30, 50, 70, 90, 128]
# (representation, n) -> encoder sequence length
LENGTHS = {("R", n): 2 * n + 2 for n in DIMENSIONS}
LENGTHS.update({("P", n): 3 * n + 1 for n in DIMENSIONS})


def target_spec(**overrides) -> ModelSpec:
    """The approved gated Universal Transformer, with optional overrides."""
    settings = dict(
        vocab_size=VOCAB, encoder_dim=512, decoder_dim=128,
        encoder_layers=1, decoder_layers=1, encoder_heads=8, decoder_heads=4,
        encoder_loops=2, decoder_loops=2, gated=True,
        arch="gated_universal_transformer",
    )
    settings.update(overrides)
    return ModelSpec(**settings)


def control_spec(**overrides) -> ModelSpec:
    """The approved compact transformer control, with optional overrides."""
    settings = dict(
        vocab_size=VOCAB, encoder_dim=384, decoder_dim=128,
        encoder_layers=2, decoder_layers=2, encoder_heads=6, decoder_heads=4,
        encoder_loops=2, decoder_loops=2, gated=False,
        arch="compact_transformer",
    )
    settings.update(overrides)
    return ModelSpec(**settings)


ALL_SPECS = [("target", target_spec), ("control", control_spec)]


# --------------------------------------------------------------------------- #
# 1. Construction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,factory", ALL_SPECS)
def test_model_constructs(label: str, factory) -> None:
    """Both approved architectures build on the CPU."""
    model = SalsaTransformer(factory())
    assert isinstance(model, torch.nn.Module)
    assert model.device.type == "cpu"
    assert model.vocab_size == VOCAB
    assert count_trainable_parameters(model) > 0


def test_models_build_from_the_shipped_configs() -> None:
    """The config files construct the architectures they claim to."""
    target = build_model(load_config(CONFIG_DIR / "target_4_5m.yaml"))
    assert target.name == "Salsa2-GatedUT"
    assert target.spec.encoder_dim == 512 and target.spec.encoder_heads == 8
    assert target.spec.encoder_layers == 1 and target.spec.encoder_loops == 2
    assert target.spec.gated is True

    control = build_model(load_config(CONFIG_DIR / "control_a.yaml"))
    assert control.name == "Salsa2-CompactTransformer"
    assert control.spec.encoder_dim == 384 and control.spec.encoder_layers == 2
    assert control.spec.gated is False


def test_model_vocab_follows_the_codec() -> None:
    """The model is built for exactly the vocabulary its codec produces."""
    config = load_config(CONFIG_DIR / "target_4_5m.yaml")
    codec = LatticeCodec.from_config(config)
    assert build_model(config).vocab_size == codec.vocabulary.size == VOCAB


def test_invalid_specs_are_rejected() -> None:
    """Impossible geometries fail at construction, not at the first forward."""
    with pytest.raises(ValueError, match="divisible"):
        target_spec(encoder_dim=513, encoder_heads=8)
    with pytest.raises(ValueError, match="even for RoPE"):
        target_spec(encoder_dim=24, encoder_heads=8)
    with pytest.raises(ValueError, match="encoder_loops must be >="):
        target_spec(encoder_layers=4, encoder_loops=2)
    with pytest.raises(ValueError, match="arch must be"):
        target_spec(arch="mamba")


# --------------------------------------------------------------------------- #
# 5/6. Weight sharing
# --------------------------------------------------------------------------- #
def test_shared_layer_is_literally_the_same_module() -> None:
    """The Universal Transformer must reuse one module, not clone it."""
    model = SalsaTransformer(target_spec(encoder_loops=4, decoder_loops=4))
    assert len(model.encoder_layers) == 1
    assert len(model.decoder_layers) == 1
    first, second = model.encoder_layer_for(0), model.encoder_layer_for(3)
    assert first is second, "looped passes must reuse the same trainable module"
    assert model.decoder_layer_for(0) is model.decoder_layer_for(3)
    # ...and the same parameter objects, so one gradient accumulates over passes.
    assert first.self_attn.q_proj.weight is second.self_attn.q_proj.weight


def test_unshared_control_uses_distinct_modules() -> None:
    """The control must NOT share: that is the whole point of the comparison."""
    model = SalsaTransformer(control_spec())
    assert len(model.encoder_layers) == 2
    assert model.encoder_layer_for(0) is not model.encoder_layer_for(1)
    assert (
        model.encoder_layer_for(0).self_attn.q_proj.weight
        is not model.encoder_layer_for(1).self_attn.q_proj.weight
    )


@pytest.mark.parametrize("loops", [1, 2, 3, 4, 8])
def test_encoder_loops_do_not_change_the_parameter_count(loops: int) -> None:
    """parameter_count(T_e=2) == parameter_count(T_e=4)."""
    reference = count_trainable_parameters(SalsaTransformer(target_spec(encoder_loops=2)))
    assert count_trainable_parameters(
        SalsaTransformer(target_spec(encoder_loops=loops))
    ) == reference


@pytest.mark.parametrize("loops", [1, 2, 3, 4, 8])
def test_decoder_loops_do_not_change_the_parameter_count(loops: int) -> None:
    """parameter_count(T_d=2) == parameter_count(T_d=4)."""
    reference = count_trainable_parameters(SalsaTransformer(target_spec(decoder_loops=2)))
    assert count_trainable_parameters(
        SalsaTransformer(target_spec(decoder_loops=loops))
    ) == reference


def test_more_loops_do_change_the_computation() -> None:
    """Sharing must not degenerate into skipping the extra passes."""
    set_seed(0)
    shallow = SalsaTransformer(target_spec(encoder_loops=2))
    deep = SalsaTransformer(target_spec(encoder_loops=4))
    deep.load_state_dict(shallow.state_dict())
    shallow.eval(), deep.eval()

    src = torch.randint(1, VOCAB, (2, 62))
    with torch.no_grad():
        assert not torch.allclose(shallow.encode(src), deep.encode(src))


# --------------------------------------------------------------------------- #
# Copy gate
# --------------------------------------------------------------------------- #
def test_copy_gate_is_learned_not_an_identity() -> None:
    """The gate has trainable weights and mixes both of its inputs."""
    gate = CopyGate(16)
    assert count_trainable_parameters(gate) == 2 * 16 * 16 + 16

    x = torch.randn(2, 3, 16)
    h = torch.randn(2, 3, 16)
    values = gate.gate_values(x, h)
    assert values.shape == x.shape
    assert torch.all((values > 0) & (values < 1))

    out = gate(x, h)
    assert not torch.allclose(out, x), "gate must not collapse to a copy"
    assert not torch.allclose(out, h), "gate must not collapse to the transform"

    out.sum().backward()
    assert gate.proj.weight.grad is not None
    assert gate.proj.bias.grad is not None
    assert torch.any(gate.proj.weight.grad != 0)


def test_copy_gate_blend_is_exact() -> None:
    """out == g*h + (1-g)*x, checked numerically."""
    gate = CopyGate(8)
    x, h = torch.randn(1, 2, 8), torch.randn(1, 2, 8)
    with torch.no_grad():
        g = gate.gate_values(x, h)
        assert torch.allclose(gate(x, h), g * h + (1 - g) * x, atol=1e-6)


def test_gated_model_exposes_gates_and_control_does_not() -> None:
    """Gates exist exactly where the architecture says they do."""
    gated = SalsaTransformer(target_spec())
    assert gated.encoder_layers[0].gate is not None
    assert gated.decoder_layers[0].gate is not None
    plain = SalsaTransformer(control_spec())
    assert plain.encoder_layers[0].gate is None
    assert plain.decoder_layers[0].gate is None


# --------------------------------------------------------------------------- #
# 15. RoPE
# --------------------------------------------------------------------------- #
def test_rope_has_no_trainable_parameters() -> None:
    """Positional encoding must cost zero budget."""
    rope = RotaryPositionalEmbedding(head_dim=64)
    assert count_trainable_parameters(rope) == 0
    assert "inv_freq" not in rope.state_dict(), "inv_freq must not enter checkpoints"


@pytest.mark.parametrize("length", [10, 62, 91, 258, 385])
def test_rope_supports_any_length(length: int) -> None:
    """The tables are built on demand for whatever length is asked for."""
    rope = RotaryPositionalEmbedding(head_dim=64)
    cos, sin = rope(length, torch.device("cpu"), torch.float32)
    assert cos.shape == sin.shape == (length, 32)
    assert torch.all(torch.isfinite(cos)) and torch.all(torch.isfinite(sin))


def test_rope_encodes_relative_position() -> None:
    """<rope(q, m), rope(k, n)> depends only on m - n."""
    rope = RotaryPositionalEmbedding(head_dim=16)
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)

    def dot(m: int, n: int) -> float:
        cq, sq = rope(1, torch.device("cpu"), torch.float32, offset=m)
        ck, sk = rope(1, torch.device("cpu"), torch.float32, offset=n)
        return float((apply_rope(q, cq, sq) * apply_rope(k, ck, sk)).sum())

    assert dot(5, 3) == pytest.approx(dot(105, 103), abs=1e-4)
    assert dot(0, 7) == pytest.approx(dot(300, 307), abs=1e-4)
    assert dot(5, 3) != pytest.approx(dot(5, 4), abs=1e-3)


def test_rope_preserves_norms() -> None:
    """A rotation must not change vector length."""
    rope = RotaryPositionalEmbedding(head_dim=32)
    x = torch.randn(2, 4, 91, 32)
    cos, sin = rope(91, torch.device("cpu"), torch.float32)
    assert torch.allclose(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_rope_rejects_bad_shapes() -> None:
    """Malformed inputs raise instead of silently broadcasting."""
    with pytest.raises(ValueError):
        RotaryPositionalEmbedding(head_dim=15)
    rope = RotaryPositionalEmbedding(head_dim=16)
    cos, sin = rope(4, torch.device("cpu"))
    with pytest.raises(ValueError):
        apply_rope(torch.randn(4, 16), cos, sin)
    with pytest.raises(ValueError):
        apply_rope(torch.randn(1, 1, 5, 16), cos, sin)


def test_parameter_count_is_independent_of_sequence_length() -> None:
    """One checkpoint serves every n and both representations."""
    model = SalsaTransformer(target_spec())
    reference = count_trainable_parameters(model)
    model.eval()
    for length in (10, 62, 91, 258, 385):
        with torch.no_grad():
            model(torch.randint(1, VOCAB, (1, length)), torch.randint(1, VOCAB, (1, 4)))
        assert count_trainable_parameters(model) == reference


# --------------------------------------------------------------------------- #
# 7-10. Forward pass
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,factory", ALL_SPECS)
def test_forward_shape_and_finiteness(label: str, factory) -> None:
    """Logits are (batch, tgt_len, vocab) and contain no NaN or inf."""
    model = SalsaTransformer(factory()).eval()
    src = torch.randint(1, VOCAB, (4, 62))
    tgt = torch.randint(1, VOCAB, (4, 4))
    with torch.no_grad():
        logits = model(src, tgt)
    assert logits.shape == (4, 4, VOCAB)
    assert torch.all(torch.isfinite(logits))
    assert logits.dtype == torch.float32


@pytest.mark.parametrize("n", DIMENSIONS)
@pytest.mark.parametrize("representation", ["R", "P"])
def test_forward_at_every_dimension_and_representation(n: int, representation: str) -> None:
    """n = 30, 50, 70, 90, 128 under both R and P, with one architecture."""
    length = LENGTHS[(representation, n)]
    model = SalsaTransformer(target_spec()).eval()
    src = torch.randint(1, VOCAB, (2, length))
    tgt = torch.randint(1, VOCAB, (2, 4))
    with torch.no_grad():
        logits = model(src, tgt)
    assert logits.shape == (2, 4, VOCAB)
    assert torch.all(torch.isfinite(logits))


def test_representation_lengths_match_the_codec() -> None:
    """The lengths used above are the ones the codec actually produces."""
    for representation, name in (("R", "representation_r"), ("P", "representation_p")):
        for n in DIMENSIONS:
            config = load_config(CONFIG_DIR / f"{name}.yaml", overrides=[f"lwe.n={n}"])
            assert LatticeCodec.from_config(config).input_length == LENGTHS[(representation, n)]


def test_forward_on_real_encoded_lwe_data() -> None:
    """End to end: phase-2 samples -> phase-3 codec -> phase-5 model."""
    config = load_config(
        CONFIG_DIR / "target_4_5m.yaml", overrides=["lwe.structure=lwe"]
    )
    codec = LatticeCodec.from_config(config)
    params = LWEParams(n=config.lwe.n, q=config.lwe.q, sigma=config.lwe.sigma,
                       hamming_weight=config.lwe.resolved_hamming_weight)
    sample = generate_lwe_sample(params, num_instances=8, seed=0)

    src_ids, tgt_ids = codec.encode_batch(sample.A, sample.b)
    model = build_model(config).eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(src_ids), torch.from_numpy(tgt_ids))
    assert logits.shape == (8, codec.output_length, codec.vocabulary.size)
    assert torch.all(torch.isfinite(logits))


def test_output_length_matches_two_digits_plus_markers() -> None:
    """q=251 in base 81 gives 2 digits; with <bos>/<eos> the target is 4 tokens."""
    config = load_config(CONFIG_DIR / "target_4_5m.yaml")
    codec = LatticeCodec.from_config(config)
    assert codec.output_encoder.width == 2
    assert codec.output_length == 4

    model = build_model(config).eval()
    tgt = torch.randint(1, VOCAB, (3, codec.output_length))
    with torch.no_grad():
        logits = model(torch.randint(1, VOCAB, (3, codec.input_length)), tgt)
    assert logits.shape == (3, 4, VOCAB)
    # Teacher forcing feeds tgt[:, :-1] and scores tgt[:, 1:].
    with torch.no_grad():
        shifted = model(torch.randint(1, VOCAB, (3, codec.input_length)), tgt[:, :-1])
    assert shifted.shape == (3, 3, VOCAB)


@pytest.mark.parametrize("src_len", [4, 10, 62, 91, 258, 385])
@pytest.mark.parametrize("tgt_len", [1, 3, 4])
def test_variable_sequence_lengths(src_len: int, tgt_len: int) -> None:
    """The model accepts any source and target length without rebuilding."""
    model = SalsaTransformer(target_spec()).eval()
    with torch.no_grad():
        logits = model(
            torch.randint(1, VOCAB, (2, src_len)), torch.randint(1, VOCAB, (2, tgt_len))
        )
    assert logits.shape == (2, tgt_len, VOCAB)


def test_batch_size_one_and_large_batches() -> None:
    """No hidden assumption about the batch dimension."""
    model = SalsaTransformer(target_spec()).eval()
    for batch in (1, 2, 33):
        with torch.no_grad():
            logits = model(
                torch.randint(1, VOCAB, (batch, 62)), torch.randint(1, VOCAB, (batch, 4))
            )
        assert logits.shape == (batch, 4, VOCAB)


def test_causal_mask_hides_the_future() -> None:
    """Logits at position t must not depend on target tokens after t."""
    model = SalsaTransformer(target_spec()).eval()
    src = torch.randint(1, VOCAB, (1, 62))
    tgt = torch.randint(1, VOCAB, (1, 4))
    altered = tgt.clone()
    altered[0, -1] = (altered[0, -1] + 1) % VOCAB

    with torch.no_grad():
        first = model(src, tgt)
        second = model(src, altered)
    assert torch.allclose(first[:, :-1], second[:, :-1], atol=1e-6)
    assert not torch.allclose(first[:, -1], second[:, -1], atol=1e-6)


def test_padding_mask_excludes_padded_positions() -> None:
    """Content at masked source positions cannot influence the output."""
    model = SalsaTransformer(target_spec()).eval()
    src = torch.randint(1, VOCAB, (2, 20))
    valid = torch.ones(2, 20, dtype=torch.bool)
    valid[:, 12:] = False
    polluted = src.clone()
    polluted[:, 12:] = torch.randint(1, VOCAB, (2, 8))
    tgt = torch.randint(1, VOCAB, (2, 4))

    with torch.no_grad():
        first = model(src, tgt, src_valid=valid)
        second = model(polluted, tgt, src_valid=valid)
    assert torch.allclose(first, second, atol=1e-6)
    assert torch.all(torch.isfinite(first))


def test_encode_and_decode_compose_into_forward() -> None:
    """forward() is exactly encode() followed by decode()."""
    model = SalsaTransformer(target_spec()).eval()
    src = torch.randint(1, VOCAB, (2, 62))
    tgt = torch.randint(1, VOCAB, (2, 4))
    with torch.no_grad():
        assert torch.allclose(
            model(src, tgt), model.decode(model.encode(src), tgt), atol=1e-6
        )


def test_forward_rejects_malformed_inputs() -> None:
    """Shape errors are raised, not broadcast around."""
    model = SalsaTransformer(target_spec()).eval()
    with pytest.raises(ValueError, match="src_ids must be"):
        model(torch.randint(1, VOCAB, (62,)), torch.randint(1, VOCAB, (1, 4)))
    with pytest.raises(ValueError, match="tgt_ids must be"):
        model(torch.randint(1, VOCAB, (1, 62)), torch.randint(1, VOCAB, (4,)))
    with pytest.raises(ValueError, match="batch dimension"):
        model.decode(torch.randn(2, 62, 512), torch.randint(1, VOCAB, (3, 4)))


def test_gradients_flow_to_every_trainable_parameter() -> None:
    """A backward pass reaches all parameters (nothing is dead weight)."""
    model = SalsaTransformer(target_spec())
    logits = model(torch.randint(1, VOCAB, (2, 62)), torch.randint(1, VOCAB, (2, 4)))
    logits.sum().backward()
    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (parameter.grad is None or torch.all(parameter.grad == 0))
    ]
    # The pad row of each embedding legitimately receives no gradient.
    assert missing == [], f"parameters without gradient: {missing}"


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
def test_rmsnorm_normalises() -> None:
    """RMSNorm scales to unit RMS and has exactly `dim` parameters."""
    norm = RMSNorm(64)
    assert count_trainable_parameters(norm) == 64
    x = torch.randn(3, 5, 64) * 10
    out = norm(x)
    assert torch.allclose(out.pow(2).mean(-1).sqrt(), torch.ones(3, 5), atol=1e-3)


def test_token_embedding_keeps_the_pad_row_at_zero() -> None:
    """The pad id must embed to zero so padding contributes nothing."""
    embedding = TokenEmbedding(VOCAB, 32, padding_idx=0)
    assert torch.all(embedding.weight[0] == 0)
    assert count_trainable_parameters(embedding) == VOCAB * 32


def test_attention_parameter_shapes() -> None:
    """Self- and cross-attention cost what the formulas say."""
    self_attn = MultiHeadAttention(query_dim=512, num_heads=8)
    assert count_trainable_parameters(self_attn) == 4 * 512 * 512

    cross_attn = MultiHeadAttention(query_dim=128, num_heads=4, kv_dim=512, use_rope=False)
    assert count_trainable_parameters(cross_attn) == 2 * 128 * 128 + 2 * 512 * 128
    assert cross_attn.rope is None


def test_cross_attention_accepts_a_different_kv_width() -> None:
    """The decoder reads a 512-wide memory with a 128-wide query stream."""
    attn = MultiHeadAttention(query_dim=128, num_heads=4, kv_dim=512, use_rope=False)
    out = attn(torch.randn(2, 4, 128), key_value=torch.randn(2, 91, 512))
    assert out.shape == (2, 4, 128)


# --------------------------------------------------------------------------- #
# 16-18. Devices, determinism, portability
# --------------------------------------------------------------------------- #
def test_model_moves_to_an_explicitly_requested_device() -> None:
    """CPU always works; an accelerator is used only when actually present."""
    from salsa.utils.device import resolve_device

    model = SalsaTransformer(target_spec())
    model.to(torch.device("cpu"))
    assert model.device.type == "cpu"

    spec = resolve_device("auto")
    model.to(spec.torch_device())
    assert model.device.type == spec.type
    src = torch.randint(1, VOCAB, (1, 62), device=spec.torch_device())
    tgt = torch.randint(1, VOCAB, (1, 4), device=spec.torch_device())
    model.eval()
    with torch.no_grad():
        assert model(src, tgt).device.type == spec.type


def test_model_runs_without_any_cuda_device() -> None:
    """The default path never touches CUDA."""
    model = SalsaTransformer(target_spec()).eval()
    assert all(p.device.type == "cpu" for p in model.parameters())
    with torch.no_grad():
        logits = model(torch.randint(1, VOCAB, (2, 62)), torch.randint(1, VOCAB, (2, 4)))
    assert logits.device.type == "cpu"


def test_model_sources_contain_no_cuda_specific_code() -> None:
    """No .cuda() calls, no device hard-coding, no custom kernels."""
    for path in sorted((REPO_ROOT / "salsa" / "models").glob("*.py")):
        source = path.read_text(encoding="utf-8").lower()
        for marker in (".cuda(", "cuda:0", "torch.cuda.", "device='cuda'", 'device="cuda"'):
            assert marker not in source, f"{path.name} contains {marker!r}"


@pytest.mark.parametrize("label,factory", ALL_SPECS)
def test_initialisation_is_reproducible(label: str, factory) -> None:
    """Same seed in, byte-identical weights out."""
    set_seed(1234)
    first = SalsaTransformer(factory())
    set_seed(1234)
    second = SalsaTransformer(factory())
    for (name_a, a), (name_b, b) in zip(first.named_parameters(), second.named_parameters()):
        assert name_a == name_b
        assert torch.equal(a, b), f"{name_a} differs between identically seeded builds"


def test_different_seeds_give_different_weights() -> None:
    """Seeding actually varies initialisation."""
    set_seed(1)
    first = SalsaTransformer(target_spec())
    set_seed(2)
    second = SalsaTransformer(target_spec())
    assert not torch.equal(
        first.encoder_layers[0].ffn.w1.weight, second.encoder_layers[0].ffn.w1.weight
    )


def test_forward_is_deterministic_in_eval_mode() -> None:
    """Two identical eval forwards give identical logits."""
    model = SalsaTransformer(target_spec()).eval()
    src = torch.randint(1, VOCAB, (2, 62))
    tgt = torch.randint(1, VOCAB, (2, 4))
    with torch.no_grad():
        assert torch.equal(model(src, tgt), model(src, tgt))


def test_describe_is_serialisable_for_run_metadata() -> None:
    """The model description can be written into metadata.json."""
    import json

    description = SalsaTransformer(target_spec()).describe()
    json.dumps(description)
    assert description["name"] == "Salsa2-GatedUT"
    assert description["positional_encoding"] == "rope"
    assert description["trainable_parameters"] == 4_131_200
    assert description["shares_encoder_weights"] is True
