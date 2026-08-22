"""Tests for parameter accounting and the hard 4-5M budget.

The budget is a research constraint, so it is enforced here rather than
described in a document: a model above 5,000,000 trainable parameters fails
these tests.  The measured count is also cross-checked against the analytical
formulas, so a silent architectural change cannot pass unnoticed.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from salsa.models import (  # noqa: E402
    COMPONENTS,
    ModelSpec,
    SalsaTransformer,
    analytical_breakdown,
    analytical_parameter_count,
    build_model,
    check_budget,
    count_all_parameters,
    count_trainable_parameters,
    format_parameter_report,
    parameter_breakdown,
    parameter_report,
    report_from_config,
    unclassified_parameters,
)
from salsa.utils.config import load_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"

HARD_LIMIT = 5_000_000
TARGET_FLOOR = 4_000_000

# Analytical predictions established in phase 4, corrected to V=85.
EXPECTED_TARGET = 4_131_200
EXPECTED_CONTROL = 4_251_520


def target_model() -> SalsaTransformer:
    """Build the approved gated Universal Transformer."""
    return build_model(load_config(CONFIG_DIR / "target_4_5m.yaml"))


def control_model() -> SalsaTransformer:
    """Build the approved compact transformer control."""
    return build_model(load_config(CONFIG_DIR / "control_a.yaml"))


MODELS = [("target_4_5m", target_model, EXPECTED_TARGET),
          ("control_a", control_model, EXPECTED_CONTROL)]


# --------------------------------------------------------------------------- #
# The hard limit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,factory,expected", MODELS)
def test_model_never_exceeds_the_hard_parameter_limit(name, factory, expected) -> None:
    """FAILS if actual_parameter_count > 5,000,000."""
    count = count_trainable_parameters(factory())
    assert count <= HARD_LIMIT, (
        f"{name} is a FAILED BUDGET configuration: {count:,} > {HARD_LIMIT:,}"
    )


@pytest.mark.parametrize("name,factory,expected", MODELS)
def test_model_reaches_the_target_floor(name, factory, expected) -> None:
    """The approved configurations must be at least 4,000,000 parameters."""
    count = count_trainable_parameters(factory())
    assert count >= TARGET_FLOOR, f"{name} is under target: {count:,} < {TARGET_FLOOR:,}"


@pytest.mark.parametrize("name,factory,expected", MODELS)
def test_exact_parameter_count(name, factory, expected) -> None:
    """The counts are exactly what phase 4 predicted at V=85."""
    assert count_trainable_parameters(factory()) == expected


def test_every_shipped_model_config_respects_its_own_ceiling() -> None:
    """No shipped configuration may build a model above its declared maximum."""
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        config = load_config(path)
        if not config.experiment.name or path.name.startswith("representation_"):
            continue
        report = report_from_config(config)
        assert report.within_budget, (
            f"{path.name}: FAILED BUDGET, {report.trainable:,} > "
            f"{config.model.max_parameters:,}"
        )


# --------------------------------------------------------------------------- #
# Measured vs analytical
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,factory,expected", MODELS)
def test_measured_count_matches_the_analytical_formula(name, factory, expected) -> None:
    """PyTorch and the formulas must agree exactly, or one of them is wrong."""
    model = factory()
    measured = count_trainable_parameters(model)
    predicted = analytical_parameter_count(model.spec)
    assert measured == predicted, (
        f"{name}: measured {measured:,} != analytical {predicted:,}"
    )


@pytest.mark.parametrize("name,factory,expected", MODELS)
def test_component_breakdown_matches_component_by_component(name, factory, expected) -> None:
    """Not just the total: every component must match its formula."""
    model = factory()
    measured = parameter_breakdown(model)
    predicted = analytical_breakdown(model.spec)
    mismatches = {k: (measured[k], predicted[k]) for k in COMPONENTS
                  if measured[k] != predicted[k]}
    assert mismatches == {}, f"{name}: component mismatches {mismatches}"


@pytest.mark.parametrize("name,factory,expected", MODELS)
def test_breakdown_sums_to_the_total(name, factory, expected) -> None:
    """The breakdown must account for every parameter, with none left over."""
    model = factory()
    assert sum(parameter_breakdown(model).values()) == count_trainable_parameters(model)


@pytest.mark.parametrize("name,factory,expected", MODELS)
def test_no_parameter_is_unclassified(name, factory, expected) -> None:
    """Nothing may fall into the 'other' bucket unnoticed."""
    model = factory()
    assert unclassified_parameters(model) == []
    assert parameter_breakdown(model)["other"] == 0


@pytest.mark.parametrize("name,factory,expected", MODELS)
def test_all_parameters_are_trainable(name, factory, expected) -> None:
    """No frozen weights are hiding in the count."""
    model = factory()
    assert count_all_parameters(model) == count_trainable_parameters(model)


def test_positional_encoding_costs_nothing() -> None:
    """RoPE contributes zero parameters, measured and predicted."""
    model = target_model()
    assert parameter_breakdown(model)["positional_encoding"] == 0
    assert analytical_breakdown(model.spec)["positional_encoding"] == 0


def test_expected_component_values_for_the_target() -> None:
    """Spot-check the individual formulas at V=85, d_e=512, d_d=128."""
    breakdown = parameter_breakdown(target_model())
    assert breakdown["encoder_embedding"] == 85 * 512 == 43_520
    assert breakdown["decoder_embedding"] == 85 * 128 == 10_880
    assert breakdown["encoder_attention"] == 4 * 512 ** 2 == 1_048_576
    assert breakdown["encoder_ffn"] == 2 * 512 * 2048 == 2_097_152
    assert breakdown["encoder_copy_gate"] == 2 * 512 ** 2 + 512 == 524_800
    assert breakdown["encoder_normalization"] == 2 * 512 + 512 == 1_536
    assert breakdown["decoder_self_attention"] == 4 * 128 ** 2 == 65_536
    assert breakdown["decoder_cross_attention"] == 2 * 128 ** 2 + 2 * 512 * 128 == 163_840
    assert breakdown["decoder_ffn"] == 2 * 128 * 512 == 131_072
    assert breakdown["decoder_copy_gate"] == 2 * 128 ** 2 + 128 == 32_896
    assert breakdown["decoder_normalization"] == 3 * 128 + 128 == 512
    assert breakdown["output_projection"] == 128 * 85 == 10_880


def test_control_has_no_gate_parameters() -> None:
    """The control's budget goes to width and depth, not gating."""
    breakdown = parameter_breakdown(control_model())
    assert breakdown["encoder_copy_gate"] == 0
    assert breakdown["decoder_copy_gate"] == 0


# --------------------------------------------------------------------------- #
# Loop invariance
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("encoder_loops", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("decoder_loops", [1, 2, 4])
def test_loop_counts_never_change_the_budget(encoder_loops: int, decoder_loops: int) -> None:
    """Effective depth is free: T_e and T_d do not appear in any formula."""
    spec = ModelSpec(
        vocab_size=85, encoder_dim=512, decoder_dim=128,
        encoder_layers=1, decoder_layers=1, encoder_heads=8, decoder_heads=4,
        encoder_loops=encoder_loops, decoder_loops=decoder_loops, gated=True,
    )
    model = SalsaTransformer(spec)
    assert count_trainable_parameters(model) == EXPECTED_TARGET
    assert analytical_parameter_count(spec) == EXPECTED_TARGET


def test_adding_a_parameter_set_does_change_the_budget() -> None:
    """Sharing saves parameters only because a second set would cost them."""
    shared = ModelSpec(vocab_size=85, encoder_dim=512, decoder_dim=128,
                       encoder_layers=1, decoder_layers=1, encoder_heads=8,
                       decoder_heads=4, encoder_loops=2, decoder_loops=2, gated=True)
    unshared = ModelSpec(**{**shared.__dict__, "encoder_layers": 2})
    assert analytical_parameter_count(unshared) > analytical_parameter_count(shared)
    assert count_trainable_parameters(SalsaTransformer(unshared)) > EXPECTED_TARGET


# --------------------------------------------------------------------------- #
# Budget classification
# --------------------------------------------------------------------------- #
def test_check_budget_classifies_correctly() -> None:
    """Over the ceiling fails; under the floor is flagged; between is OK."""
    assert check_budget(5_000_001, 5_000_000, 4_000_000) == "FAILED BUDGET"
    assert check_budget(5_000_000, 5_000_000, 4_000_000) == "OK"
    assert check_budget(3_999_999, 5_000_000, 4_000_000) == "UNDER TARGET"
    assert check_budget(4_131_200, 5_000_000, 4_000_000) == "OK"
    assert check_budget(9_000_000) == "OK"  # no ceiling supplied


def test_an_oversized_model_is_reported_as_failed_budget() -> None:
    """A configuration that busts the ceiling must say so, not be trimmed."""
    spec = ModelSpec(vocab_size=85, encoder_dim=1024, decoder_dim=512,
                     encoder_layers=2, decoder_layers=2, encoder_heads=16,
                     decoder_heads=8, encoder_loops=2, decoder_loops=2, gated=True)
    report = parameter_report(SalsaTransformer(spec), max_parameters=HARD_LIMIT,
                              min_parameters=TARGET_FLOOR)
    assert report.trainable > HARD_LIMIT
    assert report.budget_status == "FAILED BUDGET"
    assert report.within_budget is False
    assert "FAILED BUDGET" in format_parameter_report(report)


def test_an_undersized_model_is_flagged() -> None:
    """Falling below the floor is reported too, so the band is honest both ways."""
    spec = ModelSpec(vocab_size=85, encoder_dim=64, decoder_dim=32,
                     encoder_layers=1, decoder_layers=1, encoder_heads=2,
                     decoder_heads=2, encoder_loops=2, decoder_loops=2, gated=True)
    report = parameter_report(SalsaTransformer(spec), max_parameters=HARD_LIMIT,
                              min_parameters=TARGET_FLOOR)
    assert report.budget_status == "UNDER TARGET"


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,factory,expected", MODELS)
def test_report_contents(name, factory, expected) -> None:
    """The report carries everything needed for the experimental record."""
    config = load_config(CONFIG_DIR / f"{name}.yaml")
    report = report_from_config(config, model=factory())
    assert report.trainable == expected
    assert report.matches_analytical
    assert report.component_mismatches == {}
    assert report.budget_status == "OK"
    assert report.max_parameters == 5_000_000
    assert report.fp32_bytes == expected * 4
    assert 15.0 < report.fp32_megabytes < 17.0


@pytest.mark.parametrize("name,factory,expected", MODELS)
def test_report_is_json_serialisable(name, factory, expected) -> None:
    """The report goes into metadata.json alongside the config."""
    import json

    payload = parameter_report(factory(), 5_000_000, 4_000_000).to_dict()
    json.dumps(payload)
    assert payload["trainable_parameters"] == expected
    assert payload["matches_analytical"] is True
    assert payload["budget_status"] == "OK"
    assert sum(payload["breakdown"].values()) == expected


@pytest.mark.parametrize("name,factory,expected", MODELS)
def test_formatted_report_reads_correctly(name, factory, expected) -> None:
    """The human-readable report states the required facts."""
    text = format_parameter_report(parameter_report(factory(), 5_000_000, 4_000_000))
    assert f"Trainable parameters: {expected:,}" in text
    assert "FP32 size:" in text
    assert "Vocabulary: 85" in text
    assert "MISMATCH" not in text
    assert "Budget:" in text and text.rstrip().endswith("OK")


def test_tying_embeddings_removes_the_output_projection() -> None:
    """Tied weights are accounted for, not double counted."""
    untied = ModelSpec(vocab_size=85, tie_embeddings=False)
    tied = ModelSpec(vocab_size=85, tie_embeddings=True)
    difference = analytical_parameter_count(untied) - analytical_parameter_count(tied)
    assert difference == 85 * 128

    model = SalsaTransformer(tied)
    assert model.output_projection is None
    assert parameter_breakdown(model)["output_projection"] == 0
    assert count_trainable_parameters(model) == analytical_parameter_count(tied)
    with torch.no_grad():
        logits = model(torch.randint(1, 85, (2, 62)), torch.randint(1, 85, (2, 4)))
    assert logits.shape == (2, 4, 85)


def test_vocabulary_size_enters_the_count_as_expected() -> None:
    """V=85 vs V=86 differ by exactly one row of each vocabulary matrix."""
    v85 = analytical_parameter_count(ModelSpec(vocab_size=85))
    v86 = analytical_parameter_count(ModelSpec(vocab_size=86))
    assert v85 == EXPECTED_TARGET
    assert v86 - v85 == 512 + 128 + 128
    assert v86 == 4_131_968
