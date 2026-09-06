"""Tests for direct secret recovery (SALSA Algorithm 1).

The property that matters most is negative: the recovery code must be unable to
see the secret.  An attack that quietly consults ground truth would "succeed"
on every instance and prove nothing, so that isolation is asserted both
structurally (no such argument exists) and textually (the source never names
those fields).
"""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from salsa.data import LatticeCodec  # noqa: E402
from salsa.models import SalsaTransformer, build_model  # noqa: E402
from salsa.recovery import (
    SELECTION_RULES,
    BINARIZATION_METHODS,
    DirectRecovery,
    build_probe_matrix,
    build_probes,
    probe_separation,
    ring_distance,
)
from salsa.recovery.direct import CoordinateProbe  # noqa: E402
from salsa.utils.config import load_config  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
Q = 251


def small_codec(n: int = 12) -> LatticeCodec:
    """A codec matching the trained n=12 experiment."""
    return LatticeCodec(n=n, q=Q, base=81, separator=False, digit_order="lsb_first")


def small_model(codec: LatticeCodec) -> SalsaTransformer:
    """An untrained model with the right shape for the codec."""
    from salsa.models import ModelSpec

    return SalsaTransformer(
        ModelSpec(vocab_size=codec.vocabulary.size, encoder_dim=64, decoder_dim=32,
                  encoder_layers=1, decoder_layers=1, encoder_heads=2, decoder_heads=2,
                  encoder_loops=1, decoder_loops=1, gated=True)
    ).eval()


# --------------------------------------------------------------------------- #
# Probe construction
# --------------------------------------------------------------------------- #
def test_probe_matrix_is_k_times_the_identity() -> None:
    """Row i must be K * e_i and nothing else."""
    matrix = build_probe_matrix(5, 125, Q)
    assert matrix.shape == (5, 5)
    assert matrix.dtype == np.int64
    for i in range(5):
        assert matrix[i, i] == 125
        assert matrix[i, [j for j in range(5) if j != i]].tolist() == [0, 0, 0, 0]
    assert np.array_equal(matrix, np.eye(5, dtype=np.int64) * 125)


def test_probe_values_are_reduced_mod_q() -> None:
    """Only K mod q is meaningful, and it is what gets encoded."""
    assert build_probe_matrix(3, Q, Q)[0, 0] == 0
    assert build_probe_matrix(3, Q + 7, Q)[0, 0] == 7
    assert build_probe_matrix(3, 239145, Q)[0, 0] == 239145 % Q
    matrix = build_probe_matrix(4, 300, Q)
    assert matrix.min() >= 0 and matrix.max() < Q


def test_probes_cover_every_coordinate_exactly_once() -> None:
    """n probes, one per coordinate, in order."""
    probes = build_probes(6, 94, Q)
    assert len(probes) == 6
    assert [p.coordinate for p in probes] == list(range(6))
    for probe in probes:
        assert isinstance(probe, CoordinateProbe)
        assert probe.reduced_K == 94
        assert int(probe.vector.sum()) == 94
        assert probe.vector[probe.coordinate] == 94


def test_probe_construction_is_validated() -> None:
    """Impossible dimensions are rejected."""
    with pytest.raises(ValueError):
        build_probe_matrix(0, 125, Q)
    with pytest.raises(ValueError):
        build_probe_matrix(4, 125, 1)


# --------------------------------------------------------------------------- #
# The mathematics the attack relies on
# --------------------------------------------------------------------------- #
def test_probe_realises_the_intended_inner_product() -> None:
    """a . s = K * s_i mod q, which is what makes the attack possible."""
    n, K = 12, 125
    secret = np.zeros(n, dtype=np.int64)
    secret[[3, 8]] = 1                      # a local secret, used only as arithmetic
    matrix = build_probe_matrix(n, K, Q)
    products = (matrix @ secret) % Q
    for i in range(n):
        assert products[i] == (K * secret[i]) % Q
    assert products[3] == K and products[8] == K
    assert products[0] == 0


def test_separation_is_maximal_at_half_q_and_vanishes_at_the_ends() -> None:
    """A 'large K' near q is a BAD probe, not a good one."""
    assert probe_separation(125, Q) == 125
    assert probe_separation(126, Q) == 125
    assert probe_separation(1, Q) == 1
    assert probe_separation(250, Q) == 1
    assert probe_separation(Q, Q) == 0
    assert probe_separation(239145, Q) == probe_separation(239145 % Q, Q)
    assert max(probe_separation(k, Q) for k in range(Q)) == Q // 2


def test_ring_distance_wraps() -> None:
    """0 and q-1 are neighbours on Z_q."""
    assert ring_distance(np.array([250]), 0, Q).tolist() == [1]
    assert ring_distance(np.array([0]), 250, Q).tolist() == [1]
    assert ring_distance(np.array([125]), 0, Q).tolist() == [125]
    assert np.all(ring_distance(np.arange(Q), 0, Q) <= Q // 2)


# --------------------------------------------------------------------------- #
# Decision logic
# --------------------------------------------------------------------------- #
def test_anchor_rule_assigns_bits_by_proximity() -> None:
    """A prediction near K means 1; near 0 means 0."""
    codec = small_codec(4)
    recovery = DirectRecovery(small_model(codec), codec, method="anchor")
    predictions = np.array([0, 125, 5, 120])
    ok = np.ones(4, dtype=bool)
    bits, scores = recovery._decide(predictions, ok, 125)
    assert bits.tolist() == [0, 1, 0, 1]
    assert scores[0] < 0 and scores[1] > 0
    assert abs(scores[1]) == pytest.approx(1.0)


def test_undecodable_predictions_give_no_evidence() -> None:
    """An unreadable answer must not vote for either value."""
    codec = small_codec(3)
    recovery = DirectRecovery(small_model(codec), codec, method="anchor")
    bits, scores = recovery._decide(
        np.array([-1, 125, 0]), np.array([False, True, True]), 125
    )
    assert scores[0] == 0.0
    assert bits[0] == 0
    assert bits.tolist() == [0, 1, 0]


@pytest.mark.parametrize("method", BINARIZATION_METHODS)
def test_every_method_produces_binary_output(method: str) -> None:
    """All decision rules return bits and bounded scores."""
    codec = small_codec(6)
    recovery = DirectRecovery(small_model(codec), codec, method=method)
    predictions = np.array([0, 125, 3, 122, 1, 124])
    bits, scores = recovery._decide(predictions, np.ones(6, dtype=bool), 125)
    assert set(np.unique(bits).tolist()).issubset({0, 1})
    assert np.all(np.abs(scores) <= 1.0 + 1e-9)


def test_unknown_method_is_rejected() -> None:
    """A typo in the decision rule is an error, not a silent default."""
    codec = small_codec(4)
    with pytest.raises(ValueError, match="method must be"):
        DirectRecovery(small_model(codec), codec, method="magic")


def test_decision_polarity_does_not_come_from_a_secret() -> None:
    """The anchor rule is symmetric in K, with no ground-truth tiebreak.

    The released implementation resolves polarity by testing both a bit vector
    and its inverse against the true secret. Ours must give the same answer
    whatever the secret happens to be, because it never sees one.
    """
    codec = small_codec(4)
    recovery = DirectRecovery(small_model(codec), codec, method="anchor")
    predictions = np.array([2, 123, 4, 121])
    bits_a, _ = recovery._decide(predictions, np.ones(4, dtype=bool), 125)
    bits_b, _ = recovery._decide(predictions, np.ones(4, dtype=bool), 125)
    assert bits_a.tolist() == bits_b.tolist() == [0, 1, 0, 1]


# --------------------------------------------------------------------------- #
# Inference plumbing
# --------------------------------------------------------------------------- #
def test_recovery_runs_end_to_end_on_an_untrained_model() -> None:
    """The mechanics work regardless of whether the model has learned."""
    codec = small_codec(12)
    recovery = DirectRecovery(small_model(codec), codec)
    result = recovery.recover_for_k(125)

    assert result.candidate.shape == (12,)
    assert set(np.unique(result.candidate).tolist()).issubset({0, 1})
    assert len(result.outcomes) == 12
    assert result.reduced_K == 125
    assert result.separation == 125
    assert 0.0 <= result.decode_failure_rate <= 1.0
    for index, outcome in enumerate(result.outcomes):
        assert outcome.coordinate == index
        assert outcome.distance_to_zero >= 0
        assert outcome.distance_to_K >= 0


def test_predict_rejects_wrong_probe_width() -> None:
    """A probe of the wrong dimension cannot be fed to the model."""
    codec = small_codec(12)
    recovery = DirectRecovery(small_model(codec), codec)
    with pytest.raises(ValueError, match=r"shape \(m, 12\)"):
        recovery.predict(np.zeros((3, 7), dtype=np.int64))


def test_probe_encoding_matches_the_training_encoding() -> None:
    """Probes are encoded exactly like ordinary rows: same length, same vocab."""
    codec = small_codec(12)
    matrix = build_probe_matrix(12, 125, Q)
    src_ids, targets = codec.encode_batch(matrix)
    assert src_ids.shape == (12, codec.input_length) == (12, 26)
    assert targets is None                      # no b is supplied or needed
    assert src_ids.max() < codec.vocabulary.size == 85
    # ...and the encoding round-trips, so the model sees the probe we intended.
    assert np.array_equal(codec.decode_batch_inputs(src_ids), matrix)


def test_recovery_leaves_model_mode_untouched() -> None:
    """Running an attack must not silently switch the model to eval."""
    codec = small_codec(12)
    model = small_model(codec)
    model.train()
    DirectRecovery(model, codec).recover_for_k(125)
    assert model.training is True


# --------------------------------------------------------------------------- #
# Multiple K
# --------------------------------------------------------------------------- #
def test_sweep_returns_one_result_per_k() -> None:
    """Every requested K is probed and reported."""
    codec = small_codec(12)
    recovery = DirectRecovery(small_model(codec), codec)
    ks = [31, 94, 125]
    report = recovery.recover(ks)

    assert [r.K for r in report.per_k] == ks
    assert report.aggregate_candidate.shape == (12,)
    assert report.aggregate_scores.shape == (12,)
    assert report.selected_K in ks
    assert report.n == 12 and report.q == Q


def test_sweep_requires_at_least_one_k() -> None:
    """An empty sweep is an error."""
    codec = small_codec(4)
    with pytest.raises(ValueError, match="at least one K"):
        DirectRecovery(small_model(codec), codec).recover([])


def test_zero_separation_k_is_excluded_from_selection() -> None:
    """K = q carries no information, so it cannot be the chosen probe."""
    codec = small_codec(12)
    recovery = DirectRecovery(small_model(codec), codec)
    report = recovery.recover([Q, 125])
    assert probe_separation(Q, Q) == 0
    assert report.selected_K == 125


def test_selection_uses_only_predictions() -> None:
    """The chosen K maximises the mean margin, a secret-free quantity.

    Phase 27 repurposed ``selection_rule`` to name the rule in force and moved
    the descriptive text to ``selected_K_rule``; the assertion follows the
    documentation to its new field rather than being dropped.
    """
    codec = small_codec(12)
    report = DirectRecovery(small_model(codec), codec).recover([31, 94, 125])
    eligible = [r for r in report.per_k if r.separation >= report.min_separation]
    assert report.selected_K == max(eligible, key=lambda r: r.mean_margin).K
    payload = report.to_dict()
    assert "predictions only" in payload["selected_K_rule"]
    assert payload["selection_rule"] in SELECTION_RULES


# --------------------------------------------------------------------------- #
# Secret isolation
# --------------------------------------------------------------------------- #
def test_recovery_source_never_references_ground_truth() -> None:
    """Textual proof that the attack cannot reach the secret or the error."""
    for path in sorted((REPO_ROOT / "salsa" / "recovery").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        for forbidden in ("reveal_secret", "ground_truth", "LabeledSample",
                          "labeled=True", ".secret", ".error"):
            assert forbidden not in source, f"{path.name} references {forbidden}"


def test_recovery_api_has_no_parameter_that_could_carry_a_secret() -> None:
    """Structural proof: no constructor or method accepts ground truth."""
    import inspect

    for function in (DirectRecovery.__init__, DirectRecovery.recover,
                     DirectRecovery.recover_for_k, DirectRecovery.predict,
                     build_probe_matrix, build_probes):
        names = set(inspect.signature(function).parameters)
        assert not names & {"secret", "s", "error", "e", "ground_truth", "problem"}, (
            f"{function.__qualname__} exposes a channel for ground truth: {names}"
        )


def test_reports_carry_no_ground_truth() -> None:
    """Serialised recovery output must be free of any secret field."""
    codec = small_codec(12)
    payload = DirectRecovery(small_model(codec), codec).recover([125]).to_dict()
    text = str(payload)
    for forbidden in ("true_secret", "ground_truth", "secret_bits"):
        assert forbidden not in text
    assert "candidate" in payload["per_k"][0]


def test_recovery_needs_no_target_sequence() -> None:
    """The decoder is driven from <bos>, so b is never required as input.

    The released evaluator builds ``specialB`` from the true secret to satisfy
    its batching. Ours never constructs it.
    """
    codec = small_codec(12)
    source = (REPO_ROOT / "salsa" / "recovery" / "direct.py").read_text(encoding="utf-8")
    assert "specialB" not in source
    src_ids, targets = codec.encode_batch(build_probe_matrix(12, 125, Q))
    assert targets is None


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def test_recovery_config_matches_the_trained_experiment() -> None:
    """The recovery config inherits the exact instance that was trained."""
    trained = load_config(CONFIG_DIR / "control_a_n12_h2.yaml")
    recovery = load_config(CONFIG_DIR / "recovery_n12_h2.yaml")
    recovery.validate()

    assert recovery.lwe == trained.lwe
    assert recovery.encoding == trained.encoding
    assert recovery.model == trained.model
    assert recovery.experiment.seed == trained.experiment.seed
    assert recovery.recovery != trained.recovery      # only this block differs


def test_configured_k_sweep_includes_controls_and_the_optimum() -> None:
    """The sweep spans the full range of probe quality, controls included."""
    config = load_config(CONFIG_DIR / "recovery_n12_h2.yaml")
    ks = config.recovery.direct_k_values
    separations = [probe_separation(k, config.lwe.q) for k in ks]

    assert len(ks) == 10
    assert max(separations) == config.lwe.q // 2      # a near-optimal probe
    assert min(separations) == 1                       # negative controls present
    assert config.recovery.enable_distinguisher is False


def test_model_from_config_accepts_the_probes() -> None:
    """The trained architecture consumes probes without any shape change."""
    config = load_config(CONFIG_DIR / "recovery_n12_h2.yaml")
    codec = LatticeCodec.from_config(config)
    model = build_model(config).eval()
    report = DirectRecovery(model, codec).recover([125])
    assert report.aggregate_candidate.shape == (config.lwe.n,)
    assert codec.input_length == 2 * config.lwe.n + 2
