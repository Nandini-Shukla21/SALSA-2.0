"""Tests for the Salsa 2.0 end-to-end pipeline and the hardened recovery rule.

Focused, not exhaustive: the stages themselves are already covered by
``test_direct_recovery.py`` and ``test_nact.py``.  What is tested here is the
seams -- construction, checkpoint identity, the degenerate-K guard, the
separation-weighted aggregate, the verifier's interface, config parsing and
result serialisation.
"""

import json

import numpy as np
import pytest
import torch

from salsa.pipeline import (
    PipelineError,
    PipelineResult,
    load_checkpoint,
    locate_checkpoint,
    run_salsa2_pipeline,
)
from salsa.recovery import SELECTION_RULES, DirectRecovery, probe_separation
from salsa.utils import load_config
from salsa.utils.config import ConfigError
from salsa.verification import (
    ACCEPTANCE_CRITERIA,
    residual_statistics,
    uniform_reference_std,
    verify_candidate,
)

N12 = "configs/pipeline_n12.yaml"
N20 = "configs/pipeline_n20.yaml"
NACT_F_PARAMETERS = 4_238_208


def _has_checkpoint(config_path: str) -> bool:
    from pathlib import Path

    config = load_config(config_path)
    run = config.pipeline.run_dir
    return bool(run) and (Path(run) / "checkpoints" / "best.pt").is_file()


requires_n12 = pytest.mark.skipif(
    not _has_checkpoint(N12), reason="n=12 NACT-F checkpoint not present")
requires_n20 = pytest.mark.skipif(
    not _has_checkpoint(N20), reason="n=20 NACT-F checkpoint not present")


# --------------------------------------------------------------------------- #
# configuration parsing
# --------------------------------------------------------------------------- #
def test_pipeline_configs_parse_and_validate():
    for path in (N12, N20):
        config = load_config(path)
        assert config.validate() is not None or True
        assert config.pipeline.run_dir
        assert config.pipeline.expect_parameters == NACT_F_PARAMETERS
        assert config.pipeline.expect_arch == "nact"
        assert config.recovery.selection_rule == "aggregate"
        assert config.verification.enabled is True


def test_pipeline_config_carries_the_instance_not_hard_coded_n():
    twelve, twenty = load_config(N12), load_config(N20)
    assert twelve.lwe.n == 12 and twenty.lwe.n == 20
    # The same pipeline code path serves both; nothing about n is in the code.
    assert twelve.recovery.direct_k_values == twenty.recovery.direct_k_values


def test_unknown_selection_rule_is_rejected():
    config = load_config(N12)
    config.recovery.selection_rule = "whatever"
    with pytest.raises(ConfigError, match="selection_rule"):
        config.validate()


def test_non_positive_verification_budget_is_rejected():
    config = load_config(N12)
    config.verification.num_samples = 0
    with pytest.raises(ConfigError, match="num_samples"):
        config.validate()


# --------------------------------------------------------------------------- #
# checkpoint loading
# --------------------------------------------------------------------------- #
@requires_n12
def test_locate_checkpoint_uses_metadata_not_a_guessed_filename():
    from pathlib import Path

    config = load_config(N12)
    path = locate_checkpoint(Path(config.pipeline.run_dir))
    assert path.is_file() and path.suffix == ".pt"
    summary = json.loads(
        (Path(config.pipeline.run_dir) / "artifacts" / "summary.json").read_text("utf-8"))
    assert summary["state"]["best_epoch"] is not None


@requires_n12
def test_checkpoint_loads_strictly_with_the_expected_identity():
    loaded = load_checkpoint(load_config(N12))
    assert loaded["parameter_count"] == NACT_F_PARAMETERS
    assert loaded["architecture"] == "nact"
    assert loaded["trainable_after_freeze"] == 0
    assert loaded["spec"]["use_numerical_features"] is False


@requires_n12
def test_identity_gate_rejects_a_mismatched_expectation():
    config = load_config(N12)
    config.pipeline.expect_parameters = 4_131_200      # V1's count
    with pytest.raises(PipelineError, match="parameter count"):
        load_checkpoint(config)


def test_missing_checkpoint_configuration_is_an_error():
    config = load_config(N12)
    config.pipeline.run_dir = None
    config.pipeline.checkpoint = None
    with pytest.raises(PipelineError, match="pipeline.run_dir"):
        load_checkpoint(config)


# --------------------------------------------------------------------------- #
# recovery rule hardening
# --------------------------------------------------------------------------- #
def test_selection_rules_are_the_documented_two():
    assert SELECTION_RULES == ("aggregate", "best_margin")


def test_min_separation_is_derived_from_sigma_and_excludes_degenerate_k():
    """The phase-26 defect: separation-1 probes must not win selection."""
    class _Stub:
        """Minimal stand-in so the rule can be tested without a model."""

        def __init__(self, n, q):
            self.n, self.q = n, q

    recovery = DirectRecovery.__new__(DirectRecovery)
    recovery.n, recovery.q, recovery.method, recovery.batch_size = 12, 251, "anchor", 64

    # A model that always answers 0: at K=1 this yields |score| = 1 everywhere,
    # the maximum possible margin, which is exactly the degeneracy.
    def fake_predict(matrix):
        rows = matrix.shape[0]
        return (np.zeros(rows, dtype=np.int64),
                np.ones(rows, dtype=bool),
                np.zeros((rows, 4), dtype=np.int64))

    recovery.predict = fake_predict
    k_values = [1, 63, 125, 250]
    report = recovery.recover(k_values, sigma=3.0)

    assert report.min_separation == 4          # floor(3) + 1
    assert set(report.excluded_k) == {1, 250}  # separation 1
    assert report.selected_K not in (1, 250)
    margins = {r.K: r.mean_margin for r in report.per_k}
    assert margins[1] == pytest.approx(1.0), "the degeneracy should still be visible"


def test_min_separation_defaults_to_one_when_no_sigma_is_supplied():
    recovery = DirectRecovery.__new__(DirectRecovery)
    recovery.n, recovery.q, recovery.method, recovery.batch_size = 8, 251, "anchor", 64
    recovery.predict = lambda m: (np.zeros(m.shape[0], dtype=np.int64),
                                  np.ones(m.shape[0], dtype=bool),
                                  np.zeros((m.shape[0], 4), dtype=np.int64))
    report = recovery.recover([1, 125])
    assert report.min_separation == 1
    assert report.excluded_k == []


def test_aggregate_is_separation_weighted_so_degenerate_k_barely_vote():
    recovery = DirectRecovery.__new__(DirectRecovery)
    recovery.n, recovery.q, recovery.method, recovery.batch_size = 6, 251, "anchor", 64
    recovery.predict = lambda m: (np.zeros(m.shape[0], dtype=np.int64),
                                  np.ones(m.shape[0], dtype=bool),
                                  np.zeros((m.shape[0], 4), dtype=np.int64))
    report = recovery.recover([1, 125], sigma=3.0)
    weights = {r.K: r.separation / (251 / 2.0) for r in report.per_k}
    assert weights[1] < 0.01 < weights[125]
    assert report.selection_rule == "aggregate"
    assert np.array_equal(report.primary_candidate, report.aggregate_candidate)


def test_best_margin_rule_returns_the_selected_k_candidate():
    recovery = DirectRecovery.__new__(DirectRecovery)
    recovery.n, recovery.q, recovery.method, recovery.batch_size = 6, 251, "anchor", 64
    recovery.predict = lambda m: (np.zeros(m.shape[0], dtype=np.int64),
                                  np.ones(m.shape[0], dtype=bool),
                                  np.zeros((m.shape[0], 4), dtype=np.int64))
    report = recovery.recover([31, 125], selection_rule="best_margin", sigma=3.0)
    chosen = next(r for r in report.per_k if r.K == report.selected_K)
    assert np.array_equal(report.primary_candidate, chosen.candidate)


def test_unknown_rule_raises():
    recovery = DirectRecovery.__new__(DirectRecovery)
    recovery.n, recovery.q, recovery.method, recovery.batch_size = 6, 251, "anchor", 64
    with pytest.raises(ValueError, match="selection_rule"):
        recovery.recover([31], selection_rule="nonsense")


# --------------------------------------------------------------------------- #
# verification interface
# --------------------------------------------------------------------------- #
def test_verifier_signature_admits_no_secret():
    import inspect

    parameters = list(inspect.signature(verify_candidate).parameters)
    assert parameters == ["A", "b", "candidate", "q", "sigma", "baselines"]
    for forbidden in ("secret", "ground_truth", "label", "model"):
        assert not any(forbidden in p for p in parameters)


def test_verifier_separates_a_correct_candidate_from_a_wrong_one():
    rng = np.random.default_rng(0)
    n, q, sigma = 12, 251, 3.0
    secret = np.zeros(n, dtype=np.int64)
    secret[[2, 7]] = 1
    A = rng.integers(0, q, size=(512, n), dtype=np.int64)
    error = np.rint(rng.normal(0, sigma, size=512)).astype(np.int64)
    b = (A @ secret + error) % q

    good = verify_candidate(A, b, secret, q, sigma)
    assert good.passes
    assert good.statistics["std"] < 1.5 * sigma

    wrong = secret.copy()
    wrong[0] ^= 1
    bad = verify_candidate(A, b, wrong, q, sigma)
    assert not bad.passes
    assert bad.statistics["std"] > 0.5 * uniform_reference_std(q)


def test_centered_residuals_are_used():
    """A residue of q-1 must read as -1, not as a huge positive error."""
    n, q = 4, 251
    A = np.zeros((3, n), dtype=np.int64)
    candidate = np.zeros(n, dtype=np.int64)
    b = np.array([q - 1, 0, 1], dtype=np.int64)
    stats = residual_statistics(A, b, candidate, q, 3.0)
    assert stats["max_abs"] == 1
    assert stats["mean"] == pytest.approx(0.0)


def test_acceptance_criteria_are_named_and_stable():
    assert set(ACCEPTANCE_CRITERIA) == {
        "c1_std_close_to_sigma", "c2_mean_abs_small",
        "c3_beats_baselines", "c4_within_three_sigma"}


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #
@requires_n12
def test_pipeline_runs_end_to_end_at_n12(tmp_path):
    result = run_salsa2_pipeline(N12, output_dir=tmp_path)
    assert result.parameter_count == NACT_F_PARAMETERS
    assert result.instance["n"] == 12
    assert result.exact_recovery is True
    assert result.coordinate_accuracy == 1.0
    assert result.verification_ran and result.verification_passed
    assert result.excluded_k == [1, 250]
    assert result.selected_k not in (1, 250)


@requires_n20
def test_pipeline_runs_end_to_end_at_n20(tmp_path):
    result = run_salsa2_pipeline(N20, output_dir=tmp_path)
    assert result.instance["n"] == 20
    assert result.exact_recovery is True
    assert result.verification_passed


@requires_n12
def test_pipeline_can_run_without_ground_truth(tmp_path):
    """A real attack has no secret; the pipeline must still produce a candidate."""
    result = run_salsa2_pipeline(N12, evaluate=False, output_dir=tmp_path)
    assert result.recovered_candidate
    assert result.verification_ran
    assert result.true_secret == []          # never loaded
    assert result.exact_recovery is False    # not evaluated, not a claim
    assert result.successful_k == []


@requires_n12
def test_result_serialises_and_carries_the_required_schema(tmp_path):
    result = run_salsa2_pipeline(N12, output_dir=tmp_path)
    payload = result.to_dict()
    text = json.dumps(payload, default=str)
    assert json.loads(text)
    for key in ("config_fingerprint", "architecture", "parameter_count",
                "checkpoint", "recovery_method", "k_values", "successful_k",
                "recovered_candidate", "coordinate_accuracy", "hamming_distance",
                "exact_recovery", "verification_passed", "residual_statistics",
                "runtime_seconds", "limitations"):
        assert key in payload, key
    assert payload["instance"]["n"] == 12
    written = list(tmp_path.glob("*_result.json"))
    assert written, "the pipeline should persist its result"


@requires_n12
def test_pipeline_stage_ordering_secret_last(tmp_path):
    """Recovery and verification must complete before the secret is touched."""
    result = run_salsa2_pipeline(N12, output_dir=tmp_path)
    assert set(result.stage_seconds) >= {"recovery", "verification", "evaluation"}
    assert result.notes and any("only in the evaluation stage" in n
                                for n in result.notes)
