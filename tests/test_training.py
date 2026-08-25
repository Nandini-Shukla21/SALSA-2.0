"""Tests for the training pipeline: objective, metrics and the trainer.

The properties that matter most here are the ones that would invalidate every
later result while still looking like a working training run: the trainer
touching ground truth, the teacher-forcing shift being off by one, the SALSA
tolerance metric being silently redefined, or a checkpoint that does not
actually restore a run.
"""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from salsa.data import LatticeCodec  # noqa: E402
from salsa.data.lwe import LWESample  # noqa: E402
from salsa.training import (  # noqa: E402
    BatchStream,
    Trainer,
    absolute_distance,
    chance_baselines,
    circular_distance,
    decode_generated_ids,
    evaluate,
    greedy_decode,
    perfect_accuracy,
    sequence_cross_entropy,
    shift_for_teacher_forcing,
    token_accuracy,
)
from salsa.training.metrics import ThroughputMeter  # noqa: E402
from salsa.utils.config import load_config  # noqa: E402
from salsa.utils.logging import RunContext  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
VOCAB = 85


def tiny_config(tmp_path: Path, **overrides):
    """Load the smoke config, redirected into a temporary directory."""
    settings = [f"experiment.output_root={tmp_path.as_posix()}"]
    settings += [f"{k}={v}" for k, v in overrides.items()]
    return load_config(CONFIG_DIR / "smoke_train.yaml", overrides=settings)


def tiny_trainer(tmp_path: Path, **overrides) -> Trainer:
    """Build a trainer on the smoke configuration."""
    config = tiny_config(tmp_path, **overrides)
    context = RunContext.create(config, run_id="unit", console=False)
    return Trainer(config, context=context)


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def test_teacher_forcing_shift_is_off_by_one() -> None:
    """Decoder input is target[:, :-1]; the scored target is target[:, 1:]."""
    target = torch.tensor([[1, 10, 20, 2], [1, 11, 21, 2]])
    decoder_input, scored = shift_for_teacher_forcing(target)
    assert decoder_input.shape == scored.shape == (2, 3)
    assert torch.equal(decoder_input, target[:, :-1])
    assert torch.equal(scored, target[:, 1:])
    assert decoder_input.shape[1] == target.shape[1] - 1


def test_teacher_forcing_rejects_degenerate_targets() -> None:
    """A single-token target cannot be teacher forced."""
    with pytest.raises(ValueError, match="at least two"):
        shift_for_teacher_forcing(torch.tensor([[1]]))
    with pytest.raises(ValueError, match="must be"):
        shift_for_teacher_forcing(torch.tensor([1, 2, 3]))


def test_cross_entropy_ignores_padding() -> None:
    """Padded positions must not contribute to the loss or the token count."""
    logits = torch.zeros(2, 3, VOCAB)
    targets = torch.tensor([[5, 6, 0], [7, 0, 0]])
    out = sequence_cross_entropy(logits, targets, ignore_index=0)
    assert out.num_tokens == 3
    assert out.num_sequences == 2
    assert torch.isfinite(out.loss)


def test_cross_entropy_matches_a_hand_computed_value() -> None:
    """Uniform logits give exactly log(vocab) nats."""
    logits = torch.zeros(4, 2, VOCAB)
    targets = torch.randint(1, VOCAB, (4, 2))
    out = sequence_cross_entropy(logits, targets, ignore_index=0)
    assert float(out.loss) == pytest.approx(np.log(VOCAB), abs=1e-5)


def test_loss_supports_exact_aggregate_averaging() -> None:
    """total = mean * tokens, so batches of different sizes average exactly."""
    torch.manual_seed(0)
    logits = torch.randn(3, 4, VOCAB)
    targets = torch.randint(1, VOCAB, (3, 4))
    out = sequence_cross_entropy(logits, targets, ignore_index=0)
    assert out.total == pytest.approx(float(out.loss) * out.num_tokens, rel=1e-6)

    summed = sequence_cross_entropy(logits, targets, ignore_index=0, reduction="sum")
    assert float(summed.loss) == pytest.approx(out.total, rel=1e-5)


def test_loss_rejects_shape_mismatches() -> None:
    """A misaligned target is an error, not a silent broadcast."""
    with pytest.raises(ValueError, match="must agree"):
        sequence_cross_entropy(torch.zeros(2, 3, VOCAB), torch.zeros(2, 4, dtype=torch.long))


def test_loss_module_never_mentions_the_secret() -> None:
    """The objective is defined on the public pair (a, b) alone."""
    source = (REPO_ROOT / "salsa" / "training" / "losses.py").read_text(encoding="utf-8")
    for forbidden in ("reveal_secret", "ground_truth", "LabeledSample", ".secret", ".error"):
        assert forbidden not in source, f"losses.py references {forbidden}"


# --------------------------------------------------------------------------- #
# Token-level metrics
# --------------------------------------------------------------------------- #
def test_token_and_perfect_accuracy_are_exact() -> None:
    """Hand-built logits give hand-checkable counts."""
    logits = torch.full((2, 3, VOCAB), -10.0)
    targets = torch.tensor([[5, 6, 7], [5, 6, 7]])
    logits[0, 0, 5] = logits[0, 1, 6] = logits[0, 2, 7] = 10.0   # all correct
    logits[1, 0, 5] = logits[1, 1, 6] = 10.0                      # one wrong
    logits[1, 2, 8] = 10.0

    assert token_accuracy(logits, targets) == (5, 6)
    assert perfect_accuracy(logits, targets) == (1, 2)


def test_token_metrics_skip_padding() -> None:
    """Padded positions are excluded from both counts."""
    logits = torch.full((1, 3, VOCAB), -10.0)
    targets = torch.tensor([[5, 6, 0]])
    logits[0, 0, 5] = logits[0, 1, 6] = 10.0
    assert token_accuracy(logits, targets, ignore_index=0) == (2, 2)
    assert perfect_accuracy(logits, targets, ignore_index=0) == (1, 1)


# --------------------------------------------------------------------------- #
# Distances: SALSA metric vs the added diagnostic
# --------------------------------------------------------------------------- #
def test_salsa_distance_is_the_plain_absolute_difference() -> None:
    """acc_tau must use |a - b|, matching the released generators.py."""
    a = np.array([0, 10, 250, 125])
    b = np.array([250, 10, 0, 100])
    assert absolute_distance(a, b).tolist() == [250, 0, 250, 25]


def test_circular_distance_is_a_separate_additional_diagnostic() -> None:
    """The ring distance differs precisely at the wrap-around, and is smaller."""
    a = np.array([0, 10, 250, 125])
    b = np.array([250, 10, 0, 100])
    ring = circular_distance(a, b, 251)
    assert ring.tolist() == [1, 0, 1, 25]
    assert np.all(ring <= absolute_distance(a, b))


def test_circular_metrics_are_labelled_as_additions() -> None:
    """The diagnostic must never be mistaken for the paper's metric."""
    source = (REPO_ROOT / "salsa" / "training" / "metrics.py").read_text(encoding="utf-8")
    assert "NEW ADDITIONAL DIAGNOSTIC" in source
    codec = LatticeCodec(n=8, q=251, base=81, separator=False, digit_order="lsb_first")
    keys = evaluate.__doc__ and chance_baselines(codec, 0.1)
    assert "chance_acc_tau_circular_DIAGNOSTIC" in keys
    assert "chance_acc_tau" in keys


# --------------------------------------------------------------------------- #
# Chance baselines
# --------------------------------------------------------------------------- #
def test_chance_baselines_are_split_by_level() -> None:
    """Token-level and integer-level chance are different numbers."""
    codec = LatticeCodec(n=30, q=251, base=81, separator=False, digit_order="lsb_first")
    base = chance_baselines(codec, 0.1)

    assert base["chance_token_accuracy"] == pytest.approx(0.4462, abs=1e-3)
    assert base["chance_perfect_accuracy"] == pytest.approx(0.00514, abs=1e-4)
    assert base["chance_exact_accuracy"] == pytest.approx(1 / 251, abs=1e-6)
    assert base["chance_acc_tau"] == pytest.approx(0.1929, abs=1e-3)
    assert base["uniform_loss_nats"] == pytest.approx(np.log(85), abs=1e-6)

    # The token-level baseline is far above the integer-level one; conflating
    # them would make a useless model look successful.
    assert base["chance_token_accuracy"] > 20 * base["chance_exact_accuracy"]


def test_high_digit_baseline_reflects_the_representation() -> None:
    """b // 81 is 0, 1 or 2 for 96.8% of Z_251, hence a ~32% marginal."""
    codec = LatticeCodec(n=8, q=251, base=81, separator=False, digit_order="lsb_first")
    digits = chance_baselines(codec, 0.1)["chance_digit_accuracy"]
    # Low digit is b mod 81: residues 0-7 occur 4 times over Z_251, 8-80 occur 3
    # times, so the best marginal is 4/251 -- nearly uniform.
    assert digits[0] == pytest.approx(4 / 251, abs=1e-5)
    # High digit is b // 81, which is 0, 1 or 2 for 243 of 251 values.
    assert digits[1] == pytest.approx(81 / 251, abs=1e-3)
    assert digits[1] > 20 * digits[0], "the high digit is the one that inflates chance"


def test_chance_acc_tau_is_computed_not_assumed() -> None:
    """The baseline is the exact expectation over uniform guesses."""
    codec = LatticeCodec(n=8, q=251, base=81, separator=False, digit_order="lsb_first")
    values = np.arange(251)
    grid = np.abs(values[:, None] - values[None, :])
    assert chance_baselines(codec, 0.1)["chance_acc_tau"] == pytest.approx(
        float((grid <= 25.1).mean()), abs=1e-9
    )


# --------------------------------------------------------------------------- #
# Greedy decoding
# --------------------------------------------------------------------------- #
def test_greedy_decode_shapes_and_prefix(tmp_path: Path) -> None:
    """Generation starts from <bos> and produces the requested length."""
    trainer = tiny_trainer(tmp_path)
    src, _ = trainer.train_stream.batch(0)
    bos = trainer.codec.vocabulary.bos_id
    generated = greedy_decode(trainer.model, src, bos, trainer.codec.output_length - 1)
    assert generated.shape == (src.shape[0], trainer.codec.output_length)
    assert torch.all(generated[:, 0] == bos)
    trainer.context.close()


def test_greedy_decode_leaves_training_mode_untouched(tmp_path: Path) -> None:
    """A metric must not silently switch the model out of training mode."""
    trainer = tiny_trainer(tmp_path)
    src, _ = trainer.train_stream.batch(0)
    trainer.model.train()
    greedy_decode(trainer.model, src, trainer.codec.vocabulary.bos_id, 3)
    assert trainer.model.training is True
    trainer.context.close()


def test_decode_failures_are_counted_not_hidden() -> None:
    """An unreadable generation is a distinct outcome from a wrong number."""
    codec = LatticeCodec(n=4, q=251, base=81, separator=False, digit_order="lsb_first")
    good = np.array(codec.encode_output_ids(100))
    broken = good.copy()
    broken[1] = codec.vocabulary.sep_id          # a separator where a digit belongs
    values, valid = decode_generated_ids(codec, np.stack([good, broken]))
    assert valid.tolist() == [True, False]
    assert values[0] == 100
    assert values[1] == -1


# --------------------------------------------------------------------------- #
# Throughput
# --------------------------------------------------------------------------- #
def test_throughput_meter_accumulates() -> None:
    """Rates are totals over totals, not an average of averages."""
    meter = ThroughputMeter()
    meter.update(100, 300, 2.0)
    meter.update(100, 300, 8.0)
    assert meter.samples_per_second == pytest.approx(20.0)
    assert meter.tokens_per_second == pytest.approx(60.0)
    assert meter.to_dict()["compute_seconds"] == pytest.approx(10.0)
    # The meter must not use the wall-clock key: it once overwrote the run's
    # true elapsed time in every metrics row.
    assert "elapsed_seconds" not in meter.to_dict()


# --------------------------------------------------------------------------- #
# Data streaming and secret isolation
# --------------------------------------------------------------------------- #
def test_stream_yields_only_public_samples(tmp_path: Path) -> None:
    """The trainer's data source must expose no secret and no error."""
    config = tiny_config(tmp_path)
    codec = LatticeCodec.from_config(config)
    stream = BatchStream(config, "train", codec, batch_size=8)

    sample = stream.problem.batch(0, 8, labeled=False)
    assert isinstance(sample, LWESample)
    assert not hasattr(sample, "secret")
    assert not hasattr(sample, "error")
    assert not hasattr(sample, "ground_truth")

    src, tgt = stream.batch(0)
    assert src.shape == (8, codec.input_length)
    assert tgt.shape == (8, codec.output_length)
    assert src.dtype == torch.long and tgt.dtype == torch.long


def test_trainer_module_never_reads_ground_truth() -> None:
    """Static check: the training loop cannot reach the secret."""
    source = (REPO_ROOT / "salsa" / "training" / "trainer.py").read_text(encoding="utf-8")
    assert "reveal_secret" not in source
    assert "labeled=True" not in source
    assert "labeled=False" in source


def test_stream_batches_are_reproducible(tmp_path: Path) -> None:
    """Batch i is the same data however it is reached."""
    config = tiny_config(tmp_path)
    codec = LatticeCodec.from_config(config)
    first = BatchStream(config, "train", codec, batch_size=8)
    second = BatchStream(config, "train", codec, batch_size=8)
    assert torch.equal(first.batch(3)[0], second.batch(3)[0])
    assert not torch.equal(first.batch(3)[0], first.batch(4)[0])


def test_train_and_valid_streams_differ(tmp_path: Path) -> None:
    """Validation data must not be training data."""
    config = tiny_config(tmp_path)
    codec = LatticeCodec.from_config(config)
    train = BatchStream(config, "train", codec, batch_size=8)
    valid = BatchStream(config, "valid", codec, batch_size=8)
    assert not torch.equal(train.batch(0)[0], valid.batch(0)[0])
    # ...but they attack the same secret.
    assert np.array_equal(
        train.problem.reveal_secret(), valid.problem.reveal_secret()
    )


def test_validation_set_is_fixed_across_calls(tmp_path: Path) -> None:
    """A moving validation set would measure data variation, not learning."""
    trainer = tiny_trainer(tmp_path)
    first = trainer.valid_batches
    second = trainer.valid_batches
    assert first is second
    assert torch.equal(first[0][0], second[0][0])
    trainer.context.close()


# --------------------------------------------------------------------------- #
# Alignment contract
# --------------------------------------------------------------------------- #
def test_alignment_verification_passes_and_reports_shapes(tmp_path: Path) -> None:
    """The data/model contract is asserted before any gradient is taken."""
    trainer = tiny_trainer(tmp_path)
    shapes = trainer.verify_alignment()
    codec = trainer.codec

    assert shapes["src_ids"][0] == shapes["tgt_ids"][0]
    assert shapes["decoder_input"][1] == shapes["tgt_ids"][1] - 1
    assert shapes["logits"][:2] == shapes["target"]
    assert shapes["vocab_size"] == codec.vocabulary.size
    assert shapes["pad_id"] == trainer.model.spec.pad_id == 0
    trainer.context.close()


def test_pilot_configs_use_vocabulary_85() -> None:
    """The real experiments run on the 85-token SALSA 2.0 vocabulary."""
    for name in ("experiment_4_13m_n30_r", "experiment_4_25m_control_n30_r"):
        config = load_config(CONFIG_DIR / f"{name}.yaml")
        assert LatticeCodec.from_config(config).vocabulary.size == 85


# --------------------------------------------------------------------------- #
# Training steps
# --------------------------------------------------------------------------- #
def test_single_training_step_updates_the_model(tmp_path: Path) -> None:
    """One step changes weights and returns finite statistics."""
    trainer = tiny_trainer(tmp_path)
    before = trainer.model.encoder_layers[0].ffn.w1.weight.detach().clone()
    stats = trainer.train_step(trainer.train_stream.batch(0))

    assert np.isfinite(stats["loss"]) and stats["loss"] > 0
    assert np.isfinite(stats["grad_norm"])
    assert stats["lr"] > 0
    assert not torch.equal(before, trainer.model.encoder_layers[0].ffn.w1.weight)
    assert trainer.state.global_step == 1
    assert trainer.state.samples_seen == trainer.plan.batch_size
    assert trainer.state.tokens_seen > 0
    trainer.context.close()


def test_loss_decreases_over_several_steps(tmp_path: Path) -> None:
    """A functional check that optimisation is wired up, not a research result."""
    trainer = tiny_trainer(tmp_path, **{"training.lr": 0.003})
    losses = [trainer.train_step(trainer.train_stream.batch(i))["loss"] for i in range(30)]
    assert np.mean(losses[-5:]) < np.mean(losses[:5]), (
        f"loss did not fall: {np.mean(losses[:5]):.4f} -> {np.mean(losses[-5:]):.4f}"
    )
    trainer.context.close()


def test_gradient_clipping_is_configurable(tmp_path: Path) -> None:
    """grad_clip=0 disables clipping; a positive value reports the norm."""
    clipped = tiny_trainer(tmp_path / "a", **{"training.grad_clip": 1.0})
    stats = clipped.train_step(clipped.train_stream.batch(0))
    assert np.isfinite(stats["grad_norm"])
    clipped.context.close()

    unclipped = tiny_trainer(tmp_path / "b", **{"training.grad_clip": 0})
    stats = unclipped.train_step(unclipped.train_stream.batch(0))
    assert np.isnan(stats["grad_norm"])
    unclipped.context.close()


@pytest.mark.parametrize("schedule", ["cosine", "inverse_sqrt", "constant"])
def test_learning_rate_schedules_are_configurable(tmp_path: Path, schedule: str) -> None:
    """Each schedule warms up and then behaves as named."""
    trainer = tiny_trainer(tmp_path / schedule, **{"training.lr_schedule": schedule})
    rates = []
    for index in range(12):
        rates.append(trainer.train_step(trainer.train_stream.batch(index))["lr"])
    assert rates[0] < rates[3], "learning rate must warm up"
    assert all(rate > 0 for rate in rates)
    trainer.context.close()


def test_budget_ceiling_is_enforced_before_training(tmp_path: Path) -> None:
    """A model above its declared ceiling must not be trainable at all."""
    with pytest.raises(ValueError, match="FAILED BUDGET"):
        tiny_trainer(tmp_path, **{"model.max_parameters": 1000})


# --------------------------------------------------------------------------- #
# Validation and checkpoints
# --------------------------------------------------------------------------- #
def test_validation_produces_every_required_metric(tmp_path: Path) -> None:
    """Both metric levels are reported, and separately."""
    trainer = tiny_trainer(tmp_path)
    result = trainer.validate()
    row = result.to_dict("valid")

    for key in (
        "valid_loss",
        "valid_token_accuracy",
        "valid_perfect_accuracy",
        "valid_greedy_token_accuracy",
        "valid_exact_accuracy",
        "valid_acc_tau",
        "valid_tolerance",
        "valid_mean_distance",
        "valid_median_distance",
        "valid_percentiles",
        "valid_decode_failure_rate",
        "valid_acc_tau_circular_DIAGNOSTIC",
    ):
        assert key in row, f"missing metric {key}"

    assert 0.0 <= result.token_accuracy <= 1.0
    assert 0.0 <= result.acc_tau <= 1.0
    assert result.acc_tau_circular >= result.acc_tau - 1e-9, (
        "the ring distance can only ever be smaller, so its accuracy is >= SALSA's"
    )
    assert len(result.percentiles) == 10
    assert result.percentiles[-1] == pytest.approx(1.0, abs=1e-9)
    assert result.num_sequences > 0
    trainer.context.close()


def test_checkpoint_round_trip_restores_the_run(tmp_path: Path) -> None:
    """Save, perturb, reload: weights, optimiser and progress all come back."""
    trainer = tiny_trainer(tmp_path)
    for index in range(3):
        trainer.train_step(trainer.train_stream.batch(index))
    saved_state = trainer.state.to_dict()
    reference = trainer.model.encoder_layers[0].ffn.w1.weight.detach().clone()
    path = trainer.save_checkpoint("last")
    assert path.is_file()

    for index in range(3, 8):
        trainer.train_step(trainer.train_stream.batch(index))
    assert not torch.equal(reference, trainer.model.encoder_layers[0].ffn.w1.weight)

    restored = trainer.load_checkpoint(path)
    assert torch.equal(reference, trainer.model.encoder_layers[0].ffn.w1.weight)
    assert restored.to_dict() == saved_state
    assert trainer.state.global_step == 3
    trainer.context.close()


def test_checkpoint_contains_the_full_run_state(tmp_path: Path) -> None:
    """Optimiser, scheduler, step, samples seen and fingerprint are all saved."""
    trainer = tiny_trainer(tmp_path)
    trainer.train_step(trainer.train_stream.batch(0))
    payload = torch.load(trainer.save_checkpoint("last"), weights_only=False)

    for key in ("model", "optimizer", "scheduler", "state", "plan", "spec",
                "config", "config_fingerprint", "parameter_count", "rng"):
        assert key in payload, f"checkpoint missing {key}"
    assert payload["config_fingerprint"] == trainer.config.fingerprint()
    assert payload["parameter_count"] == trainer.parameters.trainable
    assert payload["state"]["samples_seen"] == trainer.plan.batch_size
    trainer.context.close()


def test_checkpoint_refuses_a_changed_configuration(tmp_path: Path) -> None:
    """Resuming into a different experiment would silently corrupt the record."""
    trainer = tiny_trainer(tmp_path)
    path = trainer.save_checkpoint("last")
    trainer.context.close()

    other = tiny_trainer(tmp_path / "other", **{"experiment.seed": 7})
    with pytest.raises(ValueError, match="fingerprint"):
        other.load_checkpoint(path)
    other.load_checkpoint(path, strict=False)  # explicit opt-out still works
    other.context.close()


def test_missing_checkpoint_raises(tmp_path: Path) -> None:
    """A missing file is reported, not silently ignored."""
    trainer = tiny_trainer(tmp_path)
    with pytest.raises(FileNotFoundError):
        trainer.load_checkpoint(tmp_path / "nope.pt")
    assert trainer.maybe_resume() is False
    trainer.context.close()


# --------------------------------------------------------------------------- #
# Full loop
# --------------------------------------------------------------------------- #
def test_short_training_run_end_to_end(tmp_path: Path) -> None:
    """The smoke configuration runs to completion and records everything."""
    trainer = tiny_trainer(tmp_path)
    state = trainer.train()

    assert state.global_step > 0
    assert state.samples_seen == state.global_step * trainer.plan.batch_size
    assert state.samples_seen <= trainer.plan.sample_budget
    assert state.stop_reason

    run_dir = trainer.context.run_dir
    assert (run_dir / "metrics.jsonl").is_file()
    assert (run_dir / "metrics.csv").is_file()
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "checkpoints" / "last.pt").is_file()
    assert (run_dir / "checkpoints" / "best.pt").is_file()
    assert (run_dir / "artifacts" / "summary.json").is_file()
    assert (run_dir / "artifacts" / "run_header.json").is_file()

    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    for key in ("train_loss", "valid_loss", "valid_acc_tau", "samples_seen",
                "tokens_seen", "samples_per_second", "tokens_per_second",
                "elapsed_seconds", "cpu_rss_mb", "parameter_count",
                "model_name", "config_fingerprint", "seed"):
        assert key in rows[0], f"metrics row missing {key}"
    assert rows[0]["config_fingerprint"] == trainer.config.fingerprint()
    assert rows[0]["parameter_count"] == trainer.parameters.trainable

    summary = json.loads((run_dir / "artifacts" / "summary.json").read_text("utf-8"))
    assert summary["parameter_count"] == trainer.parameters.trainable
    assert "chance_baselines" in summary
    assert summary["config_fingerprint"] == trainer.config.fingerprint()
    trainer.context.close()


def test_run_header_records_the_chance_baselines(tmp_path: Path) -> None:
    """Baselines are written once into the run metadata, not re-derived later."""
    trainer = tiny_trainer(tmp_path)
    trainer.log_run_header()
    header = json.loads(
        (trainer.context.run_dir / "artifacts" / "run_header.json").read_text("utf-8")
    )
    assert "chance_baselines" in header
    assert "chance_acc_tau" in header["chance_baselines"]
    assert header["parameters"]["trainable_parameters"] == trainer.parameters.trainable
    assert header["config_fingerprint"] == trainer.config.fingerprint()
    trainer.context.close()


def test_early_stopping_is_configurable(tmp_path: Path) -> None:
    """Patience 1 on a metric that cannot improve stops the run early."""
    trainer = tiny_trainer(
        tmp_path,
        **{
            "training.early_stopping_patience": 1,
            "training.monitor_metric": "epoch",
            "training.monitor_mode": "min",
            "training.max_epochs": 8,
        },
    )
    state = trainer.train()
    assert "early stopping" in state.stop_reason
    assert state.epoch < 8
    trainer.context.close()


def test_sample_budget_caps_the_run(tmp_path: Path) -> None:
    """The YAML sample budget, not the epoch count, bounds the data consumed."""
    trainer = tiny_trainer(
        tmp_path, **{"lwe.num_train_samples": 64, "training.max_epochs": 100}
    )
    assert trainer.plan.total_steps == 64 // trainer.plan.batch_size
    state = trainer.train()
    assert state.samples_seen <= 64


def test_no_cuda_specific_code_in_the_training_package() -> None:
    """Training stays CPU-first and portable.

    ``torch.cuda.is_available()`` is explicitly allowed: seeding an accelerator
    *if one happens to exist* is correct CPU-first behaviour.  What is forbidden
    is code that assumes a CUDA device.
    """
    for path in sorted((REPO_ROOT / "salsa" / "training").glob("*.py")):
        source = path.read_text(encoding="utf-8").lower()
        for marker in (".cuda(", "device='cuda'", 'device="cuda"', "cuda:0"):
            assert marker not in source, f"{path.name} contains {marker!r}"
        for line in source.splitlines():
            if "torch.cuda." in line:
                assert "is_available" in line or "manual_seed" in line, (
                    f"{path.name} uses CUDA unguarded: {line.strip()!r}"
                )
