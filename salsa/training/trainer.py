"""CPU-first training loop for the SALSA ``a -> b`` task.

What the trainer may see
------------------------
Training consumes **public LWE samples only**.  Batches are drawn with
``problem.batch(index, size, labeled=False)``, which returns an
:class:`~salsa.data.lwe.LWESample` carrying ``A`` and ``b`` and nothing else --
the secret and the error vector are not reachable from the objects the trainer
holds.  :meth:`Trainer.verify_alignment` asserts this at start-up, and a test
asserts it again.  The generator still knows the secret so that a later phase
can score an attack, but that knowledge never enters training.

Streaming
---------
No dataset is materialised.  Batch ``i`` is generated on demand from the seed
``derive_seed(data_seed, "batch", i)``, so it is reproducible on its own, a run
can resume mid-stream, and memory stays at one batch regardless of the sample
budget.

Sample budget
-------------
``lwe.num_train_samples`` is the authority: the trainer derives the step count
from it, capped by ``training.max_epochs``.  Setting a pilot budget in the YAML
therefore controls exactly how much data is generated, which matters because
sample count -- not epochs -- is the quantity an attacker pays for.
"""

import json
import math
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from ..data import LatticeCodec, build_problem
from ..data.lwe import LWESample
from ..models import (
    SalsaTransformer,
    build_model,
    count_trainable_parameters,
    parameter_report,
)
from ..utils.device import configure_threads, memory_usage_mb, resolve_device
from ..utils.logging import RunContext
from .losses import sequence_cross_entropy, shift_for_teacher_forcing
from .metrics import EvaluationResult, ThroughputMeter, chance_baselines, evaluate
from .seed import derive_seed, set_seed

__all__ = ["BatchStream", "Trainer", "TrainingPlan", "TrainingState"]


# --------------------------------------------------------------------------- #
# Data streaming
# --------------------------------------------------------------------------- #
class BatchStream:
    """Streams encoded, public ``(src_ids, tgt_ids)`` batches on demand.

    Example:
        >>> stream = BatchStream(config, "train", codec, batch_size=32)  # doctest: +SKIP
        >>> src, tgt = stream.batch(0)                                   # doctest: +SKIP
    """

    def __init__(
        self,
        config,
        split: str,
        codec: LatticeCodec,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> None:
        """Create a stream over one data split.

        Args:
            config: The experiment configuration.
            split: ``"train"``, ``"valid"`` or ``"test"``.  Splits share the
                secret but draw independent sample streams.
            codec: The codec turning integers into token ids.
            batch_size: Instances per batch.
            device: Device for the returned tensors (CPU by default).

        Raises:
            ValueError: If ``batch_size`` is not positive.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        self.config = config
        self.split = str(split)
        self.codec = codec
        self.batch_size = int(batch_size)
        self.device = device or torch.device("cpu")
        self.problem = build_problem(config, split=self.split)

    def batch(self, index: int) -> Tuple[Tensor, Tensor]:
        """Return batch ``index`` as ``(src_ids, tgt_ids)``.

        The sample requested is the **public** view: the returned object has no
        secret and no error attached.

        Args:
            index: Zero-based batch index.

        Returns:
            Two ``int64`` tensors of shape ``(batch, src_len)`` and
            ``(batch, tgt_len)``.
        """
        sample = self.problem.batch(index, self.batch_size, labeled=False)
        assert isinstance(sample, LWESample), "training must never receive ground truth"
        src_ids, tgt_ids = self.codec.encode_batch(sample.A, sample.b)
        return (
            torch.from_numpy(src_ids).to(self.device),
            torch.from_numpy(tgt_ids).to(self.device),
        )

    def iter_batches(self, count: int, start: int = 0) -> Iterator[Tuple[Tensor, Tensor]]:
        """Yield ``count`` consecutive batches starting at ``start``."""
        for offset in range(int(count)):
            yield self.batch(int(start) + offset)

    def fixed_batches(self, num_samples: int) -> List[Tuple[Tensor, Tensor]]:
        """Materialise a small, fixed evaluation set.

        Validation must be the same data at every evaluation point, otherwise
        the curve measures data variation rather than learning.  Only the
        validation split uses this, and only for a few thousand samples.

        Args:
            num_samples: Total instances in the evaluation set.

        Returns:
            A list of ``(src_ids, tgt_ids)`` batches.
        """
        count = max(1, math.ceil(int(num_samples) / self.batch_size))
        return [self.batch(index) for index in range(count)]

    def __repr__(self) -> str:
        """Describe the stream without revealing anything private."""
        return (
            f"BatchStream(split='{self.split}', batch_size={self.batch_size}, "
            f"n={self.config.lwe.n}, q={self.config.lwe.q})"
        )


# --------------------------------------------------------------------------- #
# Plan and state
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TrainingPlan:
    """The derived schedule for one run.

    Attributes:
        sample_budget: Distinct LWE instances the run may consume.
        batch_size: Training batch size.
        total_steps: Optimiser steps implied by the budget.
        steps_per_epoch: Steps between validation points.
        epochs: Number of epochs actually planned.
        warmup_steps: Linear warm-up length.
        valid_samples: Size of the fixed validation set.
    """

    sample_budget: int
    batch_size: int
    total_steps: int
    steps_per_epoch: int
    epochs: int
    warmup_steps: int
    valid_samples: int

    def to_dict(self) -> Dict[str, int]:
        """Return a serialisable view."""
        return asdict(self)


@dataclass
class TrainingState:
    """Mutable progress, saved into every checkpoint so a run can resume."""

    epoch: int = 0
    global_step: int = 0
    samples_seen: int = 0
    tokens_seen: int = 0
    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    epochs_without_improvement: int = 0
    elapsed_seconds: float = 0.0
    stop_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable view."""
        return asdict(self)


# --------------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------------- #
class Trainer:
    """Config-driven CPU trainer with checkpointing, resume and early stopping."""

    def __init__(
        self,
        config,
        context: Optional[RunContext] = None,
        model: Optional[SalsaTransformer] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Build the model, data streams, optimiser and schedule.

        Args:
            config: A validated :class:`~salsa.utils.config.Config`.
            context: An existing run context; one is created when omitted.
            model: An already-built model; one is built from the config when
                omitted.
            run_id: Run identifier used when creating the context.
        """
        self.config = config
        config.validate()

        self.seed_state = set_seed(config.experiment.seed, config.experiment.deterministic)
        self.device_spec = resolve_device(
            config.device.prefer,
            allow_gpu_fallback=config.device.allow_gpu_fallback,
            dtype=config.device.dtype,
        )
        self.threads = configure_threads(
            config.device.threads, config.device.interop_threads
        )
        self.device = self.device_spec.torch_device()

        self.context = context or RunContext.create(config, run_id=run_id)
        self.logger = self.context.logger

        self.codec = LatticeCodec.from_config(config)
        self.model = (model or build_model(config)).to(self.device)
        self.parameters = parameter_report(
            self.model, config.model.max_parameters, config.model.min_parameters
        )
        if not self.parameters.within_budget:
            raise ValueError(
                f"{self.model.name} is a FAILED BUDGET configuration: "
                f"{self.parameters.trainable:,} > {config.model.max_parameters:,}"
            )

        self.plan = self._build_plan()
        self.train_stream = BatchStream(
            config, "train", self.codec, self.plan.batch_size, self.device
        )
        self.valid_stream = BatchStream(
            config, "valid", self.codec, config.training.eval_batch_size, self.device
        )
        self._valid_batches: Optional[List[Tuple[Tensor, Tensor]]] = None

        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.state = TrainingState()
        self.throughput = ThroughputMeter()
        self.baselines = chance_baselines(self.codec, config.evaluation.tolerance)
        self._start_time = time.time()

    # -- construction helpers ---------------------------------------------- #
    def _build_plan(self) -> TrainingPlan:
        """Derive the step schedule from the configured sample budget."""
        training = self.config.training
        budget = int(self.config.lwe.num_train_samples)
        batch_size = int(training.batch_size)
        total_steps = max(1, math.ceil(budget / batch_size))
        steps_per_epoch = min(int(training.steps_per_epoch), total_steps)
        epochs = min(int(training.max_epochs), math.ceil(total_steps / steps_per_epoch))
        return TrainingPlan(
            sample_budget=budget,
            batch_size=batch_size,
            total_steps=min(total_steps, epochs * steps_per_epoch),
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            warmup_steps=min(int(training.warmup_steps), max(1, total_steps // 2)),
            valid_samples=int(self.config.lwe.num_valid_samples),
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build AdamW, excluding 1-D parameters from weight decay."""
        training = self.config.training
        decay, no_decay = [], []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            (no_decay if parameter.ndim < 2 else decay).append(parameter)
        groups = [
            {"params": decay, "weight_decay": float(training.weight_decay)},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        if training.optimizer in ("adamw", "adam"):
            return torch.optim.AdamW(
                groups,
                lr=float(training.lr),
                betas=(float(training.beta1), float(training.beta2)),
                eps=float(training.eps),
            )
        if training.optimizer == "sgd":
            return torch.optim.SGD(groups, lr=float(training.lr), momentum=0.9)
        raise ValueError(f"Unsupported optimizer '{training.optimizer}'.")

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Linear warm-up followed by the configured decay, floored at min_lr."""
        training = self.config.training
        warmup = max(0, int(self.plan.warmup_steps))
        total = max(1, int(self.plan.total_steps))
        floor = float(training.min_lr) / float(training.lr) if training.lr > 0 else 0.0
        schedule = training.lr_schedule

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return float(step + 1) / float(warmup)
            if schedule == "constant":
                return 1.0
            if schedule == "inverse_sqrt":
                return max(floor, math.sqrt(max(1, warmup) / max(1, step)))
            progress = (step - warmup) / max(1, total - warmup)
            progress = min(1.0, max(0.0, progress))
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    @property
    def valid_batches(self) -> List[Tuple[Tensor, Tensor]]:
        """The fixed validation set, built once and reused."""
        if self._valid_batches is None:
            self._valid_batches = self.valid_stream.fixed_batches(self.plan.valid_samples)
        return self._valid_batches

    # -- alignment ---------------------------------------------------------- #
    def verify_alignment(self) -> Dict[str, Any]:
        """Assert the data/model contract before a single gradient is taken.

        Checks, in order:
        the trainer receives no ground truth; batch dimensions agree; the
        decoder input is one token shorter than the target; the logits match the
        target in batch, length and vocabulary; and the padding id is consistent
        between the codec and the model.

        Returns:
            A dictionary of the verified shapes, for the run log.

        Raises:
            AssertionError: If any part of the contract is violated.
        """
        src_ids, tgt_ids = self.train_stream.batch(0)
        sample = self.train_stream.problem.batch(0, self.plan.batch_size, labeled=False)

        assert isinstance(sample, LWESample), "training data must be the public view"
        assert not hasattr(sample, "secret"), "public sample must not expose the secret"
        assert not hasattr(sample, "error"), "public sample must not expose the error"

        assert src_ids.shape[0] == tgt_ids.shape[0], (
            f"batch mismatch: src {src_ids.shape[0]} vs tgt {tgt_ids.shape[0]}"
        )
        assert src_ids.shape[1] == self.codec.input_length, (
            f"source length {src_ids.shape[1]} != codec {self.codec.input_length}"
        )
        assert tgt_ids.shape[1] == self.codec.output_length, (
            f"target length {tgt_ids.shape[1]} != codec {self.codec.output_length}"
        )

        decoder_input, target = shift_for_teacher_forcing(tgt_ids)
        assert decoder_input.shape[1] == tgt_ids.shape[1] - 1, "decoder input must be T-1"
        assert target.shape == decoder_input.shape, "shifted target must match input"

        self.model.eval()
        with torch.no_grad():
            logits = self.model(src_ids, decoder_input)
        self.model.train()

        assert logits.shape[:2] == target.shape, (
            f"logits {tuple(logits.shape)[:2]} != target {tuple(target.shape)}"
        )
        assert logits.shape[-1] == self.codec.vocabulary.size, (
            f"logit vocabulary {logits.shape[-1]} != codec {self.codec.vocabulary.size}"
        )
        assert logits.shape[-1] == self.model.vocab_size, "model/codec vocabulary mismatch"
        assert self.codec.vocabulary.pad_id == self.model.spec.pad_id, (
            "padding id differs between codec and model"
        )
        assert torch.all(torch.isfinite(logits)), "forward pass produced non-finite logits"

        shapes = {
            "src_ids": tuple(src_ids.shape),
            "tgt_ids": tuple(tgt_ids.shape),
            "decoder_input": tuple(decoder_input.shape),
            "target": tuple(target.shape),
            "logits": tuple(logits.shape),
            "vocab_size": int(self.codec.vocabulary.size),
            "pad_id": int(self.codec.vocabulary.pad_id),
        }
        self.logger.info("Alignment verified: %s", shapes)
        return shapes

    # -- training ----------------------------------------------------------- #
    def train_step(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, float]:
        """Run one optimiser step on one batch.

        Args:
            batch: ``(src_ids, tgt_ids)``.

        Returns:
            ``{"loss": ..., "grad_norm": ..., "lr": ...}``.
        """
        src_ids, tgt_ids = batch
        decoder_input, target = shift_for_teacher_forcing(tgt_ids)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(src_ids, decoder_input)
        loss_out = sequence_cross_entropy(
            logits,
            target,
            ignore_index=self.codec.vocabulary.pad_id,
            label_smoothing=float(self.config.training.label_smoothing),
        )
        loss_out.loss.backward()

        clip = float(self.config.training.grad_clip)
        grad_norm = (
            float(nn.utils.clip_grad_norm_(self.model.parameters(), clip))
            if clip > 0
            else float("nan")
        )
        self.optimizer.step()
        self.scheduler.step()

        self.state.global_step += 1
        self.state.samples_seen += int(src_ids.shape[0])
        self.state.tokens_seen += int(loss_out.num_tokens)
        return {
            "loss": float(loss_out.loss.detach()),
            "grad_norm": grad_norm,
            "lr": float(self.optimizer.param_groups[0]["lr"]),
        }

    def train_epoch(self) -> Dict[str, float]:
        """Run one epoch of ``steps_per_epoch`` optimiser steps."""
        start_step = self.state.global_step
        remaining = self.plan.total_steps - start_step
        steps = min(self.plan.steps_per_epoch, remaining)

        loss_sum, tokens_before = 0.0, self.state.tokens_seen
        epoch_start = time.time()

        for offset in range(steps):
            batch = self.train_stream.batch(start_step + offset)
            stats = self.train_step(batch)
            loss_sum += stats["loss"]

            if self.config.training.log_every > 0 and (
                self.state.global_step % self.config.training.log_every == 0
            ):
                elapsed = time.time() - epoch_start
                done = offset + 1
                self.logger.info(
                    "epoch %d | step %d/%d | loss %.4f | lr %.2e | %.1f samples/s",
                    self.state.epoch,
                    self.state.global_step,
                    self.plan.total_steps,
                    stats["loss"],
                    stats["lr"],
                    done * self.plan.batch_size / max(elapsed, 1e-9),
                )

        seconds = time.time() - epoch_start
        self.throughput.update(
            steps * self.plan.batch_size, self.state.tokens_seen - tokens_before, seconds
        )
        self.state.elapsed_seconds = time.time() - self._start_time
        return {
            "train_loss": loss_sum / max(steps, 1),
            "epoch_seconds": seconds,
            "steps": steps,
        }

    def validate(self) -> EvaluationResult:
        """Evaluate on the fixed validation set."""
        return evaluate(
            self.model,
            self.valid_batches,
            self.codec,
            tolerance=float(self.config.evaluation.tolerance),
            ignore_index=self.codec.vocabulary.pad_id,
        )

    def train(self) -> TrainingState:
        """Run the full schedule.

        Returns:
            The final :class:`TrainingState`, including the stop reason.
        """
        self.log_run_header()
        self.verify_alignment()

        while self.state.epoch < self.plan.epochs:
            if self.state.global_step >= self.plan.total_steps:
                self.state.stop_reason = "sample budget exhausted"
                break

            train_stats = self.train_epoch()
            row = self.record_metrics(train_stats)
            self.save_checkpoint("last")

            if self._update_best(row):
                self.save_checkpoint("best")

            self.state.epoch += 1

            patience = int(self.config.training.early_stopping_patience)
            if patience > 0 and self.state.epochs_without_improvement >= patience:
                self.state.stop_reason = (
                    f"early stopping: no improvement in {patience} evaluations"
                )
                self.logger.info(self.state.stop_reason)
                break
        else:
            self.state.stop_reason = "planned epochs completed"

        if not self.state.stop_reason:
            self.state.stop_reason = "sample budget exhausted"
        self.save_checkpoint("last")
        self.write_summary()
        self.logger.info("Training finished: %s", self.state.stop_reason)
        return self.state

    # -- bookkeeping -------------------------------------------------------- #
    def _update_best(self, row: Dict[str, Any]) -> bool:
        """Update the monitored metric; return True when it improved."""
        metric_name = self.config.training.monitor_metric
        if metric_name not in row:
            self.logger.warning("Monitor metric '%s' not in metrics row.", metric_name)
            return False
        value = float(row[metric_name])
        better = (
            self.state.best_metric is None
            or (self.config.training.monitor_mode == "min" and value < self.state.best_metric)
            or (self.config.training.monitor_mode == "max" and value > self.state.best_metric)
        )
        if better:
            self.state.best_metric = value
            self.state.best_epoch = self.state.epoch
            self.state.epochs_without_improvement = 0
            self.logger.info("New best %s: %.6f", metric_name, value)
        else:
            self.state.epochs_without_improvement += 1
        return better

    def record_metrics(self, train_stats: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate, assemble the metrics row and write it to the run stream."""
        result = self.validate()
        memory = memory_usage_mb()
        row: Dict[str, Any] = {
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
            "train_loss": round(float(train_stats["train_loss"]), 6),
            "samples_seen": self.state.samples_seen,
            "tokens_seen": self.state.tokens_seen,
            "epoch_seconds": round(float(train_stats["epoch_seconds"]), 3),
            "elapsed_seconds": round(self.state.elapsed_seconds, 3),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "cpu_rss_mb": round(memory["rss_mb"], 2),
            "parameter_count": self.parameters.trainable,
            "model_name": self.model.name,
            "config_fingerprint": self.config.fingerprint(),
            "seed": self.config.experiment.seed,
        }
        row.update(result.to_dict("valid"))
        throughput = self.throughput.to_dict()
        clashes = set(throughput) & set(row)
        assert not clashes, f"throughput keys would overwrite run metrics: {clashes}"
        row.update(throughput)
        self.context.metrics.log(row, step=self.state.global_step)

        self.logger.info(
            "epoch %d | train %.4f | valid %.4f | token %.3f | acc_tau %.3f | "
            "exact %.4f | %.1f samples/s",
            self.state.epoch,
            row["train_loss"],
            result.loss,
            result.token_accuracy,
            result.acc_tau,
            result.exact_accuracy,
            self.throughput.samples_per_second,
        )
        return row

    def log_run_header(self) -> None:
        """Log and persist everything needed to interpret the run."""
        self.logger.info("=" * 72)
        self.logger.info("Model: %s (%s)", self.model.name, self.model.spec.arch)
        self.logger.info("Trainable parameters: %d", self.parameters.trainable)
        self.logger.info(
            "LWE: n=%d q=%d sigma=%s h=%d structure=%s",
            self.config.lwe.n,
            self.config.lwe.q,
            self.config.lwe.sigma,
            self.config.lwe.resolved_hamming_weight,
            self.config.lwe.structure,
        )
        self.logger.info(
            "Encoding: base=%d order=%s separator=%s -> %d input tokens, %d output",
            self.codec.input_encoder.base,
            self.codec.input_encoder.digit_order,
            self.codec.separator,
            self.codec.input_length,
            self.codec.output_length,
        )
        self.logger.info("Plan: %s", self.plan.to_dict())
        self.logger.info("Device: %s | threads: %s", self.device_spec.name, self.threads)
        self.logger.info("Chance baselines: %s", self.baselines)
        self.logger.info("=" * 72)

        self.context.save_json(
            "run_header.json",
            {
                "model": self.model.describe(),
                "parameters": self.parameters.to_dict(),
                "codec": self.codec.describe(),
                "plan": self.plan.to_dict(),
                "chance_baselines": self.baselines,
                "device": self.device_spec.to_dict(),
                "threads": self.threads,
                "seed": self.seed_state.to_dict(),
                "config_fingerprint": self.config.fingerprint(),
                "torch_version": torch.__version__,
                "platform": platform.platform(),
            },
        )

    def write_summary(self) -> Path:
        """Write the final summary JSON into the run's artifact directory."""
        result = self.validate()
        summary = {
            "model_name": self.model.name,
            "architecture": self.model.spec.arch,
            "parameter_count": self.parameters.trainable,
            "config_fingerprint": self.config.fingerprint(),
            "seed": self.config.experiment.seed,
            "experiment": self.config.experiment.name,
            "plan": self.plan.to_dict(),
            "state": self.state.to_dict(),
            "final_validation": result.to_dict("valid"),
            "throughput": self.throughput.to_dict(),
            "chance_baselines": self.baselines,
            "cpu_memory_mb": memory_usage_mb(),
            "run_dir": str(self.context.run_dir),
        }
        return self.context.save_json("summary.json", summary)

    # -- checkpoints -------------------------------------------------------- #
    def checkpoint_path(self, name: str) -> Path:
        """Return the path of a named checkpoint."""
        return self.context.checkpoint_dir / f"{name}.pt"

    def save_checkpoint(self, name: str = "last") -> Path:
        """Save model, optimiser, scheduler, progress and RNG state.

        Args:
            name: Checkpoint name, e.g. ``"last"`` or ``"best"``.

        Returns:
            The path written.
        """
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "state": self.state.to_dict(),
            "plan": self.plan.to_dict(),
            "spec": self.model.spec.to_dict(),
            "config": self.config.to_dict(),
            "config_fingerprint": self.config.fingerprint(),
            "parameter_count": self.parameters.trainable,
            "rng": {
                "torch": torch.get_rng_state(),
                "numpy": np.random.get_state(),
            },
        }
        path = self.checkpoint_path(name)
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path: "Path | str", strict: bool = True) -> TrainingState:
        """Restore a checkpoint, including optimiser and progress.

        Args:
            path: Checkpoint file.
            strict: Require the configuration fingerprint to match.

        Returns:
            The restored :class:`TrainingState`.

        Raises:
            FileNotFoundError: If the checkpoint does not exist.
            ValueError: If ``strict`` and the fingerprint differs.
        """
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"No checkpoint at {target}")
        payload = torch.load(target, map_location=self.device, weights_only=False)

        saved = payload.get("config_fingerprint")
        if strict and saved != self.config.fingerprint():
            raise ValueError(
                f"Checkpoint fingerprint {saved} does not match the current "
                f"configuration {self.config.fingerprint()}; refusing to resume a run "
                "whose settings changed."
            )

        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.state = TrainingState(**payload["state"])
        rng = payload.get("rng", {})
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        self._start_time = time.time() - self.state.elapsed_seconds
        self.logger.info(
            "Resumed from %s at epoch %d / step %d (%d samples seen)",
            target,
            self.state.epoch,
            self.state.global_step,
            self.state.samples_seen,
        )
        return self.state

    def maybe_resume(self) -> bool:
        """Resume from ``last.pt`` when the config asks for it and one exists."""
        if not self.config.training.resume:
            return False
        path = self.checkpoint_path("last")
        if not path.is_file():
            return False
        self.load_checkpoint(path)
        return True
