"""Training subsystem: seeds, objective, metrics and the CPU training loop.

The loop consumes **public LWE samples only** -- ``A`` and ``b``, never the
secret or the error -- so the training signal is exactly what an attacker
holding intercepted samples would have.  See :mod:`salsa.training.trainer`.

Metrics are kept strictly separated by level: teacher-forced token accuracy,
free-running decoded-integer accuracy, and the SALSA tolerance metric
``acc_tau``.  Chance baselines are computed for each level, because at base 81
a model that has learned nothing still scores ~44% token accuracy.
"""

from .losses import LossOutput, sequence_cross_entropy, shift_for_teacher_forcing
from .metrics import (
    EvaluationResult,
    ThroughputMeter,
    absolute_distance,
    chance_baselines,
    circular_distance,
    decode_generated_ids,
    evaluate,
    greedy_decode,
    perfect_accuracy,
    token_accuracy,
)
from .seed import SeedState, derive_seed, seed_worker, set_seed, temporary_seed
from .trainer import BatchStream, Trainer, TrainingPlan, TrainingState

__all__ = [
    # seeds
    "SeedState",
    "derive_seed",
    "seed_worker",
    "set_seed",
    "temporary_seed",
    # objective
    "LossOutput",
    "sequence_cross_entropy",
    "shift_for_teacher_forcing",
    # metrics
    "EvaluationResult",
    "ThroughputMeter",
    "absolute_distance",
    "circular_distance",
    "chance_baselines",
    "decode_generated_ids",
    "evaluate",
    "greedy_decode",
    "perfect_accuracy",
    "token_accuracy",
    # training
    "BatchStream",
    "Trainer",
    "TrainingPlan",
    "TrainingState",
]
