"""Evaluation metrics, kept deliberately separate from one another.

Three different things are measured and must never be conflated:

**Token-level metrics** (teacher forced)
    ``token_accuracy`` and ``perfect_accuracy`` score the decoder while it is
    fed the *true* previous tokens.  Cheap, but optimistic: at position 2 the
    model has already been shown digit 1.

**Decoded-integer metrics** (free running)
    The model generates ``b`` greedily from ``a`` alone, the digits are decoded
    back to an integer, and that integer is compared with the true ``b``.  This
    is what an attacker actually gets.

**Tolerance accuracy** ``acc_tau`` -- THE PRIMARY COMPATIBILITY METRIC
    ``b`` carries the LWE error, so requiring ``b_hat == b`` exactly is the
    wrong success criterion.  The SALSA paper scores a prediction as correct
    when ``|b - b_hat| <= tau * q``, with ``tau = 0.1``, using the **plain
    absolute difference** of the decoded integers (verified against the released
    ``generators.py``, whose ``get_difference`` is ``abs(hyp[0] - tgt[0])``).
    That exact definition is what :attr:`EvaluationResult.acc_tau` reports, so
    the number is directly comparable with the paper.  Exact-match accuracy is
    still reported, but as a secondary number.

**NEW ADDITIONAL DIAGNOSTICS (not from the paper)**
    ``b`` lives on the ring ``Z_q``, where 0 and ``q - 1`` are neighbours, so
    the plain absolute difference overstates the error for pairs that straddle
    the wrap-around point.  ``acc_tau_circular`` and the circular distance
    statistics apply ``min(|d|, q - |d|)`` instead.  These are **additions**,
    reported alongside the SALSA metric and never in place of it.

Chance baselines matter here more than usual, and the token-level and
integer-level baselines are different numbers that must never be mixed up.
With base 81 and ``q = 251`` the high digit of ``b`` is ``b // 81``, which is 0,
1 or 2 for 96.8% of values, so a model that has learned nothing still reaches
about 32% accuracy on that token and roughly 44% *token* accuracy overall --
while its *integer-level* accuracy stays at 1/251.  A 44% token accuracy is
therefore not evidence of learning.  :func:`chance_baselines` computes every
baseline exactly, separated by level, so each measurement can be read against
its own zero point.

Nothing here uses the secret or the error: predictions are compared against the
public ``b`` only.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from ..data.encoding import EncodingError, LatticeCodec

__all__ = [
    "EvaluationResult",
    "ThroughputMeter",
    "absolute_distance",
    "chance_baselines",
    "circular_distance",
    "decode_generated_ids",
    "evaluate",
    "greedy_decode",
    "perfect_accuracy",
    "token_accuracy",
]


# --------------------------------------------------------------------------- #
# Token-level
# --------------------------------------------------------------------------- #
def token_accuracy(
    logits: Tensor, targets: Tensor, ignore_index: int = 0
) -> Tuple[int, int]:
    """Count correctly predicted tokens, ignoring padding.

    Args:
        logits: ``(batch, seq, vocab)``.
        targets: ``(batch, seq)``.
        ignore_index: Token id to skip.

    Returns:
        ``(correct, total)`` token counts.
    """
    predictions = logits.argmax(dim=-1)
    scored = targets != ignore_index
    correct = int(((predictions == targets) & scored).sum())
    return correct, int(scored.sum())


def perfect_accuracy(
    logits: Tensor, targets: Tensor, ignore_index: int = 0
) -> Tuple[int, int]:
    """Count sequences whose every scored token is correct.

    Args:
        logits: ``(batch, seq, vocab)``.
        targets: ``(batch, seq)``.
        ignore_index: Token id to skip.

    Returns:
        ``(correct_sequences, total_sequences)``.
    """
    predictions = logits.argmax(dim=-1)
    scored = targets != ignore_index
    hit = (predictions == targets) | ~scored
    return int(hit.all(dim=1).sum()), int(targets.shape[0])


# --------------------------------------------------------------------------- #
# Decoded-integer level
# --------------------------------------------------------------------------- #
def absolute_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Plain ``|a - b|`` -- the distance the SALSA paper uses.

    This is the SALSA-compatible definition, matching ``get_difference`` in the
    released ``src/envs/generators.py``.  It is what feeds
    :attr:`EvaluationResult.acc_tau`.

    Args:
        a: Integer array.
        b: Integer array of the same shape.

    Returns:
        ``|a - b|``.
    """
    return np.abs(np.asarray(a, dtype=np.int64) - np.asarray(b, dtype=np.int64))


def circular_distance(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    """Distance on the ring ``Z_q``.  **NEW ADDITIONAL DIAGNOSTIC.**

    Not part of the SALSA paper.  ``b`` lives on a torus, so 0 and ``q - 1`` are
    neighbours; this never exceeds :func:`absolute_distance` and is reported
    alongside it, never instead of it.

    Args:
        a: Integer array.
        b: Integer array of the same shape.
        q: Modulus.

    Returns:
        ``min(|a - b|, q - |a - b|)``.
    """
    difference = np.abs(np.asarray(a, dtype=np.int64) - np.asarray(b, dtype=np.int64)) % q
    return np.minimum(difference, q - difference)


@torch.no_grad()
def greedy_decode(
    model,
    src_ids: Tensor,
    bos_id: int,
    max_new_tokens: int,
    src_valid: Optional[Tensor] = None,
) -> Tensor:
    """Generate the output sequence greedily, one token at a time.

    This is plain argmax decoding, not beam search: the SALSA output is two
    digits, so three sequential steps suffice.  The encoder runs once and its
    output is reused for every step.

    Args:
        model: A :class:`~salsa.models.SalsaTransformer`.
        src_ids: Encoded ``a``, shape ``(batch, src_len)``.
        bos_id: Begin-of-sequence token id.
        max_new_tokens: How many tokens to generate after ``<bos>``.
        src_valid: Optional source validity mask.

    Returns:
        Generated ids of shape ``(batch, 1 + max_new_tokens)``, starting with
        ``<bos>``.
    """
    was_training = model.training
    model.eval()
    memory = model.encode(src_ids, src_valid=src_valid)
    generated = torch.full(
        (src_ids.shape[0], 1), int(bos_id), dtype=torch.long, device=src_ids.device
    )
    for _ in range(int(max_new_tokens)):
        logits = model.decode(memory, generated, src_valid=src_valid)
        next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=1)
    if was_training:
        model.train()
    return generated


def decode_generated_ids(
    codec: LatticeCodec, generated: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Turn generated token ids back into integers.

    A model is free to emit a token sequence that decodes to nothing at all --
    a separator where a digit belongs, for instance.  Those cases are counted
    rather than hidden, because "the model produced an unreadable answer" is a
    different failure from "the model produced a wrong number".

    Args:
        codec: The codec used to build the sequences.
        generated: Integer array ``(batch, seq)`` of generated token ids.

    Returns:
        ``(values, valid)`` where ``values`` holds the decoded integers (``-1``
        where decoding failed) and ``valid`` is a boolean mask of successes.
    """
    values = np.full(generated.shape[0], -1, dtype=np.int64)
    valid = np.zeros(generated.shape[0], dtype=bool)
    for index, row in enumerate(generated):
        try:
            values[index] = codec.decode_output_ids(row.tolist(), strict=False)
            valid[index] = True
        except EncodingError:
            continue
    return values, valid


def _entropy(values: np.ndarray) -> float:
    """Shannon entropy in nats of the empirical distribution of ``values``."""
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log(probabilities)).sum())


def chance_baselines(codec: LatticeCodec, tolerance: float) -> Dict[str, float]:
    """Compute what a model that learned nothing would score.

    Every number here follows from the representation alone, so a measured
    metric is only evidence of learning if it clears the matching baseline.

    Args:
        codec: The codec defining the output representation.
        tolerance: ``tau``, as a fraction of ``q``.

    Returns:
        A dictionary of baseline values.
    """
    q = codec.q
    values = np.arange(q, dtype=np.int64)
    digits = codec.output_encoder.digits_array(values)  # (q, width)

    per_digit_best: List[float] = []
    for position in range(digits.shape[1]):
        counts = np.bincount(digits[:, position] - digits[:, position].min())
        per_digit_best.append(float(counts.max()) / q)

    # Scored positions under teacher forcing: the digits plus the final <eos>,
    # which is fully predictable.
    scored = per_digit_best + [1.0]
    bound = tolerance * q

    # Integer-level chance: a uniformly random guess b_hat over Z_q, averaged
    # over the true b.  Computed exactly rather than approximated.
    grid = np.abs(values[:, None] - values[None, :])
    salsa_chance = float((grid <= bound).mean())
    circular_chance = float((np.minimum(grid, q - grid) <= bound).mean())

    # Loss baselines.  The *uniform* baseline is far too generous: a model that
    # learns only the output marginals already reaches the "marginal" figure
    # without knowing anything about the secret, so that is the real zero point
    # for the loss.  The floor is what a perfect attacker who knew s exactly
    # would still pay, because b carries the LWE error.
    marginal_nats = float(
        sum(_entropy(digits[:, i]) for i in range(digits.shape[1])) / len(scored)
    )

    return {
        # -- loss baselines --------------------------------------------------- #
        "uniform_loss_nats": math.log(codec.vocabulary.size),
        "marginal_loss_nats": marginal_nats,
        # -- token level (do NOT read these as evidence of learning) --------- #
        "chance_token_accuracy": float(np.mean(scored)),
        "chance_perfect_accuracy": float(np.prod(per_digit_best)),
        "chance_digit_accuracy": [round(v, 6) for v in per_digit_best],
        # -- integer level --------------------------------------------------- #
        "chance_exact_accuracy": 1.0 / q,
        "chance_acc_tau": salsa_chance,
        "chance_acc_tau_circular_DIAGNOSTIC": circular_chance,
        "tolerance": float(tolerance),
        "tolerance_absolute": float(bound),
    }


# --------------------------------------------------------------------------- #
# Aggregate result
# --------------------------------------------------------------------------- #
@dataclass
class EvaluationResult:
    """Everything measured in one validation pass.

    Attributes:
        loss: Mean token cross entropy.
        token_accuracy: Teacher-forced token accuracy.
        perfect_accuracy: Teacher-forced whole-sequence accuracy.
        greedy_token_accuracy: Token accuracy of the free-running generation.
        exact_accuracy: Fraction where the decoded integer equals ``b`` exactly.
        acc_tau: **SALSA metric.** Fraction with ``|b - b_hat| <= tau * q``.
        tolerance: The ``tau`` that was used.
        mean_distance: Mean ``|b - b_hat|`` (SALSA definition).
        median_distance: Median ``|b - b_hat|`` (SALSA definition).
        percentiles: Fraction within ``0.1 q, 0.2 q, ...`` of ``b``, SALSA
            definition.  ``percentiles[0]`` is ``acc_tau`` at ``tau = 0.1``.
        acc_tau_circular: NEW ADDITIONAL DIAGNOSTIC -- the same fraction under
            the ring distance ``min(|d|, q - |d|)``.
        mean_circular_distance: NEW ADDITIONAL DIAGNOSTIC.
        median_circular_distance: NEW ADDITIONAL DIAGNOSTIC.
        decode_failure_rate: Fraction of generations that did not decode.
        num_sequences: Sequences evaluated.
        num_tokens: Tokens scored.
    """

    loss: float = 0.0
    token_accuracy: float = 0.0
    perfect_accuracy: float = 0.0
    greedy_token_accuracy: float = 0.0
    exact_accuracy: float = 0.0
    # -- SALSA-compatible primary metric ------------------------------------- #
    acc_tau: float = 0.0
    tolerance: float = 0.1
    mean_distance: float = 0.0
    median_distance: float = 0.0
    percentiles: List[float] = field(default_factory=list)
    # -- NEW ADDITIONAL DIAGNOSTICS (ring distance; not from the paper) ------- #
    acc_tau_circular: float = 0.0
    mean_circular_distance: float = 0.0
    median_circular_distance: float = 0.0
    # ------------------------------------------------------------------------ #
    decode_failure_rate: float = 0.0
    num_sequences: int = 0
    num_tokens: int = 0

    def to_dict(self, prefix: str = "valid") -> Dict[str, Any]:
        """Flatten into a metrics row for the run's JSONL/CSV stream."""
        return {
            f"{prefix}_loss": round(self.loss, 6),
            f"{prefix}_token_accuracy": round(self.token_accuracy, 6),
            f"{prefix}_perfect_accuracy": round(self.perfect_accuracy, 6),
            f"{prefix}_greedy_token_accuracy": round(self.greedy_token_accuracy, 6),
            f"{prefix}_exact_accuracy": round(self.exact_accuracy, 6),
            f"{prefix}_acc_tau": round(self.acc_tau, 6),
            f"{prefix}_tolerance": self.tolerance,
            f"{prefix}_mean_distance": round(self.mean_distance, 4),
            f"{prefix}_median_distance": round(self.median_distance, 4),
            f"{prefix}_percentiles": str([round(p, 4) for p in self.percentiles]),
            f"{prefix}_acc_tau_circular_DIAGNOSTIC": round(self.acc_tau_circular, 6),
            f"{prefix}_mean_circular_distance_DIAGNOSTIC": round(
                self.mean_circular_distance, 4
            ),
            f"{prefix}_median_circular_distance_DIAGNOSTIC": round(
                self.median_circular_distance, 4
            ),
            f"{prefix}_decode_failure_rate": round(self.decode_failure_rate, 6),
            f"{prefix}_sequences": self.num_sequences,
            f"{prefix}_tokens": self.num_tokens,
        }


class ThroughputMeter:
    """Accumulates samples/second and tokens/second over a run."""

    def __init__(self) -> None:
        """Start an empty meter."""
        self.samples = 0
        self.tokens = 0
        self.seconds = 0.0

    def update(self, samples: int, tokens: int, seconds: float) -> None:
        """Record one interval of work."""
        self.samples += int(samples)
        self.tokens += int(tokens)
        self.seconds += float(seconds)

    @property
    def samples_per_second(self) -> float:
        """Mean samples per second, or 0 before any work."""
        return self.samples / self.seconds if self.seconds > 0 else 0.0

    @property
    def tokens_per_second(self) -> float:
        """Mean tokens per second, or 0 before any work."""
        return self.tokens / self.seconds if self.seconds > 0 else 0.0

    def to_dict(self) -> Dict[str, float]:
        """Return the accumulated throughput figures."""
        return {
            "samples_per_second": round(self.samples_per_second, 3),
            "tokens_per_second": round(self.tokens_per_second, 1),
            "elapsed_seconds": round(self.seconds, 3),
        }


# --------------------------------------------------------------------------- #
# Evaluation driver
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(
    model,
    batches: Sequence[Tuple[Tensor, Tensor]],
    codec: LatticeCodec,
    tolerance: float = 0.1,
    ignore_index: int = 0,
    label_smoothing: float = 0.0,
    compute_greedy: bool = True,
) -> EvaluationResult:
    """Run one full validation pass.

    Teacher-forced and free-running metrics are computed in the same pass but
    kept separate in the result, because they answer different questions.

    Only the public pair ``(a, b)`` is used: ``b`` is recovered by decoding the
    target token sequence, never by consulting the generator's ground truth.

    Args:
        model: A :class:`~salsa.models.SalsaTransformer`.
        batches: Sequence of ``(src_ids, tgt_ids)`` integer tensors.
        codec: The codec used to build those sequences.
        tolerance: ``tau``, as a fraction of ``q``.
        ignore_index: Padding id excluded from token metrics and the loss.
        label_smoothing: Label smoothing for the reported loss.
        compute_greedy: Also run greedy generation for the decoded metrics.

    Returns:
        An :class:`EvaluationResult`.
    """
    from .losses import sequence_cross_entropy, shift_for_teacher_forcing

    was_training = model.training
    model.eval()

    loss_sum = 0.0
    token_hits = token_total = 0
    perfect_hits = perfect_total = 0
    greedy_hits = greedy_total = 0
    predicted: List[np.ndarray] = []
    truths: List[np.ndarray] = []
    valid_masks: List[np.ndarray] = []

    bos_id = codec.vocabulary.bos_id
    generate_steps = codec.output_length - 1

    for src_ids, tgt_ids in batches:
        decoder_input, target = shift_for_teacher_forcing(tgt_ids)
        logits = model(src_ids, decoder_input)

        loss_out = sequence_cross_entropy(
            logits, target, ignore_index=ignore_index, label_smoothing=label_smoothing
        )
        loss_sum += loss_out.total

        hits, total = token_accuracy(logits, target, ignore_index)
        token_hits += hits
        token_total += total
        hits, total = perfect_accuracy(logits, target, ignore_index)
        perfect_hits += hits
        perfect_total += total

        if compute_greedy:
            generated = greedy_decode(model, src_ids, bos_id, generate_steps)
            reference = tgt_ids[:, 1 : 1 + generate_steps]
            produced = generated[:, 1 : 1 + generate_steps]
            greedy_hits += int((produced == reference).sum())
            greedy_total += int(reference.numel())

            values, valid = decode_generated_ids(codec, generated.cpu().numpy())
            predicted.append(values)
            valid_masks.append(valid)
            truths.append(
                codec.decode_batch_outputs(tgt_ids.cpu().numpy(), strict=False)
            )

    result = EvaluationResult(
        loss=loss_sum / max(token_total, 1),
        token_accuracy=token_hits / max(token_total, 1),
        perfect_accuracy=perfect_hits / max(perfect_total, 1),
        greedy_token_accuracy=greedy_hits / max(greedy_total, 1),
        tolerance=float(tolerance),
        num_sequences=perfect_total,
        num_tokens=token_total,
    )

    if compute_greedy and predicted:
        prediction = np.concatenate(predicted)
        truth = np.concatenate(truths)
        valid = np.concatenate(valid_masks)
        q = codec.q
        bound = tolerance * q

        # A generation that did not decode counts as a miss, never as a skip.
        plain = np.where(valid, absolute_distance(prediction, truth), q)
        ring = np.where(valid, circular_distance(prediction, truth, q), q)

        result.exact_accuracy = float(((plain == 0) & valid).mean())
        result.acc_tau = float((plain <= bound).mean())
        result.mean_distance = float(plain.mean())
        result.median_distance = float(np.median(plain))
        result.percentiles = [
            float((plain <= step / 10.0 * q).mean()) for step in range(1, 11)
        ]
        result.acc_tau_circular = float((ring <= bound).mean())
        result.mean_circular_distance = float(ring.mean())
        result.median_circular_distance = float(np.median(ring))
        result.decode_failure_rate = float((~valid).mean())

    if was_training:
        model.train()
    return result
