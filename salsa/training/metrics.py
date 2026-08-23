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

**Tolerance accuracy** ``acc_tau``
    ``b`` carries the LWE error, so requiring ``b_hat == b`` exactly is the
    wrong success criterion -- the paper scores a prediction as correct when
    ``|b - b_hat| <= tau * q``, with ``tau = 0.1``.  Exact-match accuracy is
    still reported, but as a secondary number.

Chance baselines matter here more than usual.  With base 81 and ``q = 251`` the
high digit of ``b`` is ``b // 81``, which is 0, 1 or 2 for 96.8% of values, so a
model that has learned nothing still reaches roughly 32% accuracy on that token
and about 44% token accuracy overall.  :func:`chance_baselines` computes these
exactly so a reported number can be read against the right zero point.

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
    "chance_baselines",
    "circular_distance",
    "decode_generated_ids",
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
def circular_distance(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    """Distance between two residues on the ring ``Z_q``.

    ``b`` lives on a torus, so 0 and ``q - 1`` are neighbours, not opposites.
    The original SALSA evaluator uses a plain ``|x - y|``; the circular distance
    is the correct notion on ``Z_q`` and never exceeds it.

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
    within = int((circular_distance(values, np.zeros_like(values), q) <= bound).sum())

    return {
        "chance_token_accuracy": float(np.mean(scored)),
        "chance_perfect_accuracy": float(np.prod(per_digit_best)),
        "chance_exact_accuracy": 1.0 / q,
        "chance_acc_tau": within / q,
        "chance_digit_accuracy": [round(v, 6) for v in per_digit_best],
        "uniform_loss_nats": math.log(codec.vocabulary.size),
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
        acc_tau: Fraction within ``tau * q`` of ``b`` (the SALSA criterion).
        tolerance: The ``tau`` that was used.
        mean_distance: Mean circular distance to ``b``.
        median_distance: Median circular distance to ``b``.
        decode_failure_rate: Fraction of generations that did not decode.
        num_sequences: Sequences evaluated.
        num_tokens: Tokens scored.
        percentiles: Fraction within ``0.1 q, 0.2 q, ...`` of ``b``.
    """

    loss: float = 0.0
    token_accuracy: float = 0.0
    perfect_accuracy: float = 0.0
    greedy_token_accuracy: float = 0.0
    exact_accuracy: float = 0.0
    acc_tau: float = 0.0
    tolerance: float = 0.1
    mean_distance: float = 0.0
    median_distance: float = 0.0
    decode_failure_rate: float = 0.0
    num_sequences: int = 0
    num_tokens: int = 0
    percentiles: List[float] = field(default_factory=list)

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
            f"{prefix}_decode_failure_rate": round(self.decode_failure_rate, 6),
            f"{prefix}_sequences": self.num_sequences,
            f"{prefix}_percentiles": str([round(p, 4) for p in self.percentiles]),
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
