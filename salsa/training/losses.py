"""Training objective for the SALSA ``a -> b`` sequence task.

The objective is ordinary token-level cross entropy over the decoder's output
sequence, with padding excluded.  Nothing cryptographic is baked into the loss:

* the loss sees only the **public** pair ``(a, b)``.  The secret ``s`` and the
  error ``e`` never enter it, so the training signal is exactly what a real
  attacker holding intercepted LWE samples would have;
* ``b`` already contains the LWE error, so the model is being asked to predict a
  noisy target.  A perfect cross-entropy of zero is therefore not attainable and
  is not the goal -- see :mod:`salsa.training.metrics` for why exact prediction
  of ``b`` is the wrong success criterion.

Inventing a bespoke "cryptographic" loss is deliberately out of scope: the
research question is whether a smaller model can learn the same mapping, which
requires holding the objective fixed.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

__all__ = ["LossOutput", "sequence_cross_entropy", "shift_for_teacher_forcing"]


@dataclass
class LossOutput:
    """Result of one loss evaluation.

    Attributes:
        loss: Mean cross entropy over the scored (non-padding) tokens.
        num_tokens: How many tokens were scored.
        num_sequences: How many sequences contributed.
    """

    loss: Tensor
    num_tokens: int
    num_sequences: int

    @property
    def total(self) -> float:
        """Sum of the per-token losses, for exact averaging across batches."""
        return float(self.loss.detach()) * self.num_tokens


def shift_for_teacher_forcing(target_ids: Tensor) -> "tuple[Tensor, Tensor]":
    """Split a target sequence into decoder input and prediction target.

    The decoder is fed ``target[:, :-1]`` and scored against ``target[:, 1:]``,
    so position ``t`` predicts token ``t+1`` given everything before it.

    Args:
        target_ids: Integer tensor of shape ``(batch, seq)`` with ``seq >= 2``.

    Returns:
        ``(decoder_input, prediction_target)``, each ``(batch, seq - 1)``.

    Raises:
        ValueError: If the sequence is shorter than two tokens.
    """
    if target_ids.dim() != 2:
        raise ValueError(
            f"target_ids must be (batch, seq), got {tuple(target_ids.shape)}."
        )
    if target_ids.shape[1] < 2:
        raise ValueError(
            "teacher forcing needs at least two target tokens (<bos> plus one)."
        )
    return target_ids[:, :-1], target_ids[:, 1:]


def sequence_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    ignore_index: int = 0,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
) -> LossOutput:
    """Token-level cross entropy with padding excluded.

    Args:
        logits: Float tensor of shape ``(batch, seq, vocab)``.
        targets: Integer tensor of shape ``(batch, seq)``.
        ignore_index: Token id excluded from the loss (the padding id).
        label_smoothing: Label smoothing in ``[0, 1)``.
        reduction: ``"mean"`` (per scored token) or ``"sum"``.

    Returns:
        A :class:`LossOutput`.

    Raises:
        ValueError: On shape mismatches or an unknown reduction.
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be (batch, seq, vocab), got {tuple(logits.shape)}.")
    if targets.dim() != 2:
        raise ValueError(f"targets must be (batch, seq), got {tuple(targets.shape)}.")
    if logits.shape[:2] != targets.shape:
        raise ValueError(
            f"logits {tuple(logits.shape)[:2]} and targets {tuple(targets.shape)} "
            "must agree on batch and sequence length."
        )
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}.")

    vocab = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab)
    flat_targets = targets.reshape(-1)
    num_tokens = int((flat_targets != ignore_index).sum())

    loss = F.cross_entropy(
        flat_logits,
        flat_targets,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction="mean" if reduction == "mean" else "sum",
    )
    if num_tokens == 0:  # pragma: no cover - defensive
        loss = torch.zeros((), dtype=logits.dtype, device=logits.device)

    return LossOutput(
        loss=loss, num_tokens=num_tokens, num_sequences=int(targets.shape[0])
    )
