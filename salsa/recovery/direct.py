"""Direct secret recovery: the chosen-input attack of SALSA Algorithm 1.

The method
----------
For each secret coordinate ``i`` the attacker feeds the model a probe

    a = K * e_i          (all coordinates zero except the i-th, which is K)

Because ``b = a . s + e mod q`` and the secret is binary, the noiseless value of
``b`` for that probe is

    a . s = K * s_i mod q       ->   s_i = 0 gives b ~ 0,  s_i = 1 gives b ~ K

so a model that has learned the secret should answer near 0 for a zero
coordinate and near ``K`` for a one.  Reading the model's answer for all ``n``
probes reconstructs a candidate secret.

What this module may see
------------------------
Everything here is computed from **public information only**: the modulus, the
dimension, the chosen ``K`` and the model's own predictions.  The true secret
and the error vector are never arguments, never imported and never referenced;
a test asserts this against the source text.  Scoring a candidate against the
truth is the evaluation layer's job, and it happens strictly after recovery has
finished.

Two deliberate departures from the released implementation
----------------------------------------------------------
**Decision rule.**  The released ``evaluator.eval_secret`` thresholds the raw
predictions at their mean/mode, produces a bit vector, and then also tries the
*inverted* vector, keeping whichever matches the true secret better.  That last
step consults ground truth, so it cannot be part of an attack.  Here the
decision is anchored instead: a prediction is assigned to whichever of the two
mathematically predicted values -- ``0`` or ``K mod q`` -- it is closer to on
the ring ``Z_q``.  This needs no ground truth and has no polarity ambiguity.
The original mean/median/mode rules remain available for comparison, with their
polarity fixed by the same anchor rather than by the secret.

**Probe magnitude.**  The released code uses very large ``K`` (e.g. 239145).
Its fixed-width encoder silently reduces those to ``K mod B^int_len`` (mod 6561
at base 81, width 2), which can leave the probe *outside* ``Z_q`` altogether.
Only ``K mod q`` is mathematically meaningful, since ``a . s = K s_i mod q``, so
this module reduces mod ``q`` and reports the reduced value.

A consequence worth stating plainly: distinguishability is governed by the ring
distance between the two hypotheses, ``min(K mod q, q - K mod q)``.  It is
maximal at ``K ~ q/2`` and collapses to nothing as ``K`` approaches 0 or ``q``.
A "large K" close to ``q`` is therefore a *bad* probe, not a good one.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from ..data.encoding import LatticeCodec
from ..training.metrics import decode_generated_ids, greedy_decode

__all__ = [
    "BINARIZATION_METHODS",
    "CoordinateProbe",
    "DirectRecovery",
    "DirectRecoveryReport",
    "KRecoveryResult",
    "build_probe_matrix",
    "probe_separation",
    "ring_distance",
]

#: Decision rules.  ``anchor`` is the default and the only one free of polarity
#: ambiguity; the rest reproduce the released code's thresholding.
BINARIZATION_METHODS = ("anchor", "mean", "median", "mode")

#: How a single candidate is chosen from the K sweep.  Both rules are
#: secret-free; they differ in robustness.
#:
#: ``aggregate``    separation-weighted vote across every K.  Each K's per-
#:                  coordinate score is weighted by its ring separation, so a
#:                  probe carrying almost no information contributes almost
#:                  nothing.  **This is the default.**
#: ``best_margin``  the single K with the highest mean absolute decision
#:                  margin.  Retained as a diagnostic.  See the warning on
#:                  :func:`DirectRecovery.recover` -- this rule is degenerate at
#:                  low separation and must be used with ``min_separation``.
SELECTION_RULES = ("aggregate", "best_margin")

#: Default guard for ``best_margin``.  A probe's two hypotheses are ``b ~ 0``
#: and ``b ~ K``, separated by ``min(K mod q, q - K mod q)``.  When that
#: separation is no larger than the error scale the two hypotheses sit inside
#: the noise and the probe cannot discriminate, so such K are excluded from
#: *selection* (they are still reported, and still vote with their tiny
#: separation weight).  ``None`` means "derive from sigma": ``floor(sigma) + 1``.
DEFAULT_MIN_SEPARATION = None


def ring_distance(x: np.ndarray, y: int, q: int) -> np.ndarray:
    """Distance from each entry of ``x`` to ``y`` on the ring ``Z_q``.

    Args:
        x: Integer array of residues.
        y: A single residue.
        q: Modulus.

    Returns:
        ``min(|x - y|, q - |x - y|)``, elementwise.
    """
    difference = np.abs(np.asarray(x, dtype=np.int64) - int(y)) % int(q)
    return np.minimum(difference, int(q) - difference)


def probe_separation(K: int, q: int) -> int:
    """Ring distance between the two hypotheses ``b ~ 0`` and ``b ~ K``.

    This is the entire information budget of a probe: when it is small the two
    secret values produce almost the same target and no model, however well
    trained, can distinguish them.

    Args:
        K: The probe multiplier.
        q: Modulus.

    Returns:
        ``min(K mod q, q - (K mod q))``, maximal at ``K = q/2``.
    """
    reduced = int(K) % int(q)
    return int(min(reduced, int(q) - reduced))


@dataclass(frozen=True)
class CoordinateProbe:
    """One probe vector targeting a single secret coordinate.

    Attributes:
        coordinate: The index ``i`` being probed.
        K: The multiplier as supplied.
        reduced_K: ``K mod q``, the value actually encoded.
        vector: The probe ``K * e_i mod q``, shape ``(n,)``.
    """

    coordinate: int
    K: int
    reduced_K: int
    vector: np.ndarray


def build_probe_matrix(n: int, K: int, q: int) -> np.ndarray:
    """Build the full probe matrix ``K * I_n mod q``.

    Row ``i`` is the probe for coordinate ``i``, so the matrix can be encoded
    and run through the model in a single batch, exactly like a batch of
    ordinary LWE rows.

    Args:
        n: Lattice dimension.
        K: Probe multiplier.
        q: Modulus.

    Returns:
        An ``int64`` array of shape ``(n, n)``.

    Raises:
        ValueError: If ``n < 1`` or ``q < 2``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    if q < 2:
        raise ValueError(f"q must be >= 2, got {q}.")
    return (np.eye(int(n), dtype=np.int64) * (int(K) % int(q))) % int(q)


def build_probes(n: int, K: int, q: int) -> List[CoordinateProbe]:
    """Return one :class:`CoordinateProbe` per coordinate."""
    matrix = build_probe_matrix(n, K, q)
    reduced = int(K) % int(q)
    return [
        CoordinateProbe(coordinate=i, K=int(K), reduced_K=reduced, vector=matrix[i])
        for i in range(int(n))
    ]


@dataclass
class CoordinateOutcome:
    """The model's answer for one probe, and the bit inferred from it.

    Attributes:
        coordinate: Index of the probed coordinate.
        K: Probe multiplier as supplied.
        reduced_K: ``K mod q``.
        predicted_tokens: The generated token ids.
        decoded_b: The decoded prediction, or -1 when it did not decode.
        decoded: Whether decoding succeeded.
        distance_to_zero: Ring distance from the prediction to 0.
        distance_to_K: Ring distance from the prediction to ``K mod q``.
        score: Signed margin in ``[-1, 1]``; positive favours ``s_i = 1``.
        decision: The recovered bit.
    """

    coordinate: int
    K: int
    reduced_K: int
    predicted_tokens: List[int]
    decoded_b: int
    decoded: bool
    distance_to_zero: int
    distance_to_K: int
    score: float
    decision: int

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable view.  Contains no ground truth."""
        return {
            "coordinate": self.coordinate,
            "K": self.K,
            "reduced_K": self.reduced_K,
            "decoded_b": self.decoded_b,
            "decoded": self.decoded,
            "distance_to_zero": self.distance_to_zero,
            "distance_to_K": self.distance_to_K,
            "score": round(self.score, 6),
            "recovered_bit": self.decision,
        }


@dataclass
class KRecoveryResult:
    """The candidate secret produced by one value of ``K``.

    Attributes:
        K: The multiplier used.
        reduced_K: ``K mod q``.
        separation: Ring distance between the two hypotheses.
        method: Decision rule applied.
        candidate: The recovered binary vector, shape ``(n,)``.
        outcomes: Per-coordinate detail.
        decode_failure_rate: Fraction of probes whose answer did not decode.
    """

    K: int
    reduced_K: int
    separation: int
    method: str
    candidate: np.ndarray
    outcomes: List[CoordinateOutcome]
    decode_failure_rate: float

    @property
    def mean_margin(self) -> float:
        """Mean absolute decision margin -- the model's decisiveness.

        This is computed from predictions alone, so it can be used to choose a
        ``K`` without ever consulting the true secret.
        """
        return float(np.mean([abs(o.score) for o in self.outcomes])) if self.outcomes else 0.0

    @property
    def recovered_weight(self) -> int:
        """Hamming weight of the candidate."""
        return int(self.candidate.sum())

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable view.  Contains no ground truth."""
        return {
            "K": self.K,
            "reduced_K": self.reduced_K,
            "separation": self.separation,
            "method": self.method,
            "candidate": self.candidate.tolist(),
            "recovered_hamming_weight": self.recovered_weight,
            "mean_margin": round(self.mean_margin, 6),
            "decode_failure_rate": round(self.decode_failure_rate, 6),
            "coordinates": [o.to_dict() for o in self.outcomes],
        }


@dataclass
class DirectRecoveryReport:
    """All candidates produced by a K sweep, plus the aggregate candidate.

    Attributes:
        n: Lattice dimension.
        q: Modulus.
        method: Decision rule used.
        per_k: One result per probed ``K``.
        aggregate_candidate: Candidate from separation-weighted voting.
        aggregate_scores: The summed weighted score per coordinate.
        selected_K: The ``K`` chosen without reference to any secret.
    """

    n: int
    q: int
    method: str
    per_k: List[KRecoveryResult]
    aggregate_candidate: np.ndarray
    aggregate_scores: np.ndarray
    selected_K: Optional[int] = None
    selection_rule: str = "aggregate"
    min_separation: int = 1
    excluded_k: List[int] = field(default_factory=list)

    @property
    def primary_candidate(self) -> np.ndarray:
        """The candidate the configured rule actually proposes.

        ``aggregate`` returns the separation-weighted vote; ``best_margin``
        returns the winning single-K candidate.  Callers that want "the answer"
        should read this rather than picking a field themselves.
        """
        if self.selection_rule == "best_margin" and self.selected_K is not None:
            for result in self.per_k:
                if result.K == self.selected_K:
                    return result.candidate
        return self.aggregate_candidate

    @property
    def candidates(self) -> List[np.ndarray]:
        """Every distinct candidate produced, aggregate first."""
        seen: List[np.ndarray] = [self.aggregate_candidate]
        for result in self.per_k:
            if not any(np.array_equal(result.candidate, c) for c in seen):
                seen.append(result.candidate)
        return seen

    def to_dict(self) -> Dict[str, Any]:
        """Return a serialisable view.  Contains no ground truth."""
        return {
            "n": self.n,
            "q": self.q,
            "method": self.method,
            "selection_rule": self.selection_rule,
            "primary_candidate": self.primary_candidate.tolist(),
            "min_separation": self.min_separation,
            "excluded_k_below_min_separation": self.excluded_k,
            "selected_K": self.selected_K,
            "selected_K_rule": ("highest mean decision margin among K whose "
                                "separation >= min_separation (predictions only)"),
            "aggregate_candidate": self.aggregate_candidate.tolist(),
            "aggregate_scores": [round(float(s), 6) for s in self.aggregate_scores],
            "distinct_candidates": len(self.candidates),
            "per_k": [r.to_dict() for r in self.per_k],
        }


class DirectRecovery:
    """Runs SALSA Algorithm 1 against a trained model.

    The constructor takes a model and a codec and nothing else: there is no
    parameter through which a secret could be supplied.

    Example:
        >>> recovery = DirectRecovery(model, codec)          # doctest: +SKIP
        >>> report = recovery.recover([31, 63, 125])          # doctest: +SKIP
        >>> report.aggregate_candidate                        # doctest: +SKIP
    """

    def __init__(
        self,
        model,
        codec: LatticeCodec,
        method: str = "anchor",
        batch_size: int = 64,
    ) -> None:
        """Create the recovery driver.

        Args:
            model: A trained :class:`~salsa.models.SalsaTransformer`.
            codec: The codec the model was trained with.
            method: Decision rule, one of :data:`BINARIZATION_METHODS`.
            batch_size: Probes per forward pass.

        Raises:
            ValueError: On an unknown method or a non-positive batch size.
        """
        if method not in BINARIZATION_METHODS:
            raise ValueError(
                f"method must be one of {list(BINARIZATION_METHODS)}, got {method!r}."
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")
        self.model = model
        self.codec = codec
        self.method = method
        self.batch_size = int(batch_size)
        self.n = int(codec.n)
        self.q = int(codec.q)

    # -- inference ---------------------------------------------------------- #
    @torch.no_grad()
    def predict(self, probe_matrix: np.ndarray) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
        """Run the model on a batch of probes.

        The decoder is driven from ``<bos>`` alone, so no target sequence -- and
        therefore no knowledge of ``b`` or of the secret -- is required.

        Args:
            probe_matrix: Integer array of shape ``(m, n)``.

        Returns:
            ``(decoded_values, decoded_ok, generated_ids)``.

        Raises:
            ValueError: If the probe shape does not match the codec.
        """
        matrix = np.asarray(probe_matrix, dtype=np.int64)
        if matrix.ndim != 2 or matrix.shape[1] != self.n:
            raise ValueError(
                f"probe matrix must have shape (m, {self.n}), got {tuple(matrix.shape)}."
            )

        was_training = self.model.training
        self.model.eval()
        values: List[np.ndarray] = []
        flags: List[np.ndarray] = []
        tokens: List[np.ndarray] = []

        bos = self.codec.vocabulary.bos_id
        steps = self.codec.output_length - 1
        for start in range(0, matrix.shape[0], self.batch_size):
            chunk = matrix[start : start + self.batch_size]
            # Encoding needs only the probe; b is not supplied.
            src_ids, _ = self.codec.encode_batch(chunk)
            generated = greedy_decode(
                self.model, torch.from_numpy(src_ids), bos, steps
            ).cpu().numpy()
            decoded, ok = decode_generated_ids(self.codec, generated)
            values.append(decoded)
            flags.append(ok)
            tokens.append(generated)
        if was_training:
            self.model.train()

        return (
            np.concatenate(values),
            np.concatenate(flags),
            np.concatenate(tokens, axis=0),
        )

    # -- decision rules ----------------------------------------------------- #
    def _decide(
        self, predictions: np.ndarray, decoded_ok: np.ndarray, reduced_K: int
    ) -> "tuple[np.ndarray, np.ndarray]":
        """Turn predictions into bits and signed margins.

        Args:
            predictions: Decoded values, one per coordinate.
            decoded_ok: Which of them decoded successfully.
            reduced_K: ``K mod q``.

        Returns:
            ``(bits, scores)``.  A probe that failed to decode yields bit 0 and
            score 0: unreadable is treated as "no evidence", never as evidence
            for either value.
        """
        to_zero = ring_distance(predictions, 0, self.q).astype(np.float64)
        to_k = ring_distance(predictions, reduced_K, self.q).astype(np.float64)

        if self.method == "anchor":
            total = to_zero + to_k
            scores = np.divide(
                to_zero - to_k, total, out=np.zeros_like(total), where=total > 0
            )
        else:
            # Reproduce the released thresholding, but anchor the polarity with
            # K instead of resolving it against the true secret.
            usable = predictions[decoded_ok]
            if usable.size == 0:
                threshold = 0.0
            elif self.method == "mean":
                threshold = float(usable.mean())
            elif self.method == "median":
                threshold = float(np.median(usable))
            else:  # mode
                values, counts = np.unique(usable, return_counts=True)
                threshold = float(values[int(np.argmax(counts))])
            above = predictions.astype(np.float64) - threshold
            # Which side of the threshold corresponds to s_i = 1 follows from
            # whether K sits above or below it -- a public quantity.
            polarity = 1.0 if float(reduced_K) >= threshold else -1.0
            span = max(float(self.q) / 2.0, 1.0)
            scores = np.clip(polarity * above / span, -1.0, 1.0)

        scores = np.where(decoded_ok, scores, 0.0)
        bits = (scores > 0).astype(np.int64)
        return bits, scores

    # -- recovery ----------------------------------------------------------- #
    def recover_for_k(self, K: int) -> KRecoveryResult:
        """Recover a candidate secret using a single ``K``.

        Args:
            K: The probe multiplier.

        Returns:
            A :class:`KRecoveryResult`.
        """
        reduced = int(K) % self.q
        probes = build_probes(self.n, K, self.q)
        matrix = np.stack([p.vector for p in probes])
        predictions, decoded_ok, tokens = self.predict(matrix)
        bits, scores = self._decide(predictions, decoded_ok, reduced)

        to_zero = ring_distance(predictions, 0, self.q)
        to_k = ring_distance(predictions, reduced, self.q)
        outcomes = [
            CoordinateOutcome(
                coordinate=probe.coordinate,
                K=int(K),
                reduced_K=reduced,
                predicted_tokens=tokens[index].tolist(),
                decoded_b=int(predictions[index]),
                decoded=bool(decoded_ok[index]),
                distance_to_zero=int(to_zero[index]),
                distance_to_K=int(to_k[index]),
                score=float(scores[index]),
                decision=int(bits[index]),
            )
            for index, probe in enumerate(probes)
        ]
        return KRecoveryResult(
            K=int(K),
            reduced_K=reduced,
            separation=probe_separation(K, self.q),
            method=self.method,
            candidate=bits,
            outcomes=outcomes,
            decode_failure_rate=float((~decoded_ok).mean()),
        )

    def recover(
        self,
        k_values: Sequence[int],
        selection_rule: str = "aggregate",
        min_separation: Optional[int] = None,
        sigma: Optional[float] = None,
    ) -> DirectRecoveryReport:
        """Run the full sweep and combine the results.

        Votes are weighted by each probe's separation, because a ``K`` close to
        0 or ``q`` carries almost no information and should not count equally
        with one near ``q/2``.  The weighting uses only public quantities.

        Why ``best_margin`` needs a guard
        ---------------------------------
        Phase 26 exposed a real failure.  At ``K = 1`` the two hypotheses are
        ``b ~ 0`` and ``b ~ 1``: adjacent.  A model that confidently answers a
        constant ``0`` is then at distance 0 from one anchor and 1 from the
        other, so ``|score| = 1`` for every coordinate and the mean margin is
        the **maximum possible**.  The margin rule therefore rewards the least
        informative probes precisely because they are least informative, and at
        n=20 it selected a negative control.  ``min_separation`` excludes such
        probes from *selection*; they are still measured and still vote, with
        the negligible weight their separation earns them.

        The guard is derived from the problem rather than picked: a probe can
        only discriminate if its two hypotheses are further apart than the error
        scale, so the default is ``floor(sigma) + 1``.

        Args:
            k_values: The multipliers to probe.
            selection_rule: ``"aggregate"`` (default, robust) or
                ``"best_margin"`` (diagnostic).
            min_separation: Smallest ring separation a ``K`` may have and still
                be eligible for ``best_margin`` selection.  ``None`` derives it
                from ``sigma`` when given, else falls back to 1.
            sigma: Configured error scale, used only to derive the default
                guard.  Never used to decide a bit.

        Returns:
            A :class:`DirectRecoveryReport`.

        Raises:
            ValueError: If ``k_values`` is empty or the rule is unknown.
        """
        if selection_rule not in SELECTION_RULES:
            raise ValueError(
                f"selection_rule must be one of {list(SELECTION_RULES)}, "
                f"got {selection_rule!r}.")
        values = [int(k) for k in k_values]
        if not values:
            raise ValueError("at least one K value is required.")

        if min_separation is None:
            min_separation = (int(math.floor(float(sigma))) + 1
                              if sigma is not None else 1)
        min_separation = max(1, int(min_separation))

        per_k = [self.recover_for_k(k) for k in values]
        aggregate = np.zeros(self.n, dtype=np.float64)
        for result in per_k:
            weight = result.separation / max(self.q / 2.0, 1.0)
            aggregate += weight * np.array([o.score for o in result.outcomes])

        eligible = [r for r in per_k if r.separation >= min_separation]
        excluded = [r.K for r in per_k if r.separation < min_separation]
        selected = max(eligible, key=lambda r: r.mean_margin).K if eligible else None

        return DirectRecoveryReport(
            n=self.n,
            q=self.q,
            method=self.method,
            per_k=per_k,
            aggregate_candidate=(aggregate > 0).astype(np.int64),
            aggregate_scores=aggregate,
            selected_K=selected,
            selection_rule=selection_rule,
            min_separation=min_separation,
            excluded_k=excluded,
        )
