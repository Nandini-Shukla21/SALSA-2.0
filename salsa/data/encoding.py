"""Integer <-> token-sequence encoding for the SALSA ``a -> b`` problem.

A sequence model cannot consume integers directly, so each element of ``Z_q``
is written as a sequence of base-``B`` digits, and a full LWE instance becomes a
token sequence::

    input   <bos> a0 a0 <sep> a1 a1 <sep> ... <sep> a(n-1) a(n-1) <eos>
    output  <bos> b  b  <eos>

with each coordinate occupying exactly ``width = ceil(log_B q)`` digit tokens
(``width = 2`` for ``q = 251`` in base 81).  The SALSA paper reports that the
representation base materially changes learning difficulty, so the base is a
configuration value here rather than a constant: representation is meant to
become an experimental variable in later phases.

Three layers, deliberately separable:

* :class:`Vocabulary` -- the explicit token/id table.  Special tokens are
  first-class objects with fixed ids, never matched by string surgery.
* :class:`IntegerEncoder` -- one integer to a digit-token list, and back.
* :class:`LatticeCodec` -- the ``a -> b`` problem: input/output sequences,
  vectorised batch encoding, and sequence-length accounting.

Correctness properties enforced by the tests:

* ``decode(encode(x)) == x`` for every ``x`` in ``Z_q``, in every supported
  base and representation;
* the readable scalar path and the fast vectorised batch path agree exactly;
* malformed sequences raise :class:`EncodingError` naming the position, rather
  than silently decoding to a wrong integer.

This module is pure NumPy: it never imports torch and is independent of the
neural model.
"""

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "BOS",
    "EOS",
    "EncodingError",
    "IntegerEncoder",
    "LatticeCodec",
    "NEG",
    "PAD",
    "POS",
    "SEP",
    "SequenceLengths",
    "Vocabulary",
    "benchmark_codec",
    "decode_integer",
    "decode_lattice_input",
    "decode_lattice_output",
    "decode_vector",
    "digit_width",
    "encode_integer",
    "encode_lattice_input",
    "encode_lattice_output",
    "encode_vector",
]

#: Special tokens.  Their ids are fixed by :class:`Vocabulary` so that a
#: checkpoint trained with one vocabulary keeps meaning the same thing.
PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
SEP = "<sep>"
POS = "<+>"
NEG = "<->"

SPECIAL_TOKENS = (PAD, BOS, EOS, SEP)
SIGN_TOKENS = (POS, NEG)

#: Integer dtype used for token id arrays.
ID_DTYPE = np.int64


class EncodingError(ValueError):
    """Raised for any malformed value, token or token sequence."""


# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #
class Vocabulary:
    """An explicit, ordered token/id table.

    Special tokens always occupy the same ids (``<pad>`` = 0, ``<bos>`` = 1,
    ``<eos>`` = 2, ``<sep>`` = 3) followed by the sign tokens when they are
    used, then the digit tokens in ascending numeric order.  Fixing the layout
    keeps a saved model's embedding table meaningful across runs.

    Example:
        >>> vocab = Vocabulary(digit_tokens=["0", "1"])
        >>> vocab.pad_id, vocab.bos_id, vocab.eos_id, vocab.sep_id
        (0, 1, 2, 3)
        >>> vocab.encode(["<bos>", "1", "0", "<eos>"])
        [1, 5, 4, 2]
    """

    def __init__(
        self, digit_tokens: Sequence[str], include_sign_tokens: bool = False
    ) -> None:
        """Build a vocabulary from a list of digit tokens.

        Args:
            digit_tokens: Digit token strings, e.g. ``["0", "1", ..., "80"]``.
            include_sign_tokens: Reserve ``<+>`` / ``<->`` for signed encoders.

        Raises:
            EncodingError: If ``digit_tokens`` is empty or contains duplicates,
                or collides with a special token.
        """
        digits = list(dict.fromkeys(str(t) for t in digit_tokens))
        if not digits:
            raise EncodingError("A vocabulary needs at least one digit token.")
        if len(digits) != len(list(digit_tokens)):
            raise EncodingError("digit_tokens contains duplicates.")
        clashes = set(digits) & set(SPECIAL_TOKENS + SIGN_TOKENS)
        if clashes:
            raise EncodingError(f"Digit tokens collide with special tokens: {clashes}.")

        tokens: List[str] = list(SPECIAL_TOKENS)
        if include_sign_tokens:
            tokens.extend(SIGN_TOKENS)
        tokens.extend(digits)

        self.tokens: List[str] = tokens
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(tokens)}
        self.id_to_token: Dict[int, str] = {i: t for i, t in enumerate(tokens)}
        self.digit_tokens: List[str] = digits
        self.has_sign_tokens = bool(include_sign_tokens)

    # -- basic properties --------------------------------------------------- #
    @property
    def size(self) -> int:
        """Number of tokens in the vocabulary."""
        return len(self.tokens)

    def __len__(self) -> int:
        """Number of tokens, so ``len(vocab)`` works."""
        return len(self.tokens)

    def __contains__(self, token: str) -> bool:
        """True when ``token`` is part of this vocabulary."""
        return token in self.token_to_id

    @property
    def pad_id(self) -> int:
        """Id of the padding token."""
        return self.token_to_id[PAD]

    @property
    def bos_id(self) -> int:
        """Id of the begin-of-sequence token."""
        return self.token_to_id[BOS]

    @property
    def eos_id(self) -> int:
        """Id of the end-of-sequence token."""
        return self.token_to_id[EOS]

    @property
    def sep_id(self) -> int:
        """Id of the coordinate separator token."""
        return self.token_to_id[SEP]

    # -- conversion --------------------------------------------------------- #
    def token_id(self, token: str) -> int:
        """Return the id of one token.

        Raises:
            EncodingError: If the token is not in the vocabulary.
        """
        try:
            return self.token_to_id[token]
        except KeyError:
            raise EncodingError(f"Token {token!r} is not in the vocabulary.") from None

    def token_at(self, index: int) -> str:
        """Return the token with id ``index``.

        Raises:
            EncodingError: If the id is out of range.
        """
        try:
            return self.id_to_token[int(index)]
        except KeyError:
            raise EncodingError(
                f"Token id {index} is outside the vocabulary (size {self.size})."
            ) from None

    def encode(self, tokens: Iterable[str]) -> List[int]:
        """Map a token sequence to ids."""
        return [self.token_id(t) for t in tokens]

    def decode(self, ids: Iterable[int]) -> List[str]:
        """Map ids back to tokens."""
        return [self.token_at(i) for i in ids]

    def digit_id_lookup(self, digit_values: Sequence[int]) -> np.ndarray:
        """Return an array mapping a digit *value* to its token id.

        Used by the vectorised batch path.  ``digit_values[k]`` is the numeric
        value of ``self.digit_tokens[k]``.

        Args:
            digit_values: Numeric values of the digit tokens, in token order.

        Returns:
            An ``int64`` array ``lookup`` such that ``lookup[value - offset]``
            is the token id, where ``offset = min(digit_values)``.
        """
        values = np.asarray(digit_values, dtype=ID_DTYPE)
        offset = int(values.min())
        lookup = np.full(int(values.max()) - offset + 1, -1, dtype=ID_DTYPE)
        for token, value in zip(self.digit_tokens, values.tolist()):
            lookup[value - offset] = self.token_to_id[token]
        return lookup

    def __repr__(self) -> str:
        """Short description of the vocabulary."""
        return (
            f"Vocabulary(size={self.size}, specials={len(SPECIAL_TOKENS)}, "
            f"digits={len(self.digit_tokens)})"
        )


# --------------------------------------------------------------------------- #
# Integer encoding
# --------------------------------------------------------------------------- #
def digit_width(base: int, max_value: int) -> int:
    """Return the number of base-``B`` digits needed to write ``max_value``.

    Args:
        base: Positional base, at least 2.
        max_value: Largest non-negative value that must be representable.

    Returns:
        The smallest ``w`` with ``base ** w > max_value`` (at least 1).

    Raises:
        EncodingError: If ``base < 2`` or ``max_value < 0``.

    Example:
        >>> digit_width(81, 250), digit_width(7, 250), digit_width(2, 250)
        (2, 3, 8)
    """
    if base < 2:
        raise EncodingError(f"base must be >= 2, got {base}.")
    if max_value < 0:
        raise EncodingError(f"max_value must be >= 0, got {max_value}.")
    width, limit = 1, base
    while limit <= max_value:
        limit *= base
        width += 1
    return width


class IntegerEncoder:
    """Encodes a single integer as a fixed-width base-``B`` digit sequence.

    Three representations are supported, all exactly invertible:

    ``standard`` (default)
        Value ``v`` in ``[0, q)`` written as ``width`` digits in ``[0, B)``,
        most significant first.  This is the SALSA representation.
    ``balanced``
        The value is first centred into ``(-q/2, q/2]`` and written with digits
        in ``[-(B-1)/2, (B-1)/2]`` (odd bases only).  A different token
        distribution over the same information, kept available because the
        paper reports representation affects learnability.
    ``signed``
        An explicit ``<+>`` / ``<->`` token followed by the magnitude digits.
        Used for values that are genuinely signed (e.g. an error vector) rather
        than elements of ``Z_q``.

    Example:
        >>> enc = IntegerEncoder(base=81, modulus=251)
        >>> enc.width
        2
        >>> enc.encode(250)
        ['3', '7']
        >>> enc.decode(['3', '7'])
        250
    """

    def __init__(
        self,
        base: int,
        modulus: Optional[int] = None,
        max_value: Optional[int] = None,
        width: Optional[int] = None,
        fixed_width: bool = True,
        balanced: bool = False,
        signed: bool = False,
        digit_order: str = "msb_first",
    ) -> None:
        """Create an integer encoder.

        Args:
            base: Positional base ``B >= 2``.
            modulus: Modulus ``q``.  When given, encoded values are reduced into
                ``[0, q)`` and decoded values are validated against it.
            max_value: Largest magnitude to support when there is no modulus.
            width: Explicit digit count.  Defaults to the smallest width that
                covers ``modulus - 1`` (or ``max_value``).
            fixed_width: Pad every value to ``width`` digits.  Required for the
                aligned, separator-free parsing a sequence model needs.
            balanced: Use the balanced (signed-digit) representation.
            signed: Emit an explicit sign token before the digits.
            digit_order: ``"msb_first"`` (default) or ``"lsb_first"``.

        Raises:
            EncodingError: On an invalid base, width, order, or a balanced
                representation with an even base.
        """
        if base < 2:
            raise EncodingError(f"base must be >= 2, got {base}.")
        if digit_order not in ("msb_first", "lsb_first"):
            raise EncodingError(
                f"digit_order must be 'msb_first' or 'lsb_first', got {digit_order!r}."
            )
        if balanced and base % 2 == 0:
            raise EncodingError(
                f"The balanced representation needs an odd base, got {base}."
            )
        if balanced and signed:
            raise EncodingError(
                "balanced and signed are mutually exclusive: balanced digits already "
                "carry the sign."
            )
        if modulus is not None and modulus < 2:
            raise EncodingError(f"modulus must be >= 2, got {modulus}.")
        if not fixed_width and balanced:
            raise EncodingError("balanced digits require fixed_width=True.")

        self.base = int(base)
        self.modulus = None if modulus is None else int(modulus)
        self.fixed_width = bool(fixed_width)
        self.balanced = bool(balanced)
        self.signed = bool(signed)
        self.digit_order = digit_order

        if width is not None:
            if width < 1:
                raise EncodingError(f"width must be >= 1, got {width}.")
            self.width = int(width)
        elif self.modulus is not None:
            self.width = digit_width(self.base, self.modulus - 1)
        elif max_value is not None:
            self.width = digit_width(self.base, int(abs(max_value)))
        else:
            raise EncodingError(
                "Provide one of modulus, max_value or width to size the encoder."
            )

        if self.modulus is not None and self.base ** self.width < self.modulus:
            raise EncodingError(
                f"width={self.width} in base {self.base} cannot represent every "
                f"value below q={self.modulus}."
            )

        # Digit alphabet, in ascending numeric order.
        if self.balanced:
            half = (self.base - 1) // 2
            self.digit_values: List[int] = list(range(-half, half + 1))
        else:
            self.digit_values = list(range(self.base))
        self.digit_tokens: List[str] = [str(v) for v in self.digit_values]

    # -- capacity ----------------------------------------------------------- #
    @property
    def capacity(self) -> int:
        """Number of distinct values ``width`` digits can express."""
        return self.base ** self.width

    @property
    def token_length(self) -> int:
        """Tokens emitted per value, including any sign token."""
        return self.width + (1 if self.signed else 0)

    # -- scalar encoding ---------------------------------------------------- #
    def _digits(self, value: int) -> List[int]:
        """Return the digit values of ``value``, least significant first."""
        if self.balanced:
            centered = self._center(value)
            digits, remaining = [], int(centered)
            half = self.base // 2
            for _ in range(self.width):
                residue = remaining % self.base
                if residue > half:
                    residue -= self.base
                digits.append(int(residue))
                remaining = (remaining - residue) // self.base
            if remaining != 0:
                raise EncodingError(
                    f"Value {value} does not fit in {self.width} balanced base-"
                    f"{self.base} digits."
                )
            return digits

        magnitude = abs(int(value)) if self.signed else int(value)
        digits = []
        remaining = magnitude
        while remaining > 0:
            digits.append(remaining % self.base)
            remaining //= self.base
        if not digits:
            digits = [0]
        if self.fixed_width:
            if len(digits) > self.width:
                raise EncodingError(
                    f"Value {value} needs {len(digits)} base-{self.base} digits but "
                    f"the encoder is {self.width} wide."
                )
            digits.extend([0] * (self.width - len(digits)))
        return digits

    def _center(self, value: int) -> int:
        """Map a value into the centred range ``(-q/2, q/2]``."""
        if self.modulus is None:
            return int(value)
        reduced = int(value) % self.modulus
        return reduced - self.modulus if reduced > self.modulus // 2 else reduced

    def prepare(self, value: int) -> int:
        """Reduce ``value`` into the encoder's domain and validate it.

        Args:
            value: Any Python integer.

        Returns:
            The value actually encoded: reduced mod ``q`` when a modulus is set,
            otherwise ``value`` itself.

        Raises:
            EncodingError: If the value is out of range for this encoder.
        """
        number = int(value)
        if self.modulus is not None:
            return number % self.modulus
        if not self.signed and number < 0:
            raise EncodingError(
                f"Negative value {number} needs signed=True or a modulus."
            )
        if abs(number) >= self.capacity:
            raise EncodingError(
                f"Value {number} exceeds the capacity {self.capacity} of a "
                f"{self.width}-digit base-{self.base} encoder."
            )
        return number

    def encode(self, value: int) -> List[str]:
        """Encode one integer as a list of digit tokens.

        Args:
            value: The integer to encode.  Reduced mod ``q`` when the encoder
                has a modulus.

        Returns:
            A list of token strings, ``width`` long (plus a sign token when
            ``signed``).

        Raises:
            EncodingError: If the value cannot be represented.
        """
        number = self.prepare(value)
        digits = self._digits(number)
        if self.digit_order == "msb_first":
            digits = list(reversed(digits))
        tokens = [str(d) for d in digits]
        if self.signed:
            tokens = [NEG if number < 0 else POS] + tokens
        return tokens

    def decode(self, tokens: Sequence[str], strict: bool = True) -> int:
        """Decode a digit-token sequence back to an integer.

        Args:
            tokens: The tokens produced by :meth:`encode`.
            strict: When True (default) a value outside ``[0, q)`` is an error.
                When False the value is reduced mod ``q`` instead -- needed when
                decoding model output, which can express values above ``q``.

        Returns:
            The decoded integer, in ``[0, q)`` when a modulus is set.

        Raises:
            EncodingError: On an empty, truncated or malformed sequence, an
                unknown digit, or (when ``strict``) an out-of-range value.
        """
        items = list(tokens)
        if not items:
            raise EncodingError("Cannot decode an empty token sequence.")

        sign = 1
        if self.signed:
            head = items.pop(0)
            if head not in SIGN_TOKENS:
                raise EncodingError(
                    f"Expected a sign token at position 0, found {head!r}."
                )
            sign = -1 if head == NEG else 1
        elif items[0] in SIGN_TOKENS:
            raise EncodingError(
                f"Unexpected sign token {items[0]!r}: this encoder is unsigned."
            )

        if self.fixed_width and len(items) != self.width:
            raise EncodingError(
                f"Expected exactly {self.width} digit tokens, got {len(items)} "
                f"({'truncated' if len(items) < self.width else 'too many'})."
            )
        if not items:
            raise EncodingError("Cannot decode a sequence with no digit tokens.")

        allowed = set(self.digit_tokens)
        digits: List[int] = []
        for position, token in enumerate(items):
            if token in SPECIAL_TOKENS or token in SIGN_TOKENS:
                raise EncodingError(
                    f"Special token {token!r} at position {position} where a digit "
                    "was expected."
                )
            if token not in allowed:
                raise EncodingError(
                    f"Invalid digit {token!r} at position {position} for base "
                    f"{self.base}"
                    + (" (balanced)" if self.balanced else "")
                    + "."
                )
            digits.append(int(token))

        if self.digit_order == "msb_first":
            digits = list(reversed(digits))

        value = 0
        for power, digit in enumerate(digits):
            value += digit * (self.base ** power)
        value *= sign

        if self.modulus is not None:
            if self.balanced:
                return int(value % self.modulus)
            if not 0 <= value < self.modulus:
                if strict:
                    raise EncodingError(
                        f"Decoded value {value} is outside [0, {self.modulus}); the "
                        "sequence does not represent an element of Z_q."
                    )
                return int(value % self.modulus)
        return int(value)

    # -- vectorised helpers ------------------------------------------------- #
    def digits_array(self, values: np.ndarray) -> np.ndarray:
        """Vectorised digit extraction.

        Args:
            values: Integer array of any shape.

        Returns:
            An array of shape ``values.shape + (width,)`` holding digit values
            in the encoder's digit order.

        Raises:
            EncodingError: If a value does not fit in ``width`` digits.
        """
        numbers = np.asarray(values, dtype=np.int64)
        if self.modulus is not None:
            numbers = numbers % self.modulus
            if self.balanced:
                numbers = np.where(
                    numbers > self.modulus // 2, numbers - self.modulus, numbers
                )
        elif not self.signed and numbers.min(initial=0) < 0:
            raise EncodingError("Negative values need signed=True or a modulus.")

        remaining = numbers.astype(np.int64, copy=True)
        digits = np.empty(numbers.shape + (self.width,), dtype=np.int64)
        half = self.base // 2
        for position in range(self.width):
            residue = remaining % self.base
            if self.balanced:
                residue = np.where(residue > half, residue - self.base, residue)
            digits[..., position] = residue
            remaining = (remaining - residue) // self.base
        if np.any(remaining != 0):
            raise EncodingError(
                f"Some values do not fit in {self.width} base-{self.base} digits."
            )
        if self.digit_order == "msb_first":
            digits = digits[..., ::-1]
        return digits

    def values_from_digits(self, digits: np.ndarray, reduce: bool = True) -> np.ndarray:
        """Inverse of :meth:`digits_array`.

        Args:
            digits: Array of shape ``(..., width)`` of digit values.
            reduce: Reduce the result mod ``q``.  Pass False to see the raw
                value the digits express, which is what a strict range check
                needs -- ``width`` digits can encode more than ``q`` values.

        Returns:
            An ``int64`` array of shape ``digits.shape[:-1]``.
        """
        block = np.asarray(digits, dtype=np.int64)
        if block.shape[-1] != self.width:
            raise EncodingError(
                f"Expected {self.width} digits in the last axis, got {block.shape[-1]}."
            )
        if self.digit_order == "msb_first":
            block = block[..., ::-1]
        powers = (self.base ** np.arange(self.width, dtype=np.int64)).astype(np.int64)
        values = (block * powers).sum(axis=-1)
        if reduce and self.modulus is not None:
            values = values % self.modulus
        return values.astype(np.int64, copy=False)

    def __repr__(self) -> str:
        """Short description of the encoder."""
        mode = "balanced" if self.balanced else ("signed" if self.signed else "standard")
        return (
            f"IntegerEncoder(base={self.base}, q={self.modulus}, width={self.width}, "
            f"mode={mode}, order={self.digit_order})"
        )


# --------------------------------------------------------------------------- #
# Free functions (explicit, no hidden global state)
# --------------------------------------------------------------------------- #
def encode_integer(value: int, encoder: IntegerEncoder) -> List[str]:
    """Encode one integer with ``encoder``."""
    return encoder.encode(value)


def decode_integer(
    tokens: Sequence[str], encoder: IntegerEncoder, strict: bool = True
) -> int:
    """Decode one integer with ``encoder``."""
    return encoder.decode(tokens, strict=strict)


def encode_vector(
    values: Sequence[int],
    encoder: IntegerEncoder,
    separator: bool = True,
) -> List[str]:
    """Encode a vector of integers as one token sequence.

    Args:
        values: The integers to encode.
        encoder: The per-integer encoder.
        separator: Insert ``<sep>`` *between* consecutive coordinates.

    Returns:
        The concatenated token list.

    Raises:
        EncodingError: On an empty vector or an unrepresentable value.
    """
    items = [int(v) for v in np.asarray(values).reshape(-1).tolist()]
    if not items:
        raise EncodingError("Cannot encode an empty vector.")
    tokens: List[str] = []
    for index, value in enumerate(items):
        if separator and index:
            tokens.append(SEP)
        tokens.extend(encoder.encode(value))
    return tokens


def decode_vector(
    tokens: Sequence[str],
    encoder: IntegerEncoder,
    separator: bool = True,
    length: Optional[int] = None,
    strict: bool = True,
) -> np.ndarray:
    """Decode a token sequence back into a vector of integers.

    Args:
        tokens: Tokens produced by :func:`encode_vector`.
        encoder: The per-integer encoder.
        separator: Whether ``<sep>`` tokens are expected between coordinates.
        length: Expected number of coordinates; checked when given.
        strict: Passed through to :meth:`IntegerEncoder.decode`.

    Returns:
        An ``int64`` array of the decoded coordinates.

    Raises:
        EncodingError: On a misplaced or missing separator, a wrong coordinate
            count, a truncated group, or an invalid digit.
    """
    items = list(tokens)
    if not items:
        raise EncodingError("Cannot decode an empty token sequence.")

    step = encoder.token_length
    groups: List[List[str]] = []

    if separator:
        current: List[str] = []
        for position, token in enumerate(items):
            if token == SEP:
                if not current:
                    raise EncodingError(
                        f"Separator at position {position} has no preceding value."
                    )
                groups.append(current)
                current = []
            else:
                current.append(token)
        if not current:
            raise EncodingError("Token sequence ends with a separator.")
        groups.append(current)
    else:
        if len(items) % step != 0:
            raise EncodingError(
                f"Token count {len(items)} is not a multiple of the value width "
                f"{step}; the sequence is truncated or malformed."
            )
        groups = [items[i : i + step] for i in range(0, len(items), step)]

    if length is not None and len(groups) != int(length):
        raise EncodingError(
            f"Expected {int(length)} coordinates, found {len(groups)}."
        )

    values = [encoder.decode(group, strict=strict) for group in groups]
    return np.asarray(values, dtype=np.int64)


# --------------------------------------------------------------------------- #
# Lattice codec
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SequenceLengths:
    """Token accounting for one encoded LWE instance.

    Recorded with every experiment: sequence length drives both compute cost
    and memory, so it is one of the efficiency metrics of this project.

    Attributes:
        input_tokens: Length of the encoder-side sequence.
        output_tokens: Length of the decoder-side sequence.
        input_digits: Digit tokens per coordinate of ``a``.
        output_digits: Digit tokens for ``b``.
        separators: Number of ``<sep>`` tokens in the input.
        specials: Number of ``<bos>``/``<eos>`` tokens across both sequences.
    """

    input_tokens: int
    output_tokens: int
    input_digits: int
    output_digits: int
    separators: int
    specials: int

    @property
    def total_tokens(self) -> int:
        """Input plus output tokens for one instance."""
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, int]:
        """Return a serialisable view for run metadata."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "input_digits_per_coordinate": self.input_digits,
            "output_digits": self.output_digits,
            "separators": self.separators,
            "specials": self.specials,
        }


class LatticeCodec:
    """Encodes the SALSA ``a -> b`` problem as model input/output sequences.

    Layout (with ``include_bos_eos`` and ``separator`` on, the defaults)::

        input   <bos> a0.. <sep> a1.. <sep> ... <sep> a(n-1).. <eos>
        output  <bos> b.. <eos>

    Example:
        >>> codec = LatticeCodec(n=3, q=251, base=81)
        >>> codec.encode_input([1, 80, 250])
        ['<bos>', '0', '1', '<sep>', '0', '80', '<sep>', '3', '7', '<eos>']
        >>> codec.decode_input(codec.encode_input([1, 80, 250])).tolist()
        [1, 80, 250]
    """

    def __init__(
        self,
        n: int,
        q: int,
        base: int = 81,
        input_base: Optional[int] = None,
        output_base: Optional[int] = None,
        separator: bool = True,
        include_bos_eos: bool = True,
        balanced: bool = False,
        digit_order: str = "msb_first",
        fixed_width: bool = True,
    ) -> None:
        """Create a codec for an ``n``-dimensional instance over ``Z_q``.

        Args:
            n: Lattice dimension (number of coordinates in ``a``).
            q: Modulus.
            base: Default digit base for both sides.
            input_base: Override for the encoder side.
            output_base: Override for the decoder side.
            separator: Emit ``<sep>`` between coordinates of ``a``.
            include_bos_eos: Wrap sequences in ``<bos>`` / ``<eos>``.
            balanced: Use the balanced digit representation on both sides.
            digit_order: ``"msb_first"`` or ``"lsb_first"``.
            fixed_width: Pad every value to a fixed digit count.  Variable width
                requires separators and disables the fixed-stride batch path.

        Raises:
            EncodingError: On invalid dimensions, bases, modulus, or a
                variable-width layout without separators.
        """
        if n < 1:
            raise EncodingError(f"n must be >= 1, got {n}.")
        if q < 2:
            raise EncodingError(f"q must be >= 2, got {q}.")
        if not fixed_width and not separator:
            raise EncodingError(
                "fixed_width=False requires separator=True: without either a fixed "
                "stride or an explicit separator, coordinate boundaries are ambiguous."
            )

        self.n = int(n)
        self.q = int(q)
        self.separator = bool(separator)
        self.include_bos_eos = bool(include_bos_eos)
        self.fixed_width = bool(fixed_width)

        self.input_encoder = IntegerEncoder(
            base=int(input_base if input_base is not None else base),
            modulus=self.q,
            balanced=balanced,
            digit_order=digit_order,
            fixed_width=self.fixed_width,
        )
        self.output_encoder = IntegerEncoder(
            base=int(output_base if output_base is not None else base),
            modulus=self.q,
            balanced=balanced,
            digit_order=digit_order,
            fixed_width=self.fixed_width,
        )

        digit_tokens = list(
            dict.fromkeys(self.input_encoder.digit_tokens + self.output_encoder.digit_tokens)
        )
        digit_tokens.sort(key=int)
        self.vocabulary = Vocabulary(digit_tokens)

        # Lookup tables and column indices for the vectorised batch path.
        self._input_lookup = self.vocabulary.digit_id_lookup(
            [int(t) for t in self.vocabulary.digit_tokens]
        )
        self._lookup_offset = min(int(t) for t in self.vocabulary.digit_tokens)

    # -- construction ------------------------------------------------------- #
    @classmethod
    def from_config(cls, config) -> "LatticeCodec":
        """Build a codec from a :class:`~salsa.utils.config.Config`."""
        encoding = config.encoding
        return cls(
            n=int(config.lwe.n),
            q=int(config.lwe.q),
            base=int(encoding.base),
            input_base=encoding.input_base,
            output_base=encoding.output_base,
            separator=bool(encoding.separator),
            include_bos_eos=bool(encoding.include_bos_eos),
            balanced=bool(encoding.balanced),
            digit_order=str(encoding.digit_order),
            fixed_width=bool(encoding.fixed_width),
        )

    # -- lengths ------------------------------------------------------------ #
    @property
    def input_length(self) -> int:
        """Number of tokens in one encoded ``a``.

        Raises:
            EncodingError: If the codec is variable-width, where no single
                sequence length exists.
        """
        self._require_fixed_width("input_length")
        digits = self.n * self.input_encoder.token_length
        separators = (self.n - 1) if self.separator else 0
        specials = 2 if self.include_bos_eos else 0
        return digits + separators + specials

    @property
    def output_length(self) -> int:
        """Number of tokens in one encoded ``b``.

        Raises:
            EncodingError: If the codec is variable-width.
        """
        self._require_fixed_width("output_length")
        specials = 2 if self.include_bos_eos else 0
        return self.output_encoder.token_length + specials

    def lengths(self) -> SequenceLengths:
        """Return the full token accounting for one instance."""
        return SequenceLengths(
            input_tokens=self.input_length,
            output_tokens=self.output_length,
            input_digits=self.input_encoder.token_length,
            output_digits=self.output_encoder.token_length,
            separators=(self.n - 1) if self.separator else 0,
            specials=(4 if self.include_bos_eos else 0),
        )

    def describe(self) -> Dict[str, object]:
        """Return a serialisable description for run metadata."""
        description: Dict[str, object] = {
            "n": self.n,
            "q": self.q,
            "input_base": self.input_encoder.base,
            "output_base": self.output_encoder.base,
            "balanced": self.input_encoder.balanced,
            "digit_order": self.input_encoder.digit_order,
            "separator": self.separator,
            "include_bos_eos": self.include_bos_eos,
            "fixed_width": self.fixed_width,
            "vocab_size": self.vocabulary.size,
        }
        description.update(self.lengths().to_dict())
        return description

    # -- token-level encoding ----------------------------------------------- #
    def encode_input(self, a: Sequence[int]) -> List[str]:
        """Encode one ``a`` vector into the model's input token sequence.

        Args:
            a: Vector of ``n`` coordinates in ``Z_q``.

        Returns:
            The token list.

        Raises:
            EncodingError: If ``a`` has the wrong length or a bad value.
        """
        values = np.asarray(a, dtype=np.int64).reshape(-1)
        if values.size != self.n:
            raise EncodingError(
                f"Expected a vector of length {self.n}, got {values.size}."
            )
        tokens = encode_vector(values, self.input_encoder, separator=self.separator)
        return self._wrap(tokens)

    def decode_input(self, tokens: Sequence[str], strict: bool = True) -> np.ndarray:
        """Decode an input token sequence back into the ``a`` vector."""
        body = self._unwrap(tokens)
        return decode_vector(
            body,
            self.input_encoder,
            separator=self.separator,
            length=self.n,
            strict=strict,
        )

    def encode_output(self, b: int) -> List[str]:
        """Encode one ``b`` value into the model's output token sequence."""
        return self._wrap(self.output_encoder.encode(int(b)))

    def decode_output(self, tokens: Sequence[str], strict: bool = True) -> int:
        """Decode an output token sequence back into ``b``.

        Args:
            tokens: Output tokens, with or without ``<bos>``/``<eos>``.
            strict: When False, a value the digits can express but ``Z_q``
                cannot is reduced mod ``q`` instead of raising.  Model output
                needs the lenient path.

        Returns:
            The decoded ``b`` in ``[0, q)``.
        """
        body = self._unwrap(tokens)
        return self.output_encoder.decode(body, strict=strict)

    def encode_pair(self, a: Sequence[int], b: int) -> Tuple[List[str], List[str]]:
        """Encode one training pair ``(a, b)``."""
        return self.encode_input(a), self.encode_output(b)

    # -- id-level encoding -------------------------------------------------- #
    def encode_input_ids(self, a: Sequence[int]) -> List[int]:
        """Encode ``a`` directly into vocabulary ids."""
        return self.vocabulary.encode(self.encode_input(a))

    def encode_output_ids(self, b: int) -> List[int]:
        """Encode ``b`` directly into vocabulary ids."""
        return self.vocabulary.encode(self.encode_output(b))

    def decode_input_ids(self, ids: Sequence[int], strict: bool = True) -> np.ndarray:
        """Decode vocabulary ids back into the ``a`` vector."""
        return self.decode_input(self.vocabulary.decode(ids), strict=strict)

    def decode_output_ids(self, ids: Sequence[int], strict: bool = True) -> int:
        """Decode vocabulary ids back into ``b``."""
        return self.decode_output(self.vocabulary.decode(ids), strict=strict)

    # -- batch encoding (vectorised) ---------------------------------------- #
    def encode_batch(
        self, A: np.ndarray, b: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Encode a whole batch of instances into id arrays.

        This is the fast path used during training.  It is a vectorised
        equivalent of calling :meth:`encode_input_ids` row by row, and a test
        asserts the two agree exactly.

        Args:
            A: Integer array of shape ``(m, n)``.
            b: Optional integer array of shape ``(m,)``.

        Returns:
            ``(input_ids, output_ids)`` as ``int64`` arrays of shape
            ``(m, input_length)`` and ``(m, output_length)``.  ``output_ids`` is
            None when ``b`` is None.

        Raises:
            EncodingError: If the shapes do not match the codec.
        """
        matrix = np.asarray(A, dtype=np.int64)
        if matrix.ndim != 2 or matrix.shape[1] != self.n:
            raise EncodingError(
                f"A must have shape (m, {self.n}), got {tuple(matrix.shape)}."
            )
        self._require_fixed_width("encode_batch")
        rows = matrix.shape[0]

        digits = self.input_encoder.digits_array(matrix)  # (m, n, width)
        digit_ids = self._to_ids(digits).reshape(rows, -1)

        input_ids = np.full((rows, self.input_length), self.vocabulary.sep_id, dtype=ID_DTYPE)
        input_ids[:, self._input_digit_columns()] = digit_ids
        if self.include_bos_eos:
            input_ids[:, 0] = self.vocabulary.bos_id
            input_ids[:, -1] = self.vocabulary.eos_id

        output_ids: Optional[np.ndarray] = None
        if b is not None:
            targets = np.asarray(b, dtype=np.int64).reshape(-1)
            if targets.size != rows:
                raise EncodingError(
                    f"b must have {rows} entries to match A, got {targets.size}."
                )
            out_digits = self.output_encoder.digits_array(targets)  # (m, width)
            output_ids = np.empty((rows, self.output_length), dtype=ID_DTYPE)
            offset = 1 if self.include_bos_eos else 0
            output_ids[:, offset : offset + self.output_encoder.width] = self._to_ids(
                out_digits
            )
            if self.include_bos_eos:
                output_ids[:, 0] = self.vocabulary.bos_id
                output_ids[:, -1] = self.vocabulary.eos_id

        return input_ids, output_ids

    def decode_batch_inputs(self, input_ids: np.ndarray, strict: bool = True) -> np.ndarray:
        """Decode a batch of input id sequences back into an ``(m, n)`` matrix."""
        ids = np.asarray(input_ids, dtype=np.int64)
        if ids.ndim != 2 or ids.shape[1] != self.input_length:
            raise EncodingError(
                f"input_ids must have shape (m, {self.input_length}), got "
                f"{tuple(ids.shape)}."
            )
        self._check_frame(ids)
        digit_ids = ids[:, self._input_digit_columns()]
        digits = self._from_ids(digit_ids).reshape(
            ids.shape[0], self.n, self.input_encoder.width
        )
        raw = self.input_encoder.values_from_digits(digits, reduce=False)
        if strict and not self.input_encoder.balanced:
            self._check_range(raw)
        return (raw % self.q).astype(np.int64, copy=False)

    def decode_batch_outputs(self, output_ids: np.ndarray, strict: bool = True) -> np.ndarray:
        """Decode a batch of output id sequences back into a ``b`` vector."""
        ids = np.asarray(output_ids, dtype=np.int64)
        if ids.ndim != 2 or ids.shape[1] != self.output_length:
            raise EncodingError(
                f"output_ids must have shape (m, {self.output_length}), got "
                f"{tuple(ids.shape)}."
            )
        self._check_frame(ids)
        offset = 1 if self.include_bos_eos else 0
        digit_ids = ids[:, offset : offset + self.output_encoder.width]
        digits = self._from_ids(digit_ids)
        raw = self.output_encoder.values_from_digits(digits, reduce=False)
        if strict and not self.output_encoder.balanced:
            self._check_range(raw)
        return (raw % self.q).astype(np.int64, copy=False)

    # -- internals ---------------------------------------------------------- #
    def _require_fixed_width(self, what: str) -> None:
        """Raise when ``what`` needs a fixed digit stride but the codec has none."""
        if not self.fixed_width:
            raise EncodingError(
                f"{what} requires fixed_width=True; this codec uses variable-width "
                "integers, which have no single token length."
            )

    def _wrap(self, tokens: List[str]) -> List[str]:
        """Add ``<bos>``/``<eos>`` when configured."""
        if not self.include_bos_eos:
            return tokens
        return [BOS] + tokens + [EOS]

    def _unwrap(self, tokens: Sequence[str]) -> List[str]:
        """Strip and validate ``<bos>``/``<eos>``.

        Sequences without the markers are accepted so that model output can be
        decoded either way, but a *misplaced* marker is always an error.
        """
        items = list(tokens)
        if not items:
            raise EncodingError("Cannot decode an empty token sequence.")
        if items[0] == BOS:
            items = items[1:]
        if items and items[-1] == EOS:
            items = items[:-1]
        for position, token in enumerate(items):
            if token in (BOS, EOS, PAD):
                raise EncodingError(
                    f"Misplaced special token {token!r} at position {position} of the "
                    "sequence body."
                )
        if not items:
            raise EncodingError("Token sequence contains no values.")
        return items

    def _input_digit_columns(self) -> np.ndarray:
        """Column indices of the digit tokens inside an input sequence."""
        width = self.input_encoder.width
        start = 1 if self.include_bos_eos else 0
        stride = width + (1 if self.separator else 0)
        starts = start + stride * np.arange(self.n, dtype=np.int64)
        return (starts[:, None] + np.arange(width, dtype=np.int64)[None, :]).reshape(-1)

    def _to_ids(self, digits: np.ndarray) -> np.ndarray:
        """Map digit *values* to vocabulary ids (vectorised)."""
        return self._input_lookup[np.asarray(digits, dtype=np.int64) - self._lookup_offset]

    def _from_ids(self, ids: np.ndarray) -> np.ndarray:
        """Map vocabulary ids back to digit values (vectorised)."""
        values = np.asarray(ids, dtype=np.int64)
        table = np.full(self.vocabulary.size, np.iinfo(np.int64).min, dtype=np.int64)
        for token in self.vocabulary.digit_tokens:
            table[self.vocabulary.token_to_id[token]] = int(token)
        decoded = table[values]
        if np.any(decoded == np.iinfo(np.int64).min):
            raise EncodingError(
                "Non-digit token found where a digit was expected while decoding a batch."
            )
        return decoded

    def _check_frame(self, ids: np.ndarray) -> None:
        """Validate the ``<bos>``/``<eos>`` frame of a batch of sequences."""
        if not self.include_bos_eos:
            return
        if not np.all(ids[:, 0] == self.vocabulary.bos_id):
            raise EncodingError("Every sequence must start with <bos>.")
        if not np.all(ids[:, -1] == self.vocabulary.eos_id):
            raise EncodingError("Every sequence must end with <eos>.")

    def _check_range(self, values: np.ndarray) -> None:
        """Raise when a decoded value falls outside ``[0, q)``."""
        if values.size and (values.min() < 0 or values.max() >= self.q):
            raise EncodingError(
                f"Decoded values fall outside [0, {self.q}); the sequence does not "
                "represent elements of Z_q."
            )

    def __repr__(self) -> str:
        """Short description of the codec."""
        return (
            f"LatticeCodec(n={self.n}, q={self.q}, "
            f"base={self.input_encoder.base}/{self.output_encoder.base}, "
            f"vocab={self.vocabulary.size}, "
            f"len={self.input_length}->{self.output_length})"
        )


# --------------------------------------------------------------------------- #
# Lattice-level free functions
# --------------------------------------------------------------------------- #
def encode_lattice_input(a: Sequence[int], codec: LatticeCodec) -> List[str]:
    """Encode one ``a`` vector into model input tokens."""
    return codec.encode_input(a)


def decode_lattice_input(
    tokens: Sequence[str], codec: LatticeCodec, strict: bool = True
) -> np.ndarray:
    """Decode model input tokens back into the ``a`` vector."""
    return codec.decode_input(tokens, strict=strict)


def encode_lattice_output(b: int, codec: LatticeCodec) -> List[str]:
    """Encode one ``b`` value into model output tokens."""
    return codec.encode_output(b)


def decode_lattice_output(
    tokens: Sequence[str], codec: LatticeCodec, strict: bool = True
) -> int:
    """Decode model output tokens back into ``b``."""
    return codec.decode_output(tokens, strict=strict)


# --------------------------------------------------------------------------- #
# Benchmarking
# --------------------------------------------------------------------------- #
def benchmark_codec(
    codec: LatticeCodec,
    num_instances: int = 1000,
    seed: int = 0,
    repeats: int = 3,
) -> Dict[str, float]:
    """Measure encoding throughput for this codec.

    Both paths are timed because they serve different purposes: the scalar path
    is the readable reference, the batch path is what training will use.

    Args:
        codec: The codec to benchmark.
        num_instances: Number of instances to encode.
        seed: Seed for the random instances.
        repeats: Number of timed repetitions; the fastest is reported.

    Returns:
        A dictionary with per-instance latency in microseconds, instances per
        second and tokens per second for both paths.
    """
    rng = np.random.default_rng(int(seed))
    matrix = rng.integers(0, codec.q, size=(int(num_instances), codec.n), dtype=np.int64)
    targets = rng.integers(0, codec.q, size=int(num_instances), dtype=np.int64)

    def _time(function) -> float:
        best = float("inf")
        for _ in range(max(1, int(repeats))):
            start = time.perf_counter()
            function()
            best = min(best, time.perf_counter() - start)
        return best

    batch_seconds = _time(lambda: codec.encode_batch(matrix, targets))
    scalar_seconds = _time(
        lambda: [codec.encode_input_ids(row) for row in matrix]
    )

    tokens = int(num_instances) * codec.lengths().total_tokens
    return {
        "num_instances": float(num_instances),
        "input_tokens_per_instance": float(codec.input_length),
        "batch_seconds": batch_seconds,
        "batch_us_per_instance": batch_seconds / num_instances * 1e6,
        "batch_instances_per_second": num_instances / batch_seconds,
        "batch_tokens_per_second": tokens / batch_seconds,
        "scalar_seconds": scalar_seconds,
        "scalar_us_per_instance": scalar_seconds / num_instances * 1e6,
        "scalar_instances_per_second": num_instances / scalar_seconds,
        "speedup_batch_over_scalar": scalar_seconds / batch_seconds,
    }
