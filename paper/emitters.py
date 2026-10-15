"""Shared machinery for emitting the manuscript as HTML and as LaTeX.

The manuscript body is written once, in :mod:`manuscript`, using a small
markup.  Both emitters read that one source, so the rendered PDF and the
editable LaTeX cannot drift apart in wording, numbering or citation.

Markup
------
``$...$``           inline mathematics, written in LaTeX and translated to
                    HTML by :func:`tex_math_to_html` for the HTML target.
``**bold**``        bold
``*italic*``        italic
``@cite{a,b}``      citation
``@ref{label}``     cross-reference to a figure, table, equation or section
``--`` / ``---``    en dash / em dash

Display equations carry both renderings explicitly, because the HTML
translator deliberately covers only the small subset of LaTeX that appears
inline; anything more elaborate is written out twice rather than guessed at.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Block types
# --------------------------------------------------------------------------- #


@dataclass
class Section:
    title: str
    label: str = ""
    level: int = 1


@dataclass
class Para:
    text: str
    lead: bool = False


@dataclass
class Equation:
    html: str
    tex: str
    label: str = ""


@dataclass
class Figure:
    path: str
    caption: str
    label: str
    width: str = "single"          # "single" or "double"


@dataclass
class Table:
    caption: str
    label: str
    header: Sequence[str]
    rows: Sequence[Sequence[str]]
    align: str = ""                # one character per column: l, c, r
    width: str = "single"
    note: str = ""


@dataclass
class ListBlock:
    items: Sequence[str]
    ordered: bool = False


@dataclass
class Algorithm:
    caption: str
    label: str
    lines: Sequence[Tuple[int, str]]   # (indent level, line)


@dataclass
class Abstract:
    text: str


@dataclass
class Keywords:
    text: str


@dataclass
class Reference:
    key: str
    authors: str
    title: str
    venue: str
    note: str = ""
    url: str = ""
    bibtype: str = "article"
    bibfields: Dict[str, str] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Inline mathematics: LaTeX -> HTML, for the subset the body text uses
# --------------------------------------------------------------------------- #

_GREEK = {
    r"\sigma": "σ", r"\tau": "τ", r"\delta": "δ", r"\epsilon": "ε",
    r"\chi": "χ", r"\mu": "μ", r"\rho": "ρ", r"\alpha": "α", r"\beta": "β",
    r"\lambda": "λ", r"\Sigma": "Σ", r"\pi": "π", r"\theta": "θ",
}
_SYMBOLS = {
    r"\approx": "≈", r"\le": "≤", r"\leq": "≤", r"\ge": "≥", r"\geq": "≥",
    r"\neq": "≠", r"\cdot": "·", r"\times": "×", r"\pm": "±",
    r"\to": "→", r"\rightarrow": "→", r"\leftarrow": "←", r"\in": "∈",
    r"\ll": "≪", r"\gg": "≫", r"\sum": "Σ", r"\infty": "∞",
    r"\lfloor": "⌊", r"\rfloor": "⌋", r"\lceil": "⌈", r"\rceil": "⌉",
    r"\langle": "⟨", r"\rangle": "⟩", r"\ast": "∗", r"\sim": "∼",
    r"\ldots": "…", r"\cdots": "⋯", r"\dots": "…", r"\mid": "|",
    r"\subseteq": "⊆", r"\notin": "∉", r"\equiv": "≡", r"\propto": "∝",
}
_SPACING = [r"\,", r"\;", r"\!", r"\ ", r"\quad", r"\qquad",
            r"\left", r"\right", r"\limits"]


def tex_math_to_html(expr: str) -> str:
    """Translate the inline-maths subset used in the body text into HTML.

    The subset is deliberately small: identifiers, sub- and superscripts,
    ``\\mathbf``, ``\\tilde``, ``\\mathbb{Z}``, ``\\mathrm``/``\\text``, the
    Greek letters and operators tabulated above, and ``\\bmod``.  Anything
    outside it belongs in a display equation, where both renderings are
    written out by hand.

    Raises:
        ValueError: If a backslash command survives translation, which means
            the body text used a construct this translator does not cover.
    """
    # Escaped braces are literal set delimiters, not grouping; take them out
    # of the way before any brace-aware rewriting and put them back last.
    out = expr.replace(r"\{", "\x01LB\x01").replace(r"\}", "\x01RB\x01")

    # Structural commands first, innermost braces outwards.
    for _ in range(6):
        new = re.sub(r"\\mathbf\{([^{}]*)\}",
                     lambda m: f"<b>{m.group(1)}</b>", out)
        new = re.sub(r"\\mathbb\{Z\}", "ℤ", new)
        new = re.sub(r"\\mathcal\{([^{}]*)\}",
                     lambda m: f'<span class="mcal">{m.group(1)}</span>', new)
        new = re.sub(r"\\(?:mathrm|text|operatorname)\{([^{}]*)\}",
                     lambda m: f'<span class="mroman">{m.group(1)}</span>',
                     new)
        new = re.sub(r"\\tilde\{([^{}]*)\}",
                     lambda m: f"{m.group(1)}\u0303", new)
        new = re.sub(r"\\hat\{([^{}]*)\}",
                     lambda m: f"{m.group(1)}\u0302", new)
        new = re.sub(r"\\bar\{([^{}]*)\}",
                     lambda m: f"{m.group(1)}\u0304", new)
        new = re.sub(r"\\sqrt\{([^{}]*)\}",
                     lambda m: '√<span class="radicand">'
                               f"{m.group(1)}</span>", new)
        new = re.sub(r"\\boldsymbol\{\\?(\w+)\}",
                     lambda m: f"<b>{_GREEK.get(chr(92) + m.group(1), m.group(1))}"
                               "</b>", new)
        if new == out:
            break
        out = new

    out = out.replace(r"\bmod", '<span class="mroman">mod</span>')
    for macro in (r"\min", r"\max", r"\log", r"\exp", r"\std", r"\Pr",
                  r"\cos", r"\sin", r"\tan", r"\ln", r"\det", r"\arg"):
        out = out.replace(macro, f'<span class="mroman">{macro[1:]}</span>')
    # Longest macro first: otherwise \le rewrites the head of \leftarrow and
    # leaves "ftarrow" behind, which the backslash check cannot catch.
    for macro, glyph in sorted(_GREEK.items(), key=lambda kv: -len(kv[0])):
        out = out.replace(macro, glyph)
    for macro, glyph in sorted(_SYMBOLS.items(), key=lambda kv: -len(kv[0])):
        out = out.replace(macro, glyph)
    for macro in sorted(_SPACING, key=len, reverse=True):
        out = out.replace(macro, " " if macro != r"\!" else "")

    # Sub- and superscripts, braced and single-character.
    for _ in range(6):
        new = re.sub(r"_\{([^{}]*)\}", lambda m: f"<sub>{m.group(1)}</sub>",
                     out)
        new = re.sub(r"\^\{([^{}]*)\}", lambda m: f"<sup>{m.group(1)}</sup>",
                     new)
        if new == out:
            break
        out = new
    out = re.sub(r"_(\w)", lambda m: f"<sub>{m.group(1)}</sub>", out)
    out = re.sub(r"\^(\w)", lambda m: f"<sup>{m.group(1)}</sup>", out)

    out = out.replace("{", "").replace("}", "")
    out = out.replace("\x01LB\x01", "{").replace("\x01RB\x01", "}")
    if "\\" in out:
        raise ValueError(f"untranslated LaTeX in inline maths: {expr!r}")

    # Numerals are upright in mathematics; identifiers are italic.  Walk the
    # text between tags so attribute values are left alone.
    pieces = re.split(r"(<[^>]*>)", out)
    for index, piece in enumerate(pieces):
        if piece.startswith("<"):
            continue
        pieces[index] = re.sub(r"\d+(?:\.\d+)?",
                               lambda m: f'<span class="mn">{m.group(0)}</span>',
                               piece)
    return f'<span class="math">{"".join(pieces)}</span>'


# --------------------------------------------------------------------------- #
# Inline markup
# --------------------------------------------------------------------------- #

_MATH_TOKEN = "\x00MATH{}\x00"


def _protect_math(text: str) -> Tuple[str, List[str]]:
    stash: List[str] = []

    def take(match: re.Match) -> str:
        stash.append(match.group(1))
        return _MATH_TOKEN.format(len(stash) - 1)

    return re.sub(r"\$([^$]*)\$", take, text), stash


def _restore(text: str, rendered: Sequence[str]) -> str:
    for index, value in enumerate(rendered):
        text = text.replace(_MATH_TOKEN.format(index), value)
    return text


class Renderer:
    """Base class holding the label and citation bookkeeping."""

    def __init__(self, refs: Sequence[Reference]) -> None:
        self.ref_order: List[str] = [r.key for r in refs]
        self.refs = {r.key: r for r in refs}
        self.labels: Dict[str, str] = {}
        self.cited: set = set()

    def citation_number(self, key: str) -> int:
        if key not in self.refs:
            raise KeyError(f"citation {key!r} has no bibliography entry")
        self.cited.add(key)
        return self.ref_order.index(key) + 1


def assign_labels(blocks: Sequence[object]) -> Dict[str, str]:
    """Walk the document once and number every labelled object."""
    labels: Dict[str, str] = {}
    section = 0
    subsection = 0
    figure = 0
    table = 0
    equation = 0
    algorithm = 0
    for block in blocks:
        if isinstance(block, Section):
            if block.level == 1:
                section += 1
                subsection = 0
                if block.label:
                    labels[block.label] = _roman(section)
            else:
                subsection += 1
                if block.label:
                    labels[block.label] = f"{_roman(section)}-{chr(64 + subsection)}"
        elif isinstance(block, Figure):
            figure += 1
            labels[block.label] = str(figure)
        elif isinstance(block, Table):
            table += 1
            labels[block.label] = _roman(table)
        elif isinstance(block, Equation) and block.label:
            equation += 1
            labels[block.label] = str(equation)
        elif isinstance(block, Algorithm):
            algorithm += 1
            labels[block.label] = str(algorithm)
    return labels


def _roman(value: int) -> str:
    numerals = [(1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
                (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
                (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I")]
    out = ""
    for amount, glyph in numerals:
        while value >= amount:
            out += glyph
            value -= amount
    return out


def equation_numbers(blocks: Sequence[object]) -> Dict[int, int]:
    """Map the index of each labelled equation to its printed number."""
    numbers: Dict[int, int] = {}
    counter = 0
    for index, block in enumerate(blocks):
        if isinstance(block, Equation) and block.label:
            counter += 1
            numbers[index] = counter
    return numbers
