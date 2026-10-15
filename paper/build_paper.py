"""Emit the manuscript as HTML (for PDF rendering) and as LaTeX.

Both targets are generated from :mod:`manuscript`, so the rendered PDF and
the editable LaTeX source carry the same wording, the same numbering and the
same citations.

Run:  python paper/build_paper.py
"""

from __future__ import annotations

import html as htmlmod
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import manuscript as M                                       # noqa: E402
from emitters import (Abstract, Algorithm, Equation, Figure,  # noqa: E402
                      Keywords, ListBlock, Para, Reference, Section, Table,
                      _protect_math, _restore, assign_labels, tex_math_to_html)

ALL_BLOCKS = list(M.BODY) + list(M.BACKMATTER)
LABELS = assign_labels(ALL_BLOCKS)
REF_ORDER = [r.key for r in M.REFERENCES]
REF_BY_KEY = {r.key: r for r in M.REFERENCES}
CITED: set = set()


def _kind(label: str) -> str:
    if label.startswith("fig:"):
        return "Fig."
    if label.startswith("tab:"):
        return "Table"
    if label.startswith("eq:"):
        return "eq."
    if label.startswith("alg:"):
        return "Alg."
    return "Section"


def _cite_numbers(keys: Sequence[str]) -> List[int]:
    numbers = []
    for key in keys:
        key = key.strip()
        if key not in REF_BY_KEY:
            raise KeyError(f"citation {key!r} has no bibliography entry")
        CITED.add(key)
        numbers.append(REF_ORDER.index(key) + 1)
    return numbers


def _resolve_ref(label: str) -> str:
    if label not in LABELS:
        raise KeyError(f"cross-reference {label!r} has no target")
    return LABELS[label]


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #

def inline_html(text: str) -> str:
    text, maths = _protect_math(text)
    text = htmlmod.escape(text, quote=False)

    text = re.sub(r"@cite\{([^}]*)\}",
                  lambda m: "[" + ", ".join(
                      str(n) for n in _cite_numbers(m.group(1).split(","))
                  ) + "]", text)
    text = re.sub(r"@ref\{([^}]*)\}",
                  lambda m: f"{_kind(m.group(1))}&nbsp;"
                            f"{_resolve_ref(m.group(1))}", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*([^*]+?)\*(?![\w*])", r"<em>\1</em>", text)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = text.replace("---", "&#8212;").replace("--", "&#8211;")
    return _restore(text, [tex_math_to_html(m) for m in maths])


CSS = r"""
@page {
  size: A4;
  margin: 17mm 15mm 18mm 15mm;
  @bottom-center {
    content: counter(page);
    font-family: "Times New Roman", Times, serif;
    font-size: 9pt;
    color: #333;
    vertical-align: top;
    padding-top: 7mm;
  }
}
@page :first { @bottom-center { content: counter(page); } }

* { box-sizing: border-box; }
html { font-size: 9.3pt; }
body {
  font-family: "Times New Roman", Times, serif;
  margin: 0; color: #111; background: #fff;
  text-align: justify; hyphens: auto; -webkit-hyphens: auto;
  line-height: 1.26;
}

/* ---- title block, spans both columns --------------------------------- */
.titleblock { column-span: all; text-align: center; margin: 0 0 5mm 0; }
h1.title {
  font-size: 17pt; font-weight: bold; line-height: 1.22;
  margin: 0 0 3.2mm 0; text-align: center; letter-spacing: -0.1px;
}
.authors { font-size: 11pt; margin: 0 0 1.4mm 0; }
.authors .name { margin: 0 2.6mm; }
.affils { font-size: 8.4pt; font-style: italic; color: #333; line-height: 1.4; }
.affils div { margin: 0.3mm 0; }
.repoline { font-size: 8.2pt; margin-top: 2.2mm; color: #333; }
.repoline code { font-style: normal; }

.abstractbox {
  column-span: all; margin: 0 9mm 5mm 9mm; font-size: 8.7pt;
  line-height: 1.30; text-align: justify;
}
.abstractbox .lbl { font-weight: bold; font-style: italic; }
.kw { margin-top: 2.2mm; }

.body { column-count: 2; column-gap: 7mm; }

/* ---- headings --------------------------------------------------------- */
h2.sec {
  font-size: 9.6pt; font-weight: bold; text-align: center;
  text-transform: uppercase; letter-spacing: 0.35px;
  margin: 4.0mm 0 1.6mm 0; break-after: avoid; page-break-after: avoid;
}
h3.sub {
  font-size: 9.3pt; font-style: italic; font-weight: normal;
  margin: 2.6mm 0 1.0mm 0; break-after: avoid; page-break-after: avoid;
}
p { margin: 0 0 1.5mm 0; text-indent: 4.2mm; orphans: 2; widows: 2; }
p.nofirst, h2.sec + p, h3.sub + p, .eq + p, figure + p, table + p,
.alg + p, ul + p, ol + p { text-indent: 0; }

ul, ol { margin: 0.8mm 0 1.8mm 0; padding-left: 4.6mm; }
li { margin: 0 0 1.1mm 0; text-align: justify; }

code {
  font-family: Consolas, "Courier New", monospace; font-size: 0.88em;
  background: #f4f4f2; padding: 0 0.6mm; border-radius: 1px;
}

/* ---- mathematics ------------------------------------------------------ */
.math { font-style: italic; white-space: nowrap; }
.math b { font-style: normal; font-weight: bold; }
.math .mroman { font-style: normal; }
.mn { font-style: normal; }
.eq .mn { font-style: normal; }
.mcal { font-family: "Palatino Linotype", Palatino, serif;
        font-style: italic; }
.math sub, .math sup { font-style: italic; }
sub, sup { font-size: 0.72em; line-height: 0; }

.eq {
  display: block; text-align: center; margin: 2.2mm 0 2.2mm 0;
  break-inside: avoid; position: relative; font-style: italic;
  padding: 0 7mm 0 1mm;
}
.eq .inner { display: inline-block; }
.eq .num {
  position: absolute; right: 0; top: 50%; transform: translateY(-50%);
  font-style: normal; font-size: 0.94em;
}
.eq b { font-style: normal; font-weight: bold; }
.eq .mroman, .eq .num { font-style: normal; }
.eqspace { display: inline-block; width: 2.4mm; }
.eqgap { display: inline-block; width: 5mm; }
.paren { font-style: normal; }
.bigop { font-style: normal; font-size: 1.22em; }
.tildevec { font-weight: bold; font-style: normal; }
.sqrt { font-style: normal; }
.radicand { border-top: 0.7px solid #111; padding: 0 0.6mm; }
.frac {
  display: inline-block; vertical-align: -0.52em; text-align: center;
  margin: 0 0.6mm;
}
.frac .num {
  display: block; position: static; transform: none;
  border-bottom: 0.7px solid #111; padding: 0 1mm 0.2mm 1mm;
  font-style: italic;
}
.frac .den { display: block; padding: 0.2mm 1mm 0 1mm; }

/* ---- figures and tables ----------------------------------------------- */
figure { margin: 2.6mm 0 2.6mm 0; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; height: auto; }
figcaption {
  font-size: 8.0pt; text-align: justify; margin-top: 1.4mm; line-height: 1.26;
}
figcaption .lbl, table caption .lbl { font-weight: bold; }

table {
  border-collapse: collapse; width: 100%; font-size: 7.9pt;
  margin: 1.2mm 0 2.6mm 0; break-inside: avoid; line-height: 1.22;
}
table caption {
  caption-side: top; text-align: justify; font-size: 8.0pt;
  margin-bottom: 1.2mm; line-height: 1.26;
}
thead { display: table-header-group; }
th {
  font-weight: bold; text-align: left; padding: 0.9mm 1.4mm 0.9mm 0;
  border-top: 0.9px solid #111; border-bottom: 0.6px solid #111;
  vertical-align: bottom;
}
td { padding: 0.65mm 1.4mm 0.65mm 0; vertical-align: top; }
tbody tr:last-child td { border-bottom: 0.9px solid #111; }
td.r, th.r { text-align: right; padding-right: 0;
             padding-left: 2.6mm; }
th:first-child, td:first-child { padding-left: 0; }
td.c, th.c { text-align: center; }
tr.grouprow td {
  font-style: italic; padding-top: 1.6mm; border-bottom: 0.4px solid #ccc;
}
.tnote { font-size: 7.5pt; margin-top: 1.0mm; line-height: 1.25;
         text-align: justify; }

.wide { column-span: all; break-inside: avoid;
        page-break-inside: avoid; }
.wide table { font-size: 7.9pt; }

/* ---- algorithm -------------------------------------------------------- */
.alg {
  border-top: 0.9px solid #111; border-bottom: 0.9px solid #111;
  margin: 2.4mm 0; padding: 1.2mm 0; break-inside: avoid; font-size: 8.1pt;
}
.alg .cap {
  font-weight: bold; border-bottom: 0.6px solid #111; padding-bottom: 0.9mm;
  margin-bottom: 1.0mm; font-size: 8.1pt; text-align: justify;
}
.alg .cap .lbl { font-weight: bold; }
.alg .line { line-height: 1.34; text-align: left; }
.alg .i1 { padding-left: 4.4mm; }
.alg .cmt { color: #555; font-style: italic; }

/* ---- references ------------------------------------------------------- */
ol.refs { padding-left: 5.4mm; font-size: 8.0pt; line-height: 1.25; }
ol.refs li { margin-bottom: 0.9mm; text-align: justify; }

.footnote {
  column-span: all; border-top: 0.6px solid #999; margin-top: 4mm;
  padding-top: 1.4mm; font-size: 7.6pt; line-height: 1.3;
}
"""


def _cells(row: Sequence[str], align: str, tag: str) -> str:
    out = []
    for index, cell in enumerate(row):
        klass = ""
        if index < len(align):
            if align[index] == "r":
                klass = ' class="r"'
            elif align[index] == "c":
                klass = ' class="c"'
        out.append(f"<{tag}{klass}>{inline_html(cell)}</{tag}>")
    return "".join(out)


def render_html() -> str:
    parts: List[str] = []
    eq_number = 0

    for block in ALL_BLOCKS:
        if isinstance(block, Section):
            if block.level == 1:
                number = LABELS.get(block.label, "")
                prefix = f"{number}.&nbsp; " if number else ""
                parts.append(
                    f'<h2 class="sec">{prefix}{inline_html(block.title)}</h2>')
            else:
                number = LABELS.get(block.label, "")
                letter = number.split("-")[-1] if "-" in number else ""
                prefix = f"{letter}.&nbsp; " if letter else ""
                parts.append(
                    f'<h3 class="sub">{prefix}{inline_html(block.title)}</h3>')

        elif isinstance(block, Para):
            parts.append(f"<p>{inline_html(block.text)}</p>")

        elif isinstance(block, Equation):
            if block.label:
                eq_number += 1
                tag = f'<span class="num">({eq_number})</span>'
            else:
                tag = ""
            parts.append(f'<div class="eq"><span class="inner">{block.html}'
                         f"</span>{tag}</div>")

        elif isinstance(block, ListBlock):
            tag = "ol" if block.ordered else "ul"
            items = "".join(f"<li>{inline_html(i)}</li>" for i in block.items)
            parts.append(f"<{tag}>{items}</{tag}>")

        elif isinstance(block, Figure):
            wide = ' class="wide"' if block.width == "double" else ""
            number = LABELS[block.label]
            parts.append(
                f"<figure{wide}><img src=\"{block.path}\" alt=\"\">"
                f'<figcaption><span class="lbl">Fig. {number}.</span> '
                f"{inline_html(block.caption)}</figcaption></figure>")

        elif isinstance(block, Table):
            number = LABELS[block.label]
            head = _cells(block.header, block.align, "th")
            body_rows = []
            for row in block.rows:
                if row and row[0].startswith("*") and all(
                        c == "" for c in row[1:]):
                    body_rows.append(
                        f'<tr class="grouprow"><td colspan="{len(block.header)}">'
                        f"{inline_html(row[0])}</td></tr>")
                else:
                    body_rows.append(
                        f"<tr>{_cells(row, block.align, 'td')}</tr>")
            note = (f'<div class="tnote">{inline_html(block.note)}</div>'
                    if block.note else "")
            wide_open = '<div class="wide">' if block.width == "double" else ""
            wide_close = "</div>" if block.width == "double" else ""
            parts.append(
                f"{wide_open}<table><caption>"
                f'<span class="lbl">Table {number}.</span> '
                f"{inline_html(block.caption)}</caption>"
                f"<thead><tr>{head}</tr></thead>"
                f"<tbody>{''.join(body_rows)}</tbody></table>"
                f"{note}{wide_close}")

        elif isinstance(block, Algorithm):
            number = LABELS[block.label]
            lines = []
            for indent, text in block.lines:
                rendered = inline_html(text)
                rendered = re.sub(
                    r"//(.*)$", lambda m: f'<span class="cmt">//{m.group(1)}'
                                          "</span>", rendered)
                klass = "line i1" if indent else "line"
                lines.append(f'<div class="{klass}">{rendered}</div>')
            parts.append(
                f'<div class="alg"><div class="cap">'
                f'<span class="lbl">Algorithm {number}.</span> '
                f"{inline_html(block.caption)}</div>{''.join(lines)}</div>")

    # ---- references ---------------------------------------------------- #
    missing = set(REF_ORDER) - CITED
    if missing:
        raise SystemExit(f"bibliography entries never cited: {sorted(missing)}")

    def dashes(text: str) -> str:
        return (htmlmod.escape(text).replace("---", "&#8212;")
                .replace("--", "&#8211;"))

    ref_items = []
    for ref in M.REFERENCES:
        pieces = [f"{dashes(ref.authors)}, "
                  f"&ldquo;{dashes(ref.title)},&rdquo; "
                  f"<em>{dashes(ref.venue)}</em>"]
        if ref.note:
            pieces.append(f", {dashes(ref.note)}")
        year = ref.bibfields.get("year", "")
        if year and year not in ref.venue and year not in ref.note:
            pieces.append(f", {year}")
        doi = ref.bibfields.get("doi")
        if doi:
            pieces.append(f". doi:&#8203;{htmlmod.escape(doi)}")
        # Print the URL only when it is not already the venue string.
        if ref.url and ref.url.split("//")[-1] not in ref.venue:
            pieces.append(f". {htmlmod.escape(ref.url)}")
        ref_items.append("<li>" + "".join(pieces) + ".</li>")

    parts.append('<h2 class="sec">References</h2>')
    parts.append(f'<ol class="refs">{"".join(ref_items)}</ol>')

    author_html = " ".join(
        f'<span class="name">{htmlmod.escape(a["name"])}'
        f'<sup>{a["affil"]}</sup></span>' for a in M.AUTHORS)
    affil_html = "".join(
        f'<div><sup>{k}</sup>&#8201;{htmlmod.escape(v)}</div>'
        for k, v in sorted(M.AFFILIATIONS.items()))

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Modified SALSA</title>
<style>{CSS}</style></head>
<body>
<div class="titleblock">
  <h1 class="title">{htmlmod.escape(M.TITLE)}</h1>
  <div class="authors">{author_html}</div>
  <div class="affils">{affil_html}</div>
  <div class="repoline">Code and experiment artefacts:
    <code>{htmlmod.escape(M.REPO)}</code></div>
</div>
<div class="abstractbox">
  <p class="nofirst"><span class="lbl">Abstract&#8212;</span>
  {inline_html(M.ABSTRACT)}</p>
  <p class="nofirst kw"><span class="lbl">Index Terms&#8212;</span>
  {inline_html(M.KEYWORDS)}</p>
</div>
<div class="body">
{chr(10).join(parts)}
</div>
</body></html>
"""


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #

_TEX_ESCAPES = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                ("}", r"\}"), ("~", r"\textasciitilde{}"),
                ("^", r"\textasciicircum{}")]


def inline_tex(text: str) -> str:
    text, maths = _protect_math(text)

    # Pull the markup out before escaping, then put LaTeX commands back.
    cites: List[str] = []
    refs: List[str] = []

    def take_cite(match: re.Match) -> str:
        keys = [k.strip() for k in match.group(1).split(",")]
        _cite_numbers(keys)
        cites.append("\\cite{" + ",".join(keys) + "}")
        return f"\x01C{len(cites) - 1}\x01"

    def take_ref(match: re.Match) -> str:
        label = match.group(1)
        _resolve_ref(label)
        kind = _kind(label)
        if kind == "eq.":
            refs.append("\\eqref{" + label + "}")
        else:
            refs.append(f"{kind}~\\ref{{{label}}}")
        return f"\x01R{len(refs) - 1}\x01"

    text = re.sub(r"@cite\{([^}]*)\}", take_cite, text)
    text = re.sub(r"@ref\{([^}]*)\}", take_ref, text)

    bolds: List[str] = []
    italics: List[str] = []
    codes: List[str] = []

    def take_bold(match: re.Match) -> str:
        bolds.append(match.group(1))
        return f"\x01B{len(bolds) - 1}\x01"

    def take_italic(match: re.Match) -> str:
        italics.append(match.group(1))
        return f"\x01I{len(italics) - 1}\x01"

    def take_code(match: re.Match) -> str:
        codes.append(match.group(1))
        return f"\x01V{len(codes) - 1}\x01"

    text = re.sub(r"\*\*(.+?)\*\*", take_bold, text)
    text = re.sub(r"(?<![\w*])\*([^*]+?)\*(?![\w*])", take_italic, text)
    text = re.sub(r"`([^`]+)`", take_code, text)

    for char, repl in _TEX_ESCAPES:
        text = text.replace(char, repl)
    text = text.replace("---", "\x02EM\x02").replace("--", "\x02EN\x02")
    text = text.replace("\x02EM\x02", "---").replace("\x02EN\x02", "--")

    for index, value in enumerate(bolds):
        text = text.replace(f"\x01B{index}\x01",
                            "\\textbf{" + inline_tex(value) + "}")
    for index, value in enumerate(italics):
        text = text.replace(f"\x01I{index}\x01",
                            "\\emph{" + inline_tex(value) + "}")
    for index, value in enumerate(codes):
        text = text.replace(f"\x01V{index}\x01",
                            "\\texttt{" + value.replace("_", r"\_") + "}")
    for index, value in enumerate(cites):
        text = text.replace(f"\x01C{index}\x01", value)
    for index, value in enumerate(refs):
        text = text.replace(f"\x01R{index}\x01", value)

    return _restore(text, [f"${m}$" for m in maths])


def render_tex() -> str:
    out: List[str] = []
    for block in ALL_BLOCKS:
        if isinstance(block, Section):
            cmd = "section" if block.level == 1 else "subsection"
            out.append(f"\\{cmd}{{{inline_tex(block.title)}}}"
                       f"\\label{{{block.label}}}" if block.label
                       else f"\\{cmd}{{{inline_tex(block.title)}}}")
        elif isinstance(block, Para):
            out.append(inline_tex(block.text))
        elif isinstance(block, Equation):
            if block.label:
                out.append("\\begin{equation}\\label{" + block.label + "}\n"
                           + block.tex + "\n\\end{equation}")
            else:
                out.append("\\begin{equation*}\n" + block.tex
                           + "\n\\end{equation*}")
        elif isinstance(block, ListBlock):
            env = "enumerate" if block.ordered else "itemize"
            items = "\n".join("  \\item " + inline_tex(i) for i in block.items)
            out.append(f"\\begin{{{env}}}\n{items}\n\\end{{{env}}}")
        elif isinstance(block, Figure):
            env = "figure*" if block.width == "double" else "figure"
            width = r"\textwidth" if block.width == "double" else r"\columnwidth"
            out.append(
                f"\\begin{{{env}}}[t]\n  \\centering\n"
                f"  \\includegraphics[width={width}]{{{block.path}}}\n"
                f"  \\caption{{{inline_tex(block.caption)}}}\n"
                f"  \\label{{{block.label}}}\n\\end{{{env}}}")
        elif isinstance(block, Table):
            env = "table*" if block.width == "double" else "table"
            align = block.align or "l" * len(block.header)
            head = " & ".join("\\textbf{" + inline_tex(h) + "}"
                              for h in block.header)
            body = []
            for row in block.rows:
                if row and row[0].startswith("*") and all(
                        c == "" for c in row[1:]):
                    body.append("\\multicolumn{" + str(len(block.header))
                                + "}{l}{" + inline_tex(row[0])
                                + "} \\\\[0.3ex]")
                else:
                    body.append(" & ".join(inline_tex(c) for c in row)
                                + " \\\\")
            note = ("\n  \\begin{tablenotes}\\footnotesize\n    \\item "
                    + inline_tex(block.note) + "\n  \\end{tablenotes}"
                    if block.note else "")
            out.append(
                f"\\begin{{{env}}}[t]\n  \\centering\n"
                f"  \\caption{{{inline_tex(block.caption)}}}\n"
                f"  \\label{{{block.label}}}\n"
                f"  \\begin{{threeparttable}}\n"
                f"  \\footnotesize\n"
                f"  \\begin{{tabular}}{{{align}}}\n"
                f"    \\toprule\n    {head} \\\\\n    \\midrule\n    "
                + "\n    ".join(body)
                + f"\n    \\bottomrule\n  \\end{{tabular}}{note}\n"
                f"  \\end{{threeparttable}}\n\\end{{{env}}}")
        elif isinstance(block, Algorithm):
            lines = []
            for indent, text in block.lines:
                rendered = inline_tex(text)
                rendered = re.sub(r"//(.*)$",
                                  lambda m: r"\textit{//" + m.group(1) + "}",
                                  rendered)
                lines.append(("\\hspace*{1.6em}" if indent else "")
                             + rendered + r" \\")
            out.append(
                "\\begin{algorithm}[t]\n"
                f"  \\caption{{{inline_tex(block.caption)}}}\n"
                f"  \\label{{{block.label}}}\n"
                "  \\footnotesize\n  \\raggedright\n"
                + "\n  ".join(lines)
                + "\n\\end{algorithm}")

    authors = " \\and ".join(
        f"\\IEEEauthorblockN{{{a['name']}}}\n"
        f"\\IEEEauthorblockA{{\\textit{{{M.AFFILIATIONS[a['affil']]}}}}}"
        for a in M.AUTHORS)

    preamble = r"""% Modified SALSA -- manuscript source.
%
% GENERATED by paper/build_paper.py from paper/manuscript.py.  Edit the
% manuscript module and regenerate, or adopt this file as the master copy;
% do not edit both.
%
% NOT COMPILED: no TeX distribution was available in the environment that
% produced this file, so this source has never been run through LaTeX.  The
% verified artefact is Modified_SALSA_Research_Paper.pdf, rendered from the
% HTML emitter.  Compile and diff against that PDF before submission.
%
% Requires: IEEEtran, graphicx, amsmath, amssymb, booktabs, threeparttable,
%           algorithm, url, hyperref.
\documentclass[conference,a4paper,10pt]{IEEEtran}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{threeparttable}
\usepackage{algorithm}
\usepackage{url}
\usepackage[hidelinks]{hyperref}
\graphicspath{{./}}
\renewcommand{\arraystretch}{1.12}

\begin{document}
"""
    title_block = (
        "\\title{" + M.TITLE + "}\n\n"
        "\\author{\\IEEEauthorblockN{" + "}\n\\IEEEauthorblockN{".join(
            a["name"] for a in M.AUTHORS) + "}\n"
        + "\n".join("\\IEEEauthorblockA{\\textit{"
                    + M.AFFILIATIONS[k] + "}}"
                    for k in sorted(M.AFFILIATIONS)) + "}\n\n"
        "\\maketitle\n\n"
        "\\begin{abstract}\n" + inline_tex(M.ABSTRACT) + "\n\\end{abstract}\n\n"
        "\\begin{IEEEkeywords}\n" + inline_tex(M.KEYWORDS)
        + "\n\\end{IEEEkeywords}\n")

    tail = ("\n\\bibliographystyle{IEEEtran}\n"
            "\\bibliography{Modified_SALSA_Research_Paper}\n\n"
            "\\end{document}\n")

    return preamble + title_block + "\n\n".join(out) + tail


def render_bib() -> str:
    entries = []
    for ref in M.REFERENCES:
        fields = {"author": ref.authors.replace(", and ", " and ")
                                        .replace(", ", " and "),
                  "title": "{" + ref.title + "}"}
        fields.update(ref.bibfields)
        body = ",\n  ".join(f"{k} = {{{v}}}" if not v.startswith("{")
                            else f"{k} = {v}"
                            for k, v in fields.items())
        entries.append(f"@{ref.bibtype}{{{ref.key},\n  {body}\n}}")
    header = ("% Bibliography for the Modified SALSA manuscript.\n"
              "% Every entry was checked against a primary source; the\n"
              "% verification trail is in Modified_SALSA_Evidence_Audit.md.\n\n")
    return header + "\n\n".join(entries) + "\n"


# A single-column stylesheet for the Word target.  Word's HTML importer does
# not implement CSS multi-column layout or paged-media margin boxes, and
# forcing it to try produces a document nobody can edit.  A clean single
# column is both faithful to the content and actually useful for
# track-changes review, which is what the DOCX is for.
WORD_CSS_OVERRIDE = """
html { font-size: 10.5pt; }
body { line-height: 1.4; }
.body { column-count: 1; }
.wide { column-span: none; }
.abstractbox { margin: 0 0 6mm 0; font-size: 10pt; }
h1.title { font-size: 18pt; }
h2.sec { font-size: 12pt; text-align: left; margin: 7mm 0 2mm 0; }
h3.sub { font-size: 11pt; margin: 4mm 0 1.5mm 0; }
table { font-size: 9pt; }
table caption { font-size: 9.5pt; }
figcaption { font-size: 9.5pt; }
ol.refs { font-size: 9.5pt; }
.alg { font-size: 9.5pt; }
figure img { max-width: 97%; }
p { text-indent: 0; margin: 0 0 2.6mm 0; }
"""


if __name__ == "__main__":
    html_text = render_html()
    (HERE / "Modified_SALSA_Research_Paper.html").write_text(
        html_text, encoding="utf-8")

    word_html = html_text.replace("</style>", WORD_CSS_OVERRIDE + "</style>")
    (HERE / "_word_source.html").write_text(word_html, encoding="utf-8")
    tex_text = render_tex()
    (HERE / "Modified_SALSA_Research_Paper.tex").write_text(
        tex_text, encoding="utf-8")
    bib_text = render_bib()
    (HERE / "Modified_SALSA_Research_Paper.bib").write_text(
        bib_text, encoding="utf-8")

    counts = {"sections": sum(1 for b in ALL_BLOCKS
                              if isinstance(b, Section) and b.level == 1),
              "figures": sum(1 for b in ALL_BLOCKS if isinstance(b, Figure)),
              "tables": sum(1 for b in ALL_BLOCKS if isinstance(b, Table)),
              "equations": sum(1 for b in ALL_BLOCKS
                               if isinstance(b, Equation) and b.label),
              "references": len(M.REFERENCES)}
    words = len(re.findall(r"\w+", re.sub(r"<[^>]+>", " ", html_text)))
    print(f"HTML  {len(html_text):>8,} bytes")
    print(f"Word  {len(word_html):>8,} bytes  (single column)")
    print(f"LaTeX {len(tex_text):>8,} bytes")
    print(f"BibTeX{len(bib_text):>8,} bytes")
    print("  " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"  approx. words={words:,}")
