"""Structural checks on the generated LaTeX and BibTeX.

This is not a substitute for compiling: no TeX distribution is available in
the environment that generates these files.  It catches the failures a
generator realistically makes --- unbalanced environments, odd maths
delimiters, dangling cross-references, citation keys with no entry.

Run:  python paper/check_tex.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE / "Modified_SALSA_Research_Paper.tex"
BIB = HERE / "Modified_SALSA_Research_Paper.bib"

ENVIRONMENTS = ["equation", "equation*", "itemize", "enumerate", "table",
                "table*", "figure", "figure*", "tabular", "threeparttable",
                "algorithm", "document", "tablenotes", "abstract",
                "IEEEkeywords"]

text = TEX.read_text(encoding="utf-8")
problems = 0

print("environments")
for env in ENVIRONMENTS:
    opens = len(re.findall(r"\\begin\{" + re.escape(env) + r"\}", text))
    closes = len(re.findall(r"\\end\{" + re.escape(env) + r"\}", text))
    if not (opens or closes):
        continue
    ok = opens == closes
    problems += 0 if ok else 1
    print(f"  {env:<16s} begin={opens:<4d} end={closes:<4d}"
          f"{'' if ok else '   <-- MISMATCH'}")

dollars = text.count("$")
ok = dollars % 2 == 0
problems += 0 if ok else 1
print(f"\nmaths delimiters: {dollars} '$' "
      f"({'balanced' if ok else 'ODD -- PROBLEM'})")

braces_open = len(re.findall(r"(?<!\\)\{", text))
braces_close = len(re.findall(r"(?<!\\)\}", text))
ok = braces_open == braces_close
problems += 0 if ok else 1
print(f"unescaped braces: {braces_open} open, {braces_close} close "
      f"({'balanced' if ok else 'MISMATCH'})")

labels = set(re.findall(r"\\label\{([^}]*)\}", text))
refs = set(re.findall(r"\\ref\{([^}]*)\}", text))
refs |= set(re.findall(r"\\eqref\{([^}]*)\}", text))
dangling = sorted(refs - labels)
unreferenced = sorted(labels - refs)
problems += len(dangling)
print(f"\ncross-references: {len(labels)} labels, {len(refs)} referenced")
print(f"  dangling references: {dangling or 'none'}")
print(f"  labels never referenced: {unreferenced or 'none'}")

cites: set[str] = set()
for group in re.findall(r"\\cite\{([^}]*)\}", text):
    cites.update(key.strip() for key in group.split(","))
bib_keys = set(re.findall(r"@\w+\{([^,]+),", BIB.read_text(encoding="utf-8")))
missing = sorted(cites - bib_keys)
uncited = sorted(bib_keys - cites)
problems += len(missing) + len(uncited)
print(f"\ncitations: {len(cites)} keys cited, {len(bib_keys)} bibliography "
      "entries")
print(f"  cited but absent from the .bib: {missing or 'none'}")
print(f"  in the .bib but never cited:    {uncited or 'none'}")

figures = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]*)\}", text)
print(f"\nfigure files referenced: {len(figures)}")
for path in figures:
    exists = (HERE / path).exists()
    problems += 0 if exists else 1
    print(f"  {'ok  ' if exists else 'MISSING'} {path}")

print(f"\n{'PASS' if problems == 0 else f'{problems} PROBLEM(S)'}")
sys.exit(0 if problems == 0 else 1)
