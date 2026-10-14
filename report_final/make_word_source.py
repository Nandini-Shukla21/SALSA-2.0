"""Derive the Word build source from the canonical final HTML.

The canonical deliverable, ``SALSA_2.0_DRDO_SAG_Final_Report.html``, carries its
Table of Contents, List of Figures and List of Tables as hand-written anchor
lists. Those are the right thing for the HTML: every entry is a working internal
link. They cannot carry page numbers, because an HTML document has no pages.

Word can generate all three as live fields with correct page numbers, so this
script produces a build source in which each list is replaced by a single marker
paragraph. ``build_docx_pdf.ps1`` substitutes a Word field for each marker.

Nothing else is changed: the prose, tables, figures and numbering are identical
in both files.

Run:  python report_final/make_word_source.py
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "SALSA_2.0_DRDO_SAG_Final_Report.html"
TARGET = HERE / "_word_source.html"


def replace_list_after(html: str, heading_id: str, marker: str) -> str:
    """Replace the ``ul.toc`` that follows a given heading with a marker."""
    anchor = f'id="{heading_id}"'
    start = html.index(anchor)
    list_start = html.index('<ul class="toc">', start)
    list_end = html.index("</ul>", list_start) + len("</ul>")
    return html[:list_start] + f"<p>{marker}</p>" + html[list_end:]


def main() -> None:
    html = SOURCE.read_text(encoding="utf-8")

    html = replace_list_after(html, "fm-lof", "[[LOF]]")
    html = replace_list_after(html, "fm-lot", "[[LOT]]")
    html = replace_list_after(html, "fm-toc", "[[TOC]]")

    # Word takes its page geometry and numbering from the COM build script, not
    # from CSS paged-media rules, and it cannot run the polyfill.
    html = html.replace('<script src="assets/paged.polyfill.js"></script>', "")

    # Word's HTML importer folds <figcaption> into the same paragraph as the
    # <img> that precedes it, so the caption never becomes a paragraph of its
    # own and the List of Figures field finds nothing to collect. Closing the
    # figure before the caption makes each caption a plain paragraph beginning
    # with "Figure n.m:", which the build script then styles as a Caption.
    html, n_caps = re.subn(
        r"<figcaption>(.*?)</figcaption>\s*</figure>",
        lambda m: '</figure>\n<p class="caption">' + m.group(1) + "</p>",
        html,
        flags=re.S,
    )
    if n_caps != 6:
        raise SystemExit(f"expected 6 figure captions, rewrote {n_caps}")
    print(f"figure captions promoted to their own paragraphs: {n_caps}")

    # Mark the front-matter / body boundary so the build script can insert a
    # real section break there and restart the page numbering.
    html = html.replace(
        '<section class="pg-main" style="counter-reset: page 1;">',
        '<p>[[SECTIONBREAK]]</p>\n<section>',
    )

    for marker in ("[[TOC]]", "[[LOF]]", "[[LOT]]", "[[SECTIONBREAK]]"):
        if marker not in html:
            raise SystemExit(f"marker {marker} was not placed")

    leftover = re.findall(r"\[\[[A-Z]+\]\]", html)
    expected = {"[[TOC]]", "[[LOF]]", "[[LOT]]", "[[SECTIONBREAK]]"}
    unexpected = set(leftover) - expected
    if unexpected:
        raise SystemExit(f"unexpected placeholders remain: {sorted(unexpected)}")

    TARGET.write_text(html, encoding="utf-8")
    print(f"wrote {TARGET.name}  ({len(html):,} bytes)")
    print(f"markers placed: {sorted(expected)}")


if __name__ == "__main__":
    main()
