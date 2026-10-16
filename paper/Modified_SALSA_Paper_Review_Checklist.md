# Modified SALSA — Pre-Submission Review Checklist

Open questions, missing data, confirmations needed and checks to run before the
manuscript goes anywhere. Written for the three named authors and their
supervisor. Nothing here is a defect in the measurements; it is the list of
things a reader, a reviewer or an institution will reasonably ask about.

Prepared: 14 October 2026, alongside `Modified_SALSA_Research_Paper.pdf`.

---

## A. Authorship, affiliation and clearance — **blocking**

These must be resolved by a human before submission. None of them can be
settled from the repository.

| # | Item | Status |
|---|---|---|
| A1 | **Author order** Shukla, Joshi, Adhana — confirm all three agree. | Assumed from the task brief; not confirmed by the co-authors |
| A2 | **Author contributions.** The statement in §XI is a **template**, not a finding. The repository history and run metadata attribute the implementation, experiments and analysis to the first author; the contributions of the second and third authors are not documented in any artifact examined. Replace the paragraph with an agreed CRediT-style statement. | **Not written** — placeholder in the PDF |
| A3 | **Second author's affiliation.** Pranay Joshi is listed under J.C. Bose University of Science and Technology, YMCA, Faridabad, as supplied in the brief. This was **not independently verified**, nor was it verified that both student authors share one affiliation or department. | **Unverified** |
| A4 | **Third author's affiliation and consent.** "Scientific Analysis Group (SAG), Defence Research and Development Organisation (DRDO), Delhi, India" is consistent with DRDO's own published material (SAG is a DRDO laboratory at Metcalfe House, Delhi-110054, working in cryptology and information security). Whether Dr Adhana has agreed to be listed as an author, and in that position, is **not established here**. | **Unconfirmed** |
| A5 | **No department names, email addresses, ORCID identifiers or a corresponding author have been supplied**, so none appear. Most venues require at least a corresponding author with an email. Add them. | **Missing by design** |
| A6 | **Institutional clearance.** The work was done during an internship at SAG, DRDO. The paper states explicitly that it is not an official DRDO or SAG publication and carries no endorsement. Confirm what internal review or clearance DRDO requires before external submission, and obtain it. | **Not obtained** |
| A7 | **Funding.** None declared, because none was documented. Confirm. | Unconfirmed |

---

## B. Things to add to the repository before the paper cites it

| # | Item | Why |
|---|---|---|
| B1 | **Add a LICENSE file.** `pyproject.toml` declares MIT but no LICENSE file is tracked. A reader cannot rely on the declaration alone. | The paper states this honestly, which is slightly awkward in print |
| B2 | **Commit the enumeration-baseline script.** Table VIII is the paper's strongest honest claim and was measured by a 40-line NumPy script that currently lives only in a scratch directory. It must be in the repository for the result to be reproducible. | Reproducibility of a headline table |
| B3 | **Commit `paper/`** — the manuscript source, figure script, emitters and build scripts — so the figures regenerate from the artifacts. | Reproducibility of every figure |
| B4 | **Decide about checkpoints.** `.gitignore` excludes `*.pt`, so the weights behind the replay results are not downloadable, and a reader must retrain (≈19 min and ≈24 min on a laptop CPU). Either publish them (they are a few tens of MB) or keep the current honest statement. | A reviewer will notice |
| B5 | Consider adding a short `REPRODUCING_THE_PAPER.md` giving the exact commands for each table. | Low effort, high value |

---

## C. Scientific gaps a reviewer will raise

Ordered by how likely they are to be raised, and with the cheapest honest
response noted.

| # | Likely question | Current position | Cheapest response |
|---|---|---|---|
| C1 | *"One secret and one seed per dimension — how do you know this isn't luck?"* | Stated plainly in the abstract, §VI-F and §IX. We quote no success rate. | Run 5 seeds × 5 secrets at n = 12 and n = 20. About 10 hours of CPU in total. **This is the single highest-value missing experiment.** |
| C2 | *"If enumeration solves it in a second, why is this interesting?"* | §VII answers directly: the contribution is representational and methodological, not cryptanalytic. | Keep the framing. Do not let a well-meaning edit soften §VII |
| C3 | *"Your equal-depth comparison can't reach significance."* | Stated: three pairs floor the two-sided sign test at p = 0.25. | Complete seeds 456 and 789 of the NACT arm; that gives 5 pairs and p = 0.0625 if unanimous. Still not 0.05, so consider a different design |
| C4 | *"The ablation removes six things at once."* | Stated; variants C, D, E designed and never run. | Run them. Each is one 19-minute training run at n = 12 |
| C5 | *"Is the effect the representation or the sequence length?"* | **Not separated**, and now acknowledged in §IX. The coordinate-token layout changes both the identity mechanism *and* halves L. | Add a learned absolute position embedding to the *baseline* digit layout, keeping L = 2n+2. One 19-minute training run at n = 12. **Do this before submitting anywhere competitive** |
| C6 | *"Why no distinguisher recovery?"* | Stated as not implemented. | Acknowledge; implementing it is real work |
| C7 | *"Your data pipeline differs from the reference."* | Table IV states all seven differences. | Nothing more needed |
| C8 | *"What happens at larger h?"* | Not tested. The paper names it as the only direction that becomes cryptanalytically interesting. | Run n = 20, h = 3 and h = 4. Cheap, and directly on the paper's own argument |
| C9 | *"Capacity was never varied."* | Stated in §IX. | A 2 M and an 8 M variant at n = 12 would bound it |

---

## D. Text and presentation checks

| # | Check | Status |
|---|---|---|
| D1 | Page count 12–14 | **14 pages** — at the top of the range. If a venue imposes a lower limit, §II and §V-C compress most easily |
| D2 | Every figure and table referenced in the text | Verified: the builder raises if a cross-reference has no target, and all 5 figures and 9 tables are cited |
| D3 | Every bibliography entry cited | Verified: the builder raises if an entry is never cited |
| D4 | Equations numbered and referenced | 9 numbered equations; (1)–(9) all referenced except (1), which is referenced implicitly as the problem statement — **check whether your venue requires every numbered equation to be cited** |
| D5 | No fabricated DOIs, pages or venues | One DOI appears (PICANTE, verified). Volume and page numbers were omitted where unverified rather than guessed |
| D6 | Title claim strength | "Compact … Using Coordinate-Aware Input Representation" is supported. It claims no superiority and no break. A more precise alternative, if you want the scope in the title: *"…at Small Lattice Dimensions: A Diagnostic Study"*. Your call |
| D7 | Abstract contains no unsupported claim | Reviewed line by line against the evidence audit |
| D8 | British/American spelling consistent | British throughout ("minimises", "centred", "analyse"). Several venues require American; a global pass will be needed |
| D9 | Figures legible at print size | All five inspected at 400 dpi; smallest type is ~6.3 pt in figure annotations, which is at the lower limit. If the venue reduces figures, regenerate at a larger base font |
| D10 | Colour figures readable in greyscale | The palette was chosen for this, but **not formally checked**. Print one page in greyscale before submitting |

---

## E. Venue suggestions

These are suggestions based on the paper's actual scope and evidence, not
verified submission requirements. **Check each venue's current call, template,
page limit, anonymity policy and deadline directly — do not rely on this
table.** No venue is likely to be an easy acceptance: the paper's central
honest finding is that its own experimental regime is not cryptanalytically
hard, which is good science and a hard sell.

| Venue type | Examples | Fit |
|---|---|---|
| **Applied-crypto / ML-security workshops** | ACM AISec (co-located with CCS); ACM WAHC; the workshop track at PQCrypto | **Best fit.** The scope — a representational finding plus a methodological caution — is workshop-shaped, and the audience is the right one |
| **IACR ePrint** | eprint.iacr.org | Worth doing regardless and immediately. No review, establishes a date, and the community reads it |
| **arXiv** | cs.CR, cs.LG | Same; do both |
| **ML venues, workshop track** | NeurIPS / ICML workshops on ML for science, or on security | The SALSA line itself published at NeurIPS. A workshop paper is realistic; the main track would need C1 and C5 resolved first |
| **Indian national conferences / journals** | Defence Science Journal (DRDO); national cryptology symposia | Plausible, and the DRDO connection is natural — but A6 must be settled first |
| **Main-track crypto conferences** | CRYPTO, EUROCRYPT, CHES, CCS, S&P | **Not yet.** Without a cryptanalytic result these will reject on scope |

Before submitting anywhere, reformat into the venue's official template. The
supplied `.tex` is IEEE-conference-shaped as a starting point, not because any
venue requires it.

---

## F. Deliverables and their status

| File | Status |
|---|---|
| `Modified_SALSA_Research_Paper.pdf` | **Built and inspected.** 14 pages, A4 (209.9 × 297.0 mm), 5 embedded figures, selectable text, embedded font subsets |
| `Modified_SALSA_Research_Paper.html` | Built. This is the source the PDF is rendered from |
| `Modified_SALSA_Research_Paper.tex` | Built. **NEVER COMPILED** — no TeX distribution exists in the environment that produced it. See G1 |
| `Modified_SALSA_Research_Paper.bib` | Built. 22 entries, all verified |
| `Modified_SALSA_Research_Paper.docx` | See G2 |
| `Modified_SALSA_Figures/` | 5 figures at 400 dpi, generated from the artifacts by `make_paper_figures.py` |
| `Modified_SALSA_Evidence_Audit.md` | Written |
| `Modified_SALSA_Paper_Review_Checklist.md` | This file |

---

## G. Known limitations of the deliverables themselves

**G1 — The LaTeX source has not been compiled.** No `pdflatex`, `xelatex`,
`lualatex`, `tectonic` or `pandoc` is installed in the environment that
produced these files, so `Modified_SALSA_Research_Paper.tex` has never been run
through LaTeX. It is generated from the same content module as the HTML, so the
wording, numbering and citations match by construction, but **the LaTeX has not
been proved to compile**. Expect to fix packaging details: it assumes
`IEEEtran`, `graphicx`, `amsmath`, `amssymb`, `booktabs`, `threeparttable`,
`algorithm`, `url` and `hyperref`. The verified artefact is the PDF. Compile
the `.tex` and diff it against the PDF before trusting it.

**G2 — The DOCX.** Microsoft Word is available in this environment and a DOCX
can be produced from the HTML through Word automation. Word will reflow the
two-column layout and will not reproduce the figure placement or the equation
rendering faithfully; treat any DOCX as an editing convenience for
track-changes review, not as a submission artefact.

**G3 — The HTML and LaTeX are parallel emissions of one source, not one
derived from the other.** They are generated in the same build from
`manuscript.py`, so they cannot differ in wording. Display equations, however,
carry two hand-written renderings — one HTML, one LaTeX — so a *typographical*
difference between them is possible even though the content is the same. The
nine display equations should be read side by side once.

**G4 — Figure 2 is a schematic drawn from the implementation**, not a diagram
reproduced from any published source. No figure from the SALSA paper has been
reproduced, adapted or traced.

---

## H. Originality and attribution

The manuscript was written from the implementation and the recorded artifacts.
No paragraph, sentence or distinctive phrasing was taken from the SALSA paper,
from the project's internal reports, or from the repository README. The SALSA
paper is cited wherever the manuscript relies on, describes or compares against
it, and §II-B describes its method in our own words with attribution.

Passages that **require human technical review before submission**, because
they make a judgement rather than report a measurement:

1. **§I, "It appears to be representation."** A causal reading of a comparison
   with one controlled difference but limited seeds. The hedging is deliberate;
   check you are comfortable with it.
2. **§VI-D, the sparse-input interpretation.** The claim that the probes are
   "the extreme point of a continuum" is an interpretation of the lift curve,
   not a measurement.
3. **§VII, the whole section.** The enumeration argument is the paper's most
   consequential claim and the one most likely to be contested. Verify the
   numbers independently by re-running the script.
4. **§VIII-A, the three reasons a checkpoint cannot be reused at larger n.**
   Read from the code; confirm against the current code before submission.
5. **§VIII-C, the super-linear sample-requirement argument.** Structural
   reasoning, not measurement, and labelled as such.
6. **Table IV.** Characterising another group's released code. Re-read the
   reference implementation and confirm each row still holds for the current
   release before publishing a criticism of it.

No claim is made anywhere about evading plagiarism or AI-detection software,
and no such claim should be added. Run the manuscript through whatever
similarity checker your institution requires; the evidence audit gives the
source for every number so that any flagged passage can be traced.
