# SALSA 2.0 internship report — validation checklist

Comparison of the generated report against the reference report
(*Summer Training Report, Solid State Physics Laboratory, DRDO*, SlideShare
item 107247311) and against the scientific-integrity requirements set for this
deliverable.

---

## 1. Reference-format conformance

The reference was inspected before drafting. Its observed structure is: title
page → declaration → acknowledgements → list of figures → list of tables →
table of contents → numbered chapters with hierarchical numbering → appendix →
bibliography; serif typography, figure captions below the figure, equations
centred on their own lines, numbered citation style, approximately 59 pages.

| Reference feature | Status | Note |
|---|---|---|
| Title page: title, host organisation, degree, institution, student name and roll number, supervisor name and designation | **Matched** | Roll number is a marked placeholder |
| Declaration page | **Matched** | Student declaration; signature line left blank |
| Acknowledgements | **Matched** | |
| List of Figures | **Matched** | Generated as a Word field with live page numbers |
| List of Tables | **Matched** | Generated as a Word field with live page numbers |
| Table of Contents with page numbers | **Matched** | Generated as a Word field, heading levels 1–3 |
| Hierarchical section numbering (1.1, 1.2.1) | **Matched** | |
| Serif typography, bold headings | **Matched** | Times New Roman 12 pt body, 1.5 line spacing |
| Figure captions below the figure | **Matched** | Word Caption style, so the List of Figures collects them |
| Tables with borders and aligned columns | **Matched** | Numeric columns right-aligned |
| Equations centred on separate lines | **Matched** | Numbered (3.1) … (7.1) and referenced in the text |
| Numbered citation style in a bibliography | **Matched** | 15 references, cited as [1] … [15] |
| Appendix | **Matched** | Three appendices, A–C |
| Approximate length ≈ 59 pages | **Comparable** | See the build output for the exact page count |

### Deliberate additions beyond the reference

| Addition | Reason |
|---|---|
| Certificate page | Expected for a DRDO internship submission. **Marked as a placeholder**; it must be reproduced on the official SAG format and signed. No signature, seal or approval number was generated. |
| Abstract | Requested in the brief; standard for a research report. |
| List of Abbreviations and Symbols | Requested in the brief; the report uses substantial notation. |
| Roman numerals in the front matter, arabic restarting at 1 in the body | Standard practice; the reference's numbering scheme was not legible from the source. |

---

## 2. Items requiring your input or your supervisor's confirmation

These appear in the document as visible bracketed placeholders. Search the
`.docx` for `[` to find them all.

| Placeholder | Where | Needed from |
|---|---|---|
| `[ROLL NUMBER - TO BE SUPPLIED]` | Title page, declaration | You |
| `[DATE - TO BE SUPPLIED]` | Certificate, declaration | You |
| `[MONTH YEAR OF SUBMISSION - TO BE CONFIRMED]` | Title page | You |
| `[UNIVERSITY LOGO - TO BE INSERTED]` | Title page | You |
| Certificate wording and format | Certificate page | **Supervisor / SAG** — the official format must replace the draft |
| Internship dates 20 April 2026 – 20 October 2026 | Certificate | Confirm against the official internship document |
| Degree-requirement wording on the title page | Title page | Confirm against your university's current format |

**Not generated, by design:** any supervisor signature, seal, approval number,
registration number, or statement that the work is approved by DRDO or SAG.

---

## 3. Scientific-integrity verification

| Check | Result |
|---|---|
| Every reported measurement traced to a recorded artefact | **Pass** — Appendix C.2 maps each class of value to its file |
| No fabricated results | **Pass** — all figures read from `results/`; none typed by hand |
| Predictions explicitly labelled | **Pass** — Chapter 7 classifies every statement as observed, literature, theoretical or extrapolated |
| No invented values for unmeasured quantities | **Pass** — loss, acc_tau, probe success and decode validity above n = 20 are stated as *not reliably estimable*; no number is given |
| Model accuracy kept separate from secret recovery | **Pass** — Tables 6.1 and 6.2 are separate, and §6.1 states the distinction explicitly |
| Single successes not presented as rates | **Pass** — stated in the abstract, §6.2, §6.8, §8.1 and the conclusion |
| Multi-secret result correctly attributed | **Pass** — §6.7 states the three checkpoints are full NACT (4,241,288), not Modified NACT |
| Enumeration baseline at h = 2 disclosed | **Pass** — §8.3, and repeated in the abstract and conclusion |
| No claim of a practical cryptographic break | **Pass** — explicitly denied in the abstract, §1.4, §6.8 and Chapter 9 |
| Citations verified against primary sources | **Pass** — all 15 checked; see §4 below |
| Architecture diagram matches the implementation | **Pass** — widths, heads, loop counts and parameter totals read from the resolved run header |
| Divergences from the published method recorded | **Pass** — §2.5 lists five material divergences |
| No confidential or restricted material | **Pass** — contains only this project's own code, configuration and results. **You should still confirm with your supervisor before external release.** |

---

## 4. Citation verification

Each reference was checked against a primary source (publisher page or arXiv
abstract) rather than from memory.

| # | Reference | Verified against |
|---|---|---|
| 1 | Regev, *J. ACM* 56(6), 2009 | Standard citation |
| 2 | Lyubashevsky, Peikert, Regev, EUROCRYPT 2010, LNCS 6110 | Standard citation |
| 3 | Wenger, Chen, Charton, Lauter, SALSA, NeurIPS 2022 | NeurIPS proceedings page; arXiv:2207.04785; ePrint 2022/935 |
| 4 | Vaswani et al., NIPS 2017 | arXiv:1706.03762 |
| 5 | Dehghani et al., Universal Transformers, ICLR 2019 | arXiv:1807.03819 |
| 6 | Csordás, Irie, Schmidhuber, ICLR 2022 | arXiv:2110.07732 — title and authors confirmed |
| 7 | Hendrycks, Gimpel, GELUs | arXiv:1606.08415 |
| 8 | Charton, *TMLR*, 2022 | arXiv:2112.01898 — TMLR publication confirmed |
| 9 | Albrecht, Player, Scott, *J. Math. Cryptol.* 9(3), 2015 | Standard citation |
| 10 | NIST FIPS 203 | csrc.nist.gov — publication date 13 August 2024 confirmed |
| 11 | Su et al., RoFormer | arXiv:2104.09864 — full author list confirmed |
| 12 | Zhang, Sennrich, RMSNorm, NeurIPS 2019 | arXiv:1910.07467 |
| 13 | Li, Sotáková, Wenger, Malhou, Garcelon, Charton, Lauter, SALSA PICANTE | arXiv:2303.04178 — full author list confirmed |
| 14 | Li, Wenger, Allen-Zhu, Charton, Lauter, SALSA VERDE | arXiv:2306.11641 — full author list confirmed |
| 15 | Loshchilov, Hutter, AdamW, ICLR 2019 | arXiv:1711.05101 |

No DOI was invented. Where a DOI was not verified, the citation gives the venue
and the arXiv identifier instead.

---

## 5. Figures and tables

All six required figures are generated from project data or from the
implementation, embedded in the document at readable size, captioned, numbered
and referenced in the body. Each carries a short note stating what it shows and
what it does not establish.

| Figure | Content | Source |
|---|---|---|
| 4.1 | Overall SALSA 2.0 workflow with the secret-free boundary | Implementation structure |
| 4.2 | Modified NACT architecture, token flow, shared blocks, removed components | Resolved run header + `salsa/models/nact.py` |
| 4.3 | Parameter counts of the four models, log scale | Recorded parameter counts |
| 5.1 | Experimental workflow and reproducibility gate | Implementation structure |
| 6.1 | Observed validation loss and acc_tau at n = 12 and n = 20 | `summary.json` of both runs |
| 7.1 | Relative computational burden, measured vs extrapolated vs theoretical | Measured throughput + stated cost models |

Tables: 2.1, 4.1, 4.2, 5.1, 5.2, 6.1–6.6, 7.1–7.3, 8.1, A.1, B.1, B.2, C.1 —
nineteen in total, each referenced in the text. The brief's required tables are
all present: related work (2.1), experimental configuration (5.2), model
architecture and parameters (4.1, 4.2), measured results (6.1–6.3), scalability
forecast (7.3), limitations and future work (8.1).

---

## 6. Final steps for you to complete

1. Fill the four bracketed placeholders listed in §2.
2. Replace the certificate page with the official SAG format and obtain the
   supervisor's signature.
3. Confirm the internship dates against the official document.
4. Open the `.docx`, press `Ctrl+A` then `F9`, and choose **Update entire
   table** for each list, to refresh the Table of Contents, List of Figures and
   List of Tables after your edits. Re-export the PDF afterwards.
5. Have your supervisor confirm that the technical content is cleared for
   inclusion before any external circulation.

---

## 7. Regenerating the document

```bash
python report/make_figures.py                 # regenerates the six figures
powershell -ExecutionPolicy Bypass -File report/build_report.ps1
```

The build script drives Microsoft Word through COM automation; no third-party
converter is installed in this environment and none is required. It produces
both `SALSA_2.0_Internship_Report.docx` and `SALSA_2.0_Internship_Report.pdf`
from `report/salsa2_report.html`.
