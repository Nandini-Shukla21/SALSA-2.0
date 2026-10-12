"""Figures for the SALSA 2.0 internship report.

READ-ONLY. Trains nothing, loads no checkpoint. Measured values are read from
recorded artifacts under ``results/``; diagrams are drawn from the architecture
as implemented in ``salsa/models/nact.py`` and the resolved run headers.

Run::

    python report/make_figures.py

Writes figures 1-6 into ``report/figures/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figures"

# Validated categorical palette (light surface) -- see dataviz reference palette.
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8982"
GRID = "#e4e3de"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=220, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote report/figures/{name}")


# --------------------------------------------------------------------------- #
# Diagram primitives
# --------------------------------------------------------------------------- #


def box(ax, x, y, w, h, text, *, face="#f4f7fb", edge=BLUE, fontsize=8.6,
        weight="normal", textcolor=INK, style="round,pad=0.012,rounding_size=0.016"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=1.5,
                                facecolor=face, edgecolor=edge, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=textcolor, zorder=4, fontweight=weight,
            linespacing=1.45)


def arrow(ax, start, end, *, color=INK_2, style="-|>", lw=1.5, rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, linewidth=lw,
                                 color=color, mutation_scale=12, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}"))


def canvas(figsize, title):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(SURFACE)
    if title:
        ax.text(0, 1.02, title, fontsize=11.5, color=INK, fontweight="bold",
                ha="left", va="bottom")
    return fig, ax


# --------------------------------------------------------------------------- #
# Figure 1 -- overall SALSA 2.0 workflow
# --------------------------------------------------------------------------- #


def figure_1() -> None:
    fig, ax = canvas((9.6, 6.4), "")

    box(ax, 0.02, 0.84, 0.28, 0.11,
        "RLWE instance generation\n"
        "$A$ = circulant rows,  $s$ binary with $h$ ones\n"
        "$b = As + e \\;\mathrm{mod}\; q$   ($q{=}251$, $\\sigma{=}3$)",
        face="#f4f7fb", edge=BLUE)
    box(ax, 0.36, 0.84, 0.26, 0.11,
        "LatticeCodec\nbase-81, LSB-first, fixed width\n"
        "$(A,b) \\rightarrow$ token ids",
        face="#f4f7fb", edge=BLUE)
    box(ax, 0.68, 0.84, 0.30, 0.11,
        "Modified NACT\n4,238,208 parameters\n"
        "encoder $n{+}2$ tokens  $\\rightarrow$  $\\hat{b}$",
        face="#eaf4ff", edge=BLUE, weight="bold")

    arrow(ax, (0.30, 0.895), (0.355, 0.895))
    arrow(ax, (0.62, 0.895), (0.675, 0.895))

    # Training branch
    box(ax, 0.68, 0.655, 0.30, 0.10,
        "Stage 1  TRAINING\ncross-entropy on digits of $b$\n"
        "public samples only, 100,032 of them",
        face="#fdf1ec", edge=ORANGE)
    arrow(ax, (0.83, 0.835), (0.83, 0.76), color=ORANGE)
    arrow(ax, (0.985, 0.705), (1.00, 0.705), color=ORANGE)
    ax.add_patch(FancyArrowPatch((0.995, 0.705), (0.995, 0.895), arrowstyle="-",
                                 color=ORANGE, linewidth=1.4, zorder=2))
    ax.text(1.005, 0.80, "weight\nupdate", fontsize=7.4, color=ORANGE,
            ha="left", va="center", rotation=90)

    # Recovery branch
    box(ax, 0.02, 0.60, 0.30, 0.13,
        "Stage 2  SECRET RECOVERY\n"
        "probe $a = K\\,e_i$ so that $a\\cdot s = K s_i$\n"
        "anchored decision on $\\hat{b}$ vs $\\{0, K\\}$\n"
        "separation-weighted vote over $K$",
        face="#eafaf3", edge=AQUA)
    box(ax, 0.38, 0.625, 0.24, 0.085,
        "candidate secret  $\\tilde{s} \\in \\{0,1\\}^n$",
        face="#ffffff", edge=AQUA, weight="bold")
    arrow(ax, (0.68, 0.80), (0.33, 0.695), color=AQUA, rad=0.10)
    ax.text(0.50, 0.775, "frozen model queried on\nchosen probe inputs",
            fontsize=7.6, color=AQUA, ha="center", va="center")
    arrow(ax, (0.32, 0.665), (0.375, 0.665), color=AQUA)

    # Verification
    box(ax, 0.38, 0.44, 0.30, 0.11,
        "Stage 3  RESIDUAL VERIFICATION\n"
        "$r = (b - A\\tilde{s}) \\;\mathrm{mod}\; q$, centred\n"
        "accept iff  $\\mathrm{std}(r) \\approx \\sigma$,  not  $\\approx q/\\sqrt{12}$",
        face="#eafaf3", edge=AQUA)
    arrow(ax, (0.50, 0.620), (0.50, 0.555), color=AQUA)

    box(ax, 0.74, 0.455, 0.24, 0.08,
        "ACCEPT / REJECT\nno ground truth used",
        face="#ffffff", edge=AQUA, weight="bold")
    arrow(ax, (0.68, 0.495), (0.735, 0.495), color=AQUA)

    # Evaluation, last and separate
    box(ax, 0.38, 0.245, 0.30, 0.10,
        "Stage 4  EVALUATION (reporting only)\n"
        "loads the true secret, runs LAST\n"
        "coordinate accuracy, Hamming distance",
        face="#f3f3f1", edge=MUTED)
    arrow(ax, (0.50, 0.435), (0.50, 0.350), color=MUTED)

    # Isolation boundary
    ax.add_patch(FancyBboxPatch((0.005, 0.415), 1.02, 0.555,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                linewidth=1.3, linestyle=(0, (5, 4)),
                                facecolor="none", edgecolor=AQUA, zorder=1))
    ax.text(0.012, 0.985,
            "SECRET-FREE REGION — stages 1-3 never receive the true secret or the error vector",
            fontsize=8.2, color=AQUA, ha="left", va="bottom", fontweight="bold")

    ax.text(0.012, 0.19,
            "The true secret enters only at stage 4, after a candidate has been produced and\n"
            "independently verified. Running the pipeline with evaluate=False omits stage 4 entirely.",
            fontsize=8.0, color=INK_2, ha="left", va="top")
    save(fig, "fig1_workflow.png")


# --------------------------------------------------------------------------- #
# Figure 2 -- Modified NACT architecture
# --------------------------------------------------------------------------- #


def figure_2() -> None:
    fig, ax = canvas((11.4, 7.8), "")
    LW, LX = 0.305, 0.015          # left (encoder) column
    MX, MW = 0.345, 0.205          # middle annotation lane
    RX, RW = 0.580, 0.405          # right (decoder) column

    # ---- encoder front end ------------------------------------------------ #
    box(ax, LX, 0.890, LW, 0.078,
        "codec input ids,  $2n+2$ tokens (V1 layout)\n"
        "$\\langle bos\\rangle\\; d_0^{(0)} d_1^{(0)} \\ldots d_0^{(n-1)} d_1^{(n-1)} \\langle eos\\rangle$",
        face="#f4f7fb", edge=BLUE, fontsize=8.0)
    box(ax, LX, 0.788, LW, 0.072,
        "lossless digit reconstruction\n$a_i = d_0^{(i)} + 81\\, d_1^{(i)}$",
        face="#f4f7fb", edge=BLUE, fontsize=8.0)
    box(ax, LX, 0.686, LW, 0.072,
        "digit embeddings, summed over slots\n$2 \\times 81 \\times 512$",
        face="#f4f7fb", edge=BLUE, fontsize=8.0)
    box(ax, LX, 0.584, LW, 0.072,
        "+ absolute coordinate embedding\n$\\mathrm{CoordEmb}[i]$,  $128 \\times 512$",
        face="#eaf4ff", edge=BLUE, weight="bold", fontsize=8.0)
    box(ax, LX, 0.492, LW, 0.062,
        "encoder sequence,  $n+2$ tokens",
        face="#ffffff", edge=BLUE, weight="bold", fontsize=8.4)
    for y in (0.890, 0.788, 0.686, 0.584):
        arrow(ax, (LX + LW / 2, y), (LX + LW / 2, y - 0.030))

    box(ax, LX, 0.232, LW, 0.250,
        "SHARED ENCODER LAYER\napplied $T_e = 2$ times\n\n"
        "RMSNorm $\\rightarrow$ self-attention\n8 heads, $d_e = 512$, RoPE, no biases\n\n"
        "RMSNorm $\\rightarrow$ FFN\n$512 \\rightarrow 2048 \\rightarrow 512$, GELU\n\n"
        "copy gate on every sub-layer",
        face="#eaf4ff", edge=BLUE, weight="bold", fontsize=7.7)
    arrow(ax, (LX + LW / 2, 0.490), (LX + LW / 2, 0.484))
    box(ax, LX, 0.152, LW, 0.062,
        "encoder memory\n$(n+2) \\times 512$", face="#ffffff", edge=BLUE, fontsize=8.2)
    arrow(ax, (LX + LW / 2, 0.230), (LX + LW / 2, 0.216))

    # ---- middle lane ------------------------------------------------------ #
    ax.text(MX, 0.900,
            "ONE TOKEN PER\nCOORDINATE\n\n"
            "the encoder sequence\nis halved relative to\nthe V1 layout:\n"
            "$2n+2 \\;\\rightarrow\\; n+2$",
            fontsize=8.2, color=BLUE, ha="left", va="top", fontweight="bold",
            linespacing=1.5)
    box(ax, MX, 0.515, MW, 0.185,
        "REMOVED by ablation F\nto give Modified NACT\n\n"
        "numerical projection\n$4 \\rightarrow 512$  (2,560)\n\n"
        "learned zero vector  (512)\n\n"
        "sparse attention bias  (8)",
        face="#f3f3f1", edge=MUTED, fontsize=7.6, textcolor=MUTED)
    ax.text(MX, 0.470,
            "one parameter set,\nreused $T_e$ times:\neffective depth is free",
            fontsize=7.8, color=BLUE, ha="left", va="top", linespacing=1.5)
    ax.text(MX, 0.300, "cross-attention\nkeys and values",
            fontsize=7.8, color=BLUE, ha="left", va="top", linespacing=1.5)
    arrow(ax, (LX + LW + 0.004, 0.183), (RX - 0.004, 0.520), color=BLUE, rad=-0.22)

    # ---- decoder column --------------------------------------------------- #
    box(ax, RX, 0.890, RW, 0.078,
        "target ids (teacher forcing)\n"
        "$\\langle bos\\rangle\\; d_0^{(b)} d_1^{(b)} \\;\\langle eos\\rangle$",
        face="#fdf1ec", edge=ORANGE, fontsize=8.0)
    box(ax, RX, 0.800, RW, 0.060,
        "decoder embedding  $85 \\times 128$", face="#fdf1ec", edge=ORANGE, fontsize=8.2)
    box(ax, RX, 0.480, RW, 0.290,
        "SHARED DECODER LAYER   -   applied $T_d = 2$ times\n\n"
        "RMSNorm $\\rightarrow$ causal self-attention\n4 heads, $d_d = 128$, RoPE\n\n"
        "RMSNorm $\\rightarrow$ cross-attention to encoder memory\n"
        "queries from 128, keys and values from 512\n\n"
        "RMSNorm $\\rightarrow$ FFN  $128 \\rightarrow 512 \\rightarrow 128$, GELU\n\n"
        "copy gate on every sub-layer",
        face="#fdf1ec", edge=ORANGE, weight="bold", fontsize=8.0)
    box(ax, RX, 0.386, RW, 0.060,
        "output projection  $128 \\times 85$  (untied)",
        face="#fdf1ec", edge=ORANGE, fontsize=8.2)
    box(ax, RX, 0.292, RW, 0.060,
        "predicted digits of $b$  $\\rightarrow$  $\\hat{b} \\in \\mathbb{Z}_q$",
        face="#ffffff", edge=ORANGE, weight="bold", fontsize=8.4)
    for y in (0.890, 0.800):
        arrow(ax, (RX + RW / 2, y), (RX + RW / 2, y - 0.030), color=ORANGE)
    arrow(ax, (RX + RW / 2, 0.476), (RX + RW / 2, 0.446), color=ORANGE)
    arrow(ax, (RX + RW / 2, 0.382), (RX + RW / 2, 0.352), color=ORANGE)

    ax.text(LX, 0.090,
            "Encoder 3,822,592 parameters   |   decoder 415,616   |   total 4,238,208.\n"
            "The coordinate table is sized by max_coordinates = 128, not by $n$, so the parameter count is identical\n"
            "at $n = 12$ and $n = 20$. Only the encoder sequence length changes, from 14 to 22 tokens.",
            fontsize=8.2, color=INK_2, ha="left", va="top", linespacing=1.6)
    save(fig, "fig2_architecture.png")


# --------------------------------------------------------------------------- #
# Figure 3 -- experimental workflow
# --------------------------------------------------------------------------- #


def figure_3() -> None:
    fig, ax = canvas((10.6, 4.6), "")

    steps = [
        (0.010, "YAML config + seed\nnact_n12_h2_te2_F\nfingerprint f148ac74", "#f4f7fb", BLUE),
        (0.208, "deterministic data\ntrain 100,032\nvalid 2,048, no reuse", "#f4f7fb", BLUE),
        (0.406, "training, 20 epochs\nAdamW, lr 5e-4, cosine\nbatch 64, 1,563 steps", "#fdf1ec", ORANGE),
        (0.604, "best checkpoint\nselected on valid_loss\n(epoch 19)", "#fdf1ec", ORANGE),
        (0.802, "prediction metrics\nloss, acc_tau, exact,\ndecode failure rate", "#fdf1ec", ORANGE),
    ]
    for x, text, face, edge in steps:
        box(ax, x, 0.760, 0.172, 0.185, text, face=face, edge=edge, fontsize=7.2)
    for x in (0.010, 0.208, 0.406, 0.604):
        arrow(ax, (x + 0.172, 0.8525), (x + 0.206, 0.8525))

    box(ax, 0.150, 0.400, 0.235, 0.215,
        "SECRET RECOVERY\n\nfrozen model, probe inputs\nsalsa/recovery/direct.py\n"
        "no secret parameter exists",
        face="#eafaf3", edge=AQUA, fontsize=7.4)
    box(ax, 0.430, 0.400, 0.235, 0.215,
        "RESIDUAL VERIFICATION\n\ninputs $(A, b, \\tilde{s}, q, \\sigma)$ only\n"
        "salsa/verification/residual.py\nfour acceptance criteria",
        face="#eafaf3", edge=AQUA, fontsize=7.4)
    box(ax, 0.710, 0.430, 0.235, 0.155,
        "EVALUATION\n\nloads the true secret\nRUNS LAST",
        face="#f3f3f1", edge=MUTED, fontsize=7.4)
    arrow(ax, (0.690, 0.755), (0.300, 0.620), color=AQUA, rad=0.16)
    arrow(ax, (0.385, 0.5075), (0.425, 0.5075), color=AQUA)
    arrow(ax, (0.665, 0.5075), (0.705, 0.5075), color=MUTED)
    ax.text(0.400, 0.700, "the trained model is frozen\nbefore recovery begins",
            fontsize=7.2, color=AQUA, ha="center", va="center")

    box(ax, 0.150, 0.090, 0.700, 0.185,
        "REPRODUCIBILITY GATE\n\n"
        "python scripts/run_pipeline.py configs/pipeline_n12.yaml     replays the whole chain in about one second.\n"
        "The checkpoint carries a config fingerprint and a mismatched configuration is refused.\n"
        "Adding --no-evaluate runs the attack with no ground truth consulted at any point.",
        face="#ffffff", edge=INK_2, fontsize=7.4)
    arrow(ax, (0.500, 0.396), (0.500, 0.280), color=INK_2)
    save(fig, "fig3_experimental_workflow.png")


# --------------------------------------------------------------------------- #
# Figure 4 -- model comparison
# --------------------------------------------------------------------------- #


def figure_4() -> None:
    names = ["Original SALSA\n(published)", "Salsa2-GatedUT\n(V1 baseline)",
             "Salsa2-NACT\n(full)", "Modified NACT\n(final)"]
    values = [51_000_000, 4_131_200, 4_241_288, 4_238_208]
    colors = [MUTED, BLUE, AQUA, ORANGE]

    fig, ax = plt.subplots(figsize=(8.6, 4.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    bars = ax.bar(names, values, color=colors, width=0.56, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("trainable parameters (log scale)", fontsize=9.6, color=INK_2)
    ax.set_ylim(1e6, 2e8)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.8, length=0)

    labels = ["~51,000,000\n(approximate,\nas published)", "4,131,200", "4,241,288", "4,238,208"]
    for bar, label in zip(bars, labels):
        ax.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=8.6, color=INK, fontweight="semibold")

    ax.axhline(4e6, color=MUTED, linewidth=1.1, linestyle=(0, (5, 3)), zorder=2)
    ax.axhline(5e6, color=MUTED, linewidth=1.1, linestyle=(0, (5, 3)), zorder=2)
    ax.text(3.46, 4.45e6, "project budget\n4M - 5M", fontsize=8.0, color=MUTED,
            ha="right", va="center")
    save(fig, "fig4_model_comparison.png")


# --------------------------------------------------------------------------- #
# Figure 5 -- observed performance vs dimension
# --------------------------------------------------------------------------- #


def figure_5() -> None:
    n12 = load(RESULTS / "nact_ablation_F" / "nact_n12_h2_te2_F" / "seed_0" / "artifacts" / "summary.json")
    n20 = load(RESULTS / "n20_nact_f_pilot" / "nact_n20_h2_te2_F" / "seed_0" / "artifacts" / "summary.json")
    chance = n12["chance_baselines"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.2), facecolor=SURFACE)
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8.8, length=0)
        ax.set_xticks([12, 20])
        ax.set_xlim(8, 24)
        ax.set_xlabel("lattice dimension  $n$   (h = 2)", fontsize=9.4, color=INK_2)

    losses = [n12["final_validation"]["valid_loss"], n20["final_validation"]["valid_loss"]]
    ax1.axhline(chance["marginal_loss_nats"], color=MUTED, linewidth=1.4,
                linestyle=(0, (5, 3)), zorder=2)
    ax1.text(23.6, chance["marginal_loss_nats"] - 0.055,
             "marginals-only baseline 1.86498", fontsize=7.8, color=INK_2,
             ha="right", va="top")
    ax1.plot([12, 20], losses, color=BLUE, linewidth=2.0, zorder=4)
    ax1.scatter([12, 20], losses, s=78, color=BLUE, zorder=5,
                edgecolor=SURFACE, linewidth=1.8)
    for x, y in zip((12, 20), losses):
        ax1.annotate(f"{y:.4f}", xy=(x, y), xytext=(0, 11),
                     textcoords="offset points", ha="center", fontsize=9.0,
                     color=INK, fontweight="semibold")
    ax1.set_ylabel("validation loss (nats / token)", fontsize=9.4, color=INK_2)
    ax1.set_ylim(0.85, 2.0)
    ax1.set_title("(a) validation loss", loc="left", fontsize=9.6, color=INK, pad=8)

    accs = [n12["final_validation"]["valid_acc_tau"], n20["final_validation"]["valid_acc_tau"]]
    ax2.axhline(chance["chance_acc_tau"], color=MUTED, linewidth=1.4,
                linestyle=(0, (5, 3)), zorder=2)
    ax2.text(23.6, chance["chance_acc_tau"] + 0.035, "chance 0.19287",
             fontsize=7.8, color=INK_2, ha="right")
    ax2.plot([12, 20], accs, color=AQUA, linewidth=2.0, zorder=4)
    ax2.scatter([12, 20], accs, s=78, color=AQUA, zorder=5,
                edgecolor=SURFACE, linewidth=1.8)
    for x, y in zip((12, 20), accs):
        ax2.annotate(f"{y:.4f}", xy=(x, y), xytext=(0, -20),
                     textcoords="offset points", ha="center", fontsize=9.0,
                     color=INK, fontweight="semibold")
    ax2.set_ylabel("acc_tau   ($\\tau = 0.1$)", fontsize=9.4, color=INK_2)
    ax2.set_ylim(0.0, 1.10)
    ax2.set_title("(b) acc_tau", loc="left", fontsize=9.6, color=INK, pad=8)

    fig.text(0.5, -0.045,
             "Two observations per panel, one training seed and one secret at each dimension. "
             "The connecting line joins the two measured points;\nit is not a fitted trend and must not be read as a performance curve.",
             fontsize=8.0, color=INK_2, ha="center", va="top")
    fig.tight_layout()
    save(fig, "fig5_observed_performance.png")


# --------------------------------------------------------------------------- #
# Figure 6 -- scalability forecast
# --------------------------------------------------------------------------- #


def figure_6() -> None:
    n12 = load(RESULTS / "nact_ablation_F" / "nact_n12_h2_te2_F" / "seed_0" / "artifacts" / "summary.json")
    n20 = load(RESULTS / "n20_nact_f_pilot" / "nact_n20_h2_te2_F" / "seed_0" / "artifacts" / "summary.json")

    dims = [12, 20, 30, 50, 70, 90, 110, 128]
    ms12 = 1000.0 / n12["throughput"]["samples_per_second"]
    ms20 = 1000.0 / n20["throughput"]["samples_per_second"]
    slope = (ms20 - ms12) / ((20 + 2) - (12 + 2))
    intercept = ms12 - slope * (12 + 2)
    ref = intercept + slope * 22

    empirical = [(intercept + slope * (n + 2)) / ref for n in dims]
    d = 512
    def flop(n):
        L = n + 2
        return L * d * d + L * L * d
    theoretical = [flop(n) / flop(20) for n in dims]

    fig, ax = plt.subplots(figsize=(9.2, 5.0), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.8, length=0)

    ax.axvspan(20, 134, color=MUTED, alpha=0.065, zorder=1)
    ax.axvline(20, color=MUTED, linewidth=1.1, linestyle=(0, (4, 3)), zorder=2)
    ax.axvline(128, color=MUTED, linewidth=1.0, linestyle=(0, (1, 2.2)), zorder=2)
    ax.text(126.5, 7.6, "max_coordinates = 128 ", fontsize=7.8, color=MUTED,
            rotation=90, ha="right", va="top")

    ax.plot(dims, theoretical, color=AQUA, linewidth=2.0, linestyle=(0, (1, 2)), zorder=4)
    ax.scatter(dims, theoretical, s=42, color=AQUA, marker="^", zorder=5)
    ax.plot(dims, empirical, color=ORANGE, linewidth=2.0, linestyle=(0, (5, 3)), zorder=4)
    ax.scatter(dims[2:], empirical[2:], s=54, facecolor="none", edgecolor=ORANGE,
               linewidth=1.9, marker="o", zorder=5)
    ax.scatter([12, 20], [ms12 / ref, 1.0], s=92, color=BLUE, zorder=7,
               edgecolor=SURFACE, linewidth=1.8)
    for x, y in ((12, ms12 / ref), (20, 1.0)):
        ax.annotate(f"{y:.2f}x", xy=(x, y), xytext=(0, 13),
                    textcoords="offset points", ha="center", fontsize=9.0,
                    color=INK, fontweight="semibold")

    ax.annotate(f"theoretical encoder FLOP proxy\n$Ld^2 + L^2 d$  $\\rightarrow$  {theoretical[-1]:.2f}x",
                xy=(110, theoretical[-2]), xytext=(60, 6.1), fontsize=8.4,
                color=INK_2, arrowprops=dict(arrowstyle="-", color=AQUA, linewidth=1.2))
    ax.annotate(f"extrapolated wall-clock, linear in $L$\n$\\rightarrow$  {empirical[-1]:.2f}x",
                xy=(112, empirical[-2] + 0.05), xytext=(56, 1.75), fontsize=8.4,
                color=INK_2, arrowprops=dict(arrowstyle="-", color=ORANGE, linewidth=1.2))
    ax.text(74, 4.4, "NO MODEL TRAINED BEYOND n = 20", fontsize=10.0,
            color=INK_2, ha="center", fontweight="bold")

    ax.set_xticks(dims)
    ax.set_xlim(8, 134)
    ax.set_ylim(0, 8.2)
    ax.set_xlabel("lattice dimension  $n$", fontsize=9.6, color=INK_2)
    ax.set_ylabel("cost per 100,000 samples,\nrelative to the measured $n = 20$ run",
                  fontsize=9.4, color=INK_2)
    ax.legend(handles=[
        Line2D([], [], color=BLUE, marker="o", linewidth=0, markersize=9,
               label="measured wall-clock (Modified NACT)"),
        Line2D([], [], color=ORANGE, marker="o", linewidth=2.0, markersize=8,
               linestyle=(0, (5, 3)), markerfacecolor="none", markeredgewidth=1.9,
               label="extrapolated wall-clock"),
        Line2D([], [], color=AQUA, marker="^", linewidth=2.0, markersize=8,
               linestyle=(0, (1, 2)), label="theoretical FLOP proxy"),
        Patch(facecolor=MUTED, alpha=0.065, label="no experimental evidence"),
    ], loc="upper left", frameon=False, fontsize=8.6, labelcolor=INK_2,
        bbox_to_anchor=(0.0, -0.16), ncol=2)

    fig.text(0.012, -0.30,
             "Excluded from both curves: the number of training samples required at each dimension, which is unknown and dominates total cost;\n"
             "the decoder, the optimiser step, data generation, validation passes and interpreter overhead. Roughly half of the measured\n"
             "per-sample time at n = 20 is independent of dimension, which is why the FLOP proxy rises faster than the wall-clock extrapolation.",
             fontsize=7.8, color=MUTED, ha="left", va="top")
    save(fig, "fig6_scalability_forecast.png")


if __name__ == "__main__":
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    figure_6()
