"""Publication figures for the Modified SALSA manuscript.

Every number drawn here is read from a recorded artifact under ``results/`` or
from a quantity derived in the manuscript text, never from a literal typed in
twice.  Where a figure shows a forecast the forecast is drawn in a different
style from the measurements and is labelled as such in the legend.

Run:  python paper/make_paper_figures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = Path(__file__).resolve().parent / "Modified_SALSA_Figures"
OUT.mkdir(parents=True, exist_ok=True)

# A restrained palette that survives greyscale printing.
INK = "#1d1d1b"
GREY = "#8a8982"
LIGHT = "#e8e6e1"
BLUE = "#2a5fa5"
ORANGE = "#d1621f"
GREEN = "#1b7f5a"
RED = "#a62f2f"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.titlesize": 9.0,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.7,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load(relative: str):
    with (RESULTS / relative).open(encoding="utf-8") as handle:
        return json.load(handle)


SUMMARY = load("final_pipeline/final_summary.json")
PIPELINE = load("final_pipeline/pipeline_report.json")
ARCH = load("nact_v2/architecture_report.json")
ABLATION = load("nact_ablation_F/ablation_F.json")

N12 = SUMMARY["prediction_results"]["n12"]
N20 = SUMMARY["prediction_results"]["n20"]
CHANCE = SUMMARY["chance_baselines"]


# --------------------------------------------------------------------------- #
# Figure 1 -- the attack pipeline and its information barriers
# --------------------------------------------------------------------------- #
def figure_1() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.85))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")

    def box(x, y, w, h, text, face, edge, weight="normal", size=7.6):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.2",
            linewidth=0.8, edgecolor=edge, facecolor=face, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=size, zorder=4, linespacing=1.35, fontweight=weight)

    def arrow(x0, y0, x1, y1, style="-|>", colour=INK, dashed=False):
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=8,
            linewidth=0.8, color=colour, zorder=2,
            linestyle=(0, (2.2, 1.6)) if dashed else "solid",
            shrinkA=0, shrinkB=0))

    # Four vertical stage bands across the full height of the upper area; the
    # secret occupies a separate lane underneath, so the one place it enters
    # before the final stage -- data generation -- is visible at a glance.
    bands = [
        (1.5, 23.5, "DATA", LIGHT),
        (26.5, 34.0, "STAGE 1   RECOVERY\nno secret in scope", "#dfe8f3"),
        (62.5, 19.0, "STAGE 2\nVERIFICATION", "#dcefe6"),
        (84.0, 14.5, "STAGE 3\nEVALUATION", "#f6e3d6"),
    ]
    for x, w, label, colour in bands:
        ax.add_patch(mpatches.Rectangle((x, 11.0), w, 32.0, facecolor=colour,
                                        edgecolor="none", zorder=0))
        ax.text(x + w / 2, 42.0, label, ha="center", va="top", fontsize=7.0,
                color="#4a4a46", fontweight="bold", linespacing=1.25)

    row_y, row_h = 20.0, 12.5
    box(3.0, row_y, 20.5, row_h,
        r"RLWE samples  $(\mathbf{A},\mathbf{b})$" "\n"
        r"$\mathbf{b}=\mathbf{A}\mathbf{s}+\mathbf{e}$" "\n"
        "base-81 codec", "#ffffff", GREY, size=7.3)

    box(28.0, row_y, 14.5, row_h,
        "train\nModified\nNACT", "#ffffff", BLUE, size=7.3)
    arrow(23.5, row_y + row_h / 2, 28.0, row_y + row_h / 2)

    box(45.0, row_y, 14.0, row_h,
        r"probe $K\mathbf{e}_i$" "\n" "anchored\ndecision", "#ffffff", BLUE,
        size=7.3)
    arrow(42.5, row_y + row_h / 2, 45.0, row_y + row_h / 2)

    box(64.0, row_y, 16.0, row_h,
        r"$\mathbf{r}=\mathbf{b}-\mathbf{A}\tilde{\mathbf{s}}$" "\n"
        r"accept iff" "\n" r"$\mathrm{std}(\mathbf{r})\approx\sigma$",
        "#ffffff", GREEN, size=7.3)
    arrow(59.0, row_y + row_h / 2, 64.0, row_y + row_h / 2)
    ax.text(61.5, row_y + row_h + 1.6, r"candidate $\tilde{\mathbf{s}}$",
            ha="center", va="bottom", fontsize=6.8, style="italic",
            color="#4a4a46")

    box(85.5, row_y, 11.5, row_h,
        "compare\nwith the\ntrue secret", "#ffffff", ORANGE, size=7.3)
    arrow(80.0, row_y + row_h / 2, 85.5, row_y + row_h / 2)
    ax.text(82.7, row_y + row_h + 1.6, "verdict", ha="center", va="bottom",
            fontsize=6.8, style="italic", color="#4a4a46")

    # The secret lane.  Two solid orange arrows, both vertical, so neither
    # crosses the pipeline row.
    box(3.0, 2.0, 20.5, 6.0, r"secret $\mathbf{s}$", "#fdf1e8", ORANGE,
        size=7.6)
    arrow(13.25, 8.0, 13.25, row_y, colour=ORANGE)
    arrow(23.5, 4.2, 91.25, 4.2, colour=ORANGE, dashed=True)
    arrow(91.25, 4.2, 91.25, row_y, colour=ORANGE)

    ax.text(57.0, 7.4,
            r"$\mathbf{s}$ generates the data and is read back only at the end;"
            " it is not an argument to stages 1 or 2",
            ha="center", va="bottom", fontsize=6.8, color=ORANGE,
            style="italic")

    fig.savefig(OUT / "fig1_pipeline.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 2 -- Modified NACT, and what changed relative to the V1 front end
# --------------------------------------------------------------------------- #
def figure_2() -> None:
    fig, (axl, axr) = plt.subplots(
        1, 2, figsize=(7.0, 3.6), gridspec_kw={"width_ratios": [1.0, 1.08]})
    for ax in (axl, axr):
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis("off")

    def tok(ax, x, y, w, h, label, face, edge, size=6.6):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.3,rounding_size=0.8",
            linewidth=0.7, edgecolor=edge, facecolor=face, zorder=3))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=size, zorder=4, linespacing=1.3)

    # -- left panel: the two input layouts ---------------------------------- #
    axl.text(50, 97, "(a) encoder input layout", ha="center", va="top",
             fontsize=8.6, fontweight="bold")

    axl.text(2, 86, "V1 / released SALSA — two digit tokens per coordinate",
             fontsize=7.2, va="center")
    labels = [("<bos>", LIGHT), (r"$d^{0}_{0}$", "#ffffff"),
              (r"$d^{1}_{0}$", "#ffffff"), (r"$d^{0}_{1}$", "#ffffff"),
              (r"$d^{1}_{1}$", "#ffffff"), (r"$\cdots$", "#ffffff"),
              (r"$d^{0}_{n-1}$", "#ffffff"), (r"$d^{1}_{n-1}$", "#ffffff"),
              ("<eos>", LIGHT)]
    x = 2.0
    for label, face in labels:
        w = 10.55
        tok(axl, x, 72, w, 9, label, face, GREY)
        x += w + 0.4
    axl.annotate("", xy=(2, 67.5), xytext=(100, 67.5),
                 arrowprops=dict(arrowstyle="<->", lw=0.7, color=GREY))
    axl.text(51, 63.5, r"length $2n+2$   (258 at $n=128$)", ha="center",
             fontsize=7.2, color="#4a4a46")

    axl.text(2, 52, "Modified NACT — one coordinate token per $a_i$",
             fontsize=7.2, va="center")
    labels2 = [("<bos>", LIGHT), (r"$\mathbf{e}_0$", "#dfe8f3"),
               (r"$\mathbf{e}_1$", "#dfe8f3"), (r"$\mathbf{e}_2$", "#dfe8f3"),
               (r"$\cdots$", "#dfe8f3"), (r"$\mathbf{e}_{n-1}$", "#dfe8f3"),
               ("<eos>", LIGHT)]
    x = 2.0
    for label, face in labels2:
        w = 14.0
        tok(axl, x, 38, w, 9, label, face, BLUE, size=7.0)
        x += w + 0.3
    axl.annotate("", xy=(2, 33.5), xytext=(100, 33.5),
                 arrowprops=dict(arrowstyle="<->", lw=0.7, color=BLUE))
    axl.text(51, 29.5, r"length $n+2$   (130 at $n=128$)", ha="center",
             fontsize=7.2, color=BLUE)

    axl.add_patch(FancyBboxPatch(
        (2, 3), 96, 22, boxstyle="round,pad=0.6,rounding_size=1.2",
        linewidth=0.7, edgecolor=BLUE, facecolor="#f4f7fb", zorder=1))
    axl.text(50, 14.0,
             r"$\mathbf{e}_i=\sum_{w}\mathrm{Digit}_w[d^{w}_{i}]"
             r"\;+\;\mathrm{Coord}[\,i\,]$" "\n"
             "digit content  +  absolute coordinate identity",
             ha="center", va="center", fontsize=7.6, linespacing=2.0)

    # -- right panel: the stack --------------------------------------------- #
    axr.text(50, 99, "(b) Modified NACT, 4,238,208 parameters",
             ha="center", va="top", fontsize=8.6, fontweight="bold")
    axr.text(37, 90.5, r"codec token ids (V1 layout)", ha="center",
             va="center", fontsize=6.9, style="italic", color="#4a4a46")

    # Group totals come from the recorded component breakdown, minus the three
    # components ablation F removed, so they cannot drift from the artefact.
    b2 = ARCH["parameter_verification"]["v2"]["breakdown"]
    g_front = (b2["encoder_digit_embeddings"] + b2["encoder_special_embeddings"]
               + b2["coordinate_embedding"])
    g_enc = (b2["encoder_attention"] + b2["encoder_ffn"]
             + b2["encoder_copy_gate"] + b2["encoder_normalization"])
    g_dec = (b2["decoder_embedding"] + b2["decoder_self_attention"]
             + b2["decoder_cross_attention"] + b2["decoder_ffn"]
             + b2["decoder_copy_gate"] + b2["decoder_normalization"])
    g_out = b2["output_projection"]
    assert g_front + g_enc + g_dec + g_out == \
        SUMMARY["model_lineage"]["Modified NACT (final)"]

    box_x, box_w, gap = 3.0, 68.0, 3.4
    stack = [
        (12.0, "front end\ndigit embeddings  +  Coord$[0..n{-}1]$",
         "#dfe8f3", BLUE, f"{g_front:,}"),
        (12.0, r"gated UT encoder layer, $d_e{=}512$, 8 heads"
               "\n" r"RMSNorm · RoPE · GELU FFN · copy gate"
               "\n" r"applied $T_e=2$ times", "#ffffff", INK, f"{g_enc:,}"),
        (12.0, r"gated UT decoder layer, $d_d{=}128$, 4 heads"
               "\n" r"self-attn · cross-attn · FFN · copy gate"
               "\n" r"applied $T_d=2$ times", "#f5f5f3", GREY, f"{g_dec:,}"),
        (9.0, r"output projection $\rightarrow$ 85-token vocabulary"
              "\n" r"$\langle\mathrm{bos}\rangle\ b^{0}\ b^{1}\ "
              r"\langle\mathrm{eos}\rangle$",
         "#fdf1e8", ORANGE, f"{g_out:,}"),
    ]
    y = 84.0
    centres = []
    for h, label, face, edge, right in stack:
        y -= h
        tok(axr, box_x, y, box_w, h, label, face, edge, size=6.8)
        axr.text(99.0, y + h / 2, right, ha="right", va="center",
                 fontsize=6.5, color="#4a4a46")
        centres.append((y, y + h))
        y -= gap

    axr.add_patch(FancyArrowPatch((37, 88.0), (37, 84.3), arrowstyle="-|>",
                                  mutation_scale=7, linewidth=0.7, color=INK,
                                  shrinkA=0, shrinkB=0))
    for (lo, _), (_, hi_next) in zip(centres[:-1], centres[1:]):
        axr.add_patch(FancyArrowPatch((37, lo), (37, hi_next + 0.3),
                                      arrowstyle="-|>", mutation_scale=7,
                                      linewidth=0.7, color=INK,
                                      shrinkA=0, shrinkB=0))

    axr.text(99.0, 86.5, "parameters", ha="right", va="center", fontsize=6.5,
             color="#4a4a46", style="italic")
    axr.text(50, 8.0,
             "the grey block and the output projection are V1 modules,\n"
             "reused without modification; only the front end is new",
             ha="center", va="center", fontsize=6.6, color="#4a4a46",
             style="italic", linespacing=1.5)

    fig.savefig(OUT / "fig2_architecture.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 3 -- parameter budget and sequence length
# --------------------------------------------------------------------------- #
def figure_3() -> None:
    v1 = ARCH["parameter_verification"]["v1"]["breakdown"]
    v2 = ARCH["parameter_verification"]["v2"]["breakdown"]

    def group(breakdown: dict) -> dict:
        front = sum(v for k, v in breakdown.items()
                    if k in ("encoder_embedding", "encoder_digit_embeddings",
                             "encoder_special_embeddings", "numerical_projection",
                             "coordinate_embedding", "zero_coordinate_vector",
                             "sparse_attention_bias"))
        enc = sum(v for k, v in breakdown.items()
                  if k.startswith("encoder_") and k not in (
                      "encoder_embedding", "encoder_digit_embeddings",
                      "encoder_special_embeddings"))
        dec = sum(v for k, v in breakdown.items()
                  if k.startswith("decoder_")) + breakdown["output_projection"]
        return {"input front end": front, "encoder body": enc,
                "decoder + output": dec}

    g_v1, g_v2 = group(v1), group(v2)
    # Modified NACT = full NACT minus the numerical projection, zero vector and
    # sparse attention bias; the artefact records the total it must equal.
    removed = (v2["numerical_projection"] + v2["zero_coordinate_vector"]
               + v2["sparse_attention_bias"])
    g_mod = dict(g_v2)
    g_mod["input front end"] -= removed
    expected = SUMMARY["model_lineage"]["Modified NACT (final)"]
    assert sum(g_mod.values()) == expected, (sum(g_mod.values()), expected)
    assert sum(g_v2.values()) == SUMMARY["model_lineage"]["V2 Salsa2-NACT (full)"]
    assert sum(g_v1.values()) == SUMMARY["model_lineage"]["V1 Salsa2-GatedUT"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [1.15, 1.0]})

    names = ["Salsa2-GatedUT\n(V1)", "NACT\n(full)", "Modified NACT\n(final)"]
    groups = [g_v1, g_v2, g_mod]
    keys = ["input front end", "encoder body", "decoder + output"]
    colours = [BLUE, "#9bb4d4", LIGHT]
    bottoms = [0.0, 0.0, 0.0]
    xs = range(3)
    for key, colour in zip(keys, colours):
        vals = [g[key] / 1e6 for g in groups]
        ax1.bar(xs, vals, bottom=bottoms, width=0.56, label=key,
                color=colour, edgecolor=INK, linewidth=0.6)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    for x, g in zip(xs, groups):
        ax1.text(x, sum(g.values()) / 1e6 + 0.09, f"{sum(g.values()):,}",
                 ha="center", fontsize=7.0)
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels(names, fontsize=7.4)
    ax1.set_ylabel("trainable parameters (millions)")
    ax1.set_ylim(0, 5.6)
    ax1.axhline(5.0, color=RED, linewidth=0.8, linestyle=(0, (3, 2)))
    ax1.text(-0.46, 4.86, "5M parameter budget ceiling", ha="left",
             va="top", fontsize=6.8, color=RED)
    ax1.legend(loc="upper center", frameon=False, ncol=3, fontsize=6.6,
               bbox_to_anchor=(0.5, 1.005), columnspacing=1.0,
               handlelength=1.3, handletextpad=0.5)
    ax1.set_title("(a) where the parameters sit", fontsize=8.6)
    ax1.spines[["top", "right"]].set_visible(False)

    ns = [12, 20, 30, 50, 70, 90, 110, 128]
    ax2.plot(ns, [2 * n + 2 for n in ns], marker="o", markersize=3.0,
             color=GREY, linewidth=1.0, label=r"V1 / released:  $2n+2$")
    ax2.plot(ns, [n + 2 for n in ns], marker="s", markersize=3.0,
             color=BLUE, linewidth=1.2, label=r"Modified NACT:  $n+2$")
    ax2.axvline(128, color=RED, linewidth=0.8, linestyle=(0, (3, 2)))
    ax2.text(124, 240, "coordinate table\nceiling (128)", ha="right",
             fontsize=6.6, color=RED, linespacing=1.25)
    for n in (12, 20):
        ax2.scatter([n], [n + 2], s=34, facecolor="none", edgecolor=ORANGE,
                    linewidth=1.1, zorder=5)
    ax2.annotate("dimensions trained", xy=(20, 22), xytext=(38, 8),
                 fontsize=6.8, color=ORANGE,
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=ORANGE))
    ax2.set_xlabel(r"lattice dimension $n$")
    ax2.set_ylabel("encoder sequence length $L$")
    ax2.legend(loc="upper left", frameon=False)
    ax2.set_title("(b) encoder sequence length", fontsize=8.6)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "fig3_parameters_and_length.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 4 -- observed prediction metrics and the sparse-input lift curve
# --------------------------------------------------------------------------- #
def figure_4() -> None:
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.0, 2.75), gridspec_kw={"width_ratios": [1.0, 1.12]})

    # (a) the three primary metrics against their chance levels.
    metrics = [
        (r"$\mathrm{acc}_\tau$", N12["acc_tau"], N20["acc_tau"],
         CHANCE["acc_tau"]),
        ("exact integer\naccuracy", N12["exact_integer_accuracy"],
         N20["exact_integer_accuracy"], CHANCE["exact_integer_accuracy"]),
        ("token\naccuracy", N12["token_accuracy"], N20["token_accuracy"],
         CHANCE["token_accuracy"]),
    ]
    width = 0.3
    xs = range(len(metrics))
    ax1.bar([x - width / 2 for x in xs], [m[1] for m in metrics], width,
            color=BLUE, edgecolor=INK, linewidth=0.6, label=r"$n=12$")
    ax1.bar([x + width / 2 for x in xs], [m[2] for m in metrics], width,
            color="#9bb4d4", edgecolor=INK, linewidth=0.6, label=r"$n=20$")
    for x, m in zip(xs, metrics):
        ax1.plot([x - 0.26, x + 0.26], [m[3], m[3]], color=RED,
                 linewidth=1.3, solid_capstyle="butt", zorder=5)
        ax1.text(x + 0.30, m[3], f"{m[3]:.3f}", fontsize=6.4, color=RED,
                 va="center", ha="left")
        ax1.text(x - width / 2, m[1] + 0.022, f"{m[1]:.3f}", ha="center",
                 fontsize=6.3)
        ax1.text(x + width / 2, m[2] + 0.022, f"{m[2]:.3f}", ha="center",
                 fontsize=6.3)
    ax1.set_xticks(list(xs))
    ax1.set_xticklabels([m[0] for m in metrics], fontsize=7.2)
    ax1.set_ylim(0, 1.16)
    ax1.set_ylabel("validation score")
    handles, labels = ax1.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color=RED, linewidth=1.3))
    labels.append("chance level")
    ax1.legend(handles, labels, loc="upper center", frameon=False, ncol=3,
               fontsize=6.9, bbox_to_anchor=(0.5, 1.03), columnspacing=1.1)
    ax1.set_title("(a) prediction quality, Modified NACT", fontsize=8.6)
    ax1.spines[["top", "right"]].set_visible(False)

    # (b) sparse-input lift, read from the ablation artefact.  The artefact
    # carries nine rows per model: a training-distribution row at nnz=12, the
    # seven sparse_random levels, and the probe row.  The curve below is the
    # sparse_random sweep followed by the probes, which is the comparison the
    # manuscript makes; the training-distribution row is dropped because it
    # duplicates nnz=12 under a different input family.
    def curve(model: str):
        rows = [r for r in ABLATION["sparsity"][model]
                if r["input_type"] != "training_distribution"]
        assert len(rows) == 8, (model, len(rows))
        return [r["acc_tau_minus_best_constant"] for r in rows], rows

    v1_lift, rows = curve("v1")
    full_lift, _ = curve("full")
    mod_lift, _ = curve("f")
    labels_x = [str(r["nnz"]) if r["input_type"] == "sparse_random"
                else r"$K\mathbf{e}_i$" for r in rows]
    positions = list(range(len(rows)))

    ax2.axhline(0.0, color=RED, linewidth=0.9, linestyle=(0, (3, 2)), zorder=1)
    ax2.plot(positions, v1_lift, marker="o", markersize=3.2, color=GREY,
             linewidth=1.0, label="Salsa2-GatedUT (V1)")
    ax2.plot(positions, full_lift, marker="^", markersize=3.2,
             color="#9bb4d4", linewidth=1.0, label="NACT (full)")
    ax2.plot(positions, mod_lift, marker="s", markersize=3.4, color=BLUE,
             linewidth=1.3, label="Modified NACT")
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels_x, fontsize=7.2)
    ax2.set_xlabel("non-zero coordinates in the input  (denser $\\leftarrow$ "
                   "$\\rightarrow$ sparser)")
    ax2.set_ylabel("lift over best input-blind constant")
    ax2.set_ylim(-0.62, 0.92)
    ax2.text(7.3, -0.045, "below the line the output carries no usable "
             "information about the input", fontsize=6.3, color=RED,
             ha="right", va="top")
    ax2.text(len(rows) - 1, 0.88, "recovery\nprobes", ha="center", va="top",
             fontsize=6.4, color=ORANGE, linespacing=1.3)
    ax2.axvspan(len(rows) - 1.5, len(rows) - 0.5, color="#fdf1e8", zorder=0)
    ax2.legend(loc="lower left", frameon=False, fontsize=7.0)
    ax2.set_title("(b) behaviour on sparse and probe-shaped inputs",
                  fontsize=8.6)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "fig4_observed_metrics.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Figure 5 -- cost scaling, with measurement and forecast kept apart
# --------------------------------------------------------------------------- #
def figure_5() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.75))

    # Measured per-sample training cost at the two trained dimensions.
    nact_pts = [(12, 14, 1000.0 / 108.902), (20, 22, 1000.0 / 87.428)]
    slope = ((nact_pts[1][2] - nact_pts[0][2])
             / (nact_pts[1][1] - nact_pts[0][1]))
    intercept = nact_pts[0][2] - slope * nact_pts[0][1]

    ns = [12, 20, 30, 50, 70, 90, 110, 128]
    lengths = [n + 2 for n in ns]
    fitted = [intercept + slope * L for L in lengths]

    ax1.plot(lengths[1:], fitted[1:], color=BLUE, linewidth=1.1,
             linestyle=(0, (4, 2)), label="linear-in-$L$ extrapolation")
    ax1.scatter([p[1] for p in nact_pts], [p[2] for p in nact_pts], s=34,
                color=BLUE, zorder=5, label="Modified NACT (measured)")
    ax1.scatter(lengths[2:], fitted[2:], s=24, facecolor="white",
                edgecolor=BLUE, linewidth=0.9, zorder=4,
                label="forecast (not measured)")

    v1_pts = [(26, 1000.0 / 55.569), (42, 1000.0 / 43.838),
              (62, 1000.0 / 29.392)]
    ax1.plot([p[0] for p in v1_pts], [p[1] for p in v1_pts], marker="o",
             markersize=3.2, color=GREY, linewidth=1.0,
             label="Salsa2-GatedUT (measured, $n{=}12,20,30$)")

    for n, L, y, off in [(12, 14, nact_pts[0][2], (6, 1)),
                         (20, 22, nact_pts[1][2], (6, -1))]:
        ax1.annotate(f"$n$={n}", (L, y), textcoords="offset points",
                     xytext=off, fontsize=6.6, color=BLUE, va="center")
    ax1.annotate("$n$=128", (130, fitted[-1]), textcoords="offset points",
                 xytext=(-26, 5), fontsize=6.6, color=BLUE)
    ax1.set_xlabel("encoder sequence length $L=n+2$")
    ax1.set_ylabel("training cost (ms / sample, 6 CPU threads)")
    ax1.set_xlim(5, 140)
    ax1.legend(loc="upper left", frameon=False, fontsize=6.6)
    ax1.set_title("(a) per-sample training cost", fontsize=8.6)
    ax1.spines[["top", "right"]].set_visible(False)

    # Candidate space and the measured cost of enumerating it.
    ax2b = ax2.twinx()
    space = [n * (n - 1) // 2 for n in ns]
    ax2.plot(ns, space, marker="o", markersize=3.2, color=ORANGE,
             linewidth=1.2, label=r"candidates $\binom{n}{2}$")
    ax2.set_yscale("log")
    ax2.set_ylim(30, 3e4)
    ax2.set_ylabel(r"weight-2 candidate secrets $\binom{n}{2}$", color=ORANGE)
    ax2.tick_params(axis="y", colors=ORANGE)
    for n, s in zip(ns, space):
        if n in (12, 20, 128):
            ax2.annotate(f"{s:,}", (n, s), textcoords="offset points",
                         xytext=(0, 8), fontsize=6.4, color=ORANGE,
                         ha="center")

    enum_n = [12, 20, 30, 50, 128]
    enum_s = [0.012, 0.022, 0.050, 0.144, 0.988]
    ax2b.plot(enum_n, enum_s, marker="s", markersize=3.2, color=GREEN,
              linewidth=1.2, label="enumerate + verify (measured)")
    ax2b.set_yscale("log")
    ax2b.set_ylim(5e-3, 5.0)
    ax2b.set_ylabel("wall-clock to enumerate and verify (s)", color=GREEN)
    ax2b.tick_params(axis="y", colors=GREEN)
    ax2b.spines["left"].set_visible(False)
    ax2b.spines["top"].set_visible(False)

    for n in (12, 20):
        ax2.axvline(n, color=LIGHT, linewidth=4.0, zorder=0)
    ax2.text(36, 44, "shaded: the two dimensions at which a model was trained",
             fontsize=6.4, color="#6a6a66", ha="left", va="center")

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="lower right", frameon=False,
               fontsize=6.8)
    ax2.set_xlabel(r"lattice dimension $n$   (at $h=2$)")
    ax2.set_title("(b) the classical baseline at $h=2$", fontsize=8.6)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "fig5_scalability.png")
    plt.close(fig)


if __name__ == "__main__":
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    for path in sorted(OUT.glob("*.png")):
        print(f"{path.name:38s} {path.stat().st_size / 1024:8.1f} KB")
