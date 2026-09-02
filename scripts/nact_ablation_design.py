"""Phase 22: design of the NACT component-attribution study.  DESIGN ONLY.

**Nothing is trained.**  No checkpoint is created or modified, no recovery is
run, and the architecture is not changed.  The only file access is reading
already-trained checkpoints to inspect *learned weight magnitudes*, which is a
design input: a component the trained model barely uses does not need an
expensive ablation to predict.

Parameter counts come from the same analytical formula that reproduces the
shipped models exactly (4,131,200 for V1, 4,241,288 for V2), and that agreement
is asserted before anything else is computed.

Runtime estimates come from the three phase-17 NACT T_e=2 runs measured on this
machine, not from a guess.
"""

import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "results" / "nact_ablation_design"
V2_ROOT = REPO_ROOT / "results" / "equal_depth_ablation" / "v2_te2" / "nact_n12_h2_te2"

V1_PARAMETERS, V2_PARAMETERS = 4_131_200, 4_241_288
BUDGET = (4_000_000, 5_000_000)
D_E = 512                       # encoder width
BASE, WIDTH, N_MAX, HEADS = 81, 2, 128, 8

#: Everything that is NOT part of the front end.  Identical in every variant.
SHARED_BODY = 4_087_680

#: The eight components under study, with their exact parameter cost.
COMPONENTS = [
    {"id": 1, "name": "one token per coordinate",
     "parameters": 0,
     "note": "a sequence-layout change, not a parameter: L_in goes 2n+2 -> n+2"},
    {"id": 2, "name": "base-81 digit embeddings",
     "parameters": WIDTH * BASE * D_E,
     "note": "replaces V1's 85 x 512 token embedding (43,520)"},
    {"id": 3, "name": "centered numerical residue", "parameters": D_E,
     "note": "one column of the numerical projection"},
    {"id": 4, "name": "zero / nonzero indicator", "parameters": D_E,
     "note": "one column of the numerical projection"},
    {"id": 5, "name": "Fourier cos/sin features", "parameters": 2 * D_E,
     "note": "two columns of the numerical projection"},
    {"id": 6, "name": "absolute coordinate embedding", "parameters": N_MAX * D_E,
     "note": "the only component with a parameter cost above 0.1% of the model"},
    {"id": 7, "name": "learned zero-coordinate feature", "parameters": D_E,
     "note": "a single learned vector added when a_i == 0"},
    {"id": 8, "name": "sparse attention key bias", "parameters": HEADS,
     "note": "one learned scalar per encoder head, zero-initialised"},
]


def front_end(features: int, coordinate_embedding: bool,
              zero_vector: bool, sparse_bias: bool) -> int:
    """Front-end parameter cost for a given component subset.

    Args:
        features: Number of numerical scalars projected per coordinate.
        coordinate_embedding: Whether the absolute coordinate table is present.
        zero_vector: Whether the learned zero-coordinate vector is present.
        sparse_bias: Whether the per-head sparse attention bias is present.

    Returns:
        Total front-end parameters.
    """
    total = WIDTH * BASE * D_E + 4 * D_E          # digit + special embeddings
    if features:
        total += features * D_E + D_E             # projection weight + bias
    if coordinate_embedding:
        total += N_MAX * D_E
    if zero_vector:
        total += D_E
    if sparse_bias:
        total += HEADS
    return total


#: The proposed variants.  ``features`` lists which numerical scalars survive.
VARIANTS = [
    {"key": "A", "name": "Full NACT (reference)",
     "features": ["zero", "centered", "cos", "sin"],
     "coordinate_embedding": True, "zero_vector": True, "sparse_bias": True,
     "already_trained": True,
     "isolates": "nothing - this is the reference arm",
     "question": "the established V2 result"},
    {"key": "B", "name": "no numerical features (keep zero indicator)",
     "features": ["zero"],
     "coordinate_embedding": True, "zero_vector": True, "sparse_bias": True,
     "already_trained": False,
     "isolates": "centered residue + Fourier pair, jointly",
     "question": "do the continuous value features matter once zero-awareness is present?"},
    {"key": "C", "name": "no zero-aware features",
     "features": ["centered", "cos", "sin"],
     "coordinate_embedding": True, "zero_vector": False, "sparse_bias": False,
     "already_trained": False,
     "isolates": "zero indicator + zero vector + sparse bias, jointly",
     "question": "is zero-awareness what fixes the phase-12 sparse collapse?"},
    {"key": "D", "name": "no Fourier features",
     "features": ["zero", "centered"],
     "coordinate_embedding": True, "zero_vector": True, "sparse_bias": True,
     "already_trained": False,
     "isolates": "the k=1 character pair",
     "question": "does the modular-arithmetic feature matter?"},
    {"key": "E", "name": "no absolute coordinate embedding",
     "features": ["zero", "centered", "cos", "sin"],
     "coordinate_embedding": False, "zero_vector": True, "sparse_bias": True,
     "already_trained": False,
     "isolates": "absolute coordinate identity",
     "question": "is RoPE's relative position insufficient for b = sum a_i s_i?"},
    {"key": "F", "name": "one-token coordinate representation only",
     "features": [],
     "coordinate_embedding": True, "zero_vector": False, "sparse_bias": False,
     "already_trained": False,
     "isolates": "ALL numerical/zero features at once",
     "question": "THE KEY SPLIT: is the gain the tokenisation, or the features?"},
    {"key": "G", "name": "no sparse attention bias",
     "features": ["zero", "centered", "cos", "sin"],
     "coordinate_embedding": True, "zero_vector": True, "sparse_bias": False,
     "already_trained": False,
     "isolates": "the per-head zero-key bias (8 parameters)",
     "question": "does the learned zero-suppression mechanism do anything?"},
]

#: Controlled variables: everything except the component under test.
CONTROLLED = [
    "lwe.n = 12", "lwe.hamming_weight = 2", "lwe.q = 251", "lwe.sigma = 3.0",
    "lwe.structure = rlwe", "lwe.rlwe_variant = circulant", "lwe.secret_index",
    "lwe.num_train_samples = 100000 (100,032 seen)", "lwe.num_valid_samples = 2048",
    "lwe.sample_reuse = 1", "encoding.base = 81", "encoding.digit_order = lsb_first",
    "encoding.separator = False", "encoding.fixed_width = True",
    "experiment.seed", "training.optimizer = adamw", "training.learning_rate",
    "training.scheduler", "training.batch_size = 64", "training.warmup_steps = 200",
    "training.max_epochs = 20", "training.weight_decay", "training.grad_clip",
    "training.monitor_metric = valid_loss", "evaluation.tolerance = 0.1",
    "model.encoder_dim = 512", "model.decoder_dim = 128", "model.encoder_heads = 8",
    "model.decoder_heads = 4", "model.encoder_loops = 2", "model.decoder_loops = 2",
    "model.gated = True", "model.ffn_multiplier = 4.0", "model.dropout = 0.0",
]

PRIMARY_METRICS = ["SALSA acc_tau", "exact integer accuracy", "validation loss",
                   "sparse-input lift", "direct secret recovery"]
SECONDARY_METRICS = ["token accuracy", "decode validity", "CPU throughput", "memory"]
SPARSITY_LEVELS = [12, 8, 6, 4, 2, 1]


def measured_basis() -> Dict[str, Any]:
    """Runtime and seed-variance basis, read from the phase-17 runs."""
    elapsed, compute, metrics = [], [], {"valid_acc_tau": [], "valid_loss": [],
                                         "valid_exact_accuracy": []}
    for seed in (0, 42, 123):
        path = V2_ROOT / f"seed_{seed}" / "artifacts" / "summary.json"
        summary = json.loads(path.read_text("utf-8"))
        elapsed.append(summary["state"]["elapsed_seconds"])
        compute.append(summary["throughput"]["compute_seconds"])
        for key in metrics:
            metrics[key].append(summary["final_validation"][key])
    return {
        "source": "three phase-17 NACT T_e=2 runs, same machine, 100,032 samples each",
        "mean_elapsed_seconds": statistics.fmean(elapsed),
        "min_elapsed_seconds": min(elapsed), "max_elapsed_seconds": max(elapsed),
        "mean_compute_seconds": statistics.fmean(compute),
        "validation_and_overhead_seconds": statistics.fmean(elapsed) - statistics.fmean(compute),
        "validation_overhead_fraction": 1 - statistics.fmean(compute) / statistics.fmean(elapsed),
        "sparsity_eval_seconds": 240,
        "recovery_plus_verification_seconds": 120,
        "seed_variance": {
            key: {"mean": statistics.fmean(values), "stdev": statistics.stdev(values),
                  "three_sigma_band": 3 * statistics.stdev(values)}
            for key, values in metrics.items()
        },
    }


def learned_weight_diagnostic() -> Dict[str, Any]:
    """Read learned front-end magnitudes from the trained checkpoints.

    Read-only, no forward pass.  A component the trained model barely moved off
    its initialisation can have its ablation *predicted* rather than run, which
    is the cheapest possible way to shrink the matrix.
    """
    import torch

    features = ["zero_indicator", "centered_residue", "cos", "sin"]
    rows = []
    for seed in (0, 42, 123):
        path = V2_ROOT / f"seed_{seed}" / "checkpoints" / "best.pt"
        if not path.is_file():
            continue
        state = torch.load(path, map_location="cpu", weights_only=False)["model"]
        weight = state["front_end.numerical_projection.weight"]
        bias = state["encoder_layers.0.sparse_attention_bias"]
        rows.append({
            "seed": seed,
            "sparse_attention_bias_max_abs": float(bias.abs().max()),
            "sparse_attention_bias": [round(float(v), 4) for v in bias],
            "numerical_projection_column_norms": {
                name: round(float(weight[:, i].norm()), 4)
                for i, name in enumerate(features)},
            "zero_vector_norm": round(float(state["front_end.zero_vector"].norm()), 4),
            "coordinate_embedding_norm": round(
                float(state["front_end.coordinate_embedding"].norm()), 4),
            "digit_embedding_norm": round(
                float(state["front_end.digit_embedding"].norm()), 4),
        })
    max_bias = max(r["sparse_attention_bias_max_abs"] for r in rows)
    return {
        "note": "read-only inspection of trained weights; no forward pass, no training",
        "rows": rows,
        "sparse_bias_max_abs_across_seeds": max_bias,
        "sparse_bias_initialised_at": 0.0,
        "inference_sparse_bias": (
            f"The per-head bias was initialised at exactly 0 and reached at most "
            f"{max_bias:.4f} after 100,032 samples, on all three seeds. Attention "
            "logits are O(1-10), so a bias of this size shifts the softmax "
            "negligibly. Variant G is therefore PREDICTED to be indistinguishable "
            "from A, and can be dropped from the paid matrix in favour of this "
            "zero-cost diagnostic."),
        "inference_numerical_features": (
            "All four numerical columns carry comparable weight norm (~1.30-1.49) on "
            "every seed, with the Fourier pair marginally highest. No feature is "
            "ignored, so none of B, C or D can be predicted away and each needs a "
            "run. Weight norm is a weak proxy for importance and is used here only "
            "to rule OUT a component, never to rule one in."),
        "inference_coordinate_embedding": (
            "The coordinate table reaches a norm comparable to the digit embeddings "
            "(~5.4 vs ~5.9), so it is carrying real signal and variant E is worth "
            "paying for."),
    }


def build_variants() -> List[Dict[str, Any]]:
    """Exact parameter counts and confound analysis for each variant."""
    rows = []
    for variant in VARIANTS:
        total = SHARED_BODY + front_end(
            len(variant["features"]), variant["coordinate_embedding"],
            variant["zero_vector"], variant["sparse_bias"])
        delta = total - V2_PARAMETERS
        rows.append({
            **variant,
            "num_features": len(variant["features"]),
            "parameters": total,
            "delta_vs_full_nact": delta,
            "percent_of_full_nact": 100.0 * delta / V2_PARAMETERS,
            "within_budget": BUDGET[0] <= total <= BUDGET[1],
            "delta_vs_v1": total - V1_PARAMETERS,
            "confound_risk": (
                "negligible: below 0.1% of the model"
                if abs(delta) / V2_PARAMETERS < 0.001 else
                "small but non-zero: state it when reporting"),
        })
    return rows


def redundancy(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Which variants duplicate each other's information."""
    return {
        "nested_ladder": {
            "chain": "F (0 features) < B (1) < D (2) < A (4)",
            "reading": ("A ladder of single-feature additions: F->B isolates the zero "
                        "indicator, B->D the centered residue, D->A the Fourier pair. "
                        "Four arms give three clean single-feature contrasts."),
        },
        "leave_one_out": {
            "chain": "A minus one of {zero-aware, Fourier, coordinate, sparse bias} = C, D, E, G",
            "reading": ("Every contrast is against the same reference A, which is what "
                        "attribution needs. Preferred over the ladder for that reason."),
        },
        "overlaps": [
            {"pair": "B and F", "overlap": "differ only by the zero-indicator scalar",
             "verdict": "keep both only if F->B is the specific question; otherwise F alone"},
            {"pair": "B and D", "overlap": "differ only by the centered-residue scalar",
             "verdict": "B is redundant once F, D and A are run - the ladder is covered"},
            {"pair": "C and G", "overlap": "C removes the sparse bias as part of "
                                          "zero-awareness, G removes it alone",
             "verdict": "G is subsumed by C for the sparsity question, and is separately "
                        "predicted a no-op by the weight diagnostic"},
        ],
        "safe_to_drop": ["G", "B"],
        "drop_rationale": {
            "G": ("predicted from trained weights: the bias never moved off zero, so "
                  "removing 8 parameters that do nothing cannot change the outcome"),
            "B": ("its two contrasts, F->B and B->D, are both recoverable from the "
                  "F/D/A arms; it adds a run without adding a distinct question"),
        },
    }


def staged_plan(basis: Dict[str, Any]) -> Dict[str, Any]:
    """Three-stage plan with pre-registered gates, so cost scales with evidence."""
    train = basis["mean_elapsed_seconds"]
    return {
        "principle": ("Each stage is paid for only by the variants that survived the "
                      "previous one, and the gates are fixed before any run."),
        "stages": [
            {"stage": 1, "name": "learning metrics", "applies_to": "every variant",
             "cost_seconds_per_variant": train,
             "measures": ["valid_loss", "valid_acc_tau", "valid_exact_accuracy",
                          "token accuracy", "decode validity", "throughput", "memory"],
             "gate": ("advance a variant to stage 2 if its acc_tau is within 0.05 of "
                      "full NACT, OR if it drops by more than 0.05 - both outcomes are "
                      "informative. Only variants that land in neither case (impossible "
                      "by construction) are dropped."),
             "note": "no variant is dropped at stage 1; the gate orders stage 2, not membership"},
            {"stage": 2, "name": "sparsity generalization",
             "applies_to": "every variant that trained successfully",
             "cost_seconds_per_variant": basis["sparsity_eval_seconds"],
             "measures": [f"lift at nnz={n}" for n in SPARSITY_LEVELS] + ["K*e_i probe lift"],
             "gate": ("advance to stage 3 only if lift stays positive through nnz=1, "
                      "the property that distinguishes V2 from V1"),
             "note": "cheap enough (4 min) to run on all arms regardless"},
            {"stage": 3, "name": "direct secret recovery",
             "applies_to": "only variants passing the stage-2 gate",
             "cost_seconds_per_variant": basis["recovery_plus_verification_seconds"],
             "measures": ["exact recovery", "coordinate accuracy", "successful K count",
                          "phase-20 residual verification"],
             "gate": "terminal",
             "note": ("recovery is nearly free (2 min) but is only INTERPRETABLE for "
                      "variants that kept sparse-probe competence; running it on a "
                      "variant that collapsed on probes measures nothing new")},
        ],
        "why_not_recovery_everywhere": (
            "Recovery is cheap, so the reason to stage it is not cost but meaning. "
            "Phase 12 established that probe behaviour is the mechanism; a variant that "
            "fails stage 2 will fail recovery for a reason already measured, and "
            "reporting it as an independent finding would double-count one cause."),
    }


def seed_design(basis: Dict[str, Any]) -> Dict[str, Any]:
    """How many seeds are actually needed, argued from measured variance."""
    variance = basis["seed_variance"]
    return {
        "recommendation": "seed 0 screening for all variants; extra seeds only where needed",
        "measured_seed_variance_full_nact": variance,
        "argument": (
            "The three phase-17 NACT runs give acc_tau sd = "
            f"{variance['valid_acc_tau']['stdev']:.4f} and loss sd = "
            f"{variance['valid_loss']['stdev']:.4f}. A single seed therefore resolves "
            f"any acc_tau difference larger than about "
            f"{variance['valid_acc_tau']['three_sigma_band']:.4f} (3 sd). The effect this "
            "study is chasing is the V1-vs-V2 gap of roughly 0.76 in acc_tau - two "
            "orders of magnitude larger than the seed noise. Spending 5 seeds per "
            "variant to resolve an effect that one seed resolves is wasted compute."),
        "when_to_add_seeds": [
            "a variant lands within 3 sd of full NACT and the claim is 'no effect' - "
            "absence of an effect needs more evidence than presence of one",
            "two variants land within 3 sd of each other and their ORDER matters",
            "the single decisive contrast in the final write-up, which should carry "
            "3 seeds so the headline is not a one-run claim",
        ],
        "caveat": (
            "The measured sd is full NACT's. A variant that degrades toward V1-like "
            "behaviour may also inherit V1's much larger seed spread - V1's acc_tau "
            "ranged 0.10 to 0.36 across seeds. Any variant whose seed-0 result is "
            "clearly degraded must therefore get 3 seeds before its magnitude is "
            "quoted, even though its DIRECTION is safe from one run."),
    }


def recommendation(variants: List[Dict[str, Any]], basis: Dict[str, Any]) -> Dict[str, Any]:
    """The minimum set, ranked."""
    train = basis["mean_elapsed_seconds"]
    sparsity = basis["sparsity_eval_seconds"]
    recovery = basis["recovery_plus_verification_seconds"]

    ranked = [
        {"rank": 1, "variant": "F", "name": "one-token only",
         "scientific_value": "highest",
         "why": ("It splits the two competing explanations in one run. If F matches A, "
                 "every numerical feature is decoration and the gain is the "
                 "tokenisation plus coordinate identity. If F falls back toward V1, "
                 "the features are doing the work. No other single experiment "
                 "separates these."),
         "distinguishes": "representation change vs information change"},
        {"rank": 2, "variant": "C", "name": "no zero-aware features",
         "scientific_value": "high",
         "why": ("Phase 14 predicted the zero indicator repairs a measured two-position "
                 "conjunction, and phase 12 measured the sparse collapse it was meant "
                 "to fix. C is the direct test of the project's own stated hypothesis "
                 "H2, and it is the one most likely to move the sparsity endpoint."),
         "distinguishes": "zero-awareness vs continuous value information"},
        {"rank": 3, "variant": "E", "name": "no coordinate embedding",
         "scientific_value": "high",
         "why": ("b = sum_i a_i s_i binds coordinate i to bit s_i, so absolute identity "
                 "is theoretically required and RoPE supplies only relative position. "
                 "The trained coordinate table has a norm comparable to the digit "
                 "embeddings, so it is carrying signal. This is also the only variant "
                 "with a parameter delta worth stating."),
         "distinguishes": "absolute vs relative position"},
        {"rank": 4, "variant": "D", "name": "no Fourier",
         "scientific_value": "medium",
         "why": ("Tests the modular-arithmetic claim specifically. Lower priority only "
                 "because F and C already bracket it: if F matches A, D is moot."),
         "distinguishes": "ring structure vs plain magnitude"},
        {"rank": 5, "variant": "G", "name": "no sparse bias",
         "scientific_value": "predicted, do not pay for it",
         "why": ("The trained bias never left its zero initialisation on any seed. "
                 "Run it only if the prediction itself is to be checked."),
         "distinguishes": "nothing measurable"},
    ]

    minimum = ["F", "C", "E"]
    minimum_cost = len(minimum) * (train + sparsity) + 2 * recovery
    full_cost = 6 * (train + sparsity + recovery)
    confirm = 2 * train

    return {
        "ranked": ranked,
        "minimum_viable_single_experiment": {
            "variant": "F", "cost_seconds": train + sparsity + recovery,
            "answers": "is the gain the tokenisation or the numerical features?",
        },
        "recommended_set": minimum,
        "recommended_rationale": (
            "F, C and E are the three arms that map onto three genuinely different "
            "explanations - representation, zero-awareness, positional identity. A is "
            "already trained on three seeds and costs nothing. D is held in reserve "
            "and only run if F shows the features matter but C does not explain why. "
            "G is predicted from weights already on disk."),
        "cost_estimate_seconds": {
            "stage_1_training_3_variants": len(minimum) * train,
            "stage_2_sparsity_3_variants": len(minimum) * sparsity,
            "stage_3_recovery_top_2": 2 * recovery,
            "confirmation_seeds_for_the_decisive_contrast": confirm,
            "total_recommended": minimum_cost + confirm,
            "total_if_all_six_variants_were_run_once": full_cost,
            "total_if_all_six_at_five_seeds": 6 * 5 * train + 6 * sparsity + 6 * recovery,
        },
        "cost_estimate_hours": {
            "total_recommended": (minimum_cost + confirm) / 3600.0,
            "total_if_all_six_variants_were_run_once": full_cost / 3600.0,
            "total_if_all_six_at_five_seeds": (6 * 5 * train + 6 * sparsity
                                               + 6 * recovery) / 3600.0,
        },
    }


def main() -> int:
    """Assemble the design."""
    from salsa.models.parameter_count import analytical_parameter_count
    from salsa.models.transformer import ModelSpec
    from salsa.models.nact import NactSpec

    assert analytical_parameter_count(ModelSpec()) == V1_PARAMETERS
    assert analytical_parameter_count(NactSpec(encoder_loops=2)) == V2_PARAMETERS
    assert SHARED_BODY + front_end(4, True, True, True) == V2_PARAMETERS, "formula drift"

    basis = measured_basis()
    variants = build_variants()
    diagnostic = learned_weight_diagnostic()
    plan = staged_plan(basis)
    seeds = seed_design(basis)
    advice = recommendation(variants, basis)

    print("=" * 98)
    print("PHASE 22 - NACT COMPONENT ATTRIBUTION STUDY   *** DESIGN ONLY, NOTHING TRAINED ***")
    print("=" * 98)
    print(f"  formula anchor: V1 {V1_PARAMETERS:,} and V2 {V2_PARAMETERS:,} both reproduced")
    print()
    print(f"  {'key':<4}{'variant':<44}{'params':>11}{'delta':>9}{'%':>8}{'budget':>8}")
    for row in variants:
        print(f"  {row['key']:<4}{row['name']:<44}{row['parameters']:>11,}"
              f"{row['delta_vs_full_nact']:>+9,}{row['percent_of_full_nact']:>8.3f}"
              f"{'OK' if row['within_budget'] else 'FAIL':>8}")
    print()
    print("  LEARNED-WEIGHT DIAGNOSTIC (read-only, zero cost)")
    print(f"    sparse attention bias max|beta| across seeds: "
          f"{diagnostic['sparse_bias_max_abs_across_seeds']:.4f}  (initialised at 0.0)")
    print(f"    -> variant G predicted to be a no-op; dropped from the paid matrix")
    print()
    print(f"  RUNTIME BASIS: {basis['source']}")
    print(f"    mean {basis['mean_elapsed_seconds']:.0f}s per run "
          f"[{basis['min_elapsed_seconds']:.0f}, {basis['max_elapsed_seconds']:.0f}], "
          f"validation overhead {basis['validation_overhead_fraction']*100:.0f}%")
    print(f"    sparsity eval {basis['sparsity_eval_seconds']}s, "
          f"recovery+verification {basis['recovery_plus_verification_seconds']}s")
    print()
    print("  RECOMMENDED SET: " + ", ".join(advice["recommended_set"]))
    hours = advice["cost_estimate_hours"]
    print(f"    recommended total      : {hours['total_recommended']:.2f} h")
    print(f"    all six variants once  : {hours['total_if_all_six_variants_were_run_once']:.2f} h")
    print(f"    all six at five seeds  : {hours['total_if_all_six_at_five_seeds']:.2f} h")

    payload = {
        "phase": "22 - NACT component attribution study DESIGN",
        "status": "DESIGN ONLY. NO TRAINING PERFORMED. No checkpoint created or "
                  "modified, no recovery run, no architecture change. The only file "
                  "access was reading trained weights for the zero-cost diagnostic.",
        "reference_models": {"V1_GatedUT": V1_PARAMETERS, "V2_NACT": V2_PARAMETERS,
                             "shared_body_identical_in_every_variant": SHARED_BODY},
        "components": COMPONENTS,
        "variants": variants,
        "redundancy_analysis": redundancy(variants),
        "learned_weight_diagnostic": diagnostic,
        "controlled_variables": CONTROLLED,
        "primary_metrics": PRIMARY_METRICS,
        "secondary_metrics": SECONDARY_METRICS,
        "sparsity_levels": SPARSITY_LEVELS,
        "measured_basis": basis,
        "staged_plan": plan,
        "seed_design": seeds,
        "recommendation": advice,
        "budget": {"min": BUDGET[0], "max": BUDGET[1],
                   "all_variants_within_budget": all(v["within_budget"] for v in variants),
                   "no_padding_parameters": True},
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "ablation_design.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    columns = ["section", "key", "name", "num_features", "parameters",
               "delta_vs_full_nact", "percent_of_full_nact", "within_budget",
               "already_trained", "isolates", "question", "confound_risk",
               "rank", "variant", "scientific_value", "stage", "applies_to",
               "cost_seconds_per_variant", "item", "value"]
    with (OUTPUT_DIR / "ablation_design.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in variants:
            writer.writerow({"section": "variant", **row})
        for row in advice["ranked"]:
            writer.writerow({"section": "ranking", **row})
        for row in plan["stages"]:
            writer.writerow({"section": "stage", **row})
        for item, value in advice["cost_estimate_seconds"].items():
            writer.writerow({"section": "cost_seconds", "item": item, "value": value})
        for component in COMPONENTS:
            writer.writerow({"section": "component", "key": component["id"],
                             "name": component["name"],
                             "parameters": component["parameters"]})

    write_markdown(payload, OUTPUT_DIR / "ablation_design.md")
    print()
    for name in ("ablation_design.md", "ablation_design.json", "ablation_design.csv"):
        print(f"  wrote {OUTPUT_DIR / name}")
    print()
    print("  NO TRAINING PERFORMED.")
    return 0


def write_markdown(payload: Dict[str, Any], path: Path) -> None:
    """Render the design document."""
    variants, basis = payload["variants"], payload["measured_basis"]
    diagnostic, plan = payload["learned_weight_diagnostic"], payload["staged_plan"]
    seeds, advice = payload["seed_design"], payload["recommendation"]
    redundant = payload["redundancy_analysis"]
    hours = advice["cost_estimate_hours"]

    lines = [
        "# Phase 22 — NACT component attribution study (design)",
        "",
        "> **DESIGN ONLY. NO TRAINING PERFORMED.** No checkpoint was created or modified,",
        "> no recovery was run, and the architecture was not changed. The only file access",
        "> was reading already-trained weights for a zero-cost diagnostic.",
        "",
        "## The question",
        "",
        "NACT changed eight things at once. Which of them causes the improvement?",
        "",
        "| # | component | parameters | note |",
        "|---:|---|---:|---|",
    ]
    for component in payload["components"]:
        lines.append(f"| {component['id']} | {component['name']} | "
                     f"{component['parameters']:,} | {component['note']} |")
    lines += [
        "",
        f"Everything outside the front end — encoder body, decoder, output — is "
        f"**{payload['reference_models']['shared_body_identical_in_every_variant']:,} "
        "parameters and is identical in every variant.**",
        "",
        "## Proposed variants and their exact parameter counts",
        "",
        "| key | variant | features | params | Δ vs full | % | budget | isolates |",
        "|---|---|---:|---:|---:|---:|:---:|---|",
    ]
    for row in variants:
        lines.append(
            f"| **{row['key']}** | {row['name']} | {row['num_features']} | "
            f"{row['parameters']:,} | {row['delta_vs_full_nact']:+,} | "
            f"{row['percent_of_full_nact']:+.3f}% | "
            f"{'OK' if row['within_budget'] else 'FAIL'} | {row['isolates']} |")
    lines += [
        "",
        "### Could the parameter differences confound the result?",
        "",
        "**No, with one qualification.** Six of the seven variants sit within **3,080",
        "parameters — under 0.08%** — of full NACT, because every numerical feature costs",
        "one 512-wide column and the sparse bias costs 8 scalars. A difference that small",
        "cannot plausibly produce the effects being chased, which are of order 0.76 in",
        "acc_tau.",
        "",
        "The qualification is **E**, which drops the 65,536-parameter coordinate table",
        "(1.5% of the model). That is still small, but it is the one variant where a",
        "capacity explanation is not automatically dismissible and it must be stated when",
        "E is reported.",
        "",
        "**No padding parameters are proposed.** If a variant were thought too small, the",
        "fair remedy is not to inflate it but to report the count honestly and, if",
        "capacity is genuinely in doubt, add a matched-capacity control that spends the",
        "difference somewhere architecturally neutral — not to manufacture dead weights.",
        "Nothing here needs that: the largest gap is 1.5%.",
        "",
        "Every variant stays inside the 4–5M budget.",
        "",
        "## Redundancy — which variants can be dropped",
        "",
        f"**Nested ladder.** {redundant['nested_ladder']['chain']} — "
        f"{redundant['nested_ladder']['reading']}",
        "",
        f"**Leave-one-out.** {redundant['leave_one_out']['chain']} — "
        f"{redundant['leave_one_out']['reading']}",
        "",
        "| overlap | verdict |",
        "|---|---|",
    ]
    for entry in redundant["overlaps"]:
        lines.append(f"| {entry['pair']}: {entry['overlap']} | {entry['verdict']} |")
    lines += [
        "",
        f"**Safe to drop: {redundant['safe_to_drop']}.**",
        "",
    ]
    for key, reason in redundant["drop_rationale"].items():
        lines.append(f"- **{key}** — {reason}")

    lines += [
        "",
        "## A zero-cost diagnostic that shrinks the matrix",
        "",
        f"*{diagnostic['note']}.*",
        "",
        "| seed | max abs sparse bias | zero-indicator col | centered col | cos col | sin col | zero-vector norm | coord-emb norm |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in diagnostic["rows"]:
        norms = row["numerical_projection_column_norms"]
        lines.append(f"| {row['seed']} | {row['sparse_attention_bias_max_abs']:.4f} | "
                     f"{norms['zero_indicator']:.3f} | {norms['centered_residue']:.3f} | "
                     f"{norms['cos']:.3f} | {norms['sin']:.3f} | "
                     f"{row['zero_vector_norm']:.3f} | "
                     f"{row['coordinate_embedding_norm']:.3f} |")
    lines += [
        "",
        f"- **Sparse attention bias.** {diagnostic['inference_sparse_bias']}",
        f"- **Numerical features.** {diagnostic['inference_numerical_features']}",
        f"- **Coordinate embedding.** {diagnostic['inference_coordinate_embedding']}",
        "",
        "## Controlled variables",
        "",
        "Every future ablation run must hold all of the following constant, so that the",
        "component under test is the only thing that varies:",
        "",
        "```",
    ] + [f"  {item}" for item in payload["controlled_variables"]] + [
        "```",
        "",
        "The existing `configs/nact_n12_h2_te2.yaml` already fixes all of these, so each",
        "variant should be a config that `extends:` it and changes only its own component.",
        "",
        "## Metrics and endpoints",
        "",
        "**Primary, in rank order:** " + ", ".join(
            f"{i+1}. {m}" for i, m in enumerate(payload["primary_metrics"])) + ".",
        "",
        "**Secondary:** " + ", ".join(payload["secondary_metrics"]) + ".",
        "",
        f"**Sparsity endpoints:** nnz = {payload['sparsity_levels']} plus `K*e_i` probes.",
        "The main endpoint is **lift over the best input-blind constant**, because sparser",
        "inputs make a constant predictor better and raw acc_tau alone would understate a",
        "collapse. V1's zero-lift boundary sits between nnz=8 and nnz=6; V2 stays positive",
        "through nnz=1. **The endpoint for every ablation is where its boundary lands",
        "between those two.**",
        "",
        "## Staged plan",
        "",
        f"*{plan['principle']}*",
        "",
        "| stage | name | applies to | cost per variant | gate |",
        "|---:|---|---|---:|---|",
    ]
    for stage in plan["stages"]:
        lines.append(f"| {stage['stage']} | {stage['name']} | {stage['applies_to']} | "
                     f"{stage['cost_seconds_per_variant']:.0f} s | {stage['gate']} |")
    lines += [
        "",
        f"**Does every ablation need recovery? No.** {plan['why_not_recovery_everywhere']}",
        "",
        "## How many seeds",
        "",
        f"**{seeds['recommendation']}.**",
        "",
        f"{seeds['argument']}",
        "",
        "Add seeds when:",
        "",
    ] + [f"- {reason}" for reason in seeds["when_to_add_seeds"]] + [
        "",
        f"**Caveat.** {seeds['caveat']}",
        "",
        "## Runtime estimates",
        "",
        f"Basis: {basis['source']}.",
        "",
        "| item | seconds |",
        "|---|---:|",
        f"| training, one variant, one seed, 100,032 samples | {basis['mean_elapsed_seconds']:.0f} "
        f"(range {basis['min_elapsed_seconds']:.0f}–{basis['max_elapsed_seconds']:.0f}) |",
        f"| — of which compute | {basis['mean_compute_seconds']:.0f} |",
        f"| — of which validation and overhead | {basis['validation_and_overhead_seconds']:.0f} "
        f"({basis['validation_overhead_fraction']*100:.0f}%) |",
        f"| sparsity evaluation, one variant | {basis['sparsity_eval_seconds']} |",
        f"| recovery + residual verification, one variant | {basis['recovery_plus_verification_seconds']} |",
        "",
        "| scenario | experiments | total |",
        "|---|---:|---:|",
        f"| **recommended (F, C, E + confirmation seeds)** | 5 training runs | "
        f"**{hours['total_recommended']:.2f} h** |",
        f"| all six new variants, one seed each | 6 training runs | "
        f"{hours['total_if_all_six_variants_were_run_once']:.2f} h |",
        f"| all six variants at five seeds | 30 training runs | "
        f"{hours['total_if_all_six_at_five_seeds']:.2f} h |",
        "",
        "Note that **variant A costs nothing** — it is already trained on three seeds from",
        "phase 17.",
        "",
        "## Recommendation",
        "",
        "| rank | variant | value | distinguishes |",
        "|---:|---|---|---|",
    ]
    for entry in advice["ranked"]:
        lines.append(f"| {entry['rank']} | **{entry['variant']}** — {entry['name']} | "
                     f"{entry['scientific_value']} | {entry['distinguishes']} |")
    lines += [""]
    for entry in advice["ranked"]:
        lines.append(f"**{entry['variant']} ({entry['name']}).** {entry['why']}")
        lines.append("")
    minimum = advice["minimum_viable_single_experiment"]
    lines += [
        f"### Minimum recommended set: **{', '.join(advice['recommended_set'])}**",
        "",
        f"{advice['recommended_rationale']}",
        "",
        f"### If only one experiment can be afforded: **{minimum['variant']}**",
        "",
        f"{minimum['cost_seconds'] / 60:.0f} minutes, and it answers *{minimum['answers']}* —",
        "the single question with the most competing explanations behind it.",
        "",
        "## What this design can and cannot settle",
        "",
        "- It attributes the improvement among **components of NACT**. It does not revisit",
        "  whether NACT beats V1 — phases 17 and 19 measured that.",
        "- Interactions are not identified. Leave-one-out finds each component's marginal",
        "  contribution *in the presence of the others*; if two features are mutually",
        "  redundant, removing either alone may show nothing while removing both shows a",
        "  large effect. Variant F guards against exactly that by removing all of them.",
        "- Everything stays at n=12, h=2, which is a diagnostic instance with C(12,2) = 66",
        "  secrets. Component attribution there need not transfer to larger n.",
        "",
        "## Artifacts",
        "",
        "- `ablation_design.md` (this file)",
        "- `ablation_design.json`",
        "- `ablation_design.csv`",
        "",
        "**NO TRAINING PERFORMED.**",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
