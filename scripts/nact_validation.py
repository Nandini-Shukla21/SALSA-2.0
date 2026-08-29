"""Phase 15 steps 4-6: architecture validation, forward benchmark, parameter proof.

**No training.**  Nothing here builds an optimizer, computes a loss or takes a
step.  Every model is put in ``eval()`` and every measurement is taken under
``torch.no_grad()``.

Three things are established, in order:

4. NACT accepts every dimension from 12 to 128 under both representations, with
   dense, sparse, single-nonzero and ``K*e_i`` probe inputs, producing finite
   logits of the V1 output shape and valid token ids.
5. V1 and V2 forward cost is measured rather than assumed, at n=30 and n=128.
6. The parameter budget is proved three ways and shown to be invariant in the
   loop count and in ``n``.

Untrained models are used throughout.  No accuracy is computed and none would
mean anything: the only questions asked are about shapes, finiteness and cost.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from salsa.data import LatticeCodec  # noqa: E402
from salsa.models import (  # noqa: E402
    ModelSpec,
    NactSpec,
    SalsaNact,
    SalsaTransformer,
    analytical_breakdown,
    count_trainable_parameters,
    parameter_breakdown,
    unclassified_parameters,
)
from salsa.recovery import build_probe_matrix  # noqa: E402
from salsa.training.metrics import decode_generated_ids, greedy_decode  # noqa: E402

DIMENSIONS = (12, 20, 30, 50, 70, 90, 128)
BENCHMARK_DIMENSIONS = (30, 128)
Q, BASE, V1_PARAMETERS, V2_PARAMETERS = 251, 81, 4_131_200, 4_241_288


def codec_for(n: int, separator: bool) -> LatticeCodec:
    """Codec matching the phase-6 encoding at dimension ``n``."""
    return LatticeCodec(n=n, q=Q, base=BASE, separator=separator,
                        digit_order="lsb_first", fixed_width=True)


def nact_for(codec: LatticeCodec) -> SalsaNact:
    """An untrained NACT consistent with ``codec``."""
    return SalsaNact(NactSpec(
        vocab_size=codec.vocabulary.size, q=codec.q,
        base=codec.input_encoder.base, digit_width=codec.input_encoder.width,
        separator=codec.separator, max_coordinates=max(128, codec.n))).eval()


def input_families(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """The four input families required by the brief, as ``(m, n)`` matrices."""
    dense = rng.integers(0, Q, size=(8, n), dtype=np.int64)

    sparse = np.zeros((8, n), dtype=np.int64)
    nnz = max(1, n // 4)
    for row in range(8):
        positions = rng.choice(n, size=nnz, replace=False)
        sparse[row, positions] = rng.integers(1, Q, size=nnz, dtype=np.int64)

    single = np.zeros((8, n), dtype=np.int64)
    single[np.arange(8), rng.choice(n, size=8)] = rng.integers(1, Q, size=8)

    probes = build_probe_matrix(n, 125, Q)[: min(8, n)]
    return {"dense_random": dense, "sparse": sparse,
            "nnz_1": single, "probe_K_times_e_i": probes}


# --------------------------------------------------------------------------- #
# Step 4
# --------------------------------------------------------------------------- #
@torch.no_grad()
def validate_dimensions() -> List[Dict[str, Any]]:
    """Forward-pass validation across dimensions, representations and input families."""
    rows: List[Dict[str, Any]] = []
    for n in DIMENSIONS:
        for representation, separator in (("R", False), ("P", True)):
            codec = codec_for(n, separator)
            model = nact_for(codec)
            parameters = count_trainable_parameters(model)
            for family, matrix in input_families(n, np.random.default_rng(n)).items():
                src_np, _ = codec.encode_batch(matrix)
                src = torch.from_numpy(src_np)
                target = torch.randint(4, 85, (src.shape[0], codec.output_length))

                memory = model.encode(src)
                logits = model(src, target)
                generated = greedy_decode(model, src, codec.vocabulary.bos_id,
                                          codec.output_length - 1)
                _, decoded_ok = decode_generated_ids(codec, generated.numpy())
                _, values = model.front_end.decode_values(src)
                features = model.front_end.numerical_features(values)

                rows.append({
                    "n": n,
                    "representation": representation,
                    "input_family": family,
                    "batch": int(src.shape[0]),
                    "source_shape": list(src.shape),
                    "source_length_matches_codec": int(src.shape[1]) == codec.input_length,
                    "expected_source_length": codec.input_length,
                    "encoder_sequence_length": int(memory.shape[1]),
                    "encoder_length_is_n_plus_2": int(memory.shape[1]) == n + 2,
                    "output_shape": list(logits.shape),
                    "output_shape_correct": tuple(logits.shape) == (
                        src.shape[0], codec.output_length, 85),
                    "vocab_size": int(codec.vocabulary.size),
                    "logits_finite": bool(torch.isfinite(logits).all()),
                    "no_nan": bool(not torch.isnan(logits).any()),
                    "logit_abs_max": round(float(logits.abs().max()), 4),
                    "memory_finite": bool(torch.isfinite(memory).all()),
                    "generated_ids_in_vocabulary": bool(
                        int(generated.min()) >= 0 and int(generated.max()) < 85),
                    "values_reconstructed_exactly": bool(
                        np.array_equal(values.numpy(), matrix)),
                    "features_finite": bool(torch.isfinite(features).all()),
                    "feature_abs_max": round(float(features.abs().max()), 6),
                    "zero_coordinates_seen": int((matrix == 0).sum()),
                    "decode_validity_UNTRAINED_not_an_accuracy": round(
                        float(decoded_ok.mean()), 4),
                    "parameter_count": parameters,
                })
    return rows


# --------------------------------------------------------------------------- #
# Step 5
# --------------------------------------------------------------------------- #
@torch.no_grad()
def benchmark(batch_size: int = 32, repeats: int = 5) -> List[Dict[str, Any]]:
    """Forward-only latency and throughput for V1 and V2.  No optimizer, no step."""
    try:
        import psutil

        process = psutil.Process()
    except ImportError:
        process = None

    rows: List[Dict[str, Any]] = []
    for n in BENCHMARK_DIMENSIONS:
        for representation, separator in (("R", False), ("P", True)):
            codec = codec_for(n, separator)
            src = torch.from_numpy(codec.encode_batch(
                np.random.default_rng(n).integers(
                    0, Q, size=(batch_size, n), dtype=np.int64))[0])
            target = torch.randint(4, 85, (batch_size, codec.output_length))

            # The T_e=2 row is the SAME WEIGHTS as the T_e=4 row -- loops reuse
            # one shared parameter set -- so it isolates what the shorter
            # sequence buys from what the extra depth spends.  Comparing V1
            # (T_e=2) against V2 at T_e=4 alone would confound the two.
            equal_depth = SalsaNact(NactSpec(
                vocab_size=codec.vocabulary.size, q=codec.q,
                base=codec.input_encoder.base, digit_width=codec.input_encoder.width,
                separator=codec.separator, max_coordinates=max(128, codec.n),
                encoder_loops=2)).eval()
            for label, model in (
                ("V1 GatedUT T_e=2", SalsaTransformer(ModelSpec()).eval()),
                ("V2 NACT T_e=4", nact_for(codec)),
                ("V2 NACT T_e=2", equal_depth),
            ):
                model(src, target)          # warm up allocator and BLAS
                gc.collect()
                before = process.memory_info().rss / 2 ** 20 if process else None

                timings = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    model(src, target)
                    timings.append(time.perf_counter() - start)
                after = process.memory_info().rss / 2 ** 20 if process else None

                encoder_length = int(model.encode(src).shape[1])
                median = float(np.median(timings))
                rows.append({
                    "n": n,
                    "representation": representation,
                    "model": label,
                    "parameter_count": count_trainable_parameters(model),
                    "source_length": int(src.shape[1]),
                    "encoder_sequence_length": encoder_length,
                    "batch_size": batch_size,
                    "repeats": repeats,
                    "encoder_loops": int(getattr(model.spec, "encoder_loops")),
                    "forward_latency_ms_median": round(median * 1e3, 3),
                    "forward_latency_ms_min": round(min(timings) * 1e3, 3),
                    "forward_throughput_samples_per_second": round(batch_size / median, 1),
                    "rss_before_mib": round(before, 1) if before else None,
                    "rss_after_mib": round(after, 1) if after else None,
                    "rss_delta_mib": round(after - before, 1) if process else None,
                })
                del model
                gc.collect()
    return rows


# --------------------------------------------------------------------------- #
# Step 6
# --------------------------------------------------------------------------- #
def verify_parameters() -> Dict[str, Any]:
    """Prove both budgets three ways and show NACT's invariances."""
    v1_spec, v2_spec = ModelSpec(), NactSpec()
    v1, v2 = SalsaTransformer(v1_spec), SalsaNact(v2_spec)

    def three_ways(model, spec) -> Dict[str, Any]:
        actual = sum(p.numel() for p in model.parameters() if p.requires_grad)
        measured = parameter_breakdown(model)
        analytical = analytical_breakdown(spec)
        return {
            "actual_sum_numel_requires_grad": actual,
            "component_breakdown_total": sum(measured.values()),
            "analytical_total": sum(analytical.values()),
            "all_three_agree": actual == sum(measured.values()) == sum(analytical.values()),
            "unclassified_parameters": unclassified_parameters(model),
            "breakdown": {k: v for k, v in measured.items() if v},
        }

    loop_invariance = {
        str(loops): count_trainable_parameters(SalsaNact(NactSpec(encoder_loops=loops)))
        for loops in (1, 2, 4, 8, 16)
    }
    decoder_loop_invariance = {
        str(loops): count_trainable_parameters(SalsaNact(NactSpec(decoder_loops=loops)))
        for loops in (1, 2, 4)
    }
    # n enters only through max_coordinates, which the design fixes at 128.
    n_invariance = {
        str(n): count_trainable_parameters(nact_for(codec_for(n, False)))
        for n in DIMENSIONS
    }

    v1_counts, v2_counts = three_ways(v1, v1_spec), three_ways(v2, v2_spec)
    difference = {
        component: v2_counts["breakdown"].get(component, 0)
                   - v1_counts["breakdown"].get(component, 0)
        for component in set(v1_counts["breakdown"]) | set(v2_counts["breakdown"])
    }
    return {
        "v1": {"expected": V1_PARAMETERS,
               "matches": v1_counts["actual_sum_numel_requires_grad"] == V1_PARAMETERS,
               **v1_counts},
        "v2": {"expected": V2_PARAMETERS,
               "matches": v2_counts["actual_sum_numel_requires_grad"] == V2_PARAMETERS,
               **v2_counts},
        "difference_v2_minus_v1": {k: v for k, v in sorted(
            difference.items(), key=lambda kv: -abs(kv[1])) if v},
        "net_difference": V2_PARAMETERS - V1_PARAMETERS,
        "encoder_loop_invariance": loop_invariance,
        "encoder_loops_do_not_change_count": len(set(loop_invariance.values())) == 1,
        "decoder_loop_invariance": decoder_loop_invariance,
        "decoder_loops_do_not_change_count": len(set(decoder_loop_invariance.values())) == 1,
        "n_invariance": n_invariance,
        "n_does_not_change_count": len(set(n_invariance.values())) == 1,
        "budget": {"min": 4_000_000, "max": 5_000_000,
                   "v1_in_budget": 4_000_000 <= V1_PARAMETERS <= 5_000_000,
                   "v2_in_budget": 4_000_000 <= V2_PARAMETERS <= 5_000_000},
    }


def main(argv: Optional[List[str]] = None) -> int:
    """Run steps 4-6 and write the artifacts."""
    parser = argparse.ArgumentParser(description="NACT architecture validation (no training).")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "results" / "nact_v2")
    args = parser.parse_args(argv)

    torch.manual_seed(0)
    print("=" * 96)
    print("PHASE 15 STEPS 4-6 - NACT VALIDATION, BENCHMARK, PARAMETER PROOF  (no training)")
    print("=" * 96)

    print("\nSTEP 6 - PARAMETER VERIFICATION")
    print("-" * 96)
    parameters = verify_parameters()
    for key in ("v1", "v2"):
        entry = parameters[key]
        print(f"  {key.upper():<4} expected {entry['expected']:>10,}  "
              f"actual {entry['actual_sum_numel_requires_grad']:>10,}  "
              f"match {entry['matches']}  three-way agreement {entry['all_three_agree']}  "
              f"unclassified {len(entry['unclassified_parameters'])}")
    print(f"  encoder loops 1/2/4/8/16 -> {sorted(set(parameters['encoder_loop_invariance'].values()))} "
          f"(invariant: {parameters['encoder_loops_do_not_change_count']})")
    print(f"  decoder loops 1/2/4      -> {sorted(set(parameters['decoder_loop_invariance'].values()))} "
          f"(invariant: {parameters['decoder_loops_do_not_change_count']})")
    print(f"  n = 12..128              -> {sorted(set(parameters['n_invariance'].values()))} "
          f"(invariant: {parameters['n_does_not_change_count']})")
    print(f"  net difference V2 - V1   = {parameters['net_difference']:+,}")
    for component, delta in parameters["difference_v2_minus_v1"].items():
        print(f"      {component:<30}{delta:>+11,}")

    print("\nSTEP 4 - ARCHITECTURE VALIDATION (no accuracy is computed or implied)")
    print("-" * 96)
    validation = validate_dimensions()
    print(f"  {'n':>5}{'rep':>5}{'family':>20}{'src':>8}{'L_enc':>7}"
          f"{'out shape':>16}{'finite':>8}{'exact a':>9}")
    for row in validation:
        print(f"  {row['n']:>5}{row['representation']:>5}{row['input_family']:>20}"
              f"{row['source_shape'][1]:>8}{row['encoder_sequence_length']:>7}"
              f"{str(tuple(row['output_shape'])):>16}"
              f"{str(row['logits_finite']):>8}"
              f"{str(row['values_reconstructed_exactly']):>9}")
    flags = ("source_length_matches_codec", "encoder_length_is_n_plus_2",
             "output_shape_correct", "logits_finite", "no_nan", "memory_finite",
             "generated_ids_in_vocabulary", "values_reconstructed_exactly",
             "features_finite")
    failures = [(r["n"], r["representation"], r["input_family"], f)
                for r in validation for f in flags if not r[f]]
    print(f"\n  {len(validation)} configurations x {len(flags)} checks = "
          f"{len(validation) * len(flags)} assertions, failures: {len(failures)}")
    if failures:
        for failure in failures:
            print(f"      FAIL {failure}")
    vocab = {r["vocab_size"] for r in validation}
    print(f"  vocabulary across every configuration: {vocab}")
    print(f"  max |logit| across every configuration: "
          f"{max(r['logit_abs_max'] for r in validation):.3f}")
    print(f"  max |numerical feature|: {max(r['feature_abs_max'] for r in validation):.6f}")

    print("\nSTEP 5 - FORWARD BENCHMARK (eval mode, no_grad, no optimizer)")
    print("-" * 96)
    bench = benchmark()
    print(f"  {'n':>5}{'rep':>5}{'model':>20}{'T_e':>5}{'params':>12}{'L_src':>7}{'L_enc':>7}"
          f"{'ms/fwd':>9}{'samples/s':>11}{'dRSS MiB':>10}")
    for row in bench:
        print(f"  {row['n']:>5}{row['representation']:>5}{row['model']:>20}"
              f"{row['encoder_loops']:>5}"
              f"{row['parameter_count']:>12,}{row['source_length']:>7}"
              f"{row['encoder_sequence_length']:>7}"
              f"{row['forward_latency_ms_median']:>9.2f}"
              f"{row['forward_throughput_samples_per_second']:>11.1f}"
              f"{(row['rss_delta_mib'] if row['rss_delta_mib'] is not None else 0):>10.1f}")

    payload = {
        "phase": "15 steps 4-6",
        "status": "NO TRAINING. Forward passes only, eval mode, torch.no_grad. "
                  "Models are untrained; no accuracy is computed or implied.",
        "parameter_verification": parameters,
        "dimension_validation": validation,
        "validation_failures": failures,
        "forward_benchmark": bench,
        "torch_version": torch.__version__,
        "threads": torch.get_num_threads(),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "architecture_report.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    import csv

    columns = ["section", "n", "representation", "input_family", "model",
               "parameter_count", "source_length", "encoder_sequence_length", "encoder_loops",
               "output_shape", "logits_finite", "no_nan",
               "values_reconstructed_exactly", "features_finite", "logit_abs_max",
               "zero_coordinates_seen", "forward_latency_ms_median",
               "forward_throughput_samples_per_second", "rss_delta_mib"]
    with (args.out / "architecture_report.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in validation:
            writer.writerow({"section": "validation",
                             **{k: v for k, v in row.items() if k in columns},
                             "source_length": row["source_shape"][1],
                             "output_shape": str(tuple(row["output_shape"]))})
        for row in bench:
            writer.writerow({"section": "benchmark", **row})

    print(f"\n  wrote {args.out / 'architecture_report.json'}")
    print(f"  wrote {args.out / 'architecture_report.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
