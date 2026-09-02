"""Phase 18: read-only audit of the trained Salsa2-NACT T_e=2 checkpoints.

Nothing is trained, resumed, re-evaluated or benchmarked; no checkpoint is
created or altered; no model, data or recovery code is touched.  The audit
loads each candidate checkpoint into a **fresh** model, runs one tiny synthetic
batch to prove the forward path is intact, and stops.

Two rules govern the selection, and both are enforced structurally:

* the best checkpoint is chosen by the run's **own recorded monitor metric**
  (``valid_loss``, minimised) -- read from each run's configuration rather than
  assumed;
* **no secret is consulted.**  The true secret is never loaded, and no recovery
  result influences the choice.  Selecting an attack's target using the answer
  would invalidate the attack.

The tiny batch exists only to show that logits come out finite and correctly
shaped.  No accuracy is computed from it and none would mean anything.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from salsa.data import LatticeCodec  # noqa: E402
from salsa.models import (  # noqa: E402
    SalsaNact,
    build_model,
    count_trainable_parameters,
)
from salsa.utils import load_config  # noqa: E402

V2_ROOT = RESULTS_V2 = REPO_ROOT / "results" / "equal_depth_ablation" / "v2_te2" / "nact_n12_h2_te2"
V1_RUN = REPO_ROOT / "results" / "control_a_n12_h2" / "control"
OUTPUT_DIR = REPO_ROOT / "results" / "v2_checkpoint_audit"

EXPECTED = {
    "arch": "nact",
    "parameter_count": 4_241_288,
    "encoder_loops": 2,
    "decoder_loops": 2,
    "n": 12,
    "hamming_weight": 2,
    "q": 251,
    "sigma": 3.0,
    "structure": "rlwe",
    "rlwe_variant": "circulant",
    "base": 81,
    "digit_order": "lsb_first",
    "separator": False,
    "fixed_width": True,
    "vocab_size": 85,
    "nact_encoder_length": 14,     # n + 2
    "output_length": 4,            # <bos> d0 d1 <eos>
}


def discover_runs(root: Path) -> List[Path]:
    """Return every run directory under ``root``, completed or not."""
    return sorted((d for d in root.iterdir() if d.is_dir()), key=lambda p: p.name)


def is_complete(run: Path) -> bool:
    """A run counts as complete when it wrote both a summary and a best checkpoint."""
    return ((run / "artifacts" / "summary.json").is_file()
            and (run / "checkpoints" / "best.pt").is_file())


def monitor_from_config(run: Path) -> Dict[str, Any]:
    """Read the selection metric the run itself recorded.

    The metric is never assumed: it comes from the run's own configuration, and
    the default is only used if the run failed to record one.
    """
    path = run / "config.yaml"
    if not path.is_file():
        return {"monitor_metric": None, "monitor_mode": None, "source": "config missing"}
    config = load_config(path)
    return {"monitor_metric": config.training.monitor_metric,
            "monitor_mode": config.training.monitor_mode,
            "source": str(path.relative_to(REPO_ROOT)).replace("\\", "/")}


def audit_run(run: Path) -> Dict[str, Any]:
    """Collect every audited field for one run, without training anything."""
    entry: Dict[str, Any] = {
        "run_dir": str(run.relative_to(REPO_ROOT)).replace("\\", "/"),
        "complete": is_complete(run),
    }
    if not entry["complete"]:
        entry["reason_incomplete"] = (
            "no artifacts/summary.json" if not (run / "artifacts" / "summary.json").is_file()
            else "no checkpoints/best.pt")
        entry["checkpoint_files"] = sorted(
            p.name for p in (run / "checkpoints").glob("*.pt")) if (run / "checkpoints").is_dir() else []
        return entry

    summary = json.loads((run / "artifacts" / "summary.json").read_text("utf-8"))
    state, valid = summary["state"], summary["final_validation"]
    checkpoint_path = run / "checkpoints" / "best.pt"

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    spec, saved = checkpoint["spec"], checkpoint["config"]

    entry.update({
        "seed": summary["seed"],
        "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "best_epoch": state["best_epoch"],
        "final_epoch": state["epoch"],
        "samples_seen": state["samples_seen"],
        "best_metric_recorded": state["best_metric"],
        "stop_reason": state["stop_reason"],
        "valid_loss": valid["valid_loss"],
        "valid_token_accuracy": valid["valid_token_accuracy"],
        "valid_greedy_token_accuracy": valid["valid_greedy_token_accuracy"],
        "valid_exact_accuracy": valid["valid_exact_accuracy"],
        "valid_acc_tau": valid["valid_acc_tau"],
        "valid_perfect_accuracy": valid["valid_perfect_accuracy"],
        "valid_decode_failure_rate": valid["valid_decode_failure_rate"],
        "valid_mean_distance": valid["valid_mean_distance"],
        "cpu_memory_rss_mb": summary["cpu_memory_mb"]["rss_mb"],
        "cpu_memory_peak_rss_mb_UNRELIABLE": summary["cpu_memory_mb"]["peak_rss_mb"],
        "parameter_count_summary": summary["parameter_count"],
        "parameter_count_checkpoint": checkpoint["parameter_count"],
        "architecture": summary["architecture"],
        "encoder_loops": spec["encoder_loops"],
        "decoder_loops": spec["decoder_loops"],
        "config_fingerprint_summary": summary["config_fingerprint"],
        "config_fingerprint_checkpoint": checkpoint["config_fingerprint"],
        # The checkpoint stores no separate weight hash; the config fingerprint
        # is the only stored fingerprint, so "match" means summary vs checkpoint.
        "checkpoint_stores_weight_fingerprint": False,
        "fingerprints_match": (summary["config_fingerprint"]
                               == checkpoint["config_fingerprint"]),
    })
    entry.update(monitor_from_config(run))

    # -- config fidelity, from the checkpoint's own recorded config ---------- #
    lwe, encoding = saved["lwe"], saved["encoding"]
    codec = LatticeCodec.from_config(load_config(run / "config.yaml"))
    checks = {
        "arch": (EXPECTED["arch"], spec["arch"]),
        "encoder_loops": (EXPECTED["encoder_loops"], spec["encoder_loops"]),
        "decoder_loops": (EXPECTED["decoder_loops"], spec["decoder_loops"]),
        "parameter_count": (EXPECTED["parameter_count"], checkpoint["parameter_count"]),
        "n": (EXPECTED["n"], lwe["n"]),
        "hamming_weight": (EXPECTED["hamming_weight"], lwe["hamming_weight"]),
        "q": (EXPECTED["q"], lwe["q"]),
        "sigma": (EXPECTED["sigma"], lwe["sigma"]),
        "structure": (EXPECTED["structure"], lwe["structure"]),
        "rlwe_variant": (EXPECTED["rlwe_variant"], lwe.get("rlwe_variant")),
        "base": (EXPECTED["base"], encoding["base"]),
        "digit_order": (EXPECTED["digit_order"], encoding["digit_order"]),
        "separator": (EXPECTED["separator"], encoding["separator"]),
        "fixed_width": (EXPECTED["fixed_width"], encoding["fixed_width"]),
        "vocab_size": (EXPECTED["vocab_size"], spec["vocab_size"]),
        "codec_vocabulary": (EXPECTED["vocab_size"], codec.vocabulary.size),
        "nact_encoder_length": (EXPECTED["nact_encoder_length"], lwe["n"] + 2),
        "codec_input_length_v1_layout": (2 * EXPECTED["n"] + 2, codec.input_length),
        "output_length": (EXPECTED["output_length"], codec.output_length),
        "seed_recorded": (saved["experiment"]["seed"], summary["seed"]),
    }
    entry["config_fidelity"] = {
        name: {"expected": expected, "actual": actual, "match": expected == actual}
        for name, (expected, actual) in checks.items()
    }
    entry["config_fidelity_all_match"] = all(
        v["match"] for v in entry["config_fidelity"].values())
    return entry


def integrity_check(run: Path) -> Dict[str, Any]:
    """Load into a FRESH model and prove the forward path works.  No training."""
    config = load_config(run / "config.yaml")
    config.validate()
    checkpoint = torch.load(run / "checkpoints" / "best.pt",
                            map_location="cpu", weights_only=False)
    report: Dict[str, Any] = {}

    model = build_model(config)
    report["fresh_model_is_SalsaNact"] = isinstance(model, SalsaNact)

    expected_keys = set(model.state_dict().keys())
    saved_keys = set(checkpoint["model"].keys())
    report["state_dict_expected_tensors"] = len(expected_keys)
    report["state_dict_saved_tensors"] = len(saved_keys)
    report["missing_keys"] = sorted(expected_keys - saved_keys)
    report["unexpected_keys"] = sorted(saved_keys - expected_keys)
    report["all_expected_keys_present"] = not (expected_keys - saved_keys)

    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    report["loads_strict"] = True
    report["load_missing"] = list(getattr(incompatible, "missing_keys", []))
    report["load_unexpected"] = list(getattr(incompatible, "unexpected_keys", []))

    model.eval()
    report["parameter_count_after_load"] = count_trainable_parameters(model)
    report["parameter_count_matches"] = (
        report["parameter_count_after_load"] == checkpoint["parameter_count"])
    report["all_parameters_on_cpu"] = all(p.device.type == "cpu" for p in model.parameters())
    report["all_parameters_finite"] = all(
        bool(torch.isfinite(p).all()) for p in model.parameters())
    report["model_config_matches_checkpoint_spec"] = (
        model.spec.arch == checkpoint["spec"]["arch"]
        and model.spec.encoder_loops == checkpoint["spec"]["encoder_loops"]
        and model.spec.decoder_loops == checkpoint["spec"]["decoder_loops"]
        and model.spec.vocab_size == checkpoint["spec"]["vocab_size"])

    # -- ONE tiny synthetic batch.  Shapes and finiteness only. ------------- #
    codec = LatticeCodec.from_config(config)
    rng = np.random.default_rng(0)
    matrix = rng.integers(0, codec.q, size=(2, codec.n), dtype=np.int64)
    matrix[1, :] = 0                       # include an all-zero row
    src_np, _ = codec.encode_batch(matrix)
    src = torch.from_numpy(src_np)
    target = torch.randint(4, 85, (2, codec.output_length))

    with torch.no_grad():
        memory = model.encode(src)
        logits = model(src, target)

    report["tiny_batch"] = {
        "note": "shape and finiteness only; NO accuracy is computed or implied",
        "batch_size": 2,
        "source_shape": list(src.shape),
        "encoder_memory_shape": list(memory.shape),
        "encoder_length_is_n_plus_2": int(memory.shape[1]) == codec.n + 2,
        "logits_shape": list(logits.shape),
        "logits_shape_correct": tuple(logits.shape) == (2, codec.output_length, 85),
        "logits_finite": bool(torch.isfinite(logits).all()),
        "logits_contain_nan": bool(torch.isnan(logits).any()),
        "logit_abs_max": round(float(logits.abs().max()), 4),
        "values_reconstructed_exactly": bool(np.array_equal(
            model.front_end.decode_values(src)[1].numpy(), matrix)),
    }
    report["integrity_all_pass"] = bool(
        report["fresh_model_is_SalsaNact"] and report["all_expected_keys_present"]
        and not report["unexpected_keys"] and report["parameter_count_matches"]
        and report["all_parameters_on_cpu"] and report["all_parameters_finite"]
        and report["model_config_matches_checkpoint_spec"]
        and report["tiny_batch"]["logits_shape_correct"]
        and report["tiny_batch"]["logits_finite"]
        and not report["tiny_batch"]["logits_contain_nan"]
        and report["tiny_batch"]["encoder_length_is_n_plus_2"])
    return report


def v1_context() -> Dict[str, Any]:
    """Metadata-only confirmation of the V1 baseline used in phase 10."""
    summary_path = V1_RUN / "artifacts" / "summary.json"
    checkpoint_path = V1_RUN / "checkpoints" / "best.pt"
    if not summary_path.is_file():
        return {"available": False}
    summary = json.loads(summary_path.read_text("utf-8"))
    saved = summary.get("experiment", {})
    config = load_config(V1_RUN / "config.yaml") if (V1_RUN / "config.yaml").is_file() else None
    return {
        "available": True,
        "note": "metadata inspection only; V1 was not reloaded or retrained",
        "run_dir": str(V1_RUN.relative_to(REPO_ROOT)).replace("\\", "/"),
        "checkpoint_path": (str(checkpoint_path.relative_to(REPO_ROOT)).replace("\\", "/")
                            if checkpoint_path.is_file() else None),
        "checkpoint_present": checkpoint_path.is_file(),
        "architecture": summary["architecture"],
        "model_name": summary["model_name"],
        "parameter_count": summary["parameter_count"],
        "n": config.lwe.n if config else None,
        "hamming_weight": config.lwe.resolved_hamming_weight if config else None,
        "seed": summary["seed"],
        "valid_loss": summary["final_validation"]["valid_loss"],
        "valid_acc_tau": summary["final_validation"]["valid_acc_tau"],
        "valid_exact_accuracy": summary["final_validation"]["valid_exact_accuracy"],
        "config_fingerprint": summary["config_fingerprint"],
        "matches_phase10_expectations": (
            summary["architecture"] == "gated_universal_transformer"
            and summary["parameter_count"] == 4_131_200
            and (config.lwe.n == 12 if config else False)
            and (config.lwe.resolved_hamming_weight == 2 if config else False)),
    }


def main() -> int:
    """Run the audit."""
    print("=" * 96)
    print("PHASE 18 - V2 NACT T_e=2 CHECKPOINT AUDIT   (read-only; no training, no recovery)")
    print("=" * 96)

    runs = discover_runs(V2_ROOT)
    audits = [audit_run(run) for run in runs]
    complete = [a for a in audits if a["complete"]]
    incomplete = [a for a in audits if not a["complete"]]

    print(f"\n  discovered {len(runs)} run directories under "
          f"{V2_ROOT.relative_to(REPO_ROOT)}".replace("\\", "/"))
    for entry in incomplete:
        print(f"    EXCLUDED {Path(entry['run_dir']).name}: {entry['reason_incomplete']} "
              f"(checkpoints present: {entry.get('checkpoint_files')})")

    print(f"\n  {len(complete)} completed checkpoint(s):")
    print(f"  {'seed':>5}{'epoch':>7}{'samples':>10}{'loss':>9}{'acc_tau':>9}{'exact':>8}"
          f"{'token':>8}{'greedy':>8}{'params':>11}{'T_e':>4}{'T_d':>4}{'fp match':>10}")
    for entry in complete:
        print(f"  {entry['seed']:>5}{entry['best_epoch']:>7}{entry['samples_seen']:>10,}"
              f"{entry['valid_loss']:>9.4f}{entry['valid_acc_tau']:>9.4f}"
              f"{entry['valid_exact_accuracy']:>8.4f}{entry['valid_token_accuracy']:>8.4f}"
              f"{entry['valid_greedy_token_accuracy']:>8.4f}"
              f"{entry['parameter_count_checkpoint']:>11,}{entry['encoder_loops']:>4}"
              f"{entry['decoder_loops']:>4}{str(entry['fingerprints_match']):>10}")

    if not complete:
        print("\n  NO COMPLETED CHECKPOINTS - nothing to select.")
        return 2

    # -- selection: the run's own monitor metric, no secret consulted ------- #
    metrics = {e["monitor_metric"] for e in complete}
    modes = {e["monitor_mode"] for e in complete}
    if len(metrics) != 1 or len(modes) != 1:
        print(f"\n  MONITOR METRIC DISAGREEMENT across runs: {metrics} / {modes} - STOPPING")
        return 2
    metric, mode = metrics.pop(), modes.pop()
    key = metric if metric in complete[0] else "valid_loss"
    reverse = (mode == "max")
    ranked = sorted(complete, key=lambda e: e[key], reverse=reverse)
    best = ranked[0]

    print(f"\n  selection metric: {metric} ({mode}) - read from each run's own config")
    print(f"  ranking: " + "  ".join(
        f"seed {e['seed']}={e[key]:.4f}" for e in ranked))
    print(f"\n  BEST: seed {best['seed']}  {best['checkpoint_path']}")

    print("\n  CONFIG FIDELITY (selected checkpoint)")
    for name, check in best["config_fidelity"].items():
        flag = "OK " if check["match"] else "MISMATCH"
        print(f"    [{flag}] {name:<28}expected {str(check['expected']):>20} | "
              f"actual {check['actual']}")

    print("\n  INTEGRITY CHECK (fresh model, one tiny synthetic batch)")
    integrity = integrity_check(V2_ROOT / f"seed_{best['seed']}")
    for name in ("fresh_model_is_SalsaNact", "loads_strict", "all_expected_keys_present",
                 "parameter_count_matches", "all_parameters_on_cpu",
                 "all_parameters_finite", "model_config_matches_checkpoint_spec"):
        print(f"    [{'OK ' if integrity[name] else 'FAIL'}] {name}")
    tiny = integrity["tiny_batch"]
    for name in ("logits_shape_correct", "logits_finite", "encoder_length_is_n_plus_2",
                 "values_reconstructed_exactly"):
        print(f"    [{'OK ' if tiny[name] else 'FAIL'}] {name}")
    print(f"    [{'OK ' if not tiny['logits_contain_nan'] else 'FAIL'}] no NaNs")
    print(f"    shapes: src {tuple(tiny['source_shape'])} -> memory "
          f"{tuple(tiny['encoder_memory_shape'])} -> logits {tuple(tiny['logits_shape'])}")

    v1 = v1_context()
    print("\n  V1 BASELINE (metadata only, not reloaded)")
    print(f"    {v1['model_name']}  {v1['parameter_count']:,} params  "
          f"n={v1['n']} h={v1['hamming_weight']}  "
          f"matches phase-10 expectations: {v1['matches_phase10_expectations']}")

    ready = bool(best["config_fidelity_all_match"] and integrity["integrity_all_pass"]
                 and best["fingerprints_match"])
    print(f"\n  READY FOR DIRECT RECOVERY: {'YES' if ready else 'NO'}")

    payload = {
        "phase": "18 - V2 NACT T_e=2 checkpoint audit",
        "status": "READ-ONLY. No training, no resume, no recovery, no distinguisher, no "
                  "verification, no sparsity rerun, no benchmark. No checkpoint created "
                  "or altered. No model, data or recovery code modified.",
        "selection_rule": {
            "metric": metric, "mode": mode,
            "source": "each run's own config.yaml training.monitor_metric",
            "secret_consulted": False,
            "recovery_result_consulted": False,
            "note": ("The true secret is never loaded by this script and no recovery "
                     "outcome influences the choice; selecting an attack target using "
                     "the answer would invalidate the attack."),
        },
        "expected_values": EXPECTED,
        "runs_discovered": len(runs),
        "completed": complete,
        "excluded": incomplete,
        "ranking": [{"seed": e["seed"], metric: e[key]} for e in ranked],
        "best_checkpoint": {
            "path": best["checkpoint_path"],
            "seed": best["seed"],
            "best_epoch": best["best_epoch"],
            "samples_seen": best["samples_seen"],
            "valid_loss": best["valid_loss"],
            "valid_acc_tau": best["valid_acc_tau"],
            "valid_exact_accuracy": best["valid_exact_accuracy"],
            "config_fingerprint": best["config_fingerprint_checkpoint"],
            "parameter_count": best["parameter_count_checkpoint"],
        },
        "integrity": integrity,
        "v1_baseline_context": v1,
        "ready_for_direct_recovery": ready,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "checkpoint_audit.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    columns = ["section", "seed", "run_dir", "checkpoint_path", "complete", "best_epoch",
               "samples_seen", "valid_loss", "valid_token_accuracy",
               "valid_greedy_token_accuracy", "valid_exact_accuracy", "valid_acc_tau",
               "valid_perfect_accuracy", "valid_decode_failure_rate",
               "cpu_memory_rss_mb", "parameter_count_checkpoint", "architecture",
               "encoder_loops", "decoder_loops", "config_fingerprint_summary",
               "config_fingerprint_checkpoint", "fingerprints_match",
               "config_fidelity_all_match", "monitor_metric", "monitor_mode",
               "check", "expected", "actual", "match"]
    with (OUTPUT_DIR / "checkpoint_audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for entry in audits:
            writer.writerow({"section": "run", **entry})
        for name, check in best["config_fidelity"].items():
            writer.writerow({"section": "config_fidelity_selected", "seed": best["seed"],
                             "check": name, "expected": check["expected"],
                             "actual": check["actual"], "match": check["match"]})
        for name, value in integrity.items():
            if isinstance(value, bool):
                writer.writerow({"section": "integrity_selected", "seed": best["seed"],
                                 "check": name, "actual": value, "match": value})

    write_markdown(payload, OUTPUT_DIR / "checkpoint_audit.md")
    for name in ("checkpoint_audit.md", "checkpoint_audit.json", "checkpoint_audit.csv"):
        print(f"  wrote {OUTPUT_DIR / name}")
    return 0


def write_markdown(payload: Dict[str, Any], path: Path) -> None:
    """Render the audit report."""
    best, integrity, v1 = (payload["best_checkpoint"], payload["integrity"],
                           payload["v1_baseline_context"])
    tiny = integrity["tiny_batch"]
    selected = next(e for e in payload["completed"] if e["seed"] == best["seed"])

    lines = [
        "# Phase 18 — V2 NACT T_e=2 checkpoint audit",
        "",
        "**Read-only.** No training, no resume, no recovery, no distinguisher, no",
        "verification, no sparsity rerun, no benchmark. No checkpoint was created or",
        "altered; no model, data or recovery code was modified.",
        "",
        "## Available checkpoints",
        "",
        f"Scanned `{Path(payload['completed'][0]['run_dir']).parent.as_posix()}` — "
        f"{payload['runs_discovered']} run directories, "
        f"{len(payload['completed'])} completed.",
        "",
        "| seed | best epoch | samples | valid loss | acc_tau | exact | token | greedy | params | T_e | T_d | fingerprints match |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for entry in payload["completed"]:
        lines.append(
            f"| {entry['seed']} | {entry['best_epoch']} | {entry['samples_seen']:,} | "
            f"{entry['valid_loss']:.4f} | {entry['valid_acc_tau']:.4f} | "
            f"{entry['valid_exact_accuracy']:.4f} | {entry['valid_token_accuracy']:.4f} | "
            f"{entry['valid_greedy_token_accuracy']:.4f} | "
            f"{entry['parameter_count_checkpoint']:,} | {entry['encoder_loops']} | "
            f"{entry['decoder_loops']} | {'yes' if entry['fingerprints_match'] else 'NO'} |")
    lines += [
        "",
        "| seed | checkpoint path | CPU RSS (MiB) | architecture | config fingerprint |",
        "|---:|---|---:|---|---|",
    ]
    for entry in payload["completed"]:
        lines.append(f"| {entry['seed']} | `{entry['checkpoint_path']}` | "
                     f"{entry['cpu_memory_rss_mb']:.1f} | {entry['architecture']} | "
                     f"`{entry['config_fingerprint_checkpoint']}` |")
    if payload["excluded"]:
        lines += ["", "**Excluded:**", ""]
        for entry in payload["excluded"]:
            lines.append(f"- `{Path(entry['run_dir']).name}` — {entry['reason_incomplete']}. "
                         f"Checkpoint files present: {entry.get('checkpoint_files')}. "
                         "Not deleted.")
    lines += [
        "",
        "The checkpoints store **no separate weight fingerprint**; the only stored",
        "fingerprint is the config fingerprint, so \"fingerprints match\" compares the one",
        "in the checkpoint against the one in the run summary.",
        "",
        "## Selection",
        "",
        f"Metric **`{payload['selection_rule']['metric']}` "
        f"({payload['selection_rule']['mode']})**, read from "
        f"{payload['selection_rule']['source']} — not assumed.",
        "",
        "Ranking: " + ", ".join(
            f"seed {r['seed']} = {r[payload['selection_rule']['metric']]:.4f}"
            for r in payload["ranking"]) + ".",
        "",
        "**No secret was consulted and no recovery result influenced the choice.** "
        f"{payload['selection_rule']['note']}",
        "",
        "```",
        "BEST CHECKPOINT:",
        f"path              {best['path']}",
        f"seed              {best['seed']}",
        f"epoch             {best['best_epoch']}",
        f"samples           {best['samples_seen']:,}",
        f"validation loss   {best['valid_loss']:.6f}",
        f"acc_tau           {best['valid_acc_tau']:.6f}",
        f"exact accuracy    {best['valid_exact_accuracy']:.6f}",
        f"fingerprint       {best['config_fingerprint']}",
        "```",
        "",
        "## Config fidelity — selected checkpoint",
        "",
        "| check | expected | actual | match |",
        "|---|---|---|:---:|",
    ]
    for name, check in selected["config_fidelity"].items():
        lines.append(f"| `{name}` | {check['expected']} | {check['actual']} | "
                     f"{'✓' if check['match'] else '**✗**'} |")
    lines += [
        "",
        f"All checks match: **{selected['config_fidelity_all_match']}**.",
        "",
        "Note on lengths: the codec still emits the **V1 token layout** of "
        f"{2 * payload['expected_values']['n'] + 2} tokens, which NACT reads and folds into "
        f"its own **{payload['expected_values']['nact_encoder_length']}-position** encoder "
        "sequence (`n+2`). Output length stays "
        f"{payload['expected_values']['output_length']} and the vocabulary stays "
        f"{payload['expected_values']['vocab_size']}, so the recovery interface is unchanged.",
        "",
        "## Integrity check",
        "",
        "Loaded into a **fresh** `SalsaNact`; one tiny synthetic batch of 2 rows (one",
        "random, one all-zero). **No accuracy was computed from it and none would mean",
        "anything.**",
        "",
        "| check | result |",
        "|---|:---:|",
        f"| checkpoint loads (`strict=True`) | {'✓' if integrity['loads_strict'] else '✗'} |",
        f"| fresh model is `SalsaNact` | {'✓' if integrity['fresh_model_is_SalsaNact'] else '✗'} |",
        f"| all expected state_dict keys present | {'✓' if integrity['all_expected_keys_present'] else '✗'} "
        f"({integrity['state_dict_saved_tensors']}/{integrity['state_dict_expected_tensors']} tensors) |",
        f"| no unexpected keys | {'✓' if not integrity['unexpected_keys'] else '✗'} |",
        f"| parameter count after load | {'✓' if integrity['parameter_count_matches'] else '✗'} "
        f"({integrity['parameter_count_after_load']:,}) |",
        f"| model config matches checkpoint spec | {'✓' if integrity['model_config_matches_checkpoint_spec'] else '✗'} |",
        f"| all parameters on CPU | {'✓' if integrity['all_parameters_on_cpu'] else '✗'} |",
        f"| all parameters finite | {'✓' if integrity['all_parameters_finite'] else '✗'} |",
        f"| logits shape correct | {'✓' if tiny['logits_shape_correct'] else '✗'} "
        f"({tuple(tiny['logits_shape'])}) |",
        f"| logits finite | {'✓' if tiny['logits_finite'] else '✗'} "
        f"(max abs {tiny['logit_abs_max']}) |",
        f"| no NaNs | {'✓' if not tiny['logits_contain_nan'] else '✗'} |",
        f"| encoder length = n+2 | {'✓' if tiny['encoder_length_is_n_plus_2'] else '✗'} "
        f"({tuple(tiny['encoder_memory_shape'])}) |",
        f"| front end reconstructs `a` exactly | {'✓' if tiny['values_reconstructed_exactly'] else '✗'} |",
        "",
        f"Shapes: `src {tuple(tiny['source_shape'])}` → `memory "
        f"{tuple(tiny['encoder_memory_shape'])}` → `logits {tuple(tiny['logits_shape'])}`.",
        "",
        "## V1 baseline, for context",
        "",
        f"*{v1['note']}.*",
        "",
        "| field | value |",
        "|---|---|",
        f"| run | `{v1['run_dir']}` |",
        f"| checkpoint | `{v1['checkpoint_path']}` |",
        f"| architecture | {v1['architecture']} |",
        f"| parameters | {v1['parameter_count']:,} |",
        f"| n / h | {v1['n']} / {v1['hamming_weight']} |",
        f"| valid loss / acc_tau / exact | {v1['valid_loss']:.4f} / "
        f"{v1['valid_acc_tau']:.4f} / {v1['valid_exact_accuracy']:.4f} |",
        f"| matches phase-10 expectations | {'✓' if v1['matches_phase10_expectations'] else '✗'} |",
        "",
        "This is the checkpoint phase 10 attacked, where direct recovery returned NO.",
        "",
        "## Caveats carried forward",
        "",
        "- The V2 seed sweep is **incomplete** (3 of 5 seeds). Selecting the best of three",
        "  is a valid secret-free rule, but it is a smaller pool than intended.",
        "- The three completed seeds are nearly indistinguishable on the selection metric,",
        "  so the choice between them is close to arbitrary on the evidence.",
        "- Nothing here says anything about recovery. Phase 17's sparsity result is",
        "  suggestive of better probe behaviour, but recovery has not been run against V2.",
        "",
        "---",
        "",
        f"**READY FOR DIRECT RECOVERY: {'YES' if payload['ready_for_direct_recovery'] else 'NO'}**",
        "",
    ]
    if payload["ready_for_direct_recovery"]:
        lines += [
            "**SELECTED CHECKPOINT:**",
            "",
            f"`{best['path']}`",
            "",
            "**REASON:**",
            "",
            f"It has the lowest recorded `valid_loss` ({best['valid_loss']:.4f}) of the "
            "three completed equal-depth NACT runs under the run's own monitor metric, "
            "loads cleanly into a fresh model with all 4,241,288 parameters and every "
            "config value matching, and was selected without consulting the secret.",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
