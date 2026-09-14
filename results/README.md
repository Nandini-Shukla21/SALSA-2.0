# `results/` — index

> **The authoritative final results are in [`final_pipeline/`](final_pipeline/).**
> Everything else in this directory is supporting or historical evidence from
> earlier project phases and is **not** a competing final result.

**Naming.** The final model is **Modified NACT**. Directory and file names
containing `nact_f` / `NACT-F` are the historical internal identifiers for it,
retained to preserve provenance.

---

## Authoritative final results

| file | what it is |
|---|---|
| [`final_pipeline/SALSA2_Final_Report.md`](final_pipeline/SALSA2_Final_Report.md) | the full technical report |
| [`final_pipeline/final_results.csv`](final_pipeline/final_results.csv) | one row per demonstrated dimension |
| [`final_pipeline/final_summary.json`](final_pipeline/final_summary.json) | machine-readable summary |
| [`final_pipeline/project_status.md`](final_pipeline/project_status.md) | what is done, demonstrated, untested |
| [`final_pipeline/final_structure.md`](final_pipeline/final_structure.md) | repository structure and maintenance notes |
| [`final_pipeline/pipeline_report.md`](final_pipeline/pipeline_report.md) | end-to-end replay validation, 12/12 checks |
| `final_pipeline/nact_f_n12_result.json` | full `PipelineResult` for n=12 |
| `final_pipeline/nact_f_n20_result.json` | full `PipelineResult` for n=20 |

**Headline:** Modified NACT, 4,238,208 parameters. Exact secret recovery and
independent residual verification at **n=12, h=2** and **n=20, h=2**.

## Authoritative checkpoints

These two are the models the final pipeline loads. They live inside the run
directories that produced them, so that each checkpoint stays next to its own
config, metrics and logs.

| dimension | run directory | loaded by |
|---|---|---|
| n=12 | `nact_ablation_F/nact_n12_h2_te2_F/seed_0/` | `configs/pipeline_n12.yaml` |
| n=20 | `n20_nact_f_pilot/nact_n20_h2_te2_F/seed_0/` | `configs/pipeline_n20.yaml` |

## Supporting evidence cited by the final report

| directory | phase | what it establishes |
|---|---|---|
| `original_fidelity_audit/` | fidelity audit | comparison against the released SALSA source |
| `recovery_generalization/` | sparsity study | V1's sparse-input collapse — the diagnosis that motivated NACT |
| `equal_depth_ablation/` | V1 vs NACT | the equal-depth comparison |
| `nact_ablation_F/` | ablation F | the ablation that produced Modified NACT, plus its sparsity evaluation |
| `nact_ablation_F_recovery/` | n=12 recovery | exact recovery at n=12 |
| `v2_verification/` | verification | independent residual verification |
| `v2_recovery_robustness/` | multi-secret | recovery reproduced across three secrets at n=12 |
| `n20_nact_f_pilot/` | n=20 training | learning at n=20 |
| `n20_nact_f_recovery/` | n=20 recovery | exact recovery at n=20 |
| `control_a_n12_h2/`, `control_b_n20_h2/` | V1 baselines | the V1 reference results |
| `direct_recovery/` | V1 recovery | V1's recovery failure at n=12 |
| `nact_v2/` | V1 vs NACT seeds | multi-seed training comparison |
| `v1_v2_audit/` | audit | the preserved snapshot of those runs |
| `control_c_n20_h3/`, `dimension_scaling/`, `pilot_gatedut_n30_r/` | early scaling | early dimension and Hamming-weight studies |

## `archive/`

Historical artifacts that no current code or documentation depends on. Kept for
scientific provenance, not needed to reproduce anything.

| directory | what it is |
|---|---|
| `architecture_v2/` | the v2 architecture design study (design only, no training) |
| `nact_ablation_design/` | the component-attribution study design |
| `n30_prediction/` | n=30 sample-requirement extrapolation — **a prediction, not an experiment** |
| `pilot_control_n30_r/` | early n=30 pilot control |
| `v2_checkpoint_audit/` | checkpoint selection audit |
| `v2_direct_recovery/` | recovery against the full NACT checkpoint |
| `pilot_comparison.csv/.json` | early pilot comparison tables |

> `n30_prediction/` contains an **extrapolation**, not a trained n=30 model.
> No model has ever been trained at n=30, n=50 or n=128 in this project.
