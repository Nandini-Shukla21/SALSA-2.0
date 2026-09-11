# SALSA 2.0 — final repository structure

Maintenance reference for the frozen project. Describes the final layout, what
each directory is for, what was removed during the final cleanup, what was
archived, and how to run and test the project.

**Final model: Modified NACT, 4,238,208 parameters.** Historical experiment
records and filenames call it `NACT-F` / `nact_f`.

---

## 1. Final structure

```text
SALSA 2.0/
│
├── salsa/                        the library
│   ├── data/                     LWE/RLWE generation, secrets, encoding
│   │   ├── lwe.py                problem instances; public/private type split
│   │   ├── rlwe.py               circulant and negacyclic constructions
│   │   ├── secrets.py            exact-Hamming-weight binary secrets
│   │   └── encoding.py           LatticeCodec, vocabulary, integer encoder
│   ├── models/                   architectures and parameter accounting
│   │   ├── transformer.py        V1 Salsa2-GatedUT and shared blocks
│   │   ├── nact.py               NACT and Modified NACT
│   │   ├── attention.py          multi-head attention, RoPE
│   │   ├── embeddings.py         token and rotary embeddings
│   │   └── parameter_count.py    three-way budget verification
│   ├── training/                 trainer, losses, metrics, seeding
│   ├── recovery/                 direct.py — direct secret recovery
│   ├── verification/             residual.py — independent residual test
│   ├── evaluation/               reserved (namespace only)
│   ├── utils/                    config, device, logging
│   └── pipeline.py               run_salsa2_pipeline — the entry point
│
├── configs/                      31 YAML experiment definitions
│   ├── pipeline_n12.yaml         final replay, n=12
│   ├── pipeline_n20.yaml         final replay, n=20
│   └── ...                       historical experiment configs
│
├── scripts/                      25 entry points and analysis tools
│   ├── run_pipeline.py           the user-facing CLI
│   ├── show_config.py            inspect a resolved configuration
│   ├── train.py                  historical / research-only
│   ├── final_documentation.py    regenerates the final CSV/JSON/status
│   ├── pipeline_report.py        replays both dimensions, writes the report
│   └── ...                       per-phase analysis and report generators
│
├── tests/                        16 test modules, 690 passed / 2 skipped
│
├── results/
│   ├── README.md                 index — what is authoritative and what is not
│   ├── final_pipeline/           ← AUTHORITATIVE FINAL RESULTS
│   ├── archive/                  historical artifacts nothing depends on
│   └── ...                       supporting evidence cited by the final report
│
├── checkpoints/                  empty placeholder (gitignored)
├── logs/                         empty placeholder (gitignored)
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 2. Purpose of each major directory

| directory | purpose |
|---|---|
| `salsa/` | the library. Everything importable; the only code the pipeline needs |
| `configs/` | every experiment is defined by a config, so results are reproducible from config plus seed |
| `scripts/` | CLI entry points and per-phase analysis tools |
| `tests/` | mathematics, parameter budgets, secret isolation, pipeline, regressions |
| `results/` | recorded artifacts. `final_pipeline/` is authoritative; see `results/README.md` |
| `checkpoints/`, `logs/` | gitignored working directories, kept as placeholders |

## 3. What constitutes the authoritative final result

**Results:** `results/final_pipeline/`. `results/README.md` indexes every file.

**Checkpoints**, loaded by the two pipeline configs:

| dimension | run directory |
|---|---|
| n=12 | `results/nact_ablation_F/nact_n12_h2_te2_F/seed_0/` |
| n=20 | `results/n20_nact_f_pilot/nact_n20_h2_te2_F/seed_0/` |

These were **deliberately left inside the run directories that produced them**,
rather than relocated to a `final/` folder. Each checkpoint therefore stays
beside its own config, metrics, logs and `summary.json`, which is what
`locate_checkpoint` reads and what makes the provenance of each model
self-evident. `results/README.md` names them explicitly so their authority is
not ambiguous despite the historical directory names.

**Headline:** Modified NACT, 4,238,208 parameters. Exact secret recovery and
independent residual verification at n=12, h=2 and n=20, h=2. Replay agreement
12/12. Tests 690 passed, 2 skipped.

## 4. What was removed in the final cleanup

Every deletion was evidence-based: each item was checked for references from
code, configs, tests and documentation before removal.

| removed | why |
|---|---|
| `results/_smoke/` | integration smoke-test output, 1,280 samples / 1 epoch. Zero references. A debug artifact, not an experiment |
| `results/smoke_train/` | smoke-test output, 512 samples. Zero references to the directory; the three apparent hits were to `configs/smoke_train.yaml`, which is retained |
| `results/nact_n12_h2/` | a training run aborted on request. Contained config, metadata and a log — **no checkpoints and no metrics**. Zero references. Held no data |
| `__pycache__/` (9 dirs) | Python bytecode caches; gitignored |
| `lightweight_salsa.egg-info/` | setuptools build artifact; gitignored |
| `.pytest_cache/` | pytest tool cache; gitignored |
| 10 empty directories under `results/` | left behind by an earlier deletion of run files |
| one stray `` ` `` line in `.gitignore` | a typo that matched nothing |

Nothing containing scientific measurements was deleted.

## 5. What was archived

Moved to `results/archive/`: historical artifacts with **zero references** from
any code, config, test or document. Kept for provenance; not needed to reproduce
anything.

`architecture_v2/`, `nact_ablation_design/`, `n30_prediction/`,
`pilot_control_n30_r/`, `v2_checkpoint_audit/`, `v2_direct_recovery/`,
`pilot_comparison.csv`, `pilot_comparison.json`.

Directories still referenced by the final documentation or its report generators
were **left in place**, because relocating them would break those scripts'
paths for no scientific gain. `results/README.md` labels each one.

> `n30_prediction/` holds an **extrapolation**, not a trained model. No model has
> been trained at n=30, n=50 or n=128 in this project.

## 6. How to run the project

Neither command trains anything; each takes about one second.

```bash
python scripts/run_pipeline.py configs/pipeline_n12.yaml
python scripts/run_pipeline.py configs/pipeline_n20.yaml
```

Attack setting, with no ground truth consulted at any point:

```bash
python scripts/run_pipeline.py configs/pipeline_n12.yaml --no-evaluate
```

From Python:

```python
from salsa import run_salsa2_pipeline

result = run_salsa2_pipeline("configs/pipeline_n12.yaml")
print(result.summary_line())
```

Regenerate the final documentation artifacts:

```bash
python scripts/pipeline_report.py        # replays both dimensions
python scripts/final_documentation.py    # CSV, JSON summary, project status
```

## 7. How to run the tests

```bash
pytest -q
```

Expected: **690 passed, 2 skipped**. The two skips are dimension guards.

## 8. Model naming

| name | meaning | parameters |
|---|---:|---:|
| Salsa2-GatedUT (V1) | compact baseline, a faithful reduction of the original architecture | 4,131,200 |
| **NACT** | the full Numerical-Aware Coordinate Transformer | 4,241,288 |
| **Modified NACT** | NACT simplified to the one-token representation — **the final model** | **4,238,208** |

In code, the two NACT variants are selected by `model.nact_variant`:
`full` → NACT, `one_token_only` → Modified NACT. The model reports
`Salsa2-NACT` and `Salsa2-Modified-NACT` respectively.

`nact_f` / `NACT-F` in directory names, config filenames and pre-existing
recorded artifacts is the **historical internal identifier for Modified NACT**.
Those names were not rewritten, because renaming recorded artifacts would damage
provenance and would not match the fingerprints stored inside them.
