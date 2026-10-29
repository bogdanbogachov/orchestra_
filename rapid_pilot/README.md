# Rapid pilot: nuisance-subspace projection for consistent routing

Implements Gates A and B of `rapid_pilot_plan_2026-10-28.md` on three tasks built from Stack Exchange
duplicate groups:

| Task | Labels | Fit / val / test groups | Train singles |
|---|---|---|---|
| `site` | 7 sites: dba, diy, electronics, gis, mechanics, networkengineering, unix | 5174 / 2587 / 5175 | 21,000 |
| `gis` | 7 tool families: esri, gdal, gee, postgis, qgis, r, web | 918 / 459 / 918 | 21,000 |
| `dba` | 5 DBMS families: mongodb, mysql, oracle, postgresql, sql-server | 513 / 256 / 515 | 13,604 |

## Protocol

**Training.** The classifier is trained only on *singles*: questions with no duplicate link at all.
Duplicate groups are split into fit / val / test (40/20/40), and no group spans splits.

**Fitting the projection.** Σ_w and Σ_b are estimated on the fit groups only. Method, layer and rank are
selected on val (val groups are capped at 800 for speed). The selected configuration is evaluated once
on test, with a paired group-level bootstrap against the baseline.

**Projection.** h ← h − QQᵀ(h − μ), applied to every token except BOS, at the output of decoder layer k
or at the final norm.

| Method | Subspace |
|---|---|
| `geig` | Proposed: Σ_w u = λ(Σ_b + εI)u |
| `sw` | Σ_w only |
| `pca` | Principal components of all fit states |
| `random` | Random orthonormal basis |

**Leakage control.** Old closed duplicates contain a "Possible Duplicate: \<title\>" blockquote. These
blockquotes are stripped, together with every link or URL to another question.

## Run on the cluster

1. Pull the repo. The data is committed in `rapid_pilot/data/` (CC BY-SA, 2024-04-01 dump snapshot).
   To rebuild it from the dumps:
   `python -I rapid_pilot/build_data.py --dumps <dir>/<site>/x/ --out rapid_pilot/data`.
2. Make sure `downloaded_models/downloaded_3_2_1b` is present.
3. Submit the job: `sbatch rapid_pilot/job.sh`. This runs a 3-task array in parallel, one GPU each,
   with a 1.5 h limit and 16 GB RAM.
4. Once all three tasks finish, print the go/no-go summary:
   `python -m rapid_pilot.summarize --runs rapid_pilot/runs`.

**GPU choice.** Peak VRAM should be well under 20 GB: a 1B model in bf16, batch 16, at most 256 tokens.
Any bf16 GPU works. A 20–40 GB MIG slice will usually schedule fastest; the slice name depends on the
cluster, so check `sinfo` / the cluster docs before changing `--gpus`. Without bf16 (e.g. V100), the code
falls back to fp32 weights with fp16 autocast, which needs about 2× the memory.

## Outputs per task (`rapid_pilot/runs/<task>/`)

| File | Contents |
|---|---|
| `train_log.json` | Validation accuracy per epoch, train seconds, peak VRAM |
| `report.json` | Fit diagnostics, the full val grid, selections, test results with bootstrap CIs, gate verdicts |
| `report.md` | Readable test table and gate verdicts |

## Gates

| Gate | Condition |
|---|---|
| A | Baseline leaves ≥ 10% of val groups inconsistent |
| B1 | `geig` gains ≥ 3 points in consistency or all-correct rate on val, with ≤ 0.5 points accuracy loss |
| B2 | It beats `sw` and `random` at the same layer and rank |
| B3 | A and B pass on ≥ 2 tasks |

If `geig` wins only at `final`, the in-stream claim reduces to post-hoc LDA.
