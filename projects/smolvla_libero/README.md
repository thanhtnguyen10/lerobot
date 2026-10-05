# SmolVLA on LIBERO: training and evaluation pipeline

Train SmolVLA (0.45B) on LIBERO with the recipe of the SmolVLA paper (Shukor et al., 2025), evaluate it on vanilla
LIBERO with the paper's protocol, and measure its robustness on LIBERO-Plus
([Fei et al., 2025](https://arxiv.org/abs/2510.13626)) with episodes paired to the vanilla ones. Built on this
LeRobot fork; all commands run from the repo root. Every choice and its source is in [`PROTOCOL.md`](PROTOCOL.md).

## Files

```
projects/smolvla_libero/
├── README.md                  this file
├── PROTOCOL.md                every training/evaluation choice, its source, and the reproducibility checks
├── setup/
│   ├── setup_envs.sh          1. build the environments and download third-party code (once)
│   └── paths.sh                  paths and pinned versions (also used by train.sh)
├── train/
│   └── train.sh               2. train SmolVLA on LIBERO
├── eval_vanilla/
│   ├── eval_vanilla.py        3. evaluate on vanilla LIBERO (SmolVLA paper protocol)
│   └── deterministic_eval.py     runs lerobot-eval with deterministic GPU kernels (used by 3 and 4)
├── eval_combined/
│   ├── eval_combined.py       4. evaluate on vanilla LIBERO + LIBERO-Plus, paired episode by episode
│   └── libero_plus_patch/
│       └── sitecustomize.py      per-episode overrides inside LIBERO-Plus processes
└── analysis/
    ├── robustness_gap.py      5. robustness-gap table (Vanilla / OOD / Δ), plots, task ranking from a step-4 run
    └── export_excel.py        6. Excel workbooks of all results (formulas over per-episode sheets)
```

`eval_combined.py` imports the vanilla protocol (`vanilla_job`) and helpers from `eval_vanilla.py`, so the vanilla
numbers of steps 3 and 4 are identical.

## 1. Setup (once)

```bash
bash projects/smolvla_libero/setup/setup_envs.sh
```

Creates, inside the repo and git-ignored: `.venv` (LeRobot's default environment from `uv.lock`, Python 3.12,
mujoco 3.8.1), `.venv-liberoplus` (the same with the LIBERO-Plus fork replacing `hf-libero`; both install as
`libero`, so they need separate environments), and `third_party/` (LIBERO-Plus @ `4976dc3` + 9.5 GB of assets,
ImageMagick from conda, LIBERO config files). Needs `uv`, `conda`, `git` and ~35 GB of disk.

## 2. Train

```bash
RUN=smolvla_libero_seed1000 SEED=1000 nohup bash projects/smolvla_libero/train/train.sh > train.log 2>&1 &
```

SmolVLA paper recipe: initialized from the SmolVLM2 weights only, VLM frozen (only the 100M-parameter action expert
is trained), 100k steps, batch 64, action chunks of 50, cosine LR 1e-4 → 2.5e-6, bf16, `torch.compile`, on
`lerobot/libero` at a pinned revision. Checkpoints: `outputs/$RUN/checkpoints/{020000,…,100000}/pretrained_model`
(`outputs/$RUN` must not exist yet). Extra flags are passed to `lerobot-train`, e.g. `--steps=200` for a quick test.
~18 h on one RTX PRO 6000 with `--num_workers=0` (this container's `/dev/shm` is 64 MB); ~8.5 h with a larger
`/dev/shm` and that flag removed.

## 3. Evaluate on vanilla LIBERO

```bash
python3 projects/smolvla_libero/eval_vanilla/eval_vanilla.py \
  --policy outputs/$RUN/checkpoints/100000/pretrained_model --out outputs/eval/${RUN}_vanilla
```

4 suites × 10 tasks × 10 episodes = 400 episodes; episode *k* of a task starts from LIBERO's stored initial state
*k* with seed `1000 + k`; a new policy call after every executed action; 256×256 images. Results: `summary.md`
(by suite, by task).

## 4. Evaluate on vanilla LIBERO + LIBERO-Plus

```bash
python3 projects/smolvla_libero/eval_combined/eval_combined.py \
  --policy outputs/$RUN/checkpoints/100000/pretrained_model --out outputs/eval/${RUN}_robustness
```

The vanilla evaluation of step 3, plus, for each of the 7 LIBERO-Plus perturbations, the same 40 tasks × up to
10 episodes with that perturbation applied. LIBERO-Plus episode *k* of a task uses the same initial state *k*,
seed and instruction as vanilla episode *k* (where the perturbation allows), so each one has a vanilla twin.
Metrics: success rate, and the gap Δ = OOD success − vanilla success, both on the same (twin) episodes.

| Perturbation (`--perturbations` name) | Identical to the vanilla episode | Episodes |
|---|---|---|
| Camera Viewpoints (`camera`) | task, initial state, seed, instruction; only the camera view changes | 396 |
| Robot Initial States (`robot`) | task, objects' initial state, seed, instruction; the robot's start pose changes | 400 |
| Light Conditions (`light`) | task, initial state, seed, instruction; only the lighting changes | 376 |
| Background Textures (`background`) | task, initial state, seed, instruction; only scene textures change | 385 |
| Sensor Noise (`noise`) | task, initial state, seed, instruction; only the image noise changes | 400 |
| Language Instructions (`language`) | task, initial state, seed; the instruction is reworded | 400 |
| Objects Layout (`layout`) | task, seed, instruction; the scene layout changes (own initial state) | 400 |

Each LIBERO-Plus variant is used once; 9 of the 280 (task, perturbation) pairs have only 4–8 variants, hence the
counts below 400. Total: 400 vanilla + 2,757 LIBERO-Plus = 3,157 episodes, ~20 h (measured: 1,189 min with 11 processes; one episode at a time on
one GPU). Runs 4 vanilla processes (one per suite) and 28 LIBERO-Plus processes (one per perturbation and suite,
started perturbation by perturbation); `summary.md` is refreshed after each, so results arrive in complete blocks.

Outputs in `--out`: `summary.md` / `summary.json` (by suite, paired comparison per perturbation, by task, by
LIBERO-Plus difficulty level), `plus_episodes.jsonl` (one row per LIBERO-Plus episode with its vanilla twin's
outcome), `run_config.json` (exact commands and versions), `vanilla/<suite>/`, `plus/<perturbation>/<suite>/`, `logs/`.

## 5. Robustness-gap table and plots

```bash
.venv/bin/python projects/smolvla_libero/analysis/robustness_gap.py --run outputs/eval/${RUN}_robustness
```

Writes to `<run>/gap/`: `gap_table.{md,csv,tex}` (Vanilla / OOD / Δ success per perturbation; Vanilla = success without perturbation on
the same episodes), `gap_by_suite.{md,csv}`, `robustness_gap.{png,pdf}` (per-perturbation plot), `gap_heatmap.{png,pdf}`
(Δ by suite × perturbation) and `task_ranking.{md,csv}` (tasks with vanilla success ≥ 80%, all variants available and
no known simulator issue, ranked by mean drop; for picking tasks for rapid experiments).

## 6. Excel workbooks

```bash
uv run --no-project --with openpyxl python projects/smolvla_libero/analysis/export_excel.py \
  --run outputs/eval/${RUN}_robustness --history outputs/eval/archive/runs.json --out outputs/eval/excel
```

`1_vanilla_libero.xlsx` (by suite vs the paper, by task), `2_libero_plus_robustness.xlsx` (robustness gap, by suite,
task × perturbation, difficulty level, task ranking) and, with `--history`, `3_vanilla_run_history.xlsx` (every earlier
evaluation listed in `runs.json`: MuJoCo comparison, eval-seed check, determinism checks). Every summary number is a
formula over per-episode sheets in the same workbook. `--history` is optional.

## Options (steps 3 and 4)

| Option | Default | Meaning |
|---|---|---|
| `--episodes` | `10` | episodes per task |
| `--seed` | `1000` | episode *k* of every task uses `seed + k` |
| `--suites` | all 4 | subset of `libero_spatial,libero_object,libero_goal,libero_10` |
| `--workers` | `11` | concurrent `lerobot-eval` processes |
| `--dry-run` | | show the plan without evaluating |
| `--perturbations` (4 only) | all 7 | subset of the names above |
| `--sample-seed` (4 only) | `0` | fixes which LIBERO-Plus variants are used, so every checkpoint gets the same ones |
| `--instructions` (4 only) | `vanilla` | non-language variants keep the vanilla instruction; `official` = LIBERO-Plus's own text |

Re-running with the same `--out` resumes: finished processes are skipped.

## Notes

- **Reproducibility.** `lerobot-eval` enables `cudnn.benchmark`, which picks GPU kernels by timing and so depends on
  GPU load; the kernel choice alone changed 8% of episode outcomes in our tests. Both evaluations run it through
  `eval_vanilla/deterministic_eval.py` with batch size 1 everywhere, so reruns reproduce every episode regardless of GPU
  load (checks in `PROTOCOL.md` §5).
- **MuJoCo version.** mujoco 3.8.1 (LeRobot's lock) differs from the version the LIBERO demonstrations were rendered
  with; it costs a few points, mostly on LIBERO-Spatial task 5 (`PROTOCOL.md` §3).
- **Parallel tasks.** LeRobot's `--env.max_parallel_tasks>1` is not used: it shares one policy across threads, which
  mixes up SmolVLA's action queue. Parallelism comes from separate processes.
- **LIBERO-Plus fixes** (in `libero_plus_patch/sitecustomize.py`): episode pairing (initial state and seed), vanilla
  instructions instead of file-name-derived ones, and `np.float_` for its fog noise under NumPy 2. Sensor noise is
  applied to the policy's observations; the videos LeRobot saves are rendered without it. LIBERO-Plus processes
  preload conda's `libjpeg` and GLib, which ImageMagick needs.
