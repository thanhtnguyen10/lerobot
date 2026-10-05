# Experimental protocol

Every choice behind the SmolVLA LIBERO / LIBERO-Plus numbers produced by this pipeline, with what it follows and
why, so it can be stated in a paper. "SmolVLA paper" = Shukor et al., 2025 (arXiv:2506.01844);
"LIBERO-Plus paper" = Fei et al., 2025 (arXiv:2510.13626).

## 1. Software and hardware

| Component | Version |
|---|---|
| LeRobot | fork of `huggingface/lerobot` at `eb7a54c6` (0.6.2) + `projects/smolvla_libero/` |
| Environment | `uv sync --locked` from LeRobot's `uv.lock`, Python 3.12.13, extras `smolvla`, `libero`, `training` |
| Key packages | torch 2.11.0+cu128, transformers 5.5.4, numpy 2.2.6, gymnasium 1.3.0, torchcodec 0.11.1 |
| Simulator | mujoco 3.8.1, robosuite 1.4.0, hf-libero 0.1.4 (vanilla LIBERO) |
| LIBERO-Plus | `sylvestf/LIBERO-plus` at `4976dc3` (the commit LeRobot's `docker/Dockerfile.benchmark.libero_plus` pins) + its asset pack, in a separate environment identical to the above except that the fork replaces `hf-libero` |
| Data | `lerobot/libero` revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` (1,693 episodes, 273,465 frames, 40 tasks) |
| VLM weights | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| Hardware | 1× NVIDIA RTX PRO 6000 Blackwell (96 GB), driver 575.57.08, CUDA 12.8 |

All of it is rebuilt by `setup/setup_envs.sh` (environments, LIBERO-Plus clone and assets, ImageMagick).

## 2. Training (follows the SmolVLA paper, Sec. 4.3)

| Setting | Value | Source |
|---|---|---|
| Initialization | SmolVLM2 weights only, no robot pretraining ("SmolVLA is only initialized from the VLM", Table 2) | paper |
| Trained parameters | action expert only (100M of 450M); VLM frozen | paper |
| Architecture | first 16 VLM layers, expert width 0.75×, interleaved cross/self-attention, flow matching | paper |
| Steps / batch | 100,000 / 64 | paper |
| Action chunk | n = 50 | paper |
| Optimizer | AdamW, β = (0.9, 0.95), cosine LR 1e-4 → 2.5e-6, 1,000 warmup steps | paper (LR); LeRobot default (warmup) |
| LR decay length | over the full 100k steps | **choice**: the paper does not state it; LeRobot's default (30k) would leave 70k steps at the minimum LR |
| Precision / speed | bf16, `torch.compile` | paper ("bfloat16 precision and torch.compile()") |
| Fixed sequence length | language padded to 48 tokens | paper ("fixed sequence length") |
| Image size | 512×512 after padding, from 256×256 dataset images | paper |
| Seed | 1000 (single training run) | choice |
| Data loading | 1 process (`--num_workers=0`; container `/dev/shm` is 64 MB) | environment constraint, affects speed only |

Training is not bit-reproducible (GPU kernels in the backward pass and `torch.compile` are not deterministic), so
the checkpoint itself is the reference artifact; evaluation of a given checkpoint is exactly reproducible (§5).

## 3. Vanilla LIBERO evaluation (follows the SmolVLA paper, Sec. 4.1/4.3)

| Setting | Value | Source |
|---|---|---|
| Suites and tasks | Spatial, Object, Goal, Long; 10 tasks each | paper |
| Trials | 10 per task (400 episodes) | paper ("10 trials per task") |
| Success | binary, LIBERO's success check | paper |
| Action execution | new policy call after every executed action (`n_action_steps=1`) | paper ("In simulation, we perform inference by sampling new observations and predicting a new action after each executed action") |
| Flow-matching steps | 10 | paper |
| Initial states | episode *k* of a task starts from LIBERO's stored initial state *k* (k = 0…9), seed `1000 + k` | standard LIBERO; choice of seed |
| Episode length | 280 / 280 / 300 / 520 steps (Spatial / Object / Goal / Long), 10 settle steps at reset | LeRobot defaults (not stated in the paper) |
| Rendering | 256×256, the training-image resolution | choice (LeRobot's default is 360×360) |
| Control | relative end-effector deltas, as in the dataset | dataset |

**Deviation to state: MuJoCo version.** The LIBERO demonstrations were rendered with MuJoCo ≤ 3.3.2; LeRobot's lock
file (used here) has 3.8.1. From 3.3.3 some scenes render differently (LIBERO issue #88), and from 3.4.0 the reset
of LIBERO-Spatial task 5 leaves the bowl tilted on the ramekin rim (LeRobot issue #4390). On an earlier checkpoint
we measured 73.5% (3.8.1) vs 77.8% (3.3.2) average success, mostly from Spatial task 5. Numbers from this pipeline
are therefore a few points below a 3.3.2 evaluation; comparisons within the pipeline are unaffected.

## 4. LIBERO-Plus evaluation (perturbations from the LIBERO-Plus paper, paired with §3)

The checkpoint is trained on standard LIBERO only; LIBERO-Plus is used for evaluation only.

| Setting | Value |
|---|---|
| Perturbations | all 7 LIBERO-Plus dimensions: Camera Viewpoints, Robot Initial States, Language Instructions, Light Conditions, Background Textures, Sensor Noise, Objects Layout |
| Episodes | for each perturbation, the same 40 tasks × up to 10 episodes; episode *k* of a task is one LIBERO-Plus variant of that task in that perturbation category |
| Pairing | that episode uses the **same stored initial state *k*, environment seed `1000 + k`, policy noise seed and instruction** as vanilla episode *k* (except where the perturbation itself changes them, below) |
| Variant choice | drawn at random per (task, perturbation) with a fixed seed (`--sample-seed 0`), each variant used once |
| Counts | camera 396, robot 400, light 376, background 385, noise 400, language 400, layout 400 = 2,757 episodes; 9 of 280 (task, perturbation) pairs have only 4–8 variants in LIBERO-Plus, and all of them are used |
| Policy settings | identical to §3 |
| Execution | one `lerobot-eval` process per (perturbation, suite), episodes one at a time; vanilla: one process per suite |

What stays identical to the vanilla episode:

| Perturbation | Identical | Changed |
|---|---|---|
| Camera, Light, Background, Noise | task, initial state, seed, instruction | the observation only |
| Robot Initial States | task, objects' initial state, seed, instruction | the robot's start pose |
| Language Instructions | task, initial state, seed | the instruction (LIBERO-Plus rewrite) |
| Objects Layout | task, seed, instruction | the scene (the variant's own stored layout) |

Three deliberate choices, implemented in `eval_combined/libero_plus_patch/sitecustomize.py` (loaded into every LIBERO-Plus process):
1. **Initial state.** LeRobot would start every one-episode LIBERO-Plus run from stored state 0; the patch uses
   state *k* so episode *k* is paired with vanilla episode *k*.
2. **Instruction.** LIBERO-Plus derives the instruction of non-language variants from the file name, so the policy
   would receive e.g. "… on the plate view 0 0 100 0 0 initstate 430". Those variants get the vanilla instruction,
   so only the intended factor changes (an open LeRobot PR, #4497, proposes the same fix).
3. **NumPy 2 compatibility.** LIBERO-Plus's fog noise uses `np.float_`, removed in NumPy 2; the patch restores it
   as `np.float64` (otherwise every fog variant crashes).

Differences from the LIBERO-Plus paper's own protocol (all 10,030 variants, 1 trial each, file-name instructions):
we use a fixed, paired subset of 2,757 variants and vanilla instructions, so our numbers measure the drop of each
perturbation relative to the same vanilla episodes and are not directly comparable to the paper's Table 2.

## 5. Reproducibility of evaluation

Same checkpoint, seed and software do not by themselves give identical episodes on a GPU: small floating-point
differences grow over a closed-loop rollout into different outcomes (two default LeRobot evaluations of the same
checkpoint disagreed on 19% of episodes). `eval_vanilla/deterministic_eval.py` runs the unmodified `lerobot-eval`
with three changes, and every episode runs with batch size 1:

| Source of variation | Handling |
|---|---|
| `cudnn.benchmark` (LeRobot default) picks GPU kernels by timing, so the choice depends on GPU load | kept off; cuDNN/cuBLAS deterministic algorithms on |
| batch size changes tensor shapes, hence kernels | batch size 1 for vanilla and LIBERO-Plus |
| the policy's noise is one random stream per process, so one episode's difference shifts all later ones | the policy is reseeded at the start of every episode with that episode's seed (`1000 + k`); a LIBERO-Plus episode uses the same seed, hence the same policy noise, as its vanilla twin |
| some GPU kernels depend on memory layout, i.e. on what ran earlier in the process | the scripts always run the same work in the same processes and order |
| GPU (EGL) rendering of the camera image can rarely differ for the same simulator state | not controllable; observed once in ~250 episodes tested, and per-episode reseeding confines it to that episode |

Verified on this machine (LIBERO-Spatial):

| Comparison | Identical episode outcomes |
|---|---|
| Deterministic, two runs at the same time (batch 10) | 50 / 50 |
| Deterministic, GPU shared with 11 evaluations vs GPU alone | 50 / 50 |
| Batch size 1, two runs | 50 / 50 |
| Same command twice, with per-episode reseeding | 10 / 10 (videos pixel-identical) |
| One task alone vs after other tasks, with per-episode reseeding | first episode pixel-identical |
| LeRobot default vs deterministic kernels | 46 / 50 (kernel choice alone changes 8% of outcomes) |
| Batch size 1 vs batch size 10 | 45 / 50 |

So rerunning the same commands with the same checkpoint, software (§1) and GPU model reproduces the evaluation,
up to rare rendering differences that affect single episodes. A different GPU model may give slightly different
outcomes.

## 6. Metrics

| Metric | Definition |
|---|---|
| Success rate | fraction of successful episodes (LIBERO's binary success check) |
| Vanilla success | success rate without perturbation on the twin episodes of the perturbed ones (same task, initial state, seed) |
| OOD success | success rate with the LIBERO-Plus perturbation |
| Success gap Δ | OOD success − vanilla success, in percentage points |

One training seed; between independently trained checkpoints we observed a few points of difference (e.g. 73.5% vs
71.5% vanilla average), which this protocol does not average out.

## 7. Reproduce

```bash
bash projects/smolvla_libero/setup/setup_envs.sh
RUN=smolvla_libero_seed1000 nohup bash projects/smolvla_libero/train/train.sh > train.log 2>&1 &
python3 projects/smolvla_libero/eval_vanilla/eval_vanilla.py \
  --policy outputs/$RUN/checkpoints/100000/pretrained_model --out outputs/eval/${RUN}_vanilla     # vanilla only
python3 projects/smolvla_libero/eval_combined/eval_combined.py \
  --policy outputs/$RUN/checkpoints/100000/pretrained_model --out outputs/eval/${RUN}_robustness
```

## 8. Upstream issues encountered

| Issue | Where | Handled by |
|---|---|---|
| LeRobot LIBERO-Plus docs claim 10 tasks/suite and 400 episodes; the fork has 10,030 tasks | LeRobot issue #4834 | protocol in §4 |
| File-name-derived instructions with perturbation tags | LeRobot PR #4497 (open) | `sitecustomize.py` |
| `np.float_` in LIBERO-Plus fog noise breaks under NumPy 2 | LIBERO-Plus | `sitecustomize.py` |
| `--env.max_parallel_tasks > 1` shares one stateful policy across threads | LeRobot issue #4327, PR #4450 (open) | never used; separate processes instead |
| MuJoCo ≥ 3.4.0 breaks LIBERO-Spatial task 5's reset | LeRobot issue #4390 | documented (§3) |
| `cudnn.benchmark` makes evaluation load-dependent | LeRobot `lerobot_eval.py` | `eval_vanilla/deterministic_eval.py` |
