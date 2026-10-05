# Evaluate one checkpoint on vanilla LIBERO and LIBERO-Plus, paired episode by episode.
#   vanilla      .venv             SmolVLA paper protocol (eval_vanilla.vanilla_job)
#   LIBERO-Plus  .venv-liberoplus  same tasks; episode k = one LIBERO-Plus variant of the task, run with vanilla
#                                  episode k's initial state, seed and instruction (where the perturbation allows)
# Each variant is used once. Metrics: success rate and Δ = OOD − vanilla success on the same episodes.
#
# Usage:  python3 projects/smolvla_libero/eval_combined/eval_combined.py --policy <ckpt>/pretrained_model --out <dir>
# Output: summary.md/.json, plus_episodes.jsonl, run_config.json, vanilla/<suite>/, plus/<pert>/<suite>/, logs/
#         (re-running the same --out resumes)

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval_vanilla"))
import eval_vanilla as ev  # noqa: E402  (vanilla protocol, launcher, success rates)

PATCH_DIR = Path(__file__).resolve().parent / "libero_plus_patch"
MAGICK_PRELOAD = ["libjpeg.so.8", "libglib-2.0.so.0", "libgobject-2.0.so.0"]  # same as setup/paths.sh

# short name -> LIBERO-Plus category (task_classification.json)
PERTURBATIONS = {
    "camera": "Camera Viewpoints",
    "robot": "Robot Initial States",
    "light": "Light Conditions",
    "background": "Background Textures",
    "noise": "Sensor Noise",
    "language": "Language Instructions",
    "layout": "Objects Layout",
}
# what stays identical to the vanilla episode
IDENTICAL = {
    "camera": "task, initial state, seed, instruction; only the camera view changes",
    "robot": "task, objects' initial state, seed, instruction; the robot's start pose changes",
    "light": "task, initial state, seed, instruction; only the lighting changes",
    "background": "task, initial state, seed, instruction; only scene textures change",
    "noise": "task, initial state, seed, instruction; only the image noise changes",
    "language": "task, initial state, seed; the instruction is reworded",
    "layout": "task, seed, instruction; the scene layout changes (own initial state)",
}


def parse_args():
    p = argparse.ArgumentParser(description="Vanilla LIBERO + LIBERO-Plus evaluation, paired episodes")
    groups = ev.add_common_args(p)

    plus = p.add_argument_group("LIBERO-Plus")
    plus.add_argument("--perturbations", default=",".join(PERTURBATIONS), help="comma-separated subset (default: all 7)")
    plus.add_argument("--sample-seed", type=int, default=0, help="fixes which variants are used")
    plus.add_argument("--instructions", choices=["vanilla", "official"], default="vanilla",
                      help="vanilla: non-language variants keep the vanilla instruction; "
                           "official: LIBERO-Plus's file-name-derived text")

    envs = groups["environments"]
    envs.add_argument("--plus-venv", type=Path, default=ev.REPO / ".venv-liberoplus")
    envs.add_argument("--plus-root", type=Path, default=ev.THIRD_PARTY / "LIBERO-plus")
    envs.add_argument("--plus-config", type=Path, default=ev.THIRD_PARTY / "libero_plus_config")
    envs.add_argument("--magick-home", type=Path, default=ev.THIRD_PARTY / "imagemagick")
    return p.parse_args()


def plus_env(args, episodes_file: Path | None) -> dict:
    # LIBERO-Plus process: fork on PYTHONPATH, its config, ImageMagick, per-episode patch
    env = ev.base_env()
    paths = [str(args.plus_root)]
    if episodes_file is not None:
        paths.insert(0, str(PATCH_DIR))
        env["LIBERO_PLUS_EPISODES"] = str(episodes_file)
    env["PYTHONPATH"] = ":".join(paths)
    env["LIBERO_CONFIG_PATH"] = str(args.plus_config)
    env["MAGICK_HOME"] = str(args.magick_home)
    env["LD_PRELOAD"] = ":".join(str(args.magick_home / "lib" / lib) for lib in MAGICK_PRELOAD)  # newer than system
    return env


# ---------------------------------------------------------------- which LIBERO-Plus variant for which episode

def build_manifest(args, suites, perturbations, tasks) -> list[dict]:
    # One entry per LIBERO-Plus episode: (suite, task, perturbation, slot k) -> a distinct variant
    classification = json.loads((args.plus_root / "libero/libero/benchmark/task_classification.json").read_text())
    manifest = []
    for suite in suites:
        names = [n for n, _ in tasks[suite]]
        pools = defaultdict(list)  # (vanilla task id, category) -> variants
        for item in classification[suite]:
            matches = [i for i, n in enumerate(names) if item["name"] == n or item["name"].startswith(n + "_")]
            if not matches:
                raise RuntimeError(f"{suite}: variant {item['name']} matches no vanilla task")
            pools[(max(matches, key=lambda i: len(names[i])), item["category"])].append(item)  # longest prefix

        for tid in range(len(names)):
            for pert in perturbations:
                pool = pools[(tid, PERTURBATIONS[pert])]
                if not pool:
                    raise RuntimeError(f"{suite} task {tid} has no {PERTURBATIONS[pert]} variant")
                order = random.Random(f"{args.sample_seed}-{suite}-task{tid}-{pert}").sample(pool, len(pool))
                for k, item in enumerate(order[:args.episodes]):  # each variant once; fewer if the pool is small
                    manifest.append({
                        "suite": suite, "vanilla_task_id": tid, "vanilla_instruction": tasks[suite][tid][1],
                        "perturbation": pert, "slot": k,
                        "task_index": item["id"] - 1,  # LIBERO-Plus ids are 1-based
                        "name": item["name"], "level": item["difficulty_level"],
                    })
    return manifest


def plus_jobs(args, manifest, perturbations) -> list[dict]:
    # One process per (perturbation, suite), in perturbation order; writes its episodes file for the patch
    jobs = []
    for pert in perturbations:
        for suite in dict.fromkeys(e["suite"] for e in manifest):
            entries = [e for e in manifest if e["perturbation"] == pert and e["suite"] == suite]
            ids = sorted(e["task_index"] for e in entries)
            assert len(ids) == len(set(ids)), f"{pert}/{suite}: a variant appears twice"
            out = args.out / "plus" / pert / suite
            episodes_file = out.parent / f"{suite}.episodes.json"
            episodes_file.parent.mkdir(parents=True, exist_ok=True)
            episodes_file.write_text(json.dumps({
                e["name"]: {"slot": e["slot"],                     # initial state k
                            "seed": args.seed + e["slot"],          # env + policy seed of vanilla episode k
                            "instruction": e["vanilla_instruction"]
                            if args.instructions == "vanilla" and pert != "language" else None}
                for e in entries}, indent=1))
            cmd = [*ev.lerobot_eval(args.plus_venv, args.policy, args.seed),  # seed overridden per episode
                   "--env.type=libero_plus", f"--env.task={suite}",
                   f"--env.task_ids=[{','.join(map(str, ids))}]",
                   ev.BATCH_FLAG, "--eval.n_episodes=1",               # one episode per variant
                   f"--output_dir={out}"]
            jobs.append({"perturbation": pert, "suite": suite, "cmd": cmd, "env": plus_env(args, episodes_file),
                         "out": out})
    return jobs


# ---------------------------------------------------------------- results

def short_pairs(args, manifest) -> dict:
    # (suite, task, perturbation) pairs with fewer than --episodes variants
    counts = defaultdict(int)
    for e in manifest:
        counts[(e["suite"], e["vanilla_task_id"], e["perturbation"])] += 1
    return {k: n for k, n in counts.items() if n < args.episodes}


def summarize(args, suites, perturbations, manifest, tasks) -> dict:
    vanilla = ev.load_vanilla(args.out)            # (suite, task, k) -> success
    shifted = {}                                   # (perturbation, suite, task_index) -> success
    for f in sorted((args.out / "plus").glob("*/*/eval_info.json")):
        for t in json.loads(f.read_text())["per_task"]:
            shifted[(f.parent.parent.name, t["task_group"], t["task_id"])] = bool(t["metrics"]["successes"][0])
    for e in manifest:
        e["success"] = shifted.get((e["perturbation"], e["suite"], e["task_index"]))
        e["vanilla_success"] = vanilla.get((e["suite"], e["vanilla_task_id"], e["slot"]))

    def van(suite=None, tid=None):
        return ev.rate([s for (sg, t, _), s in vanilla.items()
                        if (suite is None or sg == suite) and (tid is None or t == tid)])

    def shift(pert, suite=None, tid=None, level=None):
        es = [e for e in manifest if e["perturbation"] == pert and e["success"] is not None
              and (suite is None or e["suite"] == suite) and (tid is None or e["vanilla_task_id"] == tid)
              and (level is None or e["level"] == level)]
        r = ev.rate([e["success"] for e in es])
        paired = [(e["vanilla_success"], e["success"]) for e in es if e["vanilla_success"] is not None]
        n = len(paired)
        r["paired"] = {
            "vanilla_pc": round(100 * sum(v for v, _ in paired) / n, 1) if n else None,  # vanilla, same episodes
            "delta": round(100 * sum(s - v for v, s in paired) / n, 1) if n else None,   # OOD − vanilla
        }
        return r

    summary = {
        "by_suite": {s: {"vanilla": van(s), **{p: shift(p, suite=s) for p in perturbations}} for s in suites},
        "overall": {"vanilla": van(), **{p: shift(p) for p in perturbations}},
        "by_task": {s: {tid: {"vanilla": van(s, tid), **{p: shift(p, suite=s, tid=tid) for p in perturbations}}
                        for tid in range(10)} for s in suites},
        "by_level": {p: {str(lv): shift(p, level=lv) for lv in range(1, 6)} for p in perturbations},
        "n_shifted_done": sum(e["success"] is not None for e in manifest), "n_shifted": len(manifest),
    }

    def delta(row, p):
        d = row[p]["paired"]["delta"]
        return "–" if d is None else f"{d:+.1f}"

    heads = [PERTURBATIONS[p] for p in perturbations]
    o = summary["overall"]
    lines = ["# Vanilla LIBERO vs LIBERO-Plus perturbations", "",
             f"Policy: `{args.policy}`  ",
             f"Vanilla: {len(suites)} suites × 10 tasks × {args.episodes} episodes. Each perturbation: the same tasks "
             f"with up to {args.episodes} episodes (one per distinct LIBERO-Plus variant). Episode k of a task uses "
             f"LIBERO's stored initial state k and seed {args.seed}+k in both; instructions: {args.instructions}. "
             f"LIBERO-Plus episodes done: {summary['n_shifted_done']}/{summary['n_shifted']}.", "",
             "## Success rate (%) by suite", "",
             "| Suite | Vanilla | " + " | ".join(heads) + " |", "|---" * (2 + len(heads)) + "|"]
    for s in suites:
        r = summary["by_suite"][s]
        lines.append(f"| {ev.SUITE_NAMES.get(s, s)} | {ev.cell(r['vanilla'])} | "
                     + " | ".join(f"{ev.cell(r[p])} ({delta(r, p)})" for p in perturbations) + " |")
    lines.append(f"| **Average** | **{ev.cell(o['vanilla'])}** | "
                 + " | ".join(f"**{ev.cell(o[p])}** ({delta(o, p)})" for p in perturbations) + " |")
    lines += ["", "Cells: success % (change vs vanilla on the same episodes, points).", "",
              "## Paired comparison with vanilla (all suites)", "",
              "| | Episodes | Success % | Vanilla on same episodes | Δ | Identical to vanilla episode |",
              "|---|---|---|---|---|---|",
              f"| Vanilla | {o['vanilla']['n']} | {ev.cell(o['vanilla'])} | | | |"]
    for p in perturbations:
        r, pv = o[p], o[p]["paired"]
        vp = "–" if pv["vanilla_pc"] is None else f"{pv['vanilla_pc']:.1f}"
        lines.append(f"| {PERTURBATIONS[p]} | {r['n']} | {ev.cell(r)} | {vp} | {delta(o, p)} | {IDENTICAL[p]} |")
    lines += ["", "Δ = success with the perturbation − vanilla success on the same episodes (points).", "",
              "## By suite and original task", ""]
    for s in suites:
        lines += [f"### {ev.SUITE_NAMES.get(s, s)}", "", "| # | Instruction | Vanilla | " + " | ".join(heads) + " |",
                  "|---|---|---" + "|---" * len(heads) + "|"]
        for tid, (_, instr) in enumerate(tasks[s]):
            r = summary["by_task"][s][tid]
            lines.append(f"| {tid} | {instr} | {ev.cell(r['vanilla'])} | "
                         + " | ".join(ev.cell(r[p]) for p in perturbations) + " |")
        lines.append("")
    lines += ["## By LIBERO-Plus difficulty level (all suites)", "",
              "Levels are LIBERO-Plus's labels (how many of 4 reference models solved the variant).", "",
              "| Perturbation | L1 | L2 | L3 | L4 | L5 |", "|---|---|---|---|---|---|"]
    for p in perturbations:
        lines.append(f"| {PERTURBATIONS[p]} | " + " | ".join(
            f"{ev.cell(summary['by_level'][p][str(lv)])} ({summary['by_level'][p][str(lv)]['n']})"
            for lv in range(1, 6)) + " |")
    short = short_pairs(args, manifest)
    if short:
        lines += ["", f"Tasks with fewer than {args.episodes} LIBERO-Plus variants of a perturbation (all available "
                      "variants used, each once):", "", "| Suite | Task | Perturbation | Episodes |", "|---|---|---|---|"]
        lines += [f"| {ev.SUITE_NAMES.get(s, s)} | {tid} | {PERTURBATIONS[p]} | {n} |" for (s, tid, p), n in short.items()]

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    with open(args.out / "plus_episodes.jsonl", "w") as f:
        for e in manifest:
            f.write(json.dumps(e) + "\n")
    return summary


# ---------------------------------------------------------------- main

def main():
    args = parse_args()
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    perturbations = [p.strip() for p in args.perturbations.split(",") if p.strip()]
    unknown = set(perturbations) - set(PERTURBATIONS)
    if unknown:
        sys.exit(f"unknown perturbations: {sorted(unknown)}; choose from {list(PERTURBATIONS)}")
    args.out.mkdir(parents=True, exist_ok=True)

    # 1. Both environments must match except for the libero package
    vanilla_v = ev.versions(args.vanilla_venv, ev.vanilla_env(args.vanilla_config))
    plus_v = ev.versions(args.plus_venv, plus_env(args, None))
    for key in ("mujoco", "robosuite", "torch", "lerobot"):
        if vanilla_v[key] != plus_v[key]:
            sys.exit(f"{key} differs between environments: vanilla {vanilla_v[key]} vs LIBERO-Plus {plus_v[key]}")
    if "LIBERO-plus" not in plus_v["libero_path"]:
        sys.exit(f"{args.plus_venv} imports libero from {plus_v['libero_path']}, not the LIBERO-Plus fork")

    # 2. Vanilla jobs + the LIBERO-Plus variant of every (suite, task, perturbation, episode)
    tasks = ev.vanilla_tasks(args.vanilla_venv, args.vanilla_config, suites)
    vanilla_jobs = [ev.vanilla_job(args, s) for s in suites]
    manifest = build_manifest(args, suites, perturbations, tasks)
    jobs = plus_jobs(args, manifest, perturbations)

    counts = {p: sum(e["perturbation"] == p for e in manifest) for p in perturbations}
    (args.out / "run_config.json").write_text(json.dumps({
        "args": {k: str(v) for k, v in vars(args).items()},
        "versions": {"vanilla": vanilla_v, "plus": plus_v},
        "vanilla_commands": [cmd for cmd, _, _ in vanilla_jobs],
        "plus_command_example": jobs[0]["cmd"] if jobs else None,
        "episodes": {"vanilla": len(suites) * 10 * args.episodes, "shifted": len(manifest), "per_perturbation": counts,
                     "short_pairs": {f"{s}/task{t}/{p}": n for (s, t, p), n in short_pairs(args, manifest).items()}},
    }, indent=2))
    print(f"Vanilla: {len(suites)} suites x 10 tasks x {args.episodes} episodes = {len(suites) * 10 * args.episodes}")
    print(f"LIBERO-Plus: {len(manifest)} episodes (" + ", ".join(f"{p} {n}" for p, n in counts.items()) + ")")
    print(f"Processes: {len(vanilla_jobs)} vanilla + {len(jobs)} LIBERO-Plus, {args.workers} at a time; "
          f"instructions={args.instructions}")
    print(f"Versions: {vanilla_v} | LIBERO-Plus from {plus_v['libero_path']}")
    if args.dry_run:
        print("\nvanilla (first suite):", " ".join(vanilla_jobs[0][0]))
        if jobs:
            print("LIBERO-Plus (first process):", " ".join(jobs[0]["cmd"]))
        return

    # 3. Run (finished processes are skipped); summary refreshed after each
    t0, ok = time.time(), True
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(ev.run_eval, cmd, env, out, args.out / "logs" / f"vanilla_{out.name}.log")
                   for cmd, env, out in vanilla_jobs]
        futures += [pool.submit(ev.run_eval, j["cmd"], j["env"], j["out"],
                                args.out / "logs" / f"plus_{j['perturbation']}_{j['suite']}.log") for j in jobs]
        for f in as_completed(futures):
            ok &= f.result()
            summarize(args, suites, perturbations, manifest, tasks)

    o = summarize(args, suites, perturbations, manifest, tasks)["overall"]
    print(f"\nFinished in {(time.time() - t0) / 60:.0f} min. Vanilla {ev.cell(o['vanilla'])}% | "
          + " | ".join(f"{p} {ev.cell(o[p])}%" for p in perturbations) + f". Tables: {args.out / 'summary.md'}")
    if not ok:
        sys.exit("some evaluations failed; see logs/ (re-run the same command to retry them)")


if __name__ == "__main__":
    main()
