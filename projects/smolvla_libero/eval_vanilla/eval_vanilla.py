# Vanilla LIBERO evaluation, SmolVLA paper protocol (Shukor et al., 2025, Sec. 4).
# 4 suites x 10 tasks x 10 episodes; episode k starts from LIBERO's stored initial state k with seed `seed + k`.
# Also the shared code of eval_combined.py (protocol, launcher, success rates).
#
# Usage:  python3 projects/smolvla_libero/eval_vanilla/eval_vanilla.py --policy <ckpt>/pretrained_model --out <dir>
# Output: summary.md/.json, run_config.json, vanilla/<suite>/, logs/  (re-running the same --out resumes)

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------- paths

VANILLA_DIR = Path(__file__).resolve().parent
PROJECT_DIR = VANILLA_DIR.parent                     # projects/smolvla_libero
REPO = PROJECT_DIR.parents[1]
THIRD_PARTY = REPO / "third_party"
VANILLA_VENV = REPO / ".venv"
VANILLA_CONFIG = THIRD_PARTY / "libero_config"
DETERMINISTIC_EVAL = VANILLA_DIR / "deterministic_eval.py"

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
SUITE_NAMES = {"libero_spatial": "Spatial", "libero_object": "Object", "libero_goal": "Goal", "libero_10": "Long"}

# ---------------------------------------------------------------- evaluation settings (SmolVLA paper protocol)

POLICY_FLAGS = [
    "--policy.n_action_steps=1",       # new policy call after every executed action
    "--policy.compile_model=false",
]
ENV_FLAGS = [
    "--env.observation_height=256",    # training-image resolution
    "--env.observation_width=256",
    "--env.max_parallel_tasks=1",      # >1 shares one stateful policy across threads
]
BATCH_FLAG = "--eval.batch_size=1"     # same tensor shapes (hence GPU kernels) for vanilla and LIBERO-Plus


def add_common_args(parser) -> dict:
    # Arguments shared with eval_combined.py; returns the groups so it can add to them
    run = parser.add_argument_group("run")
    run.add_argument("--policy", required=True, help="checkpoint dir or hub id")
    run.add_argument("--out", required=True, type=Path, help="output dir (re-running resumes)")

    protocol = parser.add_argument_group("protocol")
    protocol.add_argument("--suites", default=",".join(SUITES), help="comma-separated subset of the 4 suites")
    protocol.add_argument("--episodes", type=int, default=10, help="episodes per task (paper: 10)")
    protocol.add_argument("--seed", type=int, default=1000, help="episode k uses seed + k")

    execution = parser.add_argument_group("execution")
    execution.add_argument("--workers", type=int, default=11, help="concurrent lerobot-eval processes")
    execution.add_argument("--dry-run", action="store_true", help="print the plan only")

    envs = parser.add_argument_group("environments")
    envs.add_argument("--vanilla-venv", type=Path, default=VANILLA_VENV)
    envs.add_argument("--vanilla-config", type=Path, default=VANILLA_CONFIG)
    return {"run": run, "protocol": protocol, "execution": execution, "environments": envs}


# ---------------------------------------------------------------- environments

def base_env() -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "LIBERO_CONFIG_PATH")}
    env["MUJOCO_GL"] = "egl"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # deterministic cuBLAS
    return env


def vanilla_env(config: Path) -> dict:
    env = base_env()
    env["LIBERO_CONFIG_PATH"] = str(config)
    return env


def py(venv: Path, code: str, env: dict) -> str:
    # Run code in a venv; return the last stdout line (LIBERO prints on import)
    out = subprocess.run([str(venv / "bin/python"), "-c", code], env=env, check=True,
                         capture_output=True, text=True).stdout
    return out.strip().splitlines()[-1]


def versions(venv: Path, env: dict) -> dict:
    # Package versions that must match between environments, and where `libero` comes from
    code = ("import importlib.metadata as m, json, libero; "
            "print(json.dumps({**{p: m.version(p) for p in ['mujoco', 'robosuite', 'torch', 'lerobot']}, "
            "'libero_path': list(libero.__path__)[0]}))")
    return json.loads(py(venv, code, env))


def vanilla_tasks(venv: Path, config: Path, suites: list[str]) -> dict:
    # {suite: [[task name, instruction], ...]} in suite order
    code = (
        "import json, io, contextlib\n"
        "with contextlib.redirect_stdout(io.StringIO()):\n"
        "    from libero.libero import benchmark\n"
        "    b = benchmark.get_benchmark_dict()\n"
        f"    s = {{n: b[n]() for n in {suites!r}}}\n"
        "print(json.dumps({n: [[t.name, t.language] for t in v.tasks] for n, v in s.items()}))"
    )
    return json.loads(py(venv, code, vanilla_env(config)))


# ---------------------------------------------------------------- launching evaluations

def lerobot_eval(venv: Path, policy: str, seed: int) -> list[str]:
    # Start of every evaluation command: lerobot-eval via deterministic_eval.py + the protocol settings
    return [str(venv / "bin/python"), str(DETERMINISTIC_EVAL),
            f"--policy.path={policy}", *POLICY_FLAGS, *ENV_FLAGS, f"--seed={seed}"]


def vanilla_job(args, suite: str) -> tuple[list[str], dict, Path]:
    # One suite: 10 tasks x N episodes, one at a time -> (command, env vars, output dir)
    out = args.out / "vanilla" / suite
    cmd = [*lerobot_eval(args.vanilla_venv, args.policy, args.seed),
           "--env.type=libero", f"--env.task={suite}",
           BATCH_FLAG, f"--eval.n_episodes={args.episodes}",
           f"--output_dir={out}"]
    return cmd, vanilla_env(args.vanilla_config), out


def run_eval(cmd: list[str], env: dict, out_dir: Path, log: Path) -> bool:
    # One lerobot-eval process; skipped if its results exist (resume)
    if (out_dir / "eval_info.json").exists():
        return True
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL).returncode
    ok = rc == 0 and (out_dir / "eval_info.json").exists()
    print(f"  {'done' if ok else 'FAILED'}: {out_dir.relative_to(out_dir.parents[1])} (log: {log})", flush=True)
    return ok


# ---------------------------------------------------------------- results

def load_vanilla(out: Path) -> dict:
    # {(suite, task id, episode k): success}
    results = {}
    for info_path in sorted((out / "vanilla").glob("*/eval_info.json")):
        for t in json.loads(info_path.read_text())["per_task"]:
            for k, s in enumerate(t["metrics"]["successes"]):
                results[(t["task_group"], t["task_id"], k)] = bool(s)
    return results


def rate(xs: list[bool]) -> dict:
    k, n = sum(xs), len(xs)
    return {"pc_success": round(100 * k / n, 1) if n else None, "n_success": k, "n": n}


def cell(r: dict | None) -> str:
    return "–" if not r or not r["n"] else f"{r['pc_success']:.1f}"


# ---------------------------------------------------------------- vanilla evaluation

def summarize(args, suites, tasks) -> dict:
    results = load_vanilla(args.out)

    def r(suite=None, tid=None):
        return rate([s for (sg, t, _), s in results.items()
                     if (suite is None or sg == suite) and (tid is None or t == tid)])

    summary = {"by_suite": {s: r(s) for s in suites}, "overall": r(),
               "by_task": {s: {tid: r(s, tid) for tid in range(10)} for s in suites}}
    lines = ["# Vanilla LIBERO (SmolVLA paper protocol)", "", f"Policy: `{args.policy}`  ",
             f"{len(suites)} suites × 10 tasks × {args.episodes} episodes; episode k uses LIBERO's stored initial state "
             f"k and seed {args.seed}+k.", "",
             "| Suite | Success % | Episodes |", "|---|---|---|"]
    for s in suites:
        x = summary["by_suite"][s]
        lines.append(f"| {SUITE_NAMES.get(s, s)} | {cell(x)} | {x['n']} |")
    o = summary["overall"]
    lines += [f"| **Average** | **{cell(o)}** | {o['n']} |", "", "## By task", ""]
    for s in suites:
        lines += [f"### {SUITE_NAMES.get(s, s)}", "", "| # | Instruction | Success % |", "|---|---|---|"]
        lines += [f"| {tid} | {instr} | {cell(summary['by_task'][s][tid])} |"
                  for tid, (_, instr) in enumerate(tasks[s])]
        lines.append("")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Vanilla LIBERO evaluation (SmolVLA paper protocol)")
    add_common_args(parser)
    args = parser.parse_args()
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    args.out.mkdir(parents=True, exist_ok=True)

    env_versions = versions(args.vanilla_venv, vanilla_env(args.vanilla_config))
    tasks = vanilla_tasks(args.vanilla_venv, args.vanilla_config, suites)
    jobs = [vanilla_job(args, s) for s in suites]
    (args.out / "run_config.json").write_text(json.dumps({
        "args": {k: str(v) for k, v in vars(args).items()}, "versions": env_versions,
        "commands": [cmd for cmd, _, _ in jobs]}, indent=2))
    print(f"Vanilla LIBERO: {len(suites)} suites x 10 tasks x {args.episodes} episodes, one process per suite")
    print(f"Versions: {env_versions}")
    if args.dry_run:
        print("\nfirst command:", " ".join(jobs[0][0]))
        return

    t0, ok = time.time(), True
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_eval, cmd, env, out, args.out / "logs" / f"vanilla_{out.name}.log")
                   for cmd, env, out in jobs]
        for f in as_completed(futures):
            ok &= f.result()
    o = summarize(args, suites, tasks)["overall"]
    print(f"\nFinished in {(time.time() - t0) / 60:.0f} min. Vanilla {cell(o)}%. Tables: {args.out / 'summary.md'}")
    if not ok:
        sys.exit("some suites failed; see logs/ (re-run the same command to retry them)")


if __name__ == "__main__":
    main()
