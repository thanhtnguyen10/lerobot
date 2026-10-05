# Robustness-gap table, plots and task ranking from an eval_combined.py run (reads <run>/summary.json).
# Writes to <run>/gap/:
#   gap_table.{md,csv,tex}     Vanilla / OOD / Δ per perturbation (Vanilla = no perturbation, same episodes)
#   gap_by_suite.{md,csv}      the same per suite
#   robustness_gap.{png,pdf}   per-perturbation plot
#   gap_heatmap.{png,pdf}      Δ by suite x perturbation
#   task_ranking.{md,csv}      candidate tasks for rapid experiments
# Usage: .venv/bin/python projects/smolvla_libero/analysis/robustness_gap.py --run outputs/eval/robustness

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

PERTURBATIONS = {"camera": "Camera Viewpoints", "robot": "Robot Initial States", "light": "Light Conditions",
                 "background": "Background Textures", "noise": "Sensor Noise", "language": "Language Instructions",
                 "layout": "Objects Layout"}
SUITE_NAMES = {"libero_spatial": "Spatial", "libero_object": "Object", "libero_goal": "Goal", "libero_10": "Long"}
KNOWN_ISSUES = {("libero_spatial", 5): "reset broken under MuJoCo >= 3.4 (LeRobot #4390)"}

# Colors: blue ramp (light = vanilla, dark = OOD), neutral ink and grid
VANILLA, OOD = "#86b6ef", "#1c5cab"
INK, INK_2, CONNECTOR, GRID, SURFACE = "#0b0b0b", "#52514e", "#cfcec8", "#e6e5e1", "#fcfcfb"
HEAT = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b"]


def gap_rows(scope: dict) -> list[dict]:
    # Vanilla / OOD / Δ per perturbation from one summary.json block
    rows = []
    for key, name in PERTURBATIONS.items():
        r, p = scope[key], scope[key]["paired"]
        rows.append({"perturbation": name, "key": key, "episodes": r["n"], "vanilla": p["vanilla_pc"],
                     "ood": r["pc_success"], "delta": p["delta"]})
    return rows


def write_tables(out: Path, overall: list[dict], by_suite: dict[str, list[dict]], vanilla: dict):
    n = sum(r["episodes"] for r in overall)
    ood_all = sum(r["ood"] * r["episodes"] for r in overall) / n
    vanilla_all = sum(r["vanilla"] * r["episodes"] for r in overall) / n
    ordered = sorted(overall, key=lambda r: r["delta"])

    lines = ["# Robustness gap: vanilla LIBERO vs LIBERO-Plus", "",
             "Vanilla = success without perturbation on the same episodes (task, initial state, seed) as the perturbed ones; "
             "OOD = success with the perturbation; Δ = OOD − Vanilla (points).", "",
             "| Perturbation | Episodes | Vanilla Success (%) | OOD Success (%) | Δ Success |",
             "|---|---|---|---|---|"]
    lines += [f"| {r['perturbation']} | {r['episodes']} | {r['vanilla']:.1f} | {r['ood']:.1f} | **{r['delta']:+.1f}** |"
              for r in ordered]
    lines += [f"| **All 7** | {n} | {vanilla_all:.1f} | {ood_all:.1f} | **{ood_all - vanilla_all:+.1f}** |", "",
              f"Vanilla LIBERO (all 400 episodes): {vanilla['pc_success']:.1f}%. The paired vanilla rate differs slightly for "
              "perturbations with fewer than 400 episodes (only the paired episodes count)."]
    (out / "gap_table.md").write_text("\n".join(lines) + "\n")

    with open(out / "gap_table.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["perturbation", "episodes", "vanilla_success", "ood_success", "delta_success"])
        for r in overall:
            w.writerow([r["perturbation"], r["episodes"], r["vanilla"], r["ood"], r["delta"]])

    tex = [r"\begin{tabular}{lrrrr}", r"\toprule",
           r"Perturbation & Episodes & Vanilla (\%) & OOD (\%) & $\Delta$ \\", r"\midrule"]
    tex += [f"{r['perturbation']} & {r['episodes']} & {r['vanilla']:.1f} & {r['ood']:.1f} & {r['delta']:+.1f} \\\\"
            for r in ordered]
    tex += [r"\midrule", f"All 7 & {n} & {vanilla_all:.1f} & {ood_all:.1f} & {ood_all - vanilla_all:+.1f} \\\\",
            r"\bottomrule", r"\end{tabular}"]
    (out / "gap_table.tex").write_text("\n".join(tex) + "\n")

    suites = list(by_suite)
    lines = ["# Δ Success by suite (points; Vanilla → OOD in brackets)", "",
             "| Perturbation | " + " | ".join(SUITE_NAMES[s] for s in suites) + " |", "|---|" + "---|" * len(suites)]
    for key, name in PERTURBATIONS.items():
        cells = [next(r for r in by_suite[s] if r["key"] == key) for s in suites]
        lines.append(f"| {name} | " + " | ".join(f"{r['delta']:+.1f} ({r['vanilla']:.0f} → {r['ood']:.0f})"
                                                for r in cells) + " |")
    (out / "gap_by_suite.md").write_text("\n".join(lines) + "\n")
    with open(out / "gap_by_suite.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suite", "perturbation", "episodes", "vanilla_success", "ood_success", "delta_success"])
        for s in suites:
            for r in by_suite[s]:
                w.writerow([SUITE_NAMES[s], r["perturbation"], r["episodes"], r["vanilla"], r["ood"], r["delta"]])


def style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0, labelsize=9)


def plot_gap(out: Path, overall: list[dict]):
    # Dumbbell: vanilla (open) -> OOD (filled) per perturbation, largest drop at the bottom
    rows = sorted(overall, key=lambda r: r["delta"], reverse=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for i, r in enumerate(rows):
        ax.plot([r["ood"], r["vanilla"]], [i, i], color=CONNECTOR, linewidth=2, solid_capstyle="round", zorder=1)
        ax.scatter(r["vanilla"], i, s=60, facecolor=SURFACE, edgecolor=VANILLA, linewidth=2, zorder=3)
        ax.scatter(r["ood"], i, s=60, facecolor=OOD, edgecolor=SURFACE, linewidth=1.5, zorder=4)
        ax.text(102, i, f"{r['delta']:+.1f}", va="center", ha="left", fontsize=9, color=INK)
    ax.set_yticks(range(len(rows)), [r["perturbation"] for r in rows], color=INK)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Success rate (%)", color=INK_2, fontsize=9)
    ax.text(102, len(rows) - 0.35, "Δ", ha="left", va="bottom", fontsize=9, color=INK_2)
    ax.scatter([], [], s=60, facecolor=SURFACE, edgecolor=VANILLA, linewidth=2, label="Vanilla (same episodes)")
    ax.scatter([], [], s=60, facecolor=OOD, edgecolor=SURFACE, label="OOD (LIBERO-Plus)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False, fontsize=8.5,
              labelcolor=INK_2, handletextpad=0.3, columnspacing=1.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"robustness_gap.{ext}", facecolor=SURFACE)
    plt.close(fig)


def plot_heatmap(out: Path, by_suite: dict[str, list[dict]]):
    # Δ by suite x perturbation; darker = larger drop
    suites, keys = list(by_suite), list(PERTURBATIONS)
    drop = [[-next(r for r in by_suite[s] if r["key"] == k)["delta"] for k in keys] for s in suites]
    fig, ax = plt.subplots(figsize=(7.2, 2.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    ax.spines["bottom"].set_visible(False)
    im = ax.imshow(drop, cmap=LinearSegmentedColormap.from_list("blue", HEAT), vmin=0, vmax=80, aspect="auto")
    for i in range(len(suites)):
        for j in range(len(keys)):
            v = drop[i][j]
            ax.text(j, i, f"{-v:+.0f}", ha="center", va="center", fontsize=9, color=SURFACE if v > 40 else INK)
    ax.set_xticks(range(len(keys)), [PERTURBATIONS[k].replace(" ", "\n", 1) for k in keys], fontsize=8, color=INK)
    ax.set_yticks(range(len(suites)), [SUITE_NAMES[s] for s in suites], color=INK)
    ax.set_xticks([x - 0.5 for x in range(1, len(keys))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(suites))], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, colors=INK_2, length=0)
    cb.set_label("Drop (points)", fontsize=8, color=INK_2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"gap_heatmap.{ext}", facecolor=SURFACE)
    plt.close(fig)


def rank_tasks(out: Path, summary: dict, tasks: dict, min_vanilla: float):
    # Qualifies: vanilla >= min_vanilla, all 7 perturbations with full variants, no known issue; ranked by mean drop
    rows = []
    for suite, by_tid in summary["by_task"].items():
        for tid, r in by_tid.items():
            tid = int(tid)
            per = {k: r[k]["pc_success"] for k in PERTURBATIONS}
            full = len({r[k]["n"] for k in PERTURBATIONS} | {r["vanilla"]["n"]}) == 1
            ood = sum(per.values()) / len(per)
            issue = KNOWN_ISSUES.get((suite, tid), "")
            reasons = [x for x, bad in [(f"vanilla < {min_vanilla:.0f}", r["vanilla"]["pc_success"] < min_vanilla),
                                        ("fewer variants", not full), (issue, bool(issue))] if bad]
            rows.append({"suite": SUITE_NAMES[suite], "task": tid, "instruction": tasks[suite][tid],
                         "vanilla": r["vanilla"]["pc_success"], "ood_mean": round(ood, 1),
                         "delta": round(ood - r["vanilla"]["pc_success"], 1), "per": per,
                         "qualifies": not reasons, "excluded_because": "; ".join(reasons)})
    rows.sort(key=lambda x: (not x["qualifies"], x["delta"]))

    head = "| Rank | Suite | # | Instruction | Vanilla | OOD mean | Δ | " + " | ".join(
        PERTURBATIONS[k].split()[0] for k in PERTURBATIONS) + " | Excluded because |"
    lines = ["# Task ranking for rapid experiments", "",
             f"Qualifies: vanilla success ≥ {min_vanilla:.0f}%, all 7 perturbations with 10 variants, no known simulator "
             "issue. Ranked by Δ = mean OOD success over the 7 perturbations − vanilla success. Per-task rates are "
             "from 10 episodes each.", "", head,
             "|---|---|---|---|---|---|---|" + "---|" * len(PERTURBATIONS) + "---|"]
    for i, x in enumerate(rows, 1):
        lines.append(f"| {i if x['qualifies'] else ''} | {x['suite']} | {x['task']} | {x['instruction']} | "
                     f"{x['vanilla']:.0f} | {x['ood_mean']:.1f} | {x['delta']:+.1f} | "
                     + " | ".join(f"{x['per'][k]:.0f}" for k in PERTURBATIONS) + f" | {x['excluded_because']} |")
    (out / "task_ranking.md").write_text("\n".join(lines) + "\n")
    with open(out / "task_ranking.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["suite", "task", "instruction", "vanilla", "ood_mean", "delta", *PERTURBATIONS, "qualifies",
                    "excluded_because"])
        for x in rows:
            w.writerow([x["suite"], x["task"], x["instruction"], x["vanilla"], x["ood_mean"], x["delta"],
                        *(x["per"][k] for k in PERTURBATIONS), x["qualifies"], x["excluded_because"]])


def main():
    parser = argparse.ArgumentParser(description="Robustness-gap table, plots and task ranking")
    parser.add_argument("--run", required=True, type=Path, help="--out dir of an eval_combined.py run")
    parser.add_argument("--out", type=Path, help="default: <run>/gap")
    parser.add_argument("--min-vanilla", type=float, default=80.0, help="min vanilla success (%%) of a candidate task")
    args = parser.parse_args()
    out = args.out or args.run / "gap"
    out.mkdir(parents=True, exist_ok=True)

    summary = json.loads((args.run / "summary.json").read_text())
    tasks = {}
    for line in (args.run / "plus_episodes.jsonl").read_text().splitlines():
        e = json.loads(line)
        tasks.setdefault(e["suite"], {})[e["vanilla_task_id"]] = e["vanilla_instruction"]
    overall = gap_rows(summary["overall"])
    by_suite = {s: gap_rows(summary["by_suite"][s]) for s in summary["by_suite"]}

    write_tables(out, overall, by_suite, summary["overall"]["vanilla"])
    plot_gap(out, overall)
    plot_heatmap(out, by_suite)
    rank_tasks(out, summary, tasks, args.min_vanilla)
    print(f"Wrote {', '.join(sorted(p.name for p in out.iterdir()))} to {out}")


if __name__ == "__main__":
    main()
