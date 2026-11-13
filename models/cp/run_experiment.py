"""Run interval CP on every local benchmark instance and compare to the authors' DP.

Published optima and C++ times come from
instance/benchmarks/computational_results_dataset_D*.csv (exact_sol, exact_time).

Writes:
    models/cp/results/cp_vs_dp.csv
    models/cp/results/cp_vs_dp_mXX.png   one figure per machine count

Usage:
    .venv/bin/python models/cp/run_experiment.py [time_limit_s] [substring_filter]
    .venv/bin/python models/cp/run_experiment.py job TIME_LIMIT REL_PATH OUT_JSON
    .venv/bin/python models/cp/run_experiment.py collect SHARD_DIR
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from docplex.cp.solver.solver import CpoSolverException

from instance.instance_reader import read_instance  # noqa: E402
from models.cp.robotic_cell_interval import solve_instance  # noqa: E402

BENCH = Path(os.environ["RCP_BENCH"]) if os.environ.get("RCP_BENCH") else ROOT / "instance" / "benchmarks"
OUT = Path(os.environ["RCP_OUT"]) if os.environ.get("RCP_OUT") else ROOT / "models" / "cp" / "results"
CSV_PATH = OUT / "cp_vs_dp.csv"

TIME_LIMIT = 30.0


def load_published():
    published = {}
    for csv_path in sorted(BENCH.glob("computational_results_dataset_*.csv")):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("exact_sol"):
                    published[row["instance_path"]] = (
                        int(row["exact_sol"]),
                        float(row["exact_time"]),
                    )
    return published


def short_label(key: str) -> str:
    name = Path(key).stem
    if "dataset_D2" in key:
        variant = Path(key).parent.name.replace("tr_", "t").replace("_pr_", "p")
        return f"D2-{variant}"
    return name.replace("_seed_00", "").replace("seed_", "s")


def time_to_best(sol) -> float | None:
    if sol is None:
        return None
    infos = sol.get_solver_infos()
    if not infos:
        return None
    for key in (
        "TimeToBestSolution",
        "TimeOfBestSolution",
        "BestSolutionTime",
        "TimeOfLastSolution",
    ):
        val = infos.get(key) if hasattr(infos, "get") else None
        if val is not None:
            return float(val)
    return None


def run_one(inst: Path, pub, pub_time):
    key = str(inst.relative_to(BENCH))
    label = short_label(key)
    m, n, _, _ = read_instance(inst)
    try:
        obj, proven, solve_time, status, sol, _carry, m, n = solve_instance(
            inst, TIME_LIMIT, log="Quiet"
        )
    except CpoSolverException as e:
        msg = str(e)
        note = "CPO community size limit" if "size limit" in msg else msg.split("\n")[0][:80]
        return {
            "instance": key,
            "label": label,
            "m": m,
            "n": n,
            "dp_obj": pub,
            "dp_time_s": pub_time,
            "cp_obj": None,
            "cp_time_s": None,
            "cp_time_to_best_s": None,
            "cp_proven": None,
            "cp_status": note,
            "gap": None,
            "note": note,
        }

    if obj is None:
        note = "no incumbent"
    elif pub is None:
        note = "no reference"
    elif obj == pub:
        note = "match" if proven else "match (not proven)"
    elif proven:
        note = "MISMATCH"
    else:
        note = f"gap +{obj - pub}"

    return {
        "instance": key,
        "label": label,
        "m": m,
        "n": n,
        "dp_obj": pub,
        "dp_time_s": pub_time,
        "cp_obj": obj,
        "cp_time_s": solve_time,
        "cp_time_to_best_s": time_to_best(sol),
        "cp_proven": proven,
        "cp_status": status,
        "gap": None if (obj is None or pub is None) else obj - pub,
        "note": note,
    }


def write_csv(rows):
    OUT.mkdir(parents=True, exist_ok=True)
    fields = [
        "instance",
        "label",
        "m",
        "n",
        "dp_obj",
        "dp_time_s",
        "cp_obj",
        "cp_time_s",
        "cp_time_to_best_s",
        "cp_proven",
        "cp_status",
        "gap",
        "note",
    ]
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def plot(rows):
    """One figure per machine count. The x-axis sweeps the job counts for that m."""
    import numpy as np
    import matplotlib.pyplot as plt

    by_m = {}
    for row in rows:
        by_m.setdefault(row["m"], []).append(row)

    written = []
    for m in sorted(by_m):
        group = sorted(by_m[m], key=lambda r: r["n"])
        comparable = [r for r in group if r["cp_obj"] is not None and r["dp_obj"] is not None]
        labels = [str(r["n"]) for r in group]
        idx = np.arange(len(group))
        width = 0.4
        dp_obj = [r["dp_obj"] for r in group]
        cp_obj = np.array(
            [r["cp_obj"] if r["cp_obj"] is not None else np.nan for r in group], dtype=float
        )
        dp_t = [r["dp_time_s"] for r in group]
        cp_t = np.array(
            [
                (r["cp_time_to_best_s"] if r["cp_time_to_best_s"] is not None else r["cp_time_s"])
                if r["cp_obj"] is not None
                else np.nan
                for r in group
            ],
            dtype=float,
        )

        fig, axes = plt.subplots(2, 1, figsize=(8.6, 7.2))

        ax = axes[0]
        ax.bar(idx - width / 2, dp_obj, width, label="DP exact_sol (authors)")
        ax.bar(idx + width / 2, cp_obj, width, label="CP incumbent")
        ax.set_ylabel("Makespan")
        ax.set_title("Objective: published DP vs interval CP")
        ax.set_xticks(idx)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Jobs  n")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

        ax = axes[1]
        ax.bar(idx - width / 2, dp_t, width, label="DP time (authors' C++)")
        ax.bar(idx + width / 2, cp_t, width, label="CP time")
        ax.set_ylabel("Seconds (log)")
        ax.set_yscale("log")
        ax.set_title("Time to prove (DP) vs CP solve time")
        ax.set_xticks(idx)
        ax.set_xticklabels(labels)
        ax.set_xlabel("Jobs  n")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

        n_skip = len(group) - len(comparable)
        fig.suptitle(
            f"m = {m}  ·  {len(comparable)}/{len(group)} CP incumbents"
            + (f", {n_skip} CPO community size limit" if n_skip else ""),
            fontsize=11,
        )
        fig.tight_layout()
        path = OUT / "cp_vs_dp_m{:02d}.png".format(m)
        fig.savefig(path, dpi=160)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close(fig)
        written.append(path)
    return written


def print_row(row):
    cp_s = f"{row['cp_time_s']:.2f}s" if row["cp_time_s"] is not None else "—"
    dp_s = f"{row['dp_time_s']:.3f}s" if row["dp_time_s"] is not None else "n/a"
    print(
        f"{row['instance']:<44} {str(row['dp_obj']):>6} {str(row['cp_obj'] if row['cp_obj'] is not None else '—'):>6} "
        f"{str(row['cp_proven'] if row['cp_proven'] is not None else '—'):>6} "
        f"{cp_s:>8} "
        f"{dp_s:>8}  "
        f"{row['note']}",
        flush=True,
    )


def finish_row(row):
    if "note" not in row:
        row["note"] = row["cp_status"]
    return row


def cmd_job():
    """One instance. Writes a JSON shard so parallel jobs do not share a file."""
    global TIME_LIMIT
    if len(sys.argv) != 5:
        raise SystemExit("usage: run_experiment.py job TIME_LIMIT REL_PATH OUT_JSON")
    TIME_LIMIT = float(sys.argv[2])
    rel = sys.argv[3]
    out = Path(sys.argv[4])
    inst = BENCH / rel
    if not inst.is_file():
        raise SystemExit(f"missing instance {inst}")
    pub, pub_time = load_published().get(rel, (None, None))
    row = finish_row(run_one(inst, pub, pub_time))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, indent=2))
    print_row(row)


def cmd_collect():
    if len(sys.argv) != 3:
        raise SystemExit("usage: run_experiment.py collect SHARD_DIR")
    shard_dir = Path(sys.argv[2])
    rows = [json.loads(path.read_text()) for path in sorted(shard_dir.glob("*.json"))]
    rows.sort(key=lambda r: r["instance"])
    if not rows:
        raise SystemExit(f"no shards in {shard_dir}")
    write_csv(rows)
    figs = plot(rows)
    n_ok = sum(1 for r in rows if r["cp_obj"] is not None)
    print(f"\n{len(rows)} instances, {n_ok} with a CP incumbent")
    print(f"csv: {CSV_PATH}")
    for fig in figs:
        print(f"fig: {fig}")


def main():
    global TIME_LIMIT
    if len(sys.argv) > 1 and sys.argv[1] == "job":
        cmd_job()
        return
    if len(sys.argv) > 1 and sys.argv[1] == "collect":
        cmd_collect()
        return

    if len(sys.argv) > 1:
        TIME_LIMIT = float(sys.argv[1])
    pattern = sys.argv[2] if len(sys.argv) > 2 else ""
    published = load_published()
    instances = sorted(p for p in BENCH.rglob("*.txt") if pattern in str(p))
    print(
        f"{'instance':<44} {'dp':>6} {'cp':>6} {'proven':>6} {'cp s':>8} {'dp s':>8}  status",
        flush=True,
    )
    rows = []
    for inst in instances:
        key = str(inst.relative_to(BENCH))
        pub, pub_time = published.get(key, (None, None))
        row = finish_row(run_one(inst, pub, pub_time))
        rows.append(row)
        print_row(row)

    write_csv(rows)
    figs = plot(rows)
    n_ok = sum(1 for r in rows if r["cp_obj"] is not None)
    print(f"\n{len(rows)} instances, {n_ok} with a CP incumbent")
    print(f"csv: {CSV_PATH}")
    for fig in figs:
        print(f"fig: {fig}")


if __name__ == "__main__":
    main()
