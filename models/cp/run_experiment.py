"""Run interval CP on every local benchmark instance and compare to the authors' DP.

Published optima and C++ times come from
instance/benchmarks/computational_results_dataset_D*.csv (exact_sol, exact_time).

Writes:
    models/cp/results/cp_vs_dp.csv
    models/cp/results/cp_vs_dp.png

Usage:
    .venv/bin/python models/cp/run_experiment.py [time_limit_s] [substring_filter]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from docplex.cp.solver.solver import CpoSolverException

from instance.instance_reader import read_instance  # noqa: E402
from models.cp.robotic_cell_interval import solve_instance  # noqa: E402

BENCH = ROOT / "instance" / "benchmarks"
OUT = ROOT / "models" / "cp" / "results"
CSV_PATH = OUT / "cp_vs_dp.csv"
PNG_PATH = OUT / "cp_vs_dp.png"

TIME_LIMIT = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0
PATTERN = sys.argv[2] if len(sys.argv) > 2 else ""


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
    import numpy as np
    import matplotlib.pyplot as plt

    comparable = [r for r in rows if r["cp_obj"] is not None and r["dp_obj"] is not None]
    labels = [r["label"] for r in rows]
    idx = np.arange(len(rows))
    width = 0.4
    dp_obj = [r["dp_obj"] for r in rows]
    cp_obj = np.array([r["cp_obj"] if r["cp_obj"] is not None else np.nan for r in rows], dtype=float)
    dp_t = [r["dp_time_s"] for r in rows]
    cp_t = np.array(
        [
            (r["cp_time_to_best_s"] if r["cp_time_to_best_s"] is not None else r["cp_time_s"])
            if r["cp_obj"] is not None
            else np.nan
            for r in rows
        ],
        dtype=float,
    )

    fig, axes = plt.subplots(2, 1, figsize=(12.5, 8.2))

    ax = axes[0]
    ax.bar(idx - width / 2, dp_obj, width, label="DP exact_sol (authors)")
    ax.bar(idx + width / 2, cp_obj, width, label="CP incumbent")
    ax.set_ylabel("Makespan")
    ax.set_title("Objective: published DP vs interval CP")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(idx - width / 2, dp_t, width, label="DP time (authors' C++)")
    ax.bar(idx + width / 2, cp_t, width, label="CP time")
    ax.set_ylabel("Seconds (log)")
    ax.set_yscale("log")
    ax.set_title("Time to prove (DP) vs CP solve time")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    n_skip = len(rows) - len(comparable)
    fig.suptitle(
        f"RCP interval CP vs Minoli & Righini DP  ·  {len(comparable)}/{len(rows)} CP incumbents"
        + (f", {n_skip} CPO community size limit" if n_skip else ""),
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=160)
    plt.close(fig)


def main():
    published = load_published()
    instances = sorted(p for p in BENCH.rglob("*.txt") if PATTERN in str(p))
    print(
        f"{'instance':<44} {'dp':>6} {'cp':>6} {'proven':>6} {'cp s':>8} {'dp s':>8}  status",
        flush=True,
    )
    rows = []
    for inst in instances:
        key = str(inst.relative_to(BENCH))
        pub, pub_time = published.get(key, (None, None))
        row = run_one(inst, pub, pub_time)
        if "note" not in row:
            row["note"] = row["cp_status"]
        rows.append(row)
        print(
            f"{row['instance']:<44} {str(row['dp_obj']):>6} {str(row['cp_obj'] if row['cp_obj'] is not None else '—'):>6} "
            f"{str(row['cp_proven'] if row['cp_proven'] is not None else '—'):>6} "
            f"{(f'{row['cp_time_s']:.2f}s' if row['cp_time_s'] is not None else '—'):>8} "
            f"{(f'{row['dp_time_s']:.3f}s' if row['dp_time_s'] is not None else 'n/a'):>8}  "
            f"{row['note']}",
            flush=True,
        )

    write_csv(rows)
    plot(rows)
    n_ok = sum(1 for r in rows if r["cp_obj"] is not None)
    print(f"\n{len(rows)} instances, {n_ok} with a CP incumbent")
    print(f"csv: {CSV_PATH}")
    print(f"fig: {PNG_PATH}")


if __name__ == "__main__":
    main()
