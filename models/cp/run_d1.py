"""Compare the interval CP against published D1 optima and C++ DP times."""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from docplex.cp.solver.solver import CpoSolverException

from models.cp.robotic_cell_interval import solve_instance  # noqa: E402

BENCH = ROOT / "instance" / "benchmarks"
time_limit = float(sys.argv[1]) if len(sys.argv) > 1 else 30
pattern = sys.argv[2] if len(sys.argv) > 2 else "dataset_D1"


def load_expected():
    expected = {}
    with open(BENCH / "computational_results_dataset_D1.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["exact_sol"]:
                expected[row["instance_path"]] = (int(row["exact_sol"]), float(row["exact_time"]))
    return expected


expected = load_expected()
instances = sorted(p for p in BENCH.rglob("*.txt") if pattern in str(p))

print(f"{'instance':<40} {'pub':>6} {'cp':>6} {'proven':>6} {'cp s':>8} {'paper s':>8}  status")
for inst in instances:
    key = str(inst.relative_to(BENCH))
    pub, pub_time = expected.get(key, (None, None))
    try:
        obj, proven, t, status, *_ = solve_instance(inst, time_limit, log="Quiet")
    except CpoSolverException as e:
        msg = str(e)
        tag = "CPO community size limit" if "size limit" in msg else msg.split("\n")[0][:40]
        print(
            f"{key:<40} {str(pub):>6} {'—':>6} {'—':>6} {'—':>8} {pub_time:>7.3f}s  {tag}",
            flush=True,
        )
        continue
    if obj is None:
        tag = "no incumbent"
    elif pub is None:
        tag = "no reference"
    elif obj == pub:
        tag = "ok" if proven else "ok (not proven)"
    elif proven:
        tag = "MISMATCH"
    else:
        tag = f"gap +{obj - pub}"
    print(
        f"{key:<40} {str(pub):>6} {str(obj):>6} {str(proven):>6} {t:>7.2f}s {pub_time:>7.3f}s  {tag}",
        flush=True,
    )
