"""
Run robotic_cell_didp.py on every instance under instance/benchmarks/ and compare the
makespan with the authors' published optimum (exact_sol column of their results CSVs).

Usage:
    python models/didp/run_benchmarks.py [time_limit_seconds] [substring_filter]

Examples:
    python models/didp/run_benchmarks.py             # 60 s per instance, all instances
    python models/didp/run_benchmarks.py 10 m_04     # 10 s per instance, only m=4 instances

Exit code is 1 if any instance was *proven* optimal with a makespan different from the
published one (a modeling bug or an unsound dual bound). A mismatch on an instance that
only hit the time limit is reported but is not a failure.
"""

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "didp" / "robotic_cell_didp.py"
BENCH = ROOT / "instance" / "benchmarks"

time_limit = sys.argv[1] if len(sys.argv) > 1 else "60"
pattern = sys.argv[2] if len(sys.argv) > 2 else ""


def load_expected():
    """Map 'dataset_D1/m_04_n_04_seed_00.txt' -> (published optimal makespan, authors' C++ solve time in s)."""
    expected = {}
    for csv_path in BENCH.glob("computational_results_dataset_*.csv"):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row["exact_sol"]:
                    expected[row["instance_path"]] = (int(row["exact_sol"]), float(row["exact_time"]))
    return expected


def run_one(instance):
    out = subprocess.run(
        [sys.executable, str(MODEL), str(instance), time_limit],
        capture_output=True, text=True, check=True,
    ).stdout
    fields = {}
    for line in out.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            fields[key.strip()] = val.strip()
    return int(fields["makespan"]), fields["optimal"] == "True", float(fields["time"].rstrip("s"))


expected = load_expected()
instances = sorted(p for p in BENCH.rglob("*.txt") if pattern in str(p))

failures = 0
summary = []
for inst in instances:
    key = str(inst.relative_to(BENCH))
    pub, pub_time = expected.get(key, (None, None))
    print(f"--- {key}", flush=True)
    got, proven, solve_time = run_one(inst)

    if pub is None:
        status = "no reference"
    elif got == pub:
        status = "ok" if proven else "ok (not proven)"
    elif proven:
        status = "MISMATCH, proven optimal -> BUG"
        failures += 1
    else:
        status = f"gap +{got - pub} (time limit)"

    pub_time_str = f"{pub_time:.3f}s" if pub_time is not None else "n/a"
    print(f"makespan:        {got}")
    print(f"published:       {pub}")
    print(f"optimal:         {proven}")
    print(f"time:            {solve_time:.3f}s")
    print(f"published time:  {pub_time_str}  (authors' C++ DP)")
    print(f"status:          {status}")
    print(flush=True)
    summary.append((key, pub, got, proven, solve_time, pub_time_str, status))

print("=" * 110)
print(f"{'instance':<45} {'published':>9} {'ours':>6} {'proven':>6} {'time':>8} {'pub time':>9}  status")
for key, pub, got, proven, solve_time, pub_time_str, status in summary:
    print(f"{key:<45} {str(pub):>9} {got:>6} {str(proven):>6} {solve_time:>7.2f}s {pub_time_str:>9}  {status}")
print()
print(f"{len(instances)} instances, {failures} proven mismatches")
sys.exit(1 if failures else 0)
