"""
CP Optimizer interval model of the Robotic Cell Problem with a fixed job order.

Stations 0 (source), 1..m (machines), m+1 (sink). One robot, no machine buffers.

Loaded move carry[i,j]: job j from station i to i+1, duration τ[i,i+1].
Occupation occupy[i,j]: job j sits on machine i from drop-off until pickup
(size ≥ p[i,j]), so the machine stays blocked while waiting for the robot.

Empty robot travel is the CPO sequence transition from the drop-off of one
loaded move (station i+1) to the pickup of the next (station k): τ[i+1,k].

Usage (from repo root):
    .venv/bin/python models/cp/robotic_cell_interval.py [instance] [time_limit_s]
"""

import sys
from pathlib import Path

from docplex.cp.config import context
from docplex.cp.model import CpoModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from instance.instance_reader import read_instance  # noqa: E402


def solve_instance(instance_path, time_limit=60, log="Quiet"):
    """Build and solve the interval CP. Returns (makespan or None, proven, solve_time, sol, carry)."""
    m, n, p, tau = read_instance(instance_path)
    H = (
        sum(p[i][k] for i in range(1, m + 1) for k in range(1, n + 1))
        + n * sum(tau[i][i + 1] for i in range(m + 1))
        + (n - 1) * tau[m + 1][0]
    )
    jobs = range(1, n + 1)
    machines = range(1, m + 1)
    mdl = CpoModel(name="rcp_interval")

    carry = {
        #move job j from machine i to machine i + 1
        (i, j): mdl.interval_var(size=tau[i][i + 1], start=(0, H), name=f"carry_{i}_{j}") #length of interval is robot traveling
        for i in range(m + 1)
        for j in jobs
    }
    occupy = {
        #completition time + time not moved since for job j on machine i
        (i, j): mdl.interval_var(size=(p[i][j], H), start=(0, H), name=f"occupy_{i}_{j}") #length of interval must be between processing time and horizon.
        for i in machines
        for j in jobs
    }

    for i in machines:
        for j in jobs:
            #start(occupy) = end(carry)
            mdl.add(mdl.start_at_end(occupy[i, j], carry[i - 1, j])) #processing on machine i starts the instant the robot finishes move i-1 -> i
            #end(occupy) = start(carry)
            mdl.add(mdl.end_at_start(occupy[i, j], carry[i, j])) #start time of carrying job j from machine i to machine i+1 is exactly the end time of occupying machine i

    for i in machines:
        mdl.add(mdl.no_overlap([occupy[i, j] for j in jobs])) #jobs cannot be on more than one machine
        for j in range(1, n):
            #end(occupy(i,j)) < start(occupy(i,j+1))
            #start time of next job j+1 occupying on same machine i has to be after job j leaves
            mdl.add(mdl.end_before_start(occupy[i, j], occupy[i, j + 1])) 

    for j in range(1, n):
        #jobs are dispatched by order from source node
        mdl.add(mdl.end_before_start(carry[0, j], carry[0, j + 1]))

    for i in machines:
        for j in range(2, n + 1):
            #blocked machine constraint: (𝑖 < 𝑚) ∧ (𝑥𝑖+1 ≠ 0).
            mdl.add(mdl.end_before_start(carry[i, j - 1], carry[i - 1, j]))

    #enumerate all loaded carry moves by station, then job
    #move job j to station i+1 from i c(i,j)
    #carry_list  = [c01, c02, c03, c04, c11, c12, c13, c14,  c21, c22, c23, c24,  c31, c32, c33, c34]
    carry_list = [carry[i, j] for i in range(m + 1) for j in jobs]

    #carry type is just the source of the move; i
    carry_types = [i for i in range(m + 1) for _ in jobs]

    #seq is the permuation of the robot moves that solve the instance
    #partial permutation that gets tighter as search continues, first value is carry[0,1]. grows with every branch
    seq = mdl.sequence_var(carry_list, types=carry_types, name="robot")

    #every entry of no_carry is travel time of robot with no job in hand

    #after a type-a carry the robot is at a+1, next type-b carry picks up at b, so gap is tau[a + 1][b]
    #no_carry[a][b] = tau[a + 1][b]
    no_carry = [[tau[a + 1][b] for b in range(m + 1)] for a in range(m + 1)]

    #1)loaded carries in seq do not overlap (one robot)
    #2) if a type a carry is immediately followed by a type b carry, then start(next carry) ≥ end(prev carry) + no_carry[type(prev)][type(next)]
    mdl.add(mdl.no_overlap(seq, no_carry, is_direct=True))

    #make it so the first move is moving job 1 from source to machine 1
    mdl.add(mdl.first(seq, carry[0, 1]))

    #minimize last carry move of job n to sink - makespan
    mdl.add(mdl.minimize(mdl.end_of(carry[m, n])))

    cpopt = ROOT / ".venv" / "bin" / "cpoptimizer"
    if cpopt.exists():
        context.solver.agent = "local"
        context.solver.local.execfile = str(cpopt)

    sol = mdl.solve(TimeLimit=time_limit, LogVerbosity=log)
    status = sol.get_solve_status() if sol else "no solution"
    proven = status == "Optimal"
    obj = None
    if sol and sol.get_objective_values():
        obj = int(sol.get_objective_values()[0])
    elif sol and sol.get_objective_value() is not None:
        obj = int(sol.get_objective_value())
    solve_time = float(sol.get_solve_time()) if sol else 0.0
    return obj, proven, solve_time, status, sol, carry, m, n


if __name__ == "__main__":
    instance_path = sys.argv[1] if len(sys.argv) > 1 else ROOT / "instance" / "paper_sample_instance.txt"
    time_limit = float(sys.argv[2]) if len(sys.argv) > 2 else 60
    obj, proven, solve_time, status, sol, carry, m, n = solve_instance(instance_path, time_limit, log="Terse")
    print(f"instance:  {instance_path}")
    print(f"m, n:      {m}, {n}")
    print(f"status:    {status}")
    print(f"makespan:  {obj}")
    print(f"optimal:   {proven}")
    print(f"time:      {solve_time:.3f}s")
    if sol and obj is not None:
        moves = []
        for i in range(m + 1):
            for j in range(1, n + 1):
                vs = sol.get_var_solution(carry[i, j])
                moves.append((vs.get_start(), i, j, vs.get_end()))
        moves.sort()
        print("robot moves (start, i -> i+1, job, end):")
        for start, i, j, end in moves:
            print(f"  t={start:4d}  carry {i}->{i + 1}  job {j}  (end {end})")

