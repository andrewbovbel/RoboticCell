import sys
from pathlib import Path

import didppy as dp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from instance.instance_reader import read_instance

instance_path = sys.argv[1] if len(sys.argv) > 1 else ROOT / "instance" / "paper_sample_instance.txt"
m, n, p, tau = read_instance(instance_path)

# Paper's "large enough constant" for idle completion times. U is the one-job-at-a-time
# makespan (always feasible), so any real C_i is < INF.
U = (
    sum(p[i][k] for i in range(1, m + 1) for k in range(1, n + 1))
    + n * sum(tau[i][i + 1] for i in range(m + 1))
    + (n - 1) * tau[m + 1][0]
)
INF = U + 1

# ---------------------------------------------------------------------------
# Model  — paper state ⟨t, y, (x_0..x_m), (C_0..C_m)⟩
# ---------------------------------------------------------------------------
model = dp.Model(maximize=False, float_cost=False)

station = model.add_object_type(number=m + 2)  # 0 = input buffer, 1..m = machines, m+1 = output buffer
job = model.add_object_type(number=n + 1)      # 0 = "no job", 1..n = jobs in processing order

"""
    t: elapsed time. Initial 0. Makespan is t at a final state.
"""
t = model.add_int_var(target=0, name="t")

"""
    Robot starts at source node = 0.
    Variable indexes τ[y, i] in eq. (9).
"""
robot = model.add_element_var(object_type=station, target=0, name="y")

"""
    The paper's vector (x_0, ..., x_m).
    x_0: next job waiting in the source (initially 1); 0 once all jobs have left it.
    x_i, i >= 1: job currently on machine i; 0 if the machine is idle.
"""
x0 = model.add_element_var(object_type=job, target=1, name="x_0")
job_on = [x0]
for i in range(1, m + 1):
    job_on.append(model.add_element_var(object_type=job, target=0, name=f"x_{i}"))

"""
    C_i: completion time of the job on station i.
    Idle machines: C_i = INF. C_0 = 0 initially, INF iff the source is empty.
    Initial C = (0, INF, ..., INF).
"""
C = [model.add_int_var(target=0, name="C_0")]
for i in range(1, m + 1):
    C.append(model.add_int_var(target=INF, name=f"C_{i}"))

travel = model.add_int_table(tau)
proc = model.add_int_table(p)

"""
Transitions: one per station i = 0..m, "carry the job on station i to i+1".

Eq. (9):  t' = max{t + τ_{y,i}, C_i} + τ_{i,i+1}
Eq. (10): y' = i + 1
Eq. (11)/(12) cases I–IV on x and C.
"""
for i in range(m + 1):
    # (9) — uses C_i even for i = 0 (the source's completion stamp)
    t_next = dp.max(t + travel[robot, i], C[i]) + travel[i, i + 1]

    preconditions = [job_on[i] != 0]  # non-idle
    if i < m:
        preconditions.append(job_on[i + 1] == 0)  # non-blocked; sink always has room

    effects = [
        (t, t_next),
        (robot, i + 1),  # (10)
    ]

    if i == 0:
        # (11)(II) / (11)(I): advance the source, or empty it after job n
        effects.append((x0, (x0 == n).if_then_else(0, x0 + 1)))
        # (12)(II) C'_0 = t' if more jobs remain; (12)(I) C'_0 = INF after job n
        effects.append((C[0], (x0 == n).if_then_else(INF, t_next)))
    else:
        # (11)(I) / (12)(I): unload machine i
        effects.append((job_on[i], 0))
        effects.append((C[i], INF))

    # destination i+1 is a machine, not the sink
    if i < m:
        # (11)(III) x'_{i+1} = k
        effects.append((job_on[i + 1], job_on[i]))
        # (12)(III) C'_{i+1} = t' + p_{i+1,k}
        effects.append((C[i + 1], t_next + proc[i + 1, job_on[i]]))

    # (11)(IV) / (12)(IV): stations not involved keep x_j and C_j (no effect)

    # cost increment is t' - t, so V(target) = t of the final state = z*
    model.add_transition(
        dp.Transition(
            name=f"move {i}->{i + 1}",
            cost=t_next - t + dp.IntExpr.state_cost(),
            preconditions=preconditions,
            effects=effects,
        )
    )

"""
Base case: final states ⟨t, m+1, (0,...,0), (INF,...,INF)⟩. y = m+1 and C = INF
are implied once every x_i is 0. Cost 0 so V = accumulated t' - t = makespan.
"""
model.add_base_case([job_on[i] == 0 for i in range(m + 1)])

"""
Dual bounds: lower bounds on remaining makespan (V of the current state).
DIDPPy takes the max of all registered bounds.
"""
fwd = [sum(tau[l][l + 1] for l in range(i, m + 1)) for i in range(m + 1)]

# Robot bound: every remaining loaded forward leg.
robot_travel = [0] + [(n - k + 1) * fwd[0] for k in range(1, n + 1)]
robot_travel_table = model.add_int_table(robot_travel)
robot_bound = robot_travel_table[x0]
for i in range(1, m + 1):
    robot_bound += (job_on[i] != 0).if_then_else(fwd[i], 0)
model.add_dual_bound(robot_bound)

# Machine bound: sequential processing on machine i.
for i in range(1, m + 1):
    tail = fwd[i] + sum(p[l][n] for l in range(i + 1, m + 1))
    remaining = [0] + [sum(p[i][w] for w in range(k, n + 1)) + tail for k in range(1, n + 1)]
    remaining_table = model.add_int_table(remaining)
    residual_i = (job_on[i] != 0).if_then_else(dp.max(0, C[i] - t), 0)
    model.add_dual_bound(residual_i + remaining_table[x0])

# Job-path bound (paper λ, using (13) only): dedicated-robot remaining path of the
# job now on station i. Using (13) even when blocked is still a valid LB on pickup time.
# Gated on x_i != 0 so we never subtract t from C_i = INF.
for i in range(m + 1):
    e_i = dp.max(C[i], t + travel[robot, i])  # (13)
    down = proc[i + 1, job_on[i]] if i < m else 0
    for l in range(i + 2, m + 1):
        down += proc[l, job_on[i]]
    path = e_i - t + fwd[i] + down
    model.add_dual_bound((job_on[i] != 0).if_then_else(path, 0))

# μ bound for last job n still in the source (16) at i = 0:
# earliest pickup of n, then dedicated run to the sink.
# μ^n_0 = e_0 + (n - k)(τ_0,1 + τ_1,0), k = x_0.
p_n_all = sum(p[l][n] for l in range(1, m + 1))
source_cycle = tau[0][1] + tau[1][0]
# extra 0→1→0 cycles to reach job n after picking the current x_0 = k; 0 if source empty
cycles_to_n = [0] + [(n - k) * source_cycle for k in range(1, n + 1)]
cycles_to_n_table = model.add_int_table(cycles_to_n)
e0 = dp.max(C[0], t + travel[robot, 0])
mu_n = e0 + cycles_to_n_table[x0]
model.add_dual_bound((x0 != 0).if_then_else(mu_n - t + fwd[0] + p_n_all, 0))

time_limit = float(sys.argv[2]) if len(sys.argv) > 2 else 60
solver = dp.CABS(model, time_limit=time_limit, quiet=True)
solution = solver.search()

# Replay matches the paper's ⟨t, y, x, C⟩ (INF printed as inf)
clock, y = 0, 0
x = [1] + [0] * m
C_val = [0] + [None] * m


def fmt_state():
    cs = " ".join("inf" if c is None else str(c) for c in C_val)
    return f"t {clock:4d}  y {y}  x {' '.join(map(str, x))}  C {cs}"


print(fmt_state())
for tr in solution.transitions:
    i = int(tr.name.split()[1].split("->")[0])
    k = x[i]
    clock = max(clock + tau[y][i], C_val[i] if C_val[i] is not None else 0) + tau[i][i + 1]
    y = i + 1
    if i > 0 or k == n:
        x[i], C_val[i] = 0, None
    else:
        x[i], C_val[i] = k + 1, clock
    if i < m:
        x[i + 1], C_val[i + 1] = k, clock + p[i + 1][k]
    print(fmt_state())

print(f"makespan: {solution.cost}")
print(f"optimal:  {solution.is_optimal}")
print(f"time:     {solution.time:.3f}s")
