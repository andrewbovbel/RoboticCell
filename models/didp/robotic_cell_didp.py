import sys
from pathlib import Path

import didppy as dp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from instance.instance_reader import read_instance  

instance_path = sys.argv[1] if len(sys.argv) > 1 else ROOT / "instance" / "paper_sample_instance.txt"
m, n, p, tau = read_instance(instance_path)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = dp.Model(maximize=False, float_cost=False)

station = model.add_object_type(number=m + 2)  # 0 = input buffer, 1..m = machines, m+1 = output buffer
job = model.add_object_type(number=n + 1)      # 0 = "no job", 1..n = jobs in processing order

"""
    Robot starts at source node = 0.
    This state variable keeps track of which station the robot is
    Variable indexes τ[y, i] in the cost
"""
robot = model.add_element_var(object_type=station, target=0, name="y")

"""
    Tracks xi from paper; indicates the first unprocessed job in station i.
    x0 is source node; where the initial value of the first unprocessed job is 1.
    If x_i, i >= 1: job is currently on machine i; 0 if the machine is idle.
"""
x0 = model.add_element_var(object_type=job, target=1, name="x_0")
job_on = [x0]
for i in range(1, m + 1):
    job_on.append(model.add_element_var(object_type=job, target=0, name=f"x_{i}"))

"""
    Important change!
    With absolute C_i, two robot routes that reach the same physical situation (same x, same remaining work on each machine) but at different clock times are different states.
    We propose residual: the remaining processing time on machine i (C_i - t), 0 if idle.
    The clock t itself is not a state variable: each transition adds its duration to the cost,
    so V(target) is the makespan.
"""
residual = [None]  # the input buffer has no processing time, so index 0 is unused
for i in range(1, m + 1):
    residual.append(model.add_int_resource_var(target=0, less_is_better=True, name=f"r_{i}"))

# Tables of constants
travel = model.add_int_table(tau)  # tau[y, i]
proc = model.add_int_table(p)      # p[i, k]

"""
Transitions: one per station i = 0..m, "carry the job on station i to i+1"

(9) with t subtracted:
delta = max(tau[y, i], C_i - t) + tau[i, i+1]

C_i - t = residual
"""

for i in range(m + 1):
    if i == 0: 
        move_duration = travel[robot,0] + travel[0,1] #the source node's next job is always available, robot does not need to wait
    else:
        move_duration = dp.max(travel[robot,i], residual[i]) + travel[i, i+1] #max(travel to i, work left on i) + travel from i to i+1

    preconditions = [job_on[i] != 0] #there must be a job to pick up
    if i < m:
        preconditions.append(job_on[i + 1] == 0) #next station must be free. if i==m, then sink always has room.

    effects = [(robot, i + 1)]  #robot position gets updated

    #cases 11,12

    if i == 0:
        #case (11)(II)
        effects.append((x0, (x0 == n).if_then_else(0, x0 + 1))) #if x0 holds job n as next job, make it empty (0), else increment to next job.
    else:
        #case (11)(I)
        effects.append((job_on[i], 0)) #for other stations, buffer is size 0 -> so pop it. 

        #case (12)(I)
        effects.append((residual[i], 0)) #station is idle.

    # the destination i+1 is a machine, not the sink (m+1)
    if i < m:
        #case (11)(III)
        effects.append((job_on[i + 1], job_on[i])) #move job k to proceeding machine i + 1
        #case (12)(III)
        effects.append((residual[i + 1], proc[i + 1, job_on[i]])) #r`_i+1 = p_{i+1,k}

    #no need for (12)(II). It states if that station is the source and more jobs remain, stamp its completion time to now, C'_0 = t'. We do not use passing time t.

    
    for j in range(1, m + 1):
        #stations j not involved in the move
        if j != i and j != i + 1:
            #case (12)(IV): clock moved forward by move_duration so decrease remaining processing time of machine j
            effects.append((residual[j], dp.max(0, residual[j] - move_duration))) #C_j − t' = C_j − (t + move_duration) = residual[j] − move_duration
            #case (11)(IV) : no update needed
            #effects.append((job_on[j],job_on[j])) is redundant

    model.add_transition(
        dp.Transition(
            name=f"move {i}->{i + 1}",
            cost=move_duration + dp.IntExpr.state_cost(),
            preconditions=preconditions,
            effects=effects,
        )
    )

"""
Base case: all jobs have reached the output buffer, i.e. every x_i = 0 (goal state)
"""
model.add_base_case([job_on[i] == 0 for i in range(m + 1)])

"""
Dual bounds (lower bounds on remaining makespan)

Why do we do lower bound on remaining makespan?
If a current search shows the lower bound + current cost being bigger or equal to than our best incumbent, stop working on that branch! 
We run the dual bound on every state to get a real time lower bound.
"""

"""
Robot bound: every remaining move must be carried by the robot.
Jobs still in the source node (k..n when x_0 = k) each need fwd[0]; job on machine i needs fwd[i].
"""
fwd = [sum(tau[l][l + 1] for l in range(i, m + 1)) for i in range(m + 1)] #fwd[i] = forward travel from station i to the output buffer

robot_travel = [0] + [(n - k + 1) * fwd[0] for k in range(1, n + 1)] #if x_0 = k the jobs in the source are k,k+1,...,n which is n - k + 1 jobs. multiply n-k+1 by fwd[0] to get how much time it takes to move all jobs to end.
robot_travel_table = model.add_int_table(robot_travel)

robot_bound = robot_travel_table[x0]
for i in range(1, m + 1):
    robot_bound += (job_on[i] != 0).if_then_else(fwd[i], 0) #if there exists a job on machine i, machine i needs fwd[i].
model.add_dual_bound(robot_bound)

"""
Machine bounds: machine i must finish its current job, process every job still in the source node, 
and then the last job (n) must travel to the sink node through machines i+1..m.
"""
for i in range(1, m + 1):
    last_job = n
    tail = fwd[i] + sum(p[after][last_job] for after in range(i + 1, m + 1)) #last job (n) must travel to the sink node through machines i+1..m
    remaining = [0] + [sum(p[i][w] for w in range(k, n + 1)) + tail for k in range(1, n + 1)] #process every job still in the source node
    remaining_table = model.add_int_table(remaining)
    model.add_dual_bound(residual[i] + remaining_table[x0]) # index x_0 = k. machine i must finish its current job, process every job left in source (n - k + 1 left), and last job has to travel to sink node.

"""
Solve
"""
time_limit = float(sys.argv[2]) if len(sys.argv) > 2 else 60  # optional 2nd CLI arg, seconds
solver = dp.CABS(model, time_limit=time_limit, quiet=True)
solution = solver.search()

# Replay the transitions to recover the paper's absolute-time states <t, y, x, C>
t, y = 0, 0
x = [1] + [0] * m
C = [0] + [None] * m  # None plays the role of infinity


def fmt_state():
    cs = " ".join("inf" if c is None else str(c) for c in C)
    return f"t {t:4d}  y {y}  x {' '.join(map(str, x))}  C {cs}"


print(fmt_state())
for tr in solution.transitions:
    i = int(tr.name.split()[1].split("->")[0])
    k = x[i]
    t = max(t + tau[y][i], C[i] if C[i] is not None else 0) + tau[i][i + 1]
    y = i + 1
    if i > 0 or k == n:
        x[i], C[i] = 0, None
    else:
        x[i], C[i] = k + 1, t
    if i < m:
        x[i + 1], C[i + 1] = k, t + p[i + 1][k]
    print(fmt_state())

print(f"makespan: {solution.cost}")
print(f"optimal:  {solution.is_optimal}")
print(f"time:     {solution.time:.3f}s")
