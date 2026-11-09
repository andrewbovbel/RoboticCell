# RoboticCell

Exact solver for the Robotic Cell Problem with a fixed job processing order, modeled in
DIDPPy (domain-independent dynamic programming). Reproduces the DP of:

> A. Minoli, G. Righini, "A dynamic programming algorithm for the Robotic Cell Problem
> with a fixed job processing order", EJOR 2026. https://doi.org/10.1016/j.ejor.2026.09.015
> Authors' C++ code, instances and results: https://github.com/alessandro-minoli/DP_for_RCP_with_a_fixed_job_processing_order

## Problem in one paragraph

`m` machines in a line, stations `0..m+1` where `0` is the input buffer (source) and `m+1` the
output buffer (sink). `n` jobs must visit machines `1..m` in order, in the fixed sequence
`1..n`. Machines hold one job and have no buffers (blocking). A single robot carries one job
at a time; `tau[a][b]` is travel time between stations, `p[i][k]` is processing time of job `k`
on machine `i`. Minimize makespan.

Conventions, everywhere in this repo: `m` = machines, `n` = jobs (paper notation; do not swap).
Job index `0` is the sentinel "no job"; real jobs are `1..n`.

## Layout

```
instance/instance_reader.py        read_instance(path) -> (m, n, p, tau); p padded to (m+2)x(n+1) with zeros
instance/paper_sample_instance.txt paper's Fig. 1 toy instance (m=3, n=4), optimum 553
instance/benchmarks/               subset of authors' dataset_D1 / dataset_D2 + their results CSVs
models/didp/robotic_cell_didp.py   the DIDPPy model + CABS solve + replay of the schedule
models/didp/run_benchmarks.py      runs the model on every instance in instance/benchmarks, compares to published optimum
```

Run (from any cwd; the model resolves the repo root from `__file__`):

```
.venv/bin/python models/didp/robotic_cell_didp.py [instance_path] [time_limit_s]
.venv/bin/python models/didp/run_benchmarks.py [time_limit_s] [substring_filter]
```

Instance file format (authors'): `m`, `n`, then `m` rows of `n` processing times, then
`(m+2)` rows of `(m+2)` travel times.

## The model (how the paper's DP maps to DIDPPy)

Paper state: `<t, y, (x_0..x_m), (C_0..C_m)>`. The DIDPPy state is the same tuple.
`INF` is the paper's ∞ (a constant larger than the one-job-at-a-time makespan U).

| paper | ours | type | target (initial state) |
|---|---|---|---|
| `t` | `t` | IntVar | 0 |
| `y` | `robot` | ElementVar(station, m+2 objects) | 0 |
| `x_0` | `x0` = `job_on[0]` | ElementVar(job, n+1 objects) | 1 |
| `x_i` | `job_on[i]` | ElementVar(job) | 0 |
| `C_0` | `C[0]` | IntVar | 0 |
| `C_i` | `C[i]` | IntVar | INF |

Transitions: one per source station `i = 0..m`, "carry job on `i` to `i+1`".
- preconditions: `job_on[i] != 0` and (`i == m` or `job_on[i+1] == 0`)
- eq. (9): `t' = max(t + travel[robot,i], C[i]) + travel[i,i+1]`
- eq. (10): `robot <- i+1`
- eqs. (11)/(12): case I unload (`x_i <- 0`, `C_i <- INF`); case II source (`x0 <- x0+1`, `C_0 <- t'`);
  case III destination (`x_{i+1} <- k`, `C_{i+1} <- t' + p[i+1,k]`); case IV unchanged
- cost = `(t' - t) + state_cost()` so V(target) is the t of a final state (z*)
- base case: all `job_on[i] == 0`, cost 0

`t` and `C_i` are ordinary IntVars, not resource variables: with additive remaining cost, a later
clock can mean less waiting, so `less_is_better` on `t` would be unsound. The paper's dominance
rule (eq. 17) is what made absolute times work in their C++; DIDPPy cannot express that rule.

Dual bounds (lower bounds on remaining makespan; DIDPPy takes the max of all registered):
- robot bound: forward travel for every remaining move (`buffer_travel[x0]` + `fwd[i]` per occupied machine)
- per-machine bound i: `max(0, C[i]-t)` if occupied, plus `sum_{w>=x0} p[i][w] + tail`, table on `x0`

## What DIDPPy cannot express from the paper

The paper's dominance rule (eq. 17, vectors `d`, `e`, bounds `lambda`, `mu`) compares states
with different configurations and uses a joint function of both states. DIDPPy's only
dominance mechanism is resource variables (identical non-resource vars + componentwise
comparison). Absolute `t` and `C` cannot be resource variables, so we currently have no
dominance beyond identical-state duplicate detection. This is why the authors' C++ scales
to m=15 and ours degrades past m~12. Implementing the full rule needs a hand-written
label-correcting search, not a DIDPPy model.

## Validation status (as of 2026-11-03)

- Toy instance: 553, replayed state trace identical to the authors' C++ verbose output.
- 22 benchmark instances: every proven result matches the published optimum (0 mismatches).
- Jobs scale fine (m=4, n=30 in 4 ms). Machines are the limiting dimension:
  m=10 ~3 s, m=12 ~7 s, m=13 n=8 found optimum but needs >60 s to prove,
  m=15 n=8 gap +6 at 60 s (authors needed 240 s on that one too).
- D2 pair at m=12 n=10: travel-heavy `tr_10_pr_10` proven in 1 s (authors 0.6 s);
  processing-heavy `tr_50_pr_300` proven in 93 s (authors 13.6 s). => the machine bound is the weak one.
- Parallel CABS (`threads=8`, Hdbs2) was measured SLOWER than sequential (3.3 s -> 5.6 s,
  6.9 s -> 7.5 s): per-state work is too small to amortize per-layer sync. Keep threads=1.

A run where `optimal: True` and makespan != published is a bug (unsound bound or wrong effect)
and must never be accepted. A mismatch with `optimal: False` is just the time limit.

## Reference oracle

Clone the authors' repo (e.g. to /tmp) and build without cmake:

```
git clone https://github.com/alessandro-minoli/DP_for_RCP_with_a_fixed_job_processing_order /tmp/dp_rcp
cd /tmp/dp_rcp && mkdir -p bin && clang++ -std=c++20 -O2 -DNDEBUG -Iinclude src/*.cpp -o bin/ref
./bin/ref <instance> exact_dp 1      # prints the optimal state sequence t,y,x,C
```

Their `MILP_formulation.py` (docplex, already in requirements.txt) is an independent model
for cross-checking small instances (m<=6, n<=8).

## Next steps, in priority order

1. ~~Tighten the machine bound with upstream jobs + live tail.~~ Tried 2026-11-03. Sound but slower:
   m=10/12 ~2×, tr_10_pr_10 1.0s→2.5s, tr_50_pr_300 missed the optimum in 60s (5642 vs 5639).
   Reverted to the cheap `x0`-indexed table. Extra per-state work and CABS beam distortion
   outweighed the tighter h. Do not retry without a cheaper encoding (one table, no OR-chain).
2. Try `dp.CAASDy` (single-pass A*) for the optimality proof; CABS re-expands layers on each beam doubling.
   A/B on `tr_50_pr_300` (93 s CABS) and `m_13_n_08` (unproven at 60 s).
3. Robot bound: account for empty return legs (careful: min empty leg can be 0).
4. Try `keep_all_layers=True`.
5. Only if m>=13 matters: port the authors' label-correcting algorithm with the full dominance
   rule to Python as a separate model (models/label_correcting/), keep the DIDPPy model as the reference.

## Pitfalls already hit

- `from instance...` only works because the model inserts the repo root into sys.path; keep that.
- `job` object type must have `n+1` objects, `station` must have `m+2`.
- Idle `C_i` is `INF` (paper's ∞), a constant `U+1` where U is the one-job-at-a-time makespan.
  Do not use `C_i - t` in a bound unless `x_i != 0` (otherwise it is INF).
- Effects are all evaluated on the pre-transition state, so `job_on[i]` on the right-hand side
  of the destination effects is the carried job even though `job_on[i]` is also being reset.
- The PDF-to-text copy of the paper loses the bar on the unloaded station index in eqs. (11)/(12);
  "i = i" there means "this component is the source station".
