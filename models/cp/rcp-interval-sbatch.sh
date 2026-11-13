#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=def-beck
#SBATCH --time=00:15:00
#SBATCH --ntasks-per-node=22
#SBATCH --mail-user=andrew.bovbel@mail.utoronto.ca
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-rcp-cp-%j.out
#SBATCH --error=slurm-rcp-cp-%j.err
#SBATCH --chdir=/scratch/bovbelan

## Interval CP vs published DP on all 22 benchmark instances.
## Each solve is capped at 120 s and they all run at once (Workers=1).
## The Slurm wall clock is 3 min: 2 min of solving, plus startup and the
## csv/plot collect. HOME is read-only on Trillium compute nodes.

ROOT="$SCRATCH/RoboticCell"
PY="$ROOT/models/cp/run_experiment.py"
DIR="$ROOT/models/cp/results/shards"
LOGS="$ROOT/models/cp/results/logs"
TIME_LIMIT=120

module load StdEnv/2023
module use ~/modulefiles
module load mycplex/22.2.0

source "$HOME/.venv/bin/activate"

command -v parallel >/dev/null || module load gnu-parallel
command -v parallel >/dev/null || { echo "ERROR: GNU parallel not available" >&2; exit 1; }

mkdir -p "$DIR" "$LOGS"

export PY DIR LOGS TIME_LIMIT
export PARALLEL_SHELL=/bin/bash

NT="${SLURM_NTASKS_PER_NODE:-$SLURM_TASKS_PER_NODE}"

parallel -j "$NT" --halt never \
  'safe=$(printf "%s" {1} | tr "/" "_"); python3 -u "$PY" job "$TIME_LIMIT" {1} "$DIR/${safe}.json" | tee "$LOGS/${safe}.txt"' \
  ::: $(cd "$ROOT/instance/benchmarks" && find . -name '*.txt' -printf '%P\n' | sort)

python3 "$PY" collect "$DIR"

echo "Done. Shards in $DIR, csv and png in $ROOT/models/cp/results"
