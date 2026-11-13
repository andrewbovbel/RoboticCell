#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=def-beck
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=168
#SBATCH --mail-user=andrew.bovbel@mail.utoronto.ca
#SBATCH --mail-type=ALL
#SBATCH --output=slurm-rcp-d1-seed00-%j.out
#SBATCH --error=slurm-rcp-d1-seed00-%j.err
#SBATCH --chdir=/scratch/bovbelan

## Dataset D1, seed 00 only: 168 instances, one per (m, n).
## m = 4..15, n = 4,6,...,30. Each pair appears once.
## Time cap is the slowest of those DP runs: m=15, n=14, seed 00,
## exact_time 1179.99 s, so TIME_LIMIT=1180. All 168 run at once
## (Workers=1). Wall clock is 30 min: the 1180 s solves, plus startup
## and the csv/plot collect.

ROOT="$SCRATCH/RoboticCell"
BENCH="$ROOT/DP_for_RCP_with_a_fixed_job_processing_order"
PY="$ROOT/models/cp/run_experiment.py"
OUT="$ROOT/models/cp/results/d1_seed00"
DIR="$OUT/shards"
LOGS="$OUT/logs"
TIME_LIMIT=1180

module load StdEnv/2023
module use ~/modulefiles
module load mycplex/22.2.0

source "$HOME/.venv/bin/activate"

command -v parallel >/dev/null || module load gnu-parallel
command -v parallel >/dev/null || { echo "ERROR: GNU parallel not available" >&2; exit 1; }

mkdir -p "$DIR" "$LOGS"

export PY DIR LOGS TIME_LIMIT
export RCP_BENCH="$BENCH"
export RCP_OUT="$OUT"
export PARALLEL_SHELL=/bin/bash

NT="${SLURM_NTASKS_PER_NODE:-$SLURM_TASKS_PER_NODE}"

mapfile -t INSTANCES < <(cd "$BENCH" && find dataset_D1 -name '*_seed_00.txt' | sort)
if [ "${#INSTANCES[@]}" -ne 168 ]; then
  echo "ERROR: expected 168 seed-00 instances, found ${#INSTANCES[@]}" >&2
  exit 1
fi

parallel -j "$NT" --halt never \
  'safe=$(printf "%s" {1} | tr "/" "_"); python3 -u "$PY" job "$TIME_LIMIT" {1} "$DIR/${safe}.json" | tee "$LOGS/${safe}.txt"' \
  ::: "${INSTANCES[@]}"

python3 "$PY" collect "$DIR"

echo "Done. Shards in $DIR, csv and one figure per m in $OUT"
