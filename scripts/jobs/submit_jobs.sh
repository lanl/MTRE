#!/usr/bin/env bash
set -euo pipefail

TOPIC_ROOT="./project/MMMU"
TOPIC_LIST="${TOPIC_ROOT}/_topic_list.txt"

# Build (or refresh) topic list
# bash 01_make_topics.sh

N=$(wc -l < "${TOPIC_LIST}")
if [[ "${N}" -eq 0 ]]; then
  echo "No topics found in ${TOPIC_ROOT}" >&2
  exit 1
fi

# Concurrency throttle (how many subjects run at once)
MAX_PAR=20   # <-- tune to your cluster’s fairshare

# Submit array 0..N-1 with concurrency cap
echo "Submitting ${N} array tasks (0..$((N-1))) with max parallel ${MAX_PAR}"
sbatch --array=0-$((N-1))%${MAX_PAR} /MTRE/scripts/jobs/02_run_mmmu_topic.sbatch
