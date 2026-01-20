#!/bin/bash
#SBATCH --job-name=LLaVA_NEXT_40       # Job name
#SBATCH --output=my_bash_job_MMMU_raw.out      # Standard output and error log
#SBATCH --ntasks=1                    # Number of tasks (1 task for a single bash command)
#SBATCH --partition gpu 
#SBATCH -A xxxxx
#SBATCH --time=2:00:00 
#SBATCH -C gpu80
#SBATCH -o /output/LLaVA_NeXT/logs/slurm-%A_%a.out

# Load any necessary modules (if needed)
# module load some_module
#conda activate /usr/projects/unsupgan/geigh_env
# Option A: cluster conda module
#!/bin/bash
module load python  # if your system uses modules

# Make sure Conda commands are available
source $(conda info --base)/etc/profile.d/conda.sh

# Activate your environment
conda activate /usr/projects/unsupgan/geigh_env

cd /usr/projects/unsupgan/MM_TDA/MTRE_R

set -euo pipefail

# --- Configuration (same as your script; edit as needed) ---
TOPIC_ROOT="./project/MMMU"
MODEL_NAME="LLaVA_NeXT"
MODEL_PATH="./project/llava-v1.6-34b-hf"
# MODEL_NAME="Intern_VL_3_5"
# MODEL_PATH="./project/InternVL3_5-GPT-OSS-20B-A4B-Preview"
SPLIT="validation"
PROMPT="oe"
THEME="safety"

# Optional: your env/modules here
# module load cuda/12.1
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your-env

# Slurm usually sets CUDA_VISIBLE_DEVICES to the GPUs you were granted.
# We'll reuse your chunking logic as-is.
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

OUT_DIR_BASE="./output/${MODEL_NAME}"
TMP_DIR_BASE="${OUT_DIR_BASE}/tmp"
LOG_DIR="${OUT_DIR_BASE}/logs"
mkdir -p "$OUT_DIR_BASE" "$TMP_DIR_BASE" "$LOG_DIR"

TOPIC_LIST_FILE="${TOPIC_ROOT}/_topic_list.txt"
if [[ ! -s "$TOPIC_LIST_FILE" ]]; then
  echo "Missing topic list: $TOPIC_LIST_FILE (run 01_make_topics.sh)" >&2
  exit 1
fi

# Map SLURM_ARRAY_TASK_ID -> topic (1-based sed index)
topic="$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$TOPIC_LIST_FILE")"
if [[ -z "${topic:-}" ]]; then
  echo "Empty topic for task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

echo "=== [$(date)] Task ${SLURM_ARRAY_TASK_ID} Topic: ${topic} (split=${SPLIT}, chunks=${CHUNKS}) ==="

tmp_dir="${TMP_DIR_BASE}/${topic}"
mkdir -p "$tmp_dir"

# Ensure empty chunk files exist
for IDX in $(seq 0 $((CHUNKS-1))); do
  chunk_file="${tmp_dir}/${CHUNKS}_${IDX}_mmmu_${topic}_${SPLIT}.jsonl"
  [[ -f "$chunk_file" ]] || : > "$chunk_file"
done

# One python per GPU, same flags you used
pids=()
for IDX in $(seq 0 $((CHUNKS-1))); do
  CUDA_VISIBLE_DEVICES="${GPULIST[$IDX]}" \
  python -m run_model \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}" \
    --split "${SPLIT}" \
    --dataset "MMMU_${topic}" \
    --prompt "${PROMPT}" \
    --theme "${THEME}" \
    --answers_file "${tmp_dir}/${CHUNKS}_${IDX}_mmmu_${topic}_${SPLIT}.jsonl" \
    --num_chunks "${CHUNKS}" \
    --chunk_idx "${IDX}" \
    --temperature 0.0 \
    --top_p 0.9 \
    --num_beams 1 \
    > "${LOG_DIR}/${topic}_chunk${IDX}.log" 2>&1 &
  pids+=($!)
done

# Wait for all chunks
for pid in "${pids[@]}"; do
  wait "$pid"
done

# Concatenate into final per-topic file
out_file="${OUT_DIR_BASE}/mmmu_${topic}_${SPLIT}.jsonl"
: > "$out_file"
for IDX in $(seq 0 $((CHUNKS-1))); do
  cat "${tmp_dir}/${CHUNKS}_${IDX}_mmmu_${topic}_${SPLIT}.jsonl" >> "$out_file"
done

echo "=== [$(date)] Finished topic: ${topic} → ${out_file} ==="
