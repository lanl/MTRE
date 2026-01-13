#!/usr/bin/env bash
set -euo pipefail

# Configuration - set these environment variables or use defaults
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
MODEL_PATH="${MODEL_PATH_INTERN_VL:-/path/to/InternVL3_5-GPT-OSS-20B-A4B-Preview}"
TOPIC_ROOT="${TOPIC_ROOT:-/path/to/MMMU}"

# --- Configuration ---
MODEL_NAME="Intern_VL_3_5"
SPLIT="validation"
PROMPT="oe"
THEME="safety"

# GPUs: comma-separated list in CUDA_VISIBLE_DEVICES or default to "0"
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
CHUNKS=${#GPULIST[@]}

OUT_DIR_BASE="${OUTPUT_DIR}/${MODEL_NAME}"
TMP_DIR_BASE="${OUT_DIR_BASE}/tmp"
LOG_DIR="${OUT_DIR_BASE}/logs"
mkdir -p "$OUT_DIR_BASE" "$TMP_DIR_BASE" "$LOG_DIR"

run_one_topic() {
  local topic="$1"
  echo "=== [$(date)] Starting topic: ${topic} (${SPLIT}, chunks=${CHUNKS}) ==="

  local tmp_dir="${TMP_DIR_BASE}/${topic}"
  mkdir -p "$tmp_dir"

  # Ensure empty chunk files exist
  for IDX in $(seq 0 $((CHUNKS-1))); do
    local chunk_file="${tmp_dir}/${CHUNKS}_${IDX}_mmmu_${topic}_${SPLIT}.jsonl"
    [[ -f "$chunk_file" ]] || touch "$chunk_file"
  done

  # Run one process per GPU
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

  # Wait for all chunks to complete
  for pid in "${pids[@]}"; do
    wait "$pid"
  done

  # Concatenate chunks
  local out_file="${OUT_DIR_BASE}/mmmu_${topic}_${SPLIT}.jsonl"
  : > "$out_file"
  for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "${tmp_dir}/${CHUNKS}_${IDX}_mmmu_${topic}_${SPLIT}.jsonl" >> "$out_file"
  done

  echo "=== [$(date)] Finished topic: ${topic} → ${out_file} ==="
}

# --- Run over all topic folders ---
for dir in "${TOPIC_ROOT}"/*/; do
  topic="$(basename "${dir%/}")"
  case "$topic" in
    _img_cache|README.md|Accounting|Electronics|Agriculture|Energy_and_Power|Economics|Architecture_and_Engineering|Art|Art_Theory|Basic_Medical_Science|Biology|Chemistry|Clinical_Medicine|Computer_Science|Design|Diagnostics_and_Laboratory_Medicine) continue ;;
  esac
  run_one_topic "$topic"
done

echo "All topics complete."


# Accounting                    Biology                              Economics         _img_cache  Mechanical_Engineering  README.md
# Agriculture                   Chemistry                            Electronics       Literature  Music                   Sociology
# Architecture_and_Engineering  Clinical_Medicine                    Energy_and_Power  Manage      Pharmacy
# Art                           Computer_Science                     Finance           Marketing   Physics
# Art_Theory                    Design                               Geography         Materials   Psychology
# Basic_Medical_Science         Diagnostics_and_Laboratory_Medicine  History           Math        Public_Health
