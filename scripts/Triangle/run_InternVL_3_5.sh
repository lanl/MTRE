#!/bin/bash

# Configuration - set these environment variables or use defaults
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
MODEL_PATH="${MODEL_PATH_INTERN_VL:-/path/to/InternVL3_5-GPT-OSS-20B-A4B-Preview}"

topic="Triangle"
split="validation"
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
tmp_dir="${OUTPUT_DIR}/Intern_VL_3_5/tmp"

# Ensure the tmp directory exists
mkdir -p "$tmp_dir"

# Create empty chunk files if they don't exist
for IDX in $(seq 0 $((CHUNKS-1))); do
    file="${tmp_dir}/${CHUNKS}_${IDX}_triangle.jsonl"
    if [ ! -f "$file" ]; then
        echo "Creating missing file: $file"
        touch "$file"
    fi
done

# Run model processes
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name Intern_VL_3_5 \
        --model_path ${MODEL_PATH} \
        --split "${split}" \
        --dataset Triangle \
        --prompt oe \
        --theme safety \
        --answers_file "${tmp_dir}/${CHUNKS}_${IDX}_triangle.jsonl" \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file="${OUTPUT_DIR}/Intern_VL_3_5/Triangle.jsonl"

# Clear out the output file if it exists
> "$output_file"

# Concatenate all chunk files
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "${tmp_dir}/${CHUNKS}_${IDX}_triangle.jsonl" >> "$output_file"
done
