#!/bin/bash

# Configuration - set these environment variables or use defaults
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
MODEL_PATH="${MODEL_PATH_LLAVA_NEXT:-/path/to/llava-v1.6-34b-hf}"

topic="Physics"
split="dev"
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}
tmp_dir="${OUTPUT_DIR}/LLaVA_NeXT/tmp"

# Ensure the tmp directory exists
mkdir -p "$tmp_dir"

# Create empty chunk files if they don't exist
for IDX in $(seq 0 $((CHUNKS-1))); do
    file="${tmp_dir}/${CHUNKS}_${IDX}_mmmu_${topic}_${split}.jsonl"
    if [ ! -f "$file" ]; then
        echo "Creating missing file: $file"
        touch "$file"
    fi
done

# Run model processes
for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name LLaVA_NeXT \
        --model_path ${MODEL_PATH} \
        --split "${split}" \
        --dataset MMMU_${topic} \
        --prompt oe \
        --theme safety \
        --answers_file "${tmp_dir}/${CHUNKS}_${IDX}_mmmu_${topic}_${split}.jsonl" \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file="${OUTPUT_DIR}/LLaVA_NeXT/mmmu_${topic}_${split}.jsonl"

# Clear out the output file if it exists
> "$output_file"

# Concatenate all chunk files
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat "${tmp_dir}/${CHUNKS}_${IDX}_mmmu_${topic}_${split}.jsonl" >> "$output_file"
done
