#!/bin/bash

# Configuration - set these environment variables or use defaults
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
MODEL_PATH="${MODEL_PATH_INTERN_VL:-/path/to/InternVL3_5-GPT-OSS-20B-A4B-Preview}"

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name Intern_VL_3_5 \
        --model_path ${MODEL_PATH} \
        --split SD_TYPO \
        --dataset MME \
        --prompt oe \
        --theme safety \
        --answers_file ${OUTPUT_DIR}/Intern_VL_3_5/tmp/${CHUNKS}_${IDX}_mme.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=${OUTPUT_DIR}/Intern_VL_3_5/mme.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${OUTPUT_DIR}/Intern_VL_3_5/tmp/${CHUNKS}_${IDX}_mme.jsonl >> "$output_file"
done
