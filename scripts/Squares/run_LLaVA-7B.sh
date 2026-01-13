#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name LLaVA-7B \
        --model_path ./LLaVA/checkpoints/liuhaotian/llava-v1.5-7b \
        --split testmini \
        --dataset Squares \
        --answers_file ./output/tmp/${CHUNKS}_${IDX}_l_squares.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.0 \
        --token_id 2 \
        --num_beams 1 &
done

wait

output_file=./output/LLaVA-7B/Squares.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/tmp/${CHUNKS}_${IDX}_l_squares.jsonl >> "$output_file"
done
