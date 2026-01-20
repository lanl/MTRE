#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name mPLUG-Owl \
        --model_path ./liuhaotian/llava-v1.5-7b \
        --split testmini \
        --dataset Squares \
        --answers_file ./output/debug/tmp/${CHUNKS}_${IDX}_squares.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.9 \
        --num_beams 1 &
done

wait

output_file=./output/debug/Squares.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/debug/tmp/${CHUNKS}_${IDX}_squares.jsonl >> "$output_file"
    rm ./output/debug/tmp/${CHUNKS}_${IDX}.jsonl
done
