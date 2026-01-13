#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name LLaMA_Adapter \
        --split testmini \
        --dataset Lines \
        --answers_file ./output/LLaMA_Adapter/tmp/${CHUNKS}_${IDX}_lines.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --top_p 0.0 \
        --num_beams 1 &
done

wait

output_file=./output/LLaMA_Adapter/Lines.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./output/LLaMA_Adapter/tmp/${CHUNKS}_${IDX}_lines.jsonl >> "$output_file"
    rm ./output/LLaMA_Adapter/tmp/${CHUNKS}_${IDX}_lines.jsonl
done
