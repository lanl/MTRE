#!/usr/bin/env bash
set -euo pipefail

# --- Match your config ---
TOPIC_ROOT="./project/MMMU"

# Build a clean, stable list of topics (exclude cache/readme)
find "${TOPIC_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" \
 | grep -vE '^(_img_cache|README\.md)$' \
 | sort -u > "${TOPIC_ROOT}/_topic_list.txt"

wc -l "${TOPIC_ROOT}/_topic_list.txt" | awk '{print "Topics:", $1}'
echo "List saved to: ${TOPIC_ROOT}/_topic_list.txt"
