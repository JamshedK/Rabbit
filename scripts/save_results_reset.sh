#!/bin/bash

# Script to save run results and clean up for next experiment
# Usage: ./save_results_reset.sh <benchmark> <run_number>
# Example: ./save_results_reset.sh tpcc 1

if [ $# -ne 2 ]; then
    echo "Usage: $0 <benchmark> <run_number>"
    echo "Example: $0 tpcc 1"
    exit 1
fi

BENCHMARK=$1
RUN_NUM=$2
DBMS="postgres"

# Create directory structure
RESULTS_DIR="final_data/${DBMS}/${BENCHMARK}/run_${RUN_NUM}"
mkdir -p "${RESULTS_DIR}"

echo "Saving results for ${BENCHMARK} run ${RUN_NUM} to ${RESULTS_DIR}"

# Move files to results directory
if [ -f "knowledge/${DBMS}/init_configs_perfs_${BENCHMARK}.json" ]; then
    mv "knowledge/${DBMS}/init_configs_perfs_${BENCHMARK}.json" "${RESULTS_DIR}/"
    echo "  ✓ Saved init_configs_perfs_${BENCHMARK}.json"
fi

if [ -f "knowledge/${DBMS}/key_knobs_${BENCHMARK}_task1_all_sorted.txt" ]; then
    mv "knowledge/${DBMS}/key_knobs_${BENCHMARK}_task1_all_sorted.txt" "${RESULTS_DIR}/"
    echo "  ✓ Saved key_knobs_${BENCHMARK}_task1_all_sorted.txt"
fi

if [ -f "knowledge/${DBMS}/key_knobs_${BENCHMARK}_task1.txt" ]; then
    mv "knowledge/${DBMS}/key_knobs_${BENCHMARK}_task1.txt" "${RESULTS_DIR}/"
    echo "  ✓ Saved key_knobs_${BENCHMARK}_task1.txt"
fi

if [ -f "knowledge/${DBMS}/suggested_knobs_value/suggested_knobs_value_${BENCHMARK}_task1.json" ]; then
    mv "knowledge/${DBMS}/suggested_knobs_value/suggested_knobs_value_${BENCHMARK}_task1.json" "${RESULTS_DIR}/"
    echo "  ✓ Saved suggested_knobs_value_${BENCHMARK}_task1.json"
fi

echo "Results saved. Ready for next run."
