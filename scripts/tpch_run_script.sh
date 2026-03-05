#!/bin/bash

# Function to run with timing - SYNCHRONOUS (wait for completion)
run_with_timing() {
    local seed=$1
    local logfile="logs/tpch_logs/220_queries_1gb_run${seed}.log"
    
    # Create log directory if it doesn't exist
    mkdir -p logs/tpch_logs
    
    # Debug: Check current directory and log path
    echo "Current directory: $(pwd)"
    echo "Log file path: $logfile"
    
    echo "Starting TPC-H run with seed ${seed} at $(date)" > $logfile
    echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')" >> $logfile
    echo "Start time (Unix): $(date +%s)" >> $logfile
    
    echo "Database recreated, starting optimization..." >> $logfile
    
    # Run synchronously (no nohup, no &) - wait for completion
    python -u run.py --db postgres --test tpch --timeout 100 -seed ${seed} >> $logfile 2>&1
    local exit_code=$?
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')" >> $logfile
    echo "End time (Unix): $(date +%s)" >> $logfile
    echo "Finished TPC-H run with seed ${seed} at $(date). Exit code: $exit_code" >> $logfile
    echo "Completed TPC-H run ${seed} with exit code $exit_code"
    
    # Save results after run completion
    local run_num=$2
    echo "Saving results for run ${run_num}..." >> $logfile
    ./scripts/save_results_reset.sh tpch ${run_num} >> $logfile 2>&1
    echo "Results saved for run ${run_num}" >> $logfile
}

# Kill any existing processes first
pkill -f "run.py" 2>/dev/null

echo "Starting sequential TPC-H optimization runs..."

# Run each optimization sequentially - one completes before the next starts
run_with_timing 1001 1
run_with_timing 2002 2
run_with_timing 3003 3
# run_with_timing 4004 4
# run_with_timing 5005 5

echo "All TPC-H runs completed! Check logs in logs/tpch_logs/"