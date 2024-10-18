#!/bin/bash

# Function to handle Ctrl+C and terminate all child processes
cleanup() {
    echo "Caught Ctrl+C! Terminating all jobs..."
    kill 0  # This will send a kill signal to all child processes in the current process group
    exit 1
}
# Trap Ctrl+C signal (SIGINT) and call the cleanup function
trap cleanup SIGINT

N_WORKERS=$1

for ((i=0; i<N_WORKERS; i++))
do
  GPU_ID=$((i % 8))
  echo Running worker $WORKER_ID on $GPU_ID
  CUDA_VISIBLE_DEVICES=$GPU_ID WORKER_ID=$i N_WORKERS=$N_WORKERS python train.py &
done

wait
echo "All training jobs completed."