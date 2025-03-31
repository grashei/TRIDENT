#!/bin/bash

# Usage: ./run.sh --cpus 28 --compile --gpus 0 1 2 3 4 5 6 7 --ids files.txt --slide_dir /mnt/data/nfs03-R6/oskar/slides --h5_dir /mnt/data/nfs03-R6/oskar/patches --feat_dir /mnt/data/nfs03-R6/oskar/features --batch_size 512 --resize 224

# Determine the directory where the script is located
SCRIPT_PATH=$(readlink -f "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")

# Change the current working directory to the script's directory
cd "$SCRIPT_DIR" || exit 1

# Default configurations
CPUS=$(nproc)  # Default to all available CPUs
USE_TASKSET=0
TASK="feat"
PATCH_DIR="slides"
JOB_DIR="./trident_processed"
PATCH_ENCODER="uni_v2,conch_v1,virchow2"
MAG="40"
PATCH_SIZE="512"

GPUS=($(seq 0 $(($(nvidia-smi -L | wc -l)-1))))  # Default to all available GPUs

while (( "$#" )); do
  case "$1" in
    --cpus)
      CPUS=$2
      echo "Setting CPUs per process to $CPUS"
      shift 2
      ;;
    --patch_dir)
      PATCH_DIR=$2  # Set CPUs per process/run
      echo "Setting patch directory to $PATCH_DIR"
      shift 2
      ;;
    --slide_ids)
      TXT_FILE=$2
      shift 2
      ;;
    --patch_encoder)
      PATCH_ENCODER=$2
      echo "Setting patch encoder to $PATCH_ENCODER"
      shift 2
      ;;
    --mag)
      MAG=$2
      echo "Setting magnification to $MAG"
      shift 2
      ;;
    --patch_size)
      PATCH_SIZE=$2
      shift 2
      ;;
    *)  # If an unknown option is passed
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
  esac
done


# Validate required inputs
if [[ -z "$PATCH_DIR" || -z "$TXT_FILE" ]]; then
  echo "Error: Missing required arguments. See usage." >&2
  exit 1
fi

echo "Using $CPUS CPUs per process."

# Read slide IDs into an array and remove duplicates
readarray -t SLIDE_IDS < <(sort -u "$TXT_FILE")

# Calculate split parameters
TOTAL_SLIDES=${#SLIDE_IDS[@]}
NUM_GPUS=${#GPUS[@]}
TOTAL_PROCESSES=$((NUM_GPUS))

echo "Found $TOTAL_SLIDES slides"

# Create output directory
SPLIT_DIR="./split_txt"
mkdir -p "$SPLIT_DIR"

# Initialize output files
PART_FILES=()
for GPU_ID in "${GPUS[@]}"; do
    PART_FILE="$SPLIT_DIR/part_gpu${GPU_ID}.txt"
    PART_FILES+=("$PART_FILE")
    # Clear the file to ensure no leftover data
    : > "$PART_FILE"
done

# Distribute slide IDs evenly across the output files
for ((i = 0; i < TOTAL_SLIDES; i++)); do
  PART_INDEX=$((i % TOTAL_PROCESSES))
  echo "${SLIDE_IDS[i]}" >> "${PART_FILES[PART_INDEX]}"
done

echo "Slide IDs split into ${TOTAL_PROCESSES} parts in $SPLIT_DIR."

# Function to run inference for each part
run_inference() {
  GPU=$1
  PART_FILE="$SPLIT_DIR/part_gpu${GPU}.txt"

  CMD="python run_batch_of_slides.py --slide_ids $PART_FILE --task $TASK --patch_dir $PATCH_DIR --job_dir $JOB_DIR --patch_encoder $PATCH_ENCODER --mag $MAG --patch_size $PATCH_SIZE"

  echo "Starting inference on GPU $GPU with $PART_FILE"

  if [[ $USE_TASKSET -eq 1 ]]; then
    # ----------------------------------------
    # CPU Affinity Calculation
    # ----------------------------------------
    
    # Define NUMA nodes and their CPU sets
    # We flatten them into arrays for easy indexing
    NODE0_CPUS=($(seq 0 55) $(seq 112 167))
    NODE1_CPUS=($(seq 56 111) $(seq 168 223))

    # Calculate total number of tasks
    TOTAL_TASKS=$(( NUM_GPUS ))

    # If you assume tasks are evenly distributed across the two nodes:
    # For example, if TOTAL_TASKS=8, half_for_node=4
    HALF_TASKS=$(( TOTAL_TASKS / 2 ))

    # Identify the global task index: 
    # We can assume tasks are numbered by GPU and P.
    # For instance, if tasks are launched in a for loop:
    # GPU=0..(NUM_GPUS-1), P=0..(PROCESSES_PER_GPU-1)
    # Then global_task_index = GPU * PROCESSES_PER_GPU + P
    GLOBAL_TASK_INDEX=$(( GPU ))

    # Determine which NUMA node this task should use
    if (( GLOBAL_TASK_INDEX < HALF_TASKS )); then
      # Use NODE0
      START_CPU_INDEX=$(( GLOBAL_TASK_INDEX * CPUS ))
      END_CPU_INDEX=$(( START_CPU_INDEX + CPUS - 1 ))

      START_CPU=${NODE0_CPUS[$START_CPU_INDEX]}
      END_CPU=${NODE0_CPUS[$END_CPU_INDEX]}
    else
      # Use NODE1
      NODE1_TASK_INDEX=$(( GLOBAL_TASK_INDEX - HALF_TASKS ))
      START_CPU_INDEX=$(( NODE1_TASK_INDEX * CPUS ))
      END_CPU_INDEX=$(( START_CPU_INDEX + CPUS - 1 ))

      START_CPU=${NODE1_CPUS[$START_CPU_INDEX]}
      END_CPU=${NODE1_CPUS[$END_CPU_INDEX]}
    fi

    CPU_RANGE="$START_CPU-$END_CPU"
    echo "Task $GLOBAL_TASK_INDEX: Using CPUs $CPU_RANGE"

    # ----------------------------------------
    # End of CPU Affinity Calculation
    # ----------------------------------------
    CMD="taskset -c $CPU_RANGE $CMD"
  fi

  if [[ $VERBOSE -eq 1 ]]; then
    OUTPUT_FILE="./split_txt/output_gpu${GPU}_proc${P}.log"
    echo "Logs redirected to $OUTPUT_FILE"
    echo "Running: $CMD" > "$OUTPUT_FILE"
    CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES=$GPU HF_TOKEN=$HF_TOKEN $CMD >> "$OUTPUT_FILE" 2>&1
  else
    echo "Running: $CMD"
    CUDA_DEVICE_ORDER="PCI_BUS_ID" CUDA_VISIBLE_DEVICES=$GPU HF_TOKEN=$HF_TOKEN $CMD
  fi
}

# Export function and variables for parallel processing
export -f run_inference
export SPLIT_DIR USE_TASKSET NUM_GPUS SLIDE_IDS TASK PATCH_DIR JOB_DIR PATCH_ENCODER MAG PATCH_SIZE

# Launch all processes in parallel
for GPU in "${GPUS[@]}"; do
    bash -c "run_inference $GPU" &
done

wait

echo "All processes launched."
