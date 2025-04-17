#!/bin/bash

#SBATCH --job-name=PixiGaze-G14-Pre     # Job name (adjust for stage/fold)
#SBATCH --output=slurm_logs/pixigaze_%A_%a.out  # Standard output log (%A=job id, %a=task id)
#SBATCH --error=slurm_logs/pixigaze_%A_%a.err   # Standard error log
#SBATCH --partition=YOUR_A100_PARTITION # !!! IMPORTANT: Specify your A100 partition name !!!
#SBATCH --gres=gpu:a100:1             # Request 1 A100 GPU
#SBATCH --nodes=1                     # Run on a single node
#SBATCH --ntasks=1                    # Run a single task (the python script)
#SBATCH --cpus-per-task=16            # Request CPUs (adjust based on data loading needs)
#SBATCH --mem=128G                    # Request memory (A100s often have lots, ViT-G needs it)
#SBATCH --time=2-00:00:00             # Max wall time (2 days, adjust as needed)
#SBATCH --array=0-0                   # Default: Run a single job (task_id 0). Change for MIT stages e.g., 0-9

# --- Environment Setup ---
mkdir -p slurm_logs # Create logs directory if it doesn't exist

echo "========================================================"
echo "SLURM JOB: $SLURM_JOB_ID, TASK: $SLURM_ARRAY_TASK_ID"
echo "Running on host: $(hostname)"
echo "Running on node: $SLURMD_NODENAME"
echo "Allocate GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "Using Pixi environment."
echo "========================================================"

module purge # Clear existing modules (recommended)
# !!! IMPORTANT: Load required modules for your environment !!!
# Examples (adjust to your cluster's setup):
# module load cuda/11.8         # Or the CUDA version compatible with your PyTorch
# module load cudnn/8.6.0       # Optional, often included with CUDA module
# module load pixi              # Load Pixi if it's installed as a module

# Activate your conda environment
# !!! PIXI CHANGE: No 'source activate'. Pixi manages the environment via 'pixi run' !!!

# Optional: Set environment variables
# export WANDB_MODE=offline # If using Weights & Biases, run offline initially
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 # Helps prevent fragmentation

# Navigate to the directory containing pixi.toml and train_dinogaze.py
# !!! IMPORTANT: Adjust path if needed !!!
cd /path/to/your/project/directory # Where pixi.toml and train_dinogaze.py are

# --- Set parameters based on the desired stage ---
STAGE="salicon_pretrain" # Options: salicon_pretrain, mit_spatial, mit_scanpath_frozen, mit_scanpath_full
MODEL_NAME="dinov2_vitg14"
LAYERS="-3 -2 -1" # As a string, argparse will handle it
BATCH_SIZE=4      # Start low for ViT-G!
LR=0.0005
MIN_LR=1e-7
NUM_WORKERS=16 # Match --cpus-per-task if possible
TRAIN_DIR="./train_dinogaze_vitg_${STAGE}" # Stage-specific training dir
DATASET_DIR="./pysaliency_datasets"
LMDB_DIR="./lmdb_cache_dinogaze_vitg"

# --- Stage specific adjustments ---
FOLD_ARG=""
if [[ "$STAGE" == "mit_spatial" || "$STAGE" == "mit_scanpath_frozen" || "$STAGE" == "mit_scanpath_full" ]]; then
    # For MIT stages, use the SLURM array task ID as the fold
    # Make sure to change the #SBATCH --array directive above (e.g., --array=0-9)
    FOLD_ARG="--fold $SLURM_ARRAY_TASK_ID"
    TRAIN_DIR="${TRAIN_DIR}_fold${SLURM_ARRAY_TASK_ID}" # Add fold to output dir
    # Adjust LR for final scanpath stage
    if [[ "$STAGE" == "mit_scanpath_full" ]]; then
        LR=1e-5
    fi
    # Adjust batch size for MIT if needed (maybe smaller dataset allows slightly larger?)
    # BATCH_SIZE=8
fi

# Ensure the training directory exists
mkdir -p $TRAIN_DIR
mkdir -p $LMDB_DIR

# --- Construct the python command arguments ---
# Note: We build the command string *without* 'python' initially,
# because 'pixi run' will prepend the correct python interpreter from the environment.
PYTHON_ARGS="train_dinogaze.py \
  --stage $STAGE \
  --model_name $MODEL_NAME \
  --layers $LAYERS \
  --batch_size $BATCH_SIZE \
  --lr $LR \
  --min_lr $MIN_LR \
  $FOLD_ARG \
  --num_workers $NUM_WORKERS \
  --train_dir $TRAIN_DIR \
  --dataset_dir $DATASET_DIR \
  --lmdb_dir $LMDB_DIR"

# --- Run the Python script using Pixi ---
echo "Running command via Pixi:"
echo "pixi run python $PYTHON_ARGS"
echo "--------------------------------------------------------"

# Execute the command within the pixi environment
# 'pixi run' ensures dependencies are met and uses the environment's python
pixi run python $PYTHON_ARGS

EXIT_STATUS=$?
echo "--------------------------------------------------------"
echo "Pixi command finished with status: $EXIT_STATUS"
echo "End time: $(date)"
echo "========================================================"

exit $EXIT_STATUS