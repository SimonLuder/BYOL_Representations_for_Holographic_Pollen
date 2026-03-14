#!/bin/bash  
#SBATCH -t 3-4:00:00 
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --hint=nomultithread                # disable hyperthreading
#SBATCH --job-name=BYOL_pollen_L
#SBATCH --output=BYOL_%j_L.outs
#SBATCH --error=BYOL_%j_L.err

# Exit immediately if any command fails
set -e

# PATH CONFIGURATION ==================================================================

REPO_PATH=/mnt/nas05/data01/simon_luder/BYOL/BYOL_Representations_for_Holographic_Pollen
SIF_PATH=/mnt/nas05/data01/simon_luder/BYOL/byol_training.sif
CONFIG_PATH=/app/config/slurm_byol_dual_config_L.yaml

# LOAD ENVIRONMENT VARIABLES (.env) ===================================================

ENV_FILE="$REPO_PATH/.env"

if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from .env"
    set -a
    source "$ENV_FILE"
    set +a
else
    echo "ERROR: .env file not found at $ENV_FILE"
    exit 1
fi

# Forward environment variables to the Singularity container
export SINGULARITYENV_WANDB_API_KEY="$WANDB_API_KEY"

# NCCL error handling for distributed training robustness
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1


# UPDATE REPOSITORY  ================================================================

echo "Updating repository..."
cd "$REPO_PATH"

git fetch origin
git pull origin main


# RUN TRAINING INSIDE SINGULARITY CONTAINER =========================================

echo "Starting training inside Singularity container..."

singularity exec --pwd /app \
    -B "$REPO_PATH:/app" \
    -B /mnt/nas05/data01/marvel/marvel-fhnw/data:/app/data \
    -B /mnt/nas05/data01/simon_luder/Data_Setup/Pollen_Datasets/data/final:/app/data/final \
    --nv "$SIF_PATH" \
    python3 /app/tools/train_lightning.py --config "$CONFIG_PATH"

echo "Training job completed."