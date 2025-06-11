#!/bin/bash

# Usage
# LLsub ./submit_maze_training_4000.sh -s 20 -g volta:2

# Initialize and Load Modules
echo "[submit_maze_training.sh] Loading modules and virtual environment"
source /etc/profile
module load anaconda/2023b

# Assume current directory is gcs-diffusion
source .robodiff/bin/activate

# Set wandb to offline since Supercloud has no internet access
echo "[submit_maze_training.sh] Setting wandb to offline"
wandb offline

# Export date and time
DATE=`date +"%Y.%m.%d"`
TIME=`date +"%H.%M.%S"`

echo "[submit_maze_training.sh] Running training code..."
python train.py --config-dir=config --config-name=train_maze_diffusion_policy_cnn_4000.yaml \
    training.seed=42 hydra.run.dir=data/outputs/${DATE}/${TIME}_maze_image_4000 \
    task.dataset.zarr_path=data/maze_image/maze_image_dataset_4000.zarr
