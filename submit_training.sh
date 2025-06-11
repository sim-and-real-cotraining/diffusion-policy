#!/bin/bash

# Usage
# LLsub ./submit_training.sh -s 20 -g volta:2

# Initialize and Load Modules
echo "[submit_training.sh] Loading modules and virtual environment"
source /etc/profile
module load anaconda/2023b

# Assume current directory is gcs-diffusion
source .robodiff/bin/activate

# Set wandb to offline since Supercloud has no internet access
echo "[submit_training.sh] Setting wandb to offline"
wandb offline

# Export date, time, environment variables
DATE=`date +"%Y.%m.%d"`
TIME=`date +"%H.%M.%S"`
HYDRA_FULL_ERROR=1

echo "[submit_training.sh] Running training code..."
# python train.py --config-dir=config --config-name=train_pusher_diffusion_policy_cnn.yaml \
#     hydra.run.dir=data/outputs/push_tee_v1_sc/ \
#     task.dataset.zarr_path=data/planar_pushing/push_tee_hybrid_dataset.zarr

python train.py --config-dir=config/underactuated --config-name=sim_750_real_50.yaml