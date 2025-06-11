#!/bin/bash

# Usage
# LLsub ./submit_training.sh -s 20 -g volta:1

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
echo "[submit_training.sh] Date: $DATE"
echo "[submit_training.sh] Time: $TIME"

CONFIG_DIR=config/planar_pushing/adversarial/gan_emb_dim/10_2000
CONFIG_NAME=10_2000_3_1.yaml
HYDRA_RUN_DIR=data/outputs/adversarial/gan_emb_dim/10_2000_3_1

python train.py --config-dir=$CONFIG_DIR --config-name=$CONFIG_NAME \
	hydra.run.dir=$HYDRA_RUN_DIR

# python train.py --config-dir=config/planar_pushing/adversarial --config-name=swap_labels_k_2.yaml \
# 	hydra.run.dir=data/outputs/adversarial/swap_labels/lambda_1.0_k_2
