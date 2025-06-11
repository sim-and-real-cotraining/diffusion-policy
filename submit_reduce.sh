#!/bin/bash

# Usage
# LLsub ./submit_reduce.sh [1,1,1]

# Initialize and Load Modules
source /etc/profile
module load anaconda/2023b

python data_generation/maze/convert_pkl_to_zarr.py \
    --data_dir data_generation/maze_data
mv data_generation/maze_data/maze_image_dataset.zarr \
    data/maze_image/maze_image_dataset.zarr
