#!/bin/bash

# Usage
# LLsub ./submit_data_generation.sh [2,40,1]

# Initialize and Load Modules
source /etc/profile
source .robodiff/bin/activate
module load anaconda/2023b
 
python data_generation/maze/generate_maze_data.py --config-name supercloud_maze_data_generation.yaml task_id=$LLSUB_RANK