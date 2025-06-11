"""
Usage: From root directory, run
python data_generation/maze/generate_maze_data.py --config-name maze_data_generation.yaml
"""

import hydra
import numpy as np
import sys
import shutil
import pathlib
from omegaconf import OmegaConf

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath(
        'config'))
)
def main(cfg: OmegaConf):
    # Generate data
    OmegaConf.resolve(cfg)
    cls = hydra.utils.get_class(cfg._target_)
    maze_data_generation_workspace = cls(cfg=cfg)
    maze_data_generation_workspace.run()

if __name__ == "__main__":
    main()