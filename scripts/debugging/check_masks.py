import numpy as np
import torch
import hydra
import pathlib

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    # Get workspace
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    
    # Get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    device = torch.device("cuda")
    policy.to(device)
    policy.eval()

    zarr_configs = cfg.task.dataset.zarr_configs
    dataset_config = cfg.task.dataset
    for i in range(len(zarr_configs)):
        dataset_config['zarr_configs'] = [zarr_configs[i]]
        dataset = hydra.utils.instantiate(dataset_config)
        train_mask = dataset.train_masks[0]
        if 'real_world_tee_data.zarr' in zarr_configs[i]['path']:
            assert np.allclose(train_mask, np.load('data/experiments/dataset_mask/train_masks_0.npy'))
            print(f"Train mask {i} is correct")
        else:
            assert np.allclose(train_mask, np.load('data/experiments/dataset_mask/train_masks_1.npy'))
            print(f"Train mask {i} is correct")

if __name__ == '__main__':
    main()