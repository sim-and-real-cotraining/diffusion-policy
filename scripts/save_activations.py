import numpy as np
import hydra
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import dill
import pathlib
import os

from diffusion_policy.dataset.planar_pushing_dataset import PlanarPushingDataset
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset

ACTIVATIONS = []

def hook_fn(module, input, output):
    global ACTIVATIONS
    ACTIVATIONS.append(input[0].detach().cpu().numpy())

def save_activations(checkpoint_dir: str, save_dir: str):
    global ACTIVATIONS

    # Create save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    checkpoint = pathlib.Path(f"{checkpoint_dir}/checkpoints/latest.ckpt")
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    # Get workspace
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace
    workspace = cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # Get normalizer
    normalizer_path = checkpoint.parent.parent.joinpath("normalizer.pt")
    normalizer = torch.load(normalizer_path)

    # Get policy from workspace
    policy = workspace.model
    policy.set_normalizer(normalizer)
    if cfg.training.use_ema:
        policy = workspace.ema_model
        policy.set_normalizer(normalizer)
    device = torch.device("cuda")
    policy.to(device)
    policy.eval()
    policy.register_forward_hook(hook_fn)

    target_module_name = "module.model.final_conv.1"
    for name, module in policy.named_modules():
        if name == target_module_name:
            module.register_forward_hook(hook_fn)
            print(f"Hook registered to module: {name}")

    # Get datasets
    zarr_configs = cfg.task.dataset.zarr_configs
    dataset_config = cfg.task.dataset
    datasets = {}
    for zarr_config in zarr_configs:
        dataset_config['zarr_configs'] = [zarr_config]
        if 'real_world_tee_data.zarr' in zarr_config['path']:
            datasets['real'] = hydra.utils.instantiate(dataset_config)
        else:
            # NOTE: to save RAM, only load 500 trajectories
            dataset_config['zarr_configs'][0]['max_train_episodes'] = 200
            datasets['sim'] = hydra.utils.instantiate(dataset_config)
    
    # Get dataloaders
    batch_size = 128
    real_dataloader = DataLoader(
        datasets['real'],
        batch_size=batch_size,
        num_workers=4,
        persistent_workers=False,
        pin_memory=True,
        shuffle=False
    )
    sim_dataloader = DataLoader(
        datasets['sim'],
        batch_size=batch_size,
        num_workers=4,
        persistent_workers=False,
        pin_memory=True,
        shuffle=False
    )

    timesteps = np.array([90, 80, 70, 60, 50, 40, 30, 20, 10, 0])
    np.save(f"{save_dir}/ddim_timesteps.npy", timesteps)
    with torch.no_grad():
        with tqdm(real_dataloader, desc=f"Real data") as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                policy.predict_action(batch, use_DDIM=True)
                assert len(ACTIVATIONS) % len(timesteps) == 0
    ACTIVATIONS = np.vstack(ACTIVATIONS)
    np.save(f"{save_dir}/real_activations.npy", ACTIVATIONS)
    split_array_and_save(ACTIVATIONS, timesteps, f'{save_dir}/activations_per_noise_level', 'real_activations')
    ACTIVATIONS = []

    with torch.no_grad():        
        with tqdm(sim_dataloader, desc=f"Sim data") as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                policy.predict_action(batch, use_DDIM=True)
                assert len(ACTIVATIONS) % len(timesteps) == 0
    ACTIVATIONS = np.vstack(ACTIVATIONS)
    np.save(f"{save_dir}/sim_activations.npy", ACTIVATIONS)
    split_array_and_save(ACTIVATIONS, timesteps, f'{save_dir}/activations_per_noise_level', 'sim_activations')
    ACTIVATIONS = []

def split_array_and_save(input_array, timestamps, save_dir, root_filename):
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    num_timesteps = len(timestamps)
    for i in range(num_timesteps):
        small_array = input_array[i::num_timesteps]
        save_path = os.path.join(save_dir, f'{root_filename}_{int(timestamps[i])}.npy')
        np.save(save_path, small_array)
        print(f"Saved: {save_path}")

def main():
    experiments = [
        # "scaled_mixing/50_500_3_1",
        # "scaled_mixing/50_2000_3_1",
        # "scaled_mixing/10_2000_3_1",
        # "scaled_mixing/10_2000",
        "scaled_mixing/10_500",
    ]
    for experiment in experiments:
        print("Saving activations for", experiment)
        save_activations(
            f"data/outputs/cotrain/{experiment}",
            f"data/experiments/cotrain/{experiment}"
        )

if __name__ == '__main__':
    main()