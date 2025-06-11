import numpy as np
import hydra
import torch
import time
from tqdm import tqdm
import zarr
from torch.utils.data import DataLoader
import dill
import pathlib

from diffusion_policy.dataset.planar_pushing_dataset import PlanarPushingDataset
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.common.pytorch_util import dict_apply

def main():
    # Load dataset
    real_zarr_configs = [
        {
            'path': 'data/planar_pushing_cotrain/real_world_tee_data.zarr',
            'max_train_episodes': None,
            'sampling_weight': 1.0,
            'val_ratio': 0.0625
        }
    ]
    sim_zarr_configs = [
        {
            'path': 'data/planar_pushing_cotrain/sim_tee_data.zarr',
            'max_train_episodes': None,
            'sampling_weight': 1.0,
            'val_ratio': 0.0101
        }
    ]
    shape_meta = {
        'action': {'shape': [2]},
        'obs': {
            'agent_pos': {'type': 'low_dim', 'shape': [3]},
            'overhead_camera': {'type': 'rgb', 'shape': [3, 128, 128]},
            'wrist_camera': {'type': 'rgb', 'shape': [3, 128, 128]},
        },
    }
    real_dataset = PlanarPushingDataset(
        zarr_configs=real_zarr_configs,
        shape_meta=shape_meta,
        horizon=16,
        n_obs_steps=2,
        pad_before=1,
        pad_after=7,
        seed=42,
    )
    real_dataloader = DataLoader(
        real_dataset,
        batch_size=128,
        num_workers=2,
        persistent_workers=False,
        pin_memory=True,
        shuffle=False
    )
    sim_dataset = PlanarPushingDataset(
        zarr_configs=sim_zarr_configs,
        shape_meta=shape_meta,
        horizon=16,
        n_obs_steps=2,
        pad_before=1,
        pad_after=7,
        seed=42,
    )
    sim_dataloader = DataLoader(
        sim_dataset,
        batch_size=128,
        num_workers=2,
        persistent_workers=False,
        pin_memory=True,
        shuffle=False
    )
    print("Finished loading datasets")

    # Load model
    checkpoint = "/home/adam/workspace/gcs-diffusion/data/outputs/cotrain/real_only/10/checkpoints/latest.ckpt"
    checkpoint = pathlib.Path(checkpoint)
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

    # get policy from workspace
    policy = workspace.model
    policy.set_normalizer(normalizer)
    if cfg.training.use_ema:
        policy = workspace.ema_model
        policy.set_normalizer(normalizer)
    device = torch.device("cuda")
    policy.to(device)
    policy.eval()

    # Compute embeddings
    embedding_dim = 131*2
    real_embeddings = []
    with torch.no_grad():
        with tqdm(real_dataloader, desc=f"Real data") as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))            
                embedding = policy.compute_obs_embedding(batch)
                for i in range(embedding.shape[0]):
                    real_embeddings.append(embedding[i].cpu().numpy())

    # Save embeddings
    real_embeddings = np.array(real_embeddings)
    np.save("real_embeddings.npy", real_embeddings)
    del real_embeddings

    sim_embeddings = []
    with torch.no_grad():
        with tqdm(sim_dataloader, desc=f"Sim data") as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))            
                embedding = policy.compute_obs_embedding(batch)
                for i in range(embedding.shape[0]):
                    sim_embeddings.append(embedding[i].cpu().numpy())
    
    # Save embeddings
    sim_embeddings = np.array(sim_embeddings)
    np.save("sim_embeddings.npy", sim_embeddings)
    del sim_embeddings   
        

if __name__ == '__main__':
    main()