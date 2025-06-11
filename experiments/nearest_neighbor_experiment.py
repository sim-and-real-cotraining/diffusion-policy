import numpy as np
import time
from tqdm import tqdm
import zarr

from diffusion_policy.dataset.planar_pushing_dataset import PlanarPushingDataset

def brute_force_nearest_neighbor_search(
        pose_query, overhead_query, wrist_query, 
        pose_dataset, overhead_dataset, wrist_dataset,
        alpha=0.5
    ):
    """
    TODO:
    - Test numpy implementation (linalg.norm on query and img and then weighted sum, then min)
    Note: For now, need to use brute force implementation since I am training a model
    and have RAM constraints
    """
    # Test numpy implementation
    pose_query = pose_query.flatten()
    overhead_query = overhead_query.flatten()
    wrist_query = wrist_query.flatten()
    
    min_dist = np.inf
    min_idx = -1
    for i in tqdm(range(len(pose_dataset))):
        pose = pose_dataset[i]
        overhead_camera = overhead_dataset[i]
        wrist_camera = wrist_dataset[i]
        dist = alpha*np.linalg.norm(pose - pose_query) + \
                (1-alpha)*np.linalg.norm(overhead_camera - overhead_query) / 2.0 + \
                (1-alpha)*np.linalg.norm(wrist_camera - wrist_query) / 2.0
        if dist < min_dist:
            min_dist = dist
            min_idx = i
    return min_idx, min_dist


def main():
    # Load dataset
    zarr_configs = [
        {
            'path': 'data/planar_pushing_cotrain/real_world_tee_data.zarr',
            'max_train_episodes': None,
            'sampling_weight': 1.0,
            'val_ratio': 0.0625
        },
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
    dataset = PlanarPushingDataset(
        zarr_configs=zarr_configs,
        shape_meta=shape_meta,
        horizon=16,
        n_obs_steps=2,
        pad_before=1,
        pad_after=7,
        seed=42,
    )
    print("Finished loading dataset")

    real_sampler = dataset.samplers[0]
    sim_sampler = dataset.samplers[1]

    print(dataset.replay_buffers[0].episode_ends)
    print(dataset.replay_buffers[1].episode_ends)
    breakpoint()

    random_pose_obs = np.random.rand(6).reshape(2,3)
    random_overhead_obs = np.random.rand(2,128,128,3)
    random_wrist_obs = np.random.rand(2,128,128,3)
    # TODO: make sure order of dims is correct

    real_pose_dataset = zarr.zeros(
        (len(real_sampler), 2*3), chunks=(2048,2*3),
    )
    real_overhead_dataset = zarr.zeros(
        (len(real_sampler), 2*128*128*3),
        chunks=(2048, 2*128*128*3),
        dtype="u1",
    )
    real_wrist_dataset = zarr.zeros(
        (len(real_sampler), 2*128*128*3),
        chunks=(2048, 2*128*128*3),
        dtype="u1",
    )
    
    for i in tqdm(range(len(real_sampler))):
        sample = real_sampler.sample_sequence(i)
        pose = sample['state'][:2].flatten()
        overhead_camera = sample['overhead_camera'][:2].flatten()
        wrist_camera = sample['wrist_camera'][:2].flatten()
        real_pose_dataset[i] = pose
        real_overhead_dataset[i] = overhead_camera
        real_wrist_dataset[i] = wrist_camera

    
    start = time.time()
    idx, dist = brute_force_nearest_neighbor_search(
        random_pose_obs, random_overhead_obs, random_wrist_obs,
        real_pose_dataset, real_overhead_dataset, real_wrist_dataset,
        alpha=0.5
    )
    print(idx, dist)
    print(time.time() - start)
    

if __name__ == '__main__':
    main()