from typing import Dict
import zarr
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class DrakeCotrainPlanarPushingHybridDataset(BaseImageDataset):
    def __init__(self,
            zarr_configs, 
            horizon=1,
            n_obs_steps=None,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            max_train_trajectories=None
    ):
        if max_train_trajectories:
            raise NotImplementedError("max_train_trajectories not yet implemented for cotraining datasets")
        
        super().__init__()
        keys = ['img', 'state', 'action', 'target']
        chunks = {
            'state': (1024, 3),
            'action': (2048, 2),
            'target': (1024, 3),
            'img': (128, 96, 96, 3),
        }
        
        replay_buffers = []
        no_scaling = False
        for zarr_config in zarr_configs:
            replay_buffers.append(
                ReplayBuffer.copy_from_path(
                    zarr_path=zarr_config['path'], 
                    store=zarr.MemoryStore(),
                    # chunks=chunks, # rechunk if read time is slow
                    keys=keys
                )
            )

            if zarr_config['sampling_ratio'] is None:
                no_scaling = True

            # Remove excess episodes. Keep last episodes
            if 'max_train_trajectories' in zarr_config:
                max_train_trajectories = zarr_config['max_train_trajectories']
                if max_train_trajectories is not None:
                    root = replay_buffers[-1].root
                    assert 0 < max_train_trajectories <= root['meta']['episode_ends'].shape[0]
                    start_idx = root['meta']['episode_ends'][-max_train_trajectories-1]
                    root['meta']['episode_ends'] = root['meta']['episode_ends'][-max_train_trajectories:] - start_idx

                    for key, value in root['data'].items():
                        root['data'][key] = value[start_idx:]
                    replay_buffers[-1].root= root

        if no_scaling:
            resize_factors = np.ones(len(replay_buffers))
        else:
            ratios = np.array([zarr_config['sampling_ratio'] for zarr_config in zarr_configs])
            sizes = np.array([replay_buffer.n_episodes for replay_buffer in replay_buffers])
            resize_factors = self._compute_resizing_factors(ratios, sizes)
        print("Resize factors for the datasets: ", resize_factors)

        # Resize and combine the replay buffers
        if not no_scaling:
            for i in range(len(replay_buffers)):
                replay_buffers[i] = self._resize_replay_buffer(replay_buffers[i], resize_factors[i])
        self.replay_buffer = self._combine_replay_buffers(replay_buffers)
        
        nepisodes = self.replay_buffer.n_episodes
        ratios = [replay_buffer.n_episodes / nepisodes for replay_buffer in replay_buffers]

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in keys:
                key_first_k[key] = n_obs_steps
        key_first_k['action'] = horizon

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'],
            'target': self.replay_buffer['target']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        # normalizer is a dict containing normalizers for the
        # 'action', 'agent_pos', 'target', and 'image' keys

        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        target = sample['target'][0].astype(np.float32)
        agent_pos = sample['state'].astype(np.float32)
        image = np.moveaxis(sample['img'],-1,1)/255

        data = {
            'obs': {
                'image': image, # T_obs, 3, 96, 96
                'agent_pos': agent_pos, # T_obs, 3
            },
            'target': target, # 3
            'action': sample['action'].astype(np.float32) # T, 2
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
    
    def _compute_resizing_factors(self, ratios, sizes):
        # Extract useful info
        assert len(ratios) == len(sizes)
        num_datasets = len(ratios)
        
        # A matrix
        _A = np.outer(ratios, sizes) - np.diag(sizes)
        ones = np.ones((1, num_datasets))
        A = np.vstack((_A, ones))

        # b vector
        b = np.zeros((num_datasets+1,))
        b[-1] = 1 # normalize sum of weights to 1

        w, res, rank, sing_vals = np.linalg.lstsq(A, b, rcond=None)
        assert np.allclose(res, np.zeros_like(res))
        w_min = np.min(w)
        return w / w_min
    
    def _resize_replay_buffer(self, replay_buffer, factor):
        if np.allclose(factor, 1.0):
            return replay_buffer
        
        root = replay_buffer.root
        length = root['meta']['episode_ends'][-1]
        nepisodes = root['meta']['episode_ends'].shape[0]

        # Over-stack episodes
        for i in range(int(np.floor(factor))):
            for key, value in root['data'].items():
                root['data'][key] = np.vstack([value, value[:length]])
            root['meta']['episode_ends'] = np.hstack([root['meta']['episode_ends'], root['meta']['episode_ends'][-1] + root['meta']['episode_ends'][:length]])

        # Remove excess episodes
        desired_nepisodes = int(np.floor(factor * nepisodes))
        root['meta']['episode_ends'] = root['meta']['episode_ends'][:desired_nepisodes]
        end_idx = root['meta']['episode_ends'][-1]
        for key, value in root['data'].items():
            root['data'][key] = value[:end_idx]

        return ReplayBuffer(root=root)
    
    def _combine_replay_buffers(self, replay_buffers):
        # Combine replay buffers into the first replay buffer
        root = replay_buffers[0].root
        for replay_buffer in replay_buffers[1:]:
            for key in root['data'].keys():
                root['data'][key] = np.vstack([root['data'][key], replay_buffer.root['data'][key]])
            root['meta']['episode_ends'] = np.hstack([root['meta']['episode_ends'], root['meta']['episode_ends'][-1] + replay_buffer.root['meta']['episode_ends']])
        return ReplayBuffer(root=root)

if __name__ == "__main__":
    import random
    import time

    num_sim = 2000
    num_real = 50
    sim_ratio = 1
    hw_ratio = 1

    zarr_configs = [
        {
            'path': 'data/planar_pushing/underactuated_data.zarr',
            'max_train_trajectories': num_sim,
            'sampling_ratio': 1.0*sim_ratio / (sim_ratio + hw_ratio)
        },
        {
            'path': 'data/planar_pushing/hw_push_tee_dataset_v2.zarr',
            'max_train_trajectories': num_real,
            'sampling_ratio': 1.0*hw_ratio / (sim_ratio + hw_ratio)
        },
    ]
    start_time = time.time()
    dataset = DrakeCotrainPlanarPushingHybridDataset(
        zarr_configs=zarr_configs,
        horizon = 16,
        n_obs_steps = 2,
        pad_before = 1,
        pad_after = 7,
        seed=42,
        val_ratio=0.05,
        max_train_episodes=None,
        max_train_trajectories=None
    )
    normalizer = dataset.get_normalizer()
    normalizer_path = f"scaled_sim_{num_sim}_real_{num_real}_{sim_ratio}:{hw_ratio}_normalizer.pt"
    torch.save(normalizer.state_dict(), normalizer_path)
    print(normalizer_path)
    print(f"finished in {time.time()-start_time:.2f}s")



    # for _ in range(10):
    #     idx = random.randint(0, len(dataset)-1)
    #     sample = dataset[idx]
    #     states = sample['obs']['agent_pos']
    #     actions = sample['action']
    #     print(f"Sample states : {states}")
    #     print(f"Sample actions: {actions}")
    #     print()
    #     breakpoint()