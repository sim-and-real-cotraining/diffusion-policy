from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset

class MazeLowdimTargetConditionedDataset(BaseLowdimDataset):
    def __init__(self, 
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            state_key='state',
            target_key = 'target',
            action_key='action',
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            max_train_trajectories=None
            ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[state_key, target_key, action_key])
        
        if max_train_trajectories:
            print(f"Downsampling trajectories to {max_train_trajectories}")
            root = self.replay_buffer.root
            assert 0 < max_train_trajectories < root['meta']['episode_ends'].shape[0]
            
            root['meta']['episode_ends'] = root['meta']['episode_ends'][:max_train_trajectories]
            end_idx = root['meta']['episode_ends'][-1]
            root['data']['state'] = root['data']['state'][:end_idx]
            root['data']['action'] = root['data']['action'][:end_idx]
            root['data']['target'] = root['data']['target'][:end_idx]
            self.replay_buffer.root= root

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask
            )
        self.state_key = state_key
        self.target_key = target_key
        self.action_key = action_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

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
            'obs': self.replay_buffer['state'],
            'target': self.replay_buffer['target']
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        data = {
            'obs': sample[self.state_key], # T, D_o
            'action': sample[self.action_key], # T, D_a
            'target': sample[self.target_key][0], # D_t, D_o
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
