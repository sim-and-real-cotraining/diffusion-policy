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

class DrakeMazeHybridDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            n_obs_steps=None,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        
        super().__init__()
        keys = ['img', 'state', 'action', 'target']
        chunks = {
            'state': (1024, 2),
            'action': (2048, 2),
            'target': (1024, 2),
            'img': (256, 52, 52, 1),
        }
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path=zarr_path, 
            store=zarr.MemoryStore(),
            # chunks=chunks, # rechunk if read time is slow
            keys=keys)

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
        # normalize binary img (0: free space, 1: obstacle, 2: robot)
        image = np.moveaxis(sample['img'].astype(np.float32),-1,1) / 2.0
        # image = np.moveaxis(sample['img'],-1,1)/255

        data = {
            'obs': {
                'image': image, # T, 1, 52, 52
                'agent_pos': agent_pos, # T, 2
            },
            'target': target, # T, 2
            'action': sample['action'].astype(np.float32) # T, 2
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    import time
    import random
    from data_generation.maze.maze_environment import print_binary_img
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--zarr_path', type=str, default=None)
    args = parser.parse_args()
    
    zarr_path = None
    if args.zarr_path is not None:
        zarr_path = args.zarr_path
    else:
        zarr_path = "data/maze_image/maze_image_dataset_4000_rechunked.zarr"

    # default params from maze image config file
    dataset = DrakeMazeHybridDataset(
            zarr_path=zarr_path, 
            horizon=16,
            n_obs_steps=2,
            pad_before=1,
            pad_after=7,
            seed=42,
            val_ratio=0.05,
            max_train_episodes=None)
    
    num_samples = 1000
    avg_time = 0
    for i in range(num_samples):
        start_time = time.time()
        item = dataset[i]
        avg_time += (time.time() - start_time) / num_samples
    print(f"Average time to get zarr sample: {avg_time*1000.0:.3f}ms")


    # visualize some of the data
    while True:
        idx = random.randint(0, len(dataset)-1)
        item = dataset[idx]
        print(f"Sample: {idx}\n")
        
        # visualize observations
        obs = item['obs']
        img = obs['image']
        agent_pos = obs['agent_pos']
        for i in range(len(img)):
            # check if agent postition contains nan
            if not np.isnan(agent_pos[i]).any():
                print("Agent position: ", agent_pos[i])
                print_binary_img(2*img[i,0])

        # visualize actions
        actions = item['action']
        for i in range(len(actions)):
            print("Action: ", actions[i])
        print("\nTarget: ", item['target'])
        print()
        
        # inspect data before moving on
        breakpoint()
            



    # find optimal chunk sizes
    # base_chunks = {
    #     'state': (dataset.n_obs_steps, 2),
    #     'action': (dataset.horizon, 2),
    #     'target': (dataset.n_obs_steps, 2),
    #     'img': (dataset.n_obs_steps, 52, 52, 1),
    # }
    
    # factors = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    # avg_times = []
    # for factor in factors:
    #     factor *= 16
    #     print("\nResults for ", factor)
    #     chunks = {
    #         'state': dataset.replay_buffer.root['data']['state'].chunks,
    #         'action': dataset.replay_buffer.root['data']['action'].chunks,
    #         'target': dataset.replay_buffer.root['data']['target'].chunks,
    #         'img': dataset.replay_buffer.root['data']['img'].chunks,
    #     }
    #     # chunks['state'] = (dataset.n_obs_steps * factor, 2)
    #     # chunks['target'] = (dataset.n_obs_steps * factor, 2)
    #     chunks['img'] = (dataset.n_obs_steps * factor, 52, 52, 1)
    #     dataset.replay_buffer = ReplayBuffer.copy_from_store(
    #             src_store=dataset.replay_buffer.root.store, 
    #             store=zarr.MemoryStore(),
    #             chunks=chunks,
    #             keys=['img', 'state', 'action', 'target'])
    #     dataset.sampler = SequenceSampler(
    #             replay_buffer=dataset.replay_buffer, 
    #             sequence_length=dataset.horizon,
    #             pad_before=dataset.pad_before, 
    #             pad_after=dataset.pad_after,
    #             episode_mask=dataset.train_mask)
    
    #     avg_time = 0
    #     for i in range(num_samples):
    #         item, time_taken = dataset[i]
    #         avg_time += time_taken / num_samples
    #     print(f"Average time to get zarr sample: {avg_time*1000.0:.3f}ms")
    #     avg_times.append(avg_time)
    
    # print(avg_times)
    
    # avg_time = 0
    # for i in range(num_samples):
    #     start_time = time.time()
    #     item = dataset.get_dummy_item(i)
    #     avg_time += (time.time() - start_time) / num_samples
    # print(f"Average time to get dummy sample: {avg_time*1000.0:.3f}ms")      

if __name__ == "__main__":
    test()