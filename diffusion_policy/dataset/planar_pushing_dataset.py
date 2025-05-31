from typing import Dict
import zarr
import torch
from torchvision import transforms
import numpy as np
import os
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class PlanarPushingDataset(BaseImageDataset):
    """
    Dataset for planar pushing that supports:
    - hybrid observations (images + end effector state)
    - multi cameras
    - cotraining with multiple datasets (datasets must share input output space)
    - dataset/loss scaling
    """
    def __init__(
        self,
        zarr_configs,
        shape_meta,
        use_one_hot_encoding=False,
        horizon=1,
        n_obs_steps=None,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        color_jitter=None,
    ):
        
        super().__init__()
        self._validate_zarr_configs(zarr_configs)

        # Set up dataset keys
        self.rgb_keys = []
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', '')
            if type == 'rgb':
                self.rgb_keys.append(key)
            
        keys = self.rgb_keys + ['state', 'action', 'target']

        # trick for saving ram
        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in keys:
                key_first_k[key] = n_obs_steps
        key_first_k['action'] = horizon

        # Load in all the zarr datasets
        self.num_datasets = len(zarr_configs)
        self.replay_buffers = []
        self.train_masks = []
        self.val_masks = []
        self.samplers = []
        self.sample_probabilities = np.zeros(len(zarr_configs))
        self.zarr_paths = []

        for i, zarr_config in enumerate(zarr_configs):
            # Extract config info
            zarr_path = zarr_config['path']
            max_train_episodes = zarr_config.get('max_train_episodes', None)
            sampling_weight = zarr_config.get('sampling_weight', None)
            
            # Set up replay buffer
            self.replay_buffers.append(ReplayBuffer.copy_from_path(
                    zarr_path=zarr_path, 
                    store=zarr.MemoryStore(),
                    keys=keys
                )
            )
            n_episodes = self.replay_buffers[-1].n_episodes

            # Set up masks
            if 'val_ratio' in zarr_config and zarr_config['val_ratio'] is not None:
                dataset_val_ratio = zarr_config['val_ratio']
            else:
                dataset_val_ratio = val_ratio
            val_mask = get_val_mask(
                n_episodes=n_episodes, 
                val_ratio=dataset_val_ratio,
                seed=seed)
            train_mask = ~val_mask
            # Note max_train_episodes is the max number of training episodes
            # not the total number of train and val episodes!
            train_mask = downsample_mask(
                mask=train_mask, 
                max_n=max_train_episodes, 
                seed=seed)
            
            self.train_masks.append(train_mask)
            self.val_masks.append(val_mask)
            
            # Set up sampler
            self.samplers.append(
                SequenceSampler(
                    replay_buffer=self.replay_buffers[-1], 
                    sequence_length=horizon,
                    pad_before=pad_before, 
                    pad_after=pad_after,
                    episode_mask=train_mask,
                    key_first_k=key_first_k
                )
            )
            
            # Set up sample probabilities and zarr paths
            if sampling_weight is not None:
                self.sample_probabilities[i] = sampling_weight
            else:
                self.sample_probabilities[i] = np.sum(train_mask)
            self.zarr_paths.append(zarr_path)
        # Normalize sample_probabilities
        self.sample_probabilities = self._normalize_sample_probabilities(self.sample_probabilities)

        # Set up color jitter
        self.color_jitter = color_jitter
        if color_jitter is not None:
            self.transforms = transforms.ColorJitter(
                brightness=self.color_jitter.get('brightness', 0),
                contrast=self.color_jitter.get('contrast', 0),
                saturation=self.color_jitter.get('saturation', 0),
                hue=self.color_jitter.get('hue', 0)
            )

        # Load other variables
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.n_obs_steps = n_obs_steps
        self.shape_meta = shape_meta
        self.use_one_hot_encoding = use_one_hot_encoding
        self.one_hot_encoding = None # if val dataset, this will not be None


    def get_validation_dataset(self, index=None):
        # Create validation dataset
        val_set = copy.copy(self)

        if index == None:
            assert self.num_datasets == 1, "Must specify validation dataset index if multiple datasets"
            index = 0
        else:
            val_set.replay_buffers = [self.replay_buffers[index]]
            val_set.train_masks = [self.train_masks[index]]
            val_set.val_masks = [self.val_masks[index]]
            val_set.zarr_paths = [self.zarr_paths[index]]
        val_set.num_datasets = 1
        val_set.sample_probabilities = np.array([1.0])

        # Set one hot encoding
        val_set.one_hot_encoding = np.zeros(self.num_datasets).astype(np.float32)
        val_set.one_hot_encoding[index] = 1

        val_set.samplers = [SequenceSampler(
            replay_buffer=self.replay_buffers[index], 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_masks[index]
        )]
        
        return val_set
    
    def get_normalizer(self, mode='limits', **kwargs):
        # compute mins and maxes
        assert mode == 'limits', "Only supports limits mode"
        low_dim_keys = ['action', 'agent_pos', 'target']
        input_stats = {}
        for replay_buffer in self.replay_buffers:
            data = {
                'action': replay_buffer['action'],
                'agent_pos': replay_buffer['state'],
                'target': replay_buffer['target']
            }
            normalizer = LinearNormalizer()
            normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)

            # Update mins and maxes
            for key in low_dim_keys:
                _max = normalizer[key].params_dict.input_stats.max
                _min = normalizer[key].params_dict.input_stats.min

                if key not in input_stats:
                    input_stats[key] = {'max': _max, 'min': _min}
                else:
                    input_stats[key]['max'] = torch.maximum(input_stats[key]['max'], _max)
                    input_stats[key]['min'] = torch.minimum(input_stats[key]['min'], _min)

        # Create normalizer
        # Normalizer is a PyTorch parameter dict containing normalizers for all the keys
        normalizer = LinearNormalizer()
        normalizer.fit_from_input_stats(input_stats_dict=input_stats)
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_sample_probabilities(self):
        return self.sample_probabilities
    
    def get_num_datasets(self):
        return self.num_datasets
    
    def get_num_episodes(self, index=None):
        if index == None:
            num_episodes = 0
            for i in range(self.num_datasets):
                num_episodes += self.replay_buffers[i].n_episodes
            return num_episodes
        else:
            return self.replay_buffers[index].n_episodes

    def __len__(self) -> int:
        length = 0
        for sampler in self.samplers:
            length += len(sampler)
        return length

    def _sample_to_data(self, sample, sampler_idx):
        target = sample['target'][0].astype(np.float32)
        agent_pos = sample['state'].astype(np.float32)

        data = {
            'obs': {
                'agent_pos': agent_pos, # T_obs, 3
            },
            'target': target, # 3
            'action': sample['action'].astype(np.float32) # T, 2
        }

        if self.use_one_hot_encoding:
            if self.one_hot_encoding is None:
                data['one_hot_encoding'] = np.zeros(self.num_datasets).astype(np.float32)
                data['one_hot_encoding'][sampler_idx] = 1
            else:
                data['one_hot_encoding'] = self.one_hot_encoding

        # Add images to data
        if self.color_jitter is None:
            for key in self.rgb_keys:
                data['obs'][key] = np.moveaxis(sample[key],-1,1)/255.0
                del sample[key]
        else:
            # Stack images and apply color jitter to ensure
            # all cameras have consistent color jitter
            keys = self.rgb_keys
            length = sample[keys[0]].shape[0]
            
            imgs = np.moveaxis(np.vstack([sample[key] for key in keys]),-1,1)/255.0
            for i in range(3):
                scale = np.random.uniform(0.75, 1.25) # TODO: these are hardcoded
                imgs[:,i,:,:] = np.clip(scale*imgs[:,i,:,:], 0, 1)
            
            # imgs = np.vstack([sample[key] for key in keys])
            imgs = self.transforms(torch.from_numpy(imgs)).numpy()
            for i, key in enumerate(keys):
                data['obs'][key] = imgs[i*length:(i+1)*length]
                del sample[key]

        return data
    
    def _validate_zarr_configs(self, zarr_configs):
        num_null_sampling_weights = 0
        N = len(zarr_configs)

        for zarr_config in zarr_configs:
            zarr_path = zarr_config['path']
            if not os.path.exists(zarr_path):
                raise ValueError(f"path {zarr_path} does not exist")
            
            max_train_episodes = zarr_config.get('max_train_episodes', None)
            if max_train_episodes is not None and max_train_episodes <= 0:
                raise ValueError(f"max_train_episodes must be greater than 0, got {max_train_episodes}")
            
            sampling_weight = zarr_config.get('sampling_weight', None)
            if sampling_weight is None:
                num_null_sampling_weights += 1
            elif sampling_weight < 0:
                raise ValueError(f"sampling_weight must be greater than or equal to 0, got {sampling_weight}")
        
        if num_null_sampling_weights not in [0, N]:
            raise ValueError("Either all or none of the zarr_configs must have a sampling_weight")
    
    def _normalize_sample_probabilities(self, sample_probabilities):
        total = np.sum(sample_probabilities)
        assert total > 0, "Sum of sampling weights must be greater than 0"
        return sample_probabilities / total
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # To sample a sequence, first sample a dataset,
        # then sample a sequence from that dataset
        # Note that this implementation does not guarantee that each unique
        # sequence is sampled on every epoch!
        
        # Get sample
        if self.num_datasets == 1:
            sampler_idx = 0
            sampler = self.samplers[sampler_idx]
            sample = sampler.sample_sequence(idx)
        else:
            sampler_idx = np.random.choice(self.num_datasets, p=self.sample_probabilities)
            sampler = self.samplers[sampler_idx]
            sample = sampler.sample_sequence(idx % len(sampler))
        
        # Process sample
        data = self._sample_to_data(sample, sampler_idx)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

if __name__ == "__main__":
    import random
    import cv2
    import tqdm
    from torch.utils.data import DataLoader

    shape_meta = {
        'action': {'shape': [2]},
        'obs': {
            'agent_pos': {'type': 'low_dim', 'shape': [3]},
            'overhead_camera': {'type': 'rgb', 'shape': [3, 128, 128]},
            'wrist_camera': {'type': 'rgb', 'shape': [3, 128, 128]},
        },
    }
    zarr_configs = [
        # {
        #     'path': 'data/planar_pushing/underactuated_data.zarr',
        #     'max_train_episodes': None,
        #     'sampling_weight': 1.0
        # },
        {
            # 'path': 'data/planar_pushing_cotrain/visual_mean_shift/visual_mean_shift_level_2.zarr',
            'path': 'data/planar_pushing_cotrain/visual_mean_shift/visual_mean_shift_level_2.zarr',
            'max_train_episodes': None,
            'sampling_weight': 1.0
        }
    ]
    n_obs_steps = 2
    color_jitter = {
        'brightness': 0.15,
        # 'contrast': 0.5,
        'saturation': 0.15,
        'hue': 0.15,
    }

    dataset = PlanarPushingDataset(
        zarr_configs=zarr_configs,
        shape_meta=shape_meta,
        horizon = 8,
        n_obs_steps = n_obs_steps,
        pad_before = 1,
        pad_after = 7,
        seed=42,
        val_ratio=0.05,
        # color_jitter=color_jitter
    )
    print("Initialized dataset")
    print("Total episodes (train + val):", dataset.get_num_episodes())
    print("Training dataset length:", len(dataset))

    # Test get validation dataset
    for i in range(dataset.get_num_datasets()):
        val_dataset = dataset.get_validation_dataset(index=i)
        print(f"Got validation dataset {i}")

    # Test normalizer
    normalizer = dataset.get_normalizer()

    for i in range(10):
        idx = random.randint(0, len(dataset)-1)
        # idx = i % len(dataset)
        sample = dataset[idx]
        states = sample['obs']['agent_pos']
        actions = sample['action']
        print(f"Sample states : {states}")
        print(f"Sample actions: {actions}")
        print(f"Sample target : {sample['target']}")
        print()
        print("Press any key to continue. Ctrl+\\ to exit.\n")

        for key, attr in sample['obs'].items():
            if key == 'agent_pos':
                continue

            for i in range(n_obs_steps):
                image_array = attr[i].detach().numpy().transpose(1, 2, 0)

                # Convert the RGB array to BGR
                image_array[:,:,0], image_array[:,:,2] = image_array[:,:,2], image_array[:,:,0].copy()
                # image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

                # Display the image using OpenCV
                cv2.imshow(f'{key}_{i}', image_array)
                cv2.waitKey(0)  # Wait for a key press to close the image window
                cv2.destroyAllWindows()
    
    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=64,
    #     num_workers=1,
    #     persistent_workers=False,
    #     pin_memory=True,
    #     shuffle=True
    # )

    # for local_epoch_idx in range(50):
    #     with tqdm.tqdm(train_dataloader) as tepoch:
    #         for batch_idx, batch in enumerate(tepoch):
    #             print(local_epoch_idx, batch_idx)

    # while True:
    #     idx = random.randint(0, len(dataset)-1)
    #     sample = dataset[idx]