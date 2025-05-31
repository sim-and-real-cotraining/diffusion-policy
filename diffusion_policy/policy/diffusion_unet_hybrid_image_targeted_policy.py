from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.mlp.mlp import MLP
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import robomimic.models.obs_core as rmobsc
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class DiffusionUnetHybridImageTargetedPolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            one_hot_encoding_dim=0,
            num_inference_steps=None,
            obs_as_global_cond=True,
            use_target_cond=False,
            target_dim=None,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_embedding_dim=None,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            num_DDIM_inference_steps=10,
            pretrained_encoder=False,
            freeze_pretrained_encoder=False,
            # parameters passed to step
            **kwargs):
        super().__init__()

        if use_target_cond:
            assert target_dim is not None
        assert one_hot_encoding_dim >= 0
        assert obs_as_global_cond

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        # each list contains the keys of the corresponding modality
        # ex. {low_dim: [agent_pos], rgb: [image], depth: [], scan: []}
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        # ex. {agent_pos: shape, image: shape}
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph', 
            pretrained_encoder=pretrained_encoder,
            freeze_pretrained_encoder=freeze_pretrained_encoder)
                
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )

        # extract the image encoder
        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmobsc.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        if obs_embedding_dim is not None:
            obs_feature_dim = obs_embedding_dim
            self.obs_embedding_projector = MLP(obs_encoder.output_shape()[0], [], obs_feature_dim)
            self.obs_embedding_projector.to('cuda' if torch.cuda.is_available() else 'cpu')
            project_obs_embedding = True
        else:
            obs_feature_dim = obs_encoder.output_shape()[0]
            project_obs_embedding = False

        # create diffusion model
        # obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps + one_hot_encoding_dim
        print(f"Input dim: {input_dim}, Global cond dim: {global_cond_dim}")

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            target_dim=target_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        
        # Create DDIM sampler
        DDIM_noise_scheduler = DDIMScheduler(
            num_train_timesteps=self.noise_scheduler.num_train_timesteps,
            beta_start=self.noise_scheduler.beta_start,
            beta_end=self.noise_scheduler.beta_end,
            beta_schedule=self.noise_scheduler.beta_schedule,
            clip_sample=self.noise_scheduler.clip_sample,
            prediction_type=self.noise_scheduler.prediction_type,
        )
        DDIM_noise_scheduler.set_timesteps(num_DDIM_inference_steps)
        self.DDIM_noise_scheduler = DDIM_noise_scheduler
        self.num_DDIM_inference_steps = num_DDIM_inference_steps
        
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.project_obs_embedding = project_obs_embedding
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.one_hot_encoding_dim = one_hot_encoding_dim
        self.use_target_cond = use_target_cond
        self.kwargs = kwargs

        # Store latest observation embedding
        # This gets updated on every forward pass or call to compute_obs_embedding
        self.latest_obs_embedding = None

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
        if project_obs_embedding:
            print("Vision projector params: %e" % sum(p.numel() for p in self.obs_embedding_projector.parameters()))
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            target_cond=None, generator=None,
            use_DDIM=False,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        if use_DDIM:
            scheduler = self.DDIM_noise_scheduler
            scheduler.set_timesteps(self.num_DDIM_inference_steps)
        else:
            scheduler = self.noise_scheduler
            scheduler.set_timesteps(self.num_inference_steps)

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond,
                target_cond=target_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor], use_DDIM=False) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs"
        - if use_target_cond is true, obs_dict must also include "target"
        result: must include "action" key
        """
        assert 'obs' in obs_dict
        if self.use_target_cond:
            assert 'target' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict['obs'])
        ntarget = None
        if self.use_target_cond:
            ntarget = self.normalizer['target'].normalize(obs_dict['target'])
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if self.project_obs_embedding:
                nobs_features = self.obs_embedding_projector(nobs_features)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if self.project_obs_embedding:
                nobs_features = self.obs_embedding_projector(nobs_features)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # append one hot encoding
        if self.one_hot_encoding_dim > 0:
            # currently only supporting global conditioning
            assert self.obs_as_global_cond
            one_hot_encoding = obs_dict['one_hot_encoding']
            global_cond = torch.cat([global_cond, one_hot_encoding], dim=-1)
        self.latest_obs_embedding = global_cond

        # handle target conditioning
        target_cond = None
        if self.use_target_cond:
            target_cond = ntarget.reshape(B, -1) # B, D_t

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            target_cond=target_cond,
            use_DDIM=use_DDIM,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    def compute_obs_embedding(self, batch):
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        key = next(iter(nobs.keys()))
        batch_size = nobs[key].shape[0]

        # handle different ways of passing observation
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if self.project_obs_embedding:
                nobs_features = self.obs_embedding_projector(nobs_features)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            nactions = self.normalizer['action'].normalize(batch['action'])
            horizon = nactions.shape[1]
            trajectory = nactions
            cond_data = trajectory
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if self.project_obs_embedding:
                nobs_features = self.obs_embedding_projector(nobs_features)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()
        
        # append one hot encoding
        if self.one_hot_encoding_dim > 0:
            # currently only supporting global conditioning
            assert self.obs_as_global_cond
            one_hot_encoding = batch['one_hot_encoding']
            global_cond = torch.cat([global_cond, one_hot_encoding], dim=-1)

        self.latest_obs_embedding = global_cond
        return global_cond

    def get_latest_obs_embedding(self):
        return self.latest_obs_embedding

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward(self, batch, noisy_trajectory, timesteps):
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        ntarget = None
        if self.use_target_cond:
            ntarget = self.normalizer['target'].normalize(batch['target'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        # print(f"Inside batch size: {batch_size}")

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if self.project_obs_embedding:
                nobs_features = self.obs_embedding_projector(nobs_features)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if self.project_obs_embedding:
                nobs_features = self.obs_embedding_projector(nobs_features)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # append one hot encoding
        if self.one_hot_encoding_dim > 0:
            # currently only supporting global conditioning
            assert self.obs_as_global_cond
            one_hot_encoding = batch['one_hot_encoding']
            global_cond = torch.cat([global_cond, one_hot_encoding], dim=-1)
        self.latest_obs_embedding = global_cond
        
        # handle target conditioning
        target_cond = None
        if self.use_target_cond:
            target_cond = ntarget.reshape(batch_size, -1) # B, D_t

        # Predict the noise residual
        return self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond,
            target_cond=target_cond)

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        ntarget = None
        if self.use_target_cond:
            ntarget = self.normalizer['target'].normalize(batch['target'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if self.project_obs_embedding:
                nobs_features = self.obs_embedding_projector(nobs_features)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            if self.project_obs_embedding:
                nobs_features = self.obs_embedding_projector(nobs_features)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        
        # append one hot encoding
        if self.one_hot_encoding_dim > 0:
            # currently only supporting global conditioning
            assert self.obs_as_global_cond
            one_hot_encoding = batch['one_hot_encoding']
            global_cond = torch.cat([global_cond, one_hot_encoding], dim=-1)
        self.latest_obs_embedding = global_cond
        
        # handle target conditioning
        target_cond = None
        if self.use_target_cond:
            target_cond = ntarget.reshape(batch_size, -1) # B, D_t

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond,
            target_cond=target_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss