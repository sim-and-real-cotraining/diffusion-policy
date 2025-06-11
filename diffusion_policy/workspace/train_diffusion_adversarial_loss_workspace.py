if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
from enum import Enum
import hydra
import torch
from omegaconf import OmegaConf, ListConfig
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import dill
import shutil
import torch
from torch.nn.parallel import DataParallel
from einops import rearrange, reduce
import torch.nn.functional as F
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_hybrid_image_targeted_policy import DiffusionUnetHybridImageTargetedPolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset
# from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)

class ModelType(Enum):
    DISCRIMINATOR = 0
    DENOISER = 1

class DataParallelWrapper(DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
        
class TrainDiffusionAdversarialLossWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        # This environment does not support impainting or local conditioning
        assert cfg.policy.obs_as_global_cond is True
        assert cfg.training.gradient_accumulate_every == 1
        # This enviornment does not currently support finetuning
        assert 'pretrained_checkpoint' not in cfg or cfg.pretrained_checkpoint is None

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model = hydra.utils.instantiate(cfg.policy)
        self.model = self.model.to(torch.device("cuda:0"))

        # configure discriminator
        self.discriminator_module = None
        if 'discriminator_module' in cfg.training:
            self.discriminator_module = cfg.training.discriminator_module
        self.discriminator = hydra.utils.instantiate(cfg.discriminator)
        self.discriminator = self.discriminator.to(torch.device("cuda:0"))

        # num_GPU = torch.cuda.device_count()
        num_GPU = 1
        print(f"Running on {num_GPU} GPU(s).")
        self.model = DataParallelWrapper(self.model, device_ids=range(num_GPU))
        self.discriminator = DataParallelWrapper(self.discriminator, device_ids=range(num_GPU))
        
        # register hooks if needed
        if self.discriminator_module is not None:
            assert num_GPU <= 1, "Hooks not supported for multi-GPU training"
            self.activation = None
            for name, module in self.model.named_modules():
                if name == self.discriminator_module:
                    module.register_forward_hook(self.hook_fn)
                    print(f"Hook registered to module: {name}")

        self.ema_model: DiffusionUnetHybridImageTargetedPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = hydra.utils.instantiate(
            cfg.optimizer, params=self.model.parameters())
        self.discriminator_optimizer = hydra.utils.instantiate(
            cfg.discriminator_optimizer, params=self.discriminator.parameters())

        # configure training state
        self.global_step = 0
        self.epoch = 0
        
        # variables for load payload
        self.bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.0])) # will be overwritten later
    
    def hook_fn(self, module, input, output):
        self.activation = input[0].reshape(input[0].shape[0], -1)

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
                # self.epoch is loaded with the last completed epoch
                # the current epoch is the next epoch (hence += 1)
                self.epoch += 1

        # configure dataset and save normalizer
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        torch.save(normalizer, os.path.join(self.output_dir, 'normalizer.pt'))

        # configure weighted BCE loss
        sampling_probabilities = dataset.get_sample_probabilities()
        pos_weight = 1.0*sampling_probabilities[0] / sampling_probabilities[1]
        self.bce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

        # configure validation datasets
        self.num_datasets = dataset.get_num_datasets()
        self.sample_probabilities = dataset.get_sample_probabilities()
        val_dataloaders = []
        for i in range(self.num_datasets):
            val_dataset = dataset.get_validation_dataset(i)
            val_dataloaders.append(DataLoader(val_dataset, **cfg.val_dataloader))
        self._print_dataset_diagnostics(cfg, dataset, train_dataloader, val_dataloaders)
        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)


        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        if not isinstance(cfg.checkpoint, ListConfig):
            # configure single checkpoint manager
            topk_managers = [TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, 'checkpoints'),
                **cfg.checkpoint.topk
            )]
            save_last_ckpt = cfg.checkpoint.save_last_ckpt
            save_last_snapshot = cfg.checkpoint.save_last_snapshot
        else:
            # configure multiple checkpoint managers
            topk_managers = []
            save_last_ckpt = False
            save_last_snapshot = False
            for ckpt_cfg in cfg.checkpoint:
                topk_managers.append(TopKCheckpointManager(
                    save_dir=os.path.join(self.output_dir, 'checkpoints'),
                    **ckpt_cfg.topk
                ))
                save_last_ckpt = save_last_ckpt or ckpt_cfg.save_last_ckpt
                save_last_snapshot = save_last_snapshot or ckpt_cfg.save_last_snapshot


        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        self.discriminator.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)
        optimizer_to(self.discriminator_optimizer, device)
        self.bce_loss.to(device)

        # save batch for sampling
        val_sampling_batches = [None] * self.num_datasets

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        model_to_optimize = ModelType.DENOISER
        discriminator_steps = 0
        resume_epoch = self.epoch
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(resume_epoch, cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                diffusion_losses = list()
                discriminator_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        batch_size = batch['action'].shape[0]
                        # print(f"Outside batch size: {batch_size}")
                        if model_to_optimize == ModelType.DENOISER:
                            # construct noisy trajectory
                            trajectory = self.model.normalizer['action'].normalize(batch['action'])
                            noise = torch.randn(trajectory.shape, device=trajectory.device)
                            # Sample a random timestep for each image
                            timesteps = torch.randint(
                                0, self.model.noise_scheduler.config.num_train_timesteps, 
                                (batch_size,), device=trajectory.device
                            ).long()
                            # Add noise to the clean images according to the noise magnitude at each timestep
                            # (this is the forward diffusion process)
                            noisy_trajectory = self.model.noise_scheduler.add_noise(
                                trajectory, noise, timesteps)
                            
                            # call to the policy's forward function: computes 1 denoising step
                            if self.discriminator_module is None:
                                pred, disc_input = self.model(batch, noisy_trajectory, timesteps, return_embedding=True)
                            else:
                                pred = self.model(batch, noisy_trajectory, timesteps)
                                disc_input = self.activation
                            
                            # forward pass through discriminator
                            if self.global_step > cfg.training.discriminator_start_step:
                                label_pred = self.discriminator(disc_input)
                                if cfg.training.swap_labels:
                                    labels = torch.ones_like(batch['label']) - batch['label']
                                else:
                                    labels = batch['label']
                                raw_discriminator_loss = self.compute_discriminator_loss(labels, label_pred)
                                discriminator_loss = raw_discriminator_loss / cfg.training.gradient_accumulate_every
                            else:
                                discriminator_loss = 0.0

                            # compute loss
                            raw_denoiser_loss = self.compute_denoiser_loss(trajectory, noise, pred)
                            denoiser_loss = raw_denoiser_loss / cfg.training.gradient_accumulate_every
                            if cfg.training.swap_labels:
                                loss = denoiser_loss + cfg.training._lambda * discriminator_loss
                            else:
                                loss = denoiser_loss - cfg.training._lambda * discriminator_loss
                            loss.backward()

                            # step optimizer
                            if self.global_step % cfg.training.gradient_accumulate_every == 0:
                                self.optimizer.step()
                                self.optimizer.zero_grad()
                                lr_scheduler.step()
                            
                            # update ema
                            if cfg.training.use_ema:
                                ema.step(self.model)

                            # logging
                            tepoch.set_postfix(loss=loss.item(), refresh=False)
                            train_losses.append(loss.item())
                            diffusion_losses.append(denoiser_loss.item())
                            if self.global_step > cfg.training.discriminator_start_step:
                                discriminator_losses.append(discriminator_loss.item())
                            
                            # optimize the discriminator on the next batch (inner optimization)
                            if self.global_step >= cfg.training.discriminator_start_step: # begin optimizing the discriminator at step C
                                model_to_optimize = ModelType.DISCRIMINATOR
                        else:
                            if self.discriminator_module is None:
                                # Compute observation embedding as disc_input
                                disc_input = self.model.compute_obs_embedding(batch)
                            else:
                                # Forward pass through denoiser to compute the activation
                                # construct noisy trajectory
                                trajectory = self.model.normalizer['action'].normalize(batch['action'])
                                noise = torch.randn(trajectory.shape, device=trajectory.device)
                                # Sample a random timestep for each image
                                timesteps = torch.randint(
                                    0, self.model.noise_scheduler.config.num_train_timesteps, 
                                    (batch_size,), device=trajectory.device
                                ).long()
                                # Add noise to the clean images according to the noise magnitude at each timestep
                                # (this is the forward diffusion process)
                                noisy_trajectory = self.model.noise_scheduler.add_noise(
                                    trajectory, noise, timesteps)
                                
                                pred = self.model(batch, noisy_trajectory, timesteps)
                                disc_input = self.activation

                            label_pred = self.discriminator(disc_input)

                            # compute loss
                            raw_discriminator_loss = self.compute_discriminator_loss(batch['label'], label_pred)
                            discriminator_loss = raw_discriminator_loss / cfg.training.gradient_accumulate_every
                            bce_loss = discriminator_loss
                            bce_loss.backward()

                            # step optimizer
                            if self.global_step % cfg.training.gradient_accumulate_every == 0:
                                self.discriminator_optimizer.step()
                                self.discriminator_optimizer.zero_grad()
                                lr_scheduler.step()
                            
                            # logging
                            discriminator_losses.append(discriminator_loss.item())

                            # optimize the denoiser on the next batch (outer optimization)
                            discriminator_steps += 1
                            if discriminator_steps >= cfg.training.discriminator_steps:
                                discriminator_steps = 0
                                model_to_optimize = ModelType.DENOISER
                        
                        step_log = {
                            'train_loss': loss, # updated every 2 batches
                            'denoiser_loss': denoiser_loss, # updated every 2 batches
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        # Compute discriminator accuracy
                        if self.global_step > cfg.training.discriminator_start_step:
                            discriminator_accuracy = self.compute_discriminator_accuracy(batch['label'], label_pred)
                            if discriminator_loss != 0.0:
                                step_log['discriminator_loss'] = discriminator_loss
                            step_log['balanced_accuracy'] = discriminator_accuracy['balanced_accuracy']
                            step_log['true_negatives_accuracy'] = discriminator_accuracy['true_negatives_accuracy']
                            step_log['true_positives_accuracy'] = discriminator_accuracy['true_positives_accuracy']
                            step_log['discrimination'] = discriminator_accuracy['discrimination']

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break
                    
                    # sample bernoulli random variable with parameter cfg.training.bernoulli_prob
                    if cfg.training.random_discriminator_schedule:
                        if random.random() < cfg.training.bernoulli_prob:
                            model_to_optimize = ModelType.DENOISER
                        else:
                            model_to_optimize = ModelType.DISCRIMINATOR

                    if cfg.training.discriminator_accuracy_threshold is not None:
                        lower = cfg.training.discriminator_accuracy_threshold[0]
                        upper = cfg.training.discriminator_accuracy_threshold[1]
                        if discriminator_accuracy['balanced_accuracy'] >= upper:
                            model_to_optimize = ModelType.DENOISER
                        elif discriminator_accuracy['balanced_accuracy'] <= lower:
                            model_to_optimize = ModelType.DISCRIMINATOR


                    # End of batch                       

                # at the end of each epoch
                # replace train_loss with epoch average
                step_log['train_loss'] = np.mean(train_losses)
                step_log['denoiser_loss'] = np.mean(diffusion_losses)
                if len(discriminator_losses) > 0:
                    step_log['discriminator_loss'] = np.mean(discriminator_losses)
                else:
                    step_log['discriminator_loss'] = 0.0

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()
                self.discriminator.eval()

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_loss_per_dataset = []
                        denoiser_loss_per_dataset = []
                        discriminator_loss_per_dataset = []
                        for dataset_idx in range(self.num_datasets):
                            val_dataloader = val_dataloaders[dataset_idx]
                            val_losses = list()
                            denoiser_losses = list()
                            discriminator_losses = list()
                            with tqdm.tqdm(val_dataloader, desc=f"Dataset {dataset_idx} validation, epoch {self.epoch}", 
                                    leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                                for batch_idx, batch in enumerate(tepoch):
                                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                    if val_sampling_batches[dataset_idx] is None:
                                        val_sampling_batches[dataset_idx] = batch
                                    # denoiser loss
                                    denoiser_loss = self.model.compute_loss(batch)

                                    # discriminator loss
                                    if self.global_step > cfg.training.discriminator_start_step:
                                        if self.discriminator_module is None:
                                            # Compute observation embedding as disc_input
                                            disc_input = self.model.compute_obs_embedding(batch)
                                        else:
                                            # Forward pass through denoiser to compute the activation
                                            # construct noisy trajectory
                                            trajectory = self.model.normalizer['action'].normalize(batch['action'])
                                            noise = torch.randn(trajectory.shape, device=trajectory.device)
                                            # Sample a random timestep for each image
                                            timesteps = torch.randint(
                                                0, self.model.noise_scheduler.config.num_train_timesteps, 
                                                (batch['action'].shape[0],), device=trajectory.device
                                            ).long()
                                            # Add noise to the clean images according to the noise magnitude at each timestep
                                            # (this is the forward diffusion process)
                                            noisy_trajectory = self.model.noise_scheduler.add_noise(
                                                trajectory, noise, timesteps)
                                            
                                            pred = self.model(batch, noisy_trajectory, timesteps)
                                            disc_input = self.activation

                                        label_pred = self.discriminator(disc_input)
                                        if cfg.training.swap_labels:
                                            labels = torch.ones_like(batch['label']) - batch['label']
                                        else:
                                            labels = batch['label']
                                        discriminator_loss = self.compute_discriminator_loss(labels, label_pred)
                                    else:
                                        discriminator_loss = 0.0
                                    
                                    if cfg.training.swap_labels:
                                        val_losses.append(denoiser_loss + cfg.training._lambda * discriminator_loss)
                                    else:
                                        val_losses.append(denoiser_loss - cfg.training._lambda * discriminator_loss)
                                    denoiser_losses.append(denoiser_loss)
                                    discriminator_losses.append(discriminator_loss)
                                    
                                    if (cfg.training.max_val_steps is not None) \
                                        and batch_idx >= (cfg.training.max_val_steps-1):
                                        break
                            # End of validation loss computation loop

                            if len(val_losses) > 0:
                                val_loss = torch.mean(torch.tensor(val_losses)).item()
                                denoiser_loss = torch.mean(torch.tensor(denoiser_losses)).item()
                                discriminator_loss = torch.mean(torch.tensor(discriminator_losses)).item()
                                val_loss_per_dataset.append(val_loss)
                                denoiser_loss_per_dataset.append(denoiser_loss)
                                discriminator_loss_per_dataset.append(discriminator_loss)
                                step_log[f'val_loss_{dataset_idx}'] = val_loss
                                step_log[f'val_denoiser_loss_{dataset_idx}'] = denoiser_loss
                                step_log[f'val_discriminator_loss_{dataset_idx}'] = discriminator_loss
                        # End val_dataloader loop

                # run diffusion sampling on a _single_ validation batch from each dataset
                if (self.epoch % cfg.training.sample_every) == 0 and cfg.training.log_val_mse:
                    with torch.no_grad():
                        val_ddpm_action_mses = []
                        val_ddim_action_mses = []
                        for dataset_idx in range(self.num_datasets):
                            # Get the validation batch for this dataset
                            val_sampling_batch = val_sampling_batches[dataset_idx]
                            val_batch = dict_apply(val_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            val_obs_dict = {'obs': batch['obs'], 'target': batch['target']}
                            val_gt_action = batch['action']
                            
                            # Evaluate MSE when diffusing with DDPM
                            if cfg.training.eval_mse_DDPM:
                                pred_action = policy.predict_action(val_obs_dict, use_DDIM=False)['action_pred']
                                mse = torch.nn.functional.mse_loss(pred_action, val_gt_action)
                                step_log[f'val_ddpm_mse_{dataset_idx}'] = mse.item()
                                val_ddpm_action_mses.append(mse.item())
                            
                            # Evaluate MSE when diffusing with DDPM
                            if cfg.training.eval_mse_DDIM:
                                pred_action = policy.predict_action(val_obs_dict, use_DDIM=True)['action_pred']
                                mse = torch.nn.functional.mse_loss(pred_action, val_gt_action)
                                step_log[f'val_ddim_mse_{dataset_idx}'] = mse.item()
                                val_ddim_action_mses.append(mse.item())
                        
                        # Compute weighted val action MSEs
                        if cfg.training.eval_mse_DDPM:
                            step_log['val_ddpm_mse'] = self.weighted_sum(self.sample_probabilities, val_ddpm_action_mses)
                        if cfg.training.eval_mse_DDIM:
                            step_log['val_ddim_mse'] = self.weighted_sum(self.sample_probabilities, val_ddim_action_mses)

                        del val_batch
                        del val_obs_dict
                        del val_gt_action
                        del pred_action
                        del mse
                
                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if save_last_ckpt:
                        self.save_checkpoint()
                    if save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_paths = []
                    for i, topk_manager in enumerate(topk_managers):
                        protected_ckpts = self._get_protected_paths(i, topk_managers)
                        ckpt_path = topk_manager.get_ckpt_path(metric_dict, protected_ckpts)
                        topk_ckpt_paths.append(ckpt_path)

                    for i, topk_ckpt_path in enumerate(topk_ckpt_paths):
                        if topk_ckpt_path is not None:
                            self.save_checkpoint(path=topk_ckpt_path)
                            break
                # last epoch => save last checkpoint
                if self.epoch == cfg.training.num_epochs-1:
                    self.save_checkpoint()
                # ========= eval end for this epoch ==========
                policy.train()
                self.discriminator.train()

                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

                # End of epoch
    
    def compute_denoiser_loss(self, trajectory, noise, pred):
        pred_type = self.model.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        
        loss = F.mse_loss(pred, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
    
    def compute_discriminator_loss(self, labels, pred):
        loss = self.bce_loss(pred, labels.float())
        return loss

    def compute_discriminator_accuracy(self, labels, pred):
        pred_labels = (pred > 0).float()
        num_zero_labels = (labels == 0).sum().float().item()
        num_one_labels = (labels == 1).sum().float().item()
        true_negatives = ((pred_labels == 0) & (labels == 0)).sum().item()
        true_positives = ((pred_labels == 1) & (labels == 1)).sum().item()
        if num_zero_labels == 0:
            return {
                'true_negatives_accuracy': float('nan'),
                'true_positives_accuracy': true_positives / num_one_labels,
                'accuracy': true_positives / num_one_labels,
                'balanced_accuracy': true_positives / num_one_labels,
                'discrimination': float('nan')
            }
        elif num_one_labels == 0:
            return {
                'true_negatives_accuracy': true_negatives / num_zero_labels,
                'true_positives_accuracy': float('nan'),
                'accuracy': true_negatives / num_zero_labels,
                'balanced_accuracy': true_negatives / num_zero_labels,
                'discrimination': float('nan')
            }
        else:
            return {
                'true_negatives_accuracy': true_negatives / num_zero_labels,
                'true_positives_accuracy': true_positives / num_one_labels,
                'accuracy': (true_negatives + true_positives) / (num_zero_labels + num_one_labels),
                'balanced_accuracy': 0.5 * (true_negatives / num_zero_labels + true_positives / num_one_labels),
                'discrimination': abs((num_zero_labels - true_negatives) / num_zero_labels - true_positives / num_one_labels)
            }
    
    def weighted_sum(self, weights, values):
        assert len(weights) == len(values)
        ws = 0
        for i in range(len(weights)):
            ws += weights[i] * values[i]
        return ws


    def _print_dataset_diagnostics(self, cfg, dataset, train_dataloader, val_dataloaders):
        print()
        print("============= Dataset Diagnostics =============")
        print(f"Number of datasets: {self.num_datasets}")
        print(f"Sample probabilities: {self.sample_probabilities}")
        print(f"[Training] Number of batches: {len(train_dataloader)}")
        for i in range(self.num_datasets):
            print(f"[Val {i}] Number of batches: {len(val_dataloaders[i])}")
        print()

        for i in range(self.num_datasets):
            val_dataset = dataset.get_validation_dataset(i)
            print(f"Dataset {i}: {dataset.zarr_paths[i]}")
            print("------------------------------------------------")
            print(f"Number of training demonstrations: {np.sum(dataset.train_masks[i])}")
            print(f"Number of validation demonstrations: {np.sum(dataset.val_masks[i])}")
            print(f"Number of training samples: {len(dataset.samplers[i])}")
            print(f"Number of validation samples: {len(val_dataset)}")
            print(f"Approx. number of training batches: {len(dataset.samplers[i]) // cfg.dataloader.batch_size}")
            print(f"Approx. number of validation batches: {len(val_dataset) // cfg.val_dataloader.batch_size}")
            print(f"Sample probability: {self.sample_probabilities[i]}")
            print()
        print("================================================")

    def _get_protected_paths(self, topk_manager_idx, topk_managers):
        """
        Returns the paths that should not be deleted by topk_manager
        """
        if len(topk_managers) == 1:
            return set()
        
        topk_manager = topk_managers[topk_manager_idx]

        protected_paths = set()
        for manager in topk_managers:
            protected_paths.update(manager.get_path_value_map().keys())
        
        # Remove the paths that can be deleted
        # If a ckpt is ONLY being tracked by topk_manager, it can be deleted
        for path in topk_manager.get_path_value_map().keys():
            protected = False
            for i, manager in enumerate(topk_managers):
                if i == topk_manager_idx:
                    continue
                if path in manager.get_path_value_map().keys():
                    protected = True
                    break
            if not protected:
                protected_paths.remove(path)
        
        return protected_paths
                

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetHybridWorkspaceNoEnv(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
