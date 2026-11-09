from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

from diffusion_policy.model.common.slice import slice_episode

avg_natural_margin_mean_debug = 0

class CplDiffusionTransformerLowdimPolicy(BaseLowdimPolicy):
    def __init__(
            self,
            model: TransformerForDiffusion,
            noise_scheduler: DDPMScheduler,
            horizon,
            obs_dim,
            action_dim,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            obs_as_cond=False,
            pred_action_steps_only=False,
            beta=1.0,
            bias_reg=1.0,
            ignore_equal_pref=False,
            clip_margin=None,
            smooth_label=0,
            confidence_weight=False,
            cw_temperature=0.03,
            unclip_win=False,
            # parameters passed to step
            **kwargs
    ):
        super().__init__()
        if pred_action_steps_only:
            assert obs_as_cond

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs
        self.gamma = 0.999

        # Parameters for preference learning
        self.beta = beta
        self.bias_reg = bias_reg
        self.ignore_equal_pref = ignore_equal_pref
        self.clip_margin = clip_margin  # None = disabled
        self.smooth_label = smooth_label    # 0 = disabled
        self.confidence_weight = confidence_weight
        self.cw_temperature = cw_temperature
        self.unclip_win = unclip_win

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            cond = nobs[:,:To]
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not self.obs_as_cond:
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        return self.model.configure_optimizers(
                weight_decay=weight_decay, 
                learning_rate=learning_rate, 
                betas=tuple(betas))

    def compute_loss_cpl_kl(
            self, batch, epoch, ref_model, n_epoch_sft=0, sft_type="pos", stride=10, equal_pref_threshold=0.05,
            debug=False
    ):
        assert sft_type in ["pos", "both"]
        observations_1, actions_1, votes_1, length_1 = batch["obs"], batch["action"], batch["votes"], batch["length"]
        observations_2, actions_2, votes_2, length_2 = batch["obs_2"], batch["action_2"], batch["votes_2"], batch["length_2"]

        diff = torch.abs(votes_1 - votes_2)
        mask_not_equal_pref = torch.squeeze(diff > equal_pref_threshold, dim=-1).type(torch.float32)
        if self.confidence_weight:
            temperature = self.cw_temperature
            confidence_weight = torch.sigmoid((diff - equal_pref_threshold) / temperature)

        # Swap so segment 1 is always the preferred/winner trajectory
        mask_pref_right = ((votes_1 < votes_2) & (diff > equal_pref_threshold)).squeeze(-1)
        actions_1[mask_pref_right], actions_2[mask_pref_right] = actions_2[mask_pref_right], actions_1[mask_pref_right]
        observations_1[mask_pref_right], observations_2[mask_pref_right] = observations_2[mask_pref_right], observations_1[mask_pref_right]
        length_1[mask_pref_right], length_2[mask_pref_right] = length_2[mask_pref_right], length_1[mask_pref_right]

        # Normalize obs and action
        nbatch_1 = self.normalizer.normalize({'obs': observations_1, 'action': actions_1})
        nbatch_2 = self.normalizer.normalize({'obs': observations_2, 'action': actions_2})

        # Slice to make it compatible with action chunking
        obs_1, action_1 = slice_episode(nbatch_1['obs'], horizon=self.horizon, stride=stride), slice_episode(nbatch_1['action'], horizon=self.horizon, stride=stride)
        obs_2, action_2 = slice_episode(nbatch_2['obs'], horizon=self.horizon, stride=stride), slice_episode(nbatch_2['action'], horizon=self.horizon, stride=stride)
        assert (len(obs_1) == len(obs_2)) and (len(action_1) == len(action_2))
        assert not self.pred_action_steps_only and self.obs_as_cond and self.noise_scheduler.config.prediction_type == 'epsilon'

        bsz = obs_1[0].shape[0]
        n_train_denoise_timesteps = self.noise_scheduler.config.num_train_timesteps
        use_bc = True if epoch < n_epoch_sft else False

        # timesteps_1 = torch.randint(0, n_train_denoise_timesteps, (bsz,), device=self.device).long()
        # timesteps_2 = torch.randint(0, n_train_denoise_timesteps, (bsz,), device=self.device).long()

        valid_count_1 = torch.zeros(bsz, device=self.device)
        valid_count_2 = torch.zeros(bsz, device=self.device)
        segment_loss_1, segment_loss_2, imitation_loss = 0.0, 0.0, 0.0
        if debug:
            raw_margin_sum_1, raw_margin_sum_2 = 0.0, 0.0
            max_raw_margin = 0.0

        for i in range(len(obs_1)):
            timesteps = torch.randint(0, n_train_denoise_timesteps, (bsz,), device=self.device).long()
            timesteps_1 = timesteps
            timesteps_2 = timesteps

            obs_1_slice, action_1_slice = obs_1[i], action_1[i]
            obs_2_slice, action_2_slice = obs_2[i], action_2[i]

            trajectory_1, cond_1 = action_1_slice, obs_1_slice[:, :self.n_obs_steps, :]
            trajectory_2, cond_2 = action_2_slice, obs_2_slice[:, :self.n_obs_steps, :]

            # This mask used to ignore padded states at the last segments
            actual_timesteps = i * stride + torch.arange(self.horizon, device=self.device)
            step_mask_1 = (actual_timesteps.unsqueeze(0) < length_1.view(-1, 1)).float()
            step_mask_2 = (actual_timesteps.unsqueeze(0) < length_2.view(-1, 1)).float()
            valid_count_1 += step_mask_1.sum(dim=-1)  # Accumulate total valid timesteps evaluated
            valid_count_2 += step_mask_2.sum(dim=-1)  # Accumulate total valid timesteps evaluated
            discounts = (self.gamma ** actual_timesteps).unsqueeze(0)  # Compute gamma discounts (shape of [1, horizon])

            # Combine discounts and step masks
            weights_1, weights_2 = discounts * step_mask_1, discounts * step_mask_2     # [bsz, horizon]

            condition_mask = self.mask_generator(trajectory_1.shape)  # generate inpainting mask
            loss_mask = (~condition_mask).float()  # compute loss mask

            # Compute for segment 1 (left)
            noise_1 = torch.randn(trajectory_1.shape, device=self.device)  # Sample noise to add to actions
            noisy_trajectory_1 = self.noise_scheduler.add_noise(trajectory_1, noise_1, timesteps_1)  # Add noise to clean action
            noisy_trajectory_1[condition_mask] = trajectory_1[condition_mask]  # apply conditioning
            pred_1 = self.model(noisy_trajectory_1, timesteps_1, cond_1)  # Predict the noise
            
            # Compute for segment 2 (right)
            if (not use_bc) or (use_bc and sft_type == "both"):
                noise_2 = torch.randn(trajectory_2.shape, device=self.device)  # Sample noise to add to actions
                noisy_trajectory_2 = self.noise_scheduler.add_noise(trajectory_2, noise_2, timesteps_2)  # Add noise to clean action
                noisy_trajectory_2[condition_mask] = trajectory_2[condition_mask]   # apply conditioning
                pred_2 = self.model(noisy_trajectory_2, timesteps_2, cond_2)  # Predict the noise
            else:
                noise_2 = noisy_trajectory_2 = pred_2 = None

            if use_bc:
                if sft_type == "pos":
                    imitation_loss_1 = torch.norm((pred_1 - noise_1) * loss_mask, dim=-1) ** 2
                    imitation_loss += torch.sum(imitation_loss_1 * step_mask_1, dim=-1)
                elif sft_type == "both":
                    imitation_loss_1 = torch.norm((pred_1 - noise_1) * loss_mask, dim=-1) ** 2
                    imitation_loss_2 = torch.norm((pred_2 - noise_2) * loss_mask, dim=-1) ** 2
                    imitation_loss += (torch.sum(imitation_loss_1 * step_mask_1, dim=-1) + torch.sum(imitation_loss_2 * step_mask_2, dim=-1))
                else:
                    raise NotImplementedError

            else:   # preference learning
                with torch.no_grad():
                    ref_pred_1 = ref_model(noisy_trajectory_1, timesteps_1, cond_1)
                    ref_pred_2 = ref_model(noisy_trajectory_2, timesteps_2, cond_2)

                slice_loss_1 = (torch.norm((pred_1 - noise_1) * loss_mask, dim=-1) ** 2 - torch.norm((ref_pred_1 - noise_1) * loss_mask, dim=-1) ** 2)
                slice_loss_2 = (torch.norm((pred_2 - noise_2) * loss_mask, dim=-1) ** 2 - torch.norm((ref_pred_2 - noise_2) * loss_mask, dim=-1) ** 2)

                if debug:
                    with torch.no_grad():
                        # Track absolute difference, zeroing out invalid padded steps
                        abs_slice_1 = torch.abs(slice_loss_1) * step_mask_1
                        abs_slice_2 = torch.abs(slice_loss_2) * step_mask_2

                        raw_margin_sum_1 += abs_slice_1.sum(dim=-1)
                        raw_margin_sum_2 += abs_slice_2.sum(dim=-1)

                        # Find the largest single-step margin spike in this batch
                        batch_max = max(abs_slice_1.max().item(), abs_slice_2.max().item())
                        max_raw_margin = max(max_raw_margin, batch_max)

                if self.clip_margin is not None:
                    # TODO: Test this Soft Clip later to avoid abruptly cut the gradient
                    # slice_loss_1 = self.clip_margin * torch.tanh(slice_loss_1 / self.clip_margin)
                    # slice_loss_2 = self.clip_margin * torch.tanh(slice_loss_2 / self.clip_margin)
                    if not self.unclip_win:
                        slice_loss_1 = torch.clamp(slice_loss_1, min=-self.clip_margin, max=self.clip_margin)
                    slice_loss_2 = torch.clamp(slice_loss_2, min=-self.clip_margin, max=self.clip_margin)

                if self.ignore_equal_pref:
                    segment_loss_1 += torch.sum(slice_loss_1 * weights_1, dim=-1) * mask_not_equal_pref
                    segment_loss_2 += torch.sum(slice_loss_2 * weights_2, dim=-1) * mask_not_equal_pref
                else:
                    segment_loss_1 += torch.sum(slice_loss_1 * weights_1, dim=-1)
                    segment_loss_2 += torch.sum(slice_loss_2 * weights_2, dim=-1)

        if use_bc:
            if sft_type == "pos":
                norm_factor = torch.clamp(valid_count_1, min=1.0)
            else:   # both
                norm_factor = (torch.clamp(valid_count_1, min=1.0) + torch.clamp(valid_count_2, min=1.0))

            imitation_loss = imitation_loss / norm_factor
            loss_total = torch.mean(imitation_loss)
            mle_loss_1, accuracy = 0.0, 0.0
        else:
            norm_factor_1 = torch.clamp(valid_count_1 / self.horizon, min=1.0)  # num of chunk that calculated
            norm_factor_2 = torch.clamp(valid_count_2 / self.horizon, min=1.0)  # num of chunk that calculated

            segment_loss_1 = -self.beta * n_train_denoise_timesteps * segment_loss_1 / norm_factor_1
            segment_loss_2 = -self.beta * n_train_denoise_timesteps * segment_loss_2 / norm_factor_2

            z = segment_loss_1 - self.bias_reg * segment_loss_2

            epsilon_smooth = self.smooth_label
            if epsilon_smooth == 0:
                # Standard CPL
                mle_loss_1 = -F.logsigmoid(z)
            else:
                # Conservative CPL blends the forward and reversed preferences
                mle_loss_1 = -(1 - epsilon_smooth) * F.logsigmoid(z) - epsilon_smooth * F.logsigmoid(-z)

            if self.confidence_weight:
                # Squeeze confidence weight to match mle_loss_1 shape (B,)
                cw = confidence_weight.squeeze(-1)

            # mle_loss_1 = -F.logsigmoid(segment_loss_1 - self.bias_reg * segment_loss_2)
            if self.ignore_equal_pref:
                # Average ONLY pairs that have unequal preferences
                valid_pairs = torch.clamp(mask_not_equal_pref.sum(), min=1.0)
                if self.confidence_weight:
                    # Apply hard mask AND soft confidence weight
                    weighted_loss = mle_loss_1 * mask_not_equal_pref * cw
                    loss_total = weighted_loss.sum() / valid_pairs
                else:
                    loss_total = (mle_loss_1 * mask_not_equal_pref).sum() / valid_pairs
                # Ignore tied pairs so they don't count as incorrect
                with torch.no_grad():
                    correct_preds = (segment_loss_1.detach() > segment_loss_2.detach()).float()
                    accuracy = ((correct_preds * mask_not_equal_pref).sum() / valid_pairs).item()
            else:
                if self.confidence_weight:
                    # Apply soft confidence weight to ALL pairs
                    weighted_loss = mle_loss_1 * cw
                    # Use weighted mean to maintain stable gradient magnitudes
                    loss_total = weighted_loss.sum() / torch.clamp(cw.sum(), min=1.0)
                else:
                    loss_total = torch.mean(mle_loss_1)
                with torch.no_grad():
                    accuracy = (segment_loss_1.detach() > segment_loss_2.detach()).float().mean().item()

            if debug:
                # Calculate the average Natural Margin per valid timestep ---
                avg_margin_1 = (raw_margin_sum_1 / torch.clamp(valid_count_1, min=1.0)).mean().item()
                avg_margin_2 = (raw_margin_sum_2 / torch.clamp(valid_count_2, min=1.0)).mean().item()
                avg_natural_margin_mean = (avg_margin_1 + avg_margin_2) / 2.0

                # Calculate suitable beta targeting a max logit of 3.0
                target_max_logit = 3.0

                print(f"\n--- DEBUG INFO ---")
                print(f"natural_margin_mean={avg_natural_margin_mean:.4f}, x1.5={avg_natural_margin_mean * 1.5:.4f}, x2={avg_natural_margin_mean * 2.0:.4f}")
                print(f"natural_margin_max={max_raw_margin:.4f}")

                assumed_clip_margin = 1.0
                suitable_beta = target_max_logit / (n_train_denoise_timesteps * self.horizon * assumed_clip_margin * (1.0 + self.bias_reg))
                print(f"Suitable beta (clip_margin={assumed_clip_margin})={suitable_beta:.8f}")
                assumed_clip_margin = avg_natural_margin_mean * 1.5
                suitable_beta = target_max_logit / (n_train_denoise_timesteps * self.horizon * assumed_clip_margin * (1.0 + self.bias_reg))
                print(f"Suitable beta (x1.5: clip_margin={assumed_clip_margin})={suitable_beta:.8f}")
                assumed_clip_margin = avg_natural_margin_mean * 2.0
                suitable_beta = target_max_logit / (n_train_denoise_timesteps * self.horizon * assumed_clip_margin * (1.0 + self.bias_reg))
                print(f"Suitable beta (x2.0: clip_margin={assumed_clip_margin})={suitable_beta:.8f}")
                print(f"------------------------------------------------------\n")

        loss_metrics = {
            'mle_loss_1': mle_loss_1.mean().item() if isinstance(mle_loss_1, torch.Tensor) else mle_loss_1,
            'segment_loss_1': segment_loss_1.mean().item() if isinstance(segment_loss_1, torch.Tensor) else segment_loss_1,
            'segment_loss_2': segment_loss_2.mean().item() if isinstance(segment_loss_2, torch.Tensor) else segment_loss_2,
            'bc_loss': imitation_loss.mean().item() if isinstance(imitation_loss, torch.Tensor) else imitation_loss,
            'accuracy': accuracy
        }

        if isinstance(segment_loss_1, torch.Tensor) and isinstance(segment_loss_2, torch.Tensor):
            scale = self.beta * n_train_denoise_timesteps
            # Argument to logsigmoid: |reward_logit| >> 5 means sigmoid is saturated -> gradients vanish
            reward_logit = (segment_loss_1 - self.bias_reg * segment_loss_2).mean().item()
            # Raw log-ratios (before beta scaling): < 0 = model improved vs ref, > 0 = drifted away
            # log_ratio_win should be ≤ 0 (improving on preferred); log_ratio_lose >> 0 = collapse
            log_ratio_win  = (-segment_loss_1 / scale).mean().item()
            log_ratio_lose = (-segment_loss_2 / scale).mean().item()
            loss_metrics.update({
                'reward_logit': reward_logit,
                'log_ratio_win': log_ratio_win,
                'log_ratio_lose': log_ratio_lose,
            })
        if self.ignore_equal_pref:
            loss_metrics.update({'total_mask_not_equal': mask_not_equal_pref.sum()})
        return loss_total, loss_metrics

    def compute_loss_sft(self, batch, stride=1, equal_pref_threshold=0.05):
        observations_1, actions_1, votes_1, length_1 = batch["obs"], batch["action"], batch["votes"], batch["length"]
        observations_2, actions_2, votes_2, length_2 = batch["obs_2"], batch["action_2"], batch["votes_2"], batch["length_2"]

        diff = torch.abs(votes_1 - votes_2)

        # Swap so segment 1 is always the preferred/winner trajectory
        mask_pref_right = ((votes_1 < votes_2) & (diff > equal_pref_threshold)).squeeze(-1)
        actions_1[mask_pref_right], actions_2[mask_pref_right] = actions_2[mask_pref_right], actions_1[mask_pref_right]
        observations_1[mask_pref_right], observations_2[mask_pref_right] = observations_2[mask_pref_right], observations_1[mask_pref_right]
        length_1[mask_pref_right], length_2[mask_pref_right] = length_2[mask_pref_right], length_1[mask_pref_right]

        nbatch_1 = self.normalizer.normalize({'obs': observations_1, 'action': actions_1})

        # Slice to make it compatible with action chunking
        obs_1, action_1 = slice_episode(nbatch_1['obs'], horizon=self.horizon, stride=stride), slice_episode(nbatch_1['action'], horizon=self.horizon, stride=stride)
        assert not self.pred_action_steps_only and self.obs_as_cond and self.noise_scheduler.config.prediction_type == 'epsilon'

        bsz = obs_1[0].shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=self.device).long()

        # Behavior cloning loss for only preferred segments (i.e., segment 1 (left))
        idx = torch.randint(0, len(obs_1), (bsz,), device=self.device)  # Sampling chunk from preferred segment (similar to BC)
        batch_idx = torch.arange(bsz, device=self.device)
        obs_1_slice, action_1_slice = obs_1[idx, batch_idx], action_1[idx, batch_idx]

        trajectory = action_1_slice
        cond = obs_1_slice[:, :self.n_obs_steps, :]

        condition_mask = self.mask_generator(trajectory.shape)  # generate inpainting mask
        loss_mask = (~condition_mask).float()
        noise = torch.randn(trajectory.shape, device=self.device)   # Sample noise that we'll add to the images
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps) # Add noise to clean action
        noisy_trajectory[condition_mask] = trajectory[condition_mask]   # apply conditioning
        pred = self.model(noisy_trajectory, timesteps, cond)    # Predict the noise

        # This mask used to ignore paddings at the last slice of segments
        mask = (self.horizon + idx * stride) <= length_1
        mask = torch.squeeze(mask.float(), dim=-1)
        imitation_loss = torch.norm((pred - noise) * loss_mask, dim=-1) ** 2
        imitation_loss = torch.sum(imitation_loss, dim=-1) * mask
        loss = imitation_loss
        loss_metrics = {
            'bc_loss': imitation_loss.mean().item(),
        }
        return torch.mean(loss), loss_metrics