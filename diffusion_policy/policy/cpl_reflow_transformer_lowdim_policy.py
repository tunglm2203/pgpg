from typing import Dict
import torch
import torch.nn.functional as F
from torch import Tensor
from collections import namedtuple

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion

from diffusion_policy.model.common.slice import slice_episode

Sample = namedtuple("Sample", "trajectories chains")


class CplReFlowTransformerLowdimPolicy(BaseLowdimPolicy):
    def __init__(
            self,
            model: TransformerForDiffusion,
            horizon,
            obs_dim,
            action_dim,
            n_action_steps,
            n_obs_steps,
            sample_t_type: str = 'uniform',
            test_denoising_steps=20,
            test_clip_intermediate_actions=True,
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
        self.model = model

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.data_shape = (self.horizon, self.action_dim)
        self.sample_t_type = sample_t_type
        self.test_denoising_steps = test_denoising_steps
        self.test_clip_intermediate_actions = test_clip_intermediate_actions
        self.act_range = (None, None)
        self.kwargs = kwargs
        self.gamma = 0.999
        assert self.n_action_steps <= self.horizon, f"To={self.n_obs_steps}, Ta={self.n_action_steps}, Tp={self.horizon} are incompatible"

        # Parameters for preference learning
        self.beta = beta
        self.bias_reg = bias_reg
        self.ignore_equal_pref = ignore_equal_pref
        self.clip_margin = clip_margin  # None = disabled
        self.smooth_label = smooth_label    # 0 = disabled
        self.confidence_weight = confidence_weight
        self.cw_temperature = cw_temperature
        self.unclip_win = unclip_win

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])

        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        cond = nobs[:, :To]

        samples = self.sample(
            cond,
            inference_steps=self.test_denoising_steps,
            record_intermediate=False,
            clip_intermediate_actions=self.test_clip_intermediate_actions
        )

        # unnormalize prediction
        naction_pred = samples.trajectories
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
        act_min = self.normalizer['action'].params_dict.input_stats.min.min().item()
        act_max = self.normalizer['action'].params_dict.input_stats.max.max().item()
        self.act_range = (act_min, act_max)
    
    def generate_trajectory(self, x1: Tensor, x0: Tensor, t: Tensor) -> Tensor:
        """Generate rectified flow trajectory xt = t * x1 + (1 - t) * x0.

        Args:
            x1: Target data tensor of shape (batch_size, horizon, action_dim).
            x0: Initial noise tensor of shape (batch_size, horizon, action_dim).
            t: Time step tensor of shape (batch_size,).

        Returns:
            Tensor: Interpolated trajectory xt of shape (batch_size, horizon, action_dim).
        """
        t_ = (torch.ones_like(x1, device=self.device) * t.view(x1.shape[0], 1, 1)).to(
            self.device)  # ReinFlow Authors revised on 04/23/2025
        xt = t_ * x1 + (1 - t_) * x0
        return xt

    def sample_time(self, batch_size: int, time_sample_type: str = 'uniform', **kwargs) -> Tensor:
        """Sample time steps from a specified distribution in [0, 1).

        Args:
            batch_size: Number of time samples to generate.
            time_sample_type: Type of distribution ('uniform', 'logitnormal', 'beta').
            **kwargs: Additional parameters for non-uniform distributions.

        Returns:
            Tensor: Time samples of shape (batch_size,).

        Raises:
            ValueError: If time_sample_type is not supported.
        """
        supported_time_sample_type = ['uniform', 'logitnormal', 'beta']
        if time_sample_type == 'uniform':
            return torch.rand(batch_size, device=self.device)
        elif time_sample_type == 'logitnormal':
            m = kwargs.get("m", 0)  # Default mean
            s = kwargs.get("s", 1)  # Default standard deviation
            normal_samples = torch.normal(mean=m, std=s, size=(batch_size,), device=self.device)
            logit_normal_samples = (1 / (1 + torch.exp(-normal_samples))).to(self.device)
            return logit_normal_samples
        elif time_sample_type == 'beta':
            alpha = kwargs.get("alpha", 1.5)  # Default alpha
            beta = kwargs.get("beta", 1.0)  # Default beta
            s = kwargs.get("s", 0.999)  # Default cutoff
            beta_distribution = torch.distributions.Beta(alpha, beta)
            beta_sample = beta_distribution.sample((batch_size,)).to(self.device)
            tau = s * (1 - beta_sample)
            return tau
        else:
            raise ValueError(
                f'Unknown time_sample_type = {time_sample_type}. Supported types: {supported_time_sample_type}')

    def generate_target(self, x1: Tensor) -> tuple:
        """Generate training targets for the velocity field.

        Args:
            x1: Real data tensor of shape (batch_size, horizon, action_dim).

        Returns:
            tuple: Contains (xt, t, obs) and v where:
                - xt: Corrupted data tensor of shape (batch_size, horizon, action_dim).
                - t: Time step tensor of shape (batch_size,).
                - v: Target velocity tensor of shape (batch_size, horizon, action_dim).
        """
        t = self.sample_time(batch_size=x1.shape[0], time_sample_type=self.sample_t_type)
        x0 = torch.randn(x1.shape, dtype=torch.float32, device=self.device)
        xt = self.generate_trajectory(x1, x0, t)
        v = x1 - x0
        return (xt, t), v

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
        horizon_in_dataloader = self.horizon + self.n_obs_steps - 1
        obs_1, action_1 = slice_episode(nbatch_1['obs'], horizon=horizon_in_dataloader, stride=stride), slice_episode(nbatch_1['action'], horizon=horizon_in_dataloader, stride=stride)
        obs_2, action_2 = slice_episode(nbatch_2['obs'], horizon=horizon_in_dataloader, stride=stride), slice_episode(nbatch_2['action'], horizon=horizon_in_dataloader, stride=stride)
        assert (len(obs_1) == len(obs_2)) and (len(action_1) == len(action_2))

        bsz = obs_1[0].shape[0]
        use_bc = True if epoch < n_epoch_sft else False

        # timesteps_1 = self.sample_time(batch_size=bsz, time_sample_type=self.sample_t_type)
        # timesteps_2 = self.sample_time(batch_size=bsz, time_sample_type=self.sample_t_type)

        valid_count_1 = torch.zeros(bsz, device=self.device)
        valid_count_2 = torch.zeros(bsz, device=self.device)
        segment_loss_1, segment_loss_2, imitation_loss = 0.0, 0.0, 0.0
        if debug:
            raw_margin_sum_1, raw_margin_sum_2 = 0.0, 0.0
            max_raw_margin = 0.0

        for i in range(len(obs_1)):
            timesteps = self.sample_time(batch_size=bsz, time_sample_type=self.sample_t_type)
            timesteps_1 = timesteps
            timesteps_2 = timesteps

            obs_1_slice, action_1_slice = obs_1[i], action_1[i]
            obs_2_slice, action_2_slice = obs_2[i], action_2[i]

            cond_seg1, x1_seg1 = obs_1_slice[:, :self.n_obs_steps, :], action_1_slice[:, self.n_obs_steps - 1:, :]
            cond_seg2, x1_seg2 = obs_2_slice[:, :self.n_obs_steps, :], action_2_slice[:, self.n_obs_steps - 1:, :]

            # This mask used to ignore padded states at the last segments
            action_slice_idx = self.n_obs_steps - 1
            actual_timesteps = i * stride + torch.arange(horizon_in_dataloader, device=self.device)
            step_mask_1 = (actual_timesteps.unsqueeze(0) < length_1.view(-1, 1)).float()
            step_mask_2 = (actual_timesteps.unsqueeze(0) < length_2.view(-1, 1)).float()
            # Slice masks to match x1_seg1's temporal dimension
            loss_mask_1 = step_mask_1[:, action_slice_idx:]
            loss_mask_2 = step_mask_2[:, action_slice_idx:]

            # Accumulate valid steps based ONLY on the predicted action portion
            valid_count_1 += loss_mask_1.sum(dim=-1)
            valid_count_2 += loss_mask_2.sum(dim=-1)
            discounts = (self.gamma ** actual_timesteps).unsqueeze(0)
            # Slice weights to match
            weights_1 = discounts[:, action_slice_idx:] * loss_mask_1
            weights_2 = discounts[:, action_slice_idx:] * loss_mask_2

            x0_seg1 = torch.randn(x1_seg1.shape, dtype=torch.float32, device=self.device)
            xt_seg1 = self.generate_trajectory(x1_seg1, x0_seg1, timesteps_1)
            v_seg1 = x1_seg1 - x0_seg1
            v_hat_seg1 = self.model(xt_seg1, timesteps_1, cond_seg1)  # Predict the velocity field

            # Compute for segment 2 (right)
            if (not use_bc) or (use_bc and sft_type == "both"):
                x0_seg2 = torch.randn(x1_seg2.shape, dtype=torch.float32, device=self.device)
                xt_seg2 = self.generate_trajectory(x1_seg2, x0_seg2, timesteps_2)
                v_seg2 = x1_seg2 - x0_seg2
                v_hat_seg2 = self.model(xt_seg2, timesteps_2, cond_seg2)
            else:
                v_seg2 = v_hat_seg2 = None

            if use_bc:
                if sft_type == "pos":
                    imitation_loss_1 = torch.norm((v_hat_seg1 - v_seg1), dim=-1) ** 2
                    imitation_loss += torch.sum(imitation_loss_1 * loss_mask_1, dim=-1)
                elif sft_type == "both":
                    imitation_loss_1 = torch.norm((v_hat_seg1 - v_seg1), dim=-1) ** 2
                    imitation_loss_2 = torch.norm((v_hat_seg2 - v_seg2), dim=-1) ** 2
                    imitation_loss += (torch.sum(imitation_loss_1 * loss_mask_1, dim=-1) + torch.sum(imitation_loss_2 * loss_mask_2, dim=-1))
                else:
                    raise NotImplementedError

            else:
                # preference learning
                with torch.no_grad():
                    ref_v_hat_seg1 = ref_model(xt_seg1, timesteps_1, cond_seg1)
                    ref_v_hat_seg2 = ref_model(xt_seg2, timesteps_2, cond_seg2)

                slice_loss_1 = (torch.norm((v_hat_seg1 - v_seg1), dim=-1) ** 2 - torch.norm((ref_v_hat_seg1 - v_seg1), dim=-1) ** 2)
                slice_loss_2 = (torch.norm((v_hat_seg2 - v_seg2), dim=-1) ** 2 - torch.norm((ref_v_hat_seg2 - v_seg2), dim=-1) ** 2)

                if debug:
                    with torch.no_grad():
                        # Track absolute difference, zeroing out invalid padded steps
                        abs_slice_1 = torch.abs(slice_loss_1) * loss_mask_1
                        abs_slice_2 = torch.abs(slice_loss_2) * loss_mask_2

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

            segment_loss_1 = -self.beta * segment_loss_1 / norm_factor_1
            segment_loss_2 = -self.beta * segment_loss_2 / norm_factor_2

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
                    valid_weight_sum = torch.clamp((mask_not_equal_pref * cw).sum(), min=1.0)
                    loss_total = weighted_loss.sum() / valid_weight_sum
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
                suitable_beta = target_max_logit / (self.horizon * assumed_clip_margin * (1.0 + self.bias_reg))
                print(f"Suitable beta (clip_margin={assumed_clip_margin})={suitable_beta:.8f}")
                assumed_clip_margin = avg_natural_margin_mean * 1.5
                suitable_beta = target_max_logit / (self.horizon * assumed_clip_margin * (1.0 + self.bias_reg))
                print(f"Suitable beta (x1.5: clip_margin={assumed_clip_margin})={suitable_beta:.8f}")
                assumed_clip_margin = avg_natural_margin_mean * 2.0
                suitable_beta = target_max_logit / (self.horizon * assumed_clip_margin * (1.0 + self.bias_reg))
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
            scale = self.beta
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
        mask_not_equal_pref = torch.squeeze(diff > equal_pref_threshold, dim=-1).type(torch.float32)

        # Convert to all left segments are preferred, i.e., actions_1 is preferred over actions_2
        # Swap so segment 1 is always the preferred/winner trajectory
        mask_pref_right = ((votes_1 < votes_2) & (diff > equal_pref_threshold)).squeeze(-1)
        actions_1[mask_pref_right], actions_2[mask_pref_right] = actions_2[mask_pref_right], actions_1[mask_pref_right]
        observations_1[mask_pref_right], observations_2[mask_pref_right] = observations_2[mask_pref_right], \
        observations_1[mask_pref_right]
        length_1[mask_pref_right], length_2[mask_pref_right] = length_2[mask_pref_right], length_1[mask_pref_right]

        # Normalize obs and action
        nbatch_1 = self.normalizer.normalize({'obs': observations_1, 'action': actions_1})
        nbatch_2 = self.normalizer.normalize({'obs': observations_2, 'action': actions_2})

        # Slice to make it compatible with action chunking
        horizon_in_dataloader = self.horizon + self.n_obs_steps - 1
        obs_1, action_1 = slice_episode(nbatch_1['obs'], horizon=horizon_in_dataloader, stride=stride), slice_episode(nbatch_1['action'], horizon=horizon_in_dataloader, stride=stride)
        obs_2, action_2 = slice_episode(nbatch_2['obs'], horizon=horizon_in_dataloader, stride=stride), slice_episode(nbatch_2['action'], horizon=horizon_in_dataloader, stride=stride)
        assert (len(obs_1) == len(obs_2)) and (len(action_1) == len(action_2))

        bsz = obs_1[0].shape[0]
        timesteps = self.sample_time(batch_size=bsz, time_sample_type=self.sample_t_type)

        # Behavior cloning loss for only preferred segments (i.e., segment 1 (left))
        idx = torch.randint(0, len(obs_1), (bsz,), device=self.device)  # Sampling chunk from preferred segment (similar to BC)
        obs_1_slice, action_1_slice = obs_1[idx], action_1[idx]
        cond_seg1, x1_seg1 = obs_1_slice[:, :self.n_obs_steps, :], action_1_slice[:, self.n_obs_steps - 1:, :]

        x0_seg1 = torch.randn(x1_seg1.shape, dtype=torch.float32, device=self.device)
        xt_seg1 = self.generate_trajectory(x1_seg1, x0_seg1, timesteps)
        v_seg1 = x1_seg1 - x0_seg1
        v_hat_seg1 = self.model(xt_seg1, timesteps, cond_seg1)  # Predict the velocity field

        mask = (horizon_in_dataloader + idx * stride) <= length_1
        mask = torch.squeeze(mask.float(), dim=-1)

        imitation_loss_1 = torch.norm((v_hat_seg1 - v_seg1), dim=-1) ** 2
        imitation_loss = torch.sum(imitation_loss_1, dim=-1) * mask

        loss = imitation_loss
        loss_metrics = {
            'bc_loss': imitation_loss.mean().item(),
        }
        return torch.mean(loss), loss_metrics
    
    @torch.no_grad()
    def sample(
            self,
            cond: dict,
            inference_steps: int,
            record_intermediate: bool = False,
            clip_intermediate_actions: bool = True,
            z: torch.Tensor = None
    ) -> Sample:
        """Sample trajectories using the learned velocity field.

                Args:
                    cond: Dictionary containing 'state' tensor of shape (batch_size, cond_steps, obs_dim).
                    inference_steps: Number of denoising steps.
                    record_intermediate: Whether to return intermediate predictions.
                    clip_intermediate_actions: Whether to clip actions to act_range.

                Returns:
                    Sample: Named tuple with 'trajectories' (and 'chains' if record_intermediate).
                """
        B = cond.shape[0]
        if record_intermediate:
            x_hat_list = torch.zeros((inference_steps,) + self.data_shape, device=self.device)
        x_hat = z if z is not None else torch.randn((B,) + self.data_shape, device=self.device)
        dt = (1 / inference_steps) * torch.ones_like(x_hat, device=self.device)
        steps = torch.linspace(0, 1 - 1 / inference_steps, inference_steps, device=self.device).repeat(B, 1)
        for i in range(inference_steps):
            t = steps[:, i]
            vt = self.model(x_hat, t, cond)
            x_hat += vt * dt
            if clip_intermediate_actions or i == inference_steps - 1:  # always clip the output action. appended by ReinFlow Authors on 04/25/2025
                x_hat = x_hat.clamp(*self.act_range)
            if record_intermediate:
                x_hat_list[i] = x_hat
        return Sample(trajectories=x_hat, chains=x_hat_list if record_intermediate else None)
