import os
import torch
import torch.nn as nn
import numpy as np
import copy
import random
import time
from termcolor import cprint

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.common.pref_replay_buffer import PrefReplayBuffer
from diffusion_policy.common.pref_sampler import PrefSequenceSampler
from diffusion_policy.preference_labeling.preference_labeling import (
    load_or_create_indices,
    load_or_compute_feats,
    precompute_pair_rewards,
    get_context_observations,
    extract_segment_pseudo_reward,
    ResNet, R3M, LIV, VIP, VC1, SigLIP2
)
from typing import Dict


class PbrlLowdimDataset(BaseLowdimDataset):
    def __init__(
            self,
            replay_buffer_1: ReplayBuffer,
            replay_buffer_2: ReplayBuffer,
            abs_action=True,
            sequence_length=1,
            gamma=0.999,
            num_queries=1,
            seed=42,
            gpu_device='cuda:0',
            dense_reward=False,
            val_ratio_data1=None,
            val_ratio_data2=None,
            dataset_1_path=None,
            dataset_2_path=None,
            task_name="can_lowdim",
            pseudo_preference=False,
            replay_buffer_expert=None,
            dataset_expert_path=None,
            feature_extractor="r3m_resnet18",
            context_num=3,
            seg_margin=0.6,
            min_progress=0.0,
            n_demos_for_preference=10,
            reward_type="orca",
            marginal_scoring=None,
    ):
        super().__init__()
        assert abs_action is True, "Only absolute action is supported"
        assert feature_extractor in ["imagenet_resnet18", "r3m_resnet18", "liv_resnet50", "vip_resnet50",
                                     "vc1_vitl", "siglip2_base"]
        self.pseudo_preference = pseudo_preference

        # Hyperparameters for pseudo-labeling
        self.seg_margin = seg_margin        # Segment must beat the other by `seg_margin` % coverage to win
        self.min_progress = min_progress    # At least one segment must achieve 'min_progress' % coverage
        self.context_num = context_num        # context window for computing ORCA
        self.n_demos_for_preference = n_demos_for_preference
        self.reward_type = reward_type
        # None -> historical behaviour (marginal for ORCA only). True -> apply to every metric.
        self.marginal_scoring = (reward_type == "orca") if marginal_scoring is None else marginal_scoring

        episode_ends_1 = replay_buffer_1.episode_ends
        episode_ends_2 = replay_buffer_2.episode_ends
        num_episodes_1 = int(len(episode_ends_1) * (1 - val_ratio_data1))
        num_episodes_2 = int(len(episode_ends_2) * (1 - val_ratio_data2))
        self.pref_replay_buffer = PrefReplayBuffer.create_empty_numpy()
        random.seed(seed)
        print(f"=====================> PbrlLowdimDataset: Num episodes (dataset_1): {num_episodes_1}, "
              f"min_len={replay_buffer_1.episode_lengths.min()}, max_len={replay_buffer_1.episode_lengths.max()}")
        print(f"=====================> PbrlLowdimDataset: Num episodes (dataset_2): {num_episodes_2},"
              f"min_len={replay_buffer_2.episode_lengths.min()}, max_len={replay_buffer_2.episode_lengths.max()}")

        # NOTE: 18/05 tri load npz or sample-and-save
        idx_path = f"logs/pbrl_indices/{task_name}/pair_{task_name}_nQ{num_queries}_L{sequence_length}_{num_episodes_1}_{num_episodes_2}.npz"
        pair_indices = load_or_create_indices(
            path=idx_path,
            num_queries=num_queries,
            num_episodes_1=num_episodes_1,
            num_episodes_2=num_episodes_2,
            episode_ends_1=episode_ends_1,
            episode_ends_2=episode_ends_2,
            sequence_length=sequence_length,
            seed=seed,
            use_cached=False,
            save_cached=True
        )

        if self.pseudo_preference:
            enc_batch = 128
            if feature_extractor == "imagenet_resnet18":
                encoder = ResNet().to(gpu_device).eval()
            elif feature_extractor == "r3m_resnet18":
                encoder = R3M().to(gpu_device).eval()
            elif feature_extractor == "liv_resnet50":
                encoder = LIV().to(gpu_device).eval()
            elif feature_extractor == "vip_resnet50":
                encoder = VIP().to(gpu_device).eval()
            elif feature_extractor == "vc1_vitl":
                encoder = VC1().to(gpu_device).eval()
                enc_batch = 32          # ViT-L/16 @224 does not fit the default batch
            elif feature_extractor == "siglip2_base":
                encoder = SigLIP2().to(gpu_device).eval()
                enc_batch = 64
            else:
                raise ValueError(f"Unknown feature extractor: {feature_extractor}")

            base_1 = os.path.join(os.path.dirname(dataset_1_path), "videos")
            base_2 = os.path.join(os.path.dirname(dataset_2_path), "videos")
            video_paths_1 = [f"{base_1}/episode_{i}.mp4" for i in range(num_episodes_1)]
            video_paths_2 = [f"{base_2}/episode_{i}.mp4" for i in range(num_episodes_2)]

            # Get visual features for trajectories in dataset_1 and dataset_2
            os.makedirs("cache", exist_ok=True)
            device = torch.device(gpu_device if torch.cuda.is_available() else "cpu")

            # NOTE: drop_last is False for both pools. All reported results were produced this
            # way, so it is pinned explicitly rather than inferred, to keep runs reproducible.
            # Features are recomputed from the videos on every run rather than reloaded.
            # A cache file is keyed only by (task, encoder, episode count), so regenerating
            # the videos -- e.g. replaying the demonstrations -- leaves a stale cache that
            # would silently be reused. Encoding costs ~1-2 min per run; correctness is
            # worth more. They are still written out, because the pseudo-label analysis
            # scripts (compute_pseudo_label_accuracy.py, analyze_accuracy_gap.py,
            # compute_label_diagnostics.py) read cache/ directly.
            start = time.time()
            feats_1 = load_or_compute_feats(
                f"cache/dataset_1_{task_name.replace('_lowdim', '')}_{feature_extractor}_n{num_episodes_1}.npz",
                video_paths_1, encoder, device, drop_last=False,
                use_cached=False, save_cached=True, batch_size=enc_batch)
            feats_2 = load_or_compute_feats(
                f"cache/dataset_2_{task_name.replace('_lowdim', '')}_{feature_extractor}_n{num_episodes_2}.npz",
                video_paths_2, encoder, device, drop_last=False,
                use_cached=False, save_cached=True, batch_size=enc_batch)
            print(f"Total time to load/encode {len(video_paths_1) + len(video_paths_2)} videos: {time.time() - start:.2f}s")

            if replay_buffer_expert is None:
                replay_buffer_expert = replay_buffer_1
                dataset_expert_path = dataset_1_path

            top_k = np.argpartition(replay_buffer_expert.episode_lengths, self.n_demos_for_preference)[:self.n_demos_for_preference]
            base_3 = os.path.join(os.path.dirname(dataset_expert_path), "videos")
            expert_paths = [f"{base_3}/episode_{i}.mp4" for i in top_k]
            expert_feats = load_or_compute_feats(f"cache/experts_{task_name}_nD{self.n_demos_for_preference}_{feature_extractor}.npz", expert_paths, encoder, device, use_cached=False, save_cached=False, batch_size=enc_batch)
            expert_ctx = [get_context_observations(f, context_num=self.context_num) for f in expert_feats]

            # ----- Precompute per-(traj, expert) rewards (the big win) -----
            print("precomputing trajectory's rewards for dataset 1...")
            dataset_rewards_1 = precompute_pair_rewards(feats_1, expert_ctx, context_num=self.context_num, reward_type=self.reward_type, marginal_scoring=self.marginal_scoring)
            print("precomputing trajectory's rewards for dataset 2...")
            dataset_rewards_2 = precompute_pair_rewards(feats_2, expert_ctx, context_num=self.context_num, reward_type=self.reward_type, marginal_scoring=self.marginal_scoring)

            assert len(dataset_rewards_1) == num_episodes_1 and len(dataset_rewards_2) == num_episodes_2

        # check if saved indices existing
        orca_match = 0
        retained_pairs = 0
        for i in range(num_queries):
            ep_idx_1, ts_idx_1, ep_idx_2, ts_idx_2 = pair_indices[i]
            ep_idx_1, ts_idx_1, ep_idx_2, ts_idx_2 = int(ep_idx_1), int(ts_idx_1), int(ep_idx_2), int(ts_idx_2)

            episode_1 = replay_buffer_1.get_episode(ep_idx_1, keys=['obs', 'action', 'reward'], copy=False)
            episode_2 = replay_buffer_2.get_episode(ep_idx_2, keys=['obs', 'action', 'reward'], copy=False)

            # Equal length processing for episode 1
            episode_1_len = len(episode_1['obs'])
            if episode_1_len >= sequence_length:
                start_1 = ts_idx_1
                length = sequence_length
                for key in episode_1.keys():
                    episode_1[key] = episode_1[key][start_1:start_1 + sequence_length]
            else:
                length = episode_1_len
                for key in episode_1.keys():
                    episode_1[key] = np.pad(episode_1[key], ((0, sequence_length - episode_1_len),) + ((0, 0),) * (episode_1[key].ndim - 1), mode='edge')

            # Equal length processing for episode 2
            episode_2_len = len(episode_2['obs'])
            if episode_2_len >= sequence_length:
                start_2 = ts_idx_2
                length_2 = sequence_length
                for key in episode_2.keys():
                    episode_2[key] = episode_2[key][start_2:start_2 + sequence_length]
            else:
                length_2 = episode_2_len
                for key in episode_2.keys():
                    episode_2[key] = np.pad(episode_2[key], ((0, sequence_length - episode_2_len),) + ((0, 0),) * (episode_2[key].ndim - 1), mode='edge')

            # Set up votes and metadata based on the presence of 'reward' in episode1
            votes = np.sum([(gamma ** t) * reward for t, reward in enumerate(episode_1['reward'])])
            votes_2 = np.sum([(gamma ** t) * reward for t, reward in enumerate(episode_2['reward'])])

            if self.pseudo_preference:
                gt = 1 if votes_2 > votes else 0

                orca_1_scores, orca_1_scores_all = extract_segment_pseudo_reward(dataset_rewards_1[ep_idx_1], ts_idx_1, sequence_length)
                orca_2_scores, orca_2_scores_all = extract_segment_pseudo_reward(dataset_rewards_2[ep_idx_2], ts_idx_2, sequence_length)
                # Max-pool over expert's trajectories
                score_1 = orca_1_scores.max()
                score_2 = orca_2_scores.max()

                # Apply Threshold & Margin
                if self.reward_type == "orca":
                    if max(score_1, score_2) < self.min_progress:
                        pref_label = -1  # Discard: Neither segment did anything useful
                    elif score_2 - score_1 > self.seg_margin:
                        pref_label = 1  # Right wins cleanly
                    elif score_1 - score_2 > self.seg_margin:
                        pref_label = 0  # Left wins cleanly
                    else:
                        pref_label = -1  # Discard: Difference is too small (noise)
                else:
                    if score_2 - score_1 > self.seg_margin:
                        pref_label = 1  # Right wins cleanly
                    else:
                        pref_label = 0  # Left wins cleanly

                if pref_label != -1:
                    retained_pairs += 1
                    orca_match += (pref_label == gt)

                    votes, votes_2 = score_1, score_2
                    # Add preferred episode to the replay buffer
                    self.pref_replay_buffer.add_pref_episode(
                        data={
                            'obs': episode_1['obs'],          # First trajectory observations (shape T, obs_dim)
                            'action': episode_1['action'],     # First trajectory actions (shape T, action_dim)
                            'obs_2': episode_2['obs'],         # Second trajectory observations
                            'action_2': episode_2['action'],    # Second trajectory actions
                        },
                        meta_data={
                            'votes': votes,                   # Vote for the first trajectory
                            'votes_2': votes_2,               # Vote for the second trajectory
                            'length': np.array([length]),     # Length of the first trajectory
                            'length_2': np.array([length_2]), # Length of the second trajectory
                            'beta_priori': np.ones([2]),
                            'beta_priori_2': np.ones([2]),
                        }
                    )

                # Calculate metrics
                retention_rate = (retained_pairs / num_queries) * 100
                accuracy = (orca_match / retained_pairs * 100) if retained_pairs > 0 else 0.0
                # end
            else:
                # Add preferred episode to the replay buffer
                self.pref_replay_buffer.add_pref_episode(
                    data={
                        'obs': episode_1['obs'],  # First trajectory observations (shape T, obs_dim)
                        'action': episode_1['action'],  # First trajectory actions (shape T, action_dim)
                        'obs_2': episode_2['obs'],  # Second trajectory observations
                        'action_2': episode_2['action'],  # Second trajectory actions
                    },
                    meta_data={
                        'votes': votes,  # Vote for the first trajectory
                        'votes_2': votes_2,  # Vote for the second trajectory
                        'length': np.array([length]),  # Length of the first trajectory
                        'length_2': np.array([length_2]),  # Length of the second trajectory
                        'beta_priori': np.ones([2]),
                        'beta_priori_2': np.ones([2]),
                    }
                )

        if self.pseudo_preference:
            assert retained_pairs > 0, f"Margin ({self.seg_margin}) is too strict! 0 pairs retained out of {num_queries}."
            cprint(f"Task={task_name.upper()}: n_expert={self.n_demos_for_preference}, n_queries={num_queries}, seq_len={sequence_length}, feat={feature_extractor}, min_progress={min_progress}, margin={seg_margin}", "green", attrs=["bold"])
            cprint(f"   -> Pairs Retained: {retained_pairs} ({retention_rate:.1f}%)", "cyan")
            cprint(f"   -> ORCA Accuracy (on retained): {accuracy:.1f}%", "green", attrs=["bold"])
            train_mask = np.ones(retained_pairs, dtype=bool)
            self.retained_pairs = retained_pairs
            self.accuracy = accuracy
            self.retention_rate = retention_rate
        else:
            train_mask = np.ones(num_queries, dtype=bool)
            self.accuracy = 0.0
            self.retained_pairs = num_queries
            self.retention_rate = 100

        self.sampler = PrefSequenceSampler(
            replay_buffer=self.pref_replay_buffer,
            sequence_length=sequence_length,
            episode_mask=train_mask,
        )

        self.gpu_device = gpu_device
        self.train_mask = train_mask
        self.sequence_length = sequence_length
        self.dense_reward = dense_reward

    def construct_pref_data(self):
        data = self.pref_replay_buffer.data
        pref_data = data.copy()
        meta = self.pref_replay_buffer.meta
        pref_data.update(meta)
        if 'episode_ends' in pref_data.keys():
            del pref_data['episode_ends']

        return pref_data

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = PrefSequenceSampler(
            replay_buffer=self.pref_replay_buffer, 
            sequence_length=self.sequence_length,
            episode_mask=~self.train_mask,
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_all_actions(self) -> torch.Tensor:
        actions = np.concatenate(self.pref_replay_buffer.data['action'], self.pref_replay_buffer.data['action_2'], dim = 0)
        return torch.from_numpy(actions)

    def __len__(self) -> int:
        return self.sampler.__len__()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        torch_data = self.sampler.sample_sequence(idx)
        return torch_data