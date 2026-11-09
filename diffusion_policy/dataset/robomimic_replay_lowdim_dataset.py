from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import copy
from termcolor import colored
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_identity_normalizer_from_stat,
    array_to_stats
)

class RobomimicReplayLowdimDataset(BaseLowdimDataset):
    def __init__(self,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_keys: List[str]=[
                'object', 
                'robot0_eef_pos', 
                'robot0_eef_quat', 
                'robot0_gripper_qpos'],
            abs_action=False,
            rotation_rep='rotation_6d',
            use_legacy_normalizer=False,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            dense_reward=False,
            include_reward=False,
            mixed_bc=False,
            filtered_bc=False,
            rollout_data=None
        ):
        obs_keys = list(obs_keys)
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

        replay_buffer = ReplayBuffer.create_empty_numpy()
        with h5py.File(dataset_path) as file:
            demos = file['data']
            for i in tqdm(range(len(demos)), desc="Loading hdf5 to ReplayBuffer"):
                demo = demos[f'demo_{i}']
                episode = _data_to_obs(
                    raw_obs=demo['obs'],
                    raw_actions=demo['actions'][:].astype(np.float32),
                    obs_keys=obs_keys,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer,
                    raw_rewards=(demo['rewards'][:] if dense_reward else demo['success'][:]) if include_reward else None,
                )
                replay_buffer.add_episode(episode)
        
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.replay_buffer = replay_buffer   # Assign replay buffer here to get normalizer from demos only
        self.abs_action = abs_action
        self.use_legacy_normalizer = use_legacy_normalizer
        self.dataset_normalizer = None
        self.get_normalizer()

        if mixed_bc or filtered_bc:
            assert max_train_episodes is None, "If we train with mixed or filtered BC, do not set: max_train_episodes"
            assert (not mixed_bc and filtered_bc) or (mixed_bc and not filtered_bc), "Only one of mixed_bc and filtered_bc can be True"
            assert rollout_data is not None
            n_rollouts_added = 0
            with h5py.File(rollout_data, 'r') as f:
                demos = list(f["data"].keys())
                inds = np.argsort([int(elem.split("_")[-1]) for elem in demos])
                demos = [demos[i] for i in inds]

                for idx in tqdm(range(len(demos)), desc="Loading rollout data to ReplayBuffer"):
                    ep = demos[idx]
                    demo = f['data'][ep]
                    if filtered_bc:
                        if (demo['successes'][:] == 0).all():  # only add successful rollouts
                            continue

                    episode = {
                        'obs': demo['obs'][:].astype(np.float32),
                        'action': demo['actions'][:].astype(np.float32),
                    }
                    if include_reward:
                        episode.update({'reward': demo['rewards'][:].astype(np.float32) if dense_reward else demo['successes'][:].astype(np.float32)})
                    self.replay_buffer.add_episode(episode)
                    n_rollouts_added += 1
            print(colored(f"=============> Number of expert demo: {train_mask.sum()}", "yellow"))
            print(colored(f"=============> Added {n_rollouts_added} rollouts to replay buffer", "yellow"))
            val_mask = np.concatenate([val_mask, np.zeros((n_rollouts_added, ), dtype=val_mask.dtype)])
            train_mask = np.concatenate([train_mask, np.ones(n_rollouts_added, dtype=train_mask.dtype)])

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)

        # self.replay_buffer = replay_buffer
        self.sampler = sampler
        # self.abs_action = abs_action
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        # self.use_legacy_normalizer = use_legacy_normalizer
        self.dataset_path = dataset_path
    
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

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        if self.dataset_normalizer is None:
            normalizer = LinearNormalizer()

            # action
            stat = array_to_stats(self.replay_buffer['action'])
            if self.abs_action:
                if stat['mean'].shape[-1] > 10:
                    # dual arm
                    this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
                else:
                    this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)

                if self.use_legacy_normalizer:
                    this_normalizer = normalizer_from_stat(stat)
            else:
                # already normalized
                this_normalizer = get_identity_normalizer_from_stat(stat)
            normalizer['action'] = this_normalizer

            # aggregate obs stats
            obs_stat = array_to_stats(self.replay_buffer['obs'])


            normalizer['obs'] = normalizer_from_stat(obs_stat)
            self.dataset_normalizer = normalizer

        return self.dataset_normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])
    
    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )
    
def _data_to_obs(raw_obs, raw_actions, obs_keys, abs_action, rotation_transformer, raw_rewards=None):
    obs = np.concatenate([
        raw_obs[key] for key in obs_keys
    ], axis=-1).astype(np.float32)

    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1,2,7)
            is_dual_arm = True

        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)
    
        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1,20)
    
    data = {
        'obs': obs,
        'action': raw_actions,
    }
    if raw_rewards is not None:
        data.update({'reward': raw_rewards.astype(np.float32)})
    return data
