from typing import Dict
import torch
import numpy as np
import copy
import h5py
from tqdm import tqdm

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset


class Hdf5LowdimDataset(BaseLowdimDataset):
    def __init__(
            self,
            dataset_dir=None,
            horizon=1,
            pad_before=0,
            pad_after=0,
            abs_action=True,
            seed=42,
            val_ratio=0.0,
            dense_reward=False
    ):
        super().__init__()
        if not abs_action:
            raise NotImplementedError("Not implemented for relative actions")

        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        with h5py.File(dataset_dir, 'r') as f:
            demos = list(f["data"].keys())
            inds = np.argsort([int(elem.split("_")[-1]) for elem in demos])
            demos = [demos[i] for i in inds]

            for idx in tqdm(range(len(demos)), desc="Loading hdf5 to ReplayBuffer"):
                ep = demos[idx]
                demo = f['data'][ep]
                episode = {
                    'obs': demo['obs'][:].astype(np.float32),
                    'action': demo['actions'][:].astype(np.float32),
                    'reward': demo['rewards'][:].astype(np.float32) if dense_reward else demo['successes'][:].astype(np.float32),
                }
                self.replay_buffer.add_episode(episode)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.dataset_path = dataset_dir

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
            'obs': self.replay_buffer['obs'],
            'action': self.replay_buffer['action'],
            'reward': self.replay_buffer['reward'],
        }
        if 'range_eps' not in kwargs:
            # to prevent blowing up dims that barely change
            kwargs['range_eps'] = 5e-2
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
