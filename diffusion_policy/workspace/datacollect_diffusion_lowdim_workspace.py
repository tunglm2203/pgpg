if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import json
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
import copy
import numpy as np
import random
import dill
import h5py
from tqdm import tqdm
from termcolor import colored
from hydra.core.hydra_config import HydraConfig

from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy
from diffusion_policy.policy.cpl_diffusion_transformer_lowdim_policy import CplDiffusionTransformerLowdimPolicy
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecorder

OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
class DatacollectDiffusionLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # Load payload from checkpoint
        if cfg.checkpoint_dir is None:
            checkpoint_dir_dict = {
                'pusht_lowdim': {
                    'datacollect_diffusion_unet_lowdim': '',
                    'datacollect_diffusion_transformer_lowdim': '',
                },
                'lift_lowdim': {
                    'datacollect_diffusion_unet_lowdim': 'logs/pretrain/lift_lowdim/train_diffusion_cnn/checkpoints/epoch=0010-test_mean_score=0.680.ckpt',
                    'datacollect_diffusion_transformer_lowdim': 'logs/pretrain/lift_lowdim/train_diffusion_transformer/checkpoints/epoch=0015-test_mean_score=0.400.ckpt',
                },
                'can_lowdim': {
                    'datacollect_diffusion_unet_lowdim': 'logs/pretrain/can_lowdim/train_diffusion_cnn/checkpoints/epoch=0015-test_mean_score=0.600.ckpt',
                    'datacollect_diffusion_transformer_lowdim': 'logs/pretrain/can_lowdim/train_diffusion_transformer/checkpoints/epoch=0060-test_mean_score=0.380.ckpt',
                    # 'datacollect_diffusion_transformer_lowdim_round2': 'logs/pbrl/can_lowdim/cplkl_pseu_dpT_ExpD10_N5000_L120_bias0.25_Eq0_thr0_1ER.4_SFTpos0_strid1_segM0.3_0_nD20_beta0.01_clip0.3_smt0_cw0_unclipW1/cplkl_pseu_dpT_ExpD10_N5000_L120_bias0.25_Eq0_thr0_1ER.4_SFTpos0_strid1_segM0.3_0_nD20_beta0.01_clip0.3_smt0_cw0_unclipW1_42/checkpoints/latest.ckpt',
                },
                'square_lowdim': {
                    'datacollect_diffusion_unet_lowdim': 'logs/pretrain/square_lowdim/train_diffusion_cnn/checkpoints/epoch=0040-test_mean_score=0.520.ckpt',
                    'datacollect_diffusion_transformer_lowdim': 'logs/pretrain/square_lowdim/train_diffusion_transformer/checkpoints/epoch=0450-test_mean_score=0.520.ckpt',
                    # 'datacollect_diffusion_transformer_lowdim_round2': 'logs/pbrl/square_lowdim/cplkl_pseu_dpT_ExpD10_N20000_L200_bias0.25_Eq0_thr0_1ER.4_SFTpos0_strid1_segM0.3_0_nD40_beta0.01_clip0.3_smt0.1_cw0_unclipW1/cplkl_pseu_dpT_ExpD10_N20000_L200_bias0.25_Eq0_thr0_1ER.4_SFTpos0_strid1_segM0.3_0_nD40_beta0.01_clip0.3_smt0.1_cw0_unclipW1_42/checkpoints/latest.ckpt',
                },
                'transport_lowdim': {
                    'datacollect_diffusion_unet_lowdim': 'logs/pretrain/transport_lowdim/train_diffusion_cnn/checkpoints/epoch=0150-test_mean_score=0.480.ckpt',
                    'datacollect_diffusion_transformer_lowdim': 'logs/pretrain/transport_lowdim/train_diffusion_transformer/checkpoints/epoch=0450-test_mean_score=0.240.ckpt',
                    # 'datacollect_diffusion_transformer_lowdim_round2': 'logs/pbrl/transport_lowdim/cplkl_pseu_dpT_ExpD10_N20000_L400_bias0.25_Eq0_thr0_1ER.4_SFTpos0_strid1_segM0.4_0_nD40_beta0.01_clip0.3_smt0.1_cw0_unclipW1/cplkl_pseu_dpT_ExpD10_N20000_L400_bias0.25_Eq0_thr0_1ER.4_SFTpos0_strid1_segM0.4_0_nD40_beta0.01_clip0.3_smt0.1_cw0_unclipW1_42/checkpoints/latest.ckpt',
                },
                'tool_hang_lowdim': {
                    'datacollect_diffusion_unet_lowdim': '',
                    'datacollect_diffusion_transformer_lowdim': 'logs/pretrain/tool_hang_lowdim/train_diffusion_transformer/checkpoints/epoch=0500-test_mean_score=0.440.ckpt',
                },
                'kitchen_lowdim': {
                    'datacollect_diffusion_unet_lowdim': '',
                    'datacollect_diffusion_transformer_lowdim': '',
                },
            }

            checkpoint_dir = checkpoint_dir_dict[cfg.task_name][cfg.name]
        else:
            checkpoint_dir = cfg.checkpoint_dir

        ckpt_file = pathlib.Path(checkpoint_dir)
        assert ckpt_file.is_file()
        print(colored(f"Collecting from: {ckpt_file}", "green", attrs=["bold"]))
        payload = torch.load(ckpt_file.open('rb'), pickle_module=dill)
        self.pretrained_cfg = payload['cfg']

        # set seed
        seed = cfg.collecting.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        if self.pretrained_cfg.policy._target_ == 'diffusion_policy.policy.diffusion_unet_lowdim_policy.DiffusionUnetLowdimPolicy':
            self.model: DiffusionUnetLowdimPolicy
            self.model = hydra.utils.instantiate(self.pretrained_cfg.policy)
            self.ema_model: DiffusionUnetLowdimPolicy = None
            if self.pretrained_cfg.training.use_ema:
                self.ema_model = copy.deepcopy(self.model)
        elif self.pretrained_cfg.policy._target_ == 'diffusion_policy.policy.cpl_diffusion_transformer_lowdim_policy.CplDiffusionTransformerLowdimPolicy':
            self.model: CplDiffusionTransformerLowdimPolicy
            self.model = hydra.utils.instantiate(self.pretrained_cfg.policy)
            self.ema_model: CplDiffusionTransformerLowdimPolicy = None
            if self.pretrained_cfg.training.use_ema:
                self.ema_model = copy.deepcopy(self.model)
        elif self.pretrained_cfg.policy._target_ == 'diffusion_policy.policy.diffusion_transformer_lowdim_policy.DiffusionTransformerLowdimPolicy':
            self.model: DiffusionTransformerLowdimPolicy
            self.model = hydra.utils.instantiate(self.pretrained_cfg.policy)
            self.ema_model: DiffusionTransformerLowdimPolicy = None
            if self.pretrained_cfg.training.use_ema:
                self.ema_model = copy.deepcopy(self.model)
        else:
            raise ValueError(f"Unknown policy type: {self.pretrained_cfg.policy._target_}")

        # Load weights from pretrained models
        exclude_keys = ['optimizer']
        self.load_payload(payload, exclude_keys=exclude_keys, include_keys=None)

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        run_dir = HydraConfig.get().run.dir
        cfg.task.env_runner['n_train_vis'] = 0
        cfg.task.env_runner['n_test_vis'] = 0
        cfg.task.env_runner['n_train'] = 0
        cfg.task.env_runner['n_test'] = cfg.collecting.num_episodes
        cfg.task.env_runner['n_envs'] = min(100, cfg.collecting.num_episodes)

        # configure env runner
        env_runner: BaseLowdimRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir,
            return_intermediate_state=True,
            collect_data=True,
            use_oracle_ac=False,
        )
        assert isinstance(env_runner, BaseLowdimRunner)
        assert env_runner.return_intermediate_state and env_runner.collect_data and (not env_runner.use_oracle_ac), "Wrong configs in collect mode"

        # device transfer
        device = torch.device(cfg.collecting.device)
        policy = self.model
        if self.ema_model is not None:
            policy = self.ema_model
        policy.to(device)

        # Collect data
        policy.eval()
        runner_log, all_episodes = env_runner.run(policy)

        # Writing data to h5 file
        rollout_num_episodes = len(all_episodes['observations'])
        data_collect_file = os.path.join(run_dir, f"collect_{cfg.task_name}.hdf5")
        data_writer = h5py.File(data_collect_file, "w")
        data_grp = data_writer.create_group("data")
        total_samples = 0
        all_successes = []
        for i in range(rollout_num_episodes):
            states = []
            successes = []
            for t in range(len(all_episodes['infos'][i])):
                states.append(all_episodes['infos'][i][t]['states'])
                successes.append(all_episodes['infos'][i][t]['success'])

            if np.sum(successes) > 0:
                first_succ_idx = np.argmax(successes) # No need to +1 here since we have success flag at reset
            else:
                first_succ_idx = len(all_episodes['actions'][i])
            states = np.array(states)
            successes = np.array(successes)
            all_successes.append(np.max(successes))

            ep_data_grp = data_grp.create_group(f"episode_{i}")
            ep_data_grp.create_dataset("obs", data=np.array(all_episodes['observations'][i][:first_succ_idx]))
            ep_data_grp.create_dataset("next_obs", data=np.array(all_episodes['observations'][i][1:first_succ_idx + 1]))
            ep_data_grp.create_dataset("actions", data=np.array(all_episodes['actions'][i][:first_succ_idx]))
            ep_data_grp.create_dataset("rewards", data=np.array(all_episodes['rewards'][i][:first_succ_idx]))
            ep_data_grp.create_dataset("dones", data=np.array(all_episodes['terminals'][i][:first_succ_idx])) # this may not contain any done
            ep_data_grp.create_dataset("states", data=states[:first_succ_idx + 1])
            ep_data_grp.create_dataset("successes", data=successes[1:first_succ_idx + 1])

            ep_data_grp.attrs["model_file"] = all_episodes['infos'][i][0]['model'] # model xml for this episode
            ep_data_grp.attrs["num_samples"] = len(all_episodes['actions'][i]) # number of transitions in this episode

            total_samples += len(all_episodes['actions'][i])

        data_grp.attrs["total"] = total_samples
        data_grp.attrs["env_args"] = json.dumps(env_runner.env_meta, indent=4)
        data_writer.close()

        json_log = dict()
        for key, value in runner_log.items():
            if 'video' not in key:
                json_log[key] = float(value)
        json.dump(json_log, open(os.path.join(run_dir, f"collect_{cfg.task_name}.json"), 'w'), indent=2, sort_keys=True)

        print(colored(f"Avg. Performance: {np.mean(all_successes):.4f}", "green", attrs=['bold']))
        print(colored(f"Dumped to: {run_dir}\n", 'green'))

        if cfg.collecting.render_image:
            del env_runner
            print(f"Rendering video from collected data...")
            replay_collected_data(data_collect_file, run_dir, cfg.task.env_runner.render_hw[0], cfg.task.env_runner.render_hw[1])


def replay_collected_data(dataset_path, run_dir, cam_width, cam_height):
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=['agentview'],
        camera_height=cam_height,
        camera_width=cam_width,
        reward_shaping=True,
    )

    # Read data from offline dataset
    f = h5py.File(dataset_path, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem.split("_")[-1]) for elem in demos])
    demos = [demos[i] for i in inds]

    video_recoder = VideoRecorder.create_h264(
        fps=10,
        codec='h264',
        input_pix_fmt='rgb24',
        crf=22,
        thread_type='FRAME',
        thread_count=1
    )
    video_path = os.path.join(run_dir, "videos")
    os.makedirs(video_path, exist_ok=True)
    for ind in tqdm(range(len(demos))):
        ep = demos[ind]

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]

        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

        env.reset()
        obs = env.reset_to(initial_state)

        # Reset video writer
        video_recoder.stop()
        video_recoder.start(f"{video_path}/episode_{ind}.mp4")
        video_recoder.write_frame(obs['agentview_image'])  # Write initial state

        traj_len = states.shape[0]
        assert video_recoder.is_ready()
        for t in tqdm(range(1, traj_len), leave=False):
            # reset to simulator state to get observation
            next_obs = env.reset_to({"states": states[t]})
            video_recoder.write_frame(next_obs['agentview_image'])


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = DatacollectDiffusionLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
