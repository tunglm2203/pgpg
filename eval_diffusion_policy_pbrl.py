"""
Usage:
python eval_diffusion_policy_pbrl.py --checkpoint logs/pbrl/square_lowdim/pgpg_diffusion_policy --output_dir eval_logs/noise_0.0

Normally invoked through bash_script/evaluation/eval_pgpg_dp_transformer.sh.
"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import argparse
import hydra
import torch
import dill
import json
import ast
from termcolor import colored
from omegaconf import OmegaConf, open_dict
from diffusion_policy.workspace.base_workspace import BaseWorkspace



def str2bool(v):
    # used for parsing boolean arguments
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def setup_noise_schedule(cfg, noise_scheduler, num_inference_steps=None):
    if noise_scheduler.lower() == "ddpm":
        if cfg.policy.noise_scheduler['_target_'] == 'diffusers.schedulers.scheduling_ddpm.DDPMScheduler':
            print(colored(f"Skip setup scheduler: Pretrained model already used DDPM scheduler", "green"))
        else:
            with open_dict(cfg.policy.noise_scheduler):
                cfg.policy.noise_scheduler = OmegaConf.create({
                    '_target_': 'diffusers.schedulers.scheduling_ddpm.DDPMScheduler',
                    'num_train_timesteps': 100,
                    'beta_start': 0.0001,
                    'beta_end': 0.02,
                    'beta_schedule': 'squaredcos_cap_v2',
                    'variance_type': 'fixed_small',
                    'clip_sample': True,
                    'prediction_type': 'epsilon'
                })
    elif noise_scheduler.lower() == "ddim":
        if cfg.policy.noise_scheduler['_target_'] == 'diffusers.schedulers.scheduling_ddim.DDIMScheduler':
            print(colored(f"Skip setup scheduler: Pretrained model already used DDIM scheduler", "green"))
        else:
            with open_dict(cfg.policy.noise_scheduler):
                cfg.policy.noise_scheduler = OmegaConf.create({
                    '_target_': 'diffusers.schedulers.scheduling_ddim.DDIMScheduler',
                    'num_train_timesteps': 100,
                    'beta_start': 0.0001,
                    'beta_end': 0.02,
                    'beta_schedule': 'squaredcos_cap_v2',
                    'clip_sample': True,
                    'set_alpha_to_one': True,
                    'steps_offset': 0,
                    'prediction_type': 'epsilon'
                })
    else:
        raise ValueError(f"Unknown noise scheduler {noise_scheduler}")

    if num_inference_steps is not None:
        cfg.policy.num_inference_steps = num_inference_steps
    print(colored(f"Noise Scheduler: {noise_scheduler.upper()}-{cfg.policy.num_inference_steps}", "green"))
    print(OmegaConf.to_yaml(cfg.policy.noise_scheduler))
    return cfg


def get_all_checkpoint_paths(checkpoint_dir, ckpt_selection="last"):
    all_checkpoints = []
    if os.path.isfile(checkpoint_dir):
        return [checkpoint_dir]
    else:
        for exp_seed in os.listdir(checkpoint_dir):
            ckpt_path = os.path.join(checkpoint_dir, exp_seed, "checkpoints")
            assert os.path.isdir(ckpt_path), f"Checkpoint directory {ckpt_path} does not exist"
            if ckpt_selection == "best":
                best_score = -float("inf")
                best_ckpt = None

                for fname in os.listdir(ckpt_path):
                    if not fname.endswith(".ckpt") or "latest" in fname:
                        continue

                    fname_no_ext = fname.replace(".ckpt", "")
                    score = float(fname_no_ext[-5:])
                    if score > best_score:
                        best_score = score
                        best_ckpt = fname

                if best_ckpt is None:
                    raise ValueError(f"No valid checkpoint found in {ckpt_path}")
                ckpt_file = best_ckpt
                ckpt_file = os.path.join(ckpt_path, ckpt_file)
            elif ckpt_selection == "last":
                ckpt_file = os.path.join(ckpt_path, "latest.ckpt")
            else:
                raise ValueError(f"Unknown ckpt_selection strategy: {ckpt_selection}")

            if os.path.isfile(ckpt_file):
                all_checkpoints.append(ckpt_file)
        return all_checkpoints


def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Get checkpoint based on '--ckpt_selection' strategy
    all_checkpoints = get_all_checkpoint_paths(args.checkpoint, args.ckpt_selection)

    exp_name = args.checkpoint.split("/")[-1] if os.path.isdir(args.checkpoint) else args.checkpoint.split("/")[-3]
    for checkpoint in all_checkpoints:
        # load checkpoint
        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
        if cfg.policy._target_ == "diffusion_policy.policy.cpl_diffusion_transformer_lowdim_policy.DiffusionTransformerLowdimPolicy":
            # To make compatible after refactor
            cfg.policy._target_ = "diffusion_policy.policy.cpl_diffusion_transformer_lowdim_policy.CplDiffusionTransformerLowdimPolicy"
        cls = hydra.utils.get_class(cfg._target_)
        cfg = setup_noise_schedule(cfg, args.noise_scheduler, args.num_inference_steps)
        workspace = cls(cfg, output_dir=None)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        policy.to(device)
        policy.eval()

        # Setup log dir
        exp_name_seed = checkpoint.split("/")[-3]
        out_path = f"{output_dir}/{cfg.task.name}/{exp_name}/{exp_name_seed}"
        os.makedirs(out_path, exist_ok=True)
        json_eval_file = f"random_s{args.seed}_To{policy.n_obs_steps}_Tp{policy.horizon}_Ta{policy.n_action_steps}.json"
        out_path = os.path.join(out_path, json_eval_file)

        # Setup eval runner
        cfg.task.env_runner['n_train_vis'] = 0
        cfg.task.env_runner['n_test_vis'] = 0
        cfg.task.env_runner['n_train'] = 0
        cfg.task.env_runner['n_test'] = args.ntest
        cfg.task.env_runner['n_envs'] = 50
        cfg.task.env_runner['n_obs_steps'] = 8
        cfg.task.env_runner['test_start_seed'] = 20000 + 10000 * args.seed

        print(colored(f"Checkpoint: {checkpoint}", 'yellow', attrs=['bold']))
        print(colored(f"Evaluation setting:", 'yellow', attrs=['bold']))
        print(colored(f"    Env:    Ta={cfg.task.env_runner.n_action_steps}, To={cfg.task.env_runner.n_obs_steps}", 'yellow', attrs=['bold']))
        print(colored(f"    Policy: Ta={policy.n_action_steps}, To={policy.n_obs_steps}, Tp={policy.horizon}", 'yellow', attrs=['bold']))

        # Instantiate env_runner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=output_dir,
            return_intermediate_state=True,
            collect_data=False
        )

        # set sampler
        env_runner.set_sampler('random', 1, 1, 0, 1.0, 0.99)    # default, do not use
        runner_log = env_runner.run(policy)
        print(colored(f"Test: {runner_log['test/mean_score']:.4f}", "green", attrs=['bold']))

        # dump log to json
        json_log = dict()
        args_dict = vars(args)
        json_log['args'] = args_dict
        for key, value in runner_log.items():
            if 'video' not in key:
                json_log[key] = float(value)
        json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)
        print(colored(f"Dumped to: {out_path}\n", 'green'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--ntest', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--ckpt_selection', type=str, default="last", choices=["best", "last"])

    # Diffusion policy setting
    parser.add_argument('--noise_scheduler', type=str, default='ddpm')
    parser.add_argument('--num_inference_steps', type=int, default=100)
    args = parser.parse_args()

    main(args)
