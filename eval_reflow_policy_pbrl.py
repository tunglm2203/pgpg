"""
Usage:
python eval_reflow_policy_pbrl.py --checkpoint logs/pbrl/square_lowdim/pgpg_flow_policy --output_dir eval_logs/noise_0.0

Normally invoked through bash_script/evaluation/eval_pgpg_reflow_transformer.sh.
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
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=None)
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=['optimizer'], include_keys=None)

        # get policy from workspace
        policy = workspace.model
        if args.use_ema:
            policy = workspace.ema_model
        policy.test_denoising_steps = args.num_inference_steps

        policy.set_normalizer(policy.normalizer)  # need to set it again to set min, max range
        policy.to(device)
        policy.eval()

        # Setup log dir
        exp_name_seed = checkpoint.split("/")[-3]
        out_path = f"{output_dir}/{cfg.task.name}/{exp_name}_nfe{policy.test_denoising_steps}/{exp_name_seed}"
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
        print(colored(f"    NFE={policy.test_denoising_steps}, use_ema={args.use_ema}",'green', attrs=['bold']))

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
    parser.add_argument('--use_ema', type=str2bool, default=True)
    parser.add_argument('--num_inference_steps', type=int, default=16)
    args = parser.parse_args()

    main(args)
