if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import numpy as np
import random
import wandb
import tqdm
import scipy.stats as stats
from termcolor import colored

from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.cpl_diffusion_transformer_lowdim_policy import CplDiffusionTransformerLowdimPolicy
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)

# %%
class PbrlDiffusionTransformerLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # configure model
        self.model: CplDiffusionTransformerLowdimPolicy
        self.model = hydra.utils.instantiate(cfg.policy)

        self.ema_model: CplDiffusionTransformerLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        self.global_step = 0
        self.epoch = 0

    def prepare_preference_dataset(self, cfg):
        # configure dataset
        dataset_1: BaseLowdimDataset
        if cfg.training.use_expert_data_1:
            dataset_1 = hydra.utils.instantiate(cfg.task.dataset, include_reward=True)
        else:
            dataset_1 = hydra.utils.instantiate(cfg.task.dataset_1)
        assert isinstance(dataset_1, BaseLowdimDataset)

        # configure dataset
        dataset_2: BaseLowdimDataset
        if cfg.training.use_expert_data_2:
            dataset_2 = hydra.utils.instantiate(cfg.task.dataset, include_reward=True)
        else:
            dataset_2 = hydra.utils.instantiate(cfg.task.dataset_2)
        assert isinstance(dataset_2, BaseLowdimDataset)

        if (not cfg.training.use_expert_data_1) and (not cfg.training.use_expert_data_2):
            # If both dataset_1 and dataset_2 are rollout data, then we pass the expert to generate preference
            dataset_expert = hydra.utils.instantiate(cfg.task.dataset, include_reward=True)
            dataset_expert_path = dataset_expert.dataset_path
            replay_expert = dataset_expert.replay_buffer
        else:
            replay_expert = dataset_expert_path = None

        pref_dataset: BaseLowdimDataset
        pref_dataset = hydra.utils.instantiate(
            cfg.task.pref_dataset,
            replay_buffer_1=dataset_1.replay_buffer, replay_buffer_2=dataset_2.replay_buffer,
            dataset_1_path=dataset_1.dataset_path, dataset_2_path=dataset_2.dataset_path,
            pseudo_preference=cfg.training.pseudo_preference,
            replay_buffer_expert=replay_expert, dataset_expert_path=dataset_expert_path
        )

        # cut online groups
        votes_1, votes_2 = pref_dataset.pref_replay_buffer.meta['votes'], pref_dataset.pref_replay_buffer.meta['votes_2']

        all_votes_1 = np.array([votes_1 for _ in range(cfg.training.preference_learning.num_rounds)])
        all_votes_2 = np.array([votes_2 for _ in range(cfg.training.preference_learning.num_rounds)])

        return pref_dataset, all_votes_1, all_votes_2

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            ckpt_path = pathlib.Path(cfg.checkpoint_dir)
            assert ckpt_path.is_file()
            print(colored(f"Resuming from checkpoint {ckpt_path}", "green", attrs=['bold']))
            self.load_checkpoint(path=ckpt_path)
            self.optimizer = self.model.get_optimizer(**cfg.optimizer)
            self.global_step = 0
            self.epoch = 0
        else:
            print(colored(f"Do not train from scratch", "red", attrs=['bold']))
            raise NotImplementedError

        device = torch.device(cfg.training.device)
        ref_policy = copy.deepcopy(self.model)
        ref_policy.train()  # tried .eval() but worse performance
        for param in ref_policy.parameters():
            param.requires_grad = False
        ref_policy.to(device)

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset, include_reward=True)
        assert isinstance(dataset, BaseLowdimDataset)
        normalizer = dataset.get_normalizer()
        del dataset     # This is only used to get normalizer
        pref_dataset, all_votes_1, all_votes_2 = self.prepare_preference_dataset(cfg)

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # configure env runner
        env_runner: BaseLowdimRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseLowdimRunner)

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
        if pref_dataset.pseudo_preference:
            # log wandb about
            wandb_run.log({
                "pseudo_preference/retained_pairs": pref_dataset.retained_pairs,
                "pseudo_preference/accuracy": pref_dataset.accuracy,
                "pseudo_preference/retained_rate": pref_dataset.retention_rate
            }, step=0)

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        json_logger = JsonLogger(log_path)
        json_logger.start()
        for round_idx in range(cfg.training.preference_learning.num_rounds):
            print(f"Round {round_idx + 1} of {cfg.training.preference_learning.num_rounds} for online training")

            local_votes_1 = np.array(all_votes_1[round_idx].T, dtype=np.float32).reshape(-1, 1)
            local_votes_2 = np.array(all_votes_2[round_idx].T, dtype=np.float32).reshape(-1, 1)

            pref_dataset.pref_replay_buffer.meta['votes'] = local_votes_1
            pref_dataset.pref_replay_buffer.meta['votes_2'] = local_votes_2
            pref_dataset.pref_replay_buffer.root['meta']['votes'] = local_votes_1
            pref_dataset.pref_replay_buffer.root['meta']['votes_2'] = local_votes_2

            train_dataloader = DataLoader(pref_dataset, **cfg.dataloader)
            self.optimizer = self.model.get_optimizer(**cfg.optimizer)

            # Place lr_scheduler here to reset in each round
            lr_scheduler = get_scheduler(
                cfg.training.lr_scheduler,
                optimizer=self.optimizer,
                num_warmup_steps=cfg.training.lr_warmup_steps,
                num_training_steps=(
                    len(train_dataloader) * cfg.training.num_epochs) \
                        // cfg.training.gradient_accumulate_every,
                last_epoch=-1,
            )

            equal_pref_threshold = cfg.training.preference_learning.equal_threshold * pref_dataset.sequence_length   # following LiRE
            stride = int(np.round(self.model.horizon * cfg.training.stride_ratio))
            debug_step = 0
            for local_epoch_idx in range(cfg.training.num_epochs):
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                if local_epoch_idx == cfg.training.n_epoch_sft:
                    print(f"Start CPL training at epoch {local_epoch_idx}: reset reference policy to the current policy")
                    ref_policy = copy.deepcopy(self.model)
                    ref_policy.train()  # # tried .eval() but worse performance
                    for param in ref_policy.parameters():
                        param.requires_grad = False
                    ref_policy.to(device)
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                               leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                        # compute loss
                        if cfg.training.cpl_loss_type == 'cplkl':
                            raw_loss, loss_metrics = self.model.compute_loss_cpl_kl(
                                batch,
                                epoch=local_epoch_idx,
                                ref_model=ref_policy.model,
                                n_epoch_sft=cfg.training.n_epoch_sft,
                                sft_type=cfg.training.sft_type,
                                stride=stride,
                                equal_pref_threshold=equal_pref_threshold,
                                debug=cfg.training.debug
                            )
                        elif cfg.training.cpl_loss_type == 'sft':
                            raw_loss, loss_metrics = self.model.compute_loss_sft(
                                batch,
                                stride=stride,
                                equal_pref_threshold=equal_pref_threshold
                            )
                        else:
                            raise NotImplementedError

                        if cfg.training.debug:
                            debug_step += 1
                            if debug_step <= 10:
                                continue
                            else:
                                exit()

                        loss = raw_loss / cfg.training.gradient_accumulate_every
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
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        step_log.update(loss_metrics)

                        is_last_batch = (batch_idx == (len(train_dataloader) - 1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                if (self.epoch % cfg.training.rollout_every) == 0 or self.epoch == cfg.training.num_epochs - 1:
                    runner_log = env_runner.run(policy)
                    # log all
                    step_log.update(runner_log)

                # checkpoint
                if self.epoch != 0 and ((self.epoch % cfg.training.checkpoint_every) == 0 or self.epoch == cfg.training.num_epochs - 1):
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value

                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1
        json_logger.stop()

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = PbrlDiffusionTransformerLowdimWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
