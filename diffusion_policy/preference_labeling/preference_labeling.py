import os
import numba
import random
import numpy as np
from tqdm import tqdm
from decord import VideoReader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms import Normalize

from diffusion_policy.preference_labeling.alignment_utils import bordered_identity_like, mask_optimal_transport_plan, dtw, dtw_path
from r3m import load_r3m

""" ========================================== All encoders ========================================== """
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
        self.model = model.eval()
        self.normalizer = Normalize(mean=torch.FloatTensor([0.485, 0.456, 0.406]),
                                    std=torch.FloatTensor([0.229, 0.224, 0.225]))

    def forward(self, obs):
        obs = obs[:, -3:] / 255.0
        h = self.normalizer(obs)
        for m in list(self.model.children())[:-1]:
            h = m(h)
        out = h.view(obs.shape[0], -1)
        return out


class R3M(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_type = "resnet18"
        model = load_r3m(self.model_type).module  # unpack parallel model
        self.model = model.eval()
        self.normalizer = Normalize(mean=torch.FloatTensor([0.485, 0.456, 0.406]),
                                    std=torch.FloatTensor([0.229, 0.224, 0.225]))

    def forward(self, obs):
        h = self.normalizer(obs)
        for m in list(self.model.convnet.children())[:-1]:
            h = m(h)
        out = h.view(obs.shape[0], -1)
        return out


class LIV(nn.Module):
    def __init__(self):
        from liv import load_liv
        super().__init__()
        self.model_type = "resnet50"
        model = load_liv(self.model_type).module  # unpack parallel model
        self.model = model.eval()

    def forward(self, obs):
        out = self.model(input=obs, modality="vision")
        return out


class VIP(nn.Module):
    def __init__(self):
        super().__init__()
        from vip import load_vip
        self.model_type = "resnet50"
        model = load_vip(self.model_type).module  # unpack parallel model
        self.model = model.eval()

    def forward(self, obs):
        out = self.model(obs)
        return out


class VC1(nn.Module):
    """VC-1 (ViT-L/16, MAE-pretrained on Ego4D+ImageNet) - Majumdar et al., NeurIPS 2023.

    The official eai-vc release is a plain MAE ViT-L/16 @224; we load its checkpoint into the
    equivalent timm architecture to avoid the eai-vc package's hydra/omegaconf pins. Eval
    preprocessing follows VC-1: resize to 224 then ImageNet normalize, on a [0, 1] tensor.
    """
    CKPT = "pretrained_encoders/vc1_vitl.pth"

    def __init__(self, ckpt_path=None):
        super().__init__()
        import timm
        model = timm.create_model("vit_large_patch16_224.mae", pretrained=False, num_classes=0)
        sd = torch.load(ckpt_path or self.CKPT, map_location="cpu")
        sd = sd.get("model", sd)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # the MAE decoder / mask token are not part of the visual encoder
        assert not [k for k in missing if not k.startswith("head")], f"missing encoder weights: {missing}"
        self.model = model.eval()
        self.normalizer = Normalize(mean=torch.FloatTensor([0.485, 0.456, 0.406]),
                                    std=torch.FloatTensor([0.229, 0.224, 0.225]))

    def forward(self, obs):
        x = F.interpolate(obs / 255.0, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
        return self.model(self.normalizer(x))


class SigLIP2(nn.Module):
    """SigLIP 2 vision tower (ViT-B/16 @224, `v2_webli` weights) - Tschannen et al., 2025.

    SigLIP preprocessing rescales to [0, 1] and normalizes with mean = std = 0.5.
    """
    def __init__(self, model_name="vit_base_patch16_siglip_224.v2_webli"):
        super().__init__()
        import timm
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0).eval()
        self.normalizer = Normalize(mean=torch.FloatTensor([0.5, 0.5, 0.5]),
                                    std=torch.FloatTensor([0.5, 0.5, 0.5]))

    def forward(self, obs):
        x = F.interpolate(obs / 255.0, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
        return self.model(self.normalizer(x))

""" ================================================================================================== """


def encode_video_with_batch(video_np, model, device, batch_size=128):
    v = torch.from_numpy(video_np).permute(0, 3, 1, 2).contiguous().float()
    feats = []
    with torch.no_grad():
        for start in range(0, len(v), batch_size):
            batch = v[start:start + batch_size].to(device)
            feat = model(batch).cpu()
            feats.append(feat)
            del batch
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(feats, dim=0).numpy()


def load_or_compute_feats(cache_path, video_paths, encoder, device, drop_last=False, use_cached=True, save_cached=True,
                          batch_size=128):
    """Load cached features or run ResNet on every video then cache to disk."""
    if use_cached and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return list(data["feats"])
    feats = []
    for p in tqdm(video_paths, desc=f"encoding -> {os.path.basename(cache_path)}"):
        vr = VideoReader(p)
        video = vr.get_batch(np.arange(0, len(vr))).asnumpy()
        if drop_last:
            video = video[:-1]   # Some rollout data contains the last state caused by last action, but we don't use
        feats.append(encode_video_with_batch(video, encoder, device, batch_size=batch_size))
        del video, vr
    if save_cached:
        np.savez(cache_path, feats=np.array(feats, dtype=object))
    return feats


def get_context_observations(observations, context_num=3):
    """Stack of context_num shifted copies of `observations`. Returns (C, L, D)."""
    observations = np.asarray(observations)
    L = len(observations)
    idx0 = np.arange(L)
    out = [observations[idx0]]
    for i in range(1, context_num):
        idx_i = (idx0 + i).clip(0, L - 1)
        out.append(observations[idx_i])
    return np.stack(out, axis=0)


def get_averaged_cost_matrix(feat_a, feat_b):
    """feat_a: (C, L_a, D), feat_b: (C, L_b, D). Returns (L_a, L_b) cosine cost."""
    assert feat_a.shape[0] == 3 and feat_b.shape[0] == 3
    a = feat_a / (np.linalg.norm(feat_a, axis=-1, keepdims=True) + 1e-8)
    b = feat_b / (np.linalg.norm(feat_b, axis=-1, keepdims=True) + 1e-8)
    sim = np.matmul(a, b.transpose(0, 2, 1))   # (C, L_a, L_b)
    return np.clip(1.0 - sim.mean(axis=0), 0.0, 2.0)


@numba.njit(cache=True, fastmath=True)
def _orca_dp(prob_matrix):
    """JIT-compiled DP recurrence for ORCA ordered-coverage matrix.

    Matches the original recurrence exactly:
        covered[i, j] = max(covered[i-1, j], covered[i, j-1] * prob[i, j])
    with special-case handling of the final column to force occupancy of the
    last subgoal (paper Eq. 6).
    """
    T, S = prob_matrix.shape
    covered = np.zeros_like(prob_matrix)
    covered[0, 0] = prob_matrix[0, 0]
    # Init col 0
    for i in range(1, T):
        a = covered[i - 1, 0]
        b = prob_matrix[i, 0]
        covered[i, 0] = a if a > b else b
    # Init row 0
    for j in range(1, S):
        covered[0, j] = covered[0, j - 1] * prob_matrix[0, j]
    # Main DP (skip last column)
    for i in range(1, T):
        for j in range(1, S - 1):
            a = covered[i - 1, j]
            b = covered[i, j - 1] * prob_matrix[i, j]
            covered[i, j] = a if a > b else b
    # Final column: force occupancy at last subgoal
    for i in range(T):
        covered[i, S - 1] = covered[i, S - 2] * prob_matrix[i, S - 1]
    return covered


""" ==================================== All pseudo-reward supported ==================================== """
def compute_orca_reward(cost_matrix, tau=1.0):
    prob = np.exp(-cost_matrix / tau)
    covered = _orca_dp(prob.astype(np.float64))
    return covered[:, -1], covered


def compute_temporal_ot_reward(cost_matrix, mask_k: int = 10, niter: int = 100, ent_reg: float = 0.01):
    """
    TemporalOT reward, as implemented in (Fu et al., Robot Policy Learning with Temporal Optimal Transport Reward, NeurIPS 2024)
    Code from https://github.com/fuyw/TemporalOT
    """

    # optimal weights
    mask = bordered_identity_like(cost_matrix.shape[0], cost_matrix.shape[1], k=mask_k)
    transport_plan = mask_optimal_transport_plan(cost_matrix, mask, niter, ent_reg)

    ot_cost = np.sum(transport_plan * cost_matrix, axis=1)
    ot_reward = -ot_cost
    return ot_reward, {"assignment": transport_plan}


def compute_dtw_reward(cost_matrix):
    """
    Compute the reward with an assignment matrix that uses dynamic time warping
    """
    _, accumulated_cost_matrix = dtw(cost_matrix)
    path = dtw_path(accumulated_cost_matrix)

    # Normalize the path so that each row sums to 1
    normalized_path = path / np.expand_dims(np.sum(path, axis=1), 1)
    dtw_cost = np.sum(cost_matrix * normalized_path, axis=1)  # size: (train_freq,)
    final_reward = -dtw_cost

    return final_reward, {"assignment": normalized_path}


""" ================================================================================================== """


def load_or_create_indices(
    path: str,
    num_queries: int,
    num_episodes_1: int,
    num_episodes_2: int,
    episode_ends_1: np.ndarray,
    episode_ends_2: np.ndarray,
    sequence_length: int,
    seed: int,
    use_cached=True,
    save_cached=True,
) -> np.ndarray:
    #Load pair indices from *path* if it already exists; otherwise sample them randomly, save them, and return.

    # Index layout — array shape ``(num_queries, 4)``:
    # [ep_idx_1, timestep_idx_1, ep_idx_2, timestep_idx_2]

    if use_cached and os.path.isfile(path):
        print(f"=====================> Loading pair indices from {path}")
        data = np.load(path)
        indices = data["indices"]
        assert indices.shape == (num_queries, 4), (
            f"Loaded indices shape {indices.shape} does not match "
        )
        return indices

    # Generate fresh indices
    print(f"=====================> Generating new pair indices → {path}")
    rng = random.Random(seed)

    # Pre-compute per-episode lengths from cumulative episode_ends.
    def episode_length(episode_ends: np.ndarray, idx: int) -> int:
        start = episode_ends[idx - 1] if idx > 0 else 0
        return int(episode_ends[idx]) - int(start)

    indices = np.zeros((num_queries, 4), dtype=np.int64)
    for i in range(num_queries):
        ep_idx_1 = rng.randrange(num_episodes_1)
        ep_idx_2 = rng.randrange(num_episodes_2)

        len_ep1 = episode_length(episode_ends_1, ep_idx_1)
        len_ep2 = episode_length(episode_ends_2, ep_idx_2)

        # Random start index; fall back to 0 when the episode is shorter than
        # the requested sequence length (the caller will pad in that case).
        max_start_1 = max(len_ep1 - sequence_length, 0)
        max_start_2 = max(len_ep2 - sequence_length, 0)

        ts_idx_1 = rng.randint(0, max_start_1)  # inclusive on both ends
        ts_idx_2 = rng.randint(0, max_start_2)

        indices[i] = [ep_idx_1, ts_idx_1, ep_idx_2, ts_idx_2]

    if save_cached:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, indices=indices)
        print(f"=====================> PbrlLowdimDataset: Saved pair indices to {path}")
    return indices


def precompute_pair_rewards(traj_feats, expert_ctx_list, context_num=3, min_cost=0.05, max_cost=0.30, reward_type="orca",
                            marginal_scoring=None):
    """For every trajectory, compute per-expert per-step rewards.

    marginal_scoring: apply Marginal Progress Scoring (Eq. 3) -- convert the metric's raw
        step-wise score into the marginal difference r_t = r̂_t - r̂_{t-1}, so that summing
        over a segment telescopes to r̂_{t+L} - r̂_t (Eq. 4) instead of double-counting.
        Default (None) preserves historical behaviour: on for ORCA, off for DTW / Temporal OT,
        which is the "naive summation" those baselines use. Set True to apply the transform
        uniformly across metrics, isolating the gain attributable to the transform itself.
    """
    assert reward_type in ("orca", "dtw", "temporal_ot"), f"Unknown reward_type: {reward_type}"
    if marginal_scoring is None:
        marginal_scoring = (reward_type == "orca")
    n_exp = len(expert_ctx_list)
    reward_all = []

    for traj_feat in tqdm(traj_feats, desc=f"precompute rewards ({reward_type})"):
        traj_ctx = get_context_observations(traj_feat, context_num=context_num)
        L = traj_feat.shape[0]
        reward_relative_to_expert = np.zeros((n_exp, L), dtype=np.float32)

        for e_idx, exp_ctx in enumerate(expert_ctx_list):
            # cost shape: (L_traj, L_exp)
            cost = get_averaged_cost_matrix(traj_ctx, exp_ctx)

            if reward_type == "orca":
                # Adaptive tau: clamp closest-distance window to safe boundaries
                min_costs = np.min(cost, axis=0, keepdims=True)
                tau_local = np.clip(min_costs * 2.0, min_cost, max_cost)
                orca_cov, _ = compute_orca_reward(cost, tau=tau_local)
                # ORCA coverage is cumulative; the marginal transform is the meaningful form.
                trajectory_rew = (np.concatenate([[orca_cov[0]], np.diff(orca_cov)])
                                  if marginal_scoring else orca_cov)
            else:
                if reward_type == "dtw":
                    trajectory_rew, _ = compute_dtw_reward(cost)
                else:  # temporal_ot
                    trajectory_rew, _ = compute_temporal_ot_reward(cost)
                if marginal_scoring:
                    trajectory_rew = np.concatenate([[trajectory_rew[0]], np.diff(trajectory_rew)])

            reward_relative_to_expert[e_idx] = trajectory_rew.astype(np.float32)

        reward_all.append(reward_relative_to_expert)

    return reward_all


def extract_segment_pseudo_reward(rew, start, seq_len):
    """
    Extracts a segment of length seq_len starting at 'start'. Pads short trajectories with 0.

    Args:
        rew: (num_experts, L) array of rewards.
        start: Start index for the segment.
        seq_len: Target sequence length.

    Returns:
        segment_sum: (num_experts,) Sum of the rewards in the segment.
        segment_rewards: (num_experts, seq_len) The individual padded rewards.
    """
    # 1. Extract whatever is available from the start index up to the sequence limit
    segment = rew[:, start: start + seq_len]

    # 2. Calculate how much padding is needed to reach seq_len
    actual_len = segment.shape[-1]
    pad_amount = seq_len - actual_len

    # 3. Apply zero-padding to the time dimension if it's too short
    if pad_amount > 0:
        # pad_width format: ((dim0_before, dim0_after), (dim1_before, dim1_after))
        segment_rewards = np.pad(segment, ((0, 0), (0, pad_amount)), mode='constant', constant_values=0.0)
    else:
        segment_rewards = segment

    # 4. Calculate the sum (summing the zero-padded array is identical to summing the original slice)
    segment_sum = segment_rewards.sum(axis=-1)

    return segment_sum, segment_rewards