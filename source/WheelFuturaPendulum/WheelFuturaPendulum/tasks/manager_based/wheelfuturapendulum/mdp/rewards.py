# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def illegal_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Terminate if joint positions or velocities contain NaN/Inf values."""
    asset: Articulation = env.scene[asset_cfg.name]
    jp = asset.data.joint_pos[:, asset_cfg.joint_ids]
    jv = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return torch.isnan(jp).any(dim=1) | torch.isinf(jp).any(dim=1) | torch.isnan(jv).any(dim=1) | torch.isinf(jv).any(dim=1)


def joint_sin_cos_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Observation: sin and cos of joint positions (avoids pi-wrap discontinuity)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.cat([torch.sin(joint_pos), torch.cos(joint_pos)], dim=-1)


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    reward = torch.sum(torch.square(joint_pos - target), dim=1)
    return torch.nan_to_num(reward, nan=0.0, posinf=100.0, neginf=-100.0)


def upright_reward_cos(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Smooth cos-based reward: 1 at target, 0 at opposite, continuous gradient everywhere."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # cos(θ - target): 1 when aligned, -1 when opposite → map to [0, 1]
    reward = (torch.cos(joint_pos - target) + 1.0) / 2.0
    return reward.squeeze(-1)
