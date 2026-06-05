# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
import math
from . import mdp

##
# Pre-defined configs
##

from WheelFuturaPendulum.assets.robots import WHEEL_FUTURA_PENDULUM_CFG  # isort:skip

##
# Scene definition
##

@configclass
class WheelFuturaPendulumSceneCfg(InteractiveSceneCfg):
    """Configuration for a wheel-futura-pendulum scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # articulation
    robot: ArticulationCfg = WHEEL_FUTURA_PENDULUM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["wheel_joint"],
        scale=30.0)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        # sin/cos encoding for pendulum joints (avoids pi-wrap discontinuity)
        joint_sin_cos = ObsTerm(func=mdp.joint_sin_cos_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint1", "joint2"])})
        # raw position for wheel joint (no wrap issue)
        wheel_pos_rel = ObsTerm(func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wheel_joint"])})
        # velocities for all joints
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # reset joint1: small random offset
    reset_joint1_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint1"]),
            "position_range": (-0.5, 0.5),
            "velocity_range": (-0.5, 0.5),
        },
    )

    # reset joint2: small random offset
    reset_joint2_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint2"]),
            "position_range": (-0.5, 0.5),
            "velocity_range": (-0.5, 0.5),
        },
    )

    # reset wheel: small random rotation
    reset_wheel_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["wheel_joint"]),
            "position_range": (-0.5, 0.5),
            "velocity_range": (-0.5, 0.5),
        },
    )

@configclass
class RewardsCfg:
    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # primary task: smooth cos-based reward for joint1 (target π/2)
    upright_j1 = RewTerm(
        func=mdp.upright_reward_cos,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint1"]), "target": math.pi / 2},
    )

    # primary task: smooth cos-based reward for joint2 (target π)
    upright_j2 = RewTerm(
        func=mdp.upright_reward_cos,
        weight=5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint2"]), "target": math.pi},
    )

    # velocity penalty for pendulum joints
    joint_vel_pend = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.002,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint[1-2]"])},
    )

    # small velocity penalty for wheel
    joint_vel_wheel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wheel_joint"])},
    )

    # action penalty: discourage wasteful wheel spinning
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.001)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_state = DoneTerm(func=mdp.illegal_state)


##
# Environment configuration
##

@configclass
class WheelFuturaPendulumEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: WheelFuturaPendulumSceneCfg = WheelFuturaPendulumSceneCfg(num_envs=1024, env_spacing=4.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5
        # viewer settings
        self.viewer.eye = (8.0, 0.0, 5.0)
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physx.enable_external_forces_every_iteration = True