"""逐个定位哪个 reset 事件导致 NaN。"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from WheelFuturaPendulum.tasks.manager_based.wheelfuturapendulum import mdp
import WheelFuturaPendulum.tasks


def test_events(label, event_dict):
    print(f"\n{'='*50}")
    print(f"TEST: {label}")
    print(f"{'='*50}")
    env_cfg = parse_env_cfg("Isaac-WheelFuturaPendulum-v0", device=args_cli.device, num_envs=1)
    env_cfg.events = type(env_cfg.events)()
    for k, v in event_dict.items():
        setattr(env_cfg.events, k, v)
    try:
        env = gym.make("Isaac-WheelFuturaPendulum-v0", cfg=env_cfg)
        for cycle in range(3):
            obs, _ = env.reset()
            for i in range(10):
                obs, rew, term, trunc, _ = env.step(torch.zeros((1,1), device=args_cli.device))
                obs_t = obs["policy"] if isinstance(obs, dict) else obs
                if torch.isnan(obs_t).any():
                    print(f"  NaN at cycle {cycle}, step {i}!")
                    env.close()
                    return
        print(f"  3 cycles x 10 steps OK - no NaN")
        env.close()
    except Exception as e:
        print(f"  ERROR: {e}")


# Test: only joint1 reset (very small range)
test_events("joint1 only (±0.1 rad)", {
    "reset_j1": EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint1"]),
                "position_range": (-0.1, 0.1), "velocity_range": (-0.1, 0.1)}
    ),
})

# Test: only joint2 reset
test_events("joint2 only (±0.1 rad)", {
    "reset_j2": EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint2"]),
                "position_range": (-0.1, 0.1), "velocity_range": (-0.1, 0.1)}
    ),
})

# Test: both pendulum joints + wheel
test_events("all 3 joints (±0.1 rad)", {
    "reset_j1": EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint1"]),
                "position_range": (-0.1, 0.1), "velocity_range": (-0.1, 0.1)}
    ),
    "reset_j2": EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint2"]),
                "position_range": (-0.1, 0.1), "velocity_range": (-0.1, 0.1)}
    ),
    "reset_wheel": EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["wheel_joint"]),
                "position_range": (-0.1, 0.1), "velocity_range": (-0.1, 0.1)}
    ),
})

simulation_app.close()
