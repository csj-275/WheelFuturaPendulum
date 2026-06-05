"""快速定位 NaN 原因 - 只测一个配置，输出到文件避免被日志淹没。"""
import argparse, sys
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

def test(desc, events_dict):
    print(f"--- {desc} ---", flush=True)
    cfg = parse_env_cfg("Isaac-WheelFuturaPendulum-v0", device=args_cli.device, num_envs=1)
    cfg.events = type(cfg.events)()
    for k, v in events_dict.items():
        setattr(cfg.events, k, v)
    env = gym.make("Isaac-WheelFuturaPendulum-v0", cfg=cfg)
    for cycle in range(2):
        obs, _ = env.reset()
        for i in range(5):
            obs, _, _, _, _ = env.step(torch.zeros((1,1), device=args_cli.device))
            ot = obs["policy"] if isinstance(obs, dict) else obs
            if torch.isnan(ot).any():
                print(f"  NAN at cycle={cycle} step={i}", flush=True)
                env.close()
                return False
    print(f"  OK (2 cycles x 5 steps)", flush=True)
    env.close()
    return True

# Test 1: no events
test("No events", {})

# Test 2: only reset_joints_by_offset (default ±0.5 rad range)
# If this fails, it means the reset function itself is causing NaN
test("reset_joints_by_offset joint1 ±0.5", {
    "r1": EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint1"]),
                "position_range": (-0.5, 0.5), "velocity_range": (-0.5, 0.5)}),
})

print("DONE", flush=True)
simulation_app.close()
