"""在 NaN 发生时打印观测值和关节状态的详细信息。"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=4096)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
import WheelFuturaPendulum.tasks

env_cfg = parse_env_cfg("Isaac-WheelFuturaPendulum-v0", device=args_cli.device, num_envs=args_cli.num_envs)
env = gym.make("Isaac-WheelFuturaPendulum-v0", cfg=env_cfg)
env.reset()

robot = env.unwrapped.scene["robot"]

for step in range(500):
    obs, rew, term, trunc, _ = env.step(torch.zeros((args_cli.num_envs, 1), device=args_cli.device))
    obs_t = obs["policy"] if isinstance(obs, dict) else obs
    if torch.isnan(obs_t).any():
        nan_mask = torch.isnan(obs_t).any(dim=1)
        nan_ids = torch.where(nan_mask)[0]
        print(f"\nNaN at step {step}, {len(nan_ids)} envs have NaN!")
        for nid in nan_ids[:5]:
            jp = robot.data.joint_pos[nid].cpu()
            jv = robot.data.joint_vel[nid].cpu()
            print(f"  Env {nid}: obs={obs_t[nid].cpu()}")
            print(f"    joint_pos={jp}")
            print(f"    joint_vel={jv}")
        break

    if step % 100 == 0:
        print(f"Step {step}: max obs={obs_t.max().item():.4f}, min={obs_t.min().item():.4f}")

env.close()
simulation_app.close()
