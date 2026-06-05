"""用随机动作模拟 PPO 初始策略。"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=1024)
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

for step in range(50):
    actions = torch.randn((args_cli.num_envs, 1), device=args_cli.device)
    obs, rew, term, trunc, _ = env.step(actions)
    obs_t = obs["policy"] if isinstance(obs, dict) else obs

    if torch.isnan(obs_t).any():
        nan_count = torch.isnan(obs_t).any(dim=1).sum().item()
        print(f"Step {step}: NaN in {nan_count}/{args_cli.num_envs} envs", flush=True)
        env.close()
        simulation_app.close()
        raise SystemExit(0)

    if step == 0:
        print(f"Step 0 OK: obs [{obs_t.min().item():.3f}, {obs_t.max().item():.3f}]", flush=True)

print(f"50 random-action steps OK, no NaN", flush=True)
env.close()
simulation_app.close()
