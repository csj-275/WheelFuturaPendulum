"""模拟RL训练循环，看是否VecEnvWrapper导致NaN。"""
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
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
import WheelFuturaPendulum.tasks

env_cfg = parse_env_cfg("Isaac-WheelFuturaPendulum-v0", device=args_cli.device, num_envs=args_cli.num_envs)
env = gym.make("Isaac-WheelFuturaPendulum-v0", cfg=env_cfg)
env = RslRlVecEnvWrapper(env)

print(f"Observation space: {env.observation_space}", flush=True)
print(f"Action space: {env.action_space}", flush=True)

obs = env.reset()
if isinstance(obs, tuple):
    obs = obs[0]
obs_t = torch.from_numpy(obs).float()
print(f"After reset: obs NaN={torch.isnan(obs_t).any()}, shape={obs_t.shape}", flush=True)

for step in range(200):
    actions = torch.randn((args_cli.num_envs, 1))
    obs, rewards, dones, infos = env.step(actions)
    obs_t = torch.from_numpy(obs).float()

    if torch.isnan(obs_t).any():
        nan_count = torch.isnan(obs_t).any(dim=1).sum().item()
        print(f"Step {step}: NaN in {nan_count}/{args_cli.num_envs} envs", flush=True)
        env.close()
        simulation_app.close()
        raise SystemExit(0)

    if step % 50 == 0:
        print(f"Step {step}: obs [{obs_t.min().item():.2f}, {obs_t.max().item():.2f}]", flush=True)

print(f"200 steps with VecEnvWrapper OK, no NaN", flush=True)
env.close()
simulation_app.close()
