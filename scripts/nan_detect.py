"""极简测试：纯零动作，看 reset 后 NaN 产生的细节。"""
import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--num_envs", type=int, default=256)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
import WheelFuturaPendulum.tasks

env_cfg = parse_env_cfg("Isaac-WheelFuturaPendulum-v0", device=args_cli.device, num_envs=args_cli.num_envs)
env = gym.make("Isaac-WheelFuturaPendulum-v0", cfg=env_cfg)
obs, _ = env.reset()
robot = env.unwrapped.scene["robot"]

print(f"default_root_state pos z: {robot.data.default_root_state[0, 2].item():.3f}", flush=True)
print(f"Joint names: {robot.data.joint_names}", flush=True)

for step in range(50):
    obs, rew, term, trunc, _ = env.step(torch.zeros((args_cli.num_envs, 1), device=args_cli.device))
    obs_t = obs["policy"] if isinstance(obs, dict) else obs
    jp = robot.data.joint_pos
    jv = robot.data.joint_vel

    nan_obs = torch.isnan(obs_t).any(dim=1)
    nan_jp = torch.isnan(jp).any(dim=1)
    nan_jv = torch.isnan(jv).any(dim=1)
    has_any_nan = nan_obs.any() or nan_jp.any() or nan_jv.any()

    if has_any_nan:
        print(f"\nStep {step}: NaN detected!", flush=True)
        print(f"  nan in obs: {nan_obs.sum().item()}/{args_cli.num_envs}", flush=True)
        print(f"  nan in joint_pos: {nan_jp.sum().item()}/{args_cli.num_envs}", flush=True)
        print(f"  nan in joint_vel: {nan_jv.sum().item()}/{args_cli.num_envs}", flush=True)
        if nan_jp.any():
            idx = torch.where(nan_jp)[0][0].item()
            print(f"  Example env {idx}: joint_pos={jp[idx].cpu()}", flush=True)
            print(f"    joint_vel={jv[idx].cpu()}", flush=True)
            print(f"    obs={obs_t[idx].cpu()}", flush=True)
        break

    if step == 0:
        print(f"Step 0: obs range [{obs_t.min().item():.3f}, {obs_t.max().item():.3f}]", flush=True)
        print(f"  joint_pos range [{jp.min().item():.3f}, {jp.max().item():.3f}]", flush=True)
        print(f"  joint_vel range [{jv.min().item():.3f}, {jv.max().item():.3f}]", flush=True)

if not has_any_nan:
    print(f"50 steps OK, no NaN", flush=True)

env.close()
simulation_app.close()
