"""测试RL环境：禁用reset事件，隔离NaN根因。"""
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
import WheelFuturaPendulum.tasks

# 1. Test with NO events
print("=" * 60)
print("TEST 1: No events, zero action")
print("=" * 60)
env_cfg = parse_env_cfg("Isaac-WheelFuturaPendulum-v0", device=args_cli.device, num_envs=1)
env_cfg.events = type(env_cfg.events)()
env = gym.make("Isaac-WheelFuturaPendulum-v0", cfg=env_cfg)
env.reset()
robot = env.unwrapped.scene["robot"]
for i in range(10):
    obs, rew, term, trunc, _ = env.step(torch.zeros((1, 1), device=args_cli.device))
    jp = robot.data.joint_pos
    jv = robot.data.joint_vel
    obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
    has_nan = torch.isnan(obs_tensor).any()
    print(f"Step {i}: obs NaN={has_nan}, pos={jp.cpu().numpy().tolist()[0]}")
    if has_nan:
        break
env.close()

# 2. Test WITH events
print("\n" + "=" * 60)
print("TEST 2: With default events, zero action")
print("=" * 60)
env_cfg2 = parse_env_cfg("Isaac-WheelFuturaPendulum-v0", device=args_cli.device, num_envs=1)
env2 = gym.make("Isaac-WheelFuturaPendulum-v0", cfg=env_cfg2)
env2.reset()
robot2 = env2.unwrapped.scene["robot"]
for i in range(10):
    obs, rew, term, trunc, _ = env2.step(torch.zeros((1, 1), device=args_cli.device))
    jp = robot2.data.joint_pos
    jv = robot2.data.joint_vel
    obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
    has_nan = torch.isnan(obs_tensor).any()
    print(f"Step {i}: obs NaN={has_nan}, pos={jp.cpu().numpy().tolist()[0]}")
    if has_nan:
        break
env2.close()

# 3. Test with events AND random/reset
print("\n" + "=" * 60)
print("TEST 3: With events + manual reset cycle")
print("=" * 60)
env_cfg3 = parse_env_cfg("Isaac-WheelFuturaPendulum-v0", device=args_cli.device, num_envs=1)
env3 = gym.make("Isaac-WheelFuturaPendulum-v0", cfg=env_cfg3)
for cycle in range(3):
    obs, _ = env3.reset()
    obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
    print(f"Reset {cycle}: obs NaN={torch.isnan(obs_tensor).any()}")
    for i in range(5):
        obs, rew, term, trunc, _ = env3.step(torch.zeros((1, 1), device=args_cli.device))
        obs_tensor = obs["policy"] if isinstance(obs, dict) else obs
        has_nan = torch.isnan(obs_tensor).any()
        if has_nan:
            print(f"  Step {i}: NaN!")
    else:
        print(f"  5 steps OK")
env3.close()

print("\nAll tests done!")
simulation_app.close()
