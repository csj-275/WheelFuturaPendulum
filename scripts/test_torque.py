# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
"""测试脚本：对动量轮施加力矩，观察两级摆杆和轮子的反应。

用途：诊断物理连接是否正确。
- 轮子转但摆杆不动 → joint1/joint2 的 stiffness/damping 没有被正确设为被动
- 摆杆跟着动 → 物理连接正确

使用方式：
    python scripts/test_torque.py
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test torque transmission on WheelFuturaPendulum.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-WheelFuturaPendulum-v0", help="Name of the task.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import WheelFuturaPendulum.tasks  # noqa: F401


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env_unwrapped = env.unwrapped
    robot = env_unwrapped.scene["robot"]

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    print(f"[INFO]: Robot DOF: {robot.num_joints}")
    print(f"[INFO]: Joint names: {robot.data.joint_names}")

    obs, _ = env.reset()

    step = 0
    header = f"{'step':>4s}  {'j1_pos':>8s}  {'j2_pos':>8s}  {'whl_pos':>8s}  "
    header += f"{'j1_vel':>8s}  {'j2_vel':>8s}  {'whl_vel':>8s}  {'torque':>6s}"
    print(f"\n{header}")
    print("-" * 75)

    while simulation_app.is_running():
        with torch.inference_mode():
            # Phase 1 (steps 0-100): push wheel forward
            if step < 100:
                actions = torch.tensor([[1.0]], device=env_unwrapped.device)
            # Phase 2 (steps 100-200): push wheel backward
            elif step < 200:
                actions = torch.tensor([[-1.0]], device=env_unwrapped.device)
            # Phase 3 (steps 200-300): zero torque, see passive behavior
            else:
                actions = torch.zeros((1, 1), device=env_unwrapped.device)

            obs, reward, terminated, truncated, _ = env.step(actions)

            if step % 5 == 0 or step < 20:
                jp = robot.data.joint_pos.cpu().numpy()[0]
                jv = robot.data.joint_vel.cpu().numpy()[0]
                print(f"{step:4d}  {jp[0]:+8.4f}  {jp[1]:+8.4f}  {jp[2]:+8.4f}  "
                      f"{jv[0]:+8.4f}  {jv[1]:+8.4f}  {jv[2]:+8.4f}  {actions[0,0].item():+6.1f}")

            step += 1
            if step > 1000:
                print("\nDone. Close window to exit.")
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
