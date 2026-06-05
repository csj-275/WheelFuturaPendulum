# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
"""诊断 NaN 根因测试脚本。"""

"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description="Debug NaN for WheelFuturaPendulum.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest follows."""
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from WheelFuturaPendulum.assets.robots import WHEEL_FUTURA_PENDULUM_CFG

@configclass
class TestSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    robot: ArticulationCfg = WHEEL_FUTURA_PENDULUM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view((2.5, 0.0, 4.0), (0.0, 0.0, 2.0))
    scene_cfg = TestSceneCfg(num_envs=1, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    print("Joint names:", robot.data.joint_names)
    print("Default root state:", robot.data.default_root_state)
    print("Joint pos:", robot.data.joint_pos)

    sim_dt = sim.get_physics_dt()
    for i in range(20):
        sim.step()
        scene.update(sim_dt)
        jp = robot.data.joint_pos
        jv = robot.data.joint_vel
        has_nan = torch.isnan(jp).any() or torch.isnan(jv).any()
        print(f"Step {i}: pos={jp.cpu().numpy().tolist()[0]}, vel={jv.cpu().numpy().tolist()[0]}, nan={has_nan}")

    print("\nDone.")
    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()
