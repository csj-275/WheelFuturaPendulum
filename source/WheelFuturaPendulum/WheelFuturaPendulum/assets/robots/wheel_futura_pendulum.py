# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# created by csj, 2026-04-26
"""Configuration for a simple WheelFuturaPendulum robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

##
# Configuration
##

WHEEL_FUTURA_PENDULUM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="./usd/WheelFuturaPendulum.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            rigid_body_enabled=True,
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=False,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={"joint1": 0.0,
                   "joint2": 0.0,
                   "wheel_joint": 0.0}
    ),
    # 关节驱动器
    actuators={
        "joint1_actuator": ImplicitActuatorCfg(
            joint_names_expr=["joint1"],
            effort_limit_sim=400.0,
            stiffness=0.0, # 被动关节，无位置控制
            damping=0.01, # 少量阻尼防止数值发散
        ),
        "joint2_actuator": ImplicitActuatorCfg(
            joint_names_expr=["joint2"],
            effort_limit_sim=400.0,
            stiffness=0.0, # 被动关节，无位置控制
            damping=0.01, # 少量阻尼防止数值发散
        ),
        "wheel_actuator": ImplicitActuatorCfg(
            joint_names_expr=["wheel_joint"],
            effort_limit_sim=400.0,
            stiffness=0.0, # 位置控制的刚度，力控设为0
            damping=0.01, # 阻尼
        ),
    },
)
"""Configuration for a simple WheelFuturaPendulum robot."""
