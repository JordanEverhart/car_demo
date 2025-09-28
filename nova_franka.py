# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Ridgeback-Manipulation robots.

The following configurations are available:

* :obj:`RIDGEBACK_FRANKA_PANDA_CFG`: Clearpath Ridgeback base with Franka Emika arm

Reference: https://github.com/ridgeback/ridgeback_manipulation
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Configuration
#/home/winter/robot/some_src/new_car/final/Collected_nova_franka_train/nova_franka_train.usd
##

NOVA_FRANKA_PANDA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"/home/winter/robot/some_src/new_car/final/Collected_nova_franka_train/nova_franka_train.usd",
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # base
            "joint_wheel_right": 0.0,
            "joint_wheel_left": 0.0,
            "joint_caster_base": 0.0,
            "joint_caster_left": 0.0,
            "joint_caster_right": 0.0,
            "joint_swing_left": 0.0,
            "joint_swing_right": 0.0,
            # franka arm
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 2.0,
            "panda_joint7": 0.741,
            # tool
            "panda_finger_joint.*": 0.04,
        },
        joint_vel={".*": 0.0},
    ),
        actuators={
        "base_move": ImplicitActuatorCfg(
            joint_names_expr=["joint_wheel.*"],
            velocity_limit=100.0,
            effort_limit=1000.0,
            stiffness=0,
            damping=100,
        ),
        "base_caster": ImplicitActuatorCfg(
            joint_names_expr=["joint_caster_base"],
            velocity_limit=100.0,
            effort_limit=1000.0,
            stiffness=0,
            damping=0,
        ),
        "base_swing": ImplicitActuatorCfg(
            joint_names_expr=["joint_swing.*"],
            velocity_limit=100.0,
            effort_limit=1000.0,
            stiffness=0,
            damping=0,
        ),
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit=30.0,             # 原为 87，改小降低力量
            velocity_limit=2.0,            # 原为 10.0，限制最大移动速度
            stiffness=150.0,               # 原为 500，减小刚度
            damping=80.0,                  # 原为 250，减小阻尼
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit=10.0,             # 原为 12.0，适当下调
            velocity_limit=2.0,
            stiffness=150.0,
            damping=80.0,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            effort_limit=200.0,
            velocity_limit=0.2,
            stiffness=1e5,
            damping=1e3,
        ),
    },
)
"""Configuration of Franka arm with Franka Hand on a Clearpath Ridgeback base using implicit actuator models.

The following control configuration is used:

* Base: velocity control
* Arm: position control with damping
* Hand: position control with damping

"""
