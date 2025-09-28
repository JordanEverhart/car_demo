# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import numpy as np
import math
import torch
from collections.abc import Sequence

from nova_franka import NOVA_FRANKA_PANDA_CFG
from nova_franka_controller import NovaFranka_Controller
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform
from scipy.spatial.transform import Rotation as R

@configclass
class NovaFrankaNavEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 240.0
    action_scale = 0.01  # [N]
    action_space = 2
    observation_space = 12
    state_space = 0
    seed = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = NOVA_FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    dist_limit = 0.1
    angle_limit = 0.01
    num_envs=512
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=0, replicate_physics=True)

    # reset
    dist_max = 30

class NovaFrankaNavEnv(DirectRLEnv):
    cfg: NovaFrankaNavEnvCfg

    def __init__(self, cfg: NovaFrankaNavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.action_scale = self.cfg.action_scale
        self.target_pos = sample_uniform(-10, 10,(self.cfg.num_envs,2),'cuda')
        self.target_quat = sample_uniform(-10, 10,(self.cfg.num_envs,4),'cuda')
        self.curren_pos= sample_uniform(-1, 1,(self.cfg.num_envs,2),'cuda')
        self.curren_quat = self.target_quat

    def _setup_scene(self):
        self.novafranka = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add articulation to scene
        self.scene.articulations["novafranka"] = self.novafranka
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # 对 actions 进行截断，确保速度不超过 3
        self.actions = torch.clamp(self.action_scale * actions.clone(), min=-4.0, max=4.0)

    def _apply_action(self) -> None:
        self.curren_pos+=self.actions
  
    def _get_observations(self) -> dict:
        # print(self.novafranka.data.root_pos_w)
        obs = torch.cat(
            (   
                self.curren_pos, #13
                self.curren_quat,
                self.target_pos, #2
                self.target_quat
            ),
            dim=-1,
        )
        if torch.isnan(obs).any():
            # print("____________________________________________________________________________")
            obs = torch.zeros_like(obs)
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        total_reward = compute_rewards(
            self.target_pos,
            self.target_quat,
            self.curren_pos,
            self.curren_quat,
        )
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check if the episode is done based on target proximity or maximum steps."""
        dist = torch.norm(self.target_pos - self.curren_pos, dim=-1)  # Distance between target and current position

        # Compute quaternion difference
        target_quat = self.target_quat
        tar_yaw = quaternion_to_yaw(target_quat)
        current_yaw = quaternion_to_yaw(self.target_quat) # Angle difference between target and current quaternion
        yaw_error = abs((tar_yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        done = ((dist < self.cfg.dist_limit) & (yaw_error < self.cfg.angle_limit)) | (dist > self.cfg.dist_max)

        return done, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset the environment to a new initial state."""
        if env_ids is None:
            env_ids = self.novafranka._ALL_INDICES
        super()._reset_idx(env_ids)
        # print("reset_",env_ids)
        device='cuda'

        # Specify the device, use CPU or CUDA based on available hardware

        # Generate random target position for each environment
        self.target_pos[env_ids] = sample_uniform(-10, 10,(len(env_ids),2),device)

        default_root_state = self.novafranka.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Generate random quaternion for target orientation
        target_quat = sample_uniform(-1, 1,(len(env_ids),4),device)
        self.target_quat[env_ids] = target_quat / target_quat.norm(dim=-1, keepdim=True)   # Normalize to make it a unit quaternion
        self.curren_quat = self.target_quat
        # # Initialize random position for the robot
        # init_pos_xy = sample_uniform(-10, 10,(len(env_ids),2),device)
        # init_pos_z =  default_root_state[:, 2:3]+0.65
        # theta = sample_uniform(-torch.pi, torch.pi, (len(env_ids), 1), device)  # 形状为 (num_envs, 1)

        # # 计算绕 Z 轴旋转对应的四元数分量：
        # w = torch.cos(theta / 2)
        # z = torch.sin(theta / 2)
        # # x, y 分量都为 0（仅绕 Z 轴旋转）
        # x = torch.zeros_like(w)
        # y = torch.zeros_like(w)

        # # 组合为 [w, x, y, z] 顺序的四元数，形状为 (num_envs, 4)
        # init_quat = torch.cat([w, x, y, z], dim=-1)
        # # print(init_quat)
        # # 如果需要确保数值精度，可再归一化一次（不过对这种构造的四元数已经是单位四元数了）
        # init_quat = init_quat / init_quat.norm(dim=-1, keepdim=True)
        
        # # Construct initial pose with random position and orientation
        # init_pose = torch.cat([init_pos_xy,init_pos_z ,init_quat], dim=1)  # Concatenate position and quaternion

        # Apply initial pose to the simulation (write to the simulation environment)
        # self.novafranka.write_root_pose_to_sim(init_pose[:, :7], env_ids)
        self.novafranka.reset(env_ids)

            
        # self.novafranka.write_root_pose_to_sim(init_pose[:, :3], env_ids)
        # print('______________________RESSET_____________________________________')

@torch.jit.script
def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z], dim=-1)

# Function to compute the angle difference between two quaternions
@torch.jit.script
def quaternion_to_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    q_diff = quaternion_multiply(q1, q2)  # q1 * q2^-1
    # Compute the angle from the quaternion (w = cos(theta/2), so theta = 2 * acos(w))
    angle = 2 * torch.acos(torch.abs(q_diff[:, 0]))  # q_diff[:, 0] is the real part (w)
    return angle.abs()

@torch.jit.script
def quaternion_to_yaw(q: torch.Tensor) -> torch.Tensor:
    # q: Tensor of shape [4] with order [qw, qx, qy, qz]
    qw = q[:,0]
    qx = q[:,1]
    qy = q[:,2]
    qz = q[:,3]
    siny = 2 * (qw * qz + qx * qy)
    cosy = 1 - 2 * (qy * qy + qz * qz)
    return torch.atan2(siny, cosy)

# Rewards computation function
@torch.jit.script
def compute_rewards(
    tar_pos: torch.Tensor,    # shape: (B, 3) 或 (3,)
    tar_quat: torch.Tensor,   # shape: (B, 4) 或 (4,), order: [qw, qx, qy, qz]
    cur_pos: torch.Tensor,    # shape: (B, 3) 或 (3,)
    cur_quat: torch.Tensor,  # shape: (B, 4) 或 (4,), order: [qw, qx, qy, qz]
) -> torch.Tensor:
    dist_max = 30
    dist_error = torch.norm(tar_pos - cur_pos, dim=-1)
        # 基础距离奖励：距离越小奖励越高
    reward_dist = ((dist_max-dist_error)/dist_max)**2
    reward_dist = torch.where(dist_error < 0.1, reward_dist + 10.0, reward_dist)
    reward_dist = torch.where(dist_error > dist_max, 0, reward_dist)
    # if (dist_error > dist_max).any():
    #     print('out_range2')

    # 计算期望朝向：从当前到目标的平面角度（以弧度计）
    diff = tar_pos - cur_pos
    # 使用 torch.atan2 计算目标平面方向
    desired_yaw = torch.atan2(diff[..., 1], diff[..., 0])
    # 当前机器人 yaw（从当前四元数提取）
    current_yaw = quaternion_to_yaw(cur_quat)
    # heading_error: 期望朝向与当前朝向之间的差值，归一化到 [-pi, pi]
    heading_error = (desired_yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi
    reward_heading =  (torch.abs(torch.abs(heading_error) - math.pi/2)/(math.pi/2))
    # reward_heading = torch.where(reward_heading, -10, reward_heading)

    # 当距离很近时，换用真正的四元数比较来计算朝向差异
    tar_yaw = quaternion_to_yaw(tar_quat)
    # 保证 dot 在 [-1,1] 内
    # 四元数相似性对应于角度差：theta = 2 * acos(|dot|)
    yaw_error = abs((tar_yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi)
    reward_orient = ((torch.pi - yaw_error)*5/torch.pi)**2
    reward_orient = torch.where(yaw_error < 0.01, reward_orient + 1000.0, reward_orient)
    reward_orient = torch.where(yaw_error < 0.05, reward_orient + 100.0, reward_orient)
    
    # 定义距离切换阈值（例如 0.1 米），当距离大于该阈值时仅考虑 heading 奖励，
    # 当距离小于该阈值时，采用完整的朝向对齐奖励。
    threshold: float = 0.1
    orientation_component = torch.where(dist_error > threshold, reward_heading, reward_orient*5)

    # 增加时间惩罚（每个 step 都减一定值）
    total_reward = reward_dist+orientation_component

    # print(total_reward)
    return total_reward


import gymnasium as gym


gym.register(
    id="Isaac-NovaFrankaNav-simpy",
    entry_point=f"nova_franka_env_simpy:NovaFrankaNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"nova_franka_env_simpy:NovaFrankaNavEnvCfg",
        "sb3_cfg_entry_point": f"sb3_ppo_cfg_simpy.yaml"
    },
)




