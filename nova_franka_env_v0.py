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
    episode_length_s = 10.0
    action_scale = 10  # [N]
    action_space = 2
    observation_space = 23
    state_space = 0
    seed = 0
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = NOVA_FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    panda_joint_name = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]
    nova_joint_name = ["joint_wheel_left","joint_wheel_right"]
    ori_joint_name = ["joint_caster_left","joint_caster_right","joint_caster_base","joint_swing_left","joint_swing_right",]
    dist_limit = 0.1
    angle_limit = 0.01

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=4.0, replicate_physics=True)

    # reset
    dist_max = 100

class NovaFrankaNavEnv(DirectRLEnv):
    cfg: NovaFrankaNavEnvCfg

    def __init__(self, cfg: NovaFrankaNavEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.panda_joint_ids = [self.novafranka.data.joint_names.index(name) for name in self.cfg.panda_joint_name]
        self.nova_joint_ids = [self.novafranka.joint_names.index(name) for name in self.cfg.nova_joint_name]
        self.ori_joint_ids = [self.novafranka.joint_names.index(name) for name in self.cfg.ori_joint_name]
        self.action_scale = self.cfg.action_scale

        self.joint_pos = self.novafranka.data.joint_pos
        self.joint_vel = self.novafranka.data.joint_vel
        chassis_state=self.get_state_of_prim('chassis_link')
        self.target_pos = chassis_state[:,:2]
        self.target_quat = chassis_state[:,3:7]

    # def base_controller(self, linear_vel, angular_vel, wheel_radius=0.1, wheel_base=0.6):
    #     """
    #     根据期望线速度（m/s）和角速度（rad/s），计算左右轮目标转速，
    #     并下发给底盘轮子（假设底盘关节名称为 joint_wheel_left 和 joint_wheel_right）。
        
    #     公式：
    #     omega_right = (2*V + omega*b) / (2*r)
    #     omega_left  = (2*V - omega*b) / (2*r)
    #     """
    #     omega_right = (2 * linear_vel + angular_vel * wheel_base) / (2 * wheel_radius)
    #     omega_left  = (2 * linear_vel - angular_vel * wheel_base) / (2 * wheel_radius)
        
    #     base_cmd = torch.cat([omega_left, omega_right],dim=1)
        
    #     self.novafranka.set_joint_velocity_target(base_cmd, joint_ids=self.nova_joint_ids)

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
        self.actions = torch.clamp(self.action_scale * actions.clone(), min=-10.0, max=10.0)

    def _apply_action(self) -> None:
        # print("Action: ",self.actions)
        # self.base_controller(self.actions[:,0:1],self.actions[:,1:])
        self.novafranka.set_joint_velocity_target(self.actions, joint_ids=self.nova_joint_ids)

    def get_state_of_prim(self, prim_name):
        try:
            # Check if robot or relevant prim exists
            if self.novafranka is None or prim_name not in self.novafranka.data.body_names:
                raise ValueError(f"Prim {prim_name} is invalid or no longer exists.")
            
            index = self.novafranka.data.body_names.index(prim_name)
            state = self.novafranka.data.body_state_w[:, index, :]
            return state
        except Exception as e:
            print(f"Error getting state of prim '{prim_name}': {e}")
            return None  # or return default state, depending on your needs
        
    def _get_observations(self) -> dict:
        chassis_state=self.get_state_of_prim('chassis_link')
        # print(self.novafranka.data.root_pos_w)
        obs = torch.cat(
            (
                # self.novafranka.data.joint_pos[:,self.panda_joint_ids] ,  #7
                # self.novafranka.data.joint_vel[:,self.nova_joint_ids],   #2
                self.novafranka.data.joint_vel[:,self.nova_joint_ids], #2
                self.novafranka.data.joint_vel_target[:,self.nova_joint_ids],#2
                chassis_state[:,], #13
                self.target_pos, #2
                self.target_quat, #4
            ),
            dim=-1,
        )
        if torch.isnan(obs).any():
            # print("____________________________________________________________________________")
            obs = torch.zeros_like(obs)
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        chassis_state=self.get_state_of_prim('chassis_link')
        total_reward = compute_rewards(
            self.target_pos,
            self.target_quat,
            chassis_state[:,:2],
            chassis_state[:,3:7]
        )
        # tar_pos = self.target_pos
        # tar_quat = self.target_quat
        # cur_pos = chassis_state[:,:2]
        # cur_quat = chassis_state[:,3:7]
        # dist_error = torch.norm(tar_pos - cur_pos, dim=-1)
        # # 基础距离奖励：距离越小奖励越高
        # reward_dist = -dist_error
        # mask_dist_01= dist_error <0.1
        # reward_dist[mask_dist_01]+=100
        # mask_dist_1= dist_error <1
        # reward_dist[mask_dist_1]+=5
        # mask_dist_3=dist_error <3
        # reward_dist[mask_dist_3]+=1
        
        # # 计算期望朝向：从当前到目标的平面角度（以弧度计）
        # diff = tar_pos - cur_pos
        # # 使用 torch.atan2 计算目标平面方向
        # desired_yaw = torch.atan2(diff[..., 1], diff[..., 0])
        # # 当前机器人 yaw（从当前四元数提取）
        # current_yaw = quaternion_to_yaw(cur_quat)
        # # heading_error: 期望朝向与当前朝向之间的差值，归一化到 [-pi, pi]
        # heading_error = (desired_yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi
        # reward_heading = 0.5 * torch.abs(torch.abs(heading_error) - math.pi/2)*(dist_error+1)

        # # 当距离很近时，换用真正的四元数比较来计算朝向差异
        # dot = torch.sum(tar_quat * cur_quat, dim=-1)
        # # 保证 dot 在 [-1,1] 内
        # dot = torch.clamp(dot, -1.0, 1.0)
        # # 四元数相似性对应于角度差：theta = 2 * acos(|dot|)
        # yaw_diff = 2 * torch.acos(torch.abs(dot))
        # reward_orient = 10*(torch.pi - 0.5 * yaw_diff)
        # mask_yaw_diff_001=yaw_diff<0.01
        # reward_orient[mask_yaw_diff_001]+=10000
        # mask_yaw_diff_01=yaw_diff<0.1
        # reward_orient[mask_yaw_diff_01]+=1000
        # mask_yaw_diff_1=yaw_diff<1
        # reward_orient[mask_yaw_diff_1]+=100
        
        # # 定义距离切换阈值（例如 0.1 米），当距离大于该阈值时仅考虑 heading 奖励，
        # # 当距离小于该阈值时，采用完整的朝向对齐奖励。
        # threshold: float = 0.1
        # orientation_component = torch.where(dist_error > threshold, reward_heading, reward_orient)

        # # 增加时间惩罚（每个 step 都减一定值）
        # time_penalty = -0.01

        # total_reward = reward_dist + orientation_component + time_penalty
        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check if the episode is done based on target proximity or maximum steps."""
        chassis_state = self.get_state_of_prim('chassis_link')
        dist = torch.norm(self.target_pos - chassis_state[:, :2], dim=-1)  # Distance between target and current position

        # Compute quaternion difference
        target_quat = self.target_quat
        chassis_quat = torch.tensor(chassis_state[:, 3:7], dtype=torch.float32)  # quaternion for chassis
        angle = quaternion_to_angle(target_quat, chassis_quat)  # Angle difference between target and current quaternion

        time_out = self.episode_length_buf >= self.max_episode_length - 1
        done = (((dist < self.cfg.dist_limit) & (angle < self.cfg.angle_limit)) 
         | (chassis_state[:, 2] > 0.5)) 
        return done, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset the environment to a new initial state."""
        if env_ids is None:
            env_ids = self.novafranka._ALL_INDICES
        super()._reset_idx(env_ids)
        # print("reset_",env_ids)
        device=self.joint_pos.device

        # Specify the device, use CPU or CUDA based on available hardware

        # Generate random target position for each environment
        self.target_pos[env_ids] = sample_uniform(-10, 10,(len(env_ids),2),device)

        default_root_state = self.novafranka.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        # Generate random quaternion for target orientation
        target_quat = sample_uniform(-1, 1,(len(env_ids),4),device)
        self.target_quat[env_ids] = target_quat / target_quat.norm(dim=-1, keepdim=True)   # Normalize to make it a unit quaternion

        # Initialize random position for the robot
        init_pos_xy = sample_uniform(-10, 10,(len(env_ids),2),device)
        init_pos_z =  default_root_state[:, 2:3]+0.65
        theta = sample_uniform(-torch.pi, torch.pi, (len(env_ids), 1), device)  # 形状为 (num_envs, 1)

        # 计算绕 Z 轴旋转对应的四元数分量：
        w = torch.cos(theta / 2)
        z = torch.sin(theta / 2)
        # x, y 分量都为 0（仅绕 Z 轴旋转）
        x = torch.zeros_like(w)
        y = torch.zeros_like(w)

        # 组合为 [w, x, y, z] 顺序的四元数，形状为 (num_envs, 4)
        init_quat = torch.cat([w, x, y, z], dim=-1)
        # print(init_quat)
        # 如果需要确保数值精度，可再归一化一次（不过对这种构造的四元数已经是单位四元数了）
        init_quat = init_quat / init_quat.norm(dim=-1, keepdim=True)
        
        # Construct initial pose with random position and orientation
        init_pose = torch.cat([init_pos_xy,init_pos_z ,init_quat], dim=1)  # Concatenate position and quaternion

        # Apply initial pose to the simulation (write to the simulation environment)
        # self.novafranka.write_root_pose_to_sim(init_pose[:, :7], env_ids)
        self.novafranka.reset(env_ids)
        # self.novafranka.write_root_pose_to_sim(init_pose[:, :3], env_ids)
        # print('______________________RESSET_____________________________________')

# def _reset_idx(self, env_ids: Sequence[int] | None):
#         if env_ids is None:
#             env_ids = self.cartpole._ALL_INDICES
#         super()._reset_idx(env_ids)

#         joint_pos = self.cartpole.data.default_joint_pos[env_ids]
#         joint_pos[:, self._pole_dof_idx] += sample_uniform(
#             self.cfg.initial_pole_angle_range[0] * math.pi,
#             self.cfg.initial_pole_angle_range[1] * math.pi,
#             joint_pos[:, self._pole_dof_idx].shape,
#             joint_pos.device,
#         )
#         joint_vel = self.cartpole.data.default_joint_vel[env_ids]

#         default_root_state = self.cartpole.data.default_root_state[env_ids]
#         default_root_state[:, :3] += self.scene.env_origins[env_ids]

#         self.joint_pos[env_ids] = joint_pos
#         self.joint_vel[env_ids] = joint_vel

#         self.cartpole.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
#         self.cartpole.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
#         self.cartpole.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
# Quaternion multiplication (using Hamilton product)
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
    return angle

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
    cur_quat: torch.Tensor    # shape: (B, 4) 或 (4,), order: [qw, qx, qy, qz]
) -> torch.Tensor:
    dist_error = torch.norm(tar_pos - cur_pos, dim=-1)
        # 基础距离奖励：距离越小奖励越高
    reward_dist = -dist_error
    reward_dist = torch.where(dist_error < 0.1, reward_dist + 100000.0, reward_dist)
    reward_dist = torch.where(dist_error < 1.0, reward_dist + 5000.0, reward_dist)
    reward_dist = torch.where(dist_error < 3.0, reward_dist + 100.0, reward_dist)
    reward_dist = torch.where(dist_error < 6.0, reward_dist + 50.0, reward_dist)
    reward_dist = torch.where(dist_error < 10.0, reward_dist + 10.0, reward_dist)
    # if (dist_error < 6.0).any():
    #     print('HAVE6')
    if (dist_error < 1.0).any():
         print('HAVE1')
    # 计算期望朝向：从当前到目标的平面角度（以弧度计）
    diff = tar_pos - cur_pos
    # 使用 torch.atan2 计算目标平面方向
    desired_yaw = torch.atan2(diff[..., 1], diff[..., 0])
    # 当前机器人 yaw（从当前四元数提取）
    current_yaw = quaternion_to_yaw(cur_quat)
    # heading_error: 期望朝向与当前朝向之间的差值，归一化到 [-pi, pi]
    heading_error = (desired_yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi
    reward_heading =  (torch.abs(torch.abs(heading_error) - math.pi/2)-math.pi/2)*(dist_error/2)
    # reward_heading = torch.where(reward_heading>50, 50, reward_heading)

    # 当距离很近时，换用真正的四元数比较来计算朝向差异
    dot = torch.sum(tar_quat * cur_quat, dim=-1)
    # 保证 dot 在 [-1,1] 内
    dot = torch.clamp(dot, -1.0, 1.0)
    # 四元数相似性对应于角度差：theta = 2 * acos(|dot|)
    yaw_diff = 2 * torch.acos(torch.abs(dot))
    reward_orient = 10*(torch.pi - 0.5 * yaw_diff)
    reward_orient = torch.where(yaw_diff < 0.01, reward_orient + 10000000.0, reward_orient)
    reward_orient = torch.where(yaw_diff < 0.1, reward_orient + 100000.0, reward_orient)
    reward_orient = torch.where(yaw_diff < 0.5, reward_orient + 5000.0, reward_orient)
    reward_orient = torch.where(yaw_diff < 1.0, reward_orient + 500.0, reward_orient)

    
    # 定义距离切换阈值（例如 0.1 米），当距离大于该阈值时仅考虑 heading 奖励，
    # 当距离小于该阈值时，采用完整的朝向对齐奖励。
    threshold: float = 0.1
    orientation_component = torch.where(dist_error > threshold, reward_heading, reward_orient)

    # 增加时间惩罚（每个 step 都减一定值）
    time_penalty = -0.01

    total_reward = reward_dist + orientation_component + time_penalty
    # print(total_reward)
    return total_reward





import gymnasium as gym


gym.register(
    id="Isaac-NovaFrankaNav-v0",
    entry_point=f"nova_franka_env:NovaFrankaNavEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"nova_franka_env:NovaFrankaNavEnvCfg",
        "sb3_cfg_entry_point": f"sb3_ppo_cfg.yaml"
    },
)




