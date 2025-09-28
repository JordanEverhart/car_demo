# -*- coding: utf-8 -*-
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from pxr import Usd, UsdGeom, Gf
from isaaclab.assets import Articulation
from nova_franka import NOVA_FRANKA_PANDA_CFG
from franka_ik import FrankaEasyIK
from isaacsim.core.utils.prims import get_prim_at_path
from pxr import UsdGeom
import torch
from isaaclab.assets import RigidObject, RigidObjectCfg
import isaaclab.sim as sim_utils

# ----------------------------
# 创建机器人接口
# ----------------------------
POS_INIT = [0.0, -0.12, 0.0]
ORI_INIT =  (1.0,0.0, 0.0, 0.0)

#注意是传入WXYZ
def add_robot(position=POS_INIT,orientation=ORI_INIT
,prim_name="/Robot_1")-> Articulation:   #事兴的场景，起始坐标应该是 8.0, -1.0, 0.0  棋子附近是0.41, -0.8, 0.0
    """在场景中加载机器人模型"""
    robot_cfg = NOVA_FRANKA_PANDA_CFG
    robot_cfg.spawn.func(prim_name, robot_cfg.spawn,
                        translation=position,
                        orientation=orientation)
    robot = Articulation(cfg=robot_cfg.replace(prim_path=prim_name))
    return robot


def interpolate_positions(current, target, n_points):
    current = np.array(current)
    target = np.array(target)
    traj = [list(current + (target - current) * (i + 1) / n_points) for i in range(n_points)]
    return traj

# ----------------------------
# 辅助函数：四元数归一化
# ----------------------------
def normalize_quaternion(q):
    q = np.array(q)
    return (q / np.linalg.norm(q)).tolist()

# ----------------------------
# 辅助函数：四元数乘法（假设顺序为 [w, x, y, z]）
# ----------------------------
def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return normalize_quaternion([w, x, y, z])

# ----------------------------
# 辅助函数：球面线性插值（slerp）生成四元数插值
# ----------------------------
def slerp_quaternion(q0, q1, t):
    q0 = np.array(q0, dtype=float)
    q1 = np.array(q1, dtype=float)
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    DOT_THRESHOLD = 0.9995
    if dot > DOT_THRESHOLD:
        result = q0 + t * (q1 - q0)
        return normalize_quaternion(result)
    theta_0 = np.arccos(dot)
    theta = theta_0 * t
    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)
    result = q0 * np.cos(theta) + q2 * np.sin(theta)
    return normalize_quaternion(result)

def interpolate_quaternions(q0, q1, n_points):
    """生成 q0 到 q1 的 n_points 个 slerp 插值点（不包含 q0，包含 q1）"""
    return [slerp_quaternion(q0, q1, (i+1)/n_points) for i in range(n_points)]

class NovaFranka_Controller:

    def __init__(self,robot,sim_dt):
        self.ik = FrankaEasyIK()
        self.robot = robot
        self.sim_dt = sim_dt
        self.panda_joint_name = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]
        self.nova_joint_name = ["joint_wheel_left","joint_wheel_right"]
        self.ori_joint_name = ["joint_caster_left","joint_caster_right","joint_caster_base","joint_swing_left","joint_swing_right",]
        self.panda_finger_names = ["panda_finger_joint1", "panda_finger_joint2"]
        self.panda_joint_ids = [robot.data.joint_names.index(name) for name in self.panda_joint_name]
        self.nova_joint_ids = [robot.data.joint_names.index(name) for name in self.nova_joint_name]
        self.ori_joint_ids = [robot.data.joint_names.index(name) for name in self.ori_joint_name]
        self.panda_finger_ids = [robot.data.joint_names.index(name) for name in self.panda_finger_names]
        self.cur_chassis_pos = None
        self.cur_chassis_quat = None
        self.cur_panda_hand_pos = None
        self.cur_panda_hand_quat = None
        self.update_robot_state()
        self.tar_chassis_pos = self.cur_chassis_pos.copy()
        self.tar_chassis_quat = self.cur_chassis_quat.copy()
        self.tar_panda_hand_pos = self.cur_panda_hand_pos.copy()
        self.tar_panda_hand_quat = self.cur_panda_hand_quat.copy()

        self.arm_cmd_traj_pos = []    # 每个元素为一个 [x, y, z]
        self.arm_cmd_traj_quat = []   # 每个元素为一个四元数 [w, x, y, z]
        self.step_time=0
        self.move_flag = True
        self.ori_flag = True
        self.mova_ang = 0
        self.linear_cmd = 0
        self.angular_cmd = 0

        ###坐标移动
        self.objects = {}
        self.object_opts = {}
        
    
    def base_controller(self, linear_vel, angular_vel, wheel_radius=0.14, wheel_base=0.3452):
        """
        根据期望线速度（m/s）和角速度（rad/s），计算左右轮目标转速，
        并下发给底盘轮子（假设底盘关节名称为 joint_wheel_left 和 joint_wheel_right）。
        
        公式：
        omega_right = (2*V + omega*b) / (2*r)
        omega_left  = (2*V - omega*b) / (2*r)
        """
        omega_right = (2 * linear_vel + angular_vel * wheel_base) / (2 * wheel_radius)
        omega_left  = (2 * linear_vel - angular_vel * wheel_base) / (2 * wheel_radius)
        
        base_cmd = torch.tensor([[omega_left, omega_right]], device=self.robot.device, dtype=torch.float32)

        print(f"cmd ω_L={omega_left:.3f}, ω_R={omega_right:.3f}")
        
        self.robot.set_joint_velocity_target(base_cmd, joint_ids=self.nova_joint_ids)

    def step(self):
        self.update_robot_state()
        # self.nova_controller()
        self.panda_controller()
        self.robot.write_data_to_sim()
        self.robot.update(self.sim_dt)  

        # 先更新对象缓存
        for obj in self.objects.values():
            obj.update(self.sim_dt)

        # 对每个物体执行写入（是否吸附由每个物体的 attract 决定）
        for name in list(self.objects.keys()):
            self.write_object_state(name)
                
        # self.step_time += 1
        # if self.step_time % 10 == 0:


    def nova_controller(self):
        """
        当当前底盘状态与目标底盘状态不一致时，采用两阶段控制：
        1. 当位置误差较大时，调整方向为目标位置方向，再前进。
        2. 当位置到达目标位置后，仅调整 yaw 角以达到目标姿态。
        """
        #当前底盘状态（平面 x,y）与 yaw 角
        cur_pos = np.array(self.cur_chassis_pos[:2])
        cur_quat = [self.cur_chassis_quat[1], self.cur_chassis_quat[2], self.cur_chassis_quat[3], self.cur_chassis_quat[0]] # 顺序： [qx, qy, qz, qw]
        current_yaw = R.from_quat(cur_quat).as_euler('zyx', degrees=False)[0]

        # 目标底盘状态（仅考虑 x,y 及 yaw）
        tar_pos = np.array(self.tar_chassis_pos[:2])
        tar_quat = [self.tar_chassis_quat[1], self.tar_chassis_quat[2], self.tar_chassis_quat[3], self.tar_chassis_quat[0]]
        target_yaw = R.from_quat(tar_quat).as_euler('zyx', degrees=False)[0]
        
        # 计算位置误差与距离
        pos_error = tar_pos - cur_pos
        dist_error = np.linalg.norm(pos_error)
        
        # 定义位置阈值
        pos_threshold = 0.15  # 当距离误差小于 0.1 m 时，认为已经到达目标位置

        if dist_error > pos_threshold and self.move_flag:
            # 远离目标时：调整方向并前进
            # 计算当前方向和目标位置之间的角度
            desired_yaw = np.arctan2(pos_error[1], pos_error[0])
            
            # 计算当前 yaw 与目标 yaw 之间的差异
            yaw_error = desired_yaw - current_yaw
            yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi  # 归一化到 [-pi, pi]
            
            # 处理角度差异大于180度的情况
            if abs(yaw_error) > np.pi:
                yaw_error = yaw_error - np.sign(yaw_error) * 2 * np.pi
            
            if abs(yaw_error) > 0.005 and self.ori_flag:
                self.linear_cmd = 0
                # 调整方向
                self.angular_cmd = min(2*yaw_error,0.2)  # 比例控制，Kp_yaw 可根据需要调整
            else:
                if abs(yaw_error) > min(0.3,dist_error):
                    self.ori_flag=True
                    self.angular_cmd = 0
                    self.linear_cmd = 0
                else:
                    self.ori_flag=False
                    self.angular_cmd = 0.2*yaw_error*min(dist_error,5)
                    self.linear_cmd = dist_error
        else:
            self.move_flag = False
            # 靠近目标时：只控制旋转
            yaw_error = target_yaw - current_yaw
            yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi  # 归一化到 [-pi, pi]
            
            # 处理角度差异大于180度的情况
            if abs(yaw_error) > np.pi:
                yaw_error = yaw_error - np.sign(yaw_error) * 2 * np.piupdate_arm_cmd_traj
            
            # 仅控制角速度来调整 yaw
            self.linear_cmd = 0.0
            self.angular_cmd = min(2*yaw_error,0.2) # 比例控制，Kp_yaw 可根据需要调整
            
            if dist_error > 2 * pos_threshold:
                self.move_flag = True
                self.linear_cmd = 0.0
                self.angular_cmd = 0.0 

        # 限制命令幅值
        self.linear_cmd = np.clip(self.linear_cmd, -5.0, 5.0)
        self.angular_cmd = np.clip(self.angular_cmd, -2.0, 2.0)
        # 调用底盘控制函数
        # print(self.angular_cmd,self.linear_cmd)
        self.base_controller(self.linear_cmd, self.angular_cmd)

        # self.linear_cmd = 5.0     # 要移动大概设成5左右比较好
        # self.angular_cmd = 0.0     # 旋转大概设成30
        # self.base_controller(self.linear_cmd, self.angular_cmd)

    def set_finger_width(self, width):
        """
        设置夹爪开闭宽度（单位米）
        Franka Panda 夹爪最大宽度大约为 0.08m (8cm)。

        参数:
            width: 目标夹爪宽度, 范围 [0.0, 0.08]
        """
        width = np.clip(width, 0.0, 0.08)

        # 单侧手指的关节位置为夹爪宽度的一半
        finger_pos = width / 2.0

        # 创建tensor并发送指令
        finger_cmd = torch.tensor([[finger_pos, finger_pos]], device=self.robot.device, dtype=torch.float32)
        self.robot.set_joint_position_target(finger_cmd, joint_ids=self.panda_finger_ids)

    def panda_controller(self):
        if self.arm_cmd_traj_quat or self.arm_cmd_traj_pos:
            if self.arm_cmd_traj_pos:
            # 取出第一个插值点
                target_pos = self.arm_cmd_traj_pos.pop(0)
            else:
                target_pos = self.tar_panda_hand_pos
            if self.arm_cmd_traj_quat:
                target_quat = self.arm_cmd_traj_quat.pop(0)
            else:
                target_quat = self.tar_panda_hand_quat
            try:
                target_joint_positions = self.ik(target_pos, target_quat)
                target_joint_positions = torch.tensor([target_joint_positions], device=self.robot.device, dtype=torch.float32)
                self.robot.set_joint_position_target(target_joint_positions, joint_ids=self.panda_joint_ids)
            except Exception as e:
                print(e)
                pass

    def set_tar_nova_pos(self,pos):
        assert len(pos) == 3, f"position length: {len(pos)} != 3"
        self.tar_chassis_pos=pos

    def set_tar_nova_quat(self,quat):
        assert len(quat) == 4, f"position length: {len(quat)} != 4"
        self.tar_chassis_quat=quat

    def set_tar_nova(self,pos,quat):
        assert len(pos) == 3, f"position length: {len(pos)} != 3"
        assert len(quat) == 4, f"position length: {len(quat)} != 4"
        self.tar_chassis_pos=pos
        self.tar_chassis_quat=quat

    def set_tar_panda_hand_pos(self,pos):
        assert len(pos) == 4, f"position length: {len(pos)} != 4"
        self.tar_panda_hand_pos=pos
        self.update_arm_cmd_traj()

    def set_tar_panda_hand_quat(self,quat):
        assert len(quat) == 4, f"position length: {len(quat)} != 4"
        self.tar_panda_hand_quat=quat
        self.update_arm_cmd_traj()

    def set_tar_pand_hand(self,pos,quat):
        assert len(pos) == 3, f"position length: {len(pos)} != 3"
        assert len(quat) == 4, f"position length: {len(quat)} != 4"

        # 直接插值规划
        self.tar_panda_hand_pos=pos
        self.tar_panda_hand_quat=quat
        self.update_arm_cmd_traj()

    def set_tar_pand_hand_from_top(self,pos,quat):
        assert len(pos) == 3, f"position length: {len(pos)} != 3"
        assert len(quat) == 4, f"position length: {len(quat)} != 4"

        self.tar_panda_hand_pos = pos
        self.tar_panda_hand_quat = quat

        # 生成 top-down 轨迹
        pos_traj, quat_traj = self._plan_top_down(
            cur_pos=self.cur_panda_hand_pos,
            cur_quat=self.cur_panda_hand_quat,
            tar_pos=self.tar_panda_hand_pos,
            tar_quat=self.tar_panda_hand_quat,
            hover_clearance=0.25,
            min_hover_z=0.6
        )
        self.arm_cmd_traj_pos = pos_traj
        self.arm_cmd_traj_quat = quat_traj

    def _plan_from_y_negative(self, cur_pos, cur_quat, tar_pos, tar_quat,
                          retreat_distance=0.15,   # y轴负方向后退距离
                          min_clearance=0.3,      # 最低 y 方向悬停距离
                          base_step=0.02,
                          min_steps=3,
                          approach_density=2.5):
        """
        三段式轨迹：先在 y 负方向远离 → 再 xy 平移到目标 y → 最后垂直靠近目标。
        和 `_plan_top_down` 类似，但方向从 z 改成了 y。
        """

        # Step 1: y负方向撤退
        y_clear = min(cur_pos[1], tar_pos[1]) - retreat_distance
        y_clear = min(y_clear, -min_clearance)
        p_back_from_cur = [cur_pos[0], y_clear, cur_pos[2]]
        p_behind_target = [tar_pos[0], y_clear, tar_pos[2]]
        p_to_target = tar_pos

        mid_quat = tar_quat

        pos_traj, quat_traj = [], []

        def _append_segment(p0, q0, p1, q1, density=1.0):
            dist = float(np.linalg.norm(np.array(p1) - np.array(p0)))
            n_pos = max(min_steps, int(np.ceil(dist / (base_step / density))))
            seg_pos = interpolate_positions(p0, p1, n_pos)
            seg_quat = interpolate_quaternions(q0, q1, n_pos)
            pos_traj.extend(seg_pos)
            quat_traj.extend(seg_quat)

        # a) Y- 方向拉开
        _append_segment(cur_pos, cur_quat, p_back_from_cur, mid_quat, density=1.0)
        # b) XY 水平移到目标位置的后方
        _append_segment(p_back_from_cur, mid_quat, p_behind_target, mid_quat, density=1.0)
        # c) 向 Y 正方向靠近目标
        _append_segment(p_behind_target, mid_quat, p_to_target, tar_quat, density=approach_density)

        return pos_traj, quat_traj
    
    def set_tar_pand_hand_from_y(self, pos, quat):
        assert len(pos) == 3 and len(quat) == 4
        self.tar_panda_hand_pos = pos
        self.tar_panda_hand_quat = quat

        pos_traj, quat_traj = self._plan_from_y_negative(
            cur_pos=self.cur_panda_hand_pos,
            cur_quat=self.cur_panda_hand_quat,
            tar_pos=pos,
            tar_quat=quat,
            retreat_distance=0.25,
            min_clearance=0.6
        )
        self.arm_cmd_traj_pos = pos_traj
        self.arm_cmd_traj_quat = quat_traj



    def get_cur_z_rotation(self):
        """
        获取机器人当前绕 Z 轴的旋转角度（以弧度为单位）。
        """
        # 使用当前四元数创建旋转对象
        quat = [self.cur_chassis_quat[1], self.cur_chassis_quat[2], self.cur_chassis_quat[3], self.cur_chassis_quat[0]]  # 转换顺序为 [qx, qy, qz, qw]
        rotation = R.from_quat(quat)
        # 获取绕 Z 轴的欧拉角（即 yaw 角）
        yaw = rotation.as_euler('zyx', degrees=False)[0]*180/np.pi  # 返回的是绕 Z 轴的旋转角度（单位：弧度）
        # print(f"当前绕 Z 轴旋转角度: {yaw} radians")
        return yaw

    def get_tar_z_rotation(self):
        """
        获取机器人当前绕 Z 轴的旋转角度（以弧度为单位）。
        """
        # 使用当前四元数创建旋转对象
        quat = [self.tar_chassis_quat[1], self.tar_chassis_quat[2], self.tar_chassis_quat[3], self.tar_chassis_quat[0]]  # 转换顺序为 [qx, qy, qz, qw]
        rotation = R.from_quat(quat)
        # 获取绕 Z 轴的欧拉角（即 yaw 角）
        yaw = rotation.as_euler('zyx', degrees=False)[0]*180/np.pi  # 返回的是绕 Z 轴的旋转角度（单位：弧度）
        # print(f"当前绕 Z 轴旋转角度: {yaw} radians")
        return yaw
    
    def get_all_joint_positions(self):
        """
        打印 Panda 机械臂的关节名称和角度（按顺序），数组形式输出
        """
        joint_positions = self.robot.data.joint_pos[0].cpu().numpy()
        joint_names = self.robot.data.joint_names

        name_to_value = {name: pos for name, pos in zip(joint_names, joint_positions)}

        # 获取 Panda 的关节角度（按顺序）
        ordered_positions = [name_to_value.get(name, float('nan')) for name in self.panda_joint_name]

        # 打印关节名
        print("Panda Joint Names:")
        print(", ".join(self.panda_joint_name))

        # 打印角度数组，带逗号分隔，带括号
        pos_str = ", ".join([f"{x:.6f}" for x in ordered_positions])
        print("Panda Joint Positions (radians):")
        print(f"[{pos_str}]")

    def set_tar_z_rotation(self, angle_z):
        """
        设置机器人绕 Z 轴的目标旋转角度。
        
        参数:
            angle_z: 绕 Z 轴的旋转角度（单位：弧度）
        """
        # 创建一个绕 Z 轴旋转的四元数
        angle_z=angle_z*np.pi/180
        rotation = R.from_euler('z', angle_z)  # 'z' 表示绕 Z 轴旋转
        quat = rotation.as_quat()  # 获取对应的四元数
        self.tar_chassis_quat = [quat[3], quat[0], quat[1], quat[2]]  # 顺序转换为 [qw, qx, qy, qz]
        print(f"目标 Z 轴旋转角度设为 {angle_z} radians")

              
    def update_arm_cmd_traj(self):
        pos_error = np.linalg.norm(np.array(self.tar_panda_hand_pos) - np.array(self.cur_panda_hand_pos))
        if pos_error > 0.01:
            print('pos_error:',self.tar_panda_hand_pos,self.cur_panda_hand_pos)
            N_interp = max(1, int(np.ceil(pos_error / 0.05)))
            self.arm_cmd_traj_pos = interpolate_positions(self.cur_panda_hand_pos, self.tar_panda_hand_pos, N_interp)

        dot_val = np.dot(self.cur_panda_hand_quat, self.tar_panda_hand_quat)
        if abs(1 - abs(dot_val)) > 0.01:
            print('angle_error:',self.tar_panda_hand_quat, self.cur_panda_hand_quat)
            N_interp_angle = max(1, int(np.ceil((1-abs(dot_val)) / 0.03)))
            self.arm_cmd_traj_quat = interpolate_quaternions(self.cur_panda_hand_quat, self.tar_panda_hand_quat, N_interp_angle)

    def _plan_top_down(self, cur_pos, cur_quat, tar_pos, tar_quat,
                    hover_clearance=0.12,     # 相对抬升量
                    min_hover_z=0.50,         # 最低悬停高度
                    base_step=0.02,           # 基础步长（米），越小越细
                    min_steps=3,              # 每段最少插值点
                    descent_density=2.5       # 下降段加密倍数（>1 更密）
                    ):
        """
        三段式轨迹：抬高(Z+) -> 水平到位(XY) -> 垂直下降(Z-)
        - base_step：基础步长，默认 2cm
        - descent_density：仅对下降段生效的加密倍数
        """
        # 1) 计算悬停高度
        hover_z = max(cur_pos[2], tar_pos[2]) + hover_clearance
        hover_z = max(hover_z, min_hover_z)

        # 2) 三个关键位姿
        p_up_from_cur  = [cur_pos[0], cur_pos[1], hover_z]
        p_above_target = [tar_pos[0], tar_pos[1], hover_z]
        p_down_to_tar  = tar_pos

        mid_quat = tar_quat  # 也可换成 cur_quat，看你的抓取策略

        pos_traj, quat_traj = [], []

        def _append_segment(p0, q0, p1, q1, density=1.0):
            # 距离、步数（下降段乘以 density）
            dist = float(np.linalg.norm(np.array(p1) - np.array(p0)))
            n_pos = max(min_steps, int(np.ceil(dist / (base_step / density))))
            seg_pos  = interpolate_positions(p0, p1, n_pos)
            seg_quat = interpolate_quaternions(q0, q1, n_pos)
            pos_traj.extend(seg_pos)
            quat_traj.extend(seg_quat)

        # a) Z+ 抬高（正常密度）
        _append_segment(cur_pos,      cur_quat, p_up_from_cur,  mid_quat, density=1.0)
        # b) XY 平移（正常密度）
        _append_segment(p_up_from_cur, mid_quat, p_above_target, mid_quat, density=1.0)
        # c) Z- 垂直下降（加密密度）
        _append_segment(p_above_target, mid_quat, p_down_to_tar, tar_quat, density=descent_density)

        return pos_traj, quat_traj


    def get_state_of_prim(self, prim_name):
        try:
            # Check if robot or relevant prim exists
            if self.robot is None or prim_name not in self.robot.data.body_names:
                raise ValueError(f"Prim {prim_name} is invalid or no longer exists.")
            
            index = self.robot.data.body_names.index(prim_name)
            state = self.robot.data.body_state_w[:, index, :]
            return state
        except Exception as e:
            print(f"Error getting state of prim '{prim_name}': {e}")
            return None  # or return default state, depending on your needs
   
    def update_robot_state(self):
        chassis_state=self.get_state_of_prim('chassis_link').cpu().numpy()
        self.cur_chassis_pos = chassis_state[0][:3]
        self.cur_chassis_quat = chassis_state[0][3:7]
        self.cur_panda_hand_pos,self.cur_panda_hand_quat = self.get_pand_hand_state()

    def get_pand_hand_state(self):
        """
        计算 panda_hand 相对于 panda_link0 的相对位姿，并使用 SciPy 标准库进行旋转转换。
        假设 get_state_of_prim(prim_name) 返回形如 [x, y, z, qx, qy, qz, qw] 的张量，
        其中四元数顺序为 [qx, qy, qz, qw]。

        Returns:
            rel_pos: 相对平移向量 [x, y, z]
            rel_quat: 相对旋转的四元数 [qx, qy, qz, qw]
        """
        # 获取 panda_hand 与 panda_link0 的状态（返回 numpy 数组）
        hand_state = self.get_state_of_prim('panda_hand').cpu().numpy()[0]
        base_state = self.get_state_of_prim('panda_link0').cpu().numpy()[0]
        
        # 分别提取位置与四元数
        hand_pos = hand_state[:3]
  
        base_pos = base_state[:3]

        hand_quat_raw = hand_state[3:7]
        # 重排为 [qx, qy, qz, qw]
        hand_quat = [hand_quat_raw[1], hand_quat_raw[2], hand_quat_raw[3], hand_quat_raw[0]]# 顺序： [qx, qy, qz, qw]

        base_quat_raw = base_state[3:7]
        base_quat = [base_quat_raw[1], base_quat_raw[2], base_quat_raw[3], base_quat_raw[0]]   # 顺序： [qx, qy, qz, qw]
        # 使用 SciPy 将四元数转换为旋转矩阵
        R_hand = R.from_quat(hand_quat).as_matrix()  # 3x3 旋转矩阵
        R_base = R.from_quat(base_quat).as_matrix()
        
        # 构造齐次变换矩阵
        T_hand = np.eye(4)
        T_hand[:3, :3] = R_hand
        T_hand[:3, 3] = hand_pos

        T_base = np.eye(4)
        T_base[:3, :3] = R_base
        T_base[:3, 3] = base_pos

        # 计算相对变换：T_rel = inv(T_base) * T_hand
        T_rel = np.linalg.inv(T_base) @ T_hand
        rel_pos = T_rel[:3, 3]
        R_rel = T_rel[:3, :3]
        
        # 将相对旋转矩阵转换为四元数
        rel_quat = R.from_matrix(R_rel).as_quat()  # 返回 [qx, qy, qz, qw]
        
        return rel_pos, rel_quat
    
    def get_pand_hand_world_pose(self):
        """
        获取 panda_hand 在世界坐标系（world frame）下的位姿。

        Returns:
            world_pos : [x, y, z]
            world_quat: [qw,qx, qy, qz]
        """
        # 获取 panda_hand 在世界系下的状态（形如 [x, y, z, ?, ?, ?, ?]）
        hand_state = self.get_state_of_prim('panda_hand').cpu().numpy()[0]

        # 位置（世界系）
        world_pos = hand_state[:3]

        quat_raw = hand_state[3:7]

        return world_pos.tolist() , quat_raw.tolist()     
    
    def world_pose_to_link0(self, world_pos, world_quat):
        """
        将任意世界坐标系下的位置和姿态转换为 panda_link0 坐标系下的相对位姿。
        
        参数：
            world_pos: [x, y, z] 世界坐标系下的位置
            world_quat: [qx, qy, qz, qw] 世界坐标系下的姿态（scipy 顺序）
            !!!!!!!!!!!!!!特别注意输入的顺序和输出的顺序不同
            
        返回：
            rel_pos: 相对于 panda_link0 的位置
            rel_quat: 相对于 panda_link0 的四元数（[w, x, y, z]）
        """
        # 获取 panda_link0 的世界状态
        base_state = self.get_state_of_prim('panda_link0').cpu().numpy()[0]
        base_pos = base_state[:3]
        base_quat_raw = base_state[3:7]
        base_quat = [base_quat_raw[1], base_quat_raw[2], base_quat_raw[3], base_quat_raw[0]]

        # 构造目标和 base 的齐次矩阵
        T_world = np.eye(4)
        T_world[:3, :3] = R.from_quat(world_quat).as_matrix()
        T_world[:3, 3] = world_pos

        T_base = np.eye(4)
        T_base[:3, :3] = R.from_quat(base_quat).as_matrix()
        T_base[:3, 3] = base_pos

        # 执行坐标转换
        T_rel = np.linalg.inv(T_base) @ T_world
        rel_pos = T_rel[:3, 3]
        rel_quat = R.from_matrix(T_rel[:3, :3]).as_quat()

        rel_quat_wxyz = [rel_quat[3], rel_quat[0], rel_quat[1], rel_quat[2]]

        return rel_pos.tolist(), rel_quat_wxyz

    
    def get_relative_transform(self, from_link: str, to_link: str):
        """
        获取 `to_link` 相对于 `from_link` 的相对变换（即 TF 中的 child 相对 parent 变换）

        参数：
            from_link: 父链接名（如 'chassis_link'）
            to_link: 子链接名（如 'panda_link0/Lidar'）

        返回：
            rel_pos: 相对位置 [x, y, z]
            rel_quat: 相对旋转四元数 [x, y, z, w] （符合 ROS 用法）
        """
        from_state = self.get_state_of_prim(from_link)
        to_state = self.get_state_of_prim(to_link)

        if from_state is None or to_state is None:
            raise ValueError(f"无法获取 {from_link} 或 {to_link} 的状态")

        from_state = from_state.cpu().numpy()[0]
        to_state = to_state.cpu().numpy()[0]

        from_pos = from_state[:3]
        to_pos = to_state[:3]

        from_quat_raw = from_state[3:7]
        to_quat_raw = to_state[3:7]

        from_quat = [from_quat_raw[1], from_quat_raw[2], from_quat_raw[3], from_quat_raw[0]]  # [x, y, z, w]
        to_quat = [to_quat_raw[1], to_quat_raw[2], to_quat_raw[3], to_quat_raw[0]]

        # 构造齐次变换矩阵
        T_from = np.eye(4)
        T_from[:3, :3] = R.from_quat(from_quat).as_matrix()
        T_from[:3, 3] = from_pos

        T_to = np.eye(4)
        T_to[:3, :3] = R.from_quat(to_quat).as_matrix()
        T_to[:3, 3] = to_pos

        # 相对变换
        T_rel = np.linalg.inv(T_from) @ T_to

        rel_pos = T_rel[:3, 3]
        rel_quat = R.from_matrix(T_rel[:3, :3]).as_quat()  # [x, y, z, w]

        return rel_pos.tolist(), rel_quat.tolist()

    
    def get_nova_world_pose(self):
    # 世界坐标系
        index = self.robot.data.body_names.index("chassis_link")
        world_pose = self.robot.data.body_state_w[:, index, :].cpu().numpy()[0]
        # print("world_pose is")
        # print(world_pose)
        
        pos = world_pose[:3].tolist()
        quat_isaac = world_pose[3:7].tolist()
        quat_ros = [quat_isaac[1], quat_isaac[2], quat_isaac[3], quat_isaac[0]]
        return pos, quat_ros

    ###############################################
    ###############################################
    ###############################################
    # 5) 内部工具：统一解析对象
    def _resolve_object(self, key):
        if isinstance(key, str):
            if key not in self.objects:
                raise KeyError(f"Object '{key}' not found in controller.objects")
            return self.objects[key], key
        if hasattr(key, "write_root_state_to_sim"):
            # 用对象本身当 key
            return key, "<unnamed>"
        raise TypeError("key must be object name (str) or RigidObject instance")


    # 6) 核心：写入物体状态（无触发器；按每个物体的 attract 开关决定吸不吸）
    def write_object_state(self, key: str, offset_ee=None, alpha=None, align_orientation=None):
        """
        无触发器版本：
        - 若该物体开启吸附：将其位置指数逼近到 末端坐标系下 offset_ee 的位置（对齐姿态可选）
        - 若关闭吸附：不改变状态（把当前状态写回，保持原位；你也可以改成直接 return）
        """
        obj, name = self._resolve_object(key)

        # 拿到该物体的配置（若入参传了就覆盖）
        opts = self.object_opts.get(name, {"attract": False, "offset": (0.00, 0.0, 0.093), "alpha": 0.30, "align_orientation": True})
        use_attract = opts.get("attract", False)
        offset = tuple(offset_ee) if offset_ee is not None else opts.get("offset", (0.00, 0.0, 0.093))
        k_alpha = float(alpha) if alpha is not None else float(opts.get("alpha", 0.30))
        do_align = bool(align_orientation) if align_orientation is not None else bool(opts.get("align_orientation", True))

        # 读取当前状态
        state = obj.data.root_state_w.clone().to(self.robot.device)  # [N, 13]

        if not use_attract:
            # 不吸附：保持现状（写回或直接 return 都行）
            obj.write_root_state_to_sim(state)
            return

        # 末端世界位姿
        ee_pos, ee_quat_wxyz = self.get_pand_hand_world_pose()
        ee_pos = np.asarray(ee_pos, dtype=float)
        ee_quat_xyzw = [ee_quat_wxyz[1], ee_quat_wxyz[2], ee_quat_wxyz[3], ee_quat_wxyz[0]]
        R_ee = R.from_quat(ee_quat_xyzw).as_matrix()

        # 末端坐标系偏移 -> 世界
        world_offset = R_ee @ np.asarray(offset, dtype=float)
        target_pos = ee_pos + world_offset

        # 位置指数逼近
        cur_pos = state[:, :3]
        tgt_pos = torch.tensor(target_pos, device=self.robot.device, dtype=torch.float32).expand_as(cur_pos)
        new_pos = (1.0 - k_alpha) * cur_pos + k_alpha * tgt_pos
        state[:, :3] = new_pos

        # 姿态对齐（可选）
        if do_align:
            wxyz = torch.tensor(ee_quat_wxyz, device=self.robot.device, dtype=torch.float32).expand(state.shape[0], 4)
            state[:, 3:7] = wxyz

        # 保持其余量（速度等）不改动；如果你希望清零速度，取消下一行注释
        state[:, 7:13] = 0.0

        obj.write_root_state_to_sim(state)



    # 2) 绑定/附加物体时，初始化每个物体的选项
    def attach_objects(self, objects: dict):
        """把 spawn_from_yaml 的返回字典挂上来，并为每个物体设置默认吸附参数。"""
        self.objects = objects or {}
        self.object_opts = {
            name: {"attract": False, "offset": (0.00, 0.0, 0.093), "alpha": 0.1, "align_orientation": True}
            for name in self.objects.keys()
        }
        print(f"[Controller] attached {len(self.objects)} objects: {list(self.objects.keys())}")


    def _default_obj_opts(self):
        return {"attract": True, "offset": (0.10, 0.0, 0.093), "alpha": 0.10, "align_orientation": True}

    def set_object_options(
        self,
        name: str,
        *,
        attract: bool | None = None,
        offset_ee: tuple[float, float, float] | list[float] | None = None,
        alpha: float | None = None,
        align_orientation: bool | None = None,
        reset: bool = False,
    ):
        """
        统一设置单个物体的吸附参数。
        - attract: 是否吸附到末端附近
        - offset_ee: 末端坐标系下的偏移 (x,y,z)
        - alpha: 位置指数逼近系数 (0~1, 越大越快)
        - align_orientation: 是否将物体姿态对齐到末端
        - reset: True 则先重置为默认，再应用本次入参
        返回：更新后的配置 dict
        """
        if name not in self.objects:
            raise KeyError(f"Object '{name}' not found in controller.objects")

        if name not in self.object_opts or reset:
            self.object_opts[name] = self._default_obj_opts()

        opts = self.object_opts[name]
        if attract is not None:
            opts["attract"] = bool(attract)
        if offset_ee is not None:
            if len(offset_ee) != 3:
                raise ValueError("offset_ee must be length-3 (x,y,z) in EE frame")
            opts["offset"] = tuple(float(x) for x in offset_ee)
        if alpha is not None:
            opts["alpha"] = float(alpha)
        if align_orientation is not None:
            opts["align_orientation"] = bool(align_orientation)

        self.object_opts[name] = opts
        print(f"[Controller] opts for '{name}': {opts}")
        return opts

    def _ee_over_object(self, obj,
                        xy_radius: float = 0.10,
                        z_range=(0.05, 0.25),
                        use_ee_axis: bool = False) -> bool:
        """
        触发器：末端在物体“上方一定距离”。
        - xy_radius: 末端与物体在水平面的距离阈值
        - z_range:   末端相对物体的高度差 [z_min, z_max]（世界系）
        - use_ee_axis=True 时，使用末端的-Z轴方向判断“上方”，否则用世界Z
        """
        # 末端位姿
        ee_pos, ee_quat_wxyz = self.get_pand_hand_world_pose()
        ee_pos = np.asarray(ee_pos, dtype=float)

        # 物体当前根状态
        state = obj.data.root_state_w  # [N, 13]
        if state.ndim == 2:
            obj_pos = state[0, :3].detach().cpu().numpy()
        else:
            # 保险：取第0个
            obj_pos = state[:3].detach().cpu().numpy()
        obj_pos = np.asarray(obj_pos, dtype=float)

        # 水平误差
        dx, dy = ee_pos[0] - obj_pos[0], ee_pos[1] - obj_pos[1]
        dist_xy = float(np.hypot(dx, dy))

        # 高度差
        dz_world = float(ee_pos[2] - obj_pos[2])

        if use_ee_axis:
            # 用末端坐标系的 -Z 方向来判断“上方”投影距离
            ee_quat_xyzw = [ee_quat_wxyz[1], ee_quat_wxyz[2], ee_quat_wxyz[3], ee_quat_wxyz[0]]
            R_ee = R.from_quat(ee_quat_xyzw).as_matrix()
            z_axis_world = -R_ee[:, 2]  # 末端指向物体的“下”方向
            vec_obj_to_ee = ee_pos - obj_pos
            height_along_ee_z = float(np.dot(vec_obj_to_ee, z_axis_world))
            # 去掉沿 z 的分量，得到“水平”误差
            proj = height_along_ee_z * z_axis_world
            horiz_vec = vec_obj_to_ee - proj
            dist_xy = float(np.linalg.norm(horiz_vec))
            dz_world = height_along_ee_z  # 触发高度改用末端-Z轴方向

        z_min, z_max = z_range
        return (dist_xy <= xy_radius) and (z_min <= dz_world <= z_max)
