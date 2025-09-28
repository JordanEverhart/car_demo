import time
import numpy as np
import torch
import threading
import tkinter as tk
from tkinter import ttk
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"renderer": "RayTracedLighting", "headless": False, "physics_enabled": True})
print("Simulation started")

from nova_franka_controller import NovaFranka_Controller, add_robot
import isaaclab.sim as sim_utils

# 初始化 Isaac Sim
# file_path = "/home/rts001/demo/real_scene_phy.usd"
# from omni.isaac.core.utils.stage import open_stage
# open_stage(usd_path=file_path)

# 加载机器人
robot = add_robot(position=(0.41, -0.8, 0.0), orientation=(0, 0, 0, 1), prim_name="/Robot_1")
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, use_fabric=True, gravity=(0.0, 0.0, -9)))
sim.set_camera_view([15, -4, 10], [0.0, 0.0, 0.0])
sim.reset()
sim_dt = sim.get_physics_dt()

# 创建控制器实例
controller = NovaFranka_Controller(robot, sim_dt)

# 初始化目标状态
init_panda_hand_pos = [0.3, 0, 0.5]
init_panda_hand_quat = [0, 1, 0, 0]
controller.set_tar_pand_hand(init_panda_hand_pos, init_panda_hand_quat)

# ----------------------------
# 仿真循环（主线程）
# ----------------------------
def simulation_loop():
    while simulation_app.is_running():
        sim.step(render=True)
        controller.step()
        sim.forward()
        time.sleep(0.03)

# ----------------------------
# GUI 窗口部分（Tkinter）
# ----------------------------
def run_gui():
    root = tk.Tk()
    root.title("NovaFranka Controller Demo")

    # 当前状态显示区域
    current_state_var = tk.StringVar()
    current_label = ttk.Label(root, textvariable=current_state_var, font=("Arial", 12))
    current_label.pack(padx=10, pady=5)

    # 目标状态显示区域
    target_state_var = tk.StringVar()
    target_label = ttk.Label(root, textvariable=target_state_var, font=("Arial", 12))
    target_label.pack(padx=10, pady=5)

    # 底盘目标状态输入区域
    chassis_pos_label = ttk.Label(root, text="目标底盘位置 (x,y,z):", font=("Arial", 12))
    chassis_pos_label.pack(padx=10, pady=2)

    # Format values to 4 decimal places
    chassis_pos_entry = ttk.Entry(root, width=30)
    chassis_pos_entry.insert(0, ",".join(f"{x:.4f}" for x in controller.cur_chassis_pos))
    chassis_pos_entry.pack(padx=10, pady=2)

    chassis_quat_label = ttk.Label(root, text="目标底盘四元数 (qw,qx,qy,qz):", font=("Arial", 12))
    chassis_quat_label.pack(padx=10, pady=2)

    # Format values to 4 decimal places
    chassis_quat_entry = ttk.Entry(root, width=30)
    chassis_quat_entry.insert(0, ",".join(f"{x:.4f}" for x in controller.cur_chassis_quat))
    chassis_quat_entry.pack(padx=10, pady=2)

    # 当前底盘角度显示区域
    chassis_angle_label = ttk.Label(root, text="当前底盘角度 (yaw):", font=("Arial", 12))
    chassis_angle_label.pack(padx=10, pady=2)

    current_angle_var = tk.StringVar()
    chassis_angle_display = ttk.Label(root, textvariable=current_angle_var, font=("Arial", 12))
    current_angle_var.set(f"{controller.get_cur_z_rotation():.4f}")
    chassis_angle_display.pack(padx=10, pady=2)

    # 目标底盘角度输入区域
    target_angle_label = ttk.Label(root, text="目标底盘角度 (yaw):", font=("Arial", 12))
    target_angle_label.pack(padx=10, pady=2)
    
    target_angle_entry = ttk.Entry(root, width=30)
    target_angle_entry.insert(0, f"{controller.tar_chassis_quat[0]:.4f}")
    target_angle_entry.pack(padx=10, pady=2)

    # 机械臂末端目标状态输入区域
    panda_pos_label = ttk.Label(root, text="目标机械臂末端位置 (x,y,z):", font=("Arial", 12))
    panda_pos_label.pack(padx=10, pady=2)

    # Format values to 4 decimal places
    panda_pos_entry = ttk.Entry(root, width=30)
    panda_pos_entry.insert(0, ",".join(f"{x:.4f}" for x in controller.cur_panda_hand_pos))
    panda_pos_entry.pack(padx=10, pady=2)

    panda_quat_label = ttk.Label(root, text="目标机械臂末端四元数 (qw, qx, qy, qz):", font=("Arial", 12))
    panda_quat_label.pack(padx=10, pady=2)

    # Format values to 4 decimal places
    panda_quat_entry = ttk.Entry(root, width=30)
    panda_quat_entry.insert(0, ",".join(f"{x:.4f}" for x in controller.cur_panda_hand_quat))
    panda_quat_entry.pack(padx=10, pady=2)

    def set_target():
        try:
            # 读取底盘目标状态
            pos_str = chassis_pos_entry.get()
            quat_str = chassis_quat_entry.get()
            pos = [float(x.strip()) for x in pos_str.split(",")]
            quat = [float(x.strip()) for x in quat_str.split(",")]
            # controller.set_tar_nova(pos, quat)
            controller.set_tar_nova_pos(pos)
            # 读取目标底盘角度
            target_angle = float(target_angle_entry.get())
            controller.set_tar_z_rotation(target_angle)  # Set the target rotation angle
            
            # 读取机械臂末端目标状态
            panda_pos_str = panda_pos_entry.get()
            panda_quat_str = panda_quat_entry.get()
            panda_pos = [float(x.strip()) for x in panda_pos_str.split(",")]
            panda_quat = [float(x.strip()) for x in panda_quat_str.split(",")]
            controller.set_tar_pand_hand(panda_pos, panda_quat)
            
            print("目标状态已更新！")
        except Exception as e:
            print("输入错误：", e)
    set_button = ttk.Button(root, text="Set Target", command=set_target)
    set_button.pack(padx=10, pady=10)     
    def update_gui():
        # 更新当前状态显示（需要调用控制器更新状态）
        controller.update_robot_state()
        cur_chassis = controller.cur_chassis_pos
        cur_chassis_quat = controller.cur_chassis_quat
        cur_panda, cur_panda_quat = controller.cur_panda_hand_pos, controller.cur_panda_hand_quat
        current_state_var.set(f"当前底盘位置: {cur_chassis}\n当前底盘四元数: {cur_chassis_quat}\n当前底盘角度:{controller.get_cur_z_rotation()}" +
                              f"当前机械臂末端位置: {cur_panda}\n当前机械臂末端四元数: {cur_panda_quat}")
        
        tar_chassis = controller.tar_chassis_pos
        tar_chassis_quat = controller.tar_chassis_quat
        tar_panda, tar_panda_quat = controller.tar_panda_hand_pos, controller.tar_panda_hand_quat
        target_state_var.set(f"目标底盘位置: {tar_chassis}\n目标底盘四元数: {tar_chassis_quat}\n目标底盘角度:{controller.get_tar_z_rotation()}" +
                             f"目标机械臂末端位置: {tar_panda}\n目标机械臂末端四元数: {tar_panda_quat}")
        root.after(200, update_gui)

    root.after(200, update_gui)
    root.mainloop()

# 启动 GUI 线程
gui_thread = threading.Thread(target=run_gui, daemon=True)
gui_thread.start()

# 运行仿真循环
simulation_loop()

simulation_app.close()
