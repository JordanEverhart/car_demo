"""
ros_callbacks.py
----------------
封装回调函数注册，不修改任何逻辑、语句、变量名，保持 demo.py 中原始风格一致。

用法：
    from ros_callbacks import setup_ros_subscribers
    setup_ros_subscribers(controller, test_pos, init_panda_hand_quat)
"""
import rospy
from std_msgs.msg import String, Float32,Float32MultiArray
from geometry_msgs.msg import Twist, Pose, PoseStamped
import torch
import time
import json  # ← 新增：用于解析 /set_object_options 的 JSON 负载
import math
import numpy as np

# ========= 工具：四元数(xyzw) → yaw，角度归一化 =========
def _quat_to_yaw(quat_xyzw):
    x, y, z, w = quat_xyzw
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)

def _angle_normalize(a):
    return math.atan2(math.sin(a), math.cos(a))

# ========= 阻塞式直线运动（基于里程） =========
def _move_linear(controller, distance_m, speed_mps=0.25, rate_hz=20):
    """
    distance_m: 期望前进(+)/后退(-)的路程（米）
    speed_mps : 线速度幅值（米/秒），自动按 distance_m 方向加符号
    """
    if speed_mps <= 0:
        speed_mps = 0.25
    sign = 1.0 if distance_m >= 0.0 else -1.0
    v_cmd = sign * speed_mps

    start_xy = np.array(controller.get_nova_world_pose()[0][:2], dtype=float)
    controller.base_controller(v_cmd, 0.0)

    rate = rospy.Rate(rate_hz)
    # 给个兜底超时（3 倍理论时间 + 3s）
    timeout = abs(distance_m) / speed_mps * 3.0 + 3.0
    t0 = rospy.Time.now().to_sec()

    while not rospy.is_shutdown():
        cur_xy = np.array(controller.get_nova_world_pose()[0][:2], dtype=float)
        moved = float(np.linalg.norm(cur_xy - start_xy))

        if moved >= abs(distance_m):   # ≥ 阈值就停
            break

        if rospy.Time.now().to_sec() - t0 > timeout:
            rospy.logwarn("move_linear 超时，强制停止。")
            break

        rate.sleep()

    controller.base_controller(0.0, 0.0)

# ========= 阻塞式原地旋转（基于累计角度） =========
def _rotate(controller, target_rad, angular_speed_rps=0.4, rate_hz=20):
    """
    target_rad: 目标旋转角度（弧度），左转 +，右转 -
    angular_speed_rps: 角速度幅值（弧度/秒）
    """
    if angular_speed_rps <= 0:
        angular_speed_rps = 0.4
    sign = 1.0 if target_rad >= 0.0 else -1.0
    w_cmd = sign * angular_speed_rps

    # 累计方式处理跨 ±pi 的情况
    _, quat0 = controller.get_nova_world_pose()
    yaw_prev = _quat_to_yaw(quat0)
    yaw_accum = 0.0

    controller.base_controller(0.0, w_cmd)

    rate = rospy.Rate(rate_hz)
    timeout = abs(target_rad) / angular_speed_rps * 3.0 + 3.0
    t0 = rospy.Time.now().to_sec()

    while not rospy.is_shutdown():
        _, quat = controller.get_nova_world_pose()
        yaw = _quat_to_yaw(quat)
        dy = _angle_normalize(yaw - yaw_prev)
        yaw_accum += dy
        yaw_prev = yaw

        if abs(yaw_accum) >= abs(target_rad):  # ≥ 阈值就停
            break

        if rospy.Time.now().to_sec() - t0 > timeout:
            rospy.logwarn("rotate 超时，强制停止。")
            break

        rate.sleep()

    controller.base_controller(0.0, 0.0)

def setup_ros_subscribers(controller, test_pos, init_panda_hand_quat):
    rospy.loginfo("✅ 正在注册 ROS 订阅回调")

    # Callback function for set_finger_width (new topic)
    def set_finger_width_callback(msg):
        rospy.loginfo(f"收到夹爪宽度指令: {msg.data} 米")
        controller.set_finger_width(msg.data)  # 调用 set_finger_width 函数

    # Callback function for set_tar_pand_hand (new topic)
    def set_tar_pand_hand_callback(msg):
        rospy.loginfo(f"收到目标位置指令: position = {msg.position}, orientation = {msg.orientation}")
        pos = [msg.position.x, msg.position.y, msg.position.z]
        quat = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        controller.set_tar_pand_hand(pos, quat)  # 调用 set_tar_pand_hand 函数

    def set_tar_pand_hand_world_callback(msg):
        rospy.loginfo(f"收到世界系末端位姿指令: position = {msg.position}, orientation = {msg.orientation}")
        
        # 世界系输入（Pose 的四元数为 xyzw）
        world_pos = [msg.position.x, msg.position.y, msg.position.z]
        
        # 判断 orientation 是否为空（全部为 0，因为None不能通过rostopic传递）
        if (msg.orientation.x == 0 and msg.orientation.y == 0 and 
            msg.orientation.z == 0 and msg.orientation.w == 0):
            # 获取当前末端位姿
            cur_pos, cur_quat_xyzw = controller.get_pand_hand_world_pose()
            world_quat_xyzw = cur_quat_xyzw
            rospy.loginfo("未提供 orientation，保持当前姿态")
        else:
            world_quat_xyzw = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]

        # 转到 link0 坐标系（注意：返回四元数为 wxyz）
        rel_pos, rel_quat_wxyz = controller.world_pose_to_link0(world_pos, world_quat_xyzw)

        # 下发到控制器
        # controller.set_tar_pand_hand_from_top(rel_pos, rel_quat_wxyz)
        controller.set_tar_pand_hand(rel_pos, rel_quat_wxyz)
        # controller.set_tar_pand_hand_from_y(rel_pos, rel_quat_wxyz)
        rospy.loginfo(f"已发布 link0 相对目标 → pos={rel_pos}, quat(wxyz)={rel_quat_wxyz}")

    def pick_joint_callback(msg):
        data = msg.data  # List[float]
        if len(data) != 7:
            rospy.logwarn(f"收到的关节数组长度不为 7，忽略。收到长度为 {len(data)}")
            return

        rospy.loginfo(f"收到 /pick 指令，目标关节角度: {data}")
        target_joint_positions = torch.tensor([data], device=controller.robot.device, dtype=torch.float32)
        controller.robot.set_joint_position_target(target_joint_positions, joint_ids=controller.panda_joint_ids)


    def command_callback(msg):
        rospy.loginfo("收到发布")
        if msg.data == "forward":
            rospy.loginfo("收到前进指令")
            controller.base_controller(1,0)
            time.sleep(17)
            controller.base_controller(0,0)
        elif msg.data == "spin":
            rospy.loginfo("收到旋转指令")
            controller.base_controller(0,20)
            time.sleep(17)
        elif msg.data == "open":
            rospy.loginfo("收到打开夹爪的指令")
            controller.set_finger_width(0.08)
        elif msg.data == "close":
            rospy.loginfo("收到关闭夹爪的指令")
            controller.set_finger_width(0.01)
        elif msg.data == "pick":
            rospy.loginfo("收到移动机械臂的指令")
            controller.set_tar_pand_hand(test_pos,init_panda_hand_quat)
        elif msg.data == "fix":
            rospy.loginfo("收到吸附物体的指令")
        elif msg.data == "stop":
            rospy.loginfo("收到暂停指令")
            controller.base_controller(0,0)
        elif msg.data == "goto":
            rospy.loginfo("收到导航指令")
            init_nova_pos = [0.41, -0.8, 0.0]
            init_nova_quant = [1,0,0,0]
            controller.set_tar_nova(init_nova_pos, init_nova_quant)
            controller.base_controller(0,0)

    def cmd_vel_callback(msg):
        linear = msg.linear.x
        angular = msg.angular.z
        rospy.loginfo("cmd_vel:linear=%.2f, angular=%.2f", linear, angular)
        controller.base_controller(linear, angular)    

    def keyboard_callback(msg):
        linear = msg.linear.x
        angular = msg.angular.z
        rospy.loginfo(f"接收到键盘控制:linear={linear:.2f}, angular={angular:.2f}")
        controller.base_controller(linear, angular)

    def set_object_options_callback(msg: String):
        """
        期待 JSON 字符串，例如：
        {"name":"SpawnedCube_0","attract":true,"offset":[0.06,0.0,0.03],"alpha":0.5,"align_orientation":false,"reset":false}
        必填字段：name
        """
        try:
            cfg = json.loads(msg.data)
            if not isinstance(cfg, dict) or "name" not in cfg:
                rospy.logwarn("set_object_options 需要 JSON 且必须包含 'name' 字段")
                return

            name = cfg["name"]
            kwargs = {}

            if "attract" in cfg:
                kwargs["attract"] = bool(cfg["attract"])

            if "offset" in cfg:
                off = cfg["offset"]
                # 支持 [x,y,z] 或 {"x":...,"y":...,"z":...}
                if isinstance(off, dict):
                    off = [off.get("x", 0.0), off.get("y", 0.0), off.get("z", 0.0)]
                if not (isinstance(off, (list, tuple)) and len(off) == 3):
                    rospy.logwarn("offset 必须是长度为 3 的列表或 {x,y,z} 字典，已忽略该字段")
                else:
                    kwargs["offset_ee"] = [float(off[0]), float(off[1]), float(off[2])]

            if "alpha" in cfg:
                kwargs["alpha"] = float(cfg["alpha"])

            if "align_orientation" in cfg:
                kwargs["align_orientation"] = bool(cfg["align_orientation"])

            if "reset" in cfg:
                kwargs["reset"] = bool(cfg["reset"])

            controller.set_object_options(name, **kwargs)
            rospy.loginfo(f"set_object_options 应用于 '{name}': {kwargs}")
        except Exception as e:
            rospy.logerr(f"set_object_options 解析或应用失败: {e}")

    def move_forward_cb(msg: Float32MultiArray):
        # data = [distance_m, optional speed_mps]
        if not msg.data:
            rospy.logwarn("/move_forward 需要至少 1 个数：距离(米)")
            return
        distance = float(msg.data[0])
        speed = float(msg.data[1]) if len(msg.data) > 1 else 0.25
        rospy.loginfo("→ 前进: 距离=%.3f m, 速度=%.3f m/s", distance, speed)
        _move_linear(controller, +abs(distance), abs(speed))

    def move_backward_cb(msg: Float32MultiArray):
        # data = [distance_m, optional speed_mps]
        if not msg.data:
            rospy.logwarn("/move_backward 需要至少 1 个数：距离(米)")
            return
        distance = float(msg.data[0])
        speed = float(msg.data[1]) if len(msg.data) > 1 else 0.25
        rospy.loginfo("→ 后退: 距离=%.3f m, 速度=%.3f m/s", distance, speed)
        _move_linear(controller, -abs(distance), abs(speed))

    def rotate_left_cb(msg: Float32MultiArray):
        # data = [deg, optional deg_per_sec]
        if not msg.data:
            rospy.logwarn("/rotate_left 需要至少 1 个数：角度(度)")
            return
        deg = float(msg.data[0])
        dps = float(msg.data[1]) if len(msg.data) > 1 else 25.0  # 默认 25°/s
        rospy.loginfo("→ 左转: 角度=%.2f°, 角速度=%.2f°/s", deg, dps)
        _rotate(controller, math.radians(+abs(deg)), math.radians(abs(dps)))

    def rotate_right_cb(msg: Float32MultiArray):
        # data = [deg, optional deg_per_sec]
        if not msg.data:
            rospy.logwarn("/rotate_right 需要至少 1 个数：角度(度)")
            return
        deg = float(msg.data[0])
        dps = float(msg.data[1]) if len(msg.data) > 1 else 25.0
        rospy.loginfo("→ 右转: 角度=%.2f°, 角速度=%.2f°/s", deg, dps)
        _rotate(controller, math.radians(-abs(deg)), math.radians(abs(dps)))

    def goto_callback(msg):
        import math
        import torch

        rospy.loginfo(f"收到世界系末端位姿指令: position = {msg.position}, orientation = {msg.orientation}")

        # 世界系输入（ROS Pose 的四元数为 xyzw）
        world_pos = [msg.position.x, msg.position.y, msg.position.z]

        # 若 orientation 全为 0，则保持当前姿态
        if (msg.orientation.x == 0 and msg.orientation.y == 0 and
            msg.orientation.z == 0 and msg.orientation.w == 0):
            cur_pos, cur_quat_xyzw = controller.get_pand_hand_world_pose()
            world_quat_xyzw = cur_quat_xyzw
            rospy.loginfo("未提供 orientation，保持当前姿态")
        else:
            world_quat_xyzw = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]

        # xyzw -> wxyz（API 需要）
        w, x, y, z = world_quat_xyzw[3], world_quat_xyzw[0], world_quat_xyzw[1], world_quat_xyzw[2]

        # 归一化（避免非单位四元数导致的数值问题）
        norm = math.sqrt(w*w + x*x + y*y + z*z)
        if norm == 0.0:
            rospy.logwarn("收到零四元数，回退为单位姿态")
            w, x, y, z = 1.0, 0.0, 0.0, 0.0
        else:
            inv = 1.0 / norm
            w, x, y, z = w*inv, x*inv, y*inv, z*inv

        # 构造 root_pose 张量：[pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
        # 放到和机器人一致的 device 上（若不可得则回退到本地可用 device）
        device = controller.robot.device
        root_pose = torch.tensor([[world_pos[0], world_pos[1], world_pos[2], w, x, y, z]],
                                dtype=torch.float32, device=device)

        # 下发到仿真（默认只作用于 env 0）
        try:
            controller.robot.write_root_pose_to_sim(root_pose)
            rospy.loginfo(f"已写入 root pose 到仿真 (env_ids=[0]): pos={world_pos}, quat(wxyz)={[w, x, y, z]}")
        except Exception as e:
            rospy.logerr(f"写入 root pose 失败: {e}")

        


    rospy.Subscriber('/robot_control', String, command_callback)
    rospy.Subscriber('/cmd_vel', Twist, cmd_vel_callback)
    rospy.Subscriber('/keyboard_cmd', Twist, keyboard_callback)
    rospy.Subscriber('/set_finger_width', Float32, set_finger_width_callback)  # 订阅夹爪宽度话题
    rospy.Subscriber('/set_target_panda_hand', Pose, set_tar_pand_hand_callback)  # 订阅目标位置和姿态话题
    rospy.Subscriber('/set_target_panda_hand_world', Pose, set_tar_pand_hand_world_callback)  # ★ 新增：世界系目标
    rospy.Subscriber('/pick', Float32MultiArray, pick_joint_callback)
    rospy.Subscriber('/set_object_options', String, set_object_options_callback)  
    rospy.Subscriber('/goto', Pose, goto_callback) # ★ 新增：设置单个物体吸附参数
    # rospy.Subscriber('/move_forward',  Float32MultiArray, move_forward_cb)
    # rospy.Subscriber('/move_backward', Float32MultiArray, move_backward_cb)
    # rospy.Subscriber('/rotate_left',   Float32MultiArray, rotate_left_cb)
    # rospy.Subscriber('/rotate_right',  Float32MultiArray, rotate_right_cb)

    rospy.loginfo("✅ 已注册 ROS 控制指令回调")

