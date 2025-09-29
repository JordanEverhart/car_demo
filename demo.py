# =============================================================================
# 基础库导入
# =============================================================================

# 标准库
import time  # 时间相关操作
import os    # 操作系统接口，用于文件路径操作

# 科学计算库
import numpy as np  # 数值计算和数组操作
import torch        # 深度学习框架，用于张量操作
from scipy.spatial.transform import Rotation as R  # 旋转矩阵和四元数操作

# Isaac Sim 核心库
from omni.isaac.kit import SimulationApp  # Isaac Sim 主应用程序
import isaaclab.sim as sim_utils          # Isaac Lab 仿真工具
from isaacsim.core.utils.prims import get_prim_at_path  # 获取场景中的对象
from isaacsim.core.utils.stage import open_stage        # 打开场景文件
from isaacsim.sensors.physx import RotatingLidarPhysX   # 旋转激光雷达传感器
from isaaclab.assets import RigidObject, RigidObjectCfg # 刚体对象定义

# Isaac Sim 扩展库
from pxr import UsdGeom, Gf, PhysxSchema  # USD 几何和物理相关
import omni.kit.commands                  # Isaac Sim 命令接口

# ROS 相关库
import rospy                              # ROS Python 客户端库
import tf2_ros                           # ROS 坐标变换库
from nav_msgs.msg import Odometry        # 里程计消息类型
from sensor_msgs.msg import PointCloud2, PointField  # 点云消息类型
import sensor_msgs.point_cloud2 as pc2   # 点云数据处理
from std_msgs.msg import String          # 字符串消息类型
from geometry_msgs.msg import Twist      # 速度控制消息类型
import geometry_msgs.msg                 # 几何消息类型
from visualization_msgs.msg import Marker # 可视化标记消息类型

# 自定义模块
from nova_franka_controller import NovaFranka_Controller, add_robot  # 机器人控制器
from object_spawner import spawn_from_yaml                           # 物体生成器
from scan_converter import ScanConverter                             # 点云转激光扫描
from navgoal import send_nav_goal                                    # 导航目标发送
from ros_callbacks import setup_ros_subscribers                      # ROS 回调函数设置
from tf_static_setup import setup_static_tf                          # 静态坐标变换设置
from motion_actions import MotionActionServers                       # 动作服务器
from tf_dynamic_odom import publish_odom_tf_and_marker               # 动态里程计发布
from lidar_publisher import publish_lidar_frame                      # 激光雷达数据发布
from scene_objects_export import publish_top_level_objects_once      # 场景物体信息发布

# =============================================================================
# Isaac Sim 仿真环境初始化
# =============================================================================

# 创建 Isaac Sim 应用程序实例
# 配置参数：
# - renderer: 使用光线追踪渲染器，提供高质量视觉效果
# - headless: 非无头模式，显示图形界面
# - physics_enabled: 启用物理仿真引擎
# - enable_ros_bridge: 启用 ROS 桥接功能，支持与 ROS 系统通信
simulation_app = SimulationApp({
    "renderer": "RayTracedLighting",\
    "headless": False,
    "physics_enabled": True,
    "enable_ros_bridge": True
})

print("Simulation started")

# =============================================================================
# Isaac Sim 扩展与场景加载
# =============================================================================

# 获取 Isaac Sim 应用实例和扩展管理器
kit = omni.kit.app.get_app()
ext_manager = kit.get_extension_manager()

# 启用激光雷达传感器扩展
# 该扩展提供旋转激光雷达传感器的物理仿真功能
ext_manager.set_extension_enabled("omni.isaac.range_sensor", True)

# 定义场景文件路径
# 主场景文件：包含货架环境的物理仿真场景
file_path = "/home/winter/robot/evaluaiton/scenario/shelf/real_scene_phy.usd"

# 备用场景文件路径（注释掉的选项）
# file_path =  "/home/winter/robot/test/Collected_real_scene_phy/real_scene_phy.usd"
 # 事兴场景/home/winter/robot/usd/demo/Collected_real_scene_phy/real_scene_phy.usd
# "/home/winter/robot/Demo/real_scene_phy.usd"只有墙

# 加载指定的 USD 场景文件到 Isaac Sim 舞台
open_stage(usd_path=file_path)

# =============================================================================
# 机器人与传感器加载
# =============================================================================

# 在场景中加载 Nova Franka 机器人
# 机器人包含移动底盘和 7 自由度机械臂
robot = add_robot()

# 定义激光雷达在机器人上的路径
# 激光雷达安装在机械臂基座 (panda_link0) 上
lidar_path = f"/Robot_1/nova_franka/panda_link0/Lidar"

# 验证激光雷达路径是否有效
print(get_prim_at_path(lidar_path).IsValid())

# 创建旋转激光雷达传感器实例
# 使用 PhysX 物理引擎进行激光雷达仿真
lidar = RotatingLidarPhysX(prim_path=lidar_path)

# =============================================================================
# 场景物体生成（必须在 sim.reset() 之前完成）
# =============================================================================

# 构建物体配置文件路径
# objects.yaml 文件定义了场景中需要生成的物体类型、位置和属性
yaml_path = os.path.join(os.path.dirname(file_path), "objects.yaml")

# 检查物体配置文件是否存在
if not os.path.exists(yaml_path):
    raise FileNotFoundError(f"找不到 {yaml_path}：请在 {os.path.dirname(file_path)} 放置 objects.yaml")

# 根据 YAML 配置文件生成场景物体
# 返回物体字典，键为物体名称，值为物体对象
objects = spawn_from_yaml(yaml_path, base_path="/World")

# =============================================================================

# =============================================================================
# 仿真上下文初始化与摄像机设置
# =============================================================================

# 创建仿真上下文实例
# 配置参数：
# - dt: 仿真时间步长 0.01 秒（100Hz）
# - use_fabric: 使用 Fabric 物理引擎，提供高性能物理仿真
# - gravity: 重力加速度 (0, 0, -3) m/s²
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, use_fabric=True, gravity=(0.0, 0.0, -3)))

# 设置摄像机视角
# 参数：[摄像机位置], [目标位置]
# 摄像机位置：(-0.31, -1.17, 1)
# 目标位置：(0.4, -0.4, 0.45)
sim.set_camera_view([-0.31, -1.17, 1], [0.4, -0.4, 0.45])

# 重置仿真状态
sim.reset()

# 执行一次仿真步进（不渲染）
sim.step(render=False)

# =============================================================================
# 激光雷达传感器初始化
# =============================================================================

# 初始化激光雷达传感器
# 注意：通常在 world.reset() 时自动初始化，这里显式调用确保正确初始化
lidar.initialize()

# 将点云数据添加到当前帧
lidar.add_point_cloud_data_to_frame()

# 启用激光雷达可视化
# - draw_points: 显示点云数据点
# - draw_lines: 显示激光射线
lidar.enable_visualization(draw_points=True, draw_lines=True)

# 获取物理仿真时间步长
sim_dt = sim.get_physics_dt()

# =============================================================================
# 机器人控制器初始化
# =============================================================================

# 创建 Nova Franka 机器人控制器实例
# 控制器负责机器人的运动控制、轨迹规划和物体操作
controller = NovaFranka_Controller(robot, sim_dt)

# 测试坐标转换功能
# 将世界坐标系下的位姿转换为机械臂基座坐标系下的相对位姿
test_pos, test_quat = controller.world_pose_to_link0(
    [0.05, 0.01, 0.60],  # 世界坐标系下的位置 (x, y, z)
    [0.706, 0.706, 0, 0]  # 世界坐标系下的四元数 (qx, qy, qz, qw)
)

# =============================================================================
# 机械臂初始位置设置
# =============================================================================

# 设置机械臂末端执行器的初始位置
# 注意：世界坐标系下垂直向下的四元数应该是 [0, 1, 0, 0]
init_panda_hand_pos = [0.3, 0, 0.5]  # 初始位置 (x, y, z)
init_panda_hand_quat = [0, 1, 0, 0]  # 初始姿态四元数 (w, x, y, z)

# 设置机械臂目标位置和姿态
controller.set_tar_pand_hand(init_panda_hand_pos, init_panda_hand_quat)

# 设置夹爪初始宽度为 8cm（最大宽度）
controller.set_finger_width(0.08)

# =============================================================================
# ROS 系统初始化
# =============================================================================

# 设置 ROS 参数：不使用仿真时间
rospy.set_param("/use_sim_time", False)

# 初始化 ROS 节点
# 节点名称：robot_simulation_node
# anonymous=True: 允许同名节点存在
rospy.init_node('robot_simulation_node', anonymous=True)

# =============================================================================
# TF 坐标变换系统初始化
# =============================================================================

# 创建 TF 缓冲区，用于存储坐标变换信息
tf_buffer = tf2_ros.Buffer()

# 创建 TF 监听器，用于查询坐标变换
tf_listener = tf2_ros.TransformListener(tf_buffer)

# 创建点云到激光扫描转换器
# 将 3D 点云数据转换为 2D 激光扫描数据
scan_converter = ScanConverter(tf_buffer=tf_buffer)

# 创建 TF 广播器，用于发布坐标变换
# 注意：只需要初始化一次
br = tf2_ros.TransformBroadcaster()

# =============================================================================
# 静态坐标变换设置
# =============================================================================

# 设置静态 TF 变换：base_link → lidar_frame
# 建立机器人底盘与激光雷达之间的固定坐标关系
setup_static_tf(
    lidar_prim_path="/Robot_1/nova_franka/panda_link0/Lidar",    # 激光雷达在场景中的路径
    base_prim_path="/Robot_1/nova_franka/chassis_link",          # 机器人底盘在场景中的路径
    parent_frame="base_link",                                     # 父坐标系：机器人底盘
    child_frame="lidar_frame"                                     # 子坐标系：激光雷达
)

# =============================================================================
# ROS 话题发布者初始化
# =============================================================================

# 创建点云数据发布者
# 话题名称：/lidar_points
# 消息类型：PointCloud2
pointcloud_pub = rospy.Publisher("/lidar_points", PointCloud2, queue_size=1)

# 创建可视化标记发布者
# 话题名称：visualization_marker
# 消息类型：Marker
marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=1)

# 创建里程计数据发布者
# 话题名称：/odom
# 消息类型：Odometry
odom_pub = rospy.Publisher('/odom', Odometry, queue_size=1)  # 增加

# =============================================================================
# ROS 话题订阅者设置
# =============================================================================

# 设置 ROS 订阅者，处理来自外部的控制命令
# 参数：控制器实例、测试位置、测试姿态
setup_ros_subscribers(controller, test_pos, test_quat)

# ROS 话题发布命令示例（注释掉的测试命令）
# rostopic pub /robot_control std_msgs/String "open" --once    # 打开夹爪
# rostopic pub /robot_control std_msgs/String "close" --once   # 关闭夹爪
# rostopic pub /robot_control std_msgs/String "forward" --once # 前进
# rostopic pub /robot_control std_msgs/String "stop" --once    # 停止
# rostopic pub /robot_control std_msgs/String "pick" --once    # 抓取
# rostopic pub /robot_control std_msgs/String "spin" --once    # 旋转
# rostopic pub /robot_control std_msgs/String "goto" --once    # 导航到目标

print("ROS 节点已启动，监听 /robot_control...")

# =============================================================================
# 物体绑定与动作服务器初始化
# =============================================================================

# 将场景中的物体绑定到控制器
# 使控制器能够管理和操作这些物体
# 注意：坐标移动功能（已注释掉的旧方法）
# controller.bind_cube_as_rigid_object("/World/cube")
controller.attach_objects(objects)

# 创建动作服务器实例
# 提供标准化的机器人动作服务接口
motion_actions = MotionActionServers(controller)

# =============================================================================
# 主仿真循环
# =============================================================================

# 获取物理仿真时间步长
sim_dt = sim.get_physics_dt()

# 初始化计数器（当前未使用）
num = 0

# 主仿真循环：持续运行直到仿真应用关闭
while simulation_app.is_running():
    # 执行物理仿真步进（启用渲染）
    sim.step(render=True)
    
    # 更新机器人控制器状态
    controller.step()

    # =========================================================================
    # 动态坐标变换发布：odom → base_link
    # =========================================================================
    # 发布机器人里程计数据和坐标变换
    # 参数说明：
    # - controller: 机器人控制器实例
    # - br: TF 广播器
    # - marker_pub: 可视化标记发布者
    # - use_carto=True: 使用 Cartographer 格式
    # - odom_pub: 里程计数据发布者
    publish_odom_tf_and_marker(controller, br, marker_pub, use_carto=True, odom_pub=odom_pub)

    # =========================================================================
    # 激光雷达数据发布
    # =========================================================================
    # 发布激光雷达点云数据到 ROS 话题
    # 参数说明：
    # - lidar: 激光雷达传感器实例
    # - pointcloud_pub: 点云数据发布者
    # - frame_id: 坐标系 ID
    publish_lidar_frame(lidar, pointcloud_pub, frame_id="lidar_frame")

    # =========================================================================
    # 环境物体信息发布
    # =========================================================================
    # 发布场景中物体的位置和状态信息
    # 参数说明：
    # - controller: 机器人控制器实例
    # - topic_name: 发布话题名称
    # - filter_by_distance: 是否按距离过滤物体
    publish_top_level_objects_once(controller=controller, topic_name="/environment_objects", 
                                   filter_by_distance=False)

    # =========================================================================
    # 机器人状态信息打印
    # =========================================================================
    
    # 打印机器人底盘在世界坐标系下的位姿
    print("base_pose, base_quat =", controller.get_nova_world_pose())

    # 获取并打印机械臂末端执行器在世界坐标系下的位姿
    rel_pos, rel_quat = controller.get_pand_hand_world_pose()
    print(f"panda_hand (in world) rel_pos = ({rel_pos[0]:.3f}, {rel_pos[1]:.3f}, {rel_pos[2]:.3f}), "
          f"rel_quat = ({rel_quat[0]:.3f}, {rel_quat[1]:.3f}, {rel_quat[2]:.3f}, {rel_quat[3]:.3f})")
    
    # 打印所有关节位置信息
    controller.get_all_joint_positions()

    # 推进仿真到下一帧
    sim.forward()
