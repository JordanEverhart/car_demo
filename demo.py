# ========== 基础库导入 ==========
from nav_msgs.msg import Odometry  # 增加
import time
import numpy as np
import torch
from omni.isaac.kit import SimulationApp
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
import tf2_ros
import geometry_msgs.msg
from geometry_msgs.msg import Twist
from scan_converter import ScanConverter
from visualization_msgs.msg import Marker
# 增加
from navgoal import send_nav_goal  # 增加

# ========== 初始化 Isaac Sim ==========
simulation_app = SimulationApp({
    "renderer": "RayTracedLighting",\
    "headless": False,
    "physics_enabled": True,
    "enable_ros_bridge": True
})

print("Simulation started")

# ========== Isaac Sim 扩展与场景加载 ==========
from nova_franka_controller import NovaFranka_Controller, add_robot
from isaacsim.core.utils.prims import get_prim_at_path
from isaacsim.core.utils.stage import open_stage
from pxr import UsdGeom, Gf, PhysxSchema
import omni.kit.commands
import rospy
from std_msgs.msg import String
from scipy.spatial.transform import Rotation as R

kit = omni.kit.app.get_app()
ext_manager = kit.get_extension_manager()
ext_manager.set_extension_enabled("omni.isaac.range_sensor", True)

file_path = "/home/winter/robot/evaluaiton/scenario/shelf/real_scene_phy.usd"

# file_path =  "/home/winter/robot/test/Collected_real_scene_phy/real_scene_phy.usd"
 # 事兴场景/home/winter/robot/usd/demo/Collected_real_scene_phy/real_scene_phy.usd
# "/home/winter/robot/Demo/real_scene_phy.usd"只有墙
open_stage(usd_path=file_path)



# ========== 加载机器人与传感器 ==========
import isaaclab.sim as sim_utils

robot = add_robot()
lidar_path = f"/Robot_1/nova_franka/panda_link0/Lidar"
print(get_prim_at_path(lidar_path).IsValid())

from isaacsim.sensors.physx import RotatingLidarPhysX

lidar = RotatingLidarPhysX(prim_path=lidar_path)



from isaaclab.assets import RigidObject, RigidObjectCfg
import isaaclab.sim as sim_utils
# ========= 在 sim.reset() 之前（设计阶段）spawn 一个刚体 Cube =========
from object_spawner import spawn_from_yaml
import os
yaml_path = os.path.join(os.path.dirname(file_path), "objects.yaml")
if not os.path.exists(yaml_path):
    raise FileNotFoundError(f"找不到 {yaml_path}：请在 {os.path.dirname(file_path)} 放置 objects.yaml")

objects = spawn_from_yaml(yaml_path, base_path="/World")

# ================================================================

# 初始化仿真上下文，并设置摄像机视角
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.01, use_fabric=True, gravity=(0.0, 0.0, -3)))
sim.set_camera_view([-0.31, -1.17, 1], [0.4, -0.4, 0.45])
sim.reset()
sim.step(render=False)

# 可选：初始化（通常在 world.reset() 时自动初始化）
lidar.initialize()
lidar.add_point_cloud_data_to_frame()
# 启用可视化（可选）
lidar.enable_visualization(draw_points=True, draw_lines=True)

sim_dt = sim.get_physics_dt()

controller = NovaFranka_Controller(robot, sim_dt)

test_pos, test_quat = controller.world_pose_to_link0(
    [0.05, 0.01, 0.60],  # 世界坐标下的位置
    [0.706, 0.706, 0, 0]  # 世界坐标下的四元数
)

# 世界坐标系下垂直向下应该不是[0, 1, 0, 0]
init_panda_hand_pos = [0.3, 0, 0.5]
init_panda_hand_quat = [0, 1, 0, 0]
controller.set_tar_pand_hand(init_panda_hand_pos, init_panda_hand_quat)
controller.set_finger_width(0.08)

# ========== ROS 节点与订阅话题 ==========
from ros_callbacks import setup_ros_subscribers

setup_ros_subscribers(controller, test_pos, test_quat)

# rostopic pub /robot_control std_msgs/String "open" --once
# rostopic pub /robot_control std_msgs/String "close" --once
# rostopic pub /robot_control std_msgs/String "forward" --once
# rostopic pub /robot_control std_msgs/String "stop" --once
# rostopic pub /robot_control std_msgs/String "pick" --once
# rostopic pub /robot_control std_msgs/String "spin" --once
# rostopic pub /robot_control std_msgs/String "goto" --once

rospy.set_param("/use_sim_time", False)
rospy.init_node('robot_simulation_node', anonymous=True)

tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

# 启动点云→激光转换器s
scan_converter = ScanConverter(tf_buffer=tf_buffer)

# 初始化 TF 广播器（只需要初始化一次）
br = tf2_ros.TransformBroadcaster()

# ============ 静态 TF: base_link → lidar_frame ============
from tf_static_setup import setup_static_tf

setup_static_tf(
    lidar_prim_path="/Robot_1/nova_franka/panda_link0/Lidar",
    base_prim_path="/Robot_1/nova_franka/chassis_link",
    parent_frame="base_link",
    child_frame="lidar_frame"
)

pointcloud_pub = rospy.Publisher("/lidar_points", PointCloud2, queue_size=1)
marker_pub = rospy.Publisher('visualization_marker', Marker, queue_size=1)
odom_pub = rospy.Publisher('/odom', Odometry, queue_size=1)  # 增加

print("ROS 节点已启动，监听 /robot_control...")

## 坐标移动
# controller.bind_cube_as_rigid_object("/World/cube")
controller.attach_objects(objects)

from motion_actions import MotionActionServers
motion_actions = MotionActionServers(controller)


# ========== 仿真主循环 ==========
sim_dt = sim.get_physics_dt()
num = 0

while simulation_app.is_running():
    sim.step(render=True)
    controller.step()

    # ===== 动态 TF: odom → base_link =====
    from tf_dynamic_odom import publish_odom_tf_and_marker

    publish_odom_tf_and_marker(controller, br, marker_pub, use_carto=True, odom_pub=odom_pub)

    from lidar_publisher import publish_lidar_frame

    publish_lidar_frame(lidar, pointcloud_pub, frame_id="lidar_frame")

    from scene_objects_export import publish_top_level_objects_once

    publish_top_level_objects_once(controller = controller ,topic_name="/environment_objects" ,
                                   filter_by_distance= False)

    print("base_pose, base_quat =", controller.get_nova_world_pose())

    rel_pos, rel_quat = controller.get_pand_hand_world_pose()
    print(f"panda_hand (in world) rel_pos = ({rel_pos[0]:.3f}, {rel_pos[1]:.3f}, {rel_pos[2]:.3f}), "
          f"rel_quat = ({rel_quat[0]:.3f}, {rel_quat[1]:.3f}, {rel_quat[2]:.3f}, {rel_quat[3]:.3f})")
    
    controller.get_all_joint_positions()

    sim.forward()
