#!/usr/bin/env python
# coding=utf-8
import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32,Float32MultiArray
import sys
sys.path.insert(0, "/home/winter/robot/evaluaiton/monitor")
from ros_cmd_utils import * # type: ignore

def publish_hand_pose_once(world_pos, world_quat, topic="/set_target_panda_hand_world", wait_secs=0.5):
    """
    一次性发布末端执行器在【世界坐标系】下的 Pose（四元数为 xyzw）。
    world_pos: [x, y, z]
    world_quat: [x, y, z, w]   # 注意 xyzw 顺序（ROS 标准）
    """
    pub = rospy.Publisher(topic, Pose, queue_size=1, latch=True)

    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = world_pos
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = world_quat

    rospy.sleep(wait_secs)  # 等待连接建立
    pub.publish(pose)
    rospy.loginfo("✅ Published once to %s: pos=%s quat(xyzw)=%s", topic, world_pos, world_quat)

def publish_gripper_width_once(width_m, topic="/set_finger_width", wait_secs=0.3):
    """
    一次性发布夹爪开度（单位：米）。控制端会将宽度转换为左右手指目标位姿。
    width_m: float, 例如 0.08（最大开度 8cm）
    """
    pub = rospy.Publisher(topic, Float32, queue_size=1, latch=True)

    msg = Float32()
    msg.data = float(width_m)

    rospy.sleep(wait_secs)  # 等待连接建立
    pub.publish(msg)
    rospy.loginfo("✅ Published once to %s: width=%.3f m", topic, width_m)


def publish_pick_command(joints):
    pub = rospy.Publisher('/pick', Float32MultiArray, queue_size=10)
    rospy.sleep(1.0)  # 等待连接建立

    # 示例：7 个关节角度（单位是 radians）
    joint_angles = joints

    msg = Float32MultiArray(data=joint_angles)
    pub.publish(msg)
    rospy.loginfo(f"已发布关节目标位置到 /pick: {joint_angles}")



# ===== 示例：直接运行本文件时各发布一次（你可以修改数值后直接运行）=====
if __name__ == "__main__":
    # rospy.init_node("testhand", anonymous=True)

    turn_right(10)
    rospy.sleep(2)
    move_forward(2)
    rospy.sleep(2)

    # world_pos = [-0.218,3.96, 0.6]
    # world_quat = [0.706, 0.706, 0.0, 0.0] # xyzw
    # publish_hand_pose_once(world_pos, world_quat)
    # rospy.sleep(8)
    # publish_pick_command([-1.269877, 0.56238806, 1.85119403, -1.28880597, -1.78089552, 2.485351, -1.090143])
    # rospy.sleep(6)
    # publish_gripper_width_once(0.8)  # 夹爪开度 4cm
    # rospy.sleep(1)


    # publish_set_object_options( 
    #     name="red_cube",
    #     attract=True,
    #     offset=[0.00, 0.0, 0.093],
    #     alpha=0.1,
    #     align_orientation=True
    # )

    # publish_gripper_width_once(0.060)  # 夹爪开度 4cm
    # rospy.sleep(1)


    # send_nav_goal(-2.45,-0.0,-90)
    # rospy.sleep(2)
    
    # publish_gripper_width_once(0.80)  # 夹爪开度 4cm
    # rospy.sleep(1)


    # publish_set_object_options( 
    #     name="red_cube",
    #     attract=False,
    # )

    # send_nav_goal(0.0,4.0,0.0)
    # rospy.sleep(2)

    # move_forward(0.2)
    # rospy.sleep(2)

    # world_pos = [0.16,3.9, 0.6]
    # world_quat = [0.706, 0.706, 0.0, 0.0] # xyzw
    # publish_hand_pose_once(world_pos, world_quat)
    # rospy.sleep(8)


    # publish_set_object_options( 
    #     name="blue_cube",
    #     attract=True,
    #     offset=[0.00, 0.0, 0.093],
    #     alpha=0.1,
    #     align_orientation=True
    # )

    # publish_gripper_width_once(0.060)  # 夹爪开度 4cm
    # rospy.sleep(1)


    # send_nav_goal(5.78,-0.0,0)

    # rospy.sleep(2)

    # publish_set_object_options( 
    #     name="blue_cube",
    #     attract=False,
    # )

    # publish_gripper_width_once(0.80)  # 夹爪开度 4cm
    # rospy.sleep(1)
    # goto(5.78,-0.0,0)