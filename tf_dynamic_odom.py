"""
tf_dynamic_odom.py
-------------------
封装 base_link 在 odom 坐标系下的动态 TF 发送，并发布圆形 CYLINDER Marker 可视区域。

用法：
    from tf_dynamic_odom import publish_odom_tf_and_marker
    publish_odom_tf_and_marker(controller, br, marker_pub)
"""
import rospy
import geometry_msgs.msg
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry

__all__ = ["publish_odom_tf_and_marker"]

def publish_odom_tf_and_marker(controller, br, marker_pub, publish_ee_tf = True,use_carto=False, odom_pub=None):
    # ===== 获取机器人当前位姿 =====
    base_pose, base_quat = controller.get_nova_world_pose()  # [x,y,z], [x,y,z,w]

    # ===== 如果不使用 Cartographer，发布 TF: odom → base_link =====
    if not use_carto:
        odom_tf = geometry_msgs.msg.TransformStamped()
        odom_tf.header.stamp = rospy.Time.now()
        odom_tf.header.frame_id = "odom"
        odom_tf.child_frame_id = "base_link"

        odom_tf.transform.translation.x = base_pose[0]
        odom_tf.transform.translation.y = base_pose[1]
        odom_tf.transform.translation.z = base_pose[2]

        odom_tf.transform.rotation.x = base_quat[0]
        odom_tf.transform.rotation.y = base_quat[1]
        odom_tf.transform.rotation.z = base_quat[2]
        odom_tf.transform.rotation.w = base_quat[3]

        br.sendTransform(odom_tf)

    # ===== 如果使用 Cartographer，发布 nav_msgs/Odometry 到 /odom =====
    else:
        if odom_pub is None:
            rospy.logwarn("use_carto=True，但未传入 odom_pub，无法发布 /odom")
        else:
            odom_msg = Odometry()
            odom_msg.header.stamp = rospy.Time.now()
            odom_msg.header.frame_id = "odom"
            odom_msg.child_frame_id = "base_link"

            odom_msg.pose.pose.position.x = base_pose[0]
            odom_msg.pose.pose.position.y = base_pose[1]
            odom_msg.pose.pose.position.z = base_pose[2]

            odom_msg.pose.pose.orientation.x = base_quat[0]
            odom_msg.pose.pose.orientation.y = base_quat[1]
            odom_msg.pose.pose.orientation.z = base_quat[2]
            odom_msg.pose.pose.orientation.w = base_quat[3]

            # 不提供速度信息，可填零（可选）
            odom_msg.twist.twist.linear.x = 0.0
            odom_msg.twist.twist.linear.y = 0.0
            odom_msg.twist.twist.linear.z = 0.0
            odom_msg.twist.twist.angular.x = 0.0
            odom_msg.twist.twist.angular.y = 0.0
            odom_msg.twist.twist.angular.z = 0.0

            odom_pub.publish(odom_msg)

    # ===== 新增：发布末端执行器 TF，默认 base_link → end_effector =====
    if publish_ee_tf:
        try:
            rel_pos, rel_quat = controller.get_pand_hand_world_pose()  # [x,y,z], [qx,qy,qz,qw]
            ee_tf = geometry_msgs.msg.TransformStamped()
            ee_tf.header.stamp = rospy.Time.now()
            ee_tf.header.frame_id = "map"      # e.g., "base_link"
            ee_tf.child_frame_id = "end_effector"        # e.g., "end_effector"
            ee_tf.transform.translation.x = float(rel_pos[0])
            ee_tf.transform.translation.y = float(rel_pos[1])
            ee_tf.transform.translation.z = float(rel_pos[2])
            ee_tf.transform.rotation.x = float(rel_quat[1])
            ee_tf.transform.rotation.y = float(rel_quat[2])
            ee_tf.transform.rotation.z = float(rel_quat[3])
            ee_tf.transform.rotation.w = float(rel_quat[0])
            br.sendTransform(ee_tf)
        except AttributeError:
            rospy.logwarn("controller 未实现 get_pand_hand_world_pose()，无法发布末端 TF")
        except Exception as e:
            rospy.logerr(f"发布末端 TF 失败：{e}")

    # ===== 发布圆形可视化 Marker（不变）=====
    marker = Marker()
    marker.header.frame_id = "base_link"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "view_range"
    marker.id = 0
    marker.type = Marker.CYLINDER
    marker.action = Marker.ADD

    marker.pose.position.x = 0.0
    marker.pose.position.y = 0.0
    marker.pose.position.z = 0.01
    marker.pose.orientation.w = 1.0

    radius = 0.5
    marker.scale.x = radius * 2
    marker.scale.y = radius * 2
    marker.scale.z = 0.01

    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 1.0
    marker.color.a = 0.2

    marker.lifetime = rospy.Duration(0)
    marker_pub.publish(marker)