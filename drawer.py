#!/usr/bin/env python
import rospy
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

def publish_circle_marker():
    rospy.init_node('view_range_marker')
    pub = rospy.Publisher('visualization_marker', Marker, queue_size=10)

    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        marker = Marker()
        marker.header.frame_id = "base_link"  # 以机器人为中心
        marker.header.stamp = rospy.Time.now()
        marker.ns = "view_range"
        marker.id = 0
        marker.type = Marker.CYLINDER  # 用圆柱体模拟圆盘
        marker.action = Marker.ADD

        marker.pose.position.x = 0
        marker.pose.position.y = 0
        marker.pose.position.z = 0  # 位于地面上
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1

        # 圆的尺寸（x 和 y 是直径）
        marker.scale.x = 4.0  # 可视范围直径，例如 4 米
        marker.scale.y = 4.0
        marker.scale.z = 0.01  # 极薄的圆盘

        # 设置为透明白色
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 0.2  # 透明度

        marker.lifetime = rospy.Duration()  # 持续显示

        pub.publish(marker)
        rate.sleep()

if __name__ == "__main__":
    try:
        publish_circle_marker()
    except rospy.ROSInterruptException:
        pass
