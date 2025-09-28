#!/usr/bin/env python3
import rospy, math
from geometry_msgs.msg import PoseStamped
from tf.transformations import quaternion_from_euler

def send_nav_goal(pub, x, y, yaw_deg):
    q = quaternion_from_euler(0, 0, math.radians(yaw_deg))

    goal = PoseStamped()
    goal.header.frame_id = "map"
    goal.header.stamp = rospy.Time.now()
    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.orientation.x, goal.pose.orientation.y = q[0], q[1]
    goal.pose.orientation.z, goal.pose.orientation.w = q[2], q[3]

    pub.publish(goal)
    rospy.loginfo(f"已发布目标 → x={x:.2f}, y={y:.2f}, yaw={yaw_deg}°")

if __name__ == "__main__":
    rospy.init_node("navgoal_sender")

    # **latch=True** 保证即便脚本稍后退出，move_base 也能拿到消息
    pub = rospy.Publisher("/move_base_simple/goal",PoseStamped, queue_size=1, latch=True)
    rospy.sleep(1.0)                        # 等待连接


    send_nav_goal(pub,0.41, -2.0, 0)

    # print("输入 x y yaw(°)，用空格分隔，例如： 2.5 -1.0 90")
    # try:
    #     while not rospy.is_shutdown():
    #         line = input(">>> ").strip()
    #         if not line:                     # 空行跳过
    #             continue
    #         try:
    #             x, y, yaw = map(float, line.split())
    #             send_nav_goal(pub, x, y, yaw)
    #         except ValueError:
    #             print("格式错误！应为  x y yaw")
    # except (KeyboardInterrupt, EOFError):
    #     pass
