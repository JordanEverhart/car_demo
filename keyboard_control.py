#!/usr/bin/env python3
# 运行方式：./run_keyboard.sh 

import rospy
from geometry_msgs.msg import Twist
import sys, select, termios, tty

def get_key():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    key = ''
    if rlist:  
        if key == '\x1b':  # 方向键（以 ESC 开头）
            key += sys.stdin.read(2)  # 读取剩余两个字符
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


# 映射方向键（↑↓←→）
move_bindings = {
    '\x1b[A': (1.0, 0.0),    # ↑：前进
    '\x1b[B': (-1.0, 0.0),   # ↓：后退
    '\x1b[D': (0.0, 1.0),    # ←：左转
    '\x1b[C': (0.0, -1.0),   # →：右转
}

if __name__ == "__main__":
    settings = termios.tcgetattr(sys.stdin)
    rospy.init_node('keyboard_control_node', anonymous=True)
    pub = rospy.Publisher('/keyboard_cmd', Twist, queue_size=10)

    rospy.loginfo("✅ 键盘控制节点已启动（带暂停/恢复功能）")
    print("使用 ↑↓←→ 控制机器人移动，空格暂停/恢复，Ctrl+C 退出。")

    linear_speed = 0.5
    angular_speed = 1.0
    paused = False  # 控制是否暂停

    try:
        while not rospy.is_shutdown():
            key = get_key()
            if key == ' ':
                paused = not paused
                if paused:
                    twist = Twist()  # 发一个 0 速度的指令确保机器人立即停下
                    pub.publish(twist)
                    rospy.logwarn("⏸ 已暂停控制（忽略方向键）")
                else:
                    rospy.loginfo("▶️ 已恢复控制（可使用方向键）")

            elif not paused and key in move_bindings:
                lin, ang = move_bindings[key]
                twist = Twist()
                twist.linear.x = lin * linear_speed
                twist.angular.z = ang * angular_speed
                rospy.loginfo(f"🚗 运动指令: linear={twist.linear.x:.2f}, angular={twist.angular.z:.2f}")
                pub.publish(twist)

            elif key == '\x03':  # Ctrl+C
                break
    except Exception as e:
        rospy.logerr(f"发生异常: {e}")
    finally:
        twist = Twist()
        pub.publish(twist)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        print("已退出并发送停止指令。")

