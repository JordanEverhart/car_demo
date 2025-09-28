#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rospy
import actionlib
from actionlib_msgs.msg import GoalStatus
from my_carto_config.msg import MoveDistanceAction, MoveDistanceFeedback, MoveDistanceResult
from my_carto_config.msg import RotateAngleAction, RotateAngleFeedback, RotateAngleResult

# ====== 你已有的两个函数从 ros_callbacks 引入（避免重复实现）======
from ros_callbacks import _move_linear, _rotate  # 确保在 ros_callbacks.py 顶部不依赖 ROS 节点初始化

class MotionActionServers:
    def __init__(self, controller):
        self.controller = controller

        self.move_srv = actionlib.SimpleActionServer(
            "move_distance", MoveDistanceAction, execute_cb=self._exec_move, auto_start=False
        )
        self.rot_srv = actionlib.SimpleActionServer(
            "rotate_angle", RotateAngleAction, execute_cb=self._exec_rotate, auto_start=False
        )

        self.move_srv.start()
        self.rot_srv.start()
        rospy.loginfo("✅ Motion actions started: /move_distance, /rotate_angle")

    def _exec_move(self, goal):
        """goal: distance_m, speed_mps"""
        fb = MoveDistanceFeedback()
        res = MoveDistanceResult()

        distance = float(goal.distance_m)
        speed = float(goal.speed_mps) if goal.speed_mps > 0 else 0.25

        # 起点
        start_xy = self._get_xy()
        # 发命令：_move_linear 内部会持续发速度并阻塞循环，我们在这里以更细粒度循环实现反馈/抢占
        rate = rospy.Rate(20)
        sign = 1.0 if distance >= 0 else -1.0
        v_cmd = sign * speed
        timeout = abs(distance) / speed * 3.0 + 3.0
        t0 = rospy.Time.now().to_sec()

        # 开始运动
        self.controller.base_controller(v_cmd, 0.0)

        success = False
        msg = "done"
        try:
            while not rospy.is_shutdown():
                # 抢占
                if self.move_srv.is_preempt_requested():
                    msg = "preempted"
                    break

                cur_xy = self._get_xy()
                moved = self._dist(cur_xy, start_xy)
                fb.moved_m = moved
                self.move_srv.publish_feedback(fb)

                # 完成判定（≥）
                if moved >= abs(distance):
                    success = True
                    break

                # 超时兜底
                if rospy.Time.now().to_sec() - t0 > timeout:
                    msg = "timeout"
                    break

                rate.sleep()
        finally:
            self.controller.base_controller(0.0, 0.0)

        res.success = success
        res.message = msg
        if success:
            self.move_srv.set_succeeded(res)
        else:
            if msg == "preempted":
                self.move_srv.set_preempted(res)
            else:
                self.move_srv.set_aborted(res)

    def _exec_rotate(self, goal):
        """goal: angle_deg, angular_speed_dps"""
        fb = RotateAngleFeedback()
        res = RotateAngleResult()

        angle = math.radians(float(goal.angle_deg))
        w_dps = float(goal.angular_speed_dps) if goal.angular_speed_dps > 0 else 25.0
        w = math.radians(w_dps)

        # 累计角度
        yaw_prev = self._get_yaw()
        yaw_accum = 0.0

        rate = rospy.Rate(20)
        sign = 1.0 if angle >= 0 else -1.0
        w_cmd = sign * w
        timeout = abs(angle) / w * 3.0 + 3.0
        t0 = rospy.Time.now().to_sec()

        self.controller.base_controller(0.0, w_cmd)

        success = False
        msg = "done"
        try:
            while not rospy.is_shutdown():
                if self.rot_srv.is_preempt_requested():
                    msg = "preempted"
                    break

                yaw = self._get_yaw()
                dy = self._normalize_angle(yaw - yaw_prev)
                yaw_accum += dy
                yaw_prev = yaw

                fb.rotated_deg = math.degrees(yaw_accum)
                self.rot_srv.publish_feedback(fb)

                if abs(yaw_accum) >= abs(angle):  # ≥
                    success = True
                    break

                if rospy.Time.now().to_sec() - t0 > timeout:
                    msg = "timeout"
                    break

                rate.sleep()
        finally:
            self.controller.base_controller(0.0, 0.0)

        res.success = success
        res.message = msg
        if success:
            self.rot_srv.set_succeeded(res)
        else:
            if msg == "preempted":
                self.rot_srv.set_preempted(res)
            else:
                self.rot_srv.set_aborted(res)

    # ===== helpers =====
    def _get_xy(self):
        pos, _ = self.controller.get_nova_world_pose()
        return (float(pos[0]), float(pos[1]))

    @staticmethod
    def _dist(a, b):
        return math.hypot(a[0]-b[0], a[1]-b[1])

    def _get_yaw(self):
        _, q = self.controller.get_nova_world_pose()  # q: [x,y,z,w]
        x, y, z, w = q
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y*y + z*z)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def _normalize_angle(a):
        return math.atan2(math.sin(a), math.cos(a))


if __name__ == "__main__":
    rospy.init_node("motion_actions_server")
    # 这里需要拿到你的 controller 实例；如果你在 demo.py 里构造了 controller，
    # 推荐在 demo.py 里 import 本文件并传入 controller 来初始化：
    # MotionActionServers(controller)
    rospy.logwarn("请在你的主程序中实例化 MotionActionServers(controller)")
    rospy.spin()
