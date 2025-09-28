import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2, LaserScan
import sensor_msgs.point_cloud2 as pc2
from tf2_ros import Buffer

class ScanConverter:
    def __init__(self, tf_buffer: Buffer):
        self.tf_buffer = tf_buffer
        self.frame_id = "lidar_frame"
        self.range_min = 0.001
        self.range_max = 100
        self.angle_min = -3.14
        self.angle_max = 3.14
        self.angle_increment = 0.0174  # 约 1°

        self.z_min = -0.5
        self.z_max = 2.0

        self.pub = rospy.Publisher("/scan", LaserScan, queue_size=1)
        self.sub = rospy.Subscriber("/lidar_points", PointCloud2, self.callback, queue_size=1)
        rospy.loginfo("✅ 自定义 ScanConverter 启动成功")

    def callback(self, msg: PointCloud2):
        try:
            points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            num_bins = int((self.angle_max - self.angle_min) / self.angle_increment)
            ranges = [self.range_max] * num_bins

            valid_points = 0
            for x, y, z in points:
                if not (self.z_min <= z <= self.z_max):
                    continue

                angle = np.arctan2(y, x)
                dist = np.sqrt(x**2 + y**2)

                if self.range_min < dist < self.range_max and self.angle_min <= angle <= self.angle_max:
                    index = int((angle - self.angle_min) / self.angle_increment)
                    if 0 <= index < num_bins:
                        ranges[index] = min(ranges[index], dist)
                        valid_points += 1

            # rospy.loginfo(f"[ScanConverter] 有效 bin: {sum(r < self.range_max for r in ranges)}/{num_bins}")

            scan_msg = LaserScan()
            scan_msg.header = msg.header
            scan_msg.angle_min = self.angle_min
            scan_msg.angle_max = self.angle_max
            scan_msg.angle_increment = self.angle_increment
            scan_msg.range_min = self.range_min
            scan_msg.range_max = self.range_max
            scan_msg.ranges = [r if r < self.range_max else float('inf') for r in ranges]
            scan_msg.time_increment = 0.0
            scan_msg.scan_time = 0.1
            scan_msg.intensities = []

            self.pub.publish(scan_msg)
        except Exception as e:
            rospy.logwarn(f"[ScanConverter] 处理失败: {e}")
