"""
lidar_publisher.py
------------------
严格复刻原始 demo 中的点云发布逻辑，不做任何简化或重排。
提供 publish_lidar_frame(lidar, pointcloud_pub) 供外部调用。
"""
import numpy as np
import rospy, std_msgs.msg
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

__all__ = ["publish_lidar_frame"]

def publish_lidar_frame(lidar, pointcloud_pub, frame_id: str = "lidar_frame"):
    """
    完整复制原来 while 循环里的点云发布代码，逻辑与打印保持一致。
    :param lidar:           RotatingLidarPhysX 对象
    :param pointcloud_pub:  rospy.Publisher('/lidar_points', PointCloud2, ...)
    :param frame_id:        PointCloud2.header.frame_id，默认 "lidar_frame"
    """
    if lidar.is_valid():

        # 获取当前帧的数据
        frame_data = lidar.get_current_frame()

        # 获取点云（单位为相对传感器的 xyz）
        if "point_cloud" in frame_data:
            point_cloud = frame_data["point_cloud"]
            print(f"点云点数: {point_cloud.shape[0]}")
            if point_cloud.shape[0] > 0:
                ranges = np.linalg.norm(point_cloud[:, :2], axis=1)
                print(f"当前帧测距范围: min={ranges.min():.3f}m, max={ranges.max():.3f}m")

            if point_cloud.shape[0] > 0:
                print("第一点坐标:", point_cloud[0])

                # 🚨 修正结构为二维 (N, 3)
                if point_cloud.ndim == 3:
                    point_cloud = point_cloud.reshape(-1, 3)

                elif point_cloud.shape[1] != 3:
                    raise ValueError(f"点云格式错误，应为 (N, 3)，当前为: {point_cloud.shape}")

                # ==== 生成 PointCloud2 消息 ====
                header = std_msgs.msg.Header()
                header.stamp = rospy.Time.now()
                header.frame_id = frame_id  # 这个名字可以和 tf 配合设置

                fields = [
                    PointField('x', 0, PointField.FLOAT32, 1),
                    PointField('y', 4, PointField.FLOAT32, 1),
                    PointField('z', 8, PointField.FLOAT32, 1)
                ]

                # 注意：要转为 list 才能用 ROS 的接口
                pc2_msg = pc2.create_cloud(header, fields, [tuple(p) for p in point_cloud])

                # ==== 发布到 /lidar_points ====
                pointcloud_pub.publish(pc2_msg)

            else:
                print("当前帧没有点云数据")
    else:
        print("Lidar 无效")
