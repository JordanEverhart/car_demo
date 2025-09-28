"""
tf_static_setup.py
------------------
严格按照您原始 demo 中的代码实现：
  1. 每次调用都读取 prim → 计算 T_rel
  2. 打印 T_rel、trans、quat 供调试
  3. 构造 geometry_msgs.msg.TransformStamped
  4. 通过 tf2_ros.StaticTransformBroadcaster 立即发送

调用示例：
    from tf_static_setup import setup_static_tf
    setup_static_tf(
        lidar_prim_path="/Robot_1/nova_franka/panda_link0/Lidar",
        base_prim_path ="/Robot_1/nova_franka/chassis_link",
        parent_frame   ="base_link",
        child_frame    ="lidar_frame"
    )
"""
import numpy as np
import rospy, tf2_ros, geometry_msgs.msg
from pxr import UsdGeom
from isaacsim.core.utils.prims import get_prim_at_path
from scipy.spatial.transform import Rotation as R


def setup_static_tf(lidar_prim_path: str,
                    base_prim_path:  str,
                    parent_frame:    str = "base_link",
                    child_frame:     str = "lidar_frame") -> None:
    """完全复刻原始代码逻辑并发送 static TF"""
    static_br = tf2_ros.StaticTransformBroadcaster()

    # 每次执行都强制刷新
    lidar_prim = get_prim_at_path(lidar_prim_path)
    base_prim  = get_prim_at_path(base_prim_path)

    # 确认 Prim 有效性
    if not lidar_prim.IsValid():
        print(f"{lidar_prim_path} 无效！")
    if not base_prim.IsValid():
        print(f"{base_prim_path} 无效！")

    # 获取世界变换矩阵
    lidar_xform = UsdGeom.Xformable(lidar_prim)
    base_xform  = UsdGeom.Xformable(base_prim)

    T_lidar = np.array(lidar_xform.ComputeLocalToWorldTransform(0.0))
    T_base  = np.array(base_xform.ComputeLocalToWorldTransform(0.0))

    # 计算相对变换
    T_rel = np.linalg.inv(T_base) @ T_lidar
    T_rel = T_rel.T

    # 强制打印验证
    print("T_rel =", T_rel)

    # 提取平移和旋转
    trans    = T_rel[:3, 3]
    rot_mat  = T_rel[:3, :3]
    rot_mat  = np.where(np.abs(rot_mat) < 1e-8, 0, rot_mat)
    quat     = R.from_matrix(rot_mat).as_quat()  # [x, y, z, w]

    # 再次打印验证
    print("trans (should be [~0, 0, ~0.33]):", trans)
    print("quat  (should be near [0,0,0,1]):", quat)

    # 构建 ROS static tf 消息
    static_t                       = geometry_msgs.msg.TransformStamped()
    static_t.header.stamp          = rospy.Time.now()
    static_t.header.frame_id       = parent_frame
    static_t.child_frame_id        = child_frame

    static_t.transform.translation.x = trans[0]
    static_t.transform.translation.y = trans[1]
    static_t.transform.translation.z = trans[2]

    static_t.transform.rotation.x    = quat[0]
    static_t.transform.rotation.y    = quat[1]
    static_t.transform.rotation.z    = quat[2]
    static_t.transform.rotation.w    = quat[3]

    # 发送 static TF
    static_br.sendTransform(static_t)
    print(f"✅ 已广播静态 TF {parent_frame} → {child_frame}")
