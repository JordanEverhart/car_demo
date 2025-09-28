from scipy.spatial.transform import Rotation as R
import math

def euler_to_quat_XYZ(x_deg, y_deg, z_deg):
    # 参数就是 roll(X), pitch(Y), yaw(Z)
    # 但要按 'zxy' 顺序执行 → 需要重新映射到正确位置
    r = R.from_euler("XYZ",[x_deg, y_deg, z_deg], degrees=True)
    q = r.as_quat()  # [x, y, z, w]
    print(f"{q[0]:.8f}, {q[1]:.8f}, {q[2]:.8f}, {q[3]:.8f}")
    return q

def quat_to_euler_XYZ(qx, qy, qz, qw):
    """
    将四元数转换为欧拉角 (X, Y, Z)，单位：度
    按 'xyz' 顺序旋转，并直接打印
    """
    r = R.from_quat([qx, qy, qz, qw])
    roll, pitch, yaw = r.as_euler('XYZ', degrees=True)
    print(f"{roll:.8f}, {pitch:.8f}, {yaw:.8f}")

# 示例
# quat_to_euler_xyz(-0.004431632409229443, 0.7076490084009328, 0.004437538,0.7065363046570653 )

euler_to_quat_XYZ(-90, 0, 90)
# quat_to_euler_XYZ(-0.50000000, 0.50000000, 0.50000000, 0.50000000)

