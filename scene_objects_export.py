"""
scene_objects_export.py
-----------------------
将当前打开的 USD 场景中 /World 下所有顶层 Xform 物体的世界坐标导出到指定文件。

用法:
    from scene_objects_export import export_top_level_objects
    export_top_level_objects(output_txt_path)
"""
from pxr import UsdGeom, Usd
from isaacsim.core.utils.stage import get_current_stage
import rospy
import json
import math
from std_msgs.msg import String


def publish_top_level_objects_once(controller=None,
                                   topic_name="/environment_objects",
                                   filter_by_distance=False,
                                   radius=1.0):
    """
    发布 /World 下的顶层 Xform 物体位置。
    - filter_by_distance=False: 发布所有物体（原始行为）
    - filter_by_distance=True : 仅发布与机器人距离 <= radius 的物体
      (此时需要传入 controller，并实现 get_nova_world_pose())
      物体一旦进入发布列表，即使之后远离机器人也不会被移除。
    """
    stage = get_current_stage()
    if stage is None:
        rospy.logwarn("[scene_objects_export] 当前Stage未加载！")
        return

    robot_pos = None
    if filter_by_distance:
        if controller is None:
            rospy.logerr("[scene_objects_export] filter_by_distance=True 但未传入 controller")
            return
        try:
            robot_pos, _ = controller.get_nova_world_pose()  # [x, y, z]
        except Exception as e:
            rospy.logerr(f"[scene_objects_export] 获取机器人位置失败: {e}")
            return

    pub = rospy.Publisher(topic_name, String, queue_size=10)
    world_prim = stage.GetPrimAtPath("/World")
    objects_data = {}
    published_objects = set()  # 用来存储已经加入发布名单的物体名称

    for child in world_prim.GetChildren():
        if not UsdGeom.Xformable(child):
            continue

        xform = UsdGeom.Xformable(child)
        world_matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        translation = world_matrix.ExtractTranslation()

        include = False  # 默认不包含物体，除非满足条件
        if filter_by_distance and robot_pos is not None:
            dx = float(translation[0]) - float(robot_pos[0])
            dy = float(translation[1]) - float(robot_pos[1])
            dz = float(translation[2]) - float(robot_pos[2])
            dist = math.sqrt(dx*dx + dy*dy + dz*dz)
            
            # 如果物体离机器人足够近，加入发布名单
            if dist <= radius and child.GetName() not in published_objects:
                include = True
                published_objects.add(child.GetName())  # 将物体加入发布名单
        else:
            include = True  # 如果不需要过滤距离，则默认加入所有物体

        # 一旦加入发布名单，即使后续远离也不会移除
        if include:
            name = child.GetName()
            objects_data[name] = [
                round(translation[0], 3),
                round(translation[1], 3),
                round(translation[2], 3)
            ]

    json_str = json.dumps(objects_data, ensure_ascii=False)
    pub.publish(json_str)