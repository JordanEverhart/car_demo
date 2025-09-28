# -*- coding: utf-8 -*-
"""
Spawn rigid objects (Isaac Lab) from a YAML config file.

Usage (call BEFORE sim.reset()):
    from object_spawner import spawn_from_yaml

    objects = spawn_from_yaml(
        yaml_path="/home/winter/robot/evaluaiton/scenario_2/objects.yaml",
        base_path="/World"
    )
    # objects: Dict[str, isaaclab.assets.RigidObject]
    # e.g. controller._cube = objects.get("cube_red_0")

Notes:
- Must be executed in the "design scene" phase (before SimulationContext.reset()).
- After reset，视图会在第 1 帧 step 后就绪；如需保险，可 sim.step(render=False) 预热 1 帧。
"""

from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import os
import math

import yaml

# Isaac Lab
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg

# Optional utils（不强依赖）
try:
    from isaaclab.utils import prims as prim_utils
except Exception:
    prim_utils = None


# -----------------------
# Helpers
# -----------------------

def _as_tuple3(x, name: str) -> Tuple[float, float, float]:
    if isinstance(x, (list, tuple)) and len(x) == 3:
        return float(x[0]), float(x[1]), float(x[2])
    raise ValueError(f"'{name}' must be a list/tuple of length 3. Got: {x}")

def _as_tuple4(x, name: str) -> Tuple[float, float, float, float]:
    if isinstance(x, (list, tuple)) and len(x) == 4:
        return float(x[0]), float(x[1]), float(x[2]), float(x[3])
    raise ValueError(f"'{name}' must be a list/tuple of length 4. Got: {x}")

def _color_tuple(x: Optional[List[float]]) -> Tuple[float, float, float]:
    if x is None:
        return (0.8, 0.8, 0.8)
    r, g, b = _as_tuple3(x, "color")
    # clamp [0,1]
    return (max(0.0, min(1.0, r)),
            max(0.0, min(1.0, g)),
            max(0.0, min(1.0, b)))

def _ensure_parent_xform(path: str):
    """确保父级 Xform 存在（可选）。"""
    if prim_utils is None:
        return
    parent = os.path.dirname(path)
    if parent and parent not in ("/", ""):
        try:
            prim_utils.create_prim(parent, "Xform")
        except Exception:
            pass


# -----------------------
# Build spawn cfg per type
# -----------------------

def _build_spawn_cfg(obj: Dict[str, Any]) -> Any:
    """
    根据 object 的 'type' 构造 sim_utils.*Cfg（Cuboid/Sphere/Cylinder/Cone/Capsule）。
    """
    typ = (obj.get("type") or "cuboid").lower()

    # 物理与外观
    mass = float(obj.get("mass", 0.25))
    kinematic_enabled = bool(obj.get("kinematic", False))
    gravity_enabled = bool(obj.get("gravity", True))
    color = _color_tuple(obj.get("color"))
    metallic = float(obj.get("metallic", 0.0))
    roughness = float(obj.get("roughness", 0.5))

    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=kinematic_enabled,
        disable_gravity=not gravity_enabled,
    )
    mass_props = sim_utils.MassPropertiesCfg(mass=mass)
    collision_props = sim_utils.CollisionPropertiesCfg()
    visual_mat = sim_utils.PreviewSurfaceCfg(
        diffuse_color=color, metallic=metallic, roughness=roughness
    )

    # 按类型选择形状
    if typ in ("cuboid", "box", "cube"):
        size = obj.get("size", [0.06, 0.06, 0.06])
        if isinstance(size, (int, float)):
            size = [size, size, size]
        sx, sy, sz = _as_tuple3(size, "size")
        return sim_utils.CuboidCfg(
            size=(sx, sy, sz),
            rigid_props=rigid_props,
            mass_props=mass_props,
            collision_props=collision_props,
            visual_material=visual_mat,
        )

    if typ == "sphere":
        radius = float(obj.get("radius", 0.05))
        return sim_utils.SphereCfg(
            radius=radius,
            rigid_props=rigid_props,
            mass_props=mass_props,
            collision_props=collision_props,
            visual_material=visual_mat,
        )

    if typ == "cylinder":
        radius = float(obj.get("radius", 0.04))
        height = float(obj.get("height", 0.1))
        return sim_utils.CylinderCfg(
            radius=radius,
            height=height,
            rigid_props=rigid_props,
            mass_props=mass_props,
            collision_props=collision_props,
            visual_material=visual_mat,
        )

    if typ == "cone":
        radius = float(obj.get("radius", 0.04))
        height = float(obj.get("height", 0.1))
        return sim_utils.ConeCfg(
            radius=radius,
            height=height,
            rigid_props=rigid_props,
            mass_props=mass_props,
            collision_props=collision_props,
            visual_material=visual_mat,
        )

    if typ == "capsule":
        radius = float(obj.get("radius", 0.03))
        height = float(obj.get("height", 0.12))
        return sim_utils.CapsuleCfg(
            radius=radius,
            height=height,
            rigid_props=rigid_props,
            mass_props=mass_props,
            collision_props=collision_props,
            visual_material=visual_mat,
        )

    raise ValueError(f"Unsupported type: {typ}")


def _build_init_state(obj: Dict[str, Any]) -> RigidObjectCfg.InitialStateCfg:
    pose = obj.get("pose", {})  # 单个 pose
    pos = pose.get("position", obj.get("position", [0.0, 0.0, 0.5]))
    rot = pose.get("orientation_wxyz", obj.get("orientation_wxyz", [1.0, 0.0, 0.0, 0.0]))
    lin_vel = pose.get("lin_vel", obj.get("lin_vel", [0.0, 0.0, 0.0]))
    ang_vel = pose.get("ang_vel", obj.get("ang_vel", [0.0, 0.0, 0.0]))
    return RigidObjectCfg.InitialStateCfg(
        pos=_as_tuple3(pos, "pose.position"),
        rot=_as_tuple4(rot, "pose.orientation_wxyz"),
        lin_vel=_as_tuple3(lin_vel, "pose.lin_vel"),
        ang_vel=_as_tuple3(ang_vel, "pose.ang_vel"),
    )


def _expand_instances(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    展开多实例：
    - 支持 count + poses（匹配数量）
    - 支持 grid: {start: [...], spacing: [...], shape: [nx,ny,nz]}
    - 否则返回单实例
    """
    name = obj.get("name", "Object")
    count = int(obj.get("count", 1))
    poses = obj.get("poses", None)
    grid = obj.get("grid", None)

    instances: List[Dict[str, Any]] = []

    if poses:
        if count not in (0, 1) and len(poses) != count:
            raise ValueError(f"'{name}': poses length ({len(poses)}) must match count ({count}).")
        for i, p in enumerate(poses):
            inst = dict(obj)  # shallow copy
            inst["pose"] = {
                "position": p.get("position", [0, 0, 0]),
                "orientation_wxyz": p.get("orientation_wxyz", [1, 0, 0, 0]),
                "lin_vel": p.get("lin_vel", [0, 0, 0]),
                "ang_vel": p.get("ang_vel", [0, 0, 0]),
            }
            inst["name"] = f"{name}_{i}"
            instances.append(inst)
        return instances

    if grid:
        start = _as_tuple3(grid.get("start", [0, 0, 0]), "grid.start")
        spacing = _as_tuple3(grid.get("spacing", [0.2, 0.2, 0.0]), "grid.spacing")
        nx, ny, nz = grid.get("shape", [1, 1, 1])
        nx, ny, nz = int(nx), int(ny), int(nz)
        i = 0
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    pos = (
                        start[0] + ix * spacing[0],
                        start[1] + iy * spacing[1],
                        start[2] + iz * spacing[2],
                    )
                    inst = dict(obj)
                    inst["pose"] = {"position": pos, "orientation_wxyz": [1, 0, 0, 0]}
                    inst["name"] = f"{name}_{i}"
                    instances.append(inst)
                    i += 1
        return instances

    # 默认：按 count 复制，同一 pose（若需要不同位置，建议用 poses/grid）
    for i in range(count):
        inst = dict(obj)
        inst["name"] = f"{name}_{i}" if count > 1 else name
        instances.append(inst)
    return instances


# -----------------------
# Public API
# -----------------------

def spawn_from_yaml(yaml_path: str,
                    base_path: str = "/World") -> Dict[str, RigidObject]:
    """
    读取 YAML 并在场景中 spawn 刚体对象（返回 {name: RigidObject} 字典）。
    注意：必须在 sim.reset() 之前调用。
    """
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    objects_cfg: List[Dict[str, Any]] = cfg.get("objects", [])
    if not isinstance(objects_cfg, list) or not objects_cfg:
        return {}

    results: Dict[str, RigidObject] = {}

    for obj in objects_cfg:
        # 展开多实例
        for inst in _expand_instances(obj):
            name = inst.get("name", "Object")
            typ = inst.get("type", "cuboid")
            # prim 路径
            prim_path = inst.get("path") or f"{base_path}/{name}"
            _ensure_parent_xform(prim_path)

            # 形状 spawn cfg
            spawn_cfg = _build_spawn_cfg(inst)
            # 初始状态
            init_state = _build_init_state(inst)

            # 组装 RigidObjectCfg 并创建
            ro_cfg = RigidObjectCfg(
                prim_path=prim_path,
                spawn=spawn_cfg,
                init_state=init_state
            )
            ro = RigidObject(cfg=ro_cfg)
            results[name] = ro
            print(f"[Spawner] Spawned {typ} -> {prim_path}")

    return results


def spawn_from_dict(config: Dict[str, Any],
                    base_path: str = "/World") -> Dict[str, RigidObject]:
    """
    如果你已有 Python 字典而不是 YAML 文件，也可以用它来生成。
    结构与 YAML 相同（包含 "objects": [...]）
    """
    tmp_path = "__inline__"
    cfg = {"objects": config.get("objects", [])}
    # 直接复用 spawn_from_yaml 的逻辑（不落地文件）
    objects_cfg: List[Dict[str, Any]] = cfg.get("objects", [])
    results: Dict[str, RigidObject] = {}
    for obj in objects_cfg:
        for inst in _expand_instances(obj):
            name = inst.get("name", "Object")
            prim_path = inst.get("path") or f"{base_path}/{name}"
            _ensure_parent_xform(prim_path)
            spawn_cfg = _build_spawn_cfg(inst)
            init_state = _build_init_state(inst)
            ro_cfg = RigidObjectCfg(prim_path=prim_path, spawn=spawn_cfg, init_state=init_state)
            ro = RigidObject(cfg=ro_cfg)
            results[name] = ro
            print(f"[Spawner] Spawned {inst.get('type','cuboid')} -> {prim_path}")
    return results
