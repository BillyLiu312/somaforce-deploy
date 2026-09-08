"""Task profiles and temporary MuJoCo scenes for HDMI sim2sim."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class HDMISim2SimTask:
    name: str
    artifact_dir: str
    motion_dir: str
    policy_steps: int
    scene_owner: str
    scene_path: str
    publish_object_names: tuple[str, ...]
    primary_object_body: str
    primary_object_joint: str
    pose_ports: tuple[tuple[str, int], ...] = ()
    fixed_object_body: str | None = None
    object_xml: str | None = None
    injected_sensor_bodies: tuple[str, ...] = ()
    largebox_mesh: str | None = None
    box_half_extents: tuple[float, float, float] | None = None


TASKS = {
    "move_suitcase": HDMISim2SimTask(
        name="move_suitcase",
        artifact_dir="artifacts/hdmi_move_suitcase/hdmi_tag",
        motion_dir="assets/mujoco/reference/hdmi_suitcase",
        policy_steps=472,
        scene_owner="upstream",
        scene_path="data/robots/g1/g1_29dof_rubberhand-suitcase.xml",
        publish_object_names=("suitcase", "pelvis"),
        primary_object_body="suitcase",
        primary_object_joint="suitcase_root",
    ),
    "push_door_hand": HDMISim2SimTask(
        name="push_door_hand",
        artifact_dir="artifacts/hdmi_push_door_hand/hdmi_tag",
        motion_dir="assets/mujoco/reference/hdmi_push_door_hand",
        policy_steps=573,
        scene_owner="upstream",
        scene_path="data/robots/g1/g1_29dof_rubberhand-suitcase.xml",
        publish_object_names=("door", "door_panel", "pelvis"),
        primary_object_body="door_panel",
        primary_object_joint="door_joint",
        fixed_object_body="door",
        object_xml="active_adaptation/assets_mjcf/objects/door/door.xml",
        injected_sensor_bodies=("door", "door_panel"),
    ),
    "push_box": HDMISim2SimTask(
        name="push_box",
        artifact_dir="artifacts/hdmi_push_box/hdmi_tag",
        motion_dir="assets/mujoco/reference/push_box",
        policy_steps=792,
        scene_owner="hdmi",
        scene_path="active_adaptation/assets_mjcf/g1_29dof_nohand/g1_29dof_nohand-eef_L-box.xml",
        publish_object_names=("box", "pelvis"),
        primary_object_body="box",
        primary_object_joint="box_root",
        object_xml="upstream:data/objects/box/box.xml",
    ),
    "move_largebox": HDMISim2SimTask(
        name="move_largebox",
        artifact_dir="artifacts/hdmi_move_largebox/hdmi_tag",
        motion_dir="assets/mujoco/reference/hdmi_move_largebox",
        policy_steps=199,
        scene_owner="upstream",
        scene_path="data/robots/g1/g1_29dof_rubberhand-suitcase.xml",
        publish_object_names=("largebox", "largebox_link", "pelvis"),
        primary_object_body="largebox_link",
        primary_object_joint="largebox_root",
        pose_ports=(("largebox", 5573), ("largebox_link", 5574)),
        largebox_mesh="active_adaptation/assets/objects/largebox/largebox.obj",
    ),
}


def get_task(name: str) -> HDMISim2SimTask:
    try:
        return TASKS[name]
    except KeyError as exc:
        raise ValueError(f"unknown HDMI sim2sim task {name!r}") from exc


def task_artifact_dir(task: HDMISim2SimTask, repo_root: Path) -> Path:
    return repo_root / task.artifact_dir


def task_motion_dir(task: HDMISim2SimTask, repo_root: Path) -> Path:
    return repo_root / task.motion_dir


def _resolve_object_xml(spec: str, *, upstream_root: Path, hdmi_root: Path) -> Path:
    if spec.startswith("upstream:"):
        return (upstream_root / spec.removeprefix("upstream:")).resolve()
    return (hdmi_root / spec).resolve()


def _largebox_object_xml(mesh_path: Path, output_path: Path) -> None:
    root = ET.Element("mujoco", {"model": "largebox"})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "mesh", {"name": "largebox_mesh", "file": str(mesh_path)})
    world = ET.SubElement(root, "worldbody")
    body = ET.SubElement(world, "body", {"name": "largebox"})
    ET.SubElement(body, "freejoint", {"name": "largebox_root"})
    link = ET.SubElement(body, "body", {"name": "largebox_link"})
    ET.SubElement(
        link,
        "geom",
        {
            "name": "largebox_geom",
            "type": "mesh",
            "mesh": "largebox_mesh",
            "mass": "1.0",
            "friction": "0.9 0.5 0.001",
            "rgba": "0.7 0.8 0.9 0.7",
        },
    )
    sensor = ET.SubElement(root, "sensor")
    for body_name in ("largebox", "largebox_link"):
        ET.SubElement(
            sensor,
            "framepos",
            {"name": f"{body_name}_pos", "objtype": "xbody", "objname": body_name},
        )
        ET.SubElement(
            sensor,
            "framequat",
            {"name": f"{body_name}_quat", "objtype": "xbody", "objname": body_name},
        )
    ET.ElementTree(root).write(output_path, encoding="unicode")


def _box_object_xml(half_extents: tuple[float, float, float], output_path: Path) -> None:
    hx, hy, hz = half_extents
    root = ET.Element("mujoco", {"model": "box"})
    world = ET.SubElement(root, "worldbody")
    body = ET.SubElement(world, "body", {"name": "box"})
    ET.SubElement(body, "freejoint", {"name": "box_root"})
    ET.SubElement(
        body,
        "geom",
        {
            "name": "box_geom",
            "type": "box",
            "size": f"{hx} {hy} {hz}",
            "pos": f"{-hx} 0 {hz}",
            "mass": "8.0",
            "friction": "0.5 0.005 0.0001",
            "rgba": "0.8 0.8 0.8 1",
        },
    )
    sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "framepos", {"name": "box_pos", "objtype": "xbody", "objname": "box"})
    ET.SubElement(sensor, "framequat", {"name": "box_quat", "objtype": "xbody", "objname": "box"})
    ET.ElementTree(root).write(output_path, encoding="unicode")


def _fixed_door_object_xml(source_path: Path, output_path: Path) -> None:
    tree = ET.parse(source_path)
    root = tree.getroot()
    door = root.find("./worldbody/body[@name='door']")
    if door is None:
        raise ValueError(f"door body is missing from {source_path}")
    for freejoint in door.findall("freejoint"):
        door.remove(freejoint)
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    ET.SubElement(
        actuator,
        "motor",
        {
            "name": "door_joint",
            "joint": "door_joint",
            "ctrllimited": "true",
            "ctrlrange": "-100 100",
        },
    )
    tree.write(output_path, encoding="unicode")


def materialize_scene(
    task: HDMISim2SimTask,
    *,
    upstream_root: Path,
    hdmi_root: Path,
    output_dir: Path,
) -> Path:
    """Create a relocatable temporary scene without modifying either source checkout."""
    owner_root = upstream_root if task.scene_owner == "upstream" else hdmi_root
    source_path = (owner_root / task.scene_path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"HDMI sim2sim source scene is missing: {source_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(source_path)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir"):
        mesh_dir = Path(compiler.get("meshdir", ""))
        if not mesh_dir.is_absolute():
            compiler.set("meshdir", str((source_path.parent / mesh_dir).resolve()))

    includes = root.findall("include")
    if task.fixed_object_body is not None:
        if len(includes) != 1:
            raise ValueError("fixed-object scene template must contain exactly one object include")
        if task.object_xml is None:
            raise ValueError("fixed object profile is missing object_xml")
        source_object = _resolve_object_xml(task.object_xml, upstream_root=upstream_root, hdmi_root=hdmi_root)
        object_xml = output_dir / "door.xml"
        _fixed_door_object_xml(source_object, object_xml)
        includes[0].set("file", str(object_xml))
    elif task.largebox_mesh is not None:
        if len(includes) != 1:
            raise ValueError("large-box scene template must contain exactly one object include")
        mesh_path = (hdmi_root / task.largebox_mesh).resolve()
        if not mesh_path.is_file():
            raise FileNotFoundError(f"large-box mesh is missing: {mesh_path}")
        object_xml = output_dir / "largebox.xml"
        _largebox_object_xml(mesh_path, object_xml)
        includes[0].set("file", str(object_xml))
    elif task.box_half_extents is not None:
        if len(includes) != 1:
            raise ValueError("box scene template must contain exactly one object include")
        object_xml = output_dir / "box.xml"
        _box_object_xml(task.box_half_extents, object_xml)
        includes[0].set("file", str(object_xml))
    elif task.object_xml is not None:
        if len(includes) != 1:
            raise ValueError("object scene template must contain exactly one object include")
        includes[0].set("file", str(_resolve_object_xml(task.object_xml, upstream_root=upstream_root, hdmi_root=hdmi_root)))
    else:
        for include in includes:
            include_path = Path(include.get("file", ""))
            if not include_path.is_absolute():
                include.set("file", str((source_path.parent / include_path).resolve()))

    if task.injected_sensor_bodies:
        sensor = ET.SubElement(root, "sensor")
        for body_name in task.injected_sensor_bodies:
            ET.SubElement(
                sensor,
                "framepos",
                {"name": f"{body_name}_pos", "objtype": "xbody", "objname": body_name},
            )
            ET.SubElement(
                sensor,
                "framequat",
                {"name": f"{body_name}_quat", "objtype": "xbody", "objname": body_name},
            )

    output_path = output_dir / f"{task.name}.xml"
    tree.write(output_path, encoding="unicode")
    return output_path
