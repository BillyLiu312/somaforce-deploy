#!/usr/bin/env python3
"""Live, no-dynamics MuJoCo viewer for marker-to-policy transforms.

The script reads two ROS2 PoseStamped marker poses, applies fixed homogeneous
transforms, and updates only the free joints of a MuJoCo G1+suitcase scene.
It intentionally does not read Unitree low-state or perform waist FK. Use it
to validate the marker geometry and pose transforms independently.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.ros2_pose_to_zmq import _pose_matrix_from_zmq_values, _wxyz_from_rotation, pose_stamped_to_zmq_values
from somaforce_deploy.torso2pelvis import make_transform, validate_transform


def _load_matrix(payload: dict[str, Any], keys: tuple[str, ...], name: str) -> np.ndarray:
    for key in keys:
        if key in payload:
            return validate_transform(payload[key], name=name)
    raise ValueError(f"calibration JSON is missing one of {keys}")


def _set_free_pose(model: Any, data: Any, joint_name: str, pose: np.ndarray) -> None:
    joint = model.joint(joint_name)
    address = int(model.jnt_qposadr[joint.id])
    data.qpos[address : address + 3] = pose[:3, 3]
    data.qpos[address + 3 : address + 7] = _wxyz_from_rotation(pose[:3, :3])


def _initialize_reference_pose(model: Any, data: Any, motion_path: Path, meta_path: Path) -> None:
    motion = np.load(motion_path, allow_pickle=False)
    meta = json.loads(meta_path.read_text())
    model_joints = {
        model.joint(i).name: i for i in range(model.njnt)
    }
    for joint_name, value in zip(meta["joint_names"], motion["joint_pos"][0], strict=True):
        if joint_name in model_joints:
            joint = model.joint(joint_name)
            data.qpos[int(model.jnt_qposadr[joint.id])] = float(value)
    body_to_joint = {"pelvis": "pelvis_root", "suitcase": "suitcase_root"}
    for body_name, joint_name in body_to_joint.items():
        if body_name not in meta["body_names"] or joint_name not in model_joints:
            continue
        body_index = meta["body_names"].index(body_name)
        joint = model.joint(joint_name)
        qpos = int(model.jnt_qposadr[joint.id])
        data.qpos[qpos : qpos + 3] = motion["body_pos_w"][0, body_index]
        data.qpos[qpos + 3 : qpos + 7] = motion["body_quat_w"][0, body_index]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torso-topic", default="/robot_g1/pose")
    parser.add_argument("--suitcase-topic", default="/suitcase/pose")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--scene", type=Path, default=REPO_ROOT.parent / "sim2real-hdmi-upstream/data/robots/g1/g1_29dof_rubberhand-suitcase.xml")
    parser.add_argument("--motion", type=Path, default=REPO_ROOT / "assets/mujoco/reference/hdmi_suitcase/motion.npz")
    parser.add_argument("--motion-meta", type=Path, default=REPO_ROOT / "assets/mujoco/reference/hdmi_suitcase/meta.json")
    parser.add_argument("--position-scale", type=float, default=0.001)
    parser.add_argument("--expected-frame-id", default="world")
    parser.add_argument("--stale-timeout", type=float, default=0.25)
    parser.add_argument("--render-hz", type=float, default=30.0)
    args = parser.parse_args()
    if args.position_scale <= 0 or args.stale_timeout <= 0 or args.render_hz <= 0:
        raise ValueError("position-scale, stale-timeout, and render-hz must be positive")

    try:
        import mujoco
        import mujoco.viewer
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    except ImportError as exc:
        raise RuntimeError("source ROS2 and use the project environment with MuJoCo") from exc

    calibration = json.loads(args.calibration.read_text())
    marker_to_torso = _load_matrix(
        calibration,
        ("marker_from_torso", "marker_to_torso", "T_M_T", "torso_marker_to_torso"),
        "marker_to_torso",
    )
    marker_to_suitcase = _load_matrix(
        calibration,
        ("marker_from_suitcase", "marker_to_suitcase", "T_Ms_S", "suitcase_marker_to_suitcase"),
        "marker_to_suitcase",
    )

    rclpy.init()
    node = rclpy.create_node("sim2real_marker_policy_mujoco")
    qos = QoSProfile(depth=30)
    qos.reliability = ReliabilityPolicy.BEST_EFFORT
    qos.durability = DurabilityPolicy.VOLATILE
    latest: dict[str, tuple[float, np.ndarray]] = {}

    def callback(name: str) -> Any:
        def receive(msg: Any) -> None:
            if args.expected_frame_id and str(msg.header.frame_id) != args.expected_frame_id:
                return
            try:
                values = pose_stamped_to_zmq_values(msg, position_scale=args.position_scale)
                latest[name] = (time.monotonic(), _pose_matrix_from_zmq_values(values))
            except ValueError as exc:
                node.get_logger().warning(f"rejecting {name} pose: {exc}")
        return receive

    node.create_subscription(PoseStamped, args.torso_topic, callback("torso_marker"), qos)
    node.create_subscription(PoseStamped, args.suitcase_topic, callback("suitcase_marker"), qos)

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    _initialize_reference_pose(model, data, args.motion, args.motion_meta)
    mujoco.mj_forward(model, data)
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -18
    viewer.cam.distance = 3.5
    period = 1.0 / args.render_hz
    last_render = time.monotonic()
    try:
        while viewer.is_running() and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            now = time.monotonic()
            if "torso_marker" not in latest or "suitcase_marker" not in latest:
                time.sleep(0.005)
                continue
            torso_time, marker_torso_world = latest["torso_marker"]
            suitcase_time, marker_suitcase_world = latest["suitcase_marker"]
            if max(now - torso_time, now - suitcase_time) > args.stale_timeout:
                node.get_logger().warning("marker pose watchdog expired")
                time.sleep(0.02)
                continue
            world_torso = marker_torso_world @ marker_to_torso
            world_suitcase = marker_suitcase_world @ marker_to_suitcase
            _set_free_pose(model, data, "pelvis_root", world_torso)
            _set_free_pose(model, data, "suitcase_root", world_suitcase)
            mujoco.mj_forward(model, data)
            viewer.cam.lookat[:] = world_torso[:3, 3]
            viewer.sync()
            sleep_for = period - (time.monotonic() - last_render)
            if sleep_for > 0:
                time.sleep(sleep_for)
            last_render = time.monotonic()
    except KeyboardInterrupt:
        return 0
    finally:
        viewer.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
