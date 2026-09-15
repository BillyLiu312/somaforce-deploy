#!/usr/bin/env python3
"""Read-only live mocap + G1 waist state check and static MuJoCo render.

This diagnostic does not create a low-command publisher. It reads ROS2
PoseStamped topics and the Unitree HG ``rt/lowstate`` topic, computes the same
torso-to-pelvis FK used by deployment, then places the resulting policy frames
in a static MuJoCo scene and renders one PNG.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sim2real.config.robots.g1 import G1_CFG
from somaforce_deploy.torso2pelvis import TorsoToPelvisFK, make_transform
from scripts.ros2_pose_to_zmq import pose_stamped_to_zmq_values, _pose_matrix_from_zmq_values, _wxyz_from_rotation


def _load_matrix(path: Path | None, key: str) -> np.ndarray:
    if path is None:
        return np.eye(4, dtype=np.float64)
    payload = json.loads(path.read_text())
    raw = payload.get(key, payload.get("matrix", payload))
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{key} must be a 4x4 matrix")
    return matrix


def _body_pose(model: Any, data: Any, name: str) -> np.ndarray:
    body = model.body(name)
    return make_transform(data.xmat[body.id].reshape(3, 3), data.xpos[body.id])


def _set_free_pose(model: Any, data: Any, joint_name: str, pose: np.ndarray) -> None:
    joint = model.joint(joint_name)
    address = int(model.jnt_qposadr[joint.id])
    data.qpos[address : address + 3] = pose[:3, 3]
    data.qpos[address + 3 : address + 7] = _wxyz_from_rotation(pose[:3, :3])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torso-topic", default="/robot_g1/pose")
    parser.add_argument("--suitcase-topic", default="/suitcase/pose")
    parser.add_argument("--interface", default="eno1")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--position-scale", type=float, default=0.001)
    parser.add_argument("--delta-p", type=float, nargs=3, default=[-0.69, -0.02, 0.51])
    parser.add_argument("--suitcase-marker-in-root", type=float, nargs=3, default=[0.0, 0.0225, 0.4])
    parser.add_argument("--scene", type=Path, default=REPO_ROOT.parent / "sim2real-hdmi-upstream/data/robots/g1/g1_29dof_rubberhand-suitcase.xml")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs/live_mocap_static/mocap_static.png")
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "outputs/live_mocap_static/report.json")
    parser.add_argument("--expected-frame-id", default="world")
    args = parser.parse_args()

    try:
        import mujoco
        import imageio.v3 as imageio
        import rclpy
        import zmq
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    except ImportError as exc:
        raise RuntimeError("root dependencies and ROS2/Unitree SDK imports are required") from exc

    ChannelFactoryInitialize(args.domain_id, args.interface)
    state_sub = ChannelSubscriber("rt/lowstate", LowState_)
    state_sub.Init(handler=None, queueLen=1)

    rclpy.init()
    node = rclpy.create_node("sim2real_live_mocap_static")
    qos = QoSProfile(depth=30)
    qos.reliability = ReliabilityPolicy.BEST_EFFORT
    qos.durability = DurabilityPolicy.VOLATILE
    latest: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    states: list[Any] = []

    def pose_callback(name: str):
        def callback(msg: Any) -> None:
            if args.expected_frame_id and str(msg.header.frame_id) != args.expected_frame_id:
                return
            values = pose_stamped_to_zmq_values(msg, position_scale=args.position_scale)
            latest[name] = (_pose_matrix_from_zmq_values(values), values, str(msg.header.frame_id))
        return callback

    node.create_subscription(PoseStamped, args.torso_topic, pose_callback("torso"), qos)
    node.create_subscription(PoseStamped, args.suitcase_topic, pose_callback("suitcase"), qos)
    deadline = time.monotonic() + args.timeout
    while rclpy.ok() and time.monotonic() < deadline and len(states) < args.samples:
        rclpy.spin_once(node, timeout_sec=0.01)
        msg = state_sub.Read()
        if msg is not None:
            states.append(msg)
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    if not all(name in latest for name in ("torso", "suitcase")):
        raise RuntimeError(f"missing live pose(s): expected torso and suitcase, got {sorted(latest)}")
    if not states:
        raise RuntimeError("no rt/lowstate sample received")

    torso_world_marker = latest["torso"][0]
    suitcase_world_marker = latest["suitcase"][0]
    torso_marker_quat = latest["torso"][1][3:]
    suitcase_marker_quat = latest["suitcase"][1][3:]
    torso_marker_rotation = _pose_matrix_from_zmq_values(latest["torso"][1])[:3, :3]
    suitcase_marker_rotation = _pose_matrix_from_zmq_values(latest["suitcase"][1])[:3, :3]

    # The marker centers are in mocap/world coordinates. delta-p is the measured
    # torso-frame origin relative to the suitcase marker center.
    marker_relative = torso_world_marker[:3, 3] - suitcase_world_marker[:3, 3]
    torso_origin_minus_marker = np.asarray(args.delta_p, dtype=np.float64) - marker_relative
    torso_from_marker = make_transform(torso_marker_rotation, -torso_origin_minus_marker)
    suitcase_marker_from_policy = make_transform(
        suitcase_marker_rotation.T,
        -suitcase_marker_rotation.T @ np.asarray(args.suitcase_marker_in_root, dtype=np.float64),
    )

    # Low-state joint positions are in the G1 canonical order used by the bridge.
    state = states[-1]
    waist_indices = [G1_CFG.joint_names.index(name) for name in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")]
    waist_angles = np.asarray(state.motor_state, dtype=object)
    waist_angles = np.asarray([float(waist_angles[index].q) for index in waist_indices], dtype=np.float64)
    fk = TorsoToPelvisFK(torso_from_marker)
    pelvis_world = fk.pelvis_from_marker(torso_world_marker, waist_angles)
    suitcase_world = suitcase_world_marker @ suitcase_marker_from_policy

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    _set_free_pose(model, data, "pelvis_root", pelvis_world)
    _set_free_pose(model, data, "suitcase_root", suitcase_world)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=720, width=960)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.azimuth = 135
    camera.elevation = -18
    camera.distance = 3.5
    camera.lookat[:] = pelvis_world[:3, 3]
    renderer.update_scene(data, camera=camera)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(args.output, renderer.render())

    report = {
        "samples": len(states),
        "torso_topic": args.torso_topic,
        "suitcase_topic": args.suitcase_topic,
        "position_scale": args.position_scale,
        "delta_p_torso_minus_suitcase": list(args.delta_p),
        "marker_relative_torso_minus_suitcase": marker_relative.tolist(),
        "torso_origin_minus_marker_world": torso_origin_minus_marker.tolist(),
        "waist_angles_rad_yaw_roll_pitch": waist_angles.tolist(),
        "torso_from_marker_T_T_M": torso_from_marker.tolist(),
        "suitcase_marker_from_policy_T_Ms_S": suitcase_marker_from_policy.tolist(),
        "pelvis_world_pose": pelvis_world.tolist(),
        "suitcase_world_pose": suitcase_world.tolist(),
        "output": str(args.output),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"saved static MuJoCo render: {args.output}")
    print(f"saved report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
