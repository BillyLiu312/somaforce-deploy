#!/usr/bin/env python3
"""Live MuJoCo view using the deployed multi-marker pose contract.

This diagnostic uses real robot/suitcase marker poses. Joint angles are
initialized from the suitcase reference frame because the G1 low-state link is
not required by this marker-only viewer.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import zmq
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.ros2_pose_to_zmq import _pose_matrix_from_zmq_values, _wxyz_from_rotation, pose_stamped_to_zmq_values
from sim2real.utils.mjviser_viewer import MjviserMujocoViewer
from sim2real.config.robots.g1 import G1_CFG
from sim2real.utils.common import LowStateMessage
from somaforce_deploy.mocap_fusion import MultiMarkerPoseFusion, load_calibration_payload, load_marker_sources
from somaforce_deploy.torso2pelvis import TorsoToPelvisFK, torso_from_pelvis


def _set_free_pose(model: Any, data: Any, joint_name: str, pose: np.ndarray) -> None:
    joint = model.joint(joint_name)
    address = int(model.jnt_qposadr[joint.id])
    data.qpos[address : address + 3] = pose[:3, 3]
    data.qpos[address + 3 : address + 7] = _wxyz_from_rotation(pose[:3, :3])


def _initialize_reference_pose(model: Any, data: Any, motion_path: Path, meta_path: Path) -> np.ndarray:
    motion = np.load(motion_path, allow_pickle=False)
    meta = json.loads(meta_path.read_text())
    model_joints = {model.joint(i).name: i for i in range(model.njnt)}
    for joint_name, value in zip(meta["joint_names"], motion["joint_pos"][0], strict=True):
        if joint_name in model_joints:
            joint = model_joints[joint_name]
            data.qpos[int(model.jnt_qposadr[joint])] = float(value)
    waist_names = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
    return np.asarray(
        [float(data.qpos[int(model.jnt_qposadr[model_joints[name]])]) for name in waist_names],
        dtype=np.float64,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=REPO_ROOT / "calibration/marker_policy.json")
    parser.add_argument("--scene", type=Path, default=REPO_ROOT.parent / "sim2real-hdmi-upstream/data/robots/g1/g1_29dof_rubberhand-suitcase.xml")
    parser.add_argument("--motion", type=Path, default=REPO_ROOT / "assets/mujoco/reference/hdmi_suitcase/motion.npz")
    parser.add_argument("--motion-meta", type=Path, default=REPO_ROOT / "assets/mujoco/reference/hdmi_suitcase/meta.json")
    parser.add_argument("--position-scale", type=float, default=0.001)
    parser.add_argument("--expected-frame-id", default="world")
    parser.add_argument("--stale-timeout", type=float, default=0.25)
    parser.add_argument("--render-hz", type=float, default=30.0)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--low-state-port", type=int, default=5590)
    args = parser.parse_args()
    if args.position_scale <= 0 or args.stale_timeout <= 0 or args.render_hz <= 0:
        raise ValueError("position-scale, stale-timeout, and render-hz must be positive")

    try:
        import mujoco
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    except ImportError as exc:
        raise RuntimeError("source ROS2 and use the project environment with MuJoCo") from exc

    calibration = load_calibration_payload(args.calibration)
    torso_sources = load_marker_sources(calibration, role="torso", legacy_topic="/robot1/pose")
    suitcase_sources = load_marker_sources(calibration, role="suitcase", legacy_topic="/suitcase1/pose")
    torso_fusion = MultiMarkerPoseFusion(torso_sources, stale_timeout_s=args.stale_timeout)
    suitcase_fusion = MultiMarkerPoseFusion(suitcase_sources, stale_timeout_s=args.stale_timeout)

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)
    reference_waist_angles = _initialize_reference_pose(model, data, args.motion, args.motion_meta)
    fk = TorsoToPelvisFK()
    mujoco.mj_forward(model, data)
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    viewer = MjviserMujocoViewer(
        model, data, label="suitcase-multimarker-live", port=args.port,
        tracked_body_id=int(pelvis_id), camera_distance=3.5,
        camera_azimuth=135.0, camera_elevation=-18.0,
    )

    rclpy.init()
    node = rclpy.create_node("sim2real_suitcase_multimarker_mujoco")
    state_socket = zmq.Context.instance().socket(zmq.SUB)
    state_socket.setsockopt(zmq.SUBSCRIBE, b"")
    state_socket.setsockopt(zmq.CONFLATE, 1)
    state_socket.setsockopt(zmq.LINGER, 0)
    state_socket.connect(f"tcp://127.0.0.1:{args.low_state_port}")
    qos = QoSProfile(depth=30)
    qos.reliability = ReliabilityPolicy.BEST_EFFORT
    qos.durability = DurabilityPolicy.VOLATILE

    def callback(source, fusion):
        def receive(msg: Any) -> None:
            if args.expected_frame_id and str(msg.header.frame_id) != args.expected_frame_id:
                return
            try:
                values = pose_stamped_to_zmq_values(msg, position_scale=args.position_scale)
                fusion.update(source.name, _pose_matrix_from_zmq_values(values), received_at=time.monotonic())
            except ValueError as exc:
                node.get_logger().warning(f"rejecting {source.topic}: {exc}")
        return receive

    for source in torso_sources:
        node.create_subscription(PoseStamped, source.topic, callback(source, torso_fusion), qos)
    for source in suitcase_sources:
        node.create_subscription(PoseStamped, source.topic, callback(source, suitcase_fusion), qos)

    print(
        "multi-marker MuJoCo viewer: "
        f"torso={[source.topic for source in torso_sources]}, "
        f"suitcase={[source.topic for source in suitcase_sources]}, "
        f"reference_waist_angles={reference_waist_angles.tolist()}, "
        f"low_state_port={args.low_state_port}", flush=True,
    )
    period = 1.0 / args.render_hz
    last_render = time.monotonic()
    latest_state: LowStateMessage | None = None
    yaw_alignment: float | None = None
    model_joint_indices = {
        name: int(model.jnt_qposadr[model.joint(name).id])
        for name in G1_CFG.joint_names
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0
    }
    waist_names = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
    try:
        while viewer.is_running() and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            try:
                while True:
                    latest_state = LowStateMessage.from_bytes(
                        state_socket.recv(flags=zmq.DONTWAIT)
                    )
            except zmq.Again:
                pass
            now = time.monotonic()
            torso = torso_fusion.resolve(now=now)
            suitcase = suitcase_fusion.resolve(now=now)
            if torso is None or suitcase is None:
                time.sleep(0.01)
                continue
            if latest_state is not None:
                for index, name in enumerate(G1_CFG.joint_names):
                    if name in model_joint_indices:
                        data.qpos[model_joint_indices[name]] = float(
                            latest_state.joint_positions[index]
                        )
                waist_angles = np.asarray(
                    [
                        latest_state.joint_positions[G1_CFG.joint_names.index(name)]
                        for name in waist_names
                    ],
                    dtype=np.float64,
                )
                base_quat = np.asarray(latest_state.quaternion, dtype=np.float64)
                base_quat /= max(float(np.linalg.norm(base_quat)), 1e-9)
                base_rotation = Rotation.from_quat(
                    [base_quat[1], base_quat[2], base_quat[3], base_quat[0]]
                ).as_matrix()
                torso_relative = torso_from_pelvis(waist_angles)
                predicted_torso_rotation = base_rotation @ torso_relative[:3, :3]
                suitcase_yaw = float(
                    np.arctan2(
                        suitcase.world_from_target[1, 0],
                        suitcase.world_from_target[0, 0],
                    )
                )
                predicted_yaw = float(
                    np.arctan2(predicted_torso_rotation[1, 0], predicted_torso_rotation[0, 0])
                )
                if yaw_alignment is None:
                    yaw_alignment = suitcase_yaw - predicted_yaw
                c, s = np.cos(yaw_alignment), np.sin(yaw_alignment)
                world_alignment = np.asarray(
                    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                    dtype=np.float64,
                )
                torso_pose = torso.world_from_target.copy()
                torso_pose[:3, :3] = world_alignment @ predicted_torso_rotation
            else:
                waist_angles = reference_waist_angles
                torso_pose = torso.world_from_target
            pelvis_world = fk.pelvis_from_torso(torso_pose, waist_angles)
            _set_free_pose(model, data, "pelvis_root", pelvis_world)
            _set_free_pose(model, data, "suitcase_root", suitcase.world_from_target)
            mujoco.mj_forward(model, data)
            viewer.sync()
            sleep_for = period - (time.monotonic() - last_render)
            if sleep_for > 0:
                time.sleep(sleep_for)
            last_render = time.monotonic()
    except KeyboardInterrupt:
        return 0
    finally:
        viewer.close()
        state_socket.close(linger=0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
