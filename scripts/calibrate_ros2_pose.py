#!/usr/bin/env python3
"""Solve a common calibration-world transform from live ROS2 poses.

The tool accepts fixed marker-frame-to-policy-frame transforms measured from
CAD or the mocap rigid-body definition. It then estimates one common
``world_from_mocap`` transform. A common transform cancels from the policy's
pelvis-relative observations; a large two-body residual means the measured
marker transforms or the physical target placement are wrong.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np


def _pose_matrix(position: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion_xyzw / np.linalg.norm(quaternion_xyzw)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    matrix[:3, 3] = position
    return matrix


def _yaw_matrix(yaw_rad: float, position: np.ndarray) -> np.ndarray:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    matrix[:3, 3] = position
    return matrix


def _rotation_angle(matrix: np.ndarray) -> float:
    value = np.clip((np.trace(matrix[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
    return float(math.acos(value))


def _yaw(rotation: np.ndarray) -> float:
    return float(math.atan2(rotation[1, 0], rotation[0, 0]))


def _wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _mean_transform(transforms: list[np.ndarray]) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = np.mean([transform[:3, 3] for transform in transforms], axis=0)
    result[:3, :3] = Rotation.from_matrix(np.stack([transform[:3, :3] for transform in transforms])).mean().as_matrix()
    return result


def _load_matrix(path: str | None, role: str) -> np.ndarray:
    if path is None:
        return np.eye(4, dtype=np.float64)
    payload = json.loads(Path(path).read_text())
    raw = payload[role] if role in payload else payload
    matrix = np.asarray(raw, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{role} marker-to-policy transform must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{role} transform last row must be [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
        raise ValueError(f"{role} transform rotation is not a proper rigid rotation")
    return matrix


def _solve(
    samples: list[dict[str, np.ndarray]],
    target_pelvis: np.ndarray,
    target_suitcase: np.ndarray,
    marker_to_policy: dict[str, np.ndarray],
) -> dict[str, Any]:
    pelvis_policy_samples: list[np.ndarray] = []
    suitcase_policy_samples: list[np.ndarray] = []
    for sample in samples:
        pelvis = sample["pelvis"]
        suitcase = sample["suitcase"]
        pelvis_policy = pelvis @ marker_to_policy["pelvis"]
        suitcase_policy = suitcase @ marker_to_policy["suitcase"]
        pelvis_policy_samples.append(pelvis_policy)
        suitcase_policy_samples.append(suitcase_policy)

    mean_pelvis = _mean_transform(pelvis_policy_samples)
    mean_suitcase = _mean_transform(suitcase_policy_samples)
    pelvis_yaw = _yaw(mean_pelvis[:3, :3])
    transform = _yaw_matrix(-pelvis_yaw, np.zeros(3, dtype=np.float64))
    transform[:3, 3] = -transform[:3, :3] @ mean_pelvis[:3, 3]

    calibrated_pelvis = transform @ mean_pelvis
    calibrated_suitcase = transform @ mean_suitcase
    relative_position = calibrated_suitcase[:3, 3] - calibrated_pelvis[:3, 3]
    relative_yaw = _wrap_angle(
        _yaw(calibrated_suitcase[:3, :3]) - _yaw(calibrated_pelvis[:3, :3])
    )
    target_relative = np.linalg.inv(target_pelvis) @ target_suitcase
    target_relative_position = target_relative[:3, 3]
    target_relative_yaw = _yaw(target_relative[:3, :3])
    position_residual = relative_position - target_relative_position
    yaw_residual = _wrap_angle(relative_yaw - target_relative_yaw)

    pelvis_positions = np.stack([pose[:3, 3] for pose in pelvis_policy_samples])
    suitcase_positions = np.stack([pose[:3, 3] for pose in suitcase_policy_samples])
    pelvis_yaws = np.unwrap([_yaw(pose[:3, :3]) for pose in pelvis_policy_samples])
    suitcase_yaws = np.unwrap([_yaw(pose[:3, :3]) for pose in suitcase_policy_samples])
    return {
        "world_from_mocap": transform.tolist(),
        "samples": len(samples),
        "stationary_noise": {
            "pelvis_position_std_m": np.std(pelvis_positions, axis=0).tolist(),
            "suitcase_position_std_m": np.std(suitcase_positions, axis=0).tolist(),
            "pelvis_yaw_std_deg": float(np.degrees(np.std(pelvis_yaws))),
            "suitcase_yaw_std_deg": float(np.degrees(np.std(suitcase_yaws))),
        },
        "calibrated_pelvis_pose": calibrated_pelvis.tolist(),
        "calibrated_suitcase_pose": calibrated_suitcase.tolist(),
        "relative_position_m": relative_position.tolist(),
        "relative_yaw_deg": float(np.degrees(relative_yaw)),
        "target_relative_pose_pelvis_to_suitcase": target_relative.tolist(),
        "relative_residual_xyz_m": position_residual.tolist(),
        "relative_residual_translation_m": float(np.linalg.norm(position_residual)),
        "relative_residual_rotation_deg": float(abs(np.degrees(yaw_residual))),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suitcase-topic", default="/suitcase/pose")
    parser.add_argument("--pelvis-topic", default="/robot_g1/pose")
    parser.add_argument("--position-scale", type=float, default=0.001)
    parser.add_argument("--expected-frame-id", default="world")
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--relative-position", type=float, nargs=3, default=[0.532, 0.0, -0.793])
    parser.add_argument("--relative-yaw-deg", type=float, default=0.0)
    parser.add_argument(
        "--marker-to-policy-json",
        type=Path,
        default=None,
        help="JSON with pelvis and suitcase 4x4 marker_to_policy matrices",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.samples < 2 or args.timeout <= 0 or args.position_scale <= 0:
        raise ValueError("samples must be >= 2, timeout and position-scale must be positive")
    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    except ImportError as exc:
        raise RuntimeError("source ROS2 before running this calibration tool") from exc

    rclpy.init()
    node = rclpy.create_node("sim2real_pose_calibrator")
    qos = QoSProfile(depth=10)
    qos.reliability = ReliabilityPolicy.BEST_EFFORT
    qos.durability = DurabilityPolicy.VOLATILE
    latest: dict[str, tuple[int, np.ndarray]] = {}
    samples: list[dict[str, np.ndarray]] = []
    last_pair_stamp: tuple[int, int] | None = None

    def callback(name: str):
        def receive(msg: Any) -> None:
            frame_id = str(msg.header.frame_id)
            if args.expected_frame_id and frame_id != args.expected_frame_id:
                node.get_logger().error(f"{name} frame_id={frame_id!r}, expected {args.expected_frame_id!r}")
                return
            p = msg.pose.position
            o = msg.pose.orientation
            stamp = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
            position = np.asarray([p.x, p.y, p.z], dtype=np.float64) * args.position_scale
            quaternion = np.asarray([o.x, o.y, o.z, o.w], dtype=np.float64)
            if not np.isfinite(position).all() or not np.isfinite(quaternion).all() or np.linalg.norm(quaternion) < 1e-9:
                return
            latest[name] = (stamp, _pose_matrix(position, quaternion))
        return receive

    node.create_subscription(PoseStamped, args.pelvis_topic, callback("pelvis"), qos)
    node.create_subscription(PoseStamped, args.suitcase_topic, callback("suitcase"), qos)
    deadline = time.monotonic() + args.timeout
    while rclpy.ok() and time.monotonic() < deadline and len(samples) < args.samples:
        rclpy.spin_once(node, timeout_sec=0.05)
        if "pelvis" not in latest or "suitcase" not in latest:
            continue
        pair_stamp = (latest["pelvis"][0], latest["suitcase"][0])
        if pair_stamp == last_pair_stamp:
            continue
        last_pair_stamp = pair_stamp
        samples.append({"pelvis": latest["pelvis"][1].copy(), "suitcase": latest["suitcase"][1].copy()})
    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    if len(samples) < args.samples:
        raise RuntimeError(f"collected {len(samples)} paired samples, expected {args.samples}")

    target_pelvis = np.eye(4, dtype=np.float64)
    target_suitcase = _yaw_matrix(math.radians(args.relative_yaw_deg), np.asarray(args.relative_position, dtype=np.float64))
    marker_to_policy = {
        "pelvis": _load_matrix(str(args.marker_to_policy_json) if args.marker_to_policy_json else None, "pelvis"),
        "suitcase": _load_matrix(str(args.marker_to_policy_json) if args.marker_to_policy_json else None, "suitcase"),
    }
    result = _solve(samples, target_pelvis, target_suitcase, marker_to_policy)
    result.update(
        {
            "schema": "sim2real_ros2_pose_calibration_v1",
            "suitcase_topic": args.suitcase_topic,
            "pelvis_topic": args.pelvis_topic,
            "position_scale": args.position_scale,
            "expected_frame_id": args.expected_frame_id,
            "target_definition": {
                "pelvis_pose": "identity calibration frame",
                "suitcase_relative_position_m": list(args.relative_position),
                "suitcase_relative_yaw_deg": args.relative_yaw_deg,
            },
            "marker_to_policy": {role: matrix.tolist() for role, matrix in marker_to_policy.items()},
            "assumption": "marker_to_policy matrices describe fixed rigid-body geometry",
        }
    )
    result["valid"] = not (
        result["relative_residual_translation_m"] > 0.03
        or result["relative_residual_rotation_deg"] > 5.0
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        print("WARNING: relative residual is high; fix rigid-body origins/axes or physical placement before policy startup")
        return 2
    print(f"saved calibration: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
