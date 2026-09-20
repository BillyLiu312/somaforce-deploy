#!/usr/bin/env python3
"""Quickly calibrate the G1 robot marker set from suitcase motion frame 0."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.ros2_pose_to_zmq import (
    _pose_matrix_from_zmq_values,
    pose_stamped_to_zmq_values,
)
from somaforce_deploy.mocap_fusion import (
    MultiMarkerPoseFusion,
    load_calibration_payload,
    load_marker_sources,
)
from somaforce_deploy.torso2pelvis import validate_transform

DEFAULT_CALIBRATION = REPO_ROOT / "calibration/marker_policy.json"
DEFAULT_MOTION = REPO_ROOT / "assets/mujoco/reference/hdmi_suitcase/motion.npz"
DEFAULT_MOTION_META = REPO_ROOT / "assets/mujoco/reference/hdmi_suitcase/meta.json"
REQUIRED_ROBOT_MARKER_NAMES = ("robot1", "robot2", "robot3")


def _mean_transform(transforms: list[np.ndarray]) -> np.ndarray:
    if not transforms:
        raise ValueError("cannot average an empty transform sequence")
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = np.mean([value[:3, 3] for value in transforms], axis=0)
    result[:3, :3] = (
        Rotation.from_matrix(np.stack([value[:3, :3] for value in transforms]))
        .mean()
        .as_matrix()
    )
    return result


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def _wxyz_transform(position: Any, quaternion: Any) -> np.ndarray:
    position = np.asarray(position, dtype=np.float64)
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("motion body pose must contain xyz and wxyz")
    if not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise ValueError("motion body pose contains non-finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-8:
        raise ValueError("motion body quaternion has zero norm")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = position
    transform[:3, :3] = Rotation.from_quat(
        (quaternion / norm)[[1, 2, 3, 0]]
    ).as_matrix()
    return transform


def load_frame_zero_suitcase_from_torso(
    motion_path: Path,
    metadata_path: Path,
    *,
    suitcase_body: str = "suitcase",
    torso_body: str = "torso_link",
) -> np.ndarray:
    metadata = json.loads(metadata_path.read_text())
    body_names = [str(name) for name in metadata.get("body_names", [])]
    if len(body_names) != len(set(body_names)):
        raise ValueError("motion metadata contains duplicate body names")
    missing = [name for name in (suitcase_body, torso_body) if name not in body_names]
    if missing:
        raise ValueError(f"motion metadata is missing bodies: {missing}")
    with np.load(motion_path) as motion:
        positions = np.asarray(motion["body_pos_w"])
        quaternions = np.asarray(motion["body_quat_w"])
        if (
            positions.ndim != 3
            or quaternions.ndim != 3
            or positions.shape[:2] != quaternions.shape[:2]
            or positions.shape[0] == 0
            or positions.shape[1] != len(body_names)
            or positions.shape[2] != 3
            or quaternions.shape[2] != 4
        ):
            raise ValueError("motion body pose arrays do not match metadata")
        suitcase_index = body_names.index(suitcase_body)
        torso_index = body_names.index(torso_body)
        world_from_suitcase = _wxyz_transform(
            positions[0, suitcase_index], quaternions[0, suitcase_index]
        )
        world_from_torso = _wxyz_transform(
            positions[0, torso_index], quaternions[0, torso_index]
        )
    return validate_transform(
        np.linalg.inv(world_from_suitcase) @ world_from_torso,
        name="motion_frame_zero.suitcase_from_torso",
    )


def estimate_marker_transform(
    transforms: list[np.ndarray],
) -> tuple[np.ndarray, dict[str, float]]:
    mean = _mean_transform(transforms)
    position_errors = np.asarray(
        [np.linalg.norm(value[:3, 3] - mean[:3, 3]) for value in transforms]
    )
    orientation_errors = np.asarray(
        [_rotation_error_deg(mean, value) for value in transforms]
    )
    return mean, {
        "samples": len(transforms),
        "position_error_m_p95": float(np.percentile(position_errors, 95)),
        "position_error_m_max": float(np.max(position_errors)),
        "orientation_error_deg_p95": float(np.percentile(orientation_errors, 95)),
        "orientation_error_deg_max": float(np.max(orientation_errors)),
    }


def validate_robot_marker_names(names: list[str]) -> None:
    expected = set(REQUIRED_ROBOT_MARKER_NAMES)
    actual = set(names)
    if actual != expected or len(names) != len(expected):
        raise ValueError(
            "quick calibration requires exactly robot1, robot2, and robot3: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def update_calibration_payload(
    payload: dict[str, Any],
    transforms: dict[str, np.ndarray],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    result = copy.deepcopy(payload)
    marker_sources = result.get("marker_sources")
    if not isinstance(marker_sources, dict):
        raise TypeError("quick calibration requires marker_sources in calibration")
    torso_sources = marker_sources.get("torso")
    if not isinstance(torso_sources, list) or not torso_sources:
        raise ValueError("quick calibration requires marker_sources.torso")
    source_names = [str(source.get("name", "")) for source in torso_sources]
    if set(source_names) != set(transforms):
        raise ValueError(
            "calibrated transforms do not exactly cover marker_sources.torso"
        )
    for source in torso_sources:
        name = str(source["name"])
        source["marker_from_target"] = validate_transform(
            transforms[name], name=f"marker_from_target[{name}]"
        ).tolist()
    # Keep the legacy single-marker field aligned with the first v2 source.
    result["marker_from_torso"] = transforms[source_names[0]].tolist()
    calibration = result.setdefault("calibration", {})
    if not isinstance(calibration, dict):
        raise TypeError("calibration metadata must be an object")
    calibration["robot_marker_quick_calibration"] = metadata
    result["valid"] = True
    return result


def write_calibration_atomic(path: Path, payload: dict[str, Any]) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    backup = path.with_name(f"{path.name}.before-robot-marker-{timestamp}")
    shutil.copy2(path, backup)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            json.dump(payload, temporary, indent=2)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return backup


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--motion-meta", type=Path, default=DEFAULT_MOTION_META)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--timeout", type=float, default=12.0)
    parser.add_argument("--position-scale", type=float, default=0.001)
    parser.add_argument("--expected-frame-id", default="world")
    parser.add_argument("--stale-timeout", type=float, default=0.10)
    parser.add_argument("--synchronization-window-s", type=float, default=0.05)
    parser.add_argument("--max-position-error-m", type=float, default=0.01)
    parser.add_argument("--max-orientation-error-deg", type=float, default=2.0)
    parser.add_argument("--suitcase-body", default="suitcase")
    parser.add_argument("--torso-body", default="torso_link")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    positive = (
        args.samples,
        args.timeout,
        args.position_scale,
        args.stale_timeout,
        args.synchronization_window_s,
        args.max_position_error_m,
        args.max_orientation_error_deg,
    )
    if any(value <= 0 for value in positive):
        raise ValueError(
            "sample count, timing, scale, and residual limits must be positive"
        )

    payload = load_calibration_payload(args.calibration)
    robot_sources = load_marker_sources(
        payload, role="torso", legacy_topic="/robot1/pose"
    )
    validate_robot_marker_names([source.name for source in robot_sources])
    suitcase_sources = load_marker_sources(
        payload, role="suitcase", legacy_topic="/suitcase1/pose"
    )
    suitcase_from_torso = load_frame_zero_suitcase_from_torso(
        args.motion,
        args.motion_meta,
        suitcase_body=args.suitcase_body,
        torso_body=args.torso_body,
    )

    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    except ImportError as exc:
        raise RuntimeError("source ROS2 and use the project virtualenv") from exc

    rclpy.init()
    node = rclpy.create_node("sim2real_quick_robot_marker_calibrator")
    qos = QoSProfile(depth=50)
    qos.reliability = ReliabilityPolicy.BEST_EFFORT
    qos.durability = DurabilityPolicy.VOLATILE
    suitcase_fusion = MultiMarkerPoseFusion(
        suitcase_sources,
        stale_timeout_s=args.stale_timeout,
        synchronization_window_s=args.synchronization_window_s,
    )
    latest_robot: dict[str, tuple[int, float, np.ndarray]] = {}
    latest_suitcase: tuple[float, np.ndarray, tuple[str, ...]] | None = None
    last_robot_stamp: dict[str, int] = {}
    samples: dict[str, list[np.ndarray]] = {source.name: [] for source in robot_sources}
    suitcase_sources_used: set[str] = set()
    received_counts = {
        source.name: 0 for source in tuple(robot_sources) + tuple(suitcase_sources)
    }
    rejected_counts = {name: 0 for name in received_counts}
    suitcase_fused_count = 0

    def parse_message(msg: Any) -> tuple[int, float, np.ndarray]:
        if (
            args.expected_frame_id
            and str(msg.header.frame_id) != args.expected_frame_id
        ):
            raise ValueError(
                f"frame_id={msg.header.frame_id!r}, expected {args.expected_frame_id!r}"
            )
        values = pose_stamped_to_zmq_values(msg, position_scale=args.position_scale)
        stamp = int(msg.header.stamp.sec) * 1_000_000_000 + int(
            msg.header.stamp.nanosec
        )
        return stamp, time.monotonic(), _pose_matrix_from_zmq_values(values)

    def robot_callback(source):
        def receive(msg: Any) -> None:
            received_counts[source.name] += 1
            try:
                latest_robot[source.name] = parse_message(msg)
            except ValueError as exc:
                rejected_counts[source.name] += 1
                node.get_logger().warning(f"rejecting {source.name}: {exc}")

        return receive

    def suitcase_callback(source):
        def receive(msg: Any) -> None:
            nonlocal latest_suitcase, suitcase_fused_count
            received_counts[source.name] += 1
            try:
                _, received_at, world_from_marker = parse_message(msg)
                suitcase_fusion.update(
                    source.name, world_from_marker, received_at=received_at
                )
                fused = suitcase_fusion.resolve(now=received_at)
                if fused is not None:
                    suitcase_fused_count += 1
                    latest_suitcase = (
                        received_at,
                        fused.world_from_target,
                        fused.used_sources,
                    )
            except ValueError as exc:
                rejected_counts[source.name] += 1
                node.get_logger().warning(f"rejecting {source.name}: {exc}")

        return receive

    for source in robot_sources:
        node.create_subscription(PoseStamped, source.topic, robot_callback(source), qos)
    for source in suitcase_sources:
        node.create_subscription(
            PoseStamped, source.topic, suitcase_callback(source), qos
        )

    print(
        "Quick calibration contract: keep G1 and suitcase stationary in the exact "
        "motion frame-0 relationship; collecting live marker samples.",
        flush=True,
    )
    deadline = time.monotonic() + args.timeout
    last_progress = time.monotonic()
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            if all(len(values) >= args.samples for values in samples.values()):
                break
            rclpy.spin_once(node, timeout_sec=0.01)
            now = time.monotonic()
            if latest_suitcase is None:
                continue
            suitcase_time, world_from_suitcase, used_sources = latest_suitcase
            if now - suitcase_time > args.stale_timeout:
                continue
            # Anchor the known frame-0 suitcase-to-torso relationship at the
            # suitcase pose measured in the current mocap world.
            world_from_torso = world_from_suitcase @ suitcase_from_torso
            for source in robot_sources:
                if len(samples[source.name]) >= args.samples:
                    continue
                observation = latest_robot.get(source.name)
                if observation is None:
                    continue
                stamp, received_at, world_from_marker = observation
                if (
                    now - received_at > args.stale_timeout
                    or abs(received_at - suitcase_time) > args.synchronization_window_s
                    or last_robot_stamp.get(source.name) == stamp
                ):
                    continue
                last_robot_stamp[source.name] = stamp
                samples[source.name].append(
                    np.linalg.inv(world_from_marker) @ world_from_torso
                )
                suitcase_sources_used.update(used_sources)
            if now - last_progress >= 1.0:
                print(
                    "calibration sampling: "
                    f"received={received_counts}, rejected={rejected_counts}, "
                    f"suitcase_fused={suitcase_fused_count}, "
                    f"samples={{{', '.join(f'{name!r}: {len(values)}' for name, values in samples.items())}}}",
                    flush=True,
                )
                last_progress = now
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    counts = {name: len(values) for name, values in samples.items()}
    incomplete = [name for name, count in counts.items() if count < args.samples]
    if incomplete:
        raise RuntimeError(
            "all robot1/robot2/robot3 markers are required for direct calibration: "
            f"incomplete={incomplete}, samples={counts}, received={received_counts}, "
            f"rejected={rejected_counts}, suitcase_fused={suitcase_fused_count}. "
            "Restore visibility of all three robot markers and the live "
            "VRPN/NOKOV stream before retrying."
        )

    transforms: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict[str, Any]] = {}
    unstable: list[str] = []
    for source in robot_sources:
        values = samples[source.name]
        transform, source_diagnostics = estimate_marker_transform(values)
        diagnostics[source.name] = source_diagnostics
        if (
            source_diagnostics["position_error_m_p95"] > args.max_position_error_m
            or source_diagnostics["orientation_error_deg_p95"]
            > args.max_orientation_error_deg
        ):
            unstable.append(source.name)
        transforms[source.name] = transform
    if unstable:
        raise RuntimeError(f"unstable directly observed robot markers: {unstable}")

    observed_names = sorted(transforms)
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reference": "motion_frame_0_suitcase_from_torso_link",
        "calibration_mode": "all_robot_markers_direct",
        "motion": str(args.motion),
        "motion_meta": str(args.motion_meta),
        "suitcase_body": args.suitcase_body,
        "torso_body": args.torso_body,
        "suitcase_from_torso": suitcase_from_torso.tolist(),
        "observed_sources": observed_names,
        "inferred_sources": [],
        "suitcase_sources_used": sorted(suitcase_sources_used),
        "missing_source_method": "disabled; all robot markers are required",
        "diagnostics": diagnostics,
    }
    updated = update_calibration_payload(payload, transforms, metadata=metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    if args.dry_run:
        print("Dry run complete; calibration file was not changed.", flush=True)
        return 0
    backup = write_calibration_atomic(args.calibration, updated)
    print(f"Updated calibration: {args.calibration}", flush=True)
    print(f"Backup: {backup}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
