#!/usr/bin/env python3
"""Refresh persistent torso marker-frame corrections from a known setup pose."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.ros2_pose_to_zmq import (  # noqa: E402
    _pose_matrix_from_zmq_values,
    pose_stamped_to_zmq_values,
)
from somaforce_deploy.mocap_fusion import (  # noqa: E402
    load_calibration_payload,
    load_marker_sources,
)


def _mean_transform(transforms: list[np.ndarray]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = np.mean([value[:3, 3] for value in transforms], axis=0)
    result[:3, :3] = Rotation.from_matrix(
        np.stack([value[:3, :3] for value in transforms])
    ).mean().as_matrix()
    return result


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def _largest_consensus(
    poses: dict[str, np.ndarray], *, position_m: float, orientation_deg: float
) -> tuple[str, ...]:
    names = tuple(poses)
    for size in range(len(names), 1, -1):
        valid = []
        for group in combinations(names, size):
            if all(
                np.linalg.norm(poses[a][:3, 3] - poses[b][:3, 3]) <= position_m
                and _rotation_error_deg(poses[a], poses[b]) <= orientation_deg
                for a, b in combinations(group, 2)
            ):
                valid.append(group)
        if len(valid) == 1:
            return valid[0]
        if len(valid) > 1:
            raise RuntimeError(f"ambiguous torso consensus groups: {valid}")
    raise RuntimeError("no two corrected torso marker sources agree")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--position-scale", type=float, default=0.001)
    parser.add_argument("--expected-frame-id", default="world")
    parser.add_argument("--minimum-suitcase-sources", type=int, default=2)
    parser.add_argument("--minimum-torso-sources", type=int, default=2)
    parser.add_argument(
        "--reference-mode",
        choices=("consensus", "suitcase"),
        default=None,
        help="consensus allows arbitrary suitcase placement; suitcase uses the recorded setup pose",
    )
    parser.add_argument("--consensus-position-m", type=float, default=0.08)
    parser.add_argument("--consensus-orientation-deg", type=float, default=12.0)
    parser.add_argument("--max-fit-position-std-mm", type=float, default=5.0)
    parser.add_argument("--max-fit-orientation-deg", type=float, default=1.0)
    args = parser.parse_args()
    if args.samples < 20 or args.timeout <= 0 or args.position_scale <= 0:
        raise ValueError("samples, timeout, and position-scale must be positive")

    calibration = load_calibration_payload(args.calibration)
    current = load_calibration_payload(args.corrections)
    reference = current.get("reference", {})
    reference_mode = args.reference_mode or str(
        reference.get("refresh_mode", "consensus")
    )
    if reference_mode not in {"consensus", "suitcase"}:
        raise ValueError(f"unsupported correction refresh mode: {reference_mode}")
    suitcase_sources = load_marker_sources(
        calibration, role="suitcase", legacy_topic="/suitcase1/pose"
    )
    torso_sources = load_marker_sources(
        calibration, role="torso", legacy_topic="/robot1/pose"
    )
    # Refresh always starts from the immutable base calibration. Existing
    # corrections are deliberately ignored so a remounted rigid body cannot
    # be double-corrected.
    effective_torso_sources = load_marker_sources(
        calibration,
        role="torso",
        legacy_topic="/robot1/pose",
        corrections=None,
    )
    effective_by_name = {source.name: source for source in effective_torso_sources}
    position = np.asarray(
        reference.get("torso_position_in_suitcase_m", (-0.67, -0.3675, 0.86)),
        dtype=np.float64,
    )
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("reference torso position must contain three finite metres")
    if reference.get("torso_rotation_in_suitcase", "identity") != "identity":
        raise ValueError("only identity torso rotation reference is currently supported")
    suitcase_from_torso = np.eye(4, dtype=np.float64)
    suitcase_from_torso[:3, 3] = position

    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    except ImportError as exc:
        raise RuntimeError("source ROS2 and use the project virtualenv") from exc

    rclpy.init()
    node = rclpy.create_node("sim2real_marker_frame_correction_refresh")
    qos = QoSProfile(depth=50)
    qos.reliability = ReliabilityPolicy.BEST_EFFORT
    qos.durability = DurabilityPolicy.VOLATILE
    latest: dict[str, tuple[float, np.ndarray]] = {}
    samples: dict[str, list[np.ndarray]] = {
        source.name: [] for source in torso_sources
    }
    last_sample = 0.0

    def callback(source):
        def receive(msg: Any) -> None:
            if args.expected_frame_id and str(msg.header.frame_id) != args.expected_frame_id:
                return
            try:
                values = pose_stamped_to_zmq_values(
                    msg, position_scale=args.position_scale
                )
                latest[source.name] = (
                    time.monotonic(), _pose_matrix_from_zmq_values(values)
                )
            except ValueError:
                return

        return receive

    for source in tuple(suitcase_sources) + tuple(torso_sources):
        node.create_subscription(PoseStamped, source.topic, callback(source), qos)

    deadline = time.monotonic() + args.timeout
    while rclpy.ok() and time.monotonic() < deadline:
        if sum(len(values) >= args.samples for values in samples.values()) >= args.minimum_torso_sources:
            break
        rclpy.spin_once(node, timeout_sec=0.01)
        now = time.monotonic()
        if now - last_sample < 0.01:
            continue
        visible_suitcase = [
            source
            for source in suitcase_sources
            if source.name in latest and now - latest[source.name][0] <= 0.10
        ]
        visible_torso = [
            source
            for source in torso_sources
            if source.name in latest and now - latest[source.name][0] <= 0.10
        ]
        if len(visible_torso) < args.minimum_torso_sources:
            continue
        if reference_mode == "suitcase" and len(visible_suitcase) < args.minimum_suitcase_sources:
            continue
        last_sample = now
        if reference_mode == "suitcase":
            world_from_suitcase = _mean_transform(
                [
                    latest[source.name][1] @ source.marker_from_target
                    for source in visible_suitcase
                ]
            )
            world_from_torso = world_from_suitcase @ suitcase_from_torso
        else:
            corrected_poses = {
                source.name: latest[source.name][1]
                @ effective_by_name[source.name].marker_from_target
                for source in visible_torso
            }
            consensus = _largest_consensus(
                corrected_poses,
                position_m=args.consensus_position_m,
                orientation_deg=args.consensus_orientation_deg,
            )
            world_from_torso = _mean_transform(
                [corrected_poses[name] for name in consensus]
            )
        for source in visible_torso:
            if len(samples[source.name]) >= args.samples:
                continue
            observed = np.linalg.inv(latest[source.name][1]) @ world_from_torso
            correction = np.linalg.inv(source.marker_from_target) @ observed
            samples[source.name].append(correction)

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()

    complete = {
        name: values for name, values in samples.items() if len(values) >= args.samples
    }
    if len(complete) < args.minimum_torso_sources:
        counts = {name: len(values) for name, values in samples.items()}
        raise RuntimeError(f"insufficient stable torso sources: {counts}")

    updated_sources: dict[str, Any] = {
        source.name: {
            "target_frame_correction": np.eye(4, dtype=np.float64).tolist(),
            "sample_count": 0,
        }
        for source in torso_sources
    }
    diagnostics = {}
    for name, values in complete.items():
        correction = _mean_transform(values)
        position_errors = np.asarray(
            [np.linalg.norm(value[:3, 3] - correction[:3, 3]) for value in values]
        )
        orientation_errors = np.asarray(
            [_rotation_error_deg(correction, value) for value in values]
        )
        position_std_mm = float(np.max(np.std([v[:3, 3] for v in values], axis=0)) * 1000)
        orientation_p95 = float(np.percentile(orientation_errors, 95))
        if (
            position_std_mm > args.max_fit_position_std_mm
            or orientation_p95 > args.max_fit_orientation_deg
        ):
            raise RuntimeError(
                f"unstable correction for {name}: position_std={position_std_mm:.3f}mm "
                f"orientation_p95={orientation_p95:.3f}deg"
            )
        updated_sources[name] = {
            "target_frame_correction": correction.tolist(),
            "sample_count": len(values),
        }
        diagnostics[name] = {
            "position_std_max_mm": position_std_mm,
            "position_error_p95_mm": float(np.percentile(position_errors * 1000, 95)),
            "orientation_error_p95_deg": orientation_p95,
            "correction_rotation_deg": float(
                np.degrees(np.linalg.norm(Rotation.from_matrix(correction[:3, :3]).as_rotvec()))
            ),
            "correction_translation_m": correction[:3, 3].tolist(),
        }

    output = {
        "schema": "sim2real_marker_frame_corrections_v1",
        "valid": True,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "reference": reference,
        "refresh_mode": reference_mode,
        "previous_corrections_ignored": True,
        "sources": {"torso": updated_sources},
        "diagnostics": diagnostics,
    }
    args.corrections.parent.mkdir(parents=True, exist_ok=True)
    if args.corrections.exists():
        backup = args.corrections.with_name(
            args.corrections.name + ".before-refresh"
        )
        shutil.copy2(args.corrections, backup)
    temporary = args.corrections.with_suffix(args.corrections.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    os.replace(temporary, args.corrections)
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
