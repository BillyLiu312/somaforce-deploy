#!/usr/bin/env python3
"""Calibrate redundant mocap rigid bodies from one known marker per object."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
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


def _source_spec(values: list[str], role: str) -> tuple[str, str]:
    name, topic = values
    if not name or not topic.startswith("/"):
        raise ValueError(f"{role} source requires NAME and absolute TOPIC")
    return name, topic


def _mean_transform(transforms: list[np.ndarray]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = np.mean([transform[:3, 3] for transform in transforms], axis=0)
    result[:3, :3] = Rotation.from_matrix(
        np.stack([transform[:3, :3] for transform in transforms])
    ).mean().as_matrix()
    return result


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(cosine))


def _calibrate_role(
    samples: list[dict[str, np.ndarray]],
    sources: list[tuple[str, str]],
    *,
    anchor_name: str,
    anchor_marker_from_target: np.ndarray,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    source_names = {name for name, _ in sources}
    if anchor_name not in source_names:
        raise ValueError(f"anchor {anchor_name!r} is not present in source list")
    estimates: dict[str, list[np.ndarray]] = {name: [] for name, _ in sources}
    for sample in samples:
        world_from_target = sample[anchor_name] @ anchor_marker_from_target
        for name in estimates:
            estimates[name].append(np.linalg.inv(sample[name]) @ world_from_target)

    result = []
    diagnostics: dict[str, object] = {}
    topic_by_name = dict(sources)
    for name, transforms in estimates.items():
        mean = _mean_transform(transforms)
        position_errors = np.asarray(
            [np.linalg.norm(transform[:3, 3] - mean[:3, 3]) for transform in transforms]
        )
        orientation_errors = np.asarray(
            [_rotation_error_deg(mean, transform) for transform in transforms]
        )
        result.append(
            {
                "name": name,
                "topic": topic_by_name[name],
                "marker_from_target": mean.tolist(),
            }
        )
        diagnostics[name] = {
            "position_error_m_p95": float(np.percentile(position_errors, 95)),
            "position_error_m_max": float(np.max(position_errors)),
            "orientation_error_deg_p95": float(
                np.percentile(orientation_errors, 95)
            ),
            "orientation_error_deg_max": float(np.max(orientation_errors)),
        }
    return result, diagnostics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role",
        choices=("torso", "suitcase", "both"),
        default="both",
        help="calibrate one object independently, or both objects together",
    )
    parser.add_argument("--anchor-calibration", type=Path, required=True)
    parser.add_argument("--torso-anchor")
    parser.add_argument("--suitcase-anchor")
    for role in ("torso", "suitcase"):
        parser.add_argument(
            f"--{role}-source",
            action="append",
            nargs=2,
            metavar=("NAME", "TOPIC"),
        )
    parser.add_argument("--position-scale", type=float, default=0.001)
    parser.add_argument("--expected-frame-id", default="world")
    parser.add_argument("--samples", type=int, default=240)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--synchronization-window-s", type=float, default=0.05)
    parser.add_argument("--max-position-error-m", type=float, default=0.01)
    parser.add_argument("--max-orientation-error-deg", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if (
        args.samples < 2
        or args.timeout <= 0
        or args.position_scale <= 0
        or args.synchronization_window_s <= 0
        or args.max_position_error_m <= 0
        or args.max_orientation_error_deg <= 0
    ):
        raise ValueError("sample count, timing, scale, and residual limits must be positive")

    selected_roles = (
        ("torso", "suitcase") if args.role == "both" else (args.role,)
    )
    sources_by_role = {
        role: [
            _source_spec(values, role)
            for values in (getattr(args, f"{role}_source") or [])
        ]
        for role in selected_roles
    }
    for role, sources in sources_by_role.items():
        if not sources:
            raise ValueError(f"--role {args.role} requires --{role}-source")
        if getattr(args, f"{role}_anchor") is None:
            raise ValueError(f"--role {args.role} requires --{role}-anchor")
    all_sources = [source for sources in sources_by_role.values() for source in sources]
    names = [name for name, _ in all_sources]
    topics = [topic for _, topic in all_sources]
    if len(names) != len(set(names)):
        raise ValueError("source names must be unique across both objects")
    if len(topics) != len(set(topics)):
        raise ValueError("source topics must be unique across both objects")

    anchor_payload = load_calibration_payload(args.anchor_calibration)
    anchor_transforms = {}
    for role in selected_roles:
        anchor_name = getattr(args, f"{role}_anchor")
        assert anchor_name is not None
        topic_by_name = dict(sources_by_role[role])
        if anchor_name not in topic_by_name:
            raise ValueError(f"{role} anchor {anchor_name!r} is not a configured source")
        known_sources = load_marker_sources(
            anchor_payload,
            role=role,
            legacy_topic=topic_by_name[anchor_name],
        )
        matching = [source for source in known_sources if source.name == anchor_name]
        if len(matching) == 1:
            anchor_transforms[role] = matching[0].marker_from_target
        elif len(known_sources) == 1:
            anchor_transforms[role] = known_sources[0].marker_from_target
        else:
            raise ValueError(
                f"anchor calibration does not identify {role} source {anchor_name!r}"
            )

    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    except ImportError as exc:
        raise RuntimeError("source ROS2 and use the project virtualenv") from exc

    rclpy.init()
    node = rclpy.create_node("sim2real_redundant_marker_calibrator")
    qos = QoSProfile(depth=30)
    qos.reliability = ReliabilityPolicy.BEST_EFFORT
    qos.durability = DurabilityPolicy.VOLATILE
    latest: dict[str, tuple[int, float, np.ndarray]] = {}
    sample_sets: dict[str, list[dict[str, np.ndarray]]] = {
        role: [] for role in selected_roles
    }
    last_stamps: dict[str, tuple[int, ...] | None] = {
        role: None for role in selected_roles
    }

    def callback(name: str):
        def receive(msg: Any) -> None:
            if args.expected_frame_id and str(msg.header.frame_id) != args.expected_frame_id:
                return
            try:
                values = pose_stamped_to_zmq_values(
                    msg, position_scale=args.position_scale
                )
            except ValueError:
                return
            stamp = int(msg.header.stamp.sec) * 1_000_000_000 + int(
                msg.header.stamp.nanosec
            )
            latest[name] = (
                stamp,
                time.monotonic(),
                _pose_matrix_from_zmq_values(values),
            )

        return receive

    for name, topic in all_sources:
        node.create_subscription(PoseStamped, topic, callback(name), qos)

    deadline = time.monotonic() + args.timeout
    while rclpy.ok() and time.monotonic() < deadline:
        if all(len(samples) >= args.samples for samples in sample_sets.values()):
            break
        rclpy.spin_once(node, timeout_sec=0.02)
        for role, sources in sources_by_role.items():
            if len(sample_sets[role]) >= args.samples:
                continue
            role_names = [name for name, _ in sources]
            if any(name not in latest for name in role_names):
                continue
            received = [latest[name][1] for name in role_names]
            if max(received) - min(received) > args.synchronization_window_s:
                continue
            stamps = tuple(latest[name][0] for name in role_names)
            if stamps == last_stamps[role]:
                continue
            last_stamps[role] = stamps
            sample_sets[role].append(
                {name: latest[name][2].copy() for name in role_names}
            )

    node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()
    missing = {
        role: len(samples)
        for role, samples in sample_sets.items()
        if len(samples) < args.samples
    }
    if missing:
        raise RuntimeError(f"insufficient synchronized marker samples: {missing}")

    marker_sources = {}
    diagnostics = {}
    for role in selected_roles:
        marker_sources[role], diagnostics[role] = _calibrate_role(
            sample_sets[role],
            sources_by_role[role],
            anchor_name=getattr(args, f"{role}_anchor"),
            anchor_marker_from_target=anchor_transforms[role],
        )
    violations = []
    for role, role_diagnostics in diagnostics.items():
        for name, values in role_diagnostics.items():
            if values["position_error_m_max"] > args.max_position_error_m:
                violations.append(f"{role}.{name}.position")
            if values["orientation_error_deg_max"] > args.max_orientation_error_deg:
                violations.append(f"{role}.{name}.orientation")

    result: dict[str, object] = {
        "schema": "sim2real_marker_to_policy_v2",
        "valid": not violations,
        "convention": "T_marker_target; calibrated from simultaneous rigid-body poses",
        "marker_sources": marker_sources,
        "calibration": {
            "samples_per_object": args.samples,
            "anchor_calibration": str(args.anchor_calibration.resolve()),
            "anchors": {
                role: getattr(args, f"{role}_anchor") for role in selected_roles
            },
            "diagnostics": diagnostics,
            "violations": violations,
        },
    }
    if "world_from_mocap" in anchor_payload:
        result["world_from_mocap"] = anchor_payload["world_from_mocap"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if not violations else 2


if __name__ == "__main__":
    raise SystemExit(main())
