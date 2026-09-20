#!/usr/bin/env python3
"""Convert measured rigid-body geometry into marker-to-policy transforms.

Inputs describe ``T_policy_marker``: marker origin in policy/body coordinates
and marker axes expressed in policy/body coordinates. The output stores its
inverse ``T_marker_policy``, which is the convention consumed by the ROS2 pose
adapter and calibration solver.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def _inverse_transform(position: list[float], rpy_deg: list[float]) -> np.ndarray:
    position_array = np.asarray(position, dtype=np.float64)
    if position_array.shape != (3,) or not np.isfinite(position_array).all():
        raise ValueError("position must be three finite values in metres")
    measured = np.eye(4, dtype=np.float64)
    measured[:3, :3] = Rotation.from_euler("xyz", rpy_deg, degrees=True).as_matrix()
    measured[:3, 3] = position_array
    return np.linalg.inv(measured)


def _marker_source(values: list[str], role: str) -> dict[str, object]:
    name, topic, *raw_numbers = values
    if not name:
        raise ValueError(f"{role} marker name must not be empty")
    if not topic.startswith("/"):
        raise ValueError(f"{role} marker topic must be absolute: {topic!r}")
    try:
        numbers = [float(value) for value in raw_numbers]
    except ValueError as exc:
        raise ValueError(f"{role} marker position/RPY must be numeric") from exc
    return {
        "name": name,
        "topic": topic,
        "marker_from_target": _inverse_transform(numbers[:3], numbers[3:]).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ("pelvis", "suitcase"):
        parser.add_argument(f"--{role}-marker-position", type=float, nargs=3)
        parser.add_argument(f"--{role}-marker-rpy-deg", type=float, nargs=3)
    for role in ("torso", "suitcase"):
        parser.add_argument(
            f"--{role}-source",
            action="append",
            nargs=8,
            metavar=("NAME", "TOPIC", "PX", "PY", "PZ", "ROLL", "PITCH", "YAW"),
            help=(
                "repeat for each rigid-body marker; position is marker origin in "
                "the target frame (m), followed by target-frame XYZ RPY (deg)"
            ),
        )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    multi_marker = bool(args.torso_source or args.suitcase_source)
    if multi_marker:
        if not args.torso_source or not args.suitcase_source:
            raise ValueError("multi-marker output requires torso and suitcase sources")
        result = {
            "schema": "sim2real_marker_to_policy_v2",
            "convention": (
                "T_marker_target maps target coordinates into each marker frame; "
                "input measurements were T_target_marker"
            ),
            "marker_sources": {
                "torso": [
                    _marker_source(values, "torso") for values in args.torso_source
                ],
                "suitcase": [
                    _marker_source(values, "suitcase")
                    for values in args.suitcase_source
                ],
            },
        }
    else:
        legacy_values = (
            args.pelvis_marker_position,
            args.pelvis_marker_rpy_deg,
            args.suitcase_marker_position,
            args.suitcase_marker_rpy_deg,
        )
        if any(value is None for value in legacy_values):
            raise ValueError(
                "provide all legacy pelvis/suitcase marker arguments, or repeat "
                "--torso-source and --suitcase-source"
            )
        result = {
            "schema": "sim2real_marker_to_policy_v1",
            "convention": "T_marker_policy; input measurements were T_policy_marker",
            "pelvis": _inverse_transform(
                args.pelvis_marker_position, args.pelvis_marker_rpy_deg
            ).tolist(),
            "suitcase": _inverse_transform(
                args.suitcase_marker_position, args.suitcase_marker_rpy_deg
            ).tolist(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"saved marker transforms: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
