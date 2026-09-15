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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ("pelvis", "suitcase"):
        parser.add_argument(f"--{role}-marker-position", type=float, nargs=3, required=True)
        parser.add_argument(f"--{role}-marker-rpy-deg", type=float, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema": "sim2real_marker_to_policy_v1",
        "convention": "T_marker_policy; input measurements were T_policy_marker",
        "pelvis": _inverse_transform(args.pelvis_marker_position, args.pelvis_marker_rpy_deg).tolist(),
        "suitcase": _inverse_transform(args.suitcase_marker_position, args.suitcase_marker_rpy_deg).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"saved marker transforms: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
