"""G1 torso-to-pelvis forward-kinematics utilities.

Frames use the convention ``T_A_B`` maps coordinates in frame B into frame A.
The G1 chain is the exact chain in the deployment MJCF:

    pelvis -> waist_yaw_link -> waist_roll_link -> torso_link

Only the three waist joint angles are required. This module is deliberately
independent of ROS, ZMQ, and a particular RobotIO backend so the same FK is
usable in MuJoCo and on the real robot.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


TORSO_OFFSET_IN_WAIST_YAW = np.asarray((-0.0039635, 0.0, 0.044), dtype=np.float64)
WAIST_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)


def _rotation(axis: str, angle: float) -> np.ndarray:
    c, s = math.cos(float(angle)), math.sin(float(angle))
    if axis == "x":
        return np.asarray(((1, 0, 0), (0, c, -s), (0, s, c)), dtype=np.float64)
    if axis == "y":
        return np.asarray(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=np.float64)
    if axis == "z":
        return np.asarray(((c, -s, 0), (s, c, 0), (0, 0, 1)), dtype=np.float64)
    raise ValueError(f"unsupported rotation axis {axis!r}")


def make_transform(rotation: Any, translation: Any) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return matrix


def validate_transform(matrix: Any, *, name: str = "transform") -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0, 0, 0, 1), atol=1e-6):
        raise ValueError(f"{name} last row must be [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant is not +1")
    return matrix


def torso_from_pelvis(waist_angles: Any) -> np.ndarray:
    """Return exact ``T_pelvis_torso`` for ``[yaw, roll, pitch]`` in radians."""
    angles = np.asarray(waist_angles, dtype=np.float64).reshape(-1)
    if angles.size != 3 or not np.isfinite(angles).all():
        raise ValueError("waist_angles must be three finite radians [yaw, roll, pitch]")
    yaw, roll, pitch = (float(value) for value in angles)
    rotation = _rotation("z", yaw) @ _rotation("x", roll) @ _rotation("y", pitch)
    translation = _rotation("z", yaw) @ TORSO_OFFSET_IN_WAIST_YAW
    return make_transform(rotation, translation)


class TorsoToPelvisFK:
    """Recover pelvis pose from a torso marker pose and waist joint angles."""

    def __init__(self, torso_from_marker: Any | None = None) -> None:
        # Input is T_torso_marker: marker pose expressed in torso coordinates.
        self.torso_from_marker = validate_transform(
            np.eye(4) if torso_from_marker is None else torso_from_marker,
            name="torso_from_marker",
        )
        self.marker_from_torso = np.linalg.inv(self.torso_from_marker)

    def pelvis_from_torso(self, torso_world: Any, waist_angles: Any) -> np.ndarray:
        torso_world = validate_transform(torso_world, name="world_from_torso")
        return torso_world @ np.linalg.inv(torso_from_pelvis(waist_angles))

    def pelvis_from_marker(self, marker_world: Any, waist_angles: Any) -> np.ndarray:
        marker_world = validate_transform(marker_world, name="world_from_marker")
        torso_world = marker_world @ self.marker_from_torso
        return self.pelvis_from_torso(torso_world, waist_angles)


def pelvis_pose_from_low_state(
    marker_world: Any,
    joint_positions: Any,
    joint_names: tuple[str, ...] | list[str],
    *,
    torso_from_marker: Any | None = None,
) -> np.ndarray:
    """Convenience wrapper using a named G1 joint-position array."""
    names = list(joint_names)
    try:
        indices = [names.index(name) for name in WAIST_JOINT_NAMES]
    except ValueError as exc:
        raise ValueError(f"low-state joint names do not contain {WAIST_JOINT_NAMES}") from exc
    positions = np.asarray(joint_positions, dtype=np.float64).reshape(-1)
    if positions.size != len(names):
        raise ValueError(f"joint_positions has {positions.size} values, expected {len(names)}")
    return TorsoToPelvisFK(torso_from_marker).pelvis_from_marker(
        marker_world,
        positions[indices],
    )
