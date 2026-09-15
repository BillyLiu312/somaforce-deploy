"""Reference-guided object pose fallback for temporary mocap occlusion.

Poses use ``[x, y, z, qw, qx, qy, qz]`` and transforms follow the
``T_A_B`` convention: coordinates in frame B are mapped into frame A.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _pose_matrix(pose: Any) -> np.ndarray:
    values = np.asarray(pose, dtype=np.float64).reshape(-1)
    if values.size != 7 or not np.isfinite(values).all():
        raise ValueError("pose must contain seven finite [x,y,z,qw,qx,qy,qz] values")
    x, y, z, w, qx, qy, qz = values
    norm = math.sqrt(w * w + qx * qx + qy * qy + qz * qz)
    if norm < 1e-9:
        raise ValueError("pose quaternion has near-zero norm")
    w, qx, qy, qz = w / norm, qx / norm, qy / norm, qz / norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = (
        (1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * w), 2 * (qx * qz + qy * w)),
        (2 * (qx * qy + qz * w), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * w)),
        (2 * (qx * qz - qy * w), 2 * (qy * qz + qx * w), 1 - 2 * (qx * qx + qy * qy)),
    )
    matrix[:3, 3] = (x, y, z)
    return matrix


def _quaternion_from_matrix(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray(
            (
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            )
        )
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            scale = math.sqrt(max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 1e-12)) * 2.0
            quaternion = np.asarray(
                (
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                )
            )
        elif index == 1:
            scale = math.sqrt(max(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], 1e-12)) * 2.0
            quaternion = np.asarray(
                (
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                )
            )
        else:
            scale = math.sqrt(max(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], 1e-12)) * 2.0
            quaternion = np.asarray(
                (
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                )
            )
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[0] < 0.0:
        quaternion *= -1.0
    return quaternion


def _matrix_pose(matrix: Any) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    return np.concatenate((matrix[:3, 3], _quaternion_from_matrix(matrix[:3, :3])))


def _rotation_error_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


@dataclass(frozen=True)
class ReacquisitionError:
    position_m: float
    orientation_deg: float


class ReferenceAttachedObjectEstimator:
    """Estimate an occluded object from live pelvis and reference-relative motion.

    The fallback can only arm after both the reference contact flag is true and
    mocap has measured a real object lift. This prevents a failed grasp from
    making a box left on the floor follow the robot synthetically.
    """

    def __init__(
        self,
        reference_pelvis_pose: Any,
        reference_object_pose: Any,
        reference_contact: Any,
        *,
        lift_threshold_m: float = 0.05,
    ) -> None:
        pelvis = np.asarray(reference_pelvis_pose, dtype=np.float64)
        object_pose = np.asarray(reference_object_pose, dtype=np.float64)
        contact = np.asarray(reference_contact, dtype=bool).reshape(-1)
        if pelvis.ndim != 2 or pelvis.shape[1] != 7:
            raise ValueError("reference pelvis poses must have shape (steps, 7)")
        if object_pose.shape != pelvis.shape or contact.shape != (pelvis.shape[0],):
            raise ValueError("reference object poses/contact do not match pelvis steps")
        if not np.isfinite(pelvis).all() or not np.isfinite(object_pose).all():
            raise ValueError("reference poses contain non-finite values")
        if not math.isfinite(lift_threshold_m) or lift_threshold_m <= 0.0:
            raise ValueError("lift_threshold_m must be positive and finite")

        self.reference_relative = np.stack(
            [
                np.linalg.inv(_pose_matrix(pelvis_pose)) @ _pose_matrix(item_pose)
                for pelvis_pose, item_pose in zip(pelvis, object_pose, strict=True)
            ]
        )
        self.reference_contact = contact
        self.lift_threshold_m = float(lift_threshold_m)
        self.baseline_z: float | None = None
        self.confirmed = False
        self.confirmed_step: int | None = None
        self.fallback_active = False
        self._reference_alignment: np.ndarray | None = None

    @classmethod
    def from_motion(
        cls,
        motion_path: str | Path,
        metadata_path: str | Path,
        *,
        object_name: str = "suitcase",
        root_name: str = "pelvis",
        lift_threshold_m: float = 0.05,
    ) -> ReferenceAttachedObjectEstimator:
        metadata = json.loads(Path(metadata_path).read_text())
        body_names = [str(name) for name in metadata["body_names"]]
        root_index = body_names.index(root_name)
        object_index = body_names.index(object_name)
        with np.load(motion_path, allow_pickle=False) as motion:
            positions = np.asarray(motion["body_pos_w"], dtype=np.float64)
            quaternions = np.asarray(motion["body_quat_w"], dtype=np.float64)
            contact = np.asarray(motion["object_contact"], dtype=bool)
        if contact.ndim == 2 and contact.shape[1] == 1:
            contact = contact[:, 0]
        pelvis = np.concatenate(
            (positions[:, root_index], quaternions[:, root_index]), axis=1
        )
        object_pose = np.concatenate(
            (positions[:, object_index], quaternions[:, object_index]), axis=1
        )
        return cls(
            pelvis,
            object_pose,
            contact,
            lift_threshold_m=lift_threshold_m,
        )

    def _step(self, reference_step: int) -> int:
        return int(np.clip(int(reference_step), 0, len(self.reference_contact) - 1))

    def begin_motion(self, object_pose: Any) -> None:
        pose = np.asarray(object_pose, dtype=np.float64).reshape(-1)
        if pose.size != 7 or not np.isfinite(pose).all():
            raise ValueError("initial object pose must contain seven finite values")
        self.baseline_z = float(pose[2])
        self.confirmed = False
        self.confirmed_step = None
        self.fallback_active = False
        self._reference_alignment = None

    def observe_live(
        self, pelvis_pose: Any, object_pose: Any, reference_step: int
    ) -> ReacquisitionError | None:
        pelvis_world = _pose_matrix(pelvis_pose)
        object_world = _pose_matrix(object_pose)
        step = self._step(reference_step)
        if self.baseline_z is None:
            self.baseline_z = float(object_world[2, 3])
        lifted = float(object_world[2, 3]) - self.baseline_z >= self.lift_threshold_m
        if self.reference_contact[step] and lifted and not self.confirmed:
            self.confirmed = True
            self.confirmed_step = step

        reacquisition = None
        if self.fallback_active and self._reference_alignment is not None:
            predicted = pelvis_world @ self._reference_alignment @ self.reference_relative[step]
            reacquisition = ReacquisitionError(
                position_m=float(np.linalg.norm(predicted[:3, 3] - object_world[:3, 3])),
                orientation_deg=_rotation_error_deg(predicted, object_world),
            )
        live_relative = np.linalg.inv(pelvis_world) @ object_world
        self._reference_alignment = live_relative @ np.linalg.inv(
            self.reference_relative[step]
        )
        self.fallback_active = False
        return reacquisition

    def estimate(self, pelvis_pose: Any, reference_step: int) -> np.ndarray | None:
        step = self._step(reference_step)
        if self._reference_alignment is None or not self.confirmed:
            return None
        if not self.fallback_active and not self.reference_contact[step]:
            return None
        self.fallback_active = True
        object_world = (
            _pose_matrix(pelvis_pose)
            @ self._reference_alignment
            @ self.reference_relative[step]
        )
        return _matrix_pose(object_world).astype(np.float32)
