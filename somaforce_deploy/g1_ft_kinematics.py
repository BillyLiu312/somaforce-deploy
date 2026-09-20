"""G1 wrist kinematics used by the real F/T adapter."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from sim2real.config.robots.g1 import G1_CFG
from sim2real.utils.common import LowStateMessage

from .ft_adapter import WristKinematics, yaw_rotation


class G1WristKinematics:
    """Evaluate wrist frames and finite-difference twists from G1 low-state."""

    def __init__(
        self,
        mjcf_path: str | Path,
        *,
        velocity_alpha: float = 0.5,
        max_derivative_dt_s: float = 0.10,
    ) -> None:
        if not 0.0 <= velocity_alpha <= 1.0:
            raise ValueError("velocity_alpha must be in [0, 1]")
        if max_derivative_dt_s <= 0.0:
            raise ValueError("max_derivative_dt_s must be positive")
        self.model = mujoco.MjModel.from_xml_path(str(Path(mjcf_path).resolve()))
        self.data = mujoco.MjData(self.model)
        self.velocity_alpha = float(velocity_alpha)
        self.max_derivative_dt_s = float(max_derivative_dt_s)
        self._joint_qpos_addresses: list[int] = []
        for name in G1_CFG.joint_names:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id < 0:
                raise ValueError(f"kinematics model is missing joint {name!r}")
            self._joint_qpos_addresses.append(int(self.model.jnt_qposadr[joint_id]))
        root_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_root"
        )
        if root_id < 0:
            raise ValueError("kinematics model is missing pelvis_root")
        self._root_qpos_address = int(self.model.jnt_qposadr[root_id])
        self._wrist_body_ids = tuple(
            self._body_id(f"{side}_wrist_yaw_link") for side in ("left", "right")
        )
        self._root_position = np.zeros(3, dtype=np.float64)
        self._previous_positions: np.ndarray | None = None
        self._previous_rotations: np.ndarray | None = None
        self._previous_time_ns: int | None = None
        self._filtered_twist = np.zeros((2, 6), dtype=np.float64)

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"kinematics model is missing body {name!r}")
        return int(body_id)

    def reset_derivative(self) -> None:
        self._previous_positions = None
        self._previous_rotations = None
        self._previous_time_ns = None
        self._filtered_twist[:] = 0.0

    def update(
        self,
        state: LowStateMessage,
        *,
        received_monotonic_ns: int,
        root_position_world: np.ndarray | None = None,
        reset_derivative: bool = False,
    ) -> WristKinematics:
        if state.joint_positions.shape != (len(G1_CFG.joint_names),):
            raise ValueError(
                "G1 low-state joint count does not match the 29-DoF kinematics model"
            )
        quaternion = np.asarray(state.quaternion, dtype=np.float64)
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-9 or not np.isfinite(quaternion).all():
            raise ValueError("G1 low-state quaternion is invalid")
        quaternion /= norm
        if root_position_world is not None:
            root_position = np.asarray(root_position_world, dtype=np.float64)
            if root_position.shape != (3,) or not np.isfinite(root_position).all():
                raise ValueError("root_position_world must be a finite 3-vector")
            self._root_position[:] = root_position

        root = self._root_qpos_address
        self.data.qpos[root : root + 3] = self._root_position
        self.data.qpos[root + 3 : root + 7] = quaternion
        for address, value in zip(
            self._joint_qpos_addresses, state.joint_positions, strict=True
        ):
            self.data.qpos[address] = float(value)
        mujoco.mj_forward(self.model, self.data)

        positions = np.stack(
            [
                np.asarray(self.data.xpos[body_id], dtype=np.float64)
                for body_id in self._wrist_body_ids
            ]
        )
        rotations = np.stack(
            [
                np.asarray(self.data.xmat[body_id], dtype=np.float64).reshape(3, 3)
                for body_id in self._wrist_body_ids
            ]
        )
        if reset_derivative:
            self.reset_derivative()
        raw_twist_world = np.zeros((2, 6), dtype=np.float64)
        if self._previous_time_ns is not None:
            dt_s = (int(received_monotonic_ns) - self._previous_time_ns) / 1e9
            if 0.0 < dt_s <= self.max_derivative_dt_s:
                raw_twist_world[:, :3] = (
                    positions - self._previous_positions
                ) / dt_s
                for index in range(2):
                    delta_world = rotations[index] @ self._previous_rotations[index].T
                    raw_twist_world[index, 3:] = (
                        Rotation.from_matrix(delta_world).as_rotvec() / dt_s
                    )
        rotation_world_base_yaw = yaw_rotation(quaternion)
        raw_twist_base_yaw = np.empty_like(raw_twist_world)
        raw_twist_base_yaw[:, :3] = (
            rotation_world_base_yaw.T @ raw_twist_world[:, :3].T
        ).T
        raw_twist_base_yaw[:, 3:] = (
            rotation_world_base_yaw.T @ raw_twist_world[:, 3:].T
        ).T
        self._filtered_twist = (
            self.velocity_alpha * self._filtered_twist
            + (1.0 - self.velocity_alpha) * raw_twist_base_yaw
        )
        self._previous_positions = positions.copy()
        self._previous_rotations = rotations.copy()
        self._previous_time_ns = int(received_monotonic_ns)
        return WristKinematics(
            base_quaternion_wxyz=quaternion,
            world_from_wrist_rotation=rotations,
            twist_base_yaw=self._filtered_twist.copy(),
            received_monotonic_ns=int(received_monotonic_ns),
        )
