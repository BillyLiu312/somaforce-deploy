"""MuJoCo contact-to-wrist F/T token mapping for sim2sim evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class ContactSample:
    """One control-step sensor sample in each wrist body-local frame."""

    wrench: np.ndarray
    wrench_base_yaw: np.ndarray
    normalized_wrench: np.ndarray
    twist: np.ndarray
    contact_probability: np.ndarray
    quality: np.ndarray
    contact_count: int
    total_force_norm: float

    @property
    def token(self) -> np.ndarray:
        return np.concatenate(
            (
                self.normalized_wrench,
                self.twist,
                self.contact_probability[:, None],
                self.quality[:, None],
            ),
            axis=-1,
        ).astype(np.float32)


class MujocoWristFTSensor:
    """Aggregate actual MuJoCo contacts between wrist geoms and an object.

    MuJoCo's contact force is converted from the contact frame to world frame,
    summed at the wrist body origin, and rotated into that wrist's local frame.
    The resulting six wrench values are followed by local angular/linear twist,
    contact probability, and quality, matching Cross's 14-D token contract.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        wrist_body_names: tuple[str, str] = (
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
        ),
        object_body_name: str = "box",
        base_body_name: str = "pelvis",
        force_scale_N: float = 100.0,
        moment_scale_Nm: float = 10.0,
        on_threshold: float = 0.05,
        off_threshold: float = 0.025,
        probability_temperature: float = 0.02,
        smoothing_alpha: float = 0.5,
    ) -> None:
        self.model = model
        self.base_body_id = self._body_id(base_body_name)
        self.wrist_body_ids = tuple(
            self._body_id(name) for name in wrist_body_names
        )
        self.object_body_id = self._body_id(object_body_name)
        self.wrist_geom_ids = tuple(
            tuple(
                geom_id
                for geom_id in range(model.ngeom)
                if int(model.geom_bodyid[geom_id]) in self._wrist_contact_body_ids(wrist_index)
            )
            for wrist_index, _body_id in enumerate(self.wrist_body_ids)
        )
        object_body_ids = self._descendant_body_ids(self.object_body_id)
        self.object_geom_ids = frozenset(
            int(geom_id)
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) in object_body_ids
        )
        if any(not ids for ids in self.wrist_geom_ids):
            raise ValueError("wrist body has no MuJoCo geoms")
        if not self.object_geom_ids:
            raise ValueError("object body has no MuJoCo geoms")
        self.force_scale_N = float(force_scale_N)
        self.moment_scale_Nm = float(moment_scale_Nm)
        self.on_threshold = float(on_threshold)
        self.off_threshold = float(off_threshold)
        self.probability_temperature = float(probability_temperature)
        self.smoothing_alpha = float(smoothing_alpha)
        if self.force_scale_N <= 0.0 or self.moment_scale_Nm <= 0.0:
            raise ValueError("wrench scales must be positive")
        if not 0.0 <= self.off_threshold < self.on_threshold:
            raise ValueError("contact thresholds must satisfy 0 <= off < on")
        if self.probability_temperature <= 0.0:
            raise ValueError("probability_temperature must be positive")
        if not 0.0 <= self.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be within [0, 1]")
        self._contact_state = np.zeros(2, dtype=bool)
        self._contact_probability = np.zeros(2, dtype=np.float64)

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"MuJoCo model is missing body {name!r}")
        return int(body_id)

    def _wrist_contact_body_ids(self, wrist_index: int) -> set[int]:
        side = "left" if wrist_index == 0 else "right"
        allowed = {self.wrist_body_ids[wrist_index]}
        for suffix in ("wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link"):
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_{suffix}"
            )
            if body_id >= 0:
                allowed.add(int(body_id))
        expanded: set[int] = set()
        for body_id in allowed:
            expanded.update(self._descendant_body_ids(body_id))
        return expanded

    def _descendant_body_ids(self, root_body_id: int) -> set[int]:
        descendants: set[int] = set()
        for body_id in range(self.model.nbody):
            current = body_id
            while current > 0:
                if current == root_body_id:
                    descendants.add(body_id)
                    break
                current = int(self.model.body_parentid[current])
        descendants.add(root_body_id)
        return descendants

    @staticmethod
    def _yaw_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
        w, x, y, z = quaternion_wxyz
        yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        cosine, sine = np.cos(yaw), np.sin(yaw)
        return np.asarray(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    @staticmethod
    def _world_from_contact(contact_frame: np.ndarray, value: np.ndarray) -> np.ndarray:
        # MuJoCo stores contact frame axes as rows; transpose maps local -> world.
        return contact_frame.reshape(3, 3).T @ value

    def sample(self, data: mujoco.MjData) -> ContactSample:
        wrench_world = np.zeros((2, 6), dtype=np.float64)
        contact_count = 0
        total_force_norm = 0.0
        wrist_geom_sets = [set(ids) for ids in self.wrist_geom_ids]

        for contact_id in range(int(data.ncon)):
            contact = data.contact[contact_id]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if geom1 in self.object_geom_ids:
                object_geom, other_geom, sign = geom1, geom2, -1.0
            elif geom2 in self.object_geom_ids:
                object_geom, other_geom, sign = geom2, geom1, 1.0
            else:
                continue
            del object_geom
            wrist_index = next(
                (index for index, ids in enumerate(wrist_geom_sets) if other_geom in ids),
                None,
            )
            if wrist_index is None:
                continue

            force_contact = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, data, contact_id, force_contact)
            frame = np.asarray(contact.frame, dtype=np.float64)
            force_world = sign * self._world_from_contact(frame, force_contact[:3])
            torque_world = sign * self._world_from_contact(frame, force_contact[3:])
            wrist_body_id = self.wrist_body_ids[wrist_index]
            offset_world = np.asarray(contact.pos, dtype=np.float64) - data.xpos[wrist_body_id]
            torque_world = torque_world + np.cross(offset_world, force_world)
            wrench_world[wrist_index, :3] += force_world
            wrench_world[wrist_index, 3:] += torque_world
            contact_count += 1
            total_force_norm += float(np.linalg.norm(force_world))

        wrench_local = np.zeros_like(wrench_world)
        wrench_base_yaw = np.zeros_like(wrench_world)
        twist_base_yaw = np.zeros((2, 6), dtype=np.float64)
        base_rotation = self._yaw_rotation(np.asarray(data.xquat[self.base_body_id]))
        for index, body_id in enumerate(self.wrist_body_ids):
            rotation_world_from_local = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
            wrench_local[index, :3] = rotation_world_from_local.T @ wrench_world[index, :3]
            wrench_local[index, 3:] = rotation_world_from_local.T @ wrench_world[index, 3:]
            wrench_base_yaw[index, :3] = base_rotation.T @ wrench_world[index, :3]
            wrench_base_yaw[index, 3:] = base_rotation.T @ wrench_world[index, 3:]
            velocity_world = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                self.model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
                velocity_world,
                0,
            )
            twist_base_yaw[index, :3] = base_rotation.T @ velocity_world[3:]
            twist_base_yaw[index, 3:] = base_rotation.T @ velocity_world[:3]

        normalized_wrench = wrench_base_yaw.copy()
        normalized_wrench[:, :3] /= self.force_scale_N
        normalized_wrench[:, 3:] /= self.moment_scale_Nm
        force_ratio = np.linalg.norm(wrench_base_yaw[:, :3], axis=-1) / self.force_scale_N
        moment_ratio = np.linalg.norm(wrench_base_yaw[:, 3:], axis=-1) / self.moment_scale_Nm
        strength = np.sqrt(0.5 * (force_ratio**2 + moment_ratio**2))
        threshold = np.where(self._contact_state, self.off_threshold, self.on_threshold)
        self._contact_state = strength >= threshold
        raw_probability = 1.0 / (
            1.0 + np.exp(-(strength - threshold) / self.probability_temperature)
        )
        self._contact_probability = (
            self.smoothing_alpha * self._contact_probability
            + (1.0 - self.smoothing_alpha) * raw_probability
        )
        quality = np.ones(2, dtype=np.float64)

        return ContactSample(
            wrench=wrench_local.astype(np.float32),
            wrench_base_yaw=wrench_base_yaw.astype(np.float32),
            normalized_wrench=normalized_wrench.astype(np.float32),
            twist=twist_base_yaw.astype(np.float32),
            contact_probability=self._contact_probability.astype(np.float32),
            quality=quality.astype(np.float32),
            contact_count=contact_count,
            total_force_norm=float(total_force_norm),
        )
