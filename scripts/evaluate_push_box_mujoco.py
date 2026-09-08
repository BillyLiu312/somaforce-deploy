#!/usr/bin/env python3
"""Evaluate the HDMI student plus Cross residual on the local G1+box scene."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from sim2real.rl_policy.inference import build_inference_module
from somaforce_deploy.contracts import ACTION_JOINT_NAMES
from somaforce_deploy.mujoco_ft import MujocoWristFTSensor
from somaforce_deploy.nominal import HDMIStudentNominal
from somaforce_deploy.residual import CrossResidual
from somaforce_deploy.runtime import DeploymentStack
from somaforce_deploy.scaffold import ScaffoldNominal, build_scaffold_observation


BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]
FUTURE_STEPS = (1, 2, 8, 16, 32)
ACTION_SCALE = {
    "hip_yaw": 0.55,
    "hip_roll": 0.35,
    "hip_pitch": 0.55,
    "knee": 0.35,
    "ankle_pitch": 0.44,
    "ankle_roll": 0.44,
    "waist_roll": 0.44,
    "waist_pitch": 0.44,
    "waist_yaw": 0.55,
    "shoulder_pitch": 0.44,
    "shoulder_roll": 0.44,
    "shoulder_yaw": 0.44,
    "elbow": 0.44,
}


def _yaw(quat: np.ndarray) -> float:
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _rotate_inverse_yaw(vector: np.ndarray, yaw: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    out = vector.copy()
    out[..., 0] = c * vector[..., 0] + s * vector[..., 1]
    out[..., 1] = -s * vector[..., 0] + c * vector[..., 1]
    return out


def _relative_position(position: np.ndarray, root_position: np.ndarray, root_yaw: float) -> np.ndarray:
    delta = np.asarray(position, dtype=np.float64) - np.asarray(root_position, dtype=np.float64)
    return _rotate_inverse_yaw(delta, root_yaw).astype(np.float32)


def _joint_scale(name: str) -> float:
    for key, value in ACTION_SCALE.items():
        if key in name:
            return value
    raise KeyError(f"no action scale for {name}")


class PushBoxEvaluator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = Path(__file__).resolve().parents[1]
        self.scene_path = self.root / "assets/mujoco/g1_29dof_nohand/g1_29dof_nohand-box.xml"
        self.motion_path = self.root / "assets/mujoco/reference/push_box/motion.npz"
        self.meta_path = self.root / "assets/mujoco/reference/push_box/meta.json"
        self.asset_meta_path = self.root / "assets/mujoco/reference/push_box/student_asset_meta.json"
        self.motion = np.load(self.motion_path, allow_pickle=False)
        self.motion_meta = json.loads(self.meta_path.read_text())
        self.asset_meta = json.loads(self.asset_meta_path.read_text())
        self.motion_body_names = list(self.motion_meta["body_names"])
        self.motion_joint_names = list(self.motion_meta["joint_names"])
        self.motion_length = int(self.motion["joint_pos"].shape[0])
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.model.opt.timestep = float(args.physics_dt)
        self.model.opt.iterations = 100
        self.model.opt.ls_iterations = 50
        # Mirror the HDMI-exported reflected motor armatures for all 29 joints
        # before constructing MjData. HDMI's asset metadata leaves joint
        # friction unspecified.
        for joint_name in self.asset_meta["joint_names_isaac"]:
            joint_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
            if joint_id < 0:
                continue
            dof = int(self.model.jnt_dofadr[joint_id])
            armature_5020 = 0.003609725
            armature_7520_14 = 0.010177520
            armature_7520_22 = 0.025101925
            armature_4010 = 0.00425
            if "hip_pitch" in joint_name or "hip_yaw" in joint_name or joint_name == "waist_yaw_joint":
                armature = armature_7520_14
            elif "hip_roll" in joint_name or "knee" in joint_name:
                armature = armature_7520_22
            elif "ankle" in joint_name or joint_name in {"waist_roll_joint", "waist_pitch_joint"}:
                armature = 2.0 * armature_5020
            elif "wrist_pitch" in joint_name or "wrist_yaw" in joint_name:
                armature = armature_4010
            else:
                armature = armature_5020
            self.model.dof_armature[dof] = armature
        self.data = mujoco.MjData(self.model)
        floor_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
        if floor_id >= 0:
            # World-geometry transforms are cached; refresh derived constants so
            # the requested plane height is reflected in contact generation.
            self.model.geom_pos[floor_id, 2] = float(args.ground_z)
            mujoco.mj_setConst(self.model, self.data)
        self._configure_collision_filters()
        if not bool(args.all_contact_geometry):
            self._restrict_contact_geometry()
        self.robot_joint_names = list(self.asset_meta["joint_names_isaac"])
        self.default_joint_pos = np.asarray(self.asset_meta["default_joint_pos"], dtype=np.float32)
        self.default_by_joint = dict(zip(self.robot_joint_names, self.default_joint_pos, strict=True))
        self.root_joint_id = self._joint_id("pelvis_root")
        self.box_joint_id = self._joint_id("box_root")
        self.root_qpos_adr = int(self.model.jnt_qposadr[self.root_joint_id])
        self.root_qvel_adr = int(self.model.jnt_dofadr[self.root_joint_id])
        self.box_qpos_adr = int(self.model.jnt_qposadr[self.box_joint_id])
        self.qpos_adrs = {name: int(self.model.jnt_qposadr[self._joint_id(name)]) for name in self.robot_joint_names}
        self.qvel_adrs = {name: int(self.model.jnt_dofadr[self._joint_id(name)]) for name in self.robot_joint_names}
        self.act_adrs = {self.model.actuator(i).name: i for i in range(self.model.nu)}
        self.box_x = float(args.box_x)
        self.zero_action = bool(args.zero_action)
        self.action_buffer = np.zeros((23, 3), dtype=np.float32)
        self.applied_action = np.zeros(23, dtype=np.float32)
        self.action_delay = 4
        self.action_alpha = 0.9
        self._set_initial_state()
        self.fixed_box = bool(args.fixed_box)
        self.box_anchor_qpos = self.data.qpos[self.box_qpos_adr : self.box_qpos_adr + 7].copy()
        self.sensor = MujocoWristFTSensor(self.model, contact_force_threshold=args.contact_threshold)
        self.student = HDMIStudentNominal(build_inference_module(str(args.student), "onnx-cpu"))
        self.scaffold_nominal = None
        if bool(args.scaffold_only):
            self.scaffold_nominal = ScaffoldNominal(self, args.scaffold_artifact)
        self.residual = None if bool(args.nominal_only) or bool(args.scaffold_only) else CrossResidual(build_inference_module(str(args.residual), "onnx-cpu"))
        self.stack = DeploymentStack(
            nominal=(self.scaffold_nominal if self.scaffold_nominal is not None else self.student),
            residual=self.residual,
            mode=("hdmi_student_baseline" if bool(args.nominal_only) or bool(args.scaffold_only) else ("hdmi_student_residual_shadow" if args.shadow else "hdmi_student_residual")),
            authority=args.authority,
            contact_gain=args.contact_gain,
        )
        current_joint = self._joint_positions()
        self.joint_history = deque((current_joint.copy() for _ in range(9)), maxlen=9)
        self.action_history = deque((np.zeros(23, dtype=np.float32) for _ in range(3)), maxlen=3)
        self.root_ang_history = deque((np.zeros(3, dtype=np.float32) for _ in range(9)), maxlen=9)
        self.gravity_history = deque((self._projected_gravity().copy() for _ in range(9)), maxlen=9)
        self.task_spec_action_names = tuple(ACTION_JOINT_NAMES)
        self.wrist_history = deque((np.zeros((2, 14), dtype=np.float32) for _ in range(16)), maxlen=16)

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"scene is missing body {name!r}")
        return int(body_id)

    def _joint_id(self, name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"scene is missing joint {name!r}")
        return int(joint_id)

    def _configure_collision_filters(self) -> None:
        """Enable only the USD-derived collision proxies and disable self-contact."""
        box_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box"))
        for geom_id in range(self.model.ngeom):
            body_id = int(self.model.geom_bodyid[geom_id])
            if body_id == 0 or body_id == box_id:
                continue
            self.model.geom_contype[geom_id] = 0
            self.model.geom_conaffinity[geom_id] = 0
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if "collision_" in name or "foot_sphere_" in name:
                self.model.geom_contype[geom_id] = 1
                self.model.geom_conaffinity[geom_id] = 0

    def _restrict_contact_geometry(self) -> None:
        """Deprecated sensing label; physical robot-floor/box contact stays enabled."""
        return

    def _set_initial_state(self) -> None:
        body_index = self.motion_body_names.index("pelvis")
        box_index = self.motion_body_names.index("box")
        if bool(self.args.scaffold_only):
            self.data.qpos[self.root_qpos_adr : self.root_qpos_adr + 3] = self.motion["body_pos_w"][0, body_index]
            self.data.qpos[self.root_qpos_adr + 3 : self.root_qpos_adr + 7] = self.motion["body_quat_w"][0, body_index]
            self.data.qpos[self.box_qpos_adr : self.box_qpos_adr + 3] = self.motion["body_pos_w"][0, box_index]
            self.data.qpos[self.box_qpos_adr + 3 : self.box_qpos_adr + 7] = self.motion["body_quat_w"][0, box_index]
        else:
            self.data.qpos[self.root_qpos_adr : self.root_qpos_adr + 3] = (0.0, 0.0, 0.76)
            self.data.qpos[self.root_qpos_adr + 3 : self.root_qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
            self.data.qpos[self.box_qpos_adr : self.box_qpos_adr + 3] = (self.box_x, 0.0, 0.06)
            self.data.qpos[self.box_qpos_adr + 3 : self.box_qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
        for name, address in self.qpos_adrs.items():
            if bool(self.args.scaffold_only) and name in self.motion_joint_names:
                self.data.qpos[address] = self.motion["joint_pos"][0, self.motion_joint_names.index(name)]
            else:
                self.data.qpos[address] = self.default_by_joint.get(name, 0.0)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _anchor_box(self) -> None:
        if not self.fixed_box:
            return
        self.data.qpos[self.box_qpos_adr : self.box_qpos_adr + 7] = self.box_anchor_qpos
        self.data.qvel[self.model.jnt_dofadr[self.box_joint_id] : self.model.jnt_dofadr[self.box_joint_id] + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _joint_positions(self) -> np.ndarray:
        return np.asarray([self.data.qpos[self.qpos_adrs[name]] for name in self.robot_joint_names], dtype=np.float32)

    def _joint_velocities(self) -> np.ndarray:
        return np.asarray([self.data.qvel[self.qvel_adrs[name]] for name in self.robot_joint_names], dtype=np.float32)

    def _projected_gravity(self) -> np.ndarray:
        quat = np.asarray(self.data.qpos[self.root_qpos_adr + 3 : self.root_qpos_adr + 7], dtype=np.float64)
        w, x, y, z = quat
        rotation = np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        return (rotation.T @ np.asarray([0.0, 0.0, -1.0])).astype(np.float32)

    def _root_ang_vel_b(self) -> np.ndarray:
        quat = np.asarray(self.data.qpos[self.root_qpos_adr + 3 : self.root_qpos_adr + 7], dtype=np.float64)
        w, x, y, z = quat
        rotation = np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        root_ang_w = np.asarray(self.data.qvel[self.root_qvel_adr + 3 : self.root_qvel_adr + 6], dtype=np.float64)
        return (rotation.T @ root_ang_w).astype(np.float32)

    def _policy_observation(self) -> np.ndarray:
        root_ang = self._root_ang_vel_b()
        positions = np.stack(tuple(self.joint_history), axis=0)
        previous = np.stack(tuple(self.action_history), axis=1)
        return np.concatenate(
            (root_ang, self._projected_gravity(), positions[[0, 1, 2, 3, 4, 8]].reshape(-1), previous.reshape(-1)),
            axis=0,
        ).astype(np.float32)[None, :]

    def _reference_command(self, step: int) -> np.ndarray:
        root_idx = self.motion_body_names.index("pelvis")
        action_indices = [self.motion_joint_names.index(name) for name in ACTION_JOINT_NAMES]
        parts = []
        for offset in FUTURE_STEPS:
            index = min(step + offset, self.motion_length - 1)
            root_pos = self.motion["body_pos_w"][index, root_idx]
            root_yaw = _yaw(self.motion["body_quat_w"][index, root_idx])
            body_indices = [self.motion_body_names.index(name) for name in BODY_NAMES]
            positions = self.motion["body_pos_w"][index, body_indices] - root_pos
            parts.append(_rotate_inverse_yaw(positions, root_yaw).reshape(-1))
        joint_future = self.motion["joint_pos"][
            [min(step + offset, self.motion_length - 1) for offset in FUTURE_STEPS]
        ][:, action_indices].reshape(-1)
        return np.concatenate((np.concatenate(parts), joint_future, np.asarray([step / self.motion_length]))).astype(np.float32)[None, :]

    def _object_observation(self) -> np.ndarray:
        root_pos = self.data.qpos[self.root_qpos_adr : self.root_qpos_adr + 3]
        root_yaw = _yaw(self.data.qpos[self.root_qpos_adr + 3 : self.root_qpos_adr + 7])
        box_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
        box_pos = np.asarray(self.data.xpos[box_body], dtype=np.float64)
        box_yaw = _yaw(self.data.xquat[box_body])
        box_rel = _relative_position(box_pos, root_pos, root_yaw)
        heading = np.asarray([np.cos(box_yaw - root_yaw), np.sin(box_yaw - root_yaw)], dtype=np.float32)
        offsets = np.asarray([[0.0, -0.2, 0.8], [0.0, 0.2, 0.8]], dtype=np.float64)
        targets = _rotate_inverse_yaw(box_pos[None, :] + offsets - root_pos[None, :], root_yaw).reshape(-1)
        return np.concatenate((box_rel[:2], heading, targets)).astype(np.float32)[None, :]

    def _proprio(self) -> np.ndarray:
        root_ang = self._root_ang_vel_b()
        positions = self._joint_positions() - self.default_joint_pos
        return np.concatenate((root_ang, self._projected_gravity(), positions, self._joint_velocities())).astype(np.float32)[None, :]

    def _gains(self, name: str) -> tuple[float, float, float, float]:
        if "hip_pitch" in name or "hip_yaw" in name or name == "waist_yaw_joint":
            kp, kd, effort = 40.179238, 2.557890, 88.0
        elif "hip_roll" in name or "knee" in name:
            kp, kd, effort = 99.098428, 6.308802, 139.0
        elif "ankle" in name or name in {"waist_roll_joint", "waist_pitch_joint"}:
            kp, kd, effort = 28.501246, 1.814446, 50.0
        elif "wrist_pitch" in name or "wrist_yaw" in name:
            kp, kd, effort = 16.778328, 1.068142, 5.0
        else:
            kp, kd, effort = 14.250623, 0.907223, 25.0
        return kp, kd, effort, 37.0

    def _substep_action(self, action: np.ndarray, substep: int) -> np.ndarray:
        if substep == 0:
            self.action_buffer[:, 1:] = self.action_buffer[:, :-1]
            self.action_buffer[:, 0] = np.asarray(action[0], dtype=np.float32)
        delayed_index = (self.action_delay - substep + int(self.args.decimation) - 1) // int(self.args.decimation)
        delayed_index = int(np.clip(delayed_index, 0, self.action_buffer.shape[1] - 1))
        self.applied_action = self.applied_action * (1.0 - self.action_alpha) + self.action_buffer[:, delayed_index] * self.action_alpha
        return self.applied_action.copy()

    def _apply_pd(self, action: np.ndarray) -> None:
        # Isaac's JointPosition action manager targets default_joint_pos for
        # every joint, then adds policy actions only on its 23 controlled joints.
        targets = self.default_joint_pos.copy()
        for index, name in enumerate(ACTION_JOINT_NAMES):
            raw_action = float(action[index])
            targets[self.robot_joint_names.index(name)] = self.default_by_joint[name] + raw_action * _joint_scale(name)
        for index, name in enumerate(self.robot_joint_names):
            actuator_index = self.act_adrs.get(name)
            if actuator_index is None:
                continue
            kp, kd, effort, _velocity = self._gains(name)
            error = float(targets[index] - self.data.qpos[self.qpos_adrs[name]])
            torque = kp * error - kd * float(self.data.qvel[self.qvel_adrs[name]])
            self.data.ctrl[actuator_index] = np.clip(torque, -effort, effort)

    def run(self) -> Path:
        records: dict[str, list[np.ndarray | float | int]] = {key: [] for key in (
            "nominal", "residual", "composed", "applied", "wrench", "tokens", "proprio", "root_pos", "box_pos", "qpos", "qvel", "contact_count", "force_norm", "step"
        )}
        contact_steps = 0
        termination_reason = "reference_complete"
        max_steps = min(self.motion_length, int(self.args.max_steps)) if self.args.max_steps else self.motion_length
        for step in range(max_steps):
            current_joint = self._joint_positions()
            self.joint_history.appendleft(current_joint)
            sample = self.sensor.sample(self.data)
            self.wrist_history.append(sample.token)
            token_history = np.stack(tuple(self.wrist_history), axis=1)[None, ...]
            self.root_ang_history.append(self._root_ang_vel_b().copy())
            self.gravity_history.append(self._projected_gravity().copy())
            scaffold_observation = build_scaffold_observation(self, step) if self.scaffold_nominal is not None else None
            if self.zero_action:
                class _Zero:
                    def __call__(self, **_kwargs):
                        return np.zeros((1, 23), dtype=np.float32)
                result = type("_Result", (), {"nominal": np.zeros((1, 23), np.float32), "residual": np.zeros((1, 23), np.float32), "composed": np.zeros((1, 23), np.float32), "applied": np.zeros((1, 23), np.float32)})()
            else:
                nominal_kwargs = {
                    "command": self._reference_command(step),
                    "policy": self._policy_observation(),
                    "object_obs": self._object_observation(),
                }
                if scaffold_observation is not None:
                    nominal_kwargs["scaffold_observation"] = scaffold_observation
                result = self.stack.step(
                    nominal_kwargs=nominal_kwargs,
                    wrist_tokens=token_history.astype(np.float32),
                    proprio=self._proprio(),
                )
            self.action_history.appendleft(result.nominal[0].copy())
            records["nominal"].append(result.nominal[0].copy())
            records["residual"].append(result.residual[0].copy())
            records["composed"].append(result.composed[0].copy())
            records["applied"].append(result.applied[0].copy())
            records["wrench"].append(sample.wrench.copy())
            records["tokens"].append(sample.token.copy())
            records["proprio"].append(self._proprio()[0].copy())
            records["root_pos"].append(self.data.qpos[self.root_qpos_adr : self.root_qpos_adr + 3].copy())
            box_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box")
            records["box_pos"].append(self.data.xpos[box_body].copy())
            records["qpos"].append(self.data.qpos.copy())
            records["qvel"].append(self.data.qvel.copy())
            records["contact_count"].append(sample.contact_count)
            records["force_norm"].append(sample.total_force_norm)
            records["step"].append(step)
            contact_steps += int(sample.contact_count > 0)
            for _ in range(int(self.args.decimation)):
                substep_action = self._substep_action(result.applied, _)
                self._apply_pd(substep_action)
                mujoco.mj_step(self.model, self.data)
                self._anchor_box()
            self.root_ang_history.append(self._root_ang_vel_b().copy())
            self.gravity_history.append(self._projected_gravity().copy())
            if bool(self.args.stop_on_instability):
                root_z = float(self.data.qpos[self.root_qpos_adr + 2])
                if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                    termination_reason = "nonfinite_state"
                    break
                if root_z < float(self.args.root_height_failure):
                    termination_reason = f"root_height<{self.args.root_height_failure}"
                    break

        actual_steps = len(records["step"])
        output = Path(self.args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        arrays = {key: np.asarray(value) for key, value in records.items()}
        arrays.update(
            metadata=np.asarray(
                json.dumps(
                    {
                        "schema": "somaforce_mujoco_push_box_eval_v1",
                        "scene": str(self.scene_path),
                        "motion": str(self.motion_path),
                        "student": str(self.args.student),
                        "residual": str(self.args.residual),
                        "control_hz": 50.0,
                        "physics_dt": float(self.args.physics_dt),
                        "ground_z": float(self.args.ground_z),
                        "decimation": int(self.args.decimation),
                        "authority": float(self.args.authority),
                        "contact_gain": float(self.args.contact_gain),
                        "shadow": bool(self.args.shadow),
                        "free_root": True,
                        "root_anchor": False,
                        "box_fixed": bool(self.fixed_box),
                        "contact_geometry": "all_model_geoms" if bool(self.args.all_contact_geometry) else "usd_collision_proxies",
                        "ft_mapping": "MuJoCo contact frame -> world wrench -> wrist-local wrench",
                        "termination_reason": termination_reason,
                        "contact_steps": int(contact_steps),
                        "total_steps": int(actual_steps),
                        "contact_fraction": float(contact_steps / actual_steps) if actual_steps else 0.0,
                        "max_force_norm": float(np.max(arrays["force_norm"]) if actual_steps else 0.0),
                        "final_box_position": arrays["box_pos"][-1].tolist() if max_steps else None,
                    },
                    sort_keys=True,
                )
            )
        )
        np.savez_compressed(output, **arrays)
        print(f"saved evaluation record: {output}")
        print(f"contact_steps={contact_steps}/{actual_steps} max_force_norm={float(np.max(arrays['force_norm']) if actual_steps else 0.0):.6f} termination={termination_reason}")
        return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--residual", type=Path, required=False, default=None)
    parser.add_argument("--nominal-only", action="store_true")
    parser.add_argument("--scaffold-only", action="store_true")
    parser.add_argument("--scaffold-artifact", type=Path, default=None)
    parser.add_argument("--zero-action", action="store_true")
    parser.add_argument("--box-x", type=float, default=0.7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--authority", type=float, default=0.1)
    parser.add_argument("--contact-gain", type=float, default=1.0)
    parser.add_argument("--contact-threshold", type=float, default=1.0)
    parser.add_argument("--physics-dt", type=float, default=0.002)
    parser.add_argument("--ground-z", type=float, default=0.0)
    parser.add_argument("--decimation", type=int, default=10)
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument("--fixed-box", action="store_true")
    parser.add_argument("--all-contact-geometry", action="store_true")
    parser.add_argument("--stop-on-instability", action="store_true")
    parser.add_argument("--root-height-failure", type=float, default=0.45)
    args = parser.parse_args()
    if args.decimation <= 0:
        raise ValueError("decimation must be positive")
    if args.scaffold_only and args.scaffold_artifact is None:
        raise ValueError("--scaffold-artifact is required with --scaffold-only")
    if not args.nominal_only and not args.scaffold_only and args.residual is None:
        raise ValueError("--residual is required unless --nominal-only is set")
    PushBoxEvaluator(args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
