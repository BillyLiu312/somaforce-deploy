#!/usr/bin/env python3
"""Headless MuJoCo physics smoke for the HDMI + Cross deploy bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from sim2real.config.robots import get_robot_cfg
from sim2real.rl_policy.inference import build_inference_module
from sim2real.sim_env.utils.mjcf import load_sim_model
from somaforce_deploy.contracts import ACTION_JOINT_NAMES
from somaforce_deploy.nominal import HDMIStudentNominal
from somaforce_deploy.residual import CrossResidual
from somaforce_deploy.runtime import DeploymentStack


ACTION_SCALE = np.asarray(
    [
        0.55, 0.55, 0.55, 0.35, 0.35, 0.44, 0.55, 0.55, 0.44,
        0.35, 0.35, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44,
        0.44, 0.44, 0.44, 0.44, 0.44,
    ],
    dtype=np.float32,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--residual", type=Path, required=True)
    parser.add_argument("--control-steps", type=int, default=25)
    parser.add_argument("--authority", type=float, default=0.0)
    parser.add_argument("--contact-gain", type=float, default=0.0)
    parser.add_argument("--shadow", action="store_true")
    args = parser.parse_args()
    if args.control_steps <= 0:
        raise ValueError("control-steps must be positive")

    robot_cfg = get_robot_cfg("g1")
    model = load_sim_model(robot_cfg)
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(robot_cfg.default_qpos, dtype=np.float64)
    mujoco.mj_forward(model, data)

    student = HDMIStudentNominal(
        build_inference_module(str(args.student), "onnx-cpu")
    )
    residual = CrossResidual(
        build_inference_module(str(args.residual), "onnx-cpu")
    )
    stack = DeploymentStack(
        nominal=student,
        residual=residual,
        mode="hdmi_student_residual_shadow" if args.shadow else "hdmi_student_residual",
        authority=args.authority,
        contact_gain=args.contact_gain,
    )

    joint_ids = []
    qpos_adrs = []
    qvel_adrs = []
    for name in ACTION_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model is missing action joint {name}")
        joint_ids.append(int(joint_id))
        qpos_adrs.append(int(model.jnt_qposadr[joint_id]))
        qvel_adrs.append(int(model.jnt_dofadr[joint_id]))

    root_height_start = float(data.qpos[2])
    for _ in range(args.control_steps):
        result = stack.step(
            nominal_kwargs={
                "command": np.zeros((1, 356), np.float32),
                "policy": np.zeros((1, 249), np.float32),
                "object_obs": np.zeros((1, 10), np.float32),
            },
            # Synthetic zero F/T token is explicit until MuJoCo task sensors are bound.
            wrist_tokens=np.zeros((1, 2, 16, 14), np.float32),
            proprio=np.zeros((1, 64), np.float32),
        )
        q_target = np.asarray(
            [data.qpos[address] for address in qpos_adrs], dtype=np.float32
        )
        q_target += result.applied[0] * ACTION_SCALE
        for index, (joint_id, qpos_address, qvel_address) in enumerate(
            zip(joint_ids, qpos_adrs, qvel_adrs, strict=True)
        ):
            del joint_id
            error = q_target[index] - data.qpos[qpos_address]
            torque = 40.0 * error - 1.0 * data.qvel[qvel_address]
            data.ctrl[index] = float(torque)
        mujoco.mj_step(model, data)

    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise RuntimeError("MuJoCo state became non-finite")
    print(
        "mujoco bundle smoke: PASS "
        f"steps={args.control_steps} nq={model.nq} nu={model.nu} "
        f"root_height={float(data.qpos[2]):.6f} "
        f"root_height_delta={float(data.qpos[2] - root_height_start):.6f} "
        f"shadow={args.shadow}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
