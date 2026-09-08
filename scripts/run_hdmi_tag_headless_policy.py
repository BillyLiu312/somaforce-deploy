#!/usr/bin/env python3
"""Run an HDMI student export through an external HDMI-tag policy runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from somaforce_deploy.hdmi_sim2sim import TASKS, get_task, task_artifact_dir, task_motion_dir
from somaforce_deploy.contracts import ACTION_JOINT_NAMES, G1_JOINT_NAMES
from somaforce_deploy.hdmi_residual_runtime import (
    RESIDUAL_FT_PORT,
    RESIDUAL_LOCKSTEP_PORT,
    HDMIResidualController,
    LockstepClient,
    ResidualFTReceiver,
    residual_proprio,
)
from somaforce_deploy.residual import CrossResidual
from sim2real.rl_policy.inference import build_inference_module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(TASKS), default="move_suitcase")
    parser.add_argument(
        "--upstream-root", type=Path, default=REPO_ROOT.parent / "sim2real-hdmi-upstream"
    )
    parser.add_argument("--robot-config", type=Path)
    parser.add_argument(
        "--policy-config",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
    )
    parser.add_argument("--model-json", type=Path, default=None)
    parser.add_argument("--motion", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--pose-timeout", type=float, default=5.0)
    parser.add_argument("--residual-mode", choices=("off", "shadow", "c1", "c2"), default="off")
    parser.add_argument(
        "--residual",
        type=Path,
        default=REPO_ROOT / "artifacts/hdmi_push_box/cross_residual.onnx",
    )
    parser.add_argument("--residual-record", type=Path, default=None)
    parser.add_argument("--residual-ft-port", type=int, default=RESIDUAL_FT_PORT)
    parser.add_argument("--ort-num-threads", type=int, default=1)
    parser.add_argument("--completion-file", type=Path, default=None)
    parser.add_argument("--lockstep-port", type=int, default=0)
    args = parser.parse_args()
    if args.ort_num_threads <= 0:
        raise ValueError("ort-num-threads must be positive")
    if args.completion_file is not None and args.completion_file.exists():
        raise FileExistsError(f"completion file already exists: {args.completion_file}")
    os.environ["SIM2REAL_ORT_NUM_THREADS"] = str(args.ort_num_threads)

    task = get_task(args.task)
    upstream_root = args.upstream_root.resolve()
    robot_config_path = args.robot_config or upstream_root / "config/robot/g1.yaml"
    artifact_dir = task_artifact_dir(task, REPO_ROOT)
    policy_config_path = args.policy_config or artifact_dir / "policy.yaml"
    requested_model_path = args.model or artifact_dir / "student.onnx"
    motion_path = args.motion or task_motion_dir(task, REPO_ROOT) / "motion.npz"
    steps = task.policy_steps if args.steps is None else args.steps
    sys.path.insert(0, str(upstream_root))
    sys.path.insert(1, str(upstream_root / "rl_policy"))
    from rl_policy.tracking import Tracking
    from rl_policy.utils import onnx_module as hdmi_onnx_module
    from utils.common import PORTS
    from utils.strings import resolve_matching_names_values

    Tracking.start_key_listener = lambda _self: None
    original_session = hdmi_onnx_module.ort.InferenceSession
    session_options = hdmi_onnx_module.ort.SessionOptions()
    session_options.intra_op_num_threads = args.ort_num_threads
    session_options.inter_op_num_threads = 1
    session_options.execution_mode = hdmi_onnx_module.ort.ExecutionMode.ORT_SEQUENTIAL

    def inference_session(path, *session_args, **session_kwargs):
        session_kwargs.setdefault("sess_options", session_options)
        return original_session(path, *session_args, **session_kwargs)

    hdmi_onnx_module.ort.InferenceSession = inference_session
    PORTS.update({f"{name}_pose": port for name, port in task.pose_ports})
    robot_config = yaml.safe_load(robot_config_path.read_text())
    robot_config["LOW_CMD_PORT"] = 5591
    policy_config = yaml.safe_load(policy_config_path.read_text())
    for group in policy_config.get("observation", {}).values():
        for item in group.values():
            if isinstance(item, dict) and "motion_path" in item:
                item["motion_path"] = str(motion_path.resolve().parent)
    model_path = requested_model_path.resolve()
    model_json = args.model_json.resolve() if args.model_json else model_path.with_suffix(".json")
    with tempfile.TemporaryDirectory(prefix="hdmi_tag_model_") as temp_dir:
        if not model_json.exists() and args.model_json is None:
            sibling_policy_json = model_path.parent / "policy.json"
            if sibling_policy_json.exists():
                model_json = sibling_policy_json
        if model_json != model_path.with_suffix(".json"):
            temp_model = Path(temp_dir) / model_path.name
            shutil.copy2(model_path, temp_model)
            shutil.copy2(model_json, temp_model.with_suffix(".json"))
            model_path = temp_model
        policy = Tracking(
            robot_config=robot_config,
            policy_config=policy_config,
            model_path=str(model_path),
            rl_rate=50,
        )
        policy.state_dict = {"action": np.zeros(policy.num_actions, dtype=np.float32)}
        policy.perf_dict = {}
        required_pose_names: set[str] = set()
        for item in policy_config.get("observation", {}).get("object", {}).values():
            if not isinstance(item, dict):
                continue
            for key in ("object_name", "root_body_name"):
                if item.get(key):
                    required_pose_names.add(str(item[key]))
        pose_deadline = time.monotonic() + float(args.pose_timeout)
        while time.monotonic() < pose_deadline:
            if all(
                policy.state_processor.get_mocap_data(f"{name}_pos") is not None
                and policy.state_processor.get_mocap_data(f"{name}_quat") is not None
                for name in required_pose_names
            ):
                break
            time.sleep(0.01)
        else:
            raise RuntimeError(
                f"timed out waiting for pose streams: {sorted(required_pose_names)}"
            )

        residual_receiver = None
        controller = None
        residual_records: dict[str, list[np.ndarray | float | int]] = {
            key: []
            for key in (
                "nominal",
                "raw_residual",
                "delta_bounded",
                "delta_gated",
                "delta_safe",
                "composed",
                "applied",
                "contact_target",
                "contact_gain",
                "wrist_token",
                "wrench_base_yaw",
                "proprio",
                "ft_sequence",
                "contact_count",
                "force_norm",
            )
        }
        if args.residual_mode != "off":
            if tuple(policy.policy_joint_names) != tuple(ACTION_JOINT_NAMES):
                raise ValueError("HDMI action order does not match Cross residual order")
            full_indices = [policy.isaac_joint_names.index(name) for name in G1_JOINT_NAMES]
            action_indices = [policy.isaac_joint_names.index(name) for name in ACTION_JOINT_NAMES]
            velocity_ids, _, velocity_values = resolve_matching_names_values(
                robot_config["joint_velocity_limit"],
                policy.isaac_joint_names,
                preserve_order=True,
                strict=False,
            )
            velocity_full = np.zeros(len(policy.isaac_joint_names), dtype=np.float32)
            velocity_full[velocity_ids] = velocity_values
            controller = HDMIResidualController(
                CrossResidual(build_inference_module(str(args.residual.resolve()), "onnx-cpu")),
                mode=args.residual_mode,
                default_joint_pos=policy.default_dof_angles[action_indices],
                action_scale=policy.action_scale,
                joint_lower=policy.joint_pos_lower_limit[action_indices],
                joint_upper=policy.joint_pos_upper_limit[action_indices],
                velocity_limit=velocity_full[action_indices],
            )
            residual_default = policy.default_dof_angles[full_indices].copy()
            residual_receiver = ResidualFTReceiver(args.residual_ft_port)
            residual_receiver.receive_latest(timeout_ms=int(args.pose_timeout * 1000))
            original_policy = policy.policy

            def residual_policy(input_dict):
                nominal, _nominal_target, next_state = original_policy(input_dict)
                frame = residual_receiver.receive_latest()
                proprio = residual_proprio(
                    policy.state_processor,
                    default_joint_pos=residual_default,
                )
                current_joint_pos = np.asarray(
                    policy.state_processor.joint_pos, dtype=np.float32
                )[action_indices][None, :]
                step_result = controller.step(
                    nominal=nominal[None, :],
                    wrist_frame=frame.token,
                    proprio=proprio,
                    current_joint_pos=current_joint_pos,
                )
                target = policy.default_dof_angles.copy()
                target[policy.controlled_joint_indices] += (
                    step_result.applied[0] * policy.action_scale
                )
                for key in (
                    "nominal",
                    "raw_residual",
                    "delta_bounded",
                    "delta_gated",
                    "delta_safe",
                    "composed",
                    "applied",
                ):
                    residual_records[key].append(getattr(step_result, key)[0].copy())
                residual_records["contact_target"].append(step_result.contact_target)
                residual_records["contact_gain"].append(step_result.contact_gain)
                residual_records["wrist_token"].append(frame.token.copy())
                residual_records["wrench_base_yaw"].append(frame.wrench_base_yaw.copy())
                residual_records["proprio"].append(proprio[0].copy())
                residual_records["ft_sequence"].append(frame.sequence)
                residual_records["contact_count"].append(frame.contact_count)
                residual_records["force_norm"].append(frame.total_force_norm)
                return nominal, target, next_state

            policy.policy = residual_policy

        state_deadline = time.monotonic() + float(args.pose_timeout)
        while not policy.state_processor._prepare_low_state():
            if time.monotonic() >= state_deadline:
                raise RuntimeError("timed out waiting for low-state stream")
            time.sleep(0.01)
        policy.use_policy_action = True
        policy.get_ready_state = False
        policy.reset()
        lockstep = (
            LockstepClient(args.lockstep_port or RESIDUAL_LOCKSTEP_PORT)
            if args.lockstep_port
            else None
        )
        for _ in range(int(steps)):
            policy._rl_step_scheduled()
            if lockstep is None:
                time.sleep(policy.rl_dt)
            else:
                lockstep.step()
        if controller is not None:
            if len(residual_records["nominal"]) != int(steps):
                raise RuntimeError(
                    "residual controller did not execute once per requested policy step"
                )
            if args.residual_record is None:
                raise ValueError("--residual-record is required in residual modes")
            args.residual_record.parent.mkdir(parents=True, exist_ok=True)
            arrays = {
                key: np.asarray(value) for key, value in residual_records.items()
            }
            residual_sha = hashlib.sha256(args.residual.read_bytes()).hexdigest()
            arrays["metadata"] = np.asarray(
                json.dumps(
                    {
                        "schema": "somaforce_hdmi_residual_sim2sim_v1",
                        "task": task.name,
                        "mode": args.residual_mode,
                        "residual": str(args.residual.resolve()),
                        "residual_sha256": residual_sha,
                        "policy_steps": int(steps),
                        "wrist_frame": "base_yaw",
                        "wrench_normalization": {"force_N": 100.0, "moment_Nm": 10.0},
                        "history_order": {
                            "wrist": "oldest_to_newest",
                            "nominal": "newest_to_oldest",
                            "executed": "newest_to_oldest",
                        },
                        "authority": f"{args.residual_mode.upper()}_per_joint",
                        "contact_ramp": {"attack": 0.1, "release": 0.1},
                        "safety": "joint_margin_then_velocity_margin",
                    },
                    sort_keys=True,
                )
            )
            np.savez_compressed(args.residual_record, **arrays)
            print(f"saved residual record: {args.residual_record}")
        if args.completion_file is not None:
            args.completion_file.parent.mkdir(parents=True, exist_ok=True)
            args.completion_file.write_text("complete\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
