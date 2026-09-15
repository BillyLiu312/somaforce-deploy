#!/usr/bin/env python3
"""Run HDMI suitcase nominal inference in shadow or proposal-only mode."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml
import zmq

from somaforce_deploy.attached_object_pose import ReferenceAttachedObjectEstimator
from somaforce_deploy.contracts import G1_JOINT_NAMES
from somaforce_deploy.nominal_proposal import NominalProposal

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM_ROOT = REPO_ROOT.parent / "sim2real-hdmi-upstream"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts/hdmi_move_suitcase/hdmi_tag"
DEFAULT_MOTION = REPO_ROOT / "assets/mujoco/reference/hdmi_suitcase/motion.npz"
DEFAULT_MOTION_META = REPO_ROOT / "assets/mujoco/reference/hdmi_suitcase/meta.json"
EXPECTED_INPUT_SHAPES = {
    "command": (1, 356),
    "policy": (1, 249),
    "object": (1, 10),
}


class ShadowCommandSender:
    """Audit replacement for HDMI CommandSender with no ZMQ or DDS resources."""

    instances: list["ShadowCommandSender"] = []

    def __init__(self, *_args, **_kwargs) -> None:
        self.send_calls = 0
        self.kp_level = 1.0
        self.instances.append(self)

    def send_command(self, *_args, **_kwargs) -> None:
        self.send_calls += 1
        raise RuntimeError("nominal shadow attempted to send a robot command")


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _yaw(quaternion_wxyz: np.ndarray) -> float:
    w, x, y, z = np.asarray(quaternion_wxyz, dtype=np.float64)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrapped_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _relative_xy(delta: np.ndarray, pelvis_quat: np.ndarray) -> np.ndarray:
    yaw = _yaw(pelvis_quat)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.asarray(
        [
            cosine * delta[0] + sine * delta[1],
            -sine * delta[0] + cosine * delta[1],
        ],
        dtype=np.float64,
    )


def _reference_placement(
    motion_path: Path, meta_path: Path
) -> dict[str, float | list[float]]:
    metadata = json.loads(meta_path.read_text())
    with np.load(motion_path, allow_pickle=False) as motion:
        pelvis_index = metadata["body_names"].index("pelvis")
        suitcase_index = metadata["body_names"].index("suitcase")
        pelvis_pos = np.asarray(motion["body_pos_w"][0, pelvis_index], dtype=np.float64)
        suitcase_pos = np.asarray(motion["body_pos_w"][0, suitcase_index], dtype=np.float64)
        pelvis_quat = np.asarray(motion["body_quat_w"][0, pelvis_index], dtype=np.float64)
        suitcase_quat = np.asarray(motion["body_quat_w"][0, suitcase_index], dtype=np.float64)
    delta = suitcase_pos - pelvis_pos
    return {
        "xy_distance_m": float(np.linalg.norm(delta[:2])),
        "relative_xy_m": _relative_xy(delta, pelvis_quat).tolist(),
        "relative_z_m": float(delta[2]),
        "relative_yaw_rad": _wrapped_angle(_yaw(suitcase_quat) - _yaw(pelvis_quat)),
    }


def _motion_init_pose(
    motion_path: Path, metadata_path: Path, destination_names: list[str]
) -> np.ndarray:
    metadata = json.loads(metadata_path.read_text())
    source_names = [str(name) for name in metadata["joint_names"]]
    missing = [name for name in destination_names if name not in source_names]
    if missing:
        raise ValueError(f"motion frame 0 is missing policy joints: {missing}")
    with np.load(motion_path, allow_pickle=False) as motion:
        source = np.asarray(motion["joint_pos"][0], dtype=np.float32)
    if source.shape != (len(source_names),) or not np.isfinite(source).all():
        raise ValueError("motion frame 0 has an invalid joint vector")
    index = {name: i for i, name in enumerate(source_names)}
    return np.asarray([source[index[name]] for name in destination_names], dtype=np.float32)


def _live_placement(
    pelvis: np.ndarray, suitcase: np.ndarray
) -> dict[str, float | list[float]]:
    delta = np.asarray(suitcase[:3], dtype=np.float64) - np.asarray(
        pelvis[:3], dtype=np.float64
    )
    return {
        "xy_distance_m": float(np.linalg.norm(delta[:2])),
        "relative_xy_m": _relative_xy(delta, pelvis[3:]).tolist(),
        "relative_z_m": float(delta[2]),
        "relative_yaw_rad": _wrapped_angle(
            _yaw(suitcase[3:]) - _yaw(pelvis[3:])
        ),
    }


def _pose_monitor(context: zmq.Context, port: int) -> zmq.Socket:
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(f"tcp://127.0.0.1:{port}")
    return socket


def _drain_pose(socket: zmq.Socket) -> np.ndarray | None:
    latest = None
    while True:
        try:
            payload = socket.recv(flags=zmq.DONTWAIT)
        except zmq.Again:
            break
        values = np.frombuffer(payload, dtype="<f4").copy()
        if values.shape != (7,) or not np.isfinite(values).all():
            raise ValueError(f"invalid pose payload on {socket.getsockopt(zmq.LAST_ENDPOINT)!r}")
        quaternion_norm = float(np.linalg.norm(values[3:]))
        if not np.isclose(quaternion_norm, 1.0, atol=1e-3):
            raise ValueError(f"pose quaternion norm is {quaternion_norm:.6f}")
        latest = values
    return latest


def _freeze_reference_at_frame_zero(policy: object) -> None:
    initialized = 0
    for group in policy.observations.values():
        for observation in group.funcs.values():
            if hasattr(observation, "t") and hasattr(observation, "motion_dataset"):
                observation.t[:] = -1
                observation.update({"paused": False})
                initialized += 1
    if initialized == 0:
        raise RuntimeError("policy has no motion observations to freeze at frame 0")
    policy.state_dict["paused"] = True


def _write_runtime_status(path: Path | None, current: str | None, status: str) -> str:
    if status != current:
        print(f"pose gate: {status}", flush=True)
        if path is not None:
            path.write_text(status + "\n")
    return status


def _pose_gate_decision(
    stale_pose: list[str],
    *,
    stabilizing: bool,
    occlusion_age: float,
    stabilization_grace: float,
    motion_pelvis_grace: float,
    allow_motion_grace: bool,
) -> str:
    if not stale_pose:
        return "ready"
    if stabilizing and occlusion_age <= stabilization_grace:
        return "stabilization_pose_grace"
    if (
        allow_motion_grace
        and not stabilizing
        and stale_pose == ["pelvis"]
        and occlusion_age <= motion_pelvis_grace
    ):
        return "motion_pelvis_grace"
    return "abort"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM_ROOT)
    parser.add_argument("--policy-config", type=Path, default=DEFAULT_ARTIFACT_DIR / "policy.yaml")
    parser.add_argument("--model", type=Path, default=DEFAULT_ARTIFACT_DIR / "student.onnx")
    parser.add_argument("--model-json", type=Path)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--motion-meta", type=Path, default=DEFAULT_MOTION_META)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, default=472)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument(
        "--reference-mode",
        choices=("frozen", "advance"),
        default="frozen",
        help="freeze motion at frame 0 for safe shadow, or advance for diagnostics",
    )
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--stale-timeout", type=float, default=0.25)
    parser.add_argument("--max-initial-error", type=float, default=0.50)
    parser.add_argument("--max-object-xy-error", type=float, default=0.20)
    parser.add_argument("--max-object-height-error", type=float, default=0.15)
    parser.add_argument("--max-object-yaw-error-deg", type=float, default=25.0)
    parser.add_argument("--ort-num-threads", type=int, default=1)
    parser.add_argument(
        "--proposal-port",
        type=int,
        help="publish local nominal proposals; this never writes low-command port 5591",
    )
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--start-file", type=Path)
    parser.add_argument("--motion-start-file", type=Path)
    parser.add_argument("--pose-status-file", type=Path)
    parser.add_argument("--start-timeout", type=float, default=60.0)
    parser.add_argument("--stabilization-timeout", type=float, default=300.0)
    parser.add_argument("--stabilization-pose-grace", type=float, default=30.0)
    parser.add_argument("--motion-pelvis-pose-grace", type=float, default=2.0)
    parser.add_argument(
        "--suitcase-attachment-fallback",
        action="store_true",
        help="use reference-relative suitcase motion after a confirmed real lift",
    )
    parser.add_argument(
        "--suitcase-lift-threshold",
        type=float,
        default=0.05,
        help="measured suitcase height increase required to confirm attachment",
    )
    parser.add_argument(
        "--suitcase-fallback-delay",
        type=float,
        default=0.06,
        help="missing-sample duration before attached suitcase prediction starts",
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.steps <= 0 or args.rate <= 0 or args.ort_num_threads <= 0:
        raise ValueError("steps, rate, and ort-num-threads must be positive")
    if any(
        value <= 0
        for value in (
            args.startup_timeout,
            args.stale_timeout,
            args.max_initial_error,
            args.max_object_xy_error,
            args.max_object_height_error,
            args.max_object_yaw_error_deg,
            args.start_timeout,
            args.stabilization_timeout,
            args.stabilization_pose_grace,
            args.motion_pelvis_pose_grace,
            args.suitcase_lift_threshold,
            args.suitcase_fallback_delay,
        )
    ):
        raise ValueError("timeouts and placement tolerances must be positive")
    if not args.validate_only and args.output is None:
        raise ValueError("--output is required unless --validate-only is used")
    proposal_enabled = args.proposal_port is not None
    if proposal_enabled:
        if not 1 <= args.proposal_port <= 65535:
            raise ValueError("proposal port must be in [1, 65535]")
        if args.reference_mode != "advance":
            raise ValueError("proposal mode requires --reference-mode advance")
        if (
            args.ready_file is None
            or args.start_file is None
            or args.motion_start_file is None
            or args.pose_status_file is None
        ):
            raise ValueError(
                "proposal mode requires ready, start, motion-start, and pose-status files"
            )
    elif any(
        path is not None
        for path in (
            args.ready_file,
            args.start_file,
            args.motion_start_file,
            args.pose_status_file,
        )
    ):
        raise ValueError("ready/start files require --proposal-port")
    if args.suitcase_attachment_fallback and args.reference_mode != "advance":
        raise ValueError("suitcase attachment fallback requires --reference-mode advance")
    if (
        args.suitcase_attachment_fallback
        and args.suitcase_fallback_delay >= args.stale_timeout
    ):
        raise ValueError("suitcase fallback delay must be shorter than stale timeout")
    os.environ["SIM2REAL_ORT_NUM_THREADS"] = str(args.ort_num_threads)

    upstream_root = args.upstream_root.resolve()
    sys.path.insert(0, str(upstream_root))
    sys.path.insert(1, str(upstream_root / "rl_policy"))
    from rl_policy import base_policy as hdmi_base_policy
    from rl_policy.tracking import Tracking
    from rl_policy.utils import onnx_module as hdmi_onnx_module

    hdmi_base_policy.CommandSender = ShadowCommandSender
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
    robot_config = yaml.safe_load((upstream_root / "config/robot/g1.yaml").read_text())
    policy_config = yaml.safe_load(args.policy_config.read_text())
    motion_dir = args.motion.resolve().parent
    for group in policy_config.get("observation", {}).values():
        for item in group.values():
            if isinstance(item, dict) and "motion_path" in item:
                item["motion_path"] = str(motion_dir)

    model_path = args.model.resolve()
    model_json = args.model_json.resolve() if args.model_json else model_path.with_suffix(".json")
    with tempfile.TemporaryDirectory(prefix="hdmi_shadow_model_") as temp_dir:
        if not model_json.exists() and args.model_json is None:
            model_json = model_path.parent / "policy.json"
        if not model_json.is_file():
            raise FileNotFoundError(f"HDMI model metadata is missing: {model_json}")
        if model_json != model_path.with_suffix(".json"):
            copied_model = Path(temp_dir) / model_path.name
            shutil.copy2(model_path, copied_model)
            shutil.copy2(model_json, copied_model.with_suffix(".json"))
            model_path = copied_model

        ShadowCommandSender.instances.clear()
        policy = Tracking(
            robot_config=robot_config,
            policy_config=policy_config,
            model_path=str(model_path),
            rl_rate=args.rate,
        )
        sender = ShadowCommandSender.instances[-1]
        if args.validate_only:
            print(
                "nominal shadow contract: "
                f"inputs={EXPECTED_INPUT_SHAPES} action={(1, policy.num_actions)} "
                f"joints={policy.num_dofs} command_sender=disabled"
            )
            return 0

        context = zmq.Context.instance()
        pose_sockets = {
            "pelvis": _pose_monitor(context, 5555),
            "suitcase": _pose_monitor(context, 5561),
        }
        latest_pose: dict[str, np.ndarray] = {}
        latest_pose_time = {name: 0.0 for name in pose_sockets}

        policy.state_dict = {"action": np.zeros(policy.num_actions, dtype=np.float32)}
        policy.perf_dict = {}
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            now = time.monotonic()
            for name, socket in pose_sockets.items():
                value = _drain_pose(socket)
                if value is not None:
                    latest_pose[name] = value
                    latest_pose_time[name] = now
            low_state_ready = policy.state_processor._prepare_low_state()
            policy_pose_ready = all(
                policy.state_processor.get_mocap_data(f"{name}_{field}") is not None
                for name in ("pelvis", "suitcase")
                for field in ("pos", "quat")
            )
            if low_state_ready and policy_pose_ready and len(latest_pose) == 2:
                break
            time.sleep(0.01)
        else:
            raise RuntimeError("timed out waiting for low-state and corrected pose streams")

        current = np.asarray(policy.state_processor.joint_pos, dtype=np.float32)
        motion_init = _motion_init_pose(
            args.motion,
            args.motion_meta,
            list(policy.isaac_joint_names),
        )
        initial_error = float(np.max(np.abs(current - motion_init)))
        if initial_error > args.max_initial_error:
            raise RuntimeError(
                f"robot is not at suitcase init pose: max error {initial_error:.3f} rad "
                f"> {args.max_initial_error:.3f} rad"
            )
        reference_placement = _reference_placement(args.motion, args.motion_meta)
        live_placement = _live_placement(latest_pose["pelvis"], latest_pose["suitcase"])
        placement_error = {
            "relative_xy_m": float(
                np.linalg.norm(
                    np.asarray(live_placement["relative_xy_m"])
                    - np.asarray(reference_placement["relative_xy_m"])
                )
            ),
            "relative_z_m": abs(
                live_placement["relative_z_m"] - reference_placement["relative_z_m"]
            ),
            "relative_yaw_deg": abs(
                math.degrees(
                    _wrapped_angle(
                        live_placement["relative_yaw_rad"]
                        - reference_placement["relative_yaw_rad"]
                    )
                )
            ),
        }
        placement_limits = {
            "relative_xy_m": args.max_object_xy_error,
            "relative_z_m": args.max_object_height_error,
            "relative_yaw_deg": args.max_object_yaw_error_deg,
        }
        failed_placement = [
            name
            for name, error in placement_error.items()
            if error > placement_limits[name]
        ]
        if failed_placement:
            raise RuntimeError(
                "suitcase placement does not match motion frame 0: "
                f"failed={failed_placement} live={live_placement} "
                f"reference={reference_placement} error={placement_error}"
            )

        policy.use_policy_action = True
        policy.get_ready_state = False
        policy.reset()
        if args.reference_mode == "frozen" or proposal_enabled:
            _freeze_reference_at_frame_zero(policy)
        attachment_estimator = None
        if args.suitcase_attachment_fallback:
            attachment_estimator = ReferenceAttachedObjectEstimator.from_motion(
                args.motion,
                args.motion_meta,
                lift_threshold_m=args.suitcase_lift_threshold,
            )

        # The upstream observation objects read mocap through this method. A
        # local override lets the runner provide one coherent effective pose
        # pair without publishing synthetic data onto the raw ZMQ streams.
        effective_mocap_data: dict[str, np.ndarray] = {}
        original_get_mocap_data = policy.state_processor.get_mocap_data

        def get_effective_mocap_data(key: str):
            if key in effective_mocap_data:
                return effective_mocap_data[key]
            return original_get_mocap_data(key)

        policy.state_processor.get_mocap_data = get_effective_mocap_data
        proposal_socket = None
        proposal_joint_indices = None
        pose_runtime_status = None
        if proposal_enabled:
            if set(policy.isaac_joint_names) != set(G1_JOINT_NAMES):
                raise RuntimeError("policy joint set does not match canonical G1 proposal order")
            proposal_joint_indices = np.asarray(
                [policy.isaac_joint_names.index(name) for name in G1_JOINT_NAMES],
                dtype=np.int64,
            )
            proposal_socket = context.socket(zmq.PUB)
            proposal_socket.setsockopt(zmq.SNDHWM, 1)
            proposal_socket.setsockopt(zmq.LINGER, 0)
            proposal_socket.bind(f"tcp://127.0.0.1:{args.proposal_port}")
            if args.ready_file.exists():
                raise RuntimeError(f"proposal ready file already exists: {args.ready_file}")
            args.ready_file.write_text("ready\n")
            deadline = time.monotonic() + args.start_timeout
            while not args.start_file.exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError("timed out waiting for guarded-controller start signal")
                time.sleep(0.02)
            stabilization_started_at = time.monotonic()
            pose_runtime_status = _write_runtime_status(
                args.pose_status_file, None, "ready"
            )
        records: dict[str, list[np.ndarray | float | int | str | bool]] = {
            key: []
            for key in (
                "time_ns",
                "phase",
                "reference_step",
                "low_state_tick",
                "low_state_age_s",
                "pelvis_age_s",
                "suitcase_age_s",
                "inference_ms",
                "loop_ms",
                "command",
                "policy",
                "object",
                "action",
                "q_target",
                "joint_pos",
                "joint_vel",
                "root_quat",
                "root_ang_vel",
                "pelvis_pose",
                "suitcase_pose",
                "suitcase_pose_measured",
                "suitcase_pose_source",
                "suitcase_attachment_confirmed",
                "suitcase_reacquisition_position_error_m",
                "suitcase_reacquisition_orientation_error_deg",
                "policy_ood_ratio",
                "command_ood_ratio",
                "object_ood_ratio",
            )
        }
        period = 1.0 / args.rate
        next_step = time.perf_counter()
        last_tick = None
        last_tick_time = time.monotonic()
        sequence = 0
        motion_step = 0
        stabilization_steps = 0
        stabilization_pose_grace_steps = 0
        motion_pose_grace_steps = 0
        attachment_fallback_steps = 0
        attachment_reacquisition_count = 0
        motion_started = False
        while motion_step < args.steps:
            stabilizing = bool(
                proposal_enabled and not args.motion_start_file.exists()
            )
            if stabilizing:
                if time.monotonic() - stabilization_started_at > args.stabilization_timeout:
                    raise RuntimeError("timed out waiting for operator motion-start signal")
                policy.state_dict["paused"] = True
                phase = "stabilize"
                reference_step = 0
                stabilization_steps += 1
            else:
                policy.state_dict["paused"] = args.reference_mode == "frozen"
                phase = "motion" if args.reference_mode == "advance" else "frozen"
                reference_step = motion_step
                motion_step += 1
            sequence += 1
            loop_start = time.perf_counter()
            now = time.monotonic()
            pose_updated = {name: False for name in pose_sockets}
            for name, socket in pose_sockets.items():
                value = _drain_pose(socket)
                if value is not None:
                    latest_pose[name] = value
                    latest_pose_time[name] = now
                    pose_updated[name] = True
            if not policy.state_processor._prepare_low_state():
                raise RuntimeError("low-state unavailable during nominal shadow")
            low_state = policy.state_processor.latest_low_state
            tick = int(low_state.tick)
            if tick != last_tick:
                last_tick = tick
                last_tick_time = now
            low_state_age = now - last_tick_time
            pose_ages = {name: now - latest_pose_time[name] for name in pose_sockets}
            if low_state_age > args.stale_timeout:
                raise RuntimeError(f"low-state stale for {low_state_age:.3f}s")
            stale_pose = [name for name, age in pose_ages.items() if age > args.stale_timeout]
            effective_pose = {
                name: value.copy() for name, value in latest_pose.items()
            }
            suitcase_pose_source = "live" if "suitcase" not in stale_pose else "held_last"
            reacquisition_position_error = math.nan
            reacquisition_orientation_error = math.nan
            if not stabilizing and not motion_started:
                motion_started = True
                if attachment_estimator is not None:
                    attachment_estimator.begin_motion(latest_pose["suitcase"])
            if attachment_estimator is not None and not stabilizing:
                if pose_updated["suitcase"] and "pelvis" not in stale_pose:
                    reacquisition = attachment_estimator.observe_live(
                        latest_pose["pelvis"],
                        latest_pose["suitcase"],
                        reference_step,
                    )
                    if reacquisition is not None:
                        reacquisition_position_error = reacquisition.position_m
                        reacquisition_orientation_error = reacquisition.orientation_deg
                        attachment_reacquisition_count += 1
                        print(
                            "suitcase marker reacquired: "
                            f"position_error={reacquisition.position_m:.3f}m "
                            f"orientation_error={reacquisition.orientation_deg:.1f}deg",
                            flush=True,
                        )
                elif pose_ages["suitcase"] > args.suitcase_fallback_delay:
                    estimate = attachment_estimator.estimate(
                        latest_pose["pelvis"], reference_step
                    )
                    if estimate is not None:
                        effective_pose["suitcase"] = estimate
                        suitcase_pose_source = "reference_attached"
                        attachment_fallback_steps += 1

            gated_stale_pose = [
                name
                for name in stale_pose
                if not (
                    name == "suitcase"
                    and suitcase_pose_source == "reference_attached"
                )
            ]
            if gated_stale_pose:
                occlusion_age = max(
                    pose_ages[name] - args.stale_timeout
                    for name in gated_stale_pose
                )
                pose_decision = _pose_gate_decision(
                    gated_stale_pose,
                    stabilizing=stabilizing,
                    occlusion_age=occlusion_age,
                    stabilization_grace=args.stabilization_pose_grace,
                    motion_pelvis_grace=args.motion_pelvis_pose_grace,
                    allow_motion_grace=proposal_enabled,
                )
                if pose_decision == "stabilization_pose_grace":
                    phase = "stabilize_pose_grace"
                    stabilization_pose_grace_steps += 1
                    pose_runtime_status = _write_runtime_status(
                        args.pose_status_file,
                        pose_runtime_status,
                        f"blocked:{','.join(gated_stale_pose)}",
                    )
                elif pose_decision == "motion_pelvis_grace":
                    phase = "motion_pose_grace"
                    motion_pose_grace_steps += 1
                    degraded = "pelvis"
                    if suitcase_pose_source == "reference_attached":
                        degraded += ",suitcase_attached"
                    pose_runtime_status = _write_runtime_status(
                        args.pose_status_file,
                        pose_runtime_status,
                        f"degraded:{degraded}",
                    )
                else:
                    if proposal_enabled:
                        pose_runtime_status = _write_runtime_status(
                            args.pose_status_file,
                            pose_runtime_status,
                            f"blocked:{','.join(gated_stale_pose)}",
                        )
                    phase_context = " during stabilization" if stabilizing else ""
                    raise RuntimeError(
                        f"corrected pose stream stale{phase_context}: {gated_stale_pose} "
                        f"for {occlusion_age:.3f}s"
                    )
            else:
                if proposal_enabled:
                    if suitcase_pose_source == "reference_attached":
                        phase = "motion_object_fallback"
                        pose_status = "degraded:suitcase_attached"
                    else:
                        pose_status = "ready"
                    pose_runtime_status = _write_runtime_status(
                        args.pose_status_file, pose_runtime_status, pose_status
                    )

            effective_mocap_data.update(
                {
                    "pelvis_pos": effective_pose["pelvis"][:3],
                    "pelvis_quat": effective_pose["pelvis"][3:],
                    "suitcase_pos": effective_pose["suitcase"][:3],
                    "suitcase_quat": effective_pose["suitcase"][3:],
                }
            )

            policy.update()
            observations = policy.prepare_obs_for_rl()
            for name, expected_shape in EXPECTED_INPUT_SHAPES.items():
                value = np.asarray(observations[name], dtype=np.float32)
                if value.shape != expected_shape or not np.isfinite(value).all():
                    raise RuntimeError(
                        f"invalid {name} observation: shape={value.shape}, expected={expected_shape}"
                    )
            policy.state_dict.update(observations)
            policy.state_dict["is_init"] = np.zeros(1, dtype=bool)
            inference_start = time.perf_counter()
            action, q_target, next_state = policy.policy(policy.state_dict)
            inference_ms = (time.perf_counter() - inference_start) * 1000.0
            action = np.asarray(action, dtype=np.float32)
            q_target = np.asarray(q_target, dtype=np.float32)
            if action.shape != (policy.num_actions,) or not np.isfinite(action).all():
                raise RuntimeError(f"invalid nominal action: shape={action.shape}")
            if q_target.shape != (policy.num_dofs,) or not np.isfinite(q_target).all():
                raise RuntimeError(f"invalid nominal q_target: shape={q_target.shape}")
            policy.state_dict = next_state
            policy.state_dict["action"] = action
            policy.state_dict["q_target"] = q_target

            if proposal_socket is not None:
                assert proposal_joint_indices is not None
                proposal = NominalProposal(
                    source_time_ns=time.monotonic_ns(),
                    sequence=sequence,
                    reference_step=reference_step,
                    action=action,
                    q_target=q_target[proposal_joint_indices],
                )
                proposal_socket.send(proposal.to_bytes(), flags=zmq.DONTWAIT)

            records["time_ns"].append(time.time_ns())
            records["phase"].append(phase)
            records["reference_step"].append(reference_step)
            records["low_state_tick"].append(tick)
            records["low_state_age_s"].append(low_state_age)
            records["pelvis_age_s"].append(pose_ages["pelvis"])
            records["suitcase_age_s"].append(pose_ages["suitcase"])
            records["inference_ms"].append(inference_ms)
            records["command"].append(observations["command"][0].copy())
            records["policy"].append(observations["policy"][0].copy())
            records["object"].append(observations["object"][0].copy())
            records["action"].append(action.copy())
            records["q_target"].append(q_target.copy())
            records["joint_pos"].append(policy.state_processor.joint_pos.copy())
            records["joint_vel"].append(policy.state_processor.joint_vel.copy())
            records["root_quat"].append(policy.state_processor.root_quat_b.copy())
            records["root_ang_vel"].append(policy.state_processor.root_ang_vel_b.copy())
            records["pelvis_pose"].append(effective_pose["pelvis"].copy())
            records["suitcase_pose"].append(effective_pose["suitcase"].copy())
            records["suitcase_pose_measured"].append(latest_pose["suitcase"].copy())
            records["suitcase_pose_source"].append(suitcase_pose_source)
            records["suitcase_attachment_confirmed"].append(
                bool(attachment_estimator and attachment_estimator.confirmed)
            )
            records["suitcase_reacquisition_position_error_m"].append(
                reacquisition_position_error
            )
            records["suitcase_reacquisition_orientation_error_deg"].append(
                reacquisition_orientation_error
            )
            for name in ("policy_ood_ratio", "command_ood_ratio", "object_ood_ratio"):
                value = np.asarray(next_state.get(name, np.nan), dtype=np.float32)
                records[name].append(float(value.reshape(-1)[0]))
            loop_ms = (time.perf_counter() - loop_start) * 1000.0
            records["loop_ms"].append(loop_ms)
            next_step += period
            sleep_s = next_step - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                # A delayed inference must not publish several reference steps in a burst.
                next_step = time.perf_counter()

        if sender.send_calls != 0:
            raise RuntimeError(f"shadow command sender was called {sender.send_calls} times")
        arrays = {name: np.asarray(values) for name, values in records.items()}
        action_abs = np.abs(arrays["action"])
        target_margin = np.minimum(
            arrays["q_target"] - policy.joint_pos_lower_limit,
            policy.joint_pos_upper_limit - arrays["q_target"],
        )
        target_error = np.abs(arrays["q_target"] - arrays["joint_pos"])
        target_steps = np.abs(np.diff(arrays["q_target"], axis=0))
        target_step_p95 = float(np.percentile(target_steps, 95)) if target_steps.size else 0.0
        target_step_max = float(np.max(target_steps)) if target_steps.size else 0.0
        violation_mask = target_margin < 0.0
        violation_count = int(np.count_nonzero(violation_mask))
        violation_steps = int(np.count_nonzero(np.any(violation_mask, axis=1)))
        ood_summary = {
            name: _percentiles(arrays[name][np.isfinite(arrays[name])])
            if np.isfinite(arrays[name]).any()
            else None
            for name in ("policy_ood_ratio", "command_ood_ratio", "object_ood_ratio")
        }
        summary = {
            "schema": "somaforce_hdmi_suitcase_nominal_runner_v2",
            "result": "pass" if violation_count == 0 else "review",
            "steps": int(arrays["time_ns"].shape[0]),
            "motion_steps": args.steps,
            "stabilization_steps": stabilization_steps,
            "stabilization_pose_grace_steps": stabilization_pose_grace_steps,
            "motion_pose_grace_steps": motion_pose_grace_steps,
            "suitcase_attachment_fallback_enabled": args.suitcase_attachment_fallback,
            "suitcase_lift_threshold_m": args.suitcase_lift_threshold,
            "suitcase_fallback_delay_s": args.suitcase_fallback_delay,
            "suitcase_attachment_confirmed_step": (
                attachment_estimator.confirmed_step
                if attachment_estimator is not None
                else None
            ),
            "suitcase_attachment_fallback_steps": attachment_fallback_steps,
            "suitcase_attachment_reacquisition_count": attachment_reacquisition_count,
            "rate_hz": args.rate,
            "reference_mode": args.reference_mode,
            "policy_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
            "policy_config": str(args.policy_config.resolve()),
            "motion": str(args.motion.resolve()),
            "initial_max_joint_error_rad": initial_error,
            "reference_placement": reference_placement,
            "live_placement": live_placement,
            "placement_error": placement_error,
            "command_sender_calls": sender.send_calls,
            "proposal_output": proposal_enabled,
            "proposal_port": args.proposal_port,
            "inference_ms": _percentiles(arrays["inference_ms"]),
            "loop_ms": _percentiles(arrays["loop_ms"]),
            "loop_overrun_count": int(np.count_nonzero(arrays["loop_ms"] > period * 1000.0)),
            "action_abs_p95": float(np.percentile(action_abs, 95)),
            "action_abs_max": float(np.max(action_abs)),
            "target_error_rad_p95": float(np.percentile(target_error, 95)),
            "target_error_rad_max": float(np.max(target_error)),
            "target_step_rad_p95": target_step_p95,
            "target_step_rad_max": target_step_max,
            "minimum_joint_limit_margin_rad": float(np.min(target_margin)),
            "joint_limit_violation_count": violation_count,
            "joint_limit_violation_step_count": violation_steps,
            "ood_ratio": ood_summary,
            "low_state_age_ms_max": float(np.max(arrays["low_state_age_s"]) * 1000.0),
            "pelvis_age_ms_max": float(np.max(arrays["pelvis_age_s"]) * 1000.0),
            "suitcase_age_ms_max": float(np.max(arrays["suitcase_age_s"]) * 1000.0),
        }
        metadata = {
            "schema": summary["schema"],
            "shadow": not proposal_enabled,
            "robot_command_output": False,
            "proposal_output": proposal_enabled,
            "proposal_joint_names": list(G1_JOINT_NAMES) if proposal_enabled else None,
            "joint_names": list(policy.isaac_joint_names),
            "policy_joint_names": list(policy.policy_joint_names),
            "summary": summary,
        }
        arrays["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **arrays)
        summary_path = args.output.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        for socket in pose_sockets.values():
            socket.close(linger=0)
        if proposal_socket is not None:
            proposal_socket.close(linger=0)
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"saved nominal inference record: {args.output}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
