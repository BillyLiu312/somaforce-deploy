#!/usr/bin/env python3
"""Run HDMI nominal inference in shadow or proposal-only mode.

The default profile remains the suitcase deployment.  ``--object-name`` and
``--aux-object-name`` are used by the push-door-hand hardware wrapper so the
same proposal/safety ABI can feed a different HDMI task.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import math
import os
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml
import zmq

from sim2real.rl_policy.inference import build_inference_module
from sim2real.utils.strings import resolve_matching_names_values
from somaforce_deploy.chunked_recording import (
    ChunkedNpzRecorder,
    atomic_savez,
    atomic_write_json,
    finalize_chunked_recording,
)
from somaforce_deploy.contracts import ACTION_JOINT_NAMES, G1_JOINT_NAMES
from somaforce_deploy.hdmi_residual_runtime import (
    HDMIResidualController,
    ResidualFTFrame,
    ResidualFTReceiver,
    residual_proprio,
)
from somaforce_deploy.nominal_proposal import NominalProposal
from somaforce_deploy.residual import CrossResidual

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UPSTREAM_ROOT = REPO_ROOT.parent / "sim2real-hdmi-upstream"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "artifacts/hdmi_move_suitcase/hdmi_tag"
DEFAULT_MOTION = REPO_ROOT / "assets/mujoco/reference/hdmi_suitcase/motion.npz"
DEFAULT_MOTION_META = REPO_ROOT / "assets/mujoco/reference/hdmi_suitcase/meta.json"
DEFAULT_RESIDUAL = REPO_ROOT / "artifacts/hdmi_push_box/cross_residual.onnx"
EXPECTED_INPUT_SHAPES = {"command": (1, 356), "policy": (1, 249)}


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


def _save_partial_record(
    output: Path,
    records: dict[str, list],
    *,
    reason: str,
    residual_mode: str,
    metadata: dict | None = None,
) -> None:
    frame_count = len(records.get("time_ns", []))
    if frame_count == 0:
        return
    arrays = {
        name: np.asarray(values[:frame_count])
        for name, values in records.items()
        if len(values) >= frame_count
    }
    record_metadata = dict(metadata or {})
    record_metadata.update(
        {
            "schema": "somaforce_hdmi_suitcase_hardware_record_v4",
            "complete": False,
            "termination_reason": reason,
            "residual_mode": residual_mode,
            "steps": frame_count,
        }
    )
    arrays["metadata"] = np.asarray(json.dumps(record_metadata, sort_keys=True))
    atomic_savez(output, arrays, compressed=True)
    atomic_write_json(
        output.with_suffix(".partial.json"),
        {
            "schema": "somaforce_hdmi_suitcase_hardware_record_v4",
            "complete": False,
            "termination_reason": reason,
            "steps": frame_count,
            "output": str(output),
        },
    )


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
    motion_path: Path, meta_path: Path, object_name: str
) -> dict[str, float | list[float]]:
    metadata = json.loads(meta_path.read_text())
    with np.load(motion_path, allow_pickle=False) as motion:
        pelvis_index = metadata["body_names"].index("pelvis")
        object_index = metadata["body_names"].index(object_name)
        pelvis_pos = np.asarray(motion["body_pos_w"][0, pelvis_index], dtype=np.float64)
        object_pos = np.asarray(
            motion["body_pos_w"][0, object_index], dtype=np.float64
        )
        pelvis_quat = np.asarray(
            motion["body_quat_w"][0, pelvis_index], dtype=np.float64
        )
        object_quat = np.asarray(
            motion["body_quat_w"][0, object_index], dtype=np.float64
        )
    delta = object_pos - pelvis_pos
    return {
        "xy_distance_m": float(np.linalg.norm(delta[:2])),
        "relative_xy_m": _relative_xy(delta, pelvis_quat).tolist(),
        "relative_z_m": float(delta[2]),
        "relative_yaw_rad": _wrapped_angle(_yaw(object_quat) - _yaw(pelvis_quat)),
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
    return np.asarray(
        [source[index[name]] for name in destination_names], dtype=np.float32
    )


def _load_reference_states(motion_path: Path) -> dict[str, np.ndarray]:
    keys = (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "object_contact",
    )
    with np.load(motion_path, allow_pickle=False) as motion:
        missing = [key for key in keys if key not in motion]
        if missing:
            raise ValueError(f"motion is missing reference state arrays: {missing}")
        result = {key: np.asarray(motion[key]).copy() for key in keys}
    lengths = {value.shape[0] for value in result.values()}
    if len(lengths) != 1 or next(iter(lengths)) <= 0:
        raise ValueError(f"reference state arrays have inconsistent lengths: {lengths}")
    return result


def _live_placement(
    pelvis: np.ndarray, object_pose: np.ndarray
) -> dict[str, float | list[float]]:
    delta = np.asarray(object_pose[:3], dtype=np.float64) - np.asarray(
        pelvis[:3], dtype=np.float64
    )
    return {
        "xy_distance_m": float(np.linalg.norm(delta[:2])),
        "relative_xy_m": _relative_xy(delta, pelvis[3:]).tolist(),
        "relative_z_m": float(delta[2]),
        "relative_yaw_rad": _wrapped_angle(_yaw(object_pose[3:]) - _yaw(pelvis[3:])),
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
            raise ValueError(
                f"invalid pose payload on {socket.getsockopt(zmq.LAST_ENDPOINT)!r}"
            )
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


def _startup_stream_timeout_message(
    *,
    low_state_ready: bool,
    policy_pose_missing: list[str],
    latest_pose_time: dict[str, float],
    now: float,
    stale_timeout: float,
) -> str:
    pose_status = []
    for name, received_at in latest_pose_time.items():
        if received_at <= 0.0:
            pose_status.append(f"{name}=never_received")
        else:
            pose_status.append(f"{name}=age_{now - received_at:.3f}s")
    missing = []
    if not low_state_ready:
        missing.append("low_state")
    missing.extend(
        name for name, received_at in latest_pose_time.items() if received_at <= 0.0
    )
    missing.extend(
        name
        for name, received_at in latest_pose_time.items()
        if received_at > 0.0 and now - received_at > stale_timeout
    )
    return (
        "startup stream timeout before proposal readiness: "
        f"missing_or_stale={sorted(set(missing))}; "
        f"low_state={'ready' if low_state_ready else 'missing'}; "
        f"corrected_pose=[{', '.join(pose_status)}]; "
        f"policy_pose_missing={policy_pose_missing}; "
        f"required_pose_age<={stale_timeout:.3f}s"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM_ROOT)
    parser.add_argument(
        "--policy-config", type=Path, default=DEFAULT_ARTIFACT_DIR / "policy.yaml"
    )
    parser.add_argument(
        "--model", type=Path, default=DEFAULT_ARTIFACT_DIR / "student.onnx"
    )
    parser.add_argument("--model-json", type=Path)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--motion-meta", type=Path, default=DEFAULT_MOTION_META)
    parser.add_argument(
        "--object-name",
        default="suitcase",
        help="primary mocap body used by the policy (default: suitcase)",
    )
    parser.add_argument(
        "--object-port", type=int, default=5561,
        help="ZMQ port for the primary object pose",
    )
    parser.add_argument(
        "--aux-object-name",
        default=None,
        help="optional second mocap body required by object observations",
    )
    parser.add_argument("--aux-object-port", type=int, default=5562)
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
    parser.add_argument("--completion-file", type=Path)
    parser.add_argument("--completion-ack-file", type=Path)
    parser.add_argument("--completion-timeout", type=float, default=10.0)
    parser.add_argument("--start-timeout", type=float, default=60.0)
    parser.add_argument("--stabilization-timeout", type=float, default=300.0)
    parser.add_argument(
        "--ft-port",
        type=int,
        default=0,
        help="consume timestamped wrist F/T frames; 0 disables F/T input",
    )
    parser.add_argument("--ft-max-age-ms", type=int, default=100)
    parser.add_argument("--ft-calibration", type=Path)
    parser.add_argument("--require-both-ft-valid", action="store_true")
    parser.add_argument(
        "--residual-mode", choices=("off", "shadow", "c1", "c2"), default="off"
    )
    parser.add_argument("--residual", type=Path, default=DEFAULT_RESIDUAL)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if (
        args.steps <= 0
        or args.rate <= 0
        or args.ort_num_threads <= 0
        or args.ft_max_age_ms <= 0
    ):
        raise ValueError("steps, rates, thread counts, and F/T max age must be positive")
    if not 0 <= args.ft_port <= 65535:
        raise ValueError("ft-port must be 0 or a valid TCP port")
    if not 1 <= args.object_port <= 65535 or not 1 <= args.aux_object_port <= 65535:
        raise ValueError("object ports must be valid TCP ports")
    if args.object_name == "pelvis" or args.aux_object_name == "pelvis":
        raise ValueError("object names must differ from pelvis")
    if args.aux_object_name and args.aux_object_name == args.object_name:
        raise ValueError("primary and auxiliary object names must differ")
    if args.aux_object_name and args.aux_object_port == args.object_port:
        raise ValueError("primary and auxiliary object ports must differ")
    if args.require_both_ft_valid and args.ft_port == 0:
        raise ValueError("--require-both-ft-valid requires --ft-port")
    if args.ft_calibration is not None:
        if args.ft_port == 0:
            raise ValueError("--ft-calibration requires --ft-port")
        if not args.ft_calibration.is_file():
            raise FileNotFoundError(
                f"F/T calibration is missing: {args.ft_calibration}"
            )
    if args.residual_mode != "off":
        if args.ft_port == 0:
            raise ValueError("residual mode requires --ft-port")
        if not args.residual.is_file():
            raise FileNotFoundError(f"Cross residual model is missing: {args.residual}")
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
            args.completion_timeout,
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
            or args.completion_file is None
            or args.completion_ack_file is None
        ):
            raise ValueError(
                "proposal mode requires ready, start, motion-start, pose-status, "
                "completion, and completion-ack files"
            )
    elif any(
        path is not None
        for path in (
            args.ready_file,
            args.start_file,
            args.motion_start_file,
            args.pose_status_file,
            args.completion_file,
            args.completion_ack_file,
        )
    ):
        raise ValueError("ready/start files require --proposal-port")
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
    model_json = (
        args.model_json.resolve()
        if args.model_json
        else model_path.with_suffix(".json")
    )
    with tempfile.TemporaryDirectory(prefix="hdmi_shadow_model_") as temp_dir:
        if not model_json.exists() and args.model_json is None:
            model_json = model_path.parent / "policy.json"
        if not model_json.is_file():
            raise FileNotFoundError(f"HDMI model metadata is missing: {model_json}")
        model_metadata = json.loads(model_json.read_text())
        model_input_shapes = model_metadata.get("in_shapes", [[[], [], []]])[0]
        if len(model_input_shapes) != 3:
            raise ValueError("HDMI model metadata must describe command, policy, and object inputs")
        expected_input_shapes = {
            "command": tuple(model_input_shapes[0]),
            "policy": tuple(model_input_shapes[1]),
            "object": tuple(model_input_shapes[2]),
        }
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
            if args.residual_mode != "off":
                residual_probe = CrossResidual(
                    build_inference_module(str(args.residual.resolve()), "onnx-cpu")
                )
                residual_probe.step(
                    wrist_tokens=np.zeros((1, 2, 16, 14), dtype=np.float32),
                    proprio=np.zeros((1, 64), dtype=np.float32),
                    a_nom_history=np.zeros((1, 23, 3), dtype=np.float32),
                    previous_a_total=np.zeros((1, 23), dtype=np.float32),
                )
            print(
                "nominal shadow contract: "
                f"inputs={expected_input_shapes} action={(1, policy.num_actions)} "
                f"joints={policy.num_dofs} command_sender=disabled "
                f"ft_port={args.ft_port} residual_mode={args.residual_mode}"
            )
            return 0

        context = zmq.Context.instance()
        pose_sockets = {
            "pelvis": _pose_monitor(context, 5555),
            args.object_name: _pose_monitor(context, args.object_port),
        }
        if args.aux_object_name:
            pose_sockets[args.aux_object_name] = _pose_monitor(
                context, args.aux_object_port
            )
        latest_pose: dict[str, np.ndarray] = {}
        latest_pose_time = {name: 0.0 for name in pose_sockets}
        ft_receiver = (
            ResidualFTReceiver(args.ft_port) if args.ft_port > 0 else None
        )
        latest_ft_frame: ResidualFTFrame | None = None

        policy.state_dict = {"action": np.zeros(policy.num_actions, dtype=np.float32)}
        policy.perf_dict = {}
        deadline = time.monotonic() + args.startup_timeout
        low_state_ready = False
        required_pose_names = {"pelvis", args.object_name}
        if args.aux_object_name:
            required_pose_names.add(args.aux_object_name)
        policy_pose_missing = [
            f"{name}_{field}"
            for name in sorted(required_pose_names)
            for field in ("pos", "quat")
        ]
        while time.monotonic() < deadline:
            now = time.monotonic()
            for name, socket in pose_sockets.items():
                value = _drain_pose(socket)
                if value is not None:
                    latest_pose[name] = value
                    latest_pose_time[name] = now
            low_state_ready = policy.state_processor._prepare_low_state()
            if ft_receiver is not None:
                latest_ft_frame = None
                try:
                    candidate = ft_receiver.receive_latest(
                        timeout_ms=0, max_age_ms=args.ft_max_age_ms
                    )
                    if (
                        not args.require_both_ft_valid
                        or np.all(candidate.token[:, 13] > 0.5)
                    ):
                        latest_ft_frame = candidate
                except RuntimeError:
                    latest_ft_frame = None
            policy_pose_missing = [
                f"{name}_{field}"
                for name in sorted(required_pose_names)
                for field in ("pos", "quat")
                if policy.state_processor.get_mocap_data(f"{name}_{field}") is None
            ]
            local_pose_ready = all(
                latest_pose_time[name] > 0.0
                and now - latest_pose_time[name] <= args.stale_timeout
                for name in pose_sockets
            )
            ft_ready = ft_receiver is None or latest_ft_frame is not None
            if (
                low_state_ready
                and not policy_pose_missing
                and local_pose_ready
                and ft_ready
            ):
                break
            time.sleep(0.01)
        else:
            now = time.monotonic()
            raise RuntimeError(
                _startup_stream_timeout_message(
                    low_state_ready=low_state_ready,
                    policy_pose_missing=policy_pose_missing,
                    latest_pose_time=latest_pose_time,
                    now=now,
                    stale_timeout=args.stale_timeout,
                )
                + (
                    "; wrist_ft=missing_or_stale"
                    if ft_receiver is not None and latest_ft_frame is None
                    else ""
                )
            )

        current = np.asarray(policy.state_processor.joint_pos, dtype=np.float32)
        motion_init = _motion_init_pose(
            args.motion,
            args.motion_meta,
            list(policy.isaac_joint_names),
        )
        initial_error = float(np.max(np.abs(current - motion_init)))
        if initial_error > args.max_initial_error:
            raise RuntimeError(
                f"robot is not at {args.object_name} init pose: max error {initial_error:.3f} rad "
                f"> {args.max_initial_error:.3f} rad"
            )
        reference_placement = _reference_placement(
            args.motion, args.motion_meta, args.object_name
        )
        live_placement = _live_placement(
            latest_pose["pelvis"], latest_pose[args.object_name]
        )
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
            failure_details = ", ".join(
                f"{name}: error={placement_error[name]:.4f}, "
                f"limit={placement_limits[name]:.4f}"
                for name in failed_placement
            )
            raise RuntimeError(
                f"{args.object_name} frame-0 placement check failed: "
                f"{failure_details}; live={live_placement}; "
                f"reference={reference_placement}"
            )

        policy.use_policy_action = True
        policy.get_ready_state = False
        policy.reset()
        if args.reference_mode == "frozen" or proposal_enabled:
            _freeze_reference_at_frame_zero(policy)

        reference_states = _load_reference_states(args.motion)
        residual_controller = None
        action_joint_indices = np.asarray(
            [policy.isaac_joint_names.index(name) for name in ACTION_JOINT_NAMES],
            dtype=np.int64,
        )
        full_joint_indices = np.asarray(
            [policy.isaac_joint_names.index(name) for name in G1_JOINT_NAMES],
            dtype=np.int64,
        )
        residual_default = policy.default_dof_angles[full_joint_indices].copy()
        if args.residual_mode != "off":
            if tuple(policy.policy_joint_names) != tuple(ACTION_JOINT_NAMES):
                raise ValueError("HDMI action order does not match Cross residual order")
            velocity_ids, _, velocity_values = resolve_matching_names_values(
                robot_config["joint_velocity_limit"],
                policy.isaac_joint_names,
                preserve_order=True,
                strict=False,
            )
            velocity_full = np.zeros(len(policy.isaac_joint_names), dtype=np.float32)
            velocity_full[velocity_ids] = velocity_values
            residual_controller = HDMIResidualController(
                CrossResidual(
                    build_inference_module(str(args.residual.resolve()), "onnx-cpu")
                ),
                mode=args.residual_mode,
                default_joint_pos=policy.default_dof_angles[action_joint_indices],
                action_scale=policy.action_scale,
                joint_lower=policy.joint_pos_lower_limit[action_joint_indices],
                joint_upper=policy.joint_pos_upper_limit[action_joint_indices],
                velocity_limit=velocity_full[action_joint_indices],
                control_dt=1.0 / args.rate,
            )

        # The upstream observation objects read mocap through this method. A
        # local override lets the runner provide one coherent live pose set.
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
                raise RuntimeError(
                    "policy joint set does not match canonical G1 proposal order"
                )
            proposal_joint_indices = np.asarray(
                [policy.isaac_joint_names.index(name) for name in G1_JOINT_NAMES],
                dtype=np.int64,
            )
            proposal_socket = context.socket(zmq.PUB)
            proposal_socket.setsockopt(zmq.SNDHWM, 1)
            proposal_socket.setsockopt(zmq.LINGER, 0)
            proposal_socket.bind(f"tcp://127.0.0.1:{args.proposal_port}")
            if args.ready_file.exists():
                raise RuntimeError(
                    f"proposal ready file already exists: {args.ready_file}"
                )
            args.ready_file.write_text("ready\n")
            deadline = time.monotonic() + args.start_timeout
            while not args.start_file.exists():
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "timed out waiting for guarded-controller start signal"
                    )
                time.sleep(0.02)
            stabilization_started_at = time.monotonic()
            pose_runtime_status = _write_runtime_status(
                args.pose_status_file, None, "ready"
            )
        records: dict[str, list[np.ndarray | float | int | str | bool]] = {
            key: []
            for key in (
                "time_ns",
                "monotonic_ns",
                "phase",
                "reference_step",
                "low_state_tick",
                "low_state_age_s",
                "pelvis_age_s",
                "suitcase_age_s",
                "object_age_s",
                "inference_ms",
                "residual_inference_ms",
                "policy_total_ms",
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
                "joint_torque",
                "pelvis_pose",
                "suitcase_pose",
                "object_pose",
                "aux_object_pose",
                "target_joint_pos",
                "target_joint_vel",
                "target_body_pos_w",
                "target_body_quat_w",
                "target_body_lin_vel_w",
                "target_body_ang_vel_w",
                "target_object_contact",
                "ft_frame_fresh",
                "ft_timestamp_ns",
                "ft_publish_monotonic_ns",
                "ft_publish_age_s",
                "ft_kinematics_monotonic_ns",
                "ft_sample_received_monotonic_ns",
                "ft_sample_age_s",
                "ft_sample_kinematics_skew_s",
                "ft_sample_source_time_ns",
                "ft_sample_sequence",
                "ft_sample_device_id",
                "ft_sample_status",
                "ft_sequence",
                "ft_contact_count",
                "ft_total_force_norm",
                "ft_token",
                "ft_wrench_base_yaw",
                "ft_wrench_sensor",
                "nominal_action",
                "raw_residual",
                "delta_bounded",
                "delta_gated",
                "delta_safe",
                "composed_action",
                "applied_action",
                "contact_target",
                "contact_gain",
                "residual_proprio",
                "wrist_token_history",
                "nominal_action_history",
                "previous_applied_action",
                "policy_ood_ratio",
                "command_ood_ratio",
                "object_ood_ratio",
            )
        }
        base_record_metadata = {
            "schema": "somaforce_hdmi_suitcase_hardware_record_v4",
            "residual_mode": args.residual_mode,
            "joint_names": list(policy.isaac_joint_names),
            "policy_joint_names": list(policy.policy_joint_names),
            "target_joint_names": json.loads(args.motion_meta.read_text())["joint_names"],
            "target_body_names": json.loads(args.motion_meta.read_text())["body_names"],
            "wrist_order": ["left", "right"],
            "policy_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
            "motion": str(args.motion.resolve()),
            "ft_calibration": (
                str(args.ft_calibration.resolve())
                if args.ft_calibration is not None
                else None
            ),
            "ft_calibration_sha256": (
                hashlib.sha256(args.ft_calibration.read_bytes()).hexdigest()
                if args.ft_calibration is not None
                else None
            ),
        }
        recorder = ChunkedNpzRecorder(
            args.output,
            tuple(records),
            base_metadata=base_record_metadata,
        )
        period = 1.0 / args.rate
        next_step = time.perf_counter()
        last_tick = None
        last_tick_time = time.monotonic()
        sequence = 0
        motion_step = 0
        stabilization_steps = 0
        termination = {"reason": "abnormal_exit"}
        stop_request = {"exit_code": None}

        def save_partial_record() -> None:
            try:
                frame_count = recorder.close(
                    records,
                    status="partial",
                    reason=termination["reason"],
                )
                partial_metadata = {
                    **base_record_metadata,
                    "complete": False,
                    "termination_reason": termination["reason"],
                    "steps": frame_count,
                }
                finalize_chunked_recording(
                    args.output,
                    metadata=partial_metadata,
                    complete=False,
                    reason=termination["reason"],
                )
                atomic_write_json(
                    args.output.with_suffix(".partial.json"),
                    {
                        "schema": partial_metadata["schema"],
                        "complete": False,
                        "termination_reason": termination["reason"],
                        "steps": frame_count,
                        "output": str(args.output),
                        "recording_directory": str(recorder.directory),
                    },
                )
            except Exception as exc:
                print(f"failed to finalize partial recording: {exc}", file=sys.stderr)

        def request_stop(signum, _frame) -> None:
            termination["reason"] = signal.Signals(signum).name
            stop_request["exit_code"] = 128 + int(signum)

        atexit.register(save_partial_record)
        previous_signal_handlers = {
            signum: signal.signal(signum, request_stop)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }
        while motion_step < args.steps:
            if stop_request["exit_code"] is not None:
                raise SystemExit(stop_request["exit_code"])
            stabilizing = bool(proposal_enabled and not args.motion_start_file.exists())
            if stabilizing:
                if (
                    time.monotonic() - stabilization_started_at
                    > args.stabilization_timeout
                ):
                    raise RuntimeError(
                        "timed out waiting for operator motion-start signal"
                    )
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
            for name, socket in pose_sockets.items():
                value = _drain_pose(socket)
                if value is not None:
                    latest_pose[name] = value
                    latest_pose_time[name] = now
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
            stale_pose = [
                name for name, age in pose_ages.items() if age > args.stale_timeout
            ]
            if stale_pose:
                if proposal_enabled:
                    pose_runtime_status = _write_runtime_status(
                        args.pose_status_file,
                        pose_runtime_status,
                        f"blocked:{','.join(stale_pose)}",
                    )
                phase_context = " during stabilization" if stabilizing else ""
                raise RuntimeError(
                    f"multi-marker pose stream stale{phase_context}: {stale_pose}"
                )
            if proposal_enabled:
                pose_runtime_status = _write_runtime_status(
                    args.pose_status_file, pose_runtime_status, "ready"
                )

            control_monotonic_ns = time.monotonic_ns()
            control_time_ns = time.time_ns()

            if ft_receiver is None:
                ft_frame = ResidualFTFrame.unavailable(
                    timestamp_ns=control_time_ns, sequence=0
                )
                ft_timing_frame = ft_frame
                ft_frame_fresh = False
            else:
                ft_frame = ft_receiver.receive_latest_or_unavailable(
                    timeout_ms=0,
                    max_age_ms=args.ft_max_age_ms,
                    now_ns=control_time_ns,
                    now_monotonic_ns=control_monotonic_ns,
                )
                ft_timing_frame = ft_receiver.latest or ft_frame
                ft_frame_fresh = ft_frame is ft_receiver.latest

            live_pose = {name: value.copy() for name, value in latest_pose.items()}

            effective_mocap_data["pelvis_pos"] = live_pose["pelvis"][:3]
            effective_mocap_data["pelvis_quat"] = live_pose["pelvis"][3:]
            for pose_name, pose in live_pose.items():
                if pose_name == "pelvis":
                    continue
                effective_mocap_data[f"{pose_name}_pos"] = pose[:3]
                effective_mocap_data[f"{pose_name}_quat"] = pose[3:]

            policy.update()
            observations = policy.prepare_obs_for_rl()
            for name, expected_shape in expected_input_shapes.items():
                value = np.asarray(observations[name], dtype=np.float32)
                if value.shape != expected_shape or not np.isfinite(value).all():
                    raise RuntimeError(
                        f"invalid {name} observation: shape={value.shape}, expected={expected_shape}"
                    )
            policy.state_dict.update(observations)
            policy.state_dict["is_init"] = np.zeros(1, dtype=bool)
            inference_start = time.perf_counter()
            nominal_action, q_target, next_state = policy.policy(policy.state_dict)
            inference_ms = (time.perf_counter() - inference_start) * 1000.0
            nominal_action = np.asarray(nominal_action, dtype=np.float32)
            q_target = np.asarray(q_target, dtype=np.float32)
            if (
                nominal_action.shape != (policy.num_actions,)
                or not np.isfinite(nominal_action).all()
            ):
                raise RuntimeError(
                    f"invalid nominal action: shape={nominal_action.shape}"
                )
            if q_target.shape != (policy.num_dofs,) or not np.isfinite(q_target).all():
                raise RuntimeError(f"invalid nominal q_target: shape={q_target.shape}")

            if residual_controller is not None:
                residual_start = time.perf_counter()
                proprio = residual_proprio(
                    policy.state_processor,
                    default_joint_pos=residual_default,
                )
                current_joint_pos = np.asarray(
                    policy.state_processor.joint_pos, dtype=np.float32
                )[action_joint_indices][None, :]
                previous_applied_action = residual_controller.executed_history[
                    0, :, 0
                ].copy()
                residual_step = residual_controller.step(
                    nominal=nominal_action[None, :],
                    wrist_frame=ft_frame.token,
                    proprio=proprio,
                    current_joint_pos=current_joint_pos,
                )
                action = residual_step.applied[0].copy()
                q_target = policy.default_dof_angles.copy()
                q_target[policy.controlled_joint_indices] += (
                    action * policy.action_scale
                )
                raw_residual = residual_step.raw_residual[0]
                delta_bounded = residual_step.delta_bounded[0]
                delta_gated = residual_step.delta_gated[0]
                delta_safe = residual_step.delta_safe[0]
                composed_action = residual_step.composed[0]
                contact_target = residual_step.contact_target
                contact_gain = residual_step.contact_gain
                wrist_token_history = residual_controller.wrist_history[0].copy()
                nominal_action_history = residual_controller.nominal_history[0].copy()
                residual_inference_ms = (
                    time.perf_counter() - residual_start
                ) * 1000.0
            else:
                action = nominal_action.copy()
                raw_residual = np.zeros_like(action)
                delta_bounded = np.zeros_like(action)
                delta_gated = np.zeros_like(action)
                delta_safe = np.zeros_like(action)
                composed_action = action.copy()
                contact_target = 0.0
                contact_gain = 0.0
                proprio = np.zeros((1, 64), dtype=np.float32)
                wrist_token_history = np.zeros((2, 16, 14), dtype=np.float32)
                nominal_action_history = np.zeros((23, 3), dtype=np.float32)
                previous_applied_action = np.zeros(23, dtype=np.float32)
                residual_inference_ms = 0.0
            policy_total_ms = (time.perf_counter() - inference_start) * 1000.0
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

            reference_index = min(
                reference_step, reference_states["joint_pos"].shape[0] - 1
            )
            raw_joint_torque = low_state.joint_torques
            if raw_joint_torque is None:
                joint_torque = np.full(policy.num_dofs, np.nan, dtype=np.float32)
            else:
                joint_torque = np.asarray(raw_joint_torque, dtype=np.float32)[
                    policy.state_processor.joint_indices_in_source
                ]
            if ft_timing_frame.publish_monotonic_ns > 0:
                ft_publish_age_s = (
                    control_monotonic_ns - ft_timing_frame.publish_monotonic_ns
                ) / 1e9
            else:
                ft_publish_age_s = (
                    control_time_ns - ft_timing_frame.timestamp_ns
                ) / 1e9
            ft_sample_age_s = np.full(2, np.nan, dtype=np.float64)
            received_mask = ft_timing_frame.sample_received_monotonic_ns >= 0
            ft_sample_age_s[received_mask] = (
                control_monotonic_ns
                - ft_timing_frame.sample_received_monotonic_ns[received_mask]
            ) / 1e9
            ft_sample_kinematics_skew_s = np.full(2, np.nan, dtype=np.float64)
            if ft_timing_frame.kinematics_monotonic_ns >= 0:
                ft_sample_kinematics_skew_s[received_mask] = (
                    ft_timing_frame.sample_received_monotonic_ns[received_mask]
                    - ft_timing_frame.kinematics_monotonic_ns
                ) / 1e9

            records["time_ns"].append(control_time_ns)
            records["monotonic_ns"].append(control_monotonic_ns)
            records["phase"].append(phase)
            records["reference_step"].append(reference_step)
            records["low_state_tick"].append(tick)
            records["low_state_age_s"].append(low_state_age)
            records["pelvis_age_s"].append(pose_ages["pelvis"])
            records["suitcase_age_s"].append(pose_ages.get("suitcase", pose_ages[args.object_name]))
            records["object_age_s"].append(pose_ages[args.object_name])
            records["inference_ms"].append(inference_ms)
            records["residual_inference_ms"].append(residual_inference_ms)
            records["policy_total_ms"].append(policy_total_ms)
            records["command"].append(observations["command"][0].copy())
            records["policy"].append(observations["policy"][0].copy())
            records["object"].append(observations["object"][0].copy())
            records["action"].append(action.copy())
            records["q_target"].append(q_target.copy())
            records["joint_pos"].append(policy.state_processor.joint_pos.copy())
            records["joint_vel"].append(policy.state_processor.joint_vel.copy())
            records["root_quat"].append(policy.state_processor.root_quat_b.copy())
            records["root_ang_vel"].append(policy.state_processor.root_ang_vel_b.copy())
            records["joint_torque"].append(joint_torque.copy())
            records["pelvis_pose"].append(live_pose["pelvis"].copy())
            records["suitcase_pose"].append(live_pose.get("suitcase", live_pose[args.object_name]).copy())
            records["object_pose"].append(live_pose[args.object_name].copy())
            records["aux_object_pose"].append(
                live_pose[args.aux_object_name].copy()
                if args.aux_object_name
                else np.full(7, np.nan, dtype=np.float32)
            )
            for key in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
                "object_contact",
            ):
                records[f"target_{key}"].append(
                    reference_states[key][reference_index].copy()
                )
            records["ft_frame_fresh"].append(ft_frame_fresh)
            records["ft_timestamp_ns"].append(ft_timing_frame.timestamp_ns)
            records["ft_publish_monotonic_ns"].append(
                ft_timing_frame.publish_monotonic_ns
            )
            records["ft_publish_age_s"].append(ft_publish_age_s)
            records["ft_kinematics_monotonic_ns"].append(
                ft_timing_frame.kinematics_monotonic_ns
            )
            records["ft_sample_received_monotonic_ns"].append(
                ft_timing_frame.sample_received_monotonic_ns.copy()
            )
            records["ft_sample_age_s"].append(ft_sample_age_s)
            records["ft_sample_kinematics_skew_s"].append(
                ft_sample_kinematics_skew_s
            )
            records["ft_sample_source_time_ns"].append(
                ft_timing_frame.sample_source_time_ns.copy()
            )
            records["ft_sample_sequence"].append(
                ft_timing_frame.sample_sequence.copy()
            )
            records["ft_sample_device_id"].append(
                ft_timing_frame.sample_device_id.copy()
            )
            records["ft_sample_status"].append(
                ft_timing_frame.sample_status.copy()
            )
            records["ft_sequence"].append(ft_timing_frame.sequence)
            records["ft_contact_count"].append(ft_timing_frame.contact_count)
            records["ft_total_force_norm"].append(
                ft_timing_frame.total_force_norm
            )
            records["ft_token"].append(ft_frame.token.copy())
            records["ft_wrench_base_yaw"].append(
                ft_timing_frame.wrench_base_yaw.copy()
            )
            records["ft_wrench_sensor"].append(
                ft_timing_frame.wrench_sensor.copy()
            )
            records["nominal_action"].append(nominal_action.copy())
            records["raw_residual"].append(raw_residual.copy())
            records["delta_bounded"].append(delta_bounded.copy())
            records["delta_gated"].append(delta_gated.copy())
            records["delta_safe"].append(delta_safe.copy())
            records["composed_action"].append(composed_action.copy())
            records["applied_action"].append(action.copy())
            records["contact_target"].append(contact_target)
            records["contact_gain"].append(contact_gain)
            records["residual_proprio"].append(proprio[0].copy())
            records["wrist_token_history"].append(wrist_token_history)
            records["nominal_action_history"].append(nominal_action_history)
            records["previous_applied_action"].append(previous_applied_action)
            for name in ("policy_ood_ratio", "command_ood_ratio", "object_ood_ratio"):
                value = np.asarray(next_state.get(name, np.nan), dtype=np.float32)
                records[name].append(float(value.reshape(-1)[0]))
            loop_ms = (time.perf_counter() - loop_start) * 1000.0
            records["loop_ms"].append(loop_ms)
            recorder.capture(records)
            if stop_request["exit_code"] is not None:
                raise SystemExit(stop_request["exit_code"])
            next_step += period
            sleep_s = next_step - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                # A delayed inference must not publish several reference steps in a burst.
                next_step = time.perf_counter()

        if proposal_socket is not None:
            assert proposal_joint_indices is not None
            atomic_write_json(
                args.completion_file,
                {
                    "schema": "somaforce_policy_motion_complete_v1",
                    "steps": args.steps,
                    "recorded_frames": len(records["time_ns"]),
                    "last_reference_step": int(reference_step),
                    "last_sequence": int(sequence),
                },
            )
            completion_deadline = time.monotonic() + args.completion_timeout
            next_completion_publish = time.perf_counter()
            while not args.completion_ack_file.exists():
                if stop_request["exit_code"] is not None:
                    raise SystemExit(stop_request["exit_code"])
                if time.monotonic() >= completion_deadline:
                    raise RuntimeError(
                        "timed out waiting for guarded-controller completion hold"
                    )
                sequence += 1
                proposal = NominalProposal(
                    source_time_ns=time.monotonic_ns(),
                    sequence=sequence,
                    reference_step=reference_step,
                    action=action,
                    q_target=q_target[proposal_joint_indices],
                )
                proposal_socket.send(proposal.to_bytes(), flags=zmq.DONTWAIT)
                next_completion_publish += period
                time.sleep(max(0.0, next_completion_publish - time.perf_counter()))

        if sender.send_calls != 0:
            raise RuntimeError(
                f"shadow command sender was called {sender.send_calls} times"
            )
        recorder.close(records, status="complete")
        arrays = {name: np.asarray(values) for name, values in records.items()}
        action_abs = np.abs(arrays["action"])
        target_margin = np.minimum(
            arrays["q_target"] - policy.joint_pos_lower_limit,
            policy.joint_pos_upper_limit - arrays["q_target"],
        )
        target_error = np.abs(arrays["q_target"] - arrays["joint_pos"])
        target_steps = np.abs(np.diff(arrays["q_target"], axis=0))
        target_step_p95 = (
            float(np.percentile(target_steps, 95)) if target_steps.size else 0.0
        )
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
            "schema": "somaforce_hdmi_nominal_hardware_record_v1",
            "complete": True,
            "object_name": args.object_name,
            "aux_object_name": args.aux_object_name,
            "result": "pass" if violation_count == 0 else "review",
            "steps": int(arrays["time_ns"].shape[0]),
            "motion_steps": args.steps,
            "stabilization_steps": stabilization_steps,
            "pose_source": "multi_marker_live_only",
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
            "ft_enabled": ft_receiver is not None,
            "ft_port": args.ft_port if ft_receiver is not None else None,
            "ft_calibration": (
                str(args.ft_calibration.resolve())
                if args.ft_calibration is not None
                else None
            ),
            "ft_calibration_sha256": (
                hashlib.sha256(args.ft_calibration.read_bytes()).hexdigest()
                if args.ft_calibration is not None
                else None
            ),
            "ft_fresh_fraction": float(np.mean(arrays["ft_frame_fresh"])),
            "ft_both_valid_fraction": float(
                np.mean(np.all(arrays["ft_token"][:, :, 13] > 0.5, axis=1))
            ),
            "ft_sample_age_ms_p95": (
                float(
                    np.percentile(
                        arrays["ft_sample_age_s"][
                            np.isfinite(arrays["ft_sample_age_s"])
                        ],
                        95,
                    )
                    * 1000.0
                )
                if np.isfinite(arrays["ft_sample_age_s"]).any()
                else None
            ),
            "ft_sample_kinematics_abs_skew_ms_p95": (
                float(
                    np.percentile(
                        np.abs(
                            arrays["ft_sample_kinematics_skew_s"][
                                np.isfinite(
                                    arrays["ft_sample_kinematics_skew_s"]
                                )
                            ]
                        ),
                        95,
                    )
                    * 1000.0
                )
                if np.isfinite(arrays["ft_sample_kinematics_skew_s"]).any()
                else None
            ),
            "residual_mode": args.residual_mode,
            "residual": (
                str(args.residual.resolve()) if args.residual_mode != "off" else None
            ),
            "residual_sha256": (
                hashlib.sha256(args.residual.read_bytes()).hexdigest()
                if args.residual_mode != "off"
                else None
            ),
            "inference_ms": _percentiles(arrays["inference_ms"]),
            "residual_inference_ms": _percentiles(
                arrays["residual_inference_ms"]
            ),
            "policy_total_ms": _percentiles(arrays["policy_total_ms"]),
            "loop_ms": _percentiles(arrays["loop_ms"]),
            "loop_overrun_count": int(
                np.count_nonzero(arrays["loop_ms"] > period * 1000.0)
            ),
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
            "object_age_ms_max": float(np.max(arrays["object_age_s"]) * 1000.0),
        }
        metadata = {
            "schema": summary["schema"],
            "shadow": not proposal_enabled,
            "robot_command_output": False,
            "proposal_output": proposal_enabled,
            "proposal_joint_names": list(G1_JOINT_NAMES) if proposal_enabled else None,
            "joint_names": list(policy.isaac_joint_names),
            "policy_joint_names": list(policy.policy_joint_names),
            "target_joint_names": json.loads(args.motion_meta.read_text())["joint_names"],
            "target_body_names": json.loads(args.motion_meta.read_text())["body_names"],
            "default_joint_pos": policy.default_dof_angles.tolist(),
            "action_scale": policy.action_scale.tolist(),
            "joint_pos_lower_limit": policy.joint_pos_lower_limit.tolist(),
            "joint_pos_upper_limit": policy.joint_pos_upper_limit.tolist(),
            "wrist_order": ["left", "right"],
            "ft_token_order": [
                "force_xyz_base_yaw_normalized",
                "moment_xyz_base_yaw_normalized",
                "linear_velocity_xyz_base_yaw",
                "angular_velocity_xyz_base_yaw",
                "contact_probability",
                "quality",
            ],
            "ft_stale_semantics": {
                "ft_token": "effective policy input; zero quality when frame is stale",
                "ft_wrench_sensor": "latest received raw sample retained for audit",
                "ft_wrench_base_yaw": "latest transformed sample retained for audit",
                "freshness_flag": "ft_frame_fresh",
            },
            "ft_calibration_config": (
                yaml.safe_load(args.ft_calibration.read_text())
                if args.ft_calibration is not None
                else None
            ),
            "time_alignment": {
                "row": "one policy control tick",
                "freshness_clock": "CLOCK_MONOTONIC on the deploy host",
                "time_ns": "CLOCK_REALTIME for cross-log correlation only",
                "ft_sample_source_time_ns": (
                    "SDK clock domain; compare directly only when SDK and deploy share a host"
                ),
            },
            "history_order": {
                "wrist_token_history": "oldest_to_newest",
                "nominal_action_history": "newest_to_oldest",
                "previous_applied_action": "previous control tick",
            },
            "summary": summary,
        }
        finalize_chunked_recording(
            args.output,
            metadata=metadata,
            complete=True,
        )
        summary_path = args.output.with_suffix(".summary.json")
        atomic_write_json(summary_path, summary)
        atexit.unregister(save_partial_record)
        for signum, previous_handler in previous_signal_handlers.items():
            signal.signal(signum, previous_handler)
        for socket in pose_sockets.values():
            socket.close(linger=0)
        if ft_receiver is not None:
            ft_receiver.close()
        if proposal_socket is not None:
            proposal_socket.close(linger=0)
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"saved nominal inference record: {args.output}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
