#!/usr/bin/env python3
"""Run SONIC with live G1/F-T input and publish guarded nominal proposals.

The runner never opens the low-command port. Robot commands remain owned by
``suitcase_safe_controller.py``; this process only subscribes to low-state,
publishes SNP2 proposals, and writes crash-recoverable policy/F-T records.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import json
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml
import zmq

from sim2real.config.robots.g1 import G1_CFG
from sim2real.rl_policy.tracking import Tracking, TrackingArgs
from sim2real.utils.strings import resolve_matching_names_values
from somaforce_deploy.chunked_recording import (
    ChunkedNpzRecorder,
    atomic_write_json,
    finalize_chunked_recording,
)
from somaforce_deploy.hdmi_residual_runtime import ResidualFTReceiver
from somaforce_deploy.nominal_proposal import SonicNominalProposal


SCHEMA = "somaforce_sonic_hardware_record_v1"
REFERENCE_FIELDS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "object_contact",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--motion-root", type=Path, required=True)
    parser.add_argument("--source-motion", type=Path, required=True)
    parser.add_argument("--source-motion-meta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=573)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--proposal-port", type=int, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--start-file", type=Path, required=True)
    parser.add_argument("--motion-start-file", type=Path, required=True)
    parser.add_argument("--completion-file", type=Path, required=True)
    parser.add_argument("--completion-ack-file", type=Path, required=True)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--start-timeout", type=float, default=60.0)
    parser.add_argument("--stabilization-timeout", type=float, default=300.0)
    parser.add_argument("--completion-timeout", type=float, default=10.0)
    parser.add_argument("--state-timeout", type=float, default=0.25)
    parser.add_argument("--ft-port", type=int, default=5580)
    parser.add_argument("--ft-max-age-ms", type=int, default=100)
    parser.add_argument("--ft-calibration", type=Path, required=True)
    parser.add_argument("--require-both-ft-valid", action="store_true")
    parser.add_argument("--ort-num-threads", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--reference-time-scale", type=float, default=1.0)
    parser.add_argument("--max-proposal-step", type=float, default=0.50)
    parser.add_argument("--proposal-limit-tolerance", type=float, default=0.05)
    return parser


def _load_reference(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        missing = [name for name in REFERENCE_FIELDS if name not in source]
        if missing:
            raise ValueError(f"source HDMI motion is missing fields: {missing}")
        reference = {
            name: np.asarray(source[name]).copy() for name in REFERENCE_FIELDS
        }
    lengths = {value.shape[0] for value in reference.values()}
    if len(lengths) != 1 or next(iter(lengths)) <= 0:
        raise ValueError(f"source HDMI motion has inconsistent lengths: {lengths}")
    return reference


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _ft_timing(frame, *, control_time_ns: int, control_monotonic_ns: int):
    if frame.publish_monotonic_ns > 0:
        publish_age_s = (control_monotonic_ns - frame.publish_monotonic_ns) / 1e9
    else:
        publish_age_s = (control_time_ns - frame.timestamp_ns) / 1e9
    sample_age_s = np.full(2, np.nan, dtype=np.float64)
    received = frame.sample_received_monotonic_ns >= 0
    sample_age_s[received] = (
        control_monotonic_ns - frame.sample_received_monotonic_ns[received]
    ) / 1e9
    sample_skew_s = np.full(2, np.nan, dtype=np.float64)
    if frame.kinematics_monotonic_ns >= 0:
        sample_skew_s[received] = (
            frame.sample_received_monotonic_ns[received]
            - frame.kinematics_monotonic_ns
        ) / 1e9
    return publish_age_s, sample_age_s, sample_skew_s


def main() -> int:
    args = _parser().parse_args()
    if any(
        value <= 0
        for value in (
            args.steps,
            args.rate,
            args.startup_timeout,
            args.start_timeout,
            args.stabilization_timeout,
            args.completion_timeout,
            args.state_timeout,
            args.ft_max_age_ms,
            args.ort_num_threads,
            args.chunk_size,
            args.reference_time_scale,
            args.max_proposal_step,
        )
    ):
        raise ValueError(
            "steps, rates, timeouts, thread count, and chunk size must be positive"
        )
    if args.proposal_limit_tolerance < 0:
        raise ValueError("proposal-limit-tolerance must be non-negative")
    if not 1 <= args.proposal_port <= 65535 or not 1 <= args.ft_port <= 65535:
        raise ValueError("proposal and F/T ports must be valid TCP ports")
    for path in (
        args.policy_config,
        args.motion_root / "manifest.json",
        args.source_motion,
        args.source_motion_meta,
        args.ft_calibration,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"required SONIC runtime file is missing: {path}")
    model_path = args.model or args.policy_config.with_suffix(".onnx")
    if not model_path.is_file():
        raise FileNotFoundError(f"SONIC ONNX is missing: {model_path}")
    os.environ["SIM2REAL_ORT_NUM_THREADS"] = str(args.ort_num_threads)

    reference = _load_reference(args.source_motion)

    config = yaml.safe_load(args.policy_config.read_text())
    config["model_path"] = str(model_path.resolve())
    with tempfile.TemporaryDirectory(prefix="sonic_proposal_") as temp_dir:
        config_path = Path(temp_dir) / "policy.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        policy = Tracking(
            args=TrackingArgs(
                policy_config=str(config_path),
                robot="g1",
                rl_rate=args.rate,
                inference_backend="onnx-cpu",
                robot_io="zmq",
                robot_io_command_output=False,
                robot_interface="",
                controller="none",
                record=False,
                motion_backend="npz",
                motion_path=str(args.motion_root.resolve()),
            )
        )
        canonical_joint_names = list(G1_CFG.joint_names)
        if set(policy.joint_names_simulation) != set(canonical_joint_names):
            raise RuntimeError("SONIC simulation joints do not match canonical G1")
        mapped_policy_names = [
            canonical_joint_names[index] for index in policy.controlled_joint_indices
        ]
        if mapped_policy_names != list(policy.policy_joint_names):
            raise RuntimeError("SONIC policy-to-G1 joint mapping is inconsistent")
        if args.steps > policy.motion_length:
            raise ValueError(
                f"requested {args.steps} steps but runtime motion has "
                f"{policy.motion_length}"
            )
        lower_ids, _, lower_values = resolve_matching_names_values(
            G1_CFG.joint_pos_lower_limit,
            canonical_joint_names,
            preserve_order=True,
            strict=False,
        )
        upper_ids, _, upper_values = resolve_matching_names_values(
            G1_CFG.joint_pos_upper_limit,
            canonical_joint_names,
            preserve_order=True,
            strict=False,
        )
        joint_lower = np.zeros(29, dtype=np.float32)
        joint_upper = np.zeros(29, dtype=np.float32)
        joint_lower[lower_ids] = lower_values
        joint_upper[upper_ids] = upper_values

        context = zmq.Context.instance()
        proposal_socket = context.socket(zmq.PUB)
        proposal_socket.setsockopt(zmq.SNDHWM, 1)
        proposal_socket.setsockopt(zmq.LINGER, 0)
        proposal_socket.bind(f"tcp://127.0.0.1:{args.proposal_port}")
        ft_receiver = ResidualFTReceiver(args.ft_port)

        startup_deadline = time.monotonic() + args.startup_timeout
        ft_frame = None
        while time.monotonic() < startup_deadline:
            low_state_ready = policy.state_processor._prepare_low_state()
            try:
                candidate = ft_receiver.receive_latest(
                    timeout_ms=0, max_age_ms=args.ft_max_age_ms
                )
                if (
                    not args.require_both_ft_valid
                    or np.all(candidate.token[:, 13] > 0.5)
                ):
                    ft_frame = candidate
            except RuntimeError:
                ft_frame = None
            if low_state_ready and ft_frame is not None:
                break
            time.sleep(0.01)
        else:
            raise RuntimeError(
                "timed out waiting for fresh G1 low-state and valid wrist F/T"
            )

        # Heading alignment must be initialized from a real robot quaternion.
        policy.state_dict = {
            "action": np.zeros(policy.num_actions, dtype=np.float32),
            "paused": True,
            "control_mode": "policy",
        }
        policy.reset()
        policy.state_dict["control_mode"] = "policy"

        fields = (
            "time_ns", "monotonic_ns", "phase", "reference_step",
            "low_state_tick", "low_state_age_s", "inference_ms", "loop_ms",
            "g1_input", "proprioception", "action", "q_target", "joint_pos",
            "joint_vel", "root_quat", "root_ang_vel", "joint_torque",
            "target_joint_pos", "target_joint_vel", "target_body_pos_w",
            "target_body_quat_w", "target_body_lin_vel_w",
            "target_body_ang_vel_w", "target_object_contact", "ft_frame_fresh",
            "ft_timestamp_ns", "ft_publish_monotonic_ns", "ft_publish_age_s",
            "ft_kinematics_monotonic_ns", "ft_sample_received_monotonic_ns",
            "ft_sample_age_s", "ft_sample_kinematics_skew_s",
            "ft_sample_source_time_ns", "ft_sample_sequence",
            "ft_sample_device_id", "ft_sample_status", "ft_sequence",
            "ft_contact_count", "ft_total_force_norm", "ft_token",
            "ft_wrench_base_yaw", "ft_wrench_sensor",
        )
        records: dict[str, list] = {name: [] for name in fields}
        base_metadata = {
            "schema": SCHEMA,
            "nominal_backend": "sonic",
            "joint_names": canonical_joint_names,
            "policy_joint_names": list(policy.policy_joint_names),
            "target_joint_names": list(policy.motion_joint_names),
            "target_body_names": list(policy.motion_body_names),
            "wrist_order": ["left", "right"],
            "policy_config": str(args.policy_config.resolve()),
            "policy_config_sha256": hashlib.sha256(
                args.policy_config.read_bytes()
            ).hexdigest(),
            "policy": str(model_path.resolve()),
            "policy_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "motion_root": str(args.motion_root.resolve()),
            "source_motion": str(args.source_motion.resolve()),
            "source_motion_sha256": hashlib.sha256(
                args.source_motion.read_bytes()
            ).hexdigest(),
            "source_motion_meta": str(args.source_motion_meta.resolve()),
            "source_motion_meta_sha256": hashlib.sha256(
                args.source_motion_meta.read_bytes()
            ).hexdigest(),
            "ft_calibration": str(args.ft_calibration.resolve()),
            "ft_calibration_sha256": hashlib.sha256(
                args.ft_calibration.read_bytes()
            ).hexdigest(),
            "ft_required": True,
            "reference_time_scale": args.reference_time_scale,
            "max_proposal_step_rad": args.max_proposal_step,
            "proposal_limit_tolerance_rad": args.proposal_limit_tolerance,
        }
        recorder = ChunkedNpzRecorder(
            args.output,
            fields,
            base_metadata=base_metadata,
            chunk_size=args.chunk_size,
        )
        termination = {"reason": "abnormal_exit"}
        stop_request = {"exit_code": None}

        def finalize_partial() -> None:
            try:
                frames = recorder.close(
                    records, status="partial", reason=termination["reason"]
                )
                metadata = {
                    **base_metadata,
                    "complete": False,
                    "termination_reason": termination["reason"],
                    "steps": frames,
                }
                finalize_chunked_recording(
                    args.output,
                    metadata=metadata,
                    complete=False,
                    reason=termination["reason"],
                )
                atomic_write_json(
                    args.output.with_suffix(".partial.json"),
                    {
                        "schema": SCHEMA,
                        "complete": False,
                        "termination_reason": termination["reason"],
                        "steps": frames,
                        "output": str(args.output.resolve()),
                        "recording_directory": str(recorder.directory.resolve()),
                    },
                )
            except Exception as exc:
                print(
                    f"failed to finalize partial SONIC recording: {exc}",
                    file=sys.stderr,
                )

        def request_stop(signum, _frame) -> None:
            termination["reason"] = signal.Signals(signum).name
            stop_request["exit_code"] = 128 + int(signum)

        atexit.register(finalize_partial)
        previous_signal_handlers = {
            signum: signal.signal(signum, request_stop)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }

        args.ready_file.write_text("ready\n")
        start_deadline = time.monotonic() + args.start_timeout
        while not args.start_file.exists():
            if stop_request["exit_code"] is not None:
                raise SystemExit(stop_request["exit_code"])
            if time.monotonic() >= start_deadline:
                raise RuntimeError("timed out waiting for guarded-controller start")
            time.sleep(0.02)

        period = 1.0 / args.rate
        next_tick = time.perf_counter()
        last_tick = None
        last_tick_time = time.monotonic()
        sequence = 0
        motion_step = 0
        stabilization_steps = 0
        stabilization_started_at = time.monotonic()
        motion_started = False
        last_action = np.zeros(policy.num_actions, dtype=np.float32)
        last_q_target = np.asarray(policy.state_processor.joint_pos, dtype=np.float32)
        last_ft_warning = 0.0
        ft_invalid_streak = 0

        while motion_step < args.steps:
            if stop_request["exit_code"] is not None:
                raise SystemExit(stop_request["exit_code"])
            stabilizing = not args.motion_start_file.exists()
            if stabilizing:
                if (
                    time.monotonic() - stabilization_started_at
                    > args.stabilization_timeout
                ):
                    raise RuntimeError(
                        "timed out waiting for operator motion-start signal"
                    )
                phase = "stabilize"
                reference_step = 0
                policy.state_dict["paused"] = True
                stabilization_steps += 1
            else:
                if not motion_started:
                    # Tracking.update increments before sampling; -1 makes the
                    # first moving update consume reference frame zero.
                    policy.motion_t[:] = -1
                    motion_started = True
                phase = "motion"
                reference_step = motion_step
                policy.state_dict["paused"] = False
                motion_step += 1

            loop_started = time.perf_counter()
            control_monotonic_ns = time.monotonic_ns()
            control_time_ns = time.time_ns()
            if not policy.state_processor._prepare_low_state():
                raise RuntimeError("G1 low-state unavailable during SONIC inference")
            state = policy.state_processor.latest_state
            assert state is not None
            tick = int(state.tick)
            if tick != last_tick:
                last_tick = tick
                last_tick_time = time.monotonic()
            low_state_age_s = time.monotonic() - last_tick_time
            if low_state_age_s > args.state_timeout:
                raise RuntimeError(f"G1 low-state stale for {low_state_age_s:.3f}s")

            ft_frame = ft_receiver.receive_latest_or_unavailable(
                timeout_ms=0,
                max_age_ms=args.ft_max_age_ms,
                now_ns=control_time_ns,
                now_monotonic_ns=control_monotonic_ns,
            )
            ft_frame_fresh = ft_frame is ft_receiver.latest
            both_ft_valid = bool(np.all(ft_frame.token[:, 13] > 0.5))
            if not ft_frame_fresh or not both_ft_valid:
                ft_invalid_streak += 1
                now = time.monotonic()
                if now - last_ft_warning >= 1.0:
                    print(
                        "WARNING: wrist F/T unavailable during SONIC inference: "
                        f"fresh={ft_frame_fresh} "
                        f"quality={ft_frame.token[:, 13].tolist()} "
                        f"streak={ft_invalid_streak}; continuing nominal control",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_ft_warning = now
            elif ft_invalid_streak:
                print(
                    f"wrist F/T recovered after {ft_invalid_streak} invalid frames",
                    file=sys.stderr,
                    flush=True,
                )
                ft_invalid_streak = 0

            policy.update()
            observations, _ = policy.prepare_obs_for_rl()
            expected_shapes = {"g1_input": (640,), "proprioception": (930,)}
            for name, expected_shape in expected_shapes.items():
                value = np.asarray(observations[name], dtype=np.float32)
                if value.shape != expected_shape or not np.isfinite(value).all():
                    raise RuntimeError(
                        f"invalid SONIC {name}: shape={value.shape}, "
                        f"expected={expected_shape}"
                    )
            policy.state_dict.update(observations)
            policy.state_dict["is_init"] = np.zeros(1, dtype=bool)
            inference_started = time.perf_counter()
            action, q_target, next_state = policy.policy(policy.state_dict)
            inference_ms = (time.perf_counter() - inference_started) * 1000.0
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            q_target = np.asarray(q_target, dtype=np.float32).reshape(-1)
            if action.shape != (29,) or not np.isfinite(action).all():
                raise RuntimeError(f"SONIC returned invalid action shape={action.shape}")
            if q_target.shape != (29,) or not np.isfinite(q_target).all():
                raise RuntimeError(f"SONIC returned invalid q_target shape={q_target.shape}")
            if phase == "motion":
                violations = np.flatnonzero(
                    (q_target < joint_lower - args.proposal_limit_tolerance)
                    | (q_target > joint_upper + args.proposal_limit_tolerance)
                )
                if violations.size:
                    details = ",".join(
                        f"{canonical_joint_names[index]}={q_target[index]:.3f}"
                        f" not_in [{joint_lower[index]:.3f},{joint_upper[index]:.3f}]"
                        for index in violations
                    )
                    termination["reason"] = (
                        f"proposal_joint_limit:ref={reference_step}:{details}"
                    )
                    raise RuntimeError(termination["reason"])
                target_step = np.abs(q_target - last_q_target)
                max_step_index = int(np.argmax(target_step))
                if target_step[max_step_index] > args.max_proposal_step:
                    termination["reason"] = (
                        "proposal_step_limit:"
                        f"ref={reference_step}:"
                        f"joint={canonical_joint_names[max_step_index]}:"
                        f"step={target_step[max_step_index]:.3f}:"
                        f"limit={args.max_proposal_step:.3f}"
                    )
                    raise RuntimeError(termination["reason"])

            sequence += 1
            proposal_socket.send(
                SonicNominalProposal(
                    source_time_ns=time.monotonic_ns(),
                    sequence=sequence,
                    reference_step=reference_step,
                    action=action,
                    q_target=q_target,
                ).to_bytes(),
                flags=zmq.DONTWAIT,
            )
            policy.state_dict = next_state
            policy.state_dict["action"] = action
            policy.state_dict["paused"] = stabilizing
            policy.state_dict["control_mode"] = "policy"
            last_action = action.copy()
            last_q_target = q_target.copy()

            publish_age_s, sample_age_s, sample_skew_s = _ft_timing(
                ft_frame,
                control_time_ns=control_time_ns,
                control_monotonic_ns=control_monotonic_ns,
            )
            runtime_reference = policy.motion_dataset.get_slice(
                np.asarray([0], dtype=np.int64),
                np.asarray([reference_step], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
            )
            source_index = min(
                int(round(reference_step / args.reference_time_scale)),
                reference["object_contact"].shape[0] - 1,
            )
            records["time_ns"].append(control_time_ns)
            records["monotonic_ns"].append(control_monotonic_ns)
            records["phase"].append(phase)
            records["reference_step"].append(reference_step)
            records["low_state_tick"].append(tick)
            records["low_state_age_s"].append(low_state_age_s)
            records["inference_ms"].append(inference_ms)
            records["g1_input"].append(observations["g1_input"].copy())
            records["proprioception"].append(observations["proprioception"].copy())
            records["action"].append(action.copy())
            records["q_target"].append(q_target.copy())
            records["joint_pos"].append(policy.state_processor.joint_pos.copy())
            records["joint_vel"].append(policy.state_processor.joint_vel.copy())
            records["root_quat"].append(policy.state_processor.root_quat_w.copy())
            records["root_ang_vel"].append(policy.state_processor.root_ang_vel_b.copy())
            records["joint_torque"].append(state.joint_torque.copy())
            for name in REFERENCE_FIELDS[:-1]:
                records[f"target_{name}"].append(
                    np.asarray(getattr(runtime_reference, name)[0, 0]).copy()
                )
            records["target_object_contact"].append(
                reference["object_contact"][source_index].copy()
            )
            records["ft_frame_fresh"].append(ft_frame_fresh)
            records["ft_timestamp_ns"].append(ft_frame.timestamp_ns)
            records["ft_publish_monotonic_ns"].append(ft_frame.publish_monotonic_ns)
            records["ft_publish_age_s"].append(publish_age_s)
            records["ft_kinematics_monotonic_ns"].append(
                ft_frame.kinematics_monotonic_ns
            )
            records["ft_sample_received_monotonic_ns"].append(
                ft_frame.sample_received_monotonic_ns.copy()
            )
            records["ft_sample_age_s"].append(sample_age_s)
            records["ft_sample_kinematics_skew_s"].append(sample_skew_s)
            records["ft_sample_source_time_ns"].append(
                ft_frame.sample_source_time_ns.copy()
            )
            records["ft_sample_sequence"].append(ft_frame.sample_sequence.copy())
            records["ft_sample_device_id"].append(ft_frame.sample_device_id.copy())
            records["ft_sample_status"].append(ft_frame.sample_status.copy())
            records["ft_sequence"].append(ft_frame.sequence)
            records["ft_contact_count"].append(ft_frame.contact_count)
            records["ft_total_force_norm"].append(ft_frame.total_force_norm)
            records["ft_token"].append(ft_frame.token.copy())
            records["ft_wrench_base_yaw"].append(ft_frame.wrench_base_yaw.copy())
            records["ft_wrench_sensor"].append(ft_frame.wrench_sensor.copy())
            loop_ms = (time.perf_counter() - loop_started) * 1000.0
            records["loop_ms"].append(loop_ms)
            recorder.capture(records)

            next_tick += period
            sleep_s = next_tick - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                # Never issue catch-up proposal bursts after a delayed cycle.
                next_tick = time.perf_counter()

        atomic_write_json(
            args.completion_file,
            {
                "schema": "somaforce_policy_motion_complete_v1",
                "backend": "sonic",
                "steps": args.steps,
                "recorded_frames": len(records["time_ns"]),
                "stabilization_steps": stabilization_steps,
            },
        )
        completion_deadline = time.monotonic() + args.completion_timeout
        next_completion_publish = time.perf_counter()
        while not args.completion_ack_file.exists():
            if stop_request["exit_code"] is not None:
                raise SystemExit(stop_request["exit_code"])
            if time.monotonic() >= completion_deadline:
                raise RuntimeError(
                    "timed out waiting for safe-controller completion hold"
                )
            sequence += 1
            proposal_socket.send(
                SonicNominalProposal(
                    source_time_ns=time.monotonic_ns(),
                    sequence=sequence,
                    reference_step=args.steps - 1,
                    action=last_action,
                    q_target=last_q_target,
                ).to_bytes(),
                flags=zmq.DONTWAIT,
            )
            next_completion_publish += period
            time.sleep(max(0.0, next_completion_publish - time.perf_counter()))

        recorder.close(records, status="complete")
        arrays = {name: np.asarray(values) for name, values in records.items()}
        summary = {
            "schema": SCHEMA,
            "complete": True,
            "result": "pass",
            "steps": int(arrays["time_ns"].shape[0]),
            "motion_steps": args.steps,
            "stabilization_steps": stabilization_steps,
            "rate_hz": args.rate,
            "inference_ms": _percentiles(arrays["inference_ms"]),
            "loop_ms": _percentiles(arrays["loop_ms"]),
            "loop_overrun_count": int(
                np.count_nonzero(arrays["loop_ms"] > period * 1000.0)
            ),
            "ft_fresh_fraction": float(np.mean(arrays["ft_frame_fresh"])),
            "ft_both_valid_fraction": float(
                np.mean(np.all(arrays["ft_token"][:, :, 13] > 0.5, axis=1))
            ),
            "ft_force_norm_max": float(np.max(arrays["ft_total_force_norm"])),
            "target_step_rad_max": (
                float(np.max(np.abs(np.diff(arrays["q_target"], axis=0))))
                if arrays["q_target"].shape[0] > 1
                else 0.0
            ),
        }
        finalize_chunked_recording(
            args.output,
            metadata={**base_metadata, "summary": summary},
            complete=True,
        )
        atomic_write_json(args.output.with_suffix(".summary.json"), summary)
        termination["reason"] = "complete"
        atexit.unregister(finalize_partial)
        for signum, previous_handler in previous_signal_handlers.items():
            signal.signal(signum, previous_handler)
        ft_receiver.close()
        proposal_socket.close(linger=0)
        policy.robot_io.close()
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"saved SONIC hardware record: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
