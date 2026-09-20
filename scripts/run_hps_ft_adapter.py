#!/usr/bin/env python3
"""Publish two HPS wrist sensors as SomaForce residual F/T frames."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import zmq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sim2real.config.robots.g1 import G1_CFG
from sim2real.utils.common import LowStateMessage, PoseMessage
from somaforce_deploy.ft_adapter import (
    FTAdapterConfig,
    FTTokenAdapter,
    TCPWrenchClient,
    estimate_stationary_tare,
)
from somaforce_deploy.g1_ft_kinematics import G1WristKinematics
from somaforce_deploy.hdmi_residual_runtime import RESIDUAL_FT_PORT, ResidualFTPublisher


DEFAULT_MJCF = (
    REPO_ROOT.parent
    / "sim2real-hdmi-upstream/data/robots/g1/g1_29dof_rubberhand-suitcase.xml"
)


def _subscriber(context: zmq.Context, port: int) -> zmq.Socket:
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(f"tcp://127.0.0.1:{port}")
    return socket


def _drain(socket: zmq.Socket) -> bytes | None:
    latest = None
    while True:
        try:
            latest = socket.recv(flags=zmq.DONTWAIT)
        except zmq.Again:
            return latest


def _snapshot_wrench_samples(clients):
    """Snapshot both latest samples before taking their comparison timestamp."""
    samples = (clients[0].latest(), clients[1].latest())
    return samples, time.monotonic_ns()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--mjcf", type=Path, default=DEFAULT_MJCF)
    parser.add_argument("--left-host", default="127.0.0.1")
    parser.add_argument("--left-port", type=int, default=9000)
    parser.add_argument("--right-host", default="127.0.0.1")
    parser.add_argument("--right-port", type=int, default=9001)
    parser.add_argument("--low-state-port", type=int, default=5590)
    parser.add_argument("--pelvis-port", type=int, default=5555)
    parser.add_argument("--no-pelvis", action="store_true")
    parser.add_argument("--output-port", type=int, default=RESIDUAL_FT_PORT)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--stats-period", type=float, default=1.0)
    parser.add_argument("--tare-on-start", action="store_true")
    parser.add_argument("--tare-samples", type=int, default=200)
    parser.add_argument("--tare-timeout", type=float, default=15.0)
    parser.add_argument("--tare-max-force-std", type=float, default=1.0)
    parser.add_argument("--tare-max-moment-std", type=float, default=0.10)
    parser.add_argument("--tare-trigger-file", type=Path)
    parser.add_argument("--tare-output", type=Path)
    parser.add_argument(
        "--allow-legacy-sensor-protocol",
        action="store_true",
        help="accept the old timestamp_ms JSON for bench testing only",
    )
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--require-residual-authority", action="store_true")
    return parser


def _collect_tare(args, clients, config, state_socket, kinematics):
    latest_state: LowStateMessage | None = None

    def update_state() -> None:
        nonlocal latest_state
        payload = _drain(state_socket)
        if payload is not None:
            latest_state = LowStateMessage.from_bytes(payload)

    if args.tare_trigger_file is None:
        input(
            "Remove external wrist loads, keep the configured hands installed and "
            "stationary at the calibration pose, "
            "then press Enter to collect the runtime bias calibration..."
        )
    else:
        print(f"waiting for tare trigger: {args.tare_trigger_file}", flush=True)
        deadline = time.monotonic() + args.tare_timeout
        while not args.tare_trigger_file.exists():
            update_state()
            if time.monotonic() >= deadline:
                raise RuntimeError("timed out waiting for tare trigger file")
            time.sleep(0.02)

    samples = ([], [])
    last_keys: list[tuple[int, int] | None] = [None, None]
    deadline = time.monotonic() + args.tare_timeout
    while min(len(side) for side in samples) < args.tare_samples:
        update_state()
        if time.monotonic() >= deadline:
            counts = [len(side) for side in samples]
            raise RuntimeError(
                f"timed out collecting stationary F/T tare: counts={counts}"
            )
        for index, client in enumerate(clients):
            sample = client.latest()
            if sample is None:
                continue
            key = (sample.source_time_ns, sample.sequence)
            if key == last_keys[index]:
                continue
            last_keys[index] = key
            samples[index].append(sample)
        time.sleep(0.001)
    update_state()
    tare_kinematics = None
    if any(item.gravity_compensation for item in config.wrists):
        if latest_state is None:
            raise RuntimeError(
                "gravity-aware bias calibration has no G1 low-state sample"
            )
        tare_kinematics = kinematics.update(
            latest_state,
            received_monotonic_ns=time.monotonic_ns(),
            reset_derivative=True,
        )
    result = estimate_stationary_tare(
        samples,
        config,
        minimum_samples=args.tare_samples,
        max_force_std_N=args.tare_max_force_std,
        max_moment_std_Nm=args.tare_max_moment_std,
        kinematics=tare_kinematics,
    )
    gravity_enabled = [item.gravity_compensation for item in config.wrists]
    signs = np.asarray([item.measurement_sign for item in config.wrists])[:, None]
    record = {
        "schema": "somaforce_ft_runtime_bias_v2",
        "created_time_ns": time.time_ns(),
        "device_ids": [item.device_id for item in config.wrists],
        "sample_count": result.sample_count,
        "bias_sensor": result.bias_sensor.tolist(),
        "mean_wrench_sensor": (
            result.bias_sensor + signs * result.gravity_wrench_sensor
        ).tolist(),
        "standard_deviation": result.standard_deviation.tolist(),
        "calibration_pose_required": True,
        "method": (
            "gravity_aware_stationary_bias"
            if any(gravity_enabled)
            else "stationary_tare"
        ),
        "gravity_compensation": gravity_enabled,
        "gravity_wrench_sensor": result.gravity_wrench_sensor.tolist(),
        "downstream_mass_kg": [item.downstream_mass_kg for item in config.wrists],
        "downstream_com_sensor_m": [
            item.downstream_com_sensor_m.tolist() for item in config.wrists
        ],
        "low_state_tick": (
            None if latest_state is None else int(latest_state.tick)
        ),
        "base_quaternion_wxyz": (
            None if tare_kinematics is None
            else tare_kinematics.base_quaternion_wxyz.tolist()
        ),
        "joint_positions": (
            None if latest_state is None else latest_state.joint_positions.tolist()
        ),
        "joint_names": list(G1_CFG.joint_names),
        "world_from_wrist_rotation": (
            None if tare_kinematics is None
            else tare_kinematics.world_from_wrist_rotation.tolist()
        ),
        "kinematics_monotonic_ns": (
            None if tare_kinematics is None
            else tare_kinematics.received_monotonic_ns
        ),
    }
    print(json.dumps(record, indent=2), flush=True)
    if args.tare_output is not None:
        args.tare_output.parent.mkdir(parents=True, exist_ok=True)
        args.tare_output.write_text(json.dumps(record, indent=2) + "\n")
    return result


def main() -> int:
    args = _build_parser().parse_args()
    if (
        args.rate <= 0.0
        or args.stats_period <= 0.0
        or args.tare_samples <= 1
        or args.tare_timeout <= 0.0
        or args.tare_max_force_std <= 0.0
        or args.tare_max_moment_std <= 0.0
    ):
        raise ValueError("rates, tare sample count, timeouts, and limits must be positive")
    if args.tare_trigger_file is not None and not args.tare_on_start:
        raise ValueError("--tare-trigger-file requires --tare-on-start")
    if not args.mjcf.is_file():
        raise FileNotFoundError(f"G1 kinematics MJCF does not exist: {args.mjcf}")
    config = FTAdapterConfig.from_yaml(args.calibration)
    if args.require_residual_authority:
        config.require_residual_authority()
    kinematics = G1WristKinematics(args.mjcf)
    if args.validate_only:
        print(
            "HPS F/T adapter contract: "
            f"device_ids={[item.device_id for item in config.wrists]} "
            f"rate={args.rate:g}Hz token=(2,14) output_port={args.output_port} "
            f"require_runtime_tare={config.require_runtime_tare}"
            f" validation_scope={config.validation_scope}"
            f" gravity_compensation={[item.gravity_compensation for item in config.wrists]}"
            f" downstream_mass_kg={[item.downstream_mass_kg for item in config.wrists]}"
            f" downstream_com_sensor_m={[item.downstream_com_sensor_m.tolist() for item in config.wrists]}"
        )
        return 0
    if config.require_runtime_tare and not args.tare_on_start:
        raise ValueError(
            "calibration requires --tare-on-start before publishing valid F/T data"
        )

    context = zmq.Context.instance()
    state_socket = _subscriber(context, args.low_state_port)
    pelvis_socket = None if args.no_pelvis else _subscriber(context, args.pelvis_port)

    clients = (
        TCPWrenchClient(
            args.left_host,
            args.left_port,
            allow_legacy=args.allow_legacy_sensor_protocol,
        ),
        TCPWrenchClient(
            args.right_host,
            args.right_port,
            allow_legacy=args.allow_legacy_sensor_protocol,
        ),
    )
    for client in clients:
        client.start()
    try:
        if args.tare_on_start:
            tare = _collect_tare(
                args,
                clients,
                config,
                state_socket,
                kinematics,
            )
        else:
            tare = None
    except Exception:
        for client in clients:
            client.close()
        state_socket.close(linger=0)
        if pelvis_socket is not None:
            pelvis_socket.close(linger=0)
        raise
    publisher = ResidualFTPublisher(args.output_port)
    adapter = FTTokenAdapter(config)
    if tare is not None:
        adapter.apply_runtime_bias(tare.bias_sensor)
    latest_state: LowStateMessage | None = None
    latest_kinematics = None
    latest_pelvis: np.ndarray | None = None
    latest_pelvis_ns = 0
    pelvis_was_fresh = False
    last_tick: int | None = None
    last_stats = time.monotonic()
    next_step = time.perf_counter()
    period = 1.0 / args.rate
    try:
        while True:
            loop_monotonic_ns = time.monotonic_ns()
            payload = _drain(state_socket)
            state_updated = False
            if payload is not None:
                state = LowStateMessage.from_bytes(payload)
                if last_tick is None or int(state.tick) != last_tick:
                    latest_state = state
                    last_tick = int(state.tick)
                    state_updated = True
            if pelvis_socket is not None:
                pose_payload = _drain(pelvis_socket)
                if pose_payload is not None:
                    pose = PoseMessage.from_bytes(pose_payload)
                    latest_pelvis = pose.position.astype(np.float64)
                    latest_pelvis_ns = loop_monotonic_ns
            pelvis_fresh = bool(
                latest_pelvis is not None
                and (loop_monotonic_ns - latest_pelvis_ns) / 1e9
                <= config.robot_state_timeout_s
            )
            if state_updated and latest_state is not None:
                latest_kinematics = kinematics.update(
                    latest_state,
                    received_monotonic_ns=loop_monotonic_ns,
                    root_position_world=(latest_pelvis if pelvis_fresh else None),
                    reset_derivative=pelvis_fresh and not pelvis_was_fresh,
                )
            pelvis_was_fresh = pelvis_fresh
            samples, frame_monotonic_ns = _snapshot_wrench_samples(clients)
            frame = adapter.build_frame(
                samples,
                latest_kinematics,
                now_monotonic_ns=frame_monotonic_ns,
            )
            publisher.send(frame)

            now = time.monotonic()
            if now - last_stats >= args.stats_period:
                quality = frame.token[:, 13].astype(int).tolist()
                probability = frame.token[:, 12].tolist()
                sample_age_ms = [
                    (
                        None
                        if sample is None
                        else round(
                            (frame_monotonic_ns - sample.received_monotonic_ns)
                            / 1e6,
                            3,
                        )
                    )
                    for sample in samples
                ]
                sample_status = [
                    None if sample is None else sample.status for sample in samples
                ]
                print(
                    "F/T adapter: "
                    f"sequence={frame.sequence} quality={quality} "
                    f"contact_probability={np.round(probability, 3).tolist()} "
                    f"force_norm={frame.total_force_norm:.2f}N "
                    f"raw_received={[client.received for client in clients]} "
                    f"raw_rejected={[client.rejected for client in clients]} "
                    f"raw_age_ms={sample_age_ms} raw_status={sample_status} "
                    f"pelvis_fresh={pelvis_fresh}",
                    flush=True,
                )
                last_stats = now
            next_step += period
            sleep_s = next_step - time.perf_counter()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_step = time.perf_counter()
    except KeyboardInterrupt:
        return 0
    finally:
        for client in clients:
            client.close()
        state_socket.close(linger=0)
        if pelvis_socket is not None:
            pelvis_socket.close(linger=0)
        publisher.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
