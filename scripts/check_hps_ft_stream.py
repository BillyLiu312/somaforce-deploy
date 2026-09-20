#!/usr/bin/env python3
"""Validate and optionally record the deploy-side residual F/T stream."""
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

from somaforce_deploy.hdmi_residual_runtime import (  # noqa: E402
    RESIDUAL_FT_PORT,
    ResidualFTFrame,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=RESIDUAL_FT_PORT)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--minimum-rate", type=float, default=45.0)
    parser.add_argument("--max-age-ms", type=float, default=100.0)
    parser.add_argument("--require-both-valid", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.duration <= 0.0 or args.minimum_rate <= 0.0 or args.max_age_ms <= 0.0:
        raise ValueError("duration, minimum-rate, and max-age-ms must be positive")

    context = zmq.Context.instance()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(f"tcp://{args.host}:{args.port}")
    receive_times: list[float] = []
    receive_monotonic_times_ns: list[int] = []
    receive_wall_times_ns: list[int] = []
    frames: list[ResidualFTFrame] = []
    deadline = time.monotonic() + args.duration
    try:
        while time.monotonic() < deadline:
            if not socket.poll(timeout=20, flags=zmq.POLLIN):
                continue
            frame = ResidualFTFrame.from_bytes(socket.recv())
            receive_times.append(time.monotonic())
            receive_monotonic_times_ns.append(time.monotonic_ns())
            receive_wall_times_ns.append(time.time_ns())
            frames.append(frame)
    finally:
        socket.close(linger=0)

    if len(frames) < 2:
        raise RuntimeError(f"received only {len(frames)} residual F/T frames")
    elapsed = receive_times[-1] - receive_times[0]
    rate = (len(frames) - 1) / elapsed if elapsed > 0.0 else 0.0
    sequences = np.asarray([frame.sequence for frame in frames], dtype=np.int64)
    if np.any(np.diff(sequences) <= 0):
        raise RuntimeError("residual F/T sequence is duplicated or inverted")
    ages_ms = np.asarray(
        [
            (
                receive_mono_ns - frame.publish_monotonic_ns
                if frame.publish_monotonic_ns > 0
                else receive_wall_ns - frame.timestamp_ns
            )
            / 1e6
            for receive_mono_ns, receive_wall_ns, frame in zip(
                receive_monotonic_times_ns,
                receive_wall_times_ns,
                frames,
                strict=True,
            )
        ],
        dtype=np.float64,
    )
    tokens = np.stack([frame.token for frame in frames])
    wrenches = np.stack([frame.wrench_base_yaw for frame in frames])
    sample_received_ns = np.stack(
        [frame.sample_received_monotonic_ns for frame in frames]
    )
    sample_sequences = np.stack([frame.sample_sequence for frame in frames])
    sample_ages_ms = (
        np.asarray(receive_monotonic_times_ns, dtype=np.int64)[:, None]
        - sample_received_ns
    ) / 1e6
    sample_ages_ms[sample_received_ns < 0] = np.nan
    kinematics_ns = np.asarray(
        [frame.kinematics_monotonic_ns for frame in frames], dtype=np.int64
    )
    sample_kinematics_skew_ms = (
        sample_received_ns - kinematics_ns[:, None]
    ) / 1e6
    sample_kinematics_skew_ms[
        (sample_received_ns < 0) | (kinematics_ns[:, None] < 0)
    ] = np.nan
    if not np.isfinite(tokens).all() or not np.isfinite(wrenches).all():
        raise RuntimeError("residual F/T stream contains non-finite values")
    if rate < args.minimum_rate:
        raise RuntimeError(
            f"residual F/T rate {rate:.1f} Hz is below {args.minimum_rate:.1f} Hz"
        )
    if float(np.min(ages_ms)) < -10.0:
        raise RuntimeError(
            f"residual F/T timestamp is {abs(np.min(ages_ms)):.1f} ms in the future"
        )
    if float(np.max(ages_ms)) > args.max_age_ms:
        raise RuntimeError(
            f"residual F/T max age {np.max(ages_ms):.1f} ms exceeds "
            f"{args.max_age_ms:.1f} ms"
        )
    quality = tokens[:, :, 13]
    force_norms = np.linalg.norm(wrenches[:, :, :3], axis=2)
    moment_norms = np.linalg.norm(wrenches[:, :, 3:], axis=2)
    if args.require_both_valid and not np.all(quality == 1.0):
        valid_fraction = np.mean(quality == 1.0, axis=0)
        raise RuntimeError(
            f"both sensors were not continuously valid: {valid_fraction.tolist()}"
        )

    summary = {
        "frames": len(frames),
        "rate_hz": rate,
        "max_age_ms": float(np.max(ages_ms)),
        "sequence_gap_count": int(np.count_nonzero(np.diff(sequences) != 1)),
        "valid_fraction": np.mean(quality == 1.0, axis=0).tolist(),
        "sample_age_ms_p95": [
            (
                float(np.nanpercentile(sample_ages_ms[:, side], 95))
                if np.isfinite(sample_ages_ms[:, side]).any()
                else None
            )
            for side in range(2)
        ],
        "sample_sequence_gap_count": [
            int(
                np.count_nonzero(
                    (np.diff(sample_sequences[:, side]) != 1)
                    & (sample_sequences[1:, side] >= 0)
                    & (sample_sequences[:-1, side] >= 0)
                )
            )
            for side in range(2)
        ],
        "sample_kinematics_abs_skew_ms_p95": [
            (
                float(
                    np.nanpercentile(
                        np.abs(sample_kinematics_skew_ms[:, side]), 95
                    )
                )
                if np.isfinite(sample_kinematics_skew_ms[:, side]).any()
                else None
            )
            for side in range(2)
        ],
        "mean_wrench_base_yaw": np.mean(wrenches, axis=0).tolist(),
        "force_norm_N_p95": np.percentile(force_norms, 95, axis=0).tolist(),
        "moment_norm_Nm_p95": np.percentile(moment_norms, 95, axis=0).tolist(),
        "max_force_N": np.max(force_norms, axis=0).tolist(),
        "max_moment_Nm": np.max(moment_norms, axis=0).tolist(),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            sequence=sequences,
            timestamp_ns=np.asarray([frame.timestamp_ns for frame in frames]),
            publish_monotonic_ns=np.asarray(
                [frame.publish_monotonic_ns for frame in frames]
            ),
            kinematics_monotonic_ns=kinematics_ns,
            sample_received_monotonic_ns=sample_received_ns,
            sample_source_time_ns=np.stack(
                [frame.sample_source_time_ns for frame in frames]
            ),
            sample_sequence=sample_sequences,
            sample_device_id=np.stack(
                [frame.sample_device_id for frame in frames]
            ),
            sample_status=np.stack([frame.sample_status for frame in frames]),
            receive_time_ns=np.asarray(receive_wall_times_ns),
            receive_monotonic_ns=np.asarray(receive_monotonic_times_ns),
            token=tokens,
            wrench_base_yaw=wrenches,
            wrench_sensor=np.stack([frame.wrench_sensor for frame in frames]),
            contact_count=np.asarray([frame.contact_count for frame in frames]),
            force_norm=np.asarray([frame.total_force_norm for frame in frames]),
            metadata=np.asarray(json.dumps(summary, sort_keys=True)),
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
