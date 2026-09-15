#!/usr/bin/env python3
"""Check the three local streams consumed by the HDMI suitcase runtime."""

from __future__ import annotations

import argparse
import time

import numpy as np
import zmq

from sim2real.utils.common import LowStateMessage, PoseMessage


STREAMS = {
    "pelvis": (5555, PoseMessage.from_bytes),
    "suitcase": (5561, PoseMessage.from_bytes),
    "low_state": (5590, LowStateMessage.from_bytes),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=3.0)
    args = parser.parse_args()
    if args.duration <= 0:
        raise ValueError("duration must be positive")

    context = zmq.Context.instance()
    poller = zmq.Poller()
    sockets = {}
    for name, (port, _) in STREAMS.items():
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        socket.connect(f"tcp://127.0.0.1:{port}")
        sockets[socket] = name
        poller.register(socket, zmq.POLLIN)

    received = {name: [] for name in STREAMS}
    latest = {}
    deadline = time.monotonic() + args.duration
    while time.monotonic() < deadline:
        for socket, _ in poller.poll(timeout=20):
            name = sockets[socket]
            latest[name] = STREAMS[name][1](socket.recv())
            received[name].append(time.monotonic())

    failures = []
    for name, times in received.items():
        rate = 0.0 if len(times) < 2 else (len(times) - 1) / (times[-1] - times[0])
        print(f"{name}: frames={len(times)} rate={rate:.1f}Hz")
        if not times:
            failures.append(f"missing {name}")
    for name in ("pelvis", "suitcase"):
        if name not in latest:
            continue
        message = latest[name]
        values = np.concatenate((message.position, message.quaternion))
        if not np.isfinite(values).all():
            failures.append(f"non-finite {name}")
        print(
            f"{name}: position_m={message.position.tolist()} "
            f"quat_norm={np.linalg.norm(message.quaternion):.6f}"
        )
    if "low_state" in latest:
        message = latest["low_state"]
        arrays = (
            message.quaternion,
            message.gyroscope,
            message.joint_positions,
            message.joint_velocities,
            message.joint_torques,
        )
        if message.joint_positions.size != 29:
            failures.append(f"low_state joint count {message.joint_positions.size} != 29")
        if not all(value is not None and np.isfinite(value).all() for value in arrays):
            failures.append("non-finite low_state")
        print(
            f"low_state: joints={message.joint_positions.size} "
            f"quat_norm={np.linalg.norm(message.quaternion):.6f} tick={message.tick}"
        )
    for socket in sockets:
        socket.close(linger=0)
    if failures:
        raise RuntimeError("; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
