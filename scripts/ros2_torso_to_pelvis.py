#!/usr/bin/env python3
"""Publish a pelvis pose reconstructed from a torso marker and waist FK."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Allow direct execution as ``python scripts/ros2_torso_to_pelvis.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim2real.config.robots.g1 import G1_CFG
from sim2real.utils.common import LowStateMessage
from scripts.ros2_pose_to_zmq import (
    _pose_matrix_from_zmq_values,
    _wxyz_from_rotation,
    pose_stamped_to_zmq_values,
    pose_values_to_bytes,
    stale_watchdog_expired,
)
from somaforce_deploy.torso2pelvis import TorsoToPelvisFK, validate_transform


def _load_torso_from_marker(path: str | None) -> np.ndarray:
    if path is None:
        return np.eye(4, dtype=np.float64)
    payload = json.loads(Path(path).read_text())
    if "marker_from_torso" in payload:
        marker_from_torso = validate_transform(
            payload["marker_from_torso"], name="marker_from_torso"
        )
        return np.linalg.inv(marker_from_torso)
    raw = payload.get("torso_from_marker", payload.get("matrix", payload))
    return validate_transform(raw, name="torso_from_marker")


def relay_inputs_fresh(
    torso_time: float,
    state_time: float,
    *,
    now: float,
    stale_timeout: float,
) -> bool:
    return (
        now - torso_time <= stale_timeout
        and now - state_time <= stale_timeout
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torso-topic", default="/robot_torso/pose")
    parser.add_argument("--pelvis-port", type=int, default=5555)
    parser.add_argument("--low-state-host", default="127.0.0.1")
    parser.add_argument("--low-state-port", type=int, default=5590)
    parser.add_argument("--torso-from-marker-json", type=Path, default=None)
    parser.add_argument("--position-scale", type=float, default=0.001)
    parser.add_argument("--expected-frame-id", default="world")
    parser.add_argument("--stale-timeout", type=float, default=0.25)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument("--stats-period", type=float, default=1.0)
    parser.add_argument("--exit-on-stale", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if (
        args.position_scale <= 0
        or args.stale_timeout <= 0
        or args.startup_timeout <= 0
        or args.stats_period <= 0
    ):
        raise ValueError(
            "position-scale, stale-timeout, startup-timeout, and stats-period must be positive"
        )
    try:
        import rclpy
        import zmq
        from geometry_msgs.msg import PoseStamped
        from rclpy.executors import ExternalShutdownException
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    except ImportError as exc:
        raise RuntimeError("source ROS2 and use the project virtualenv before running this script") from exc

    fk = TorsoToPelvisFK(_load_torso_from_marker(str(args.torso_from_marker_json) if args.torso_from_marker_json else None))
    context = zmq.Context()
    state_socket = context.socket(zmq.SUB)
    state_socket.setsockopt(zmq.SUBSCRIBE, b"")
    state_socket.setsockopt(zmq.CONFLATE, 1)
    state_socket.setsockopt(zmq.RCVTIMEO, 0)
    state_socket.connect(f"tcp://{args.low_state_host}:{args.low_state_port}")
    pose_socket = context.socket(zmq.PUB)
    pose_socket.setsockopt(zmq.SNDHWM, 1)
    pose_socket.setsockopt(zmq.LINGER, 0)
    pose_socket.bind(f"tcp://*:{args.pelvis_port}")

    rclpy.init()
    node = rclpy.create_node("sim2real_torso_to_pelvis")
    qos = QoSProfile(depth=10)
    qos.reliability = ReliabilityPolicy.BEST_EFFORT
    qos.durability = DurabilityPolicy.VOLATILE
    latest_torso: tuple[float, np.ndarray] | None = None
    latest_state: tuple[float, LowStateMessage] | None = None
    last_pair: tuple[float, int] | None = None
    count = 0
    rejected = 0

    def torso_callback(msg: Any) -> None:
        nonlocal latest_torso, rejected
        try:
            if args.expected_frame_id and str(msg.header.frame_id) != args.expected_frame_id:
                raise ValueError(f"frame_id={msg.header.frame_id!r}, expected {args.expected_frame_id!r}")
            values = pose_stamped_to_zmq_values(msg, position_scale=args.position_scale)
            latest_torso = (time.monotonic(), _pose_matrix_from_zmq_values(values))
        except Exception as exc:
            rejected += 1
            node.get_logger().warning(f"rejecting torso pose: {exc}")

    node.create_subscription(PoseStamped, args.torso_topic, torso_callback, qos)
    node.get_logger().info(
        f"torso -> pelvis FK: {args.torso_topic} + low_state:{args.low_state_port} -> ZMQ *:{args.pelvis_port}"
    )
    last_stats = time.monotonic()
    started_at = last_stats
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            while True:
                try:
                    payload = state_socket.recv(flags=zmq.DONTWAIT)
                except zmq.Again:
                    break
                try:
                    latest_state = (time.monotonic(), LowStateMessage.from_bytes(payload))
                except Exception as exc:
                    rejected += 1
                    node.get_logger().warning(f"rejecting low_state: {exc}")
            if latest_torso is not None and latest_state is not None:
                torso_time, torso_world = latest_torso
                state_time, state = latest_state
                pair = (torso_time, state.tick)
                inputs_fresh = relay_inputs_fresh(
                    torso_time,
                    state_time,
                    now=time.monotonic(),
                    stale_timeout=args.stale_timeout,
                )
                if inputs_fresh and pair != last_pair:
                    waist_indices = [G1_CFG.joint_names.index(name) for name in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")]
                    pelvis_world = fk.pelvis_from_marker(torso_world, state.joint_positions[waist_indices])
                    values = np.concatenate(
                        [pelvis_world[:3, 3].astype(np.float32), _wxyz_from_rotation(pelvis_world[:3, :3]).astype(np.float32)]
                    )
                    try:
                        pose_socket.send(pose_values_to_bytes(values), flags=zmq.NOBLOCK)
                        count += 1
                        last_pair = pair
                    except zmq.Again:
                        pass
            now = time.monotonic()
            if now - last_stats >= args.stats_period:
                torso_age = math.inf if latest_torso is None else now - latest_torso[0]
                state_age = math.inf if latest_state is None else now - latest_state[0]
                node.get_logger().info(
                    f"torso_count={count}, torso_age={torso_age:.3f}s, low_state_age={state_age:.3f}s, rejected={rejected}"
                )
                last_stats = now
                if args.exit_on_stale and stale_watchdog_expired(
                    (torso_age, state_age),
                    elapsed_s=now - started_at,
                    startup_timeout_s=args.startup_timeout,
                    stale_timeout_s=args.stale_timeout,
                ):
                    node.get_logger().error("torso/low_state watchdog expired")
                    return 2
    except (KeyboardInterrupt, ExternalShutdownException):
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        state_socket.close(linger=0)
        pose_socket.close(linger=0)
        context.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
