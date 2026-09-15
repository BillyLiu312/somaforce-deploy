#!/usr/bin/env python3
"""Relay ROS 2 PoseStamped topics to the sim2real ZMQ pose ABI.

The sim2real policy consumes seven little-endian float32 values:
``[x, y, z, qw, qx, qy, qz]``.  ROS geometry messages store the quaternion
as ``[x, y, z, w]``.  VRPN/NOKOV installations commonly report millimetres;
the default scale therefore converts millimetres to metres.

Run this file with the project virtualenv plus ROS Python paths, for example:

    source /opt/ros/humble/setup.bash
    source ~/catkin_vr/install/setup.bash
    PYTHONPATH=/opt/ros/humble/local/lib/python3.10/dist-packages:$PYTHONPATH \
      .venv/bin/python scripts/ros2_pose_to_zmq.py
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


DEFAULT_SUITCASE_TOPIC = "/suitcase/pose"
DEFAULT_PELVIS_TOPIC = "/robot_g1/pose"
DEFAULT_SUITCASE_PORT = 5561
DEFAULT_PELVIS_PORT = 5555


def _finite_vector(values: Any, size: int, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size != size:
        raise ValueError(f"{name} must contain {size} values, got {vector.size}")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} contains non-finite values: {vector.tolist()}")
    return vector


def pose_stamped_to_zmq_values(msg: Any, *, position_scale: float = 0.001) -> np.ndarray:
    """Convert a PoseStamped-like object to the repository's pose payload.

    ``position_scale`` is multiplied into ROS positions.  Use ``0.001`` for
    VRPN/NOKOV millimetres and ``1.0`` when the producer already publishes m.
    The returned quaternion is normalized and ordered ``w, x, y, z``.
    """
    if not math.isfinite(float(position_scale)) or position_scale <= 0:
        raise ValueError("position_scale must be a finite positive number")
    pose = getattr(msg, "pose", msg)
    position_msg = getattr(pose, "position")
    orientation_msg = getattr(pose, "orientation")
    position = _finite_vector(
        [position_msg.x, position_msg.y, position_msg.z], 3, "position"
    )
    ros_xyzw = _finite_vector(
        [orientation_msg.x, orientation_msg.y, orientation_msg.z, orientation_msg.w],
        4,
        "orientation",
    )
    norm = float(np.linalg.norm(ros_xyzw))
    if norm < 1e-9:
        raise ValueError("orientation quaternion has near-zero norm")
    wxyz = np.asarray(
        [ros_xyzw[3], ros_xyzw[0], ros_xyzw[1], ros_xyzw[2]], dtype=np.float32
    )
    wxyz /= np.float32(norm)
    return np.concatenate(
        [(position * float(position_scale)).astype(np.float32), wxyz]
    ).astype(np.float32, copy=False)


def _pose_matrix_from_zmq_values(values: np.ndarray) -> np.ndarray:
    """Build a homogeneous matrix from ``[x,y,z,w,x,y,z]`` values."""
    x, y, z, w, qx, qy, qz = np.asarray(values, dtype=np.float64)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * w), 2 * (qx * qz + qy * w)],
            [2 * (qx * qy + qz * w), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * w)],
            [2 * (qx * qz - qy * w), 2 * (qy * qz + qx * w), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )
    matrix[:3, 3] = [x, y, z]
    return matrix


def _wxyz_from_rotation(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        diagonal = np.diag(rotation)
        index = int(np.argmax(diagonal))
        if index == 0:
            s = math.sqrt(max(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2], 1e-12)) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / s
            x = 0.25 * s
            y = (rotation[0, 1] + rotation[1, 0]) / s
            z = (rotation[0, 2] + rotation[2, 0]) / s
        elif index == 1:
            s = math.sqrt(max(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2], 1e-12)) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / s
            x = (rotation[0, 1] + rotation[1, 0]) / s
            y = 0.25 * s
            z = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = math.sqrt(max(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1], 1e-12)) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / s
            x = (rotation[0, 2] + rotation[2, 0]) / s
            y = (rotation[1, 2] + rotation[2, 1]) / s
            z = 0.25 * s
    quaternion = np.asarray([w, x, y, z], dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def _load_calibration(path: str | None) -> tuple[np.ndarray | None, dict[str, np.ndarray]]:
    if path is None:
        return None, {}
    payload = json.loads(__import__("pathlib").Path(path).read_text())
    if payload.get("valid", True) is not True:
        raise ValueError("calibration file is marked invalid; fix residuals before use")
    matrix = None
    if "world_from_mocap" in payload:
        matrix = np.asarray(payload["world_from_mocap"], dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("world_from_mocap must be a finite 4x4 matrix")
        if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError("world_from_mocap last row must be [0, 0, 0, 1]")
        rotation = matrix[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
            raise ValueError("world_from_mocap rotation is not a proper rigid rotation")
    marker_to_policy: dict[str, np.ndarray] = {}
    for role, raw_matrix in dict(payload.get("marker_to_policy", {})).items():
        candidate = np.asarray(raw_matrix, dtype=np.float64)
        if candidate.shape != (4, 4) or not np.isfinite(candidate).all():
            raise ValueError(f"marker_to_policy[{role!r}] must be a finite 4x4 matrix")
        if not np.allclose(candidate[3], [0, 0, 0, 1], atol=1e-6):
            raise ValueError(f"marker_to_policy[{role!r}] last row must be [0, 0, 0, 1]")
        rotation = candidate[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
            raise ValueError(f"marker_to_policy[{role!r}] rotation is not a proper rigid rotation")
        marker_to_policy[str(role)] = candidate
    for role, key in (("torso", "marker_from_torso"), ("suitcase", "marker_from_suitcase")):
        if key in payload and role not in marker_to_policy:
            candidate = np.asarray(payload[key], dtype=np.float64)
            if candidate.shape != (4, 4) or not np.isfinite(candidate).all():
                raise ValueError(f"{key} must be a finite 4x4 matrix")
            if not np.allclose(candidate[3], [0, 0, 0, 1], atol=1e-6):
                raise ValueError(f"{key} last row must be [0, 0, 0, 1]")
            rotation = candidate[:3, :3]
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
                raise ValueError(f"{key} rotation is not a proper rigid rotation")
            marker_to_policy[role] = candidate
    return matrix, marker_to_policy


def pose_values_to_bytes(values: Any) -> bytes:
    """Encode converted values exactly like ``PoseMessage.to_bytes``."""
    # The deployed ABI is explicitly little-endian, independent of host ABI.
    vector = _finite_vector(values, 7, "pose payload").astype("<f4")
    return vector.tobytes()


def stale_watchdog_expired(
    ages: tuple[float, ...],
    *,
    elapsed_s: float,
    startup_timeout_s: float,
    stale_timeout_s: float,
) -> bool:
    return elapsed_s >= startup_timeout_s and max(ages) > stale_timeout_s


def _self_test() -> int:
    class Point:
        x, y, z = 1000.0, -250.0, 500.0

    class Quaternion:
        x, y, z, w = 0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)

    class Pose:
        position = Point()
        orientation = Quaternion()

    class Message:
        pose = Pose()

    values = pose_stamped_to_zmq_values(Message())
    expected = np.asarray([1.0, -0.25, 0.5, 0.70710677, 0.0, 0.0, 0.70710677], dtype=np.float32)
    if not np.allclose(values, expected, atol=1e-6):
        raise AssertionError(f"unexpected conversion: {values}")
    payload = pose_values_to_bytes(values)
    if not np.allclose(np.frombuffer(payload, dtype=np.float32), expected, atol=1e-6):
        raise AssertionError("unexpected byte payload")
    print("pose adapter self-test: PASS")
    return 0


@dataclass
class _Relay:
    role: str
    topic: str
    port: int
    position_scale: float
    publisher: Any
    expected_frame_id: str
    world_from_mocap: np.ndarray | None = None
    marker_to_policy: np.ndarray | None = None
    last_received: float = 0.0
    count: int = 0
    rejected: int = 0

    def callback(self, msg: Any) -> None:
        try:
            frame_id = str(getattr(getattr(msg, "header", None), "frame_id", ""))
            if self.expected_frame_id and frame_id != self.expected_frame_id:
                raise ValueError(
                    f"frame_id {frame_id!r} does not match expected "
                    f"{self.expected_frame_id!r}"
                )
            values = pose_stamped_to_zmq_values(
                msg, position_scale=self.position_scale
            )
            if self.world_from_mocap is not None or self.marker_to_policy is not None:
                transformed = _pose_matrix_from_zmq_values(values)
                if self.marker_to_policy is not None:
                    transformed = transformed @ self.marker_to_policy
                if self.world_from_mocap is not None:
                    transformed = self.world_from_mocap @ transformed
                values = np.concatenate(
                    [transformed[:3, 3].astype(np.float32), _wxyz_from_rotation(transformed[:3, :3]).astype(np.float32)]
                )
            # PUB sockets may be briefly back-pressured while a subscriber joins;
            # dropping that frame is preferable to blocking the ROS callback.
            import zmq

            self.publisher.send(pose_values_to_bytes(values), flags=zmq.NOBLOCK)
        except Exception as exc:
            self.rejected += 1
            print(f"rejecting {self.topic} pose: {exc}", flush=True)
            return
        self.last_received = time.monotonic()
        self.count += 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suitcase-topic", default=DEFAULT_SUITCASE_TOPIC)
    parser.add_argument("--pelvis-topic", default=DEFAULT_PELVIS_TOPIC)
    parser.add_argument(
        "--suitcase-only",
        action="store_true",
        help="publish only suitcase; use when a torso-to-pelvis solver owns port 5555",
    )
    parser.add_argument("--suitcase-port", type=int, default=DEFAULT_SUITCASE_PORT)
    parser.add_argument("--pelvis-port", type=int, default=DEFAULT_PELVIS_PORT)
    parser.add_argument(
        "--position-scale",
        type=float,
        default=0.001,
        help="multiply ROS positions by this value (default: 0.001, mm -> m)",
    )
    parser.add_argument("--bind-address", default="*")
    parser.add_argument(
        "--expected-frame-id",
        default="world",
        help="reject PoseStamped messages with another header.frame_id; empty disables",
    )
    parser.add_argument(
        "--transform-json",
        default=None,
        help="JSON from calibrate_ros2_pose.py containing world_from_mocap",
    )
    parser.add_argument("--stale-timeout", type=float, default=0.25)
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=10.0,
        help="allow initial ROS/DDS discovery before enforcing the stale watchdog",
    )
    parser.add_argument(
        "--exit-on-stale",
        action="store_true",
        help="exit with status 2 when either pose stops updating",
    )
    parser.add_argument("--stats-period", type=float, default=1.0)
    parser.add_argument("--node-name", default="sim2real_pose_adapter")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.self_test:
        return _self_test()
    if args.stale_timeout <= 0 or args.startup_timeout <= 0 or args.stats_period <= 0:
        raise ValueError("stale-timeout, startup-timeout, and stats-period must be positive")
    if args.suitcase_port == args.pelvis_port:
        raise ValueError("suitcase and pelvis ports must be different")
    world_from_mocap, marker_to_policy = _load_calibration(args.transform_json)

    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from rclpy.executors import ExternalShutdownException
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        import zmq
    except ImportError as exc:
        raise RuntimeError(
            "ROS2 and pyzmq are required. Source ROS, then run with the project "
            "venv (see the module docstring)."
        ) from exc

    context = zmq.Context()
    relays: list[_Relay] = []
    relay_specs = [("suitcase", args.suitcase_topic, args.suitcase_port)]
    if not args.suitcase_only:
        relay_specs.append(("pelvis", args.pelvis_topic, args.pelvis_port))
    for role, topic, port in relay_specs:
        publisher = context.socket(zmq.PUB)
        publisher.setsockopt(zmq.SNDHWM, 1)
        publisher.setsockopt(zmq.LINGER, 0)
        publisher.bind(f"tcp://{args.bind_address}:{port}")
        relays.append(
            _Relay(
                role=role,
                topic=topic,
                port=port,
                position_scale=args.position_scale,
                publisher=publisher,
                expected_frame_id=args.expected_frame_id,
                world_from_mocap=world_from_mocap,
                marker_to_policy=marker_to_policy.get(role),
            )
        )

    rclpy.init()
    node = rclpy.create_node(args.node_name)
    qos = QoSProfile(depth=10)
    qos.reliability = ReliabilityPolicy.BEST_EFFORT
    qos.durability = DurabilityPolicy.VOLATILE
    for relay in relays:
        node.create_subscription(PoseStamped, relay.topic, relay.callback, qos)
    route_text = ", ".join(
        f"{relay.topic}=>{args.bind_address}:{relay.port}" for relay in relays
    )
    node.get_logger().info(
        f"ROS2 -> ZMQ relay: {route_text}, position_scale={args.position_scale:g}, "
        f"frame_id={args.expected_frame_id!r}, transform={'enabled' if world_from_mocap is not None else 'none'}"
    )

    last_stats = time.monotonic()
    started_at = last_stats
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.monotonic()
            if now - last_stats >= args.stats_period:
                states = []
                stale = False
                for relay in relays:
                    age = math.inf if relay.last_received == 0 else now - relay.last_received
                    stale |= age > args.stale_timeout
                    states.append(f"{relay.topic}:count={relay.count},age={age:.3f}s,rejected={relay.rejected}")
                node.get_logger().info("; ".join(states))
                last_stats = now
                if args.exit_on_stale and stale_watchdog_expired(
                    tuple(
                        math.inf
                        if relay.last_received == 0
                        else now - relay.last_received
                        for relay in relays
                    ),
                    elapsed_s=now - started_at,
                    startup_timeout_s=args.startup_timeout,
                    stale_timeout_s=args.stale_timeout,
                ):
                    node.get_logger().error("pose watchdog expired")
                    return 2
    except (KeyboardInterrupt, ExternalShutdownException):
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        for relay in relays:
            relay.publisher.close(linger=0)
        context.term()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
