"""Cross-contract residual state and local MuJoCo F/T transport."""
from __future__ import annotations

from dataclasses import dataclass
import struct
import time

import numpy as np
import zmq

from .contracts import ACTION_DIM, ACTION_JOINT_NAMES, G1_JOINT_NAMES, require_array
from .residual import CrossResidual


RESIDUAL_FT_PORT = 5580
RESIDUAL_LOCKSTEP_PORT = 5581
_FT_HEADER = struct.Struct("<Qqif")
_TOKEN_FLOATS = 2 * 14
_WRENCH_FLOATS = 2 * 6


@dataclass(frozen=True)
class ResidualFTFrame:
    timestamp_ns: int
    sequence: int
    contact_count: int
    total_force_norm: float
    token: np.ndarray
    wrench_base_yaw: np.ndarray

    def to_bytes(self) -> bytes:
        token = require_array(self.token, (2, 14), "token")
        wrench = require_array(self.wrench_base_yaw, (2, 6), "wrench_base_yaw")
        return _FT_HEADER.pack(
            int(self.timestamp_ns),
            int(self.sequence),
            int(self.contact_count),
            float(self.total_force_norm),
        ) + np.concatenate((token.reshape(-1), wrench.reshape(-1))).astype("<f4").tobytes()

    @classmethod
    def from_bytes(cls, payload: bytes) -> "ResidualFTFrame":
        expected = _FT_HEADER.size + (_TOKEN_FLOATS + _WRENCH_FLOATS) * 4
        if len(payload) != expected:
            raise ValueError(f"residual F/T payload must contain {expected} bytes")
        timestamp_ns, sequence, contact_count, force_norm = _FT_HEADER.unpack_from(payload)
        values = np.frombuffer(payload, dtype="<f4", offset=_FT_HEADER.size)
        return cls(
            timestamp_ns=timestamp_ns,
            sequence=sequence,
            contact_count=contact_count,
            total_force_norm=force_norm,
            token=values[:_TOKEN_FLOATS].reshape(2, 14).copy(),
            wrench_base_yaw=values[_TOKEN_FLOATS:].reshape(2, 6).copy(),
        )


class ResidualFTPublisher:
    def __init__(self, port: int = RESIDUAL_FT_PORT) -> None:
        self.socket = zmq.Context.instance().socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{int(port)}")

    def send(self, frame: ResidualFTFrame) -> None:
        try:
            self.socket.send(frame.to_bytes(), flags=zmq.DONTWAIT)
        except zmq.Again:
            pass


class ResidualFTReceiver:
    def __init__(self, port: int = RESIDUAL_FT_PORT, host: str = "127.0.0.1") -> None:
        self.socket = zmq.Context.instance().socket(zmq.SUB)
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.CONFLATE, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{host}:{int(port)}")

    def receive_latest(
        self, *, timeout_ms: int = 1000, max_age_ms: int = 1000
    ) -> ResidualFTFrame:
        if not self.socket.poll(timeout=int(timeout_ms), flags=zmq.POLLIN):
            raise RuntimeError("timed out waiting for residual wrist F/T frame")
        payload = self.socket.recv()
        while self.socket.poll(timeout=0, flags=zmq.POLLIN):
            payload = self.socket.recv()
        frame = ResidualFTFrame.from_bytes(payload)
        if time.time_ns() - frame.timestamp_ns > int(max_age_ms) * 1_000_000:
            raise RuntimeError("residual wrist F/T frame is stale")
        return frame


class LockstepServer:
    def __init__(self, port: int = RESIDUAL_LOCKSTEP_PORT) -> None:
        self.socket = zmq.Context.instance().socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{int(port)}")

    def requested(self) -> bool:
        if not self.socket.poll(timeout=0, flags=zmq.POLLIN):
            return False
        if self.socket.recv() != b"step":
            raise RuntimeError("invalid residual lockstep request")
        return True

    def complete(self) -> None:
        self.socket.send(b"complete")


class LockstepClient:
    def __init__(self, port: int = RESIDUAL_LOCKSTEP_PORT, host: str = "127.0.0.1") -> None:
        self.socket = zmq.Context.instance().socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f"tcp://{host}:{int(port)}")

    def step(self, *, timeout_ms: int = 1000) -> None:
        self.socket.send(b"step")
        if not self.socket.poll(timeout=int(timeout_ms), flags=zmq.POLLIN):
            raise RuntimeError("timed out waiting for residual lockstep completion")
        if self.socket.recv() != b"complete":
            raise RuntimeError("invalid residual lockstep completion")


@dataclass(frozen=True)
class ResidualControlStep:
    nominal: np.ndarray
    raw_residual: np.ndarray
    delta_bounded: np.ndarray
    delta_gated: np.ndarray
    delta_safe: np.ndarray
    composed: np.ndarray
    applied: np.ndarray
    contact_target: float
    contact_gain: float


def c1_authority() -> np.ndarray:
    values = np.full(ACTION_DIM, 0.04, dtype=np.float32)
    values[[2, 5, 8]] = 0.05
    values[[11, 12, 15, 16, 19, 20, 21, 22]] = 0.10
    return values


class HDMIResidualController:
    """Mirror the deployable Cross actor history, gating, and safety order."""

    def __init__(
        self,
        residual: CrossResidual,
        *,
        mode: str,
        default_joint_pos: np.ndarray,
        action_scale: np.ndarray,
        joint_lower: np.ndarray,
        joint_upper: np.ndarray,
        velocity_limit: np.ndarray,
        control_dt: float = 0.02,
    ) -> None:
        if mode not in {"shadow", "c1"}:
            raise ValueError("residual mode must be shadow or c1")
        self.residual = residual
        self.mode = mode
        self.default_joint_pos = require_array(default_joint_pos, (ACTION_DIM,), "default_joint_pos")
        self.action_scale = require_array(action_scale, (ACTION_DIM,), "action_scale")
        self.joint_lower = require_array(joint_lower, (ACTION_DIM,), "joint_lower")
        self.joint_upper = require_array(joint_upper, (ACTION_DIM,), "joint_upper")
        self.velocity_limit = require_array(velocity_limit, (ACTION_DIM,), "velocity_limit")
        self.control_dt = float(control_dt)
        if self.control_dt <= 0.0 or np.any(self.action_scale <= 0.0):
            raise ValueError("control_dt and action_scale must be positive")
        self.authority = c1_authority()
        self.reset()

    def reset(self) -> None:
        self.wrist_history = np.zeros((1, 2, 16, 14), dtype=np.float32)
        self.nominal_history = np.zeros((1, ACTION_DIM, 3), dtype=np.float32)
        self.executed_history = np.zeros((1, ACTION_DIM, 3), dtype=np.float32)
        self.contact_gain = 0.0

    @staticmethod
    def _push_newest(history: np.ndarray, value: np.ndarray) -> None:
        history[:, :, 1:] = history[:, :, :-1].copy()
        history[:, :, 0] = value

    def step(
        self,
        *,
        nominal: np.ndarray,
        wrist_frame: np.ndarray,
        proprio: np.ndarray,
        current_joint_pos: np.ndarray,
    ) -> ResidualControlStep:
        nominal = require_array(nominal, (1, ACTION_DIM), "nominal")
        wrist_frame = require_array(wrist_frame, (2, 14), "wrist_frame")
        proprio = require_array(proprio, (1, 64), "proprio")
        current_joint_pos = require_array(current_joint_pos, (1, ACTION_DIM), "current_joint_pos")
        self.wrist_history[:, :, :-1] = self.wrist_history[:, :, 1:].copy()
        self.wrist_history[:, :, -1] = wrist_frame
        self._push_newest(self.nominal_history, nominal)
        raw = self.residual.step(
            wrist_tokens=self.wrist_history,
            proprio=proprio,
            a_nom_history=self.nominal_history,
            previous_a_total=self.executed_history[:, :, 0],
        )
        probability = wrist_frame[:, 12]
        quality = wrist_frame[:, 13]
        contact_target = float(1.0 - np.prod(1.0 - probability * quality))
        step = 0.1 if contact_target > self.contact_gain else -0.1
        self.contact_gain = float(
            np.clip(self.contact_gain + step, min(self.contact_gain, contact_target), max(self.contact_gain, contact_target))
        )
        bounded = np.tanh(raw).astype(np.float32) * self.authority[None, :]
        gated = bounded * np.float32(self.contact_gain)

        q_nom = self.default_joint_pos[None, :] + nominal * self.action_scale[None, :]
        joint_lower = np.minimum(
            (self.joint_lower[None, :] - q_nom) / self.action_scale[None, :], 0.0
        )
        joint_upper = np.maximum(
            (self.joint_upper[None, :] - q_nom) / self.action_scale[None, :], 0.0
        )
        delta_safe = np.clip(gated, joint_lower, joint_upper)
        nominal_step = q_nom - current_joint_pos
        physical_step = self.velocity_limit[None, :] * self.control_dt
        velocity_lower = np.minimum(
            (-physical_step - nominal_step) / self.action_scale[None, :], 0.0
        )
        velocity_upper = np.maximum(
            (physical_step - nominal_step) / self.action_scale[None, :], 0.0
        )
        delta_safe = np.clip(delta_safe, velocity_lower, velocity_upper).astype(np.float32)
        composed = (nominal + delta_safe).astype(np.float32)
        applied = nominal if self.mode == "shadow" else composed
        if not np.isfinite(applied).all():
            raise RuntimeError("residual composition produced a non-finite action")
        self._push_newest(self.executed_history, applied)
        return ResidualControlStep(
            nominal=nominal,
            raw_residual=raw,
            delta_bounded=bounded,
            delta_gated=gated,
            delta_safe=delta_safe,
            composed=composed,
            applied=applied.copy(),
            contact_target=contact_target,
            contact_gain=self.contact_gain,
        )


def residual_proprio(state_processor: object, *, default_joint_pos: np.ndarray) -> np.ndarray:
    default_joint_pos = require_array(default_joint_pos, (29,), "default_joint_pos")
    indices = [state_processor.joint_names.index(name) for name in G1_JOINT_NAMES]
    joint_pos = np.asarray(state_processor.joint_pos, dtype=np.float32)[indices]
    joint_vel = np.asarray(state_processor.joint_vel, dtype=np.float32)[indices]
    w, x, y, z = np.asarray(state_processor.root_quat_b, dtype=np.float64)
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    gravity = (rotation.T @ np.asarray([0.0, 0.0, -1.0])).astype(np.float32)
    return np.concatenate(
        (
            np.asarray(state_processor.root_ang_vel_b, dtype=np.float32),
            gravity,
            joint_pos - default_joint_pos,
            joint_vel,
        )
    )[None, :].astype(np.float32)
