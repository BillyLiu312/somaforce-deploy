"""Real wrist F/T samples to the Cross residual token contract.

The adapter deliberately separates raw sensor transport from robot kinematics.
This keeps all calibration, frame transforms, freshness gates, and token
semantics on the deployment side where they can be tested against training.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import socket
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from .hdmi_residual_runtime import ResidualFTFrame


RAW_SENSOR_SCHEMA = "hps6axis.wrench.v1"
CALIBRATION_SCHEMA = "somaforce_ft_calibration_v1"
WRIST_SIDES = ("left", "right")


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _proper_transform(value: Any, name: str) -> np.ndarray:
    matrix = _finite_array(value, (4, 4), name)
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation is not proper")
    return matrix


def yaw_rotation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = _finite_array(quaternion_wxyz, (4,), "base quaternion")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-9:
        raise ValueError("base quaternion has near-zero norm")
    w, x, y, z = quaternion / norm
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )


@dataclass(frozen=True)
class RawWrenchSample:
    device_id: int
    status: int
    source_time_ns: int
    sequence: int
    wrench_sensor: np.ndarray
    received_monotonic_ns: int
    clock: str = "monotonic"

    @classmethod
    def from_json_line(
        cls,
        line: str | bytes,
        *,
        received_monotonic_ns: int | None = None,
        allow_legacy: bool = False,
        legacy_sequence: int = 0,
    ) -> "RawWrenchSample":
        try:
            payload = json.loads(line)
        except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("raw F/T sample is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("raw F/T sample must be a JSON object")

        schema = payload.get("schema")
        if schema == RAW_SENSOR_SCHEMA:
            try:
                source_time_ns = int(payload["monotonic_ns"])
                sequence = int(payload["sequence"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "versioned raw F/T sample is missing timestamp/sequence"
                ) from exc
            clock = "monotonic"
        elif allow_legacy and "timestamp_ms" in payload:
            try:
                source_time_ns = int(payload["timestamp_ms"]) * 1_000_000
            except (TypeError, ValueError) as exc:
                raise ValueError("legacy raw F/T timestamp_ms is invalid") from exc
            sequence = int(legacy_sequence)
            clock = "realtime"
        else:
            raise ValueError(
                f"raw F/T schema must be {RAW_SENSOR_SCHEMA!r}; "
                "legacy messages require allow_legacy=True"
            )
        if source_time_ns < 0 or sequence < 0:
            raise ValueError("raw F/T timestamps and sequence must be non-negative")
        try:
            device_id = int(payload["device_id"])
            status = int(payload["status"])
            wrench = np.asarray(
                [payload[key] for key in ("fx", "fy", "fz", "mx", "my", "mz")],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("raw F/T sample is missing a typed wrench field") from exc
        if device_id < 0 or not 0 <= status <= 255:
            raise ValueError("raw F/T device_id/status is out of range")
        if not np.isfinite(wrench).all():
            raise ValueError("raw F/T wrench contains non-finite values")
        return cls(
            device_id=device_id,
            status=status,
            source_time_ns=source_time_ns,
            sequence=sequence,
            wrench_sensor=wrench,
            received_monotonic_ns=(
                time.monotonic_ns()
                if received_monotonic_ns is None
                else int(received_monotonic_ns)
            ),
            clock=clock,
        )


@dataclass(frozen=True)
class WristCalibration:
    device_id: int
    measurement_sign: float
    bias_sensor: np.ndarray
    wrist_from_sensor: np.ndarray
    force_limit_N: float
    moment_limit_Nm: float
    downstream_mass_kg: float = 0.0
    downstream_com_sensor_m: np.ndarray | None = None
    gravity_compensation: bool = False

    def __post_init__(self) -> None:
        if self.device_id < 0:
            raise ValueError("device_id must be non-negative")
        if self.measurement_sign not in (-1.0, 1.0):
            raise ValueError("measurement_sign must be -1 or 1")
        object.__setattr__(
            self, "bias_sensor", _finite_array(self.bias_sensor, (6,), "bias_sensor")
        )
        object.__setattr__(
            self,
            "wrist_from_sensor",
            _proper_transform(self.wrist_from_sensor, "wrist_from_sensor"),
        )
        if self.force_limit_N <= 0.0 or self.moment_limit_Nm <= 0.0:
            raise ValueError("wrench limits must be positive")
        if self.downstream_mass_kg < 0.0:
            raise ValueError("downstream_mass_kg must be non-negative")
        com = (
            np.zeros(3)
            if self.downstream_com_sensor_m is None
            else self.downstream_com_sensor_m
        )
        object.__setattr__(
            self,
            "downstream_com_sensor_m",
            _finite_array(com, (3,), "downstream_com_sensor_m"),
        )


@dataclass(frozen=True)
class FTAdapterConfig:
    wrists: tuple[WristCalibration, WristCalibration]
    force_scale_N: float = 100.0
    moment_scale_Nm: float = 10.0
    stale_timeout_s: float = 0.10
    robot_state_timeout_s: float = 0.10
    contact_on_threshold: float = 0.05
    contact_off_threshold: float = 0.025
    probability_temperature: float = 0.02
    smoothing_alpha: float = 0.5
    require_runtime_tare: bool = False
    validation_scope: str = "unvalidated"
    upstream_distal_load_compensated: bool = False

    def __post_init__(self) -> None:
        if len(self.wrists) != 2:
            raise ValueError("exactly two wrist calibrations are required")
        positive = (
            self.force_scale_N,
            self.moment_scale_Nm,
            self.stale_timeout_s,
            self.robot_state_timeout_s,
            self.probability_temperature,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("scales, timeouts, and probability temperature must be positive")
        if not 0.0 <= self.contact_off_threshold < self.contact_on_threshold:
            raise ValueError("contact thresholds must satisfy 0 <= off < on")
        if not 0.0 <= self.smoothing_alpha <= 1.0:
            raise ValueError("smoothing_alpha must be in [0, 1]")
        if self.validation_scope not in {
            "unvalidated",
            "runtime_tare_and_residual_shadow",
            "residual_authority",
        }:
            raise ValueError(f"unsupported validation_scope: {self.validation_scope!r}")

    def require_residual_authority(self) -> None:
        if self.validation_scope != "residual_authority":
            raise ValueError(
                "F/T calibration is not validated for residual authority; "
                f"validation_scope={self.validation_scope!r}"
            )
        if not self.upstream_distal_load_compensated and not all(
            wrist.gravity_compensation for wrist in self.wrists
        ):
            raise ValueError(
                "residual authority requires gravity compensation on both wrists "
                "or upstream_distal_load_compensated=true"
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FTAdapterConfig":
        if payload.get("schema") != CALIBRATION_SCHEMA:
            raise ValueError(f"calibration schema must be {CALIBRATION_SCHEMA!r}")
        if payload.get("valid") is not True:
            raise ValueError("F/T calibration is not marked valid")
        raw_sides = payload.get("sides")
        if not isinstance(raw_sides, Mapping):
            raise ValueError("calibration sides must be a mapping")
        wrists: list[WristCalibration] = []
        for side in WRIST_SIDES:
            raw = raw_sides.get(side)
            if not isinstance(raw, Mapping):
                raise ValueError(f"missing {side} wrist calibration")
            gravity = raw.get("gravity_compensation", {})
            if not isinstance(gravity, Mapping):
                raise ValueError(f"{side}.gravity_compensation must be a mapping")
            wrists.append(
                WristCalibration(
                    device_id=int(raw["device_id"]),
                    measurement_sign=float(raw["measurement_sign"]),
                    bias_sensor=np.asarray(raw["bias_sensor"], dtype=np.float64),
                    wrist_from_sensor=np.asarray(raw["wrist_from_sensor"], dtype=np.float64),
                    force_limit_N=float(raw.get("force_limit_N", 500.0)),
                    moment_limit_Nm=float(raw.get("moment_limit_Nm", 50.0)),
                    downstream_mass_kg=float(gravity.get("mass_kg", 0.0)),
                    downstream_com_sensor_m=np.asarray(
                        gravity.get("com_sensor_m", (0.0, 0.0, 0.0)),
                        dtype=np.float64,
                    ),
                    gravity_compensation=bool(gravity.get("enabled", False)),
                )
            )
        token = payload.get("token", {})
        if not isinstance(token, Mapping):
            raise ValueError("calibration token must be a mapping")
        return cls(
            wrists=(wrists[0], wrists[1]),
            force_scale_N=float(token.get("force_scale_N", 100.0)),
            moment_scale_Nm=float(token.get("moment_scale_Nm", 10.0)),
            stale_timeout_s=float(token.get("stale_timeout_s", 0.10)),
            robot_state_timeout_s=float(token.get("robot_state_timeout_s", 0.10)),
            contact_on_threshold=float(token.get("contact_on_threshold", 0.05)),
            contact_off_threshold=float(token.get("contact_off_threshold", 0.025)),
            probability_temperature=float(token.get("probability_temperature", 0.02)),
            smoothing_alpha=float(token.get("smoothing_alpha", 0.5)),
            require_runtime_tare=bool(token.get("require_runtime_tare", False)),
            validation_scope=str(payload.get("validation_scope", "unvalidated")),
            upstream_distal_load_compensated=bool(
                payload.get("upstream_distal_load_compensated", False)
            ),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "FTAdapterConfig":
        payload = yaml.safe_load(Path(path).read_text())
        if not isinstance(payload, Mapping):
            raise ValueError("F/T calibration YAML must contain a mapping")
        return cls.from_mapping(payload)


@dataclass(frozen=True)
class WristKinematics:
    base_quaternion_wxyz: np.ndarray
    world_from_wrist_rotation: np.ndarray
    twist_base_yaw: np.ndarray
    received_monotonic_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "base_quaternion_wxyz",
            _finite_array(self.base_quaternion_wxyz, (4,), "base_quaternion_wxyz"),
        )
        object.__setattr__(
            self,
            "world_from_wrist_rotation",
            _finite_array(
                self.world_from_wrist_rotation,
                (2, 3, 3),
                "world_from_wrist_rotation",
            ),
        )
        object.__setattr__(
            self,
            "twist_base_yaw",
            _finite_array(self.twist_base_yaw, (2, 6), "twist_base_yaw"),
        )


def transform_sensor_wrench(
    sample: RawWrenchSample,
    calibration: WristCalibration,
    *,
    bias_sensor: np.ndarray | None = None,
    world_from_wrist_rotation: np.ndarray,
    world_from_base_yaw_rotation: np.ndarray,
    gravity_m_s2: float = 9.80665,
) -> np.ndarray:
    """Return `[force, moment]` at the wrist origin in the base-yaw frame."""
    rotation_world_wrist = _finite_array(
        world_from_wrist_rotation, (3, 3), "world_from_wrist_rotation"
    )
    rotation_world_base_yaw = _finite_array(
        world_from_base_yaw_rotation, (3, 3), "world_from_base_yaw_rotation"
    )
    active_bias = (
        calibration.bias_sensor
        if bias_sensor is None
        else _finite_array(bias_sensor, (6,), "active bias_sensor")
    )
    corrected = calibration.measurement_sign * (sample.wrench_sensor - active_bias)
    rotation_wrist_sensor = calibration.wrist_from_sensor[:3, :3]
    if calibration.gravity_compensation and calibration.downstream_mass_kg > 0.0:
        corrected = corrected - gravity_wrench_sensor(
            calibration,
            world_from_wrist_rotation=rotation_world_wrist,
            gravity_m_s2=gravity_m_s2,
        )

    force_wrist = rotation_wrist_sensor @ corrected[:3]
    moment_wrist = (
        rotation_wrist_sensor @ corrected[3:]
        + np.cross(calibration.wrist_from_sensor[:3, 3], force_wrist)
    )
    rotation_base_yaw_wrist = rotation_world_base_yaw.T @ rotation_world_wrist
    return np.concatenate(
        (
            rotation_base_yaw_wrist @ force_wrist,
            rotation_base_yaw_wrist @ moment_wrist,
        )
    ).astype(np.float32)


def gravity_wrench_sensor(
    calibration: WristCalibration,
    *,
    world_from_wrist_rotation: np.ndarray,
    gravity_m_s2: float = 9.80665,
) -> np.ndarray:
    """Return the configured distal-load gravity wrench in the sensor frame."""
    rotation_world_wrist = _finite_array(
        world_from_wrist_rotation, (3, 3), "world_from_wrist_rotation"
    )
    if gravity_m_s2 <= 0.0:
        raise ValueError("gravity_m_s2 must be positive")
    rotation_world_sensor = (
        rotation_world_wrist @ calibration.wrist_from_sensor[:3, :3]
    )
    gravity_force_sensor = rotation_world_sensor.T @ np.asarray(
        (0.0, 0.0, -calibration.downstream_mass_kg * gravity_m_s2),
        dtype=np.float64,
    )
    gravity_moment_sensor = np.cross(
        calibration.downstream_com_sensor_m, gravity_force_sensor
    )
    return np.concatenate((gravity_force_sensor, gravity_moment_sensor))


class FTTokenAdapter:
    """Build one non-blocking two-wrist Cross token per control step."""

    def __init__(self, config: FTAdapterConfig) -> None:
        self.config = config
        self._contact_state = np.zeros(2, dtype=bool)
        self._contact_probability = np.zeros(2, dtype=np.float64)
        self._active_bias_sensor = np.stack(
            [calibration.bias_sensor for calibration in config.wrists]
        ).astype(np.float64)
        self.sequence = 0

    @property
    def active_bias_sensor(self) -> np.ndarray:
        return self._active_bias_sensor.copy()

    def apply_runtime_tare(self, bias_sensor: np.ndarray) -> None:
        if any(item.gravity_compensation for item in self.config.wrists):
            raise ValueError(
                "single-pose runtime tare cannot be combined with gravity compensation"
            )
        self._active_bias_sensor[:] = _finite_array(
            bias_sensor, (2, 6), "runtime tare bias_sensor"
        )
        self._contact_state[:] = False
        self._contact_probability[:] = 0.0

    def apply_runtime_bias(self, bias_sensor: np.ndarray) -> None:
        """Apply a bias estimate that may already exclude modeled gravity."""
        self._active_bias_sensor[:] = _finite_array(
            bias_sensor, (2, 6), "runtime bias_sensor"
        )
        self._contact_state[:] = False
        self._contact_probability[:] = 0.0

    def _sample_valid(
        self,
        sample: RawWrenchSample | None,
        calibration: WristCalibration,
        *,
        wrist_index: int,
        now_monotonic_ns: int,
    ) -> bool:
        if (
            sample is None
            or sample.device_id != calibration.device_id
            or sample.status != 0
        ):
            return False
        age_s = (now_monotonic_ns - sample.received_monotonic_ns) / 1e9
        if age_s < 0.0 or age_s > self.config.stale_timeout_s:
            return False
        force = sample.wrench_sensor[:3] - self._active_bias_sensor[wrist_index, :3]
        moment = sample.wrench_sensor[3:] - self._active_bias_sensor[wrist_index, 3:]
        return bool(
            np.linalg.norm(force) <= calibration.force_limit_N
            and np.linalg.norm(moment) <= calibration.moment_limit_Nm
        )

    def build_frame(
        self,
        samples: tuple[RawWrenchSample | None, RawWrenchSample | None],
        kinematics: WristKinematics | None,
        *,
        now_monotonic_ns: int | None = None,
        publish_time_ns: int | None = None,
    ) -> ResidualFTFrame:
        now_mono = (
            time.monotonic_ns()
            if now_monotonic_ns is None
            else int(now_monotonic_ns)
        )
        state_fresh = kinematics is not None and (
            0.0
            <= (now_mono - kinematics.received_monotonic_ns) / 1e9
            <= self.config.robot_state_timeout_s
        )
        quality = np.zeros(2, dtype=np.float64)
        wrench_base_yaw = np.zeros((2, 6), dtype=np.float32)
        twist = np.zeros((2, 6), dtype=np.float32)
        if state_fresh:
            assert kinematics is not None
            rotation_world_base_yaw = yaw_rotation(kinematics.base_quaternion_wxyz)
            twist[:] = kinematics.twist_base_yaw
            for index, (sample, calibration) in enumerate(
                zip(samples, self.config.wrists, strict=True)
            ):
                if self._sample_valid(
                    sample,
                    calibration,
                    wrist_index=index,
                    now_monotonic_ns=now_mono,
                ):
                    assert sample is not None
                    wrench_base_yaw[index] = transform_sensor_wrench(
                        sample,
                        calibration,
                        bias_sensor=self._active_bias_sensor[index],
                        world_from_wrist_rotation=(
                            kinematics.world_from_wrist_rotation[index]
                        ),
                        world_from_base_yaw_rotation=rotation_world_base_yaw,
                    )
                    quality[index] = 1.0

        normalized = wrench_base_yaw.astype(np.float64)
        normalized[:, :3] /= self.config.force_scale_N
        normalized[:, 3:] /= self.config.moment_scale_Nm
        force_ratio = np.linalg.norm(normalized[:, :3], axis=1)
        moment_ratio = np.linalg.norm(normalized[:, 3:], axis=1)
        strength = np.sqrt(0.5 * (force_ratio**2 + moment_ratio**2))
        thresholds = np.where(
            self._contact_state,
            self.config.contact_off_threshold,
            self.config.contact_on_threshold,
        )
        self._contact_state = (strength >= thresholds) & (quality > 0.0)
        raw_probability = 1.0 / (
            1.0
            + np.exp(
                -np.clip(
                    (strength - thresholds) / self.config.probability_temperature,
                    -60.0,
                    60.0,
                )
            )
        )
        raw_probability *= quality
        self._contact_probability = (
            self.config.smoothing_alpha * self._contact_probability
            + (1.0 - self.config.smoothing_alpha) * raw_probability
        )
        token = np.concatenate(
            (
                normalized.astype(np.float32),
                twist,
                self._contact_probability[:, None].astype(np.float32),
                quality[:, None].astype(np.float32),
            ),
            axis=1,
        )
        frame = ResidualFTFrame(
            timestamp_ns=(
                time.time_ns() if publish_time_ns is None else int(publish_time_ns)
            ),
            sequence=self.sequence,
            contact_count=int(np.count_nonzero(self._contact_state)),
            total_force_norm=float(
                np.linalg.norm(wrench_base_yaw[:, :3], axis=1).sum()
            ),
            token=token,
            wrench_base_yaw=wrench_base_yaw,
            publish_monotonic_ns=now_mono,
            kinematics_monotonic_ns=(
                -1 if kinematics is None else kinematics.received_monotonic_ns
            ),
            sample_received_monotonic_ns=np.asarray(
                [
                    -1 if sample is None else sample.received_monotonic_ns
                    for sample in samples
                ],
                dtype=np.int64,
            ),
            sample_source_time_ns=np.asarray(
                [-1 if sample is None else sample.source_time_ns for sample in samples],
                dtype=np.int64,
            ),
            sample_sequence=np.asarray(
                [-1 if sample is None else sample.sequence for sample in samples],
                dtype=np.int64,
            ),
            sample_device_id=np.asarray(
                [-1 if sample is None else sample.device_id for sample in samples],
                dtype=np.int64,
            ),
            sample_status=np.asarray(
                [-1 if sample is None else sample.status for sample in samples],
                dtype=np.int64,
            ),
            wrench_sensor=np.stack(
                [
                    np.zeros(6, dtype=np.float32)
                    if sample is None
                    else sample.wrench_sensor.astype(np.float32)
                    for sample in samples
                ]
            ),
        )
        self.sequence += 1
        return frame


@dataclass(frozen=True)
class TareResult:
    bias_sensor: np.ndarray
    standard_deviation: np.ndarray
    sample_count: int
    gravity_wrench_sensor: np.ndarray


def estimate_stationary_tare(
    samples: tuple[list[RawWrenchSample], list[RawWrenchSample]],
    config: FTAdapterConfig,
    *,
    minimum_samples: int,
    max_force_std_N: float,
    max_moment_std_Nm: float,
    kinematics: WristKinematics | None = None,
) -> TareResult:
    """Estimate electronic bias, excluding configured distal-load gravity."""
    if minimum_samples <= 1:
        raise ValueError("minimum_samples must be greater than one")
    if max_force_std_N <= 0.0 or max_moment_std_Nm <= 0.0:
        raise ValueError("tare standard-deviation limits must be positive")
    arrays: list[np.ndarray] = []
    for side, side_samples, calibration in zip(
        WRIST_SIDES, samples, config.wrists, strict=True
    ):
        if len(side_samples) < minimum_samples:
            raise ValueError(
                f"{side} tare has {len(side_samples)} samples; "
                f"requires {minimum_samples}"
            )
        for sample in side_samples:
            if sample.device_id != calibration.device_id:
                raise ValueError(f"{side} tare device ID does not match calibration")
            if sample.status != 0:
                raise ValueError(f"{side} tare contains nonzero sensor status")
        arrays.append(
            np.stack(
                [sample.wrench_sensor for sample in side_samples[-minimum_samples:]]
            )
        )
    values = np.stack(arrays)
    standard_deviation = np.std(values, axis=1, ddof=1)
    if np.any(standard_deviation[:, :3] > max_force_std_N):
        raise ValueError(
            "tare rejected: force standard deviation exceeds stationary limit"
        )
    if np.any(standard_deviation[:, 3:] > max_moment_std_Nm):
        raise ValueError(
            "tare rejected: moment standard deviation exceeds stationary limit"
        )
    gravity_wrenches = np.zeros((2, 6), dtype=np.float64)
    if any(item.gravity_compensation for item in config.wrists):
        if kinematics is None:
            raise ValueError(
                "gravity-aware stationary bias estimation requires wrist kinematics"
            )
        for index, calibration in enumerate(config.wrists):
            if calibration.gravity_compensation:
                gravity_wrenches[index] = gravity_wrench_sensor(
                    calibration,
                    world_from_wrist_rotation=(
                        kinematics.world_from_wrist_rotation[index]
                    ),
                )
    bias_sensor = np.mean(values, axis=1)
    for index, calibration in enumerate(config.wrists):
        bias_sensor[index] -= (
            calibration.measurement_sign * gravity_wrenches[index]
        )
    return TareResult(
        bias_sensor=bias_sensor,
        standard_deviation=standard_deviation,
        sample_count=minimum_samples,
        gravity_wrench_sensor=gravity_wrenches,
    )


class TCPWrenchClient:
    """Reconnectable latest-value client for one newline-delimited sensor stream."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        allow_legacy: bool = False,
        reconnect_delay_s: float = 0.25,
    ) -> None:
        if not 1 <= int(port) <= 65535:
            raise ValueError("sensor TCP port must be in [1, 65535]")
        self.host = str(host)
        self.port = int(port)
        self.allow_legacy = bool(allow_legacy)
        self.reconnect_delay_s = float(reconnect_delay_s)
        self._lock = threading.Lock()
        self._latest: RawWrenchSample | None = None
        self._stop = threading.Event()
        self._legacy_sequence = 0
        self.received = 0
        self.rejected = 0
        self.reconnects = 0
        self.last_error = ""
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def latest(self) -> RawWrenchSample | None:
        with self._lock:
            return self._latest

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with socket.create_connection(
                    (self.host, self.port), timeout=1.0
                ) as connection:
                    self.reconnects += 1
                    connection_previous_sequence: int | None = None
                    connection.settimeout(1.0)
                    stream = connection.makefile("rb")
                    while not self._stop.is_set():
                        try:
                            line = stream.readline()
                        except socket.timeout:
                            continue
                        if not line:
                            raise ConnectionError("sensor stream closed")
                        received_ns = time.monotonic_ns()
                        try:
                            sample = RawWrenchSample.from_json_line(
                                line,
                                received_monotonic_ns=received_ns,
                                allow_legacy=self.allow_legacy,
                                legacy_sequence=self._legacy_sequence,
                            )
                        except ValueError as exc:
                            self.rejected += 1
                            self.last_error = str(exc)
                            continue
                        self._legacy_sequence += 1
                        with self._lock:
                            if (
                                connection_previous_sequence is not None
                                and sample.sequence <= connection_previous_sequence
                            ):
                                self.rejected += 1
                                self.last_error = "non-increasing sensor sequence"
                                continue
                            self._latest = sample
                            connection_previous_sequence = sample.sequence
                            self.received += 1
            except (OSError, ConnectionError) as exc:
                self.last_error = str(exc)
                self._stop.wait(self.reconnect_delay_s)
