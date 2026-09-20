"""Redundant rigid-body marker calibration and pose fusion."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from .torso2pelvis import validate_transform


@dataclass(frozen=True)
class MarkerSource:
    name: str
    topic: str
    marker_from_target: np.ndarray

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("marker source name must not be empty")
        if not self.topic.startswith("/"):
            raise ValueError(f"marker topic must be absolute: {self.topic!r}")
        object.__setattr__(
            self,
            "marker_from_target",
            validate_transform(
                self.marker_from_target,
                name=f"marker_sources[{self.name!r}].marker_from_target",
            ),
        )


@dataclass(frozen=True)
class FusedPose:
    world_from_target: np.ndarray
    used_sources: tuple[str, ...]
    rejected_sources: tuple[str, ...]


def load_calibration_payload(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    import json

    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError("marker calibration must be a JSON object")
    if payload.get("valid", True) is not True:
        raise ValueError("marker calibration is marked invalid")
    return payload


def load_world_from_mocap(payload: Mapping[str, Any]) -> np.ndarray | None:
    raw = payload.get("world_from_mocap")
    if raw is None:
        return None
    return validate_transform(raw, name="world_from_mocap")


def _legacy_transform(payload: Mapping[str, Any], role: str) -> np.ndarray:
    aliases = (role, "pelvis") if role == "torso" else (role,)
    marker_to_policy = payload.get("marker_to_policy", {})
    if isinstance(marker_to_policy, Mapping):
        for alias in aliases:
            if alias in marker_to_policy:
                return validate_transform(
                    marker_to_policy[alias], name=f"marker_to_policy[{alias!r}]"
                )
    for alias in aliases:
        key = f"marker_from_{alias}"
        if key in payload:
            return validate_transform(payload[key], name=key)
    if role == "torso" and "torso_from_marker" in payload:
        return np.linalg.inv(
            validate_transform(payload["torso_from_marker"], name="torso_from_marker")
        )
    return np.eye(4, dtype=np.float64)


def load_marker_sources(
    payload: Mapping[str, Any],
    *,
    role: str,
    legacy_topic: str,
    corrections: Mapping[str, Any] | None = None,
) -> tuple[MarkerSource, ...]:
    """Load v2 sources while retaining the old single-marker file format."""
    marker_sources = payload.get("marker_sources", {})
    raw_sources = (
        marker_sources.get(role) if isinstance(marker_sources, Mapping) else None
    )
    if raw_sources is None and role == "torso" and isinstance(marker_sources, Mapping):
        raw_sources = marker_sources.get("pelvis")
    if raw_sources is None:
        return (
            MarkerSource(
                name=role,
                topic=legacy_topic,
                marker_from_target=_legacy_transform(payload, role),
            ),
        )
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(f"marker_sources.{role} must be a non-empty list")

    sources: list[MarkerSource] = []
    for index, raw in enumerate(raw_sources):
        if not isinstance(raw, Mapping):
            raise ValueError(f"marker_sources.{role}[{index}] must be an object")
        name = str(raw.get("name", f"{role}_{index + 1}"))
        topic = str(raw.get("topic", ""))
        transform = raw.get("marker_from_target")
        for key in (f"marker_from_{role}", "marker_to_policy", "transform"):
            if transform is None and key in raw:
                transform = raw[key]
        if transform is None:
            raise ValueError(
                f"marker_sources.{role}[{index}] is missing marker_from_target"
            )
        correction = np.eye(4, dtype=np.float64)
        if corrections is not None:
            raw_roles = corrections.get("sources", {})
            if isinstance(raw_roles, Mapping):
                raw_role = raw_roles.get(role, {})
                if isinstance(raw_role, Mapping) and name in raw_role:
                    raw_correction = raw_role[name]
                    if isinstance(raw_correction, Mapping):
                        raw_correction = raw_correction.get(
                            "target_frame_correction", np.eye(4)
                        )
                    correction = validate_transform(
                        raw_correction,
                        name=f"corrections.sources.{role}.{name}",
                    )
        sources.append(MarkerSource(name, topic, np.asarray(transform) @ correction))

    names = [source.name for source in sources]
    topics = [source.topic for source in sources]
    if len(names) != len(set(names)):
        raise ValueError(f"marker_sources.{role} contains duplicate names")
    if len(topics) != len(set(topics)):
        raise ValueError(f"marker_sources.{role} contains duplicate topics")
    return tuple(sources)


def _rotation_distance(first: np.ndarray, second: np.ndarray) -> float:
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(math.acos(cosine))


def _mean_pose(poses: list[np.ndarray]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = np.mean([pose[:3, 3] for pose in poses], axis=0)
    result[:3, :3] = (
        Rotation.from_matrix(np.stack([pose[:3, :3] for pose in poses]))
        .mean()
        .as_matrix()
    )
    return result


def _scaled_transform(transform: np.ndarray, scale: float) -> np.ndarray:
    """Scale a local SE(3) correction toward identity."""
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = transform[:3, 3] * float(scale)
    rotvec = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
    result[:3, :3] = Rotation.from_rotvec(rotvec * float(scale)).as_matrix()
    return result


class MultiMarkerPoseFusion:
    """Convert redundant marker poses to one target pose with consensus gating."""

    def __init__(
        self,
        sources: tuple[MarkerSource, ...] | list[MarkerSource],
        *,
        stale_timeout_s: float,
        synchronization_window_s: float = 0.05,
        position_consensus_m: float = 0.08,
        orientation_consensus_deg: float = 12.0,
        source_switch_blend_s: float = 0.0,
        preferred_source_name: str | None = None,
    ) -> None:
        if (
            stale_timeout_s <= 0.0
            or synchronization_window_s <= 0.0
            or position_consensus_m <= 0.0
        ):
            raise ValueError(
                "marker timeouts and consensus thresholds must be positive"
            )
        if orientation_consensus_deg <= 0.0:
            raise ValueError("marker orientation consensus must be positive")
        if source_switch_blend_s < 0.0:
            raise ValueError("marker source-switch blend duration must be non-negative")
        self.sources = tuple(sources)
        if not self.sources:
            raise ValueError("at least one marker source is required")
        self._by_name = {source.name: source for source in self.sources}
        if len(self._by_name) != len(self.sources):
            raise ValueError("marker source names must be unique")
        if preferred_source_name is not None and preferred_source_name not in self._by_name:
            raise ValueError(
                f"preferred marker source {preferred_source_name!r} is not configured"
            )
        self.stale_timeout_s = float(stale_timeout_s)
        self.synchronization_window_s = min(
            float(synchronization_window_s), self.stale_timeout_s
        )
        self.position_consensus_m = float(position_consensus_m)
        self.orientation_consensus_rad = math.radians(orientation_consensus_deg)
        self.source_switch_blend_s = float(source_switch_blend_s)
        self.preferred_source_name = preferred_source_name
        self._observations: dict[str, tuple[float, np.ndarray]] = {}
        self._last_used_sources: tuple[str, ...] | None = None
        self._last_output_pose: np.ndarray | None = None
        self._last_resolve_succeeded = False
        self._blend_started_at: float | None = None
        self._blend_from_correction = np.eye(4, dtype=np.float64)

    def update(
        self,
        source_name: str,
        world_from_marker: Any,
        *,
        received_at: float,
    ) -> None:
        try:
            source = self._by_name[source_name]
        except KeyError as exc:
            raise ValueError(f"unknown marker source {source_name!r}") from exc
        marker_pose = validate_transform(
            world_from_marker, name=f"world_from_marker[{source_name!r}]"
        )
        self._observations[source_name] = (
            float(received_at),
            marker_pose @ source.marker_from_target,
        )

    def source_ages(self, *, now: float) -> dict[str, float]:
        return {
            source.name: (
                math.inf
                if source.name not in self._observations
                else float(now) - self._observations[source.name][0]
            )
            for source in self.sources
        }

    def resolve(self, *, now: float) -> FusedPose | None:
        fresh = [
            (source, self._observations[source.name][1])
            for source in self.sources
            if source.name in self._observations
            and float(now) - self._observations[source.name][0] <= self.stale_timeout_s
        ]
        if not fresh:
            self._last_resolve_succeeded = False
            return None
        newest_received_at = max(
            self._observations[source.name][0] for source, _ in fresh
        )
        fresh = [
            (source, pose)
            for source, pose in fresh
            if newest_received_at - self._observations[source.name][0]
            <= self.synchronization_window_s
        ]
        preferred = next(
            (
                candidate
                for candidate in fresh
                if candidate[0].name == self.preferred_source_name
            ),
            None,
        )
        if preferred is not None:
            selected = [preferred]
        elif len(fresh) == 1:
            selected = fresh
        else:

            def pair_consistent(first: int, second: int) -> bool:
                first_pose = fresh[first][1]
                second_pose = fresh[second][1]
                return bool(
                    np.linalg.norm(first_pose[:3, 3] - second_pose[:3, 3])
                    <= self.position_consensus_m
                    and _rotation_distance(first_pose, second_pose)
                    <= self.orientation_consensus_rad
                )

            consensus_groups: list[tuple[int, ...]] = []
            for size in range(len(fresh), 1, -1):
                consensus_groups = [
                    group
                    for group in combinations(range(len(fresh)), size)
                    if all(
                        pair_consistent(first, second)
                        for first, second in combinations(group, 2)
                    )
                ]
                if consensus_groups:
                    break
            if not consensus_groups:
                self._last_resolve_succeeded = False
                return None
            else:
                if len(consensus_groups) > 1:
                    self._last_resolve_succeeded = False
                    return None
                group = consensus_groups[0]
                selected = [fresh[index] for index in group]

        used_names = tuple(source.name for source, _ in selected)
        rejected_names = tuple(
            source.name for source, _ in fresh if source.name not in used_names
        )
        raw_pose = _mean_pose([candidate for _, candidate in selected])
        source_set_changed = (
            self._last_used_sources is not None
            and used_names != self._last_used_sources
        )
        recovered_after_gap = (
            self._last_output_pose is not None and not self._last_resolve_succeeded
        )
        if (
            self.source_switch_blend_s > 0.0
            and self._last_output_pose is not None
            and (source_set_changed or recovered_after_gap)
        ):
            self._blend_from_correction = (
                np.linalg.inv(raw_pose) @ self._last_output_pose
            )
            self._blend_started_at = float(now)

        pose = raw_pose
        if self._blend_started_at is not None:
            alpha = np.clip(
                (float(now) - self._blend_started_at) / self.source_switch_blend_s,
                0.0,
                1.0,
            )
            if alpha >= 1.0:
                self._blend_started_at = None
            else:
                correction = _scaled_transform(
                    self._blend_from_correction, 1.0 - float(alpha)
                )
                pose = raw_pose @ correction

        pose = validate_transform(pose, name="fused world_from_target")
        self._last_used_sources = used_names
        self._last_output_pose = pose.copy()
        self._last_resolve_succeeded = True
        return FusedPose(pose, used_names, rejected_names)
