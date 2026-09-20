"""Validated local transport contract for nominal joint-target proposals."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

from somaforce_deploy.contracts import ACTION_DIM, G1_JOINT_NAMES, require_array


_MAGIC = b"SNP1"
_SONIC_MAGIC = b"SNP2"
_HEADER = struct.Struct("<4sQQI")
_Q_TARGET_DIM = len(G1_JOINT_NAMES)
NOMINAL_PROPOSAL_SIZE = _HEADER.size + 4 * (ACTION_DIM + _Q_TARGET_DIM)
SONIC_NOMINAL_PROPOSAL_SIZE = _HEADER.size + 4 * (_Q_TARGET_DIM + _Q_TARGET_DIM)


@dataclass(frozen=True)
class NominalProposal:
    """One timestamped HDMI output in canonical G1 joint order."""

    source_time_ns: int
    sequence: int
    reference_step: int
    action: np.ndarray
    q_target: np.ndarray

    def __post_init__(self) -> None:
        for name in ("source_time_ns", "sequence", "reference_step"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if int(self.source_time_ns) == 0:
            raise ValueError("source_time_ns must be positive")
        object.__setattr__(
            self, "action", require_array(self.action, (ACTION_DIM,), "action").copy()
        )
        object.__setattr__(
            self,
            "q_target",
            require_array(self.q_target, (_Q_TARGET_DIM,), "q_target").copy(),
        )

    def to_bytes(self) -> bytes:
        header = _HEADER.pack(
            _MAGIC,
            int(self.source_time_ns),
            int(self.sequence),
            int(self.reference_step),
        )
        return header + self.action.tobytes() + self.q_target.tobytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> "NominalProposal":
        if len(data) != NOMINAL_PROPOSAL_SIZE:
            raise ValueError(
                "nominal proposal has invalid size: "
                f"{len(data)} != {NOMINAL_PROPOSAL_SIZE}"
            )
        magic, source_time_ns, sequence, reference_step = _HEADER.unpack_from(data)
        if magic != _MAGIC:
            raise ValueError("nominal proposal has invalid magic/version")
        values = np.frombuffer(data, dtype="<f4", offset=_HEADER.size).copy()
        return cls(
            source_time_ns=source_time_ns,
            sequence=sequence,
            reference_step=reference_step,
            action=values[:ACTION_DIM],
            q_target=values[ACTION_DIM:],
        )


@dataclass(frozen=True)
class SonicNominalProposal:
    """SONIC proposal with its native 29-D action and canonical G1 target."""

    source_time_ns: int
    sequence: int
    reference_step: int
    action: np.ndarray
    q_target: np.ndarray

    def __post_init__(self) -> None:
        for name in ("source_time_ns", "sequence", "reference_step"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if int(self.source_time_ns) == 0:
            raise ValueError("source_time_ns must be positive")
        object.__setattr__(self, "action", require_array(self.action, (_Q_TARGET_DIM,), "action").copy())
        object.__setattr__(self, "q_target", require_array(self.q_target, (_Q_TARGET_DIM,), "q_target").copy())

    def to_bytes(self) -> bytes:
        header = _HEADER.pack(
            _SONIC_MAGIC,
            int(self.source_time_ns),
            int(self.sequence),
            int(self.reference_step),
        )
        return header + self.action.tobytes() + self.q_target.tobytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> "SonicNominalProposal":
        if len(data) != SONIC_NOMINAL_PROPOSAL_SIZE:
            raise ValueError(
                f"Sonic nominal proposal has invalid size: {len(data)} != "
                f"{SONIC_NOMINAL_PROPOSAL_SIZE}"
            )
        magic, source_time_ns, sequence, reference_step = _HEADER.unpack_from(data)
        if magic != _SONIC_MAGIC:
            raise ValueError("Sonic nominal proposal has invalid magic/version")
        values = np.frombuffer(data, dtype="<f4", offset=_HEADER.size).copy()
        return cls(
            source_time_ns=source_time_ns,
            sequence=sequence,
            reference_step=reference_step,
            action=values[:_Q_TARGET_DIM],
            q_target=values[_Q_TARGET_DIM:],
        )


def proposal_from_bytes(data: bytes) -> NominalProposal | SonicNominalProposal:
    """Decode either the HDMI (SNP1) or native SONIC (SNP2) proposal."""
    if data[:4] == _MAGIC:
        return NominalProposal.from_bytes(data)
    if data[:4] == _SONIC_MAGIC:
        return SonicNominalProposal.from_bytes(data)
    raise ValueError("nominal proposal has invalid magic/version")
