"""Shared HDMI/Sonic plus Cross-residual deployment runtime."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .contracts import ACTION_DIM
from .residual import CrossResidual, compose_action
from .safety import Watchdog

DeploymentMode = Literal[
    "hdmi_student_baseline",
    "hdmi_student_residual",
    "sonic_baseline",
    "sonic_residual",
    "hdmi_student_residual_shadow",
    "sonic_residual_shadow",
]


@dataclass(frozen=True)
class StepResult:
    nominal: np.ndarray
    residual: np.ndarray
    composed: np.ndarray
    applied: np.ndarray
    shadow: bool


class ActionHistory:
    def __init__(self, length: int = 3) -> None:
        if length <= 0:
            raise ValueError("history length must be positive")
        self._values: deque[np.ndarray] = deque(maxlen=length)
        self.reset()

    def reset(self) -> None:
        self._values.clear()
        zero = np.zeros((1, ACTION_DIM), dtype=np.float32)
        self._values.extend(zero.copy() for _ in range(self._values.maxlen or 1))

    def append(self, action: np.ndarray) -> None:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (1, ACTION_DIM):
            raise ValueError(f"action history expects {(1, ACTION_DIM)}, got {value.shape}")
        self._values.append(value.copy())

    @property
    def values(self) -> np.ndarray:
        return np.stack(tuple(self._values), axis=1).astype(np.float32)

    @property
    def previous(self) -> np.ndarray:
        return self._values[-1].copy()


class DeploymentStack:
    """Compose a nominal policy and optional Cross residual at one control step."""

    def __init__(
        self,
        *,
        nominal: Any,
        residual: CrossResidual | None = None,
        mode: DeploymentMode | str = "hdmi_student_baseline",
        authority: float = 0.0,
        contact_gain: float = 0.0,
        action_limit: float = 1.0,
        watchdog: Watchdog | None = None,
    ) -> None:
        mode = str(mode)
        base_mode = mode.removesuffix("_shadow")
        if base_mode not in {"hdmi_student_baseline", "hdmi_student_residual", "sonic_baseline", "sonic_residual"}:
            raise ValueError(f"unsupported deployment mode: {mode}")
        if not 0.0 <= float(authority) <= 1.0:
            raise ValueError("authority must be in [0, 1]")
        if not 0.0 <= float(contact_gain) <= 1.0:
            raise ValueError("contact_gain must be in [0, 1]")
        if base_mode.endswith("_residual") and residual is None:
            raise ValueError(f"{mode} requires a residual policy")
        self.nominal = nominal
        self.residual = residual
        self.mode = mode
        self._base_mode = base_mode
        self.authority = float(authority)
        self.contact_gain = float(contact_gain)
        self.action_limit = float(action_limit)
        self.watchdog = watchdog or Watchdog()
        self.history = ActionHistory(length=3)

    @property
    def residual_enabled(self) -> bool:
        return self._base_mode.endswith("_residual") and self.residual is not None

    @property
    def shadow(self) -> bool:
        return self.mode.endswith("_shadow")

    def reset(self) -> None:
        self.history.reset()

    def step(
        self,
        *,
        nominal_kwargs: dict[str, Any],
        wrist_tokens: np.ndarray | None = None,
        proprio: np.ndarray | None = None,
        ft_timestamp_ns: int | None = None,
        state_timestamp_ns: int | None = None,
        now_ns: int | None = None,
    ) -> StepResult:
        if state_timestamp_ns is not None and ft_timestamp_ns is not None:
            self.watchdog.require_fresh(
                state_timestamp_ns=state_timestamp_ns,
                ft_timestamp_ns=ft_timestamp_ns,
                now_ns=now_ns,
            )
        nominal = np.asarray(self.nominal.step(**nominal_kwargs), dtype=np.float32)
        if nominal.shape != (1, ACTION_DIM):
            raise ValueError(f"nominal policy must return {(1, ACTION_DIM)}, got {nominal.shape}")

        residual = np.zeros_like(nominal)
        if self.residual_enabled:
            if wrist_tokens is None or proprio is None:
                raise ValueError("residual mode requires wrist_tokens and proprio")
            residual = self.residual.step(
                wrist_tokens=wrist_tokens,
                proprio=proprio,
                a_nom_history=self.history.values,
                previous_a_total=self.history.previous,
            )
        composed = compose_action(
            nominal,
            residual,
            authority=self.authority,
            contact_gain=self.contact_gain,
            action_limit=self.action_limit,
        )
        applied = nominal if self.shadow else composed
        self.watchdog.require_finite_action(applied)
        self.history.append(applied)
        return StepResult(
            nominal=nominal,
            residual=residual,
            composed=composed,
            applied=applied,
            shadow=self.shadow,
        )
