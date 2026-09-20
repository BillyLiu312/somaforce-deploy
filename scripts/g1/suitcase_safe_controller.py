#!/usr/bin/env python3
"""Own G1 low commands and guard zero/hold/init/nominal-pilot modes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import queue
import threading
import time
from pathlib import Path

import numpy as np
import yaml
import zmq

from sim2real.config.robots.g1 import G1_CFG
from sim2real.rl_policy.robot_io.zmq import ZMQRobotIO
from sim2real.utils.strings import resolve_matching_names_values
from somaforce_deploy.nominal_proposal import NominalProposal


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_CONFIG = (
    REPO_ROOT / "artifacts/hdmi_move_suitcase/hdmi_tag/policy.yaml"
)
DEFAULT_MOTION = REPO_ROOT / "assets/mujoco/reference/hdmi_suitcase/motion.npz"
DEFAULT_MOTION_META = REPO_ROOT / "assets/mujoco/reference/hdmi_suitcase/meta.json"


def _resolve_joint_values(
    config: object,
    *,
    label: str,
    require_all: bool = True,
) -> np.ndarray:
    indices, names, values = resolve_matching_names_values(
        config,
        G1_CFG.joint_names,
        preserve_order=True,
        strict=False,
    )
    if require_all and len(indices) != len(G1_CFG.joint_names):
        missing = [name for name in G1_CFG.joint_names if name not in names]
        raise ValueError(f"{label} does not cover all G1 joints: missing={missing}")
    result = np.zeros(len(G1_CFG.joint_names), dtype=np.float32)
    result[indices] = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{label} contains non-finite values")
    return result


def _load_motion_init_pose(motion_path: Path, metadata_path: Path) -> np.ndarray:
    metadata = json.loads(metadata_path.read_text())
    joint_names = [str(name) for name in metadata.get("joint_names", [])]
    if len(joint_names) != len(set(joint_names)):
        raise ValueError("motion metadata contains duplicate joint names")
    missing = [name for name in G1_CFG.joint_names if name not in joint_names]
    extra = [name for name in joint_names if name not in G1_CFG.joint_names]
    if missing or extra:
        raise ValueError(f"motion joint names do not match G1: missing={missing} extra={extra}")
    with np.load(motion_path) as motion:
        if "joint_pos" not in motion or motion["joint_pos"].ndim != 2:
            raise ValueError("motion must contain a two-dimensional joint_pos array")
        if motion["joint_pos"].shape[0] == 0:
            raise ValueError("motion joint_pos has no frames")
        frame_zero = np.asarray(motion["joint_pos"][0], dtype=np.float32)
    if frame_zero.shape != (len(joint_names),) or not np.isfinite(frame_zero).all():
        raise ValueError("motion frame 0 has an invalid joint vector")
    source_index = {name: index for index, name in enumerate(joint_names)}
    return np.asarray(
        [frame_zero[source_index[name]] for name in G1_CFG.joint_names],
        dtype=np.float32,
    )


def _upright_tilt_rad(quaternion_wxyz: np.ndarray) -> float:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        return math.inf
    norm = float(np.linalg.norm(quaternion))
    if norm < 1e-6:
        return math.inf
    _, x, y, _ = quaternion / norm
    body_z_world_z = 1.0 - 2.0 * (x * x + y * y)
    return float(math.acos(np.clip(body_z_world_z, -1.0, 1.0)))


class GuardedNominalPilot:
    """Bound nominal proposals before they can become robot commands."""

    def __init__(
        self,
        init_target: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        *,
        authority: float,
        joint_limit_margin: float,
        max_target_step: float,
        proposal_timeout_s: float,
        proposal_start_timeout_s: float,
        max_tilt_rad: float,
        max_joint_speed: float,
        direct_policy_targets: bool = False,
    ) -> None:
        if not 0.0 < authority <= 1.0:
            raise ValueError("pilot authority must be in (0, 1]")
        if any(
            value <= 0
            for value in (
                joint_limit_margin,
                max_target_step,
                proposal_timeout_s,
                proposal_start_timeout_s,
                max_tilt_rad,
                max_joint_speed,
            )
        ):
            raise ValueError("pilot safety limits must be positive")
        self.init_target = np.asarray(init_target, dtype=np.float32)
        self.lower = np.asarray(lower, dtype=np.float32) + float(joint_limit_margin)
        self.upper = np.asarray(upper, dtype=np.float32) - float(joint_limit_margin)
        if np.any(self.lower >= self.upper):
            raise ValueError("joint limit margin leaves an empty range")
        self.authority = float(authority)
        self.max_target_step = float(max_target_step)
        self.proposal_timeout_s = float(proposal_timeout_s)
        self.proposal_start_timeout_s = float(proposal_start_timeout_s)
        self.max_tilt_rad = float(max_tilt_rad)
        self.max_joint_speed = float(max_joint_speed)
        self.direct_policy_targets = bool(direct_policy_targets)
        self.started_at: float | None = None
        self.last_target: np.ndarray | None = None
        self.last_sequence: int | None = None
        self.last_raw_target: np.ndarray | None = None
        self.last_clipped_target: np.ndarray | None = None

    def start(self, current: np.ndarray, *, now: float) -> None:
        self.started_at = float(now)
        self.last_target = np.asarray(current, dtype=np.float32).copy()
        self.last_sequence = None
        self.last_raw_target = None
        self.last_clipped_target = None

    def target(
        self,
        current: np.ndarray,
        proposal: NominalProposal | None,
        *,
        now: float,
        monotonic_ns: int,
        quaternion_wxyz: np.ndarray,
        joint_velocity: np.ndarray,
    ) -> tuple[np.ndarray, str | None]:
        if self.started_at is None or self.last_target is None:
            raise RuntimeError("nominal pilot was not started")
        tilt = _upright_tilt_rad(quaternion_wxyz)
        if tilt > self.max_tilt_rad:
            return np.asarray(current, dtype=np.float32).copy(), "tilt"
        velocity = np.asarray(joint_velocity, dtype=np.float32)
        if velocity.shape != self.init_target.shape or not np.isfinite(velocity).all():
            return np.asarray(current, dtype=np.float32).copy(), "joint_velocity_invalid"
        if float(np.max(np.abs(velocity))) > self.max_joint_speed:
            return np.asarray(current, dtype=np.float32).copy(), "joint_velocity"
        if proposal is None:
            if now - self.started_at <= self.proposal_start_timeout_s:
                return self.last_target.copy(), None
            return np.asarray(current, dtype=np.float32).copy(), "proposal_start_timeout"
        age_s = (int(monotonic_ns) - int(proposal.source_time_ns)) / 1e9
        if age_s < -0.01 or age_s > self.proposal_timeout_s:
            return np.asarray(current, dtype=np.float32).copy(), "proposal_stale"
        if self.last_sequence is not None and proposal.sequence < self.last_sequence:
            return np.asarray(current, dtype=np.float32).copy(), "proposal_sequence"

        raw = np.asarray(proposal.q_target, dtype=np.float32)
        if self.direct_policy_targets:
            self.last_sequence = int(proposal.sequence)
            self.last_raw_target = raw.copy()
            self.last_clipped_target = raw.copy()
            self.last_target = raw.copy()
            return raw.copy(), None
        blended = self.init_target + self.authority * (raw - self.init_target)
        clipped = np.clip(blended, self.lower, self.upper)
        delta = np.clip(
            clipped - self.last_target,
            -self.max_target_step,
            self.max_target_step,
        )
        guarded = (self.last_target + delta).astype(np.float32)
        self.last_sequence = int(proposal.sequence)
        self.last_raw_target = raw.copy()
        self.last_clipped_target = clipped.astype(np.float32)
        self.last_target = guarded.copy()
        return guarded, None


class SafePoseStateMachine:
    """Generate targets and enter a latched fail-safe hold after an abort."""

    MODES = frozenset({"zero", "hold", "init", "pilot"})

    def __init__(
        self,
        init_joint_pos: np.ndarray,
        *,
        init_duration_s: float,
        pilot: GuardedNominalPilot | None = None,
    ) -> None:
        if init_duration_s <= 0:
            raise ValueError("init_duration_s must be positive")
        self.init_joint_pos = np.asarray(init_joint_pos, dtype=np.float32)
        self.init_duration_s = float(init_duration_s)
        self.pilot = pilot
        self.mode = "hold"
        self.hold_target: np.ndarray | None = None
        self.init_start: np.ndarray | None = None
        self.init_started_at: float | None = None
        self.init_completed = False
        self.last_abort_reason: str | None = None

    def initialize(self, joint_pos: np.ndarray) -> None:
        self.hold_target = np.asarray(joint_pos, dtype=np.float32).copy()

    def set_mode(self, mode: str, joint_pos: np.ndarray, *, now: float) -> None:
        if mode not in self.MODES:
            raise ValueError(f"unsupported safe control mode: {mode}")
        if mode == "pilot" and self.pilot is None:
            raise ValueError("nominal pilot is not configured")
        current = np.asarray(joint_pos, dtype=np.float32)
        self.mode = mode
        self.last_abort_reason = None
        if mode == "hold":
            self.hold_target = current.copy()
        elif mode == "init":
            self.init_completed = False
            self.init_start = current.copy()
            self.init_started_at = float(now)
        elif mode == "pilot":
            assert self.pilot is not None
            self.pilot.start(current, now=now)

    def target(
        self,
        joint_pos: np.ndarray,
        *,
        now: float,
        monotonic_ns: int | None = None,
        proposal: NominalProposal | None = None,
        quaternion_wxyz: np.ndarray | None = None,
        joint_velocity: np.ndarray | None = None,
    ) -> np.ndarray:
        current = np.asarray(joint_pos, dtype=np.float32)
        if self.hold_target is None:
            self.initialize(current)
        if self.mode == "zero":
            # Match sim2real's zero-policy mode: no policy offset, target follows
            # measured joint position while configured PD damping stays active.
            return current.copy()
        if self.mode == "hold":
            return self.hold_target.copy()
        if self.mode == "pilot":
            assert self.pilot is not None
            guarded, abort_reason = self.pilot.target(
                current,
                proposal,
                now=now,
                monotonic_ns=time.monotonic_ns() if monotonic_ns is None else monotonic_ns,
                quaternion_wxyz=(
                    np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                    if quaternion_wxyz is None
                    else quaternion_wxyz
                ),
                joint_velocity=(
                    np.zeros_like(current) if joint_velocity is None else joint_velocity
                ),
            )
            if abort_reason is not None:
                self.mode = "hold"
                self.hold_target = current.copy()
                self.last_abort_reason = abort_reason
                return self.hold_target.copy()
            return guarded
        assert self.init_start is not None and self.init_started_at is not None
        progress = np.clip(
            (float(now) - self.init_started_at) / self.init_duration_s,
            0.0,
            1.0,
        )
        smooth = progress * progress * (3.0 - 2.0 * progress)
        target = self.init_start + (self.init_joint_pos - self.init_start) * smooth
        if progress >= 1.0:
            self.mode = "hold"
            self.hold_target = self.init_joint_pos.copy()
            self.init_completed = True
        return target.astype(np.float32)


def _command_reader(path: Path, commands: queue.SimpleQueue[str]) -> None:
    with path.open() as stream:
        for line in stream:
            commands.put(line.strip().lower())


def _write_status(path: Path | None, status: str) -> None:
    print(f"safe controller mode: {status}", flush=True)
    if path is not None:
        path.write_text(status + "\n")


def _proposal_socket(port: int) -> zmq.Socket:
    socket = zmq.Context.instance().socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.connect(f"tcp://127.0.0.1:{port}")
    return socket


def _drain_proposal(socket: zmq.Socket | None) -> NominalProposal | None:
    if socket is None:
        return None
    latest = None
    while True:
        try:
            payload = socket.recv(flags=zmq.DONTWAIT)
        except zmq.Again:
            break
        latest = NominalProposal.from_bytes(payload)
    return latest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY_CONFIG)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--motion-meta", type=Path, default=DEFAULT_MOTION_META)
    parser.add_argument("--command-fifo", type=Path)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--status-file", type=Path)
    parser.add_argument("--pilot-log", type=Path)
    parser.add_argument("--proposal-port", type=int)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--init-duration", type=float, default=10.0)
    parser.add_argument("--initial-state-timeout", type=float, default=10.0)
    parser.add_argument("--state-timeout", type=float, default=0.25)
    parser.add_argument("--activation-grace", type=float, default=5.0)
    parser.add_argument("--pilot-authority", type=float, default=1.0)
    parser.add_argument(
        "--direct-policy-targets",
        action="store_true",
        help="apply raw nominal q_target without authority, position, or slew filtering",
    )
    parser.add_argument("--joint-limit-margin", type=float, default=0.05)
    parser.add_argument("--max-target-step", type=float, default=0.08)
    parser.add_argument("--proposal-timeout", type=float, default=0.10)
    parser.add_argument("--proposal-start-timeout", type=float, default=2.0)
    parser.add_argument("--max-tilt-deg", type=float, default=180.0)
    parser.add_argument("--max-joint-speed", type=float, default=12.0)
    parser.add_argument("--pilot-init-tolerance", type=float, default=0.50)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if (
        args.rate <= 0
        or args.initial_state_timeout <= 0
        or args.state_timeout <= 0
        or args.activation_grace < 0
        or args.pilot_init_tolerance <= 0
    ):
        raise ValueError("rate and timeouts must be positive")
    config_bytes = args.policy_config.read_bytes()
    config = yaml.safe_load(config_bytes)
    init_target = _load_motion_init_pose(args.motion, args.motion_meta)
    kp = _resolve_joint_values(config["joint_kp"], label="joint_kp")
    kd = _resolve_joint_values(config["joint_kd"], label="joint_kd")
    lower = _resolve_joint_values(G1_CFG.joint_pos_lower_limit, label="joint lower limits")
    upper = _resolve_joint_values(G1_CFG.joint_pos_upper_limit, label="joint upper limits")
    if np.any(init_target < lower) or np.any(init_target > upper):
        raise ValueError("suitcase motion frame 0 exceeds G1 joint limits")
    pilot = None
    if args.proposal_port is not None:
        if not 1 <= args.proposal_port <= 65535:
            raise ValueError("proposal port must be in [1, 65535]")
        pilot = GuardedNominalPilot(
            init_target,
            lower,
            upper,
            authority=args.pilot_authority,
            joint_limit_margin=args.joint_limit_margin,
            max_target_step=args.max_target_step,
            proposal_timeout_s=args.proposal_timeout,
            proposal_start_timeout_s=args.proposal_start_timeout,
            max_tilt_rad=math.radians(args.max_tilt_deg),
            max_joint_speed=args.max_joint_speed,
            direct_policy_targets=args.direct_policy_targets,
        )
    print(
        "safe controller contract: "
        f"joints={len(init_target)} modes=zero,hold,init"
        f"{',pilot' if pilot is not None else ''} init=motion_frame_0 "
        f"target_mode={'direct_policy' if args.direct_policy_targets else 'filtered'} "
        f"pilot_authority={args.pilot_authority:.3f} "
        f"pilot_init_tolerance={args.pilot_init_tolerance:.3f} "
        f"max_target_step={'ignored' if args.direct_policy_targets else f'{args.max_target_step:.3f}'} "
        f"max_tilt_deg={args.max_tilt_deg:.1f} "
        f"max_joint_speed={args.max_joint_speed:.1f} "
        f"policy_config_sha256={hashlib.sha256(config_bytes).hexdigest()}"
    )
    if args.validate_only:
        return 0

    backend = ZMQRobotIO(G1_CFG)
    commands: queue.SimpleQueue[str] = queue.SimpleQueue()
    if args.command_fifo is not None:
        threading.Thread(
            target=_command_reader,
            args=(args.command_fifo, commands),
            daemon=True,
        ).start()

    state = None
    deadline = time.monotonic() + args.initial_state_timeout
    while time.monotonic() < deadline:
        state = backend.read_state()
        if state is not None:
            break
        time.sleep(0.01)
    if state is None:
        backend.close()
        raise RuntimeError("timed out waiting for initial G1 low-state")

    machine = SafePoseStateMachine(
        init_target,
        init_duration_s=args.init_duration,
        pilot=pilot,
    )
    current = np.asarray(state.qpos[7:], dtype=np.float32)
    machine.initialize(current)
    _write_status(args.status_file, "hold")
    last_tick = int(state.tick)
    last_state_time = time.monotonic()
    controller_started_at = last_state_time
    period = 1.0 / args.rate
    next_step = time.monotonic()
    shutdown_at: float | None = None
    first_command = True
    proposal_socket = _proposal_socket(args.proposal_port) if args.proposal_port else None
    latest_proposal: NominalProposal | None = None
    pilot_active_reported = False
    pilot_log = None
    if args.pilot_log is not None:
        args.pilot_log.parent.mkdir(parents=True, exist_ok=True)
        pilot_log = args.pilot_log.open("a", buffering=1)
    try:
        while shutdown_at is None or time.monotonic() < shutdown_at:
            next_step += period
            latest = backend.read_state()
            now = time.monotonic()
            if latest is not None and int(latest.tick) != last_tick:
                state = latest
                current = np.asarray(state.qpos[7:], dtype=np.float32)
                last_tick = int(state.tick)
                last_state_time = now
            try:
                proposal = _drain_proposal(proposal_socket)
            except ValueError as exc:
                print(f"rejected malformed nominal proposal: {exc}", flush=True)
                proposal = None
                if machine.mode == "pilot":
                    machine.set_mode("hold", current, now=now)
                    machine.last_abort_reason = "proposal_invalid"
                    _write_status(args.status_file, "pilot_abort:proposal_invalid")
            if proposal is not None:
                latest_proposal = proposal
            if (
                now - controller_started_at > args.activation_grace
                and now - last_state_time > args.state_timeout
            ):
                raise RuntimeError(
                    f"G1 low-state stale for {now - last_state_time:.3f}s"
                )
            while not commands.empty():
                command = commands.get()
                if command in {"z", "zero"}:
                    machine.set_mode("zero", current, now=now)
                    _write_status(args.status_file, "zero")
                elif command in {"h", "hold"}:
                    machine.set_mode("hold", current, now=now)
                    _write_status(args.status_file, "hold")
                elif command in {"i", "init"}:
                    machine.set_mode("init", current, now=now)
                    _write_status(args.status_file, "init")
                elif command in {"p", "pilot"}:
                    init_error = float(np.max(np.abs(current - init_target)))
                    if pilot is None:
                        print("ignored pilot command; proposal input is disabled", flush=True)
                    elif not machine.init_completed or init_error > args.pilot_init_tolerance:
                        print(
                            "ignored pilot command; completed motion-frame-0 init required "
                            f"(max error={init_error:.3f} rad)",
                            flush=True,
                        )
                    else:
                        latest_proposal = None
                        pilot_active_reported = False
                        machine.set_mode("pilot", current, now=now)
                        _write_status(args.status_file, "pilot_waiting")
                elif command in {"q", "quit", "exit"}:
                    machine.set_mode("zero", current, now=now)
                    _write_status(args.status_file, "quitting")
                    shutdown_at = now + 0.5
                elif command:
                    allowed = "z h i p q" if pilot is not None else "z h i q"
                    print(f"ignored command {command!r}; allowed: {allowed}", flush=True)
            previous_mode = machine.mode
            target = machine.target(
                current,
                now=now,
                monotonic_ns=time.monotonic_ns(),
                proposal=latest_proposal,
                quaternion_wxyz=np.asarray(state.qpos[3:7], dtype=np.float32),
                joint_velocity=np.asarray(state.qvel[6:], dtype=np.float32),
            )
            if previous_mode == "init" and machine.mode == "hold":
                _write_status(args.status_file, "init_complete")
            if previous_mode == "pilot" and machine.mode == "pilot":
                if latest_proposal is not None and not pilot_active_reported:
                    pilot_active_reported = True
                    _write_status(args.status_file, "pilot_active")
            elif previous_mode == "pilot" and machine.mode == "hold":
                _write_status(args.status_file, f"pilot_abort:{machine.last_abort_reason}")
            if pilot_log is not None and previous_mode == "pilot":
                record = {
                    "time_ns": time.time_ns(),
                    "monotonic_ns": time.monotonic_ns(),
                    "status": machine.mode,
                    "abort_reason": machine.last_abort_reason,
                    "direct_policy_targets": args.direct_policy_targets,
                    "proposal_sequence": (
                        None if latest_proposal is None else latest_proposal.sequence
                    ),
                    "reference_step": (
                        None if latest_proposal is None else latest_proposal.reference_step
                    ),
                    "proposal_source_time_ns": (
                        None if latest_proposal is None else latest_proposal.source_time_ns
                    ),
                    "tilt_deg": math.degrees(_upright_tilt_rad(state.qpos[3:7])),
                    "max_joint_speed_rad_s": float(np.max(np.abs(state.qvel[6:]))),
                    "raw_q_target": (
                        None if latest_proposal is None else latest_proposal.q_target.tolist()
                    ),
                    "applied_q_target": target.tolist(),
                    "joint_pos": current.tolist(),
                }
                pilot_log.write(json.dumps(record, separators=(",", ":")) + "\n")
            if not np.isfinite(target).all():
                raise RuntimeError("safe controller generated an invalid joint target")
            zeros = np.zeros_like(target)
            backend.write_command(target, zeros, zeros, kp, kd)
            if first_command:
                first_command = False
                if args.ready_file is not None:
                    args.ready_file.write_text("ready\n")
            sleep_s = next_step - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                # Never issue catch-up command bursts after a delayed cycle.
                next_step = time.monotonic()
    finally:
        backend.close()
        if proposal_socket is not None:
            proposal_socket.close(linger=0)
        if pilot_log is not None:
            pilot_log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
