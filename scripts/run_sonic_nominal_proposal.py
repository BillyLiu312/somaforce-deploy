#!/usr/bin/env python3
"""Run a SONIC NPZ reference and publish guarded G1 nominal proposals.

This is the SONIC counterpart to ``run_hdmi_suitcase_nominal_shadow.py``. It
never writes the low-command socket; ``suitcase_safe_controller.py`` remains
the only process allowed to command a physical G1.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml
import zmq

from sim2real.rl_policy.base_policy import BasePolicyArgs
from sim2real.rl_policy.tracking import Tracking, TrackingArgs
from somaforce_deploy.nominal_proposal import SonicNominalProposal


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-config", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--motion-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=573)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--proposal-port", type=int, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--start-file", type=Path, required=True)
    parser.add_argument("--motion-start-file", type=Path, required=True)
    parser.add_argument("--completion-file", type=Path, required=True)
    parser.add_argument("--completion-ack-file", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.steps <= 0 or args.rate <= 0:
        raise ValueError("steps and rate must be positive")
    if not args.motion_root.is_dir():
        raise FileNotFoundError(f"SONIC motion root is missing: {args.motion_root}")
    config = yaml.safe_load(args.policy_config.read_text())
    if args.model is not None:
        config["model_path"] = str(args.model.resolve())
    with tempfile.TemporaryDirectory(prefix="sonic_proposal_") as temp_dir:
        config_path = Path(temp_dir) / "policy.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        policy_args = TrackingArgs(
            policy_config=str(config_path),
            robot="g1",
            rl_rate=args.rate,
            inference_backend="onnx-cpu",
            robot_io="zmq",
            robot_interface="",
            controller="keyboard",
            record=False,
            motion_backend="npz",
            motion_path=str(args.motion_root.resolve()),
        )
        # BasePolicy's keyboard controller is constructed for compatibility,
        # but no controller or ActionManager command is used in this runner.
        policy = Tracking(args=policy_args)
        policy.controller.close = lambda: None
        policy.state_dict = {
            "action": np.zeros(policy.num_actions, dtype=np.float32),
            "paused": False,
            "control_mode": "policy",
        }
        policy.reset()
        policy.state_dict["paused"] = False
        policy.state_dict["control_mode"] = "policy"

        context = zmq.Context.instance()
        proposal_socket = context.socket(zmq.PUB)
        proposal_socket.setsockopt(zmq.SNDHWM, 1)
        proposal_socket.setsockopt(zmq.LINGER, 0)
        proposal_socket.bind(f"tcp://127.0.0.1:{args.proposal_port}")
        args.ready_file.write_text("ready\n")
        deadline = time.monotonic() + 60.0
        while not args.start_file.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for safe-controller start")
            time.sleep(0.02)
        while not policy.state_processor._prepare_low_state():
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for G1 low-state")
            time.sleep(0.01)

        records: list[dict[str, object]] = []
        period = 1.0 / args.rate
        next_tick = time.perf_counter()
        for step in range(args.steps):
            if not args.motion_start_file.exists():
                time.sleep(0.01)
                continue
            if not policy.state_processor._prepare_low_state():
                raise RuntimeError("G1 low-state became unavailable")
            policy.update()
            observations, _ = policy.prepare_obs_for_rl()
            policy.state_dict.update(observations)
            policy.state_dict["is_init"] = np.zeros(1, dtype=bool)
            action, q_target, next_state = policy.policy(policy.state_dict)
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            q_target = np.asarray(q_target, dtype=np.float32).reshape(-1)
            if q_target.shape != (policy.num_dofs,) or not np.isfinite(q_target).all():
                raise RuntimeError(f"SONIC returned invalid q_target shape={q_target.shape}")
            if tuple(policy.joint_names_simulation) != tuple(policy_args_for_g1(policy)):
                raise RuntimeError("SONIC simulation joint order is not canonical G1")
            proposal_socket.send(
                SonicNominalProposal(
                    source_time_ns=time.monotonic_ns(),
                    sequence=step + 1,
                    reference_step=step,
                    action=action,
                    q_target=q_target,
                ).to_bytes(),
                flags=zmq.DONTWAIT,
            )
            records.append({"step": step, "q_target": q_target.copy(), "action": action.copy()})
            policy.state_dict = next_state
            policy.state_dict["action"] = action
            policy.state_dict["paused"] = False
            next_tick += period
            time.sleep(max(0.0, next_tick - time.perf_counter()))

        args.completion_file.write_text(json.dumps({"steps": len(records), "backend": "sonic"}) + "\n")
        deadline = time.monotonic() + 10.0
        while not args.completion_ack_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output,
            q_target=np.asarray([record["q_target"] for record in records], dtype=np.float32),
            action=np.asarray([record["action"] for record in records], dtype=np.float32),
            backend=np.asarray("sonic"),
        )
        proposal_socket.close(linger=0)
        policy.robot_io.close()
    return 0


def policy_args_for_g1(policy: Tracking) -> list[str]:
    """Return the canonical physical qpos order used by the proposal ABI."""
    from sim2real.config.robots.g1 import G1_CFG

    return list(G1_CFG.joint_names)


if __name__ == "__main__":
    raise SystemExit(main())
