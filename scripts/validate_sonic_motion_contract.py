#!/usr/bin/env python3
"""Validate Sonic reference-motion inputs before loading a checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from somaforce_deploy.hdmi_sim2sim import TASKS, task_motion_dir
from somaforce_deploy.contracts import G1_JOINT_NAMES
from somaforce_deploy.sonic import (
    SONIC_FUTURE_STEPS,
    SONIC_CONTROL_HZ,
    SONIC_TO_CROSS_INDICES,
    validate_sonic_bundle,
    validate_sonic_onnx,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = ("move_suitcase", "push_door_hand", "push_box", "move_largebox")


def _load_motion(path: Path) -> dict[str, Any]:
    meta_path = path.with_name("meta.json")
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"motion and meta.json are both required under {path.parent}")
    meta = json.loads(meta_path.read_text())
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    required = {
        "joint_pos": (None, None),
        "joint_vel": (None, None),
        "body_pos_w": (None, None, 3),
        "body_quat_w": (None, None, 4),
        "body_lin_vel_w": (None, None, 3),
        "body_ang_vel_w": (None, None, 3),
    }
    for name, expected in required.items():
        if name not in arrays:
            raise ValueError(f"{path}: missing motion array {name!r}")
        actual = arrays[name].shape
        if len(actual) != len(expected) or any(
            wanted is not None and actual[index] != wanted
            for index, wanted in enumerate(expected)
        ):
            raise ValueError(f"{path}: {name} has shape {actual}, expected {expected}")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{path}: {name} contains non-finite values")

    joint_names = tuple(meta.get("joint_names", ()))
    body_names = tuple(meta.get("body_names", ()))
    canonical = tuple(G1_JOINT_NAMES)
    missing = [name for name in canonical if name not in joint_names]
    canonical_indices = [joint_names.index(name) for name in canonical if name in joint_names]
    issues: list[str] = []
    if missing:
        issues.append(f"missing canonical Sonic joints: {missing}")
    if canonical_indices != sorted(canonical_indices):
        issues.append("canonical Sonic joints are not in motion order")
    if arrays["joint_pos"].shape[1] != len(joint_names):
        issues.append("joint_pos width does not match meta joint_names")
    if "pelvis" not in body_names:
        raise ValueError(f"{path}: body_names must contain pelvis")
    fps = float(meta.get("fps", 0.0))
    if not np.isclose(fps, SONIC_CONTROL_HZ):
        raise ValueError(f"{path}: expected fps={SONIC_CONTROL_HZ}, got {fps}")
    frame_count = int(arrays["joint_pos"].shape[0])
    if frame_count <= max(SONIC_FUTURE_STEPS):
        issues.append(f"only {frame_count} frames for future step {max(SONIC_FUTURE_STEPS)}")
    return {
        "path": str(path.resolve()),
        "frames": frame_count,
        "fps": fps,
        "future_steps": list(SONIC_FUTURE_STEPS),
        "lookahead_s": max(SONIC_FUTURE_STEPS) / SONIC_CONTROL_HZ,
        "joint_count": len(joint_names),
        "canonical_joint_count": len(canonical),
        "extra_joint_names": [name for name in joint_names if name not in canonical],
        "compatible": not issues,
        "issues": issues,
        "body_count": len(body_names),
        "object_contact_present": "object_contact" in arrays,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(TASKS), action="append")
    parser.add_argument("--motion", type=Path, action="append")
    parser.add_argument("--sonic-onnx", type=Path, help="Legacy single-graph Sonic export")
    parser.add_argument("--sonic-encoder", type=Path)
    parser.add_argument("--sonic-decoder", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.task and args.motion:
        parser.error("use either --task or --motion, not both")

    paths = list(args.motion or ())
    if args.task:
        paths = [task_motion_dir(TASKS[name], REPO_ROOT) / "motion.npz" for name in args.task]
    if not paths:
        paths = [task_motion_dir(TASKS[name], REPO_ROOT) / "motion.npz" for name in DEFAULT_TASKS]

    report: dict[str, Any] = {
        "schema": "sonic_motion_contract_v1",
        "control_hz": SONIC_CONTROL_HZ,
        "future_steps": list(SONIC_FUTURE_STEPS),
        "cross_selection": SONIC_TO_CROSS_INDICES.tolist(),
        "motions": [_load_motion(path.expanduser().resolve()) for path in paths],
    }
    if args.sonic_onnx is not None and (args.sonic_encoder is not None or args.sonic_decoder is not None):
        parser.error("use either --sonic-onnx or --sonic-encoder/--sonic-decoder")
    if (args.sonic_encoder is None) != (args.sonic_decoder is None):
        parser.error("--sonic-encoder and --sonic-decoder must be provided together")
    if args.sonic_onnx is not None:
        report["onnx"] = validate_sonic_onnx(args.sonic_onnx)
    if args.sonic_encoder is not None:
        report["onnx"] = validate_sonic_bundle(args.sonic_encoder, args.sonic_decoder)

    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
        print(f"saved Sonic motion contract: {args.output}")
    return 0 if all(item["compatible"] for item in report["motions"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
