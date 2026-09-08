#!/usr/bin/env python3
"""Run an HDMI student export through an external HDMI-tag policy runtime."""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upstream-root", type=Path, default=REPO_ROOT.parent / "sim2real-hdmi-upstream"
    )
    parser.add_argument("--robot-config", type=Path)
    parser.add_argument(
        "--policy-config",
        type=Path,
        default=REPO_ROOT / "artifacts/hdmi_move_suitcase/hdmi_tag/policy.yaml",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT / "artifacts/hdmi_move_suitcase/hdmi_tag/student.onnx",
    )
    parser.add_argument("--model-json", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=472)
    parser.add_argument("--pose-timeout", type=float, default=5.0)
    args = parser.parse_args()

    upstream_root = args.upstream_root.resolve()
    robot_config_path = args.robot_config or upstream_root / "config/robot/g1.yaml"
    sys.path.insert(0, str(upstream_root))
    sys.path.insert(1, str(upstream_root / "rl_policy"))
    from rl_policy.tracking import Tracking

    Tracking.start_key_listener = lambda _self: None
    robot_config = yaml.safe_load(robot_config_path.read_text())
    robot_config["LOW_CMD_PORT"] = 5591
    policy_config = yaml.safe_load(args.policy_config.read_text())
    for group in policy_config.get("observation", {}).values():
        for item in group.values():
            if isinstance(item, dict) and "motion_path" in item:
                motion_path = Path(item["motion_path"])
                if not motion_path.is_absolute():
                    item["motion_path"] = str((REPO_ROOT / motion_path).resolve())
    model_path = args.model.resolve()
    model_json = args.model_json.resolve() if args.model_json else model_path.with_suffix(".json")
    with tempfile.TemporaryDirectory(prefix="hdmi_tag_model_") as temp_dir:
        if not model_json.exists() and args.model_json is None:
            sibling_policy_json = model_path.parent / "policy.json"
            if sibling_policy_json.exists():
                model_json = sibling_policy_json
        if model_json != model_path.with_suffix(".json"):
            temp_model = Path(temp_dir) / model_path.name
            shutil.copy2(model_path, temp_model)
            shutil.copy2(model_json, temp_model.with_suffix(".json"))
            model_path = temp_model
        policy = Tracking(
            robot_config=robot_config,
            policy_config=policy_config,
            model_path=str(model_path),
            rl_rate=50,
        )
        policy.state_dict = {"action": np.zeros(policy.num_actions, dtype=np.float32)}
        policy.perf_dict = {}
        object_name = "suitcase"
        for item in policy_config.get("observation", {}).get("object", {}).values():
            if isinstance(item, dict) and item.get("object_name"):
                object_name = str(item["object_name"])
                break
        pose_deadline = time.monotonic() + float(args.pose_timeout)
        while time.monotonic() < pose_deadline:
            if policy.state_processor.get_mocap_data(f"{object_name}_pos") is not None:
                break
            time.sleep(0.01)
        policy.use_policy_action = True
        policy.get_ready_state = False
        policy.reset()
        for _ in range(int(args.steps)):
            policy._rl_step_scheduled()
            time.sleep(policy.rl_dt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
