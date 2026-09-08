#!/usr/bin/env python3
"""Run and record the external HDMI-tag MuJoCo simulator headlessly."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


class _HeadlessViewer:
    def __init__(self, seconds: float) -> None:
        self.deadline = time.monotonic() + float(seconds)
        self.cam = type("Camera", (), {})()

    def is_running(self) -> bool:
        return time.monotonic() < self.deadline

    def sync(self) -> None:
        return None

    def close(self) -> None:
        self.deadline = time.monotonic()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--upstream-root", type=Path, default=REPO_ROOT.parent / "sim2real-hdmi-upstream"
    )
    parser.add_argument("--robot-config", type=Path)
    parser.add_argument("--scene-config", type=Path)
    parser.add_argument(
        "--motion",
        type=Path,
        default=REPO_ROOT / "assets/mujoco/reference/hdmi_suitcase/motion.npz",
    )
    parser.add_argument(
        "--motion-meta",
        type=Path,
        default=REPO_ROOT / "assets/mujoco/reference/hdmi_suitcase/meta.json",
    )
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--sim-dt", type=float, default=0.002)
    parser.add_argument("--disable-elastic-band", action="store_true")
    parser.add_argument("--elastic-band-release-after", type=float, default=0.0)
    parser.add_argument("--initialize-motion-frame", action="store_true")
    args = parser.parse_args()

    upstream_root = args.upstream_root.resolve()
    robot_config_path = args.robot_config or upstream_root / "config/robot/g1.yaml"
    scene_config_path = (
        args.scene_config
        or upstream_root / "config/scene/g1_29dof_rubberhand-suitcase.yaml"
    )
    sys.path.insert(0, str(upstream_root))

    mujoco.viewer.launch_passive = lambda *_, **__: _HeadlessViewer(args.seconds)
    from sim_env.hdmi import HDMI

    robot_config = yaml.safe_load(robot_config_path.read_text())
    robot_config["LOW_CMD_PORT"] = 5591
    scene_config = yaml.safe_load(scene_config_path.read_text())
    scene_path = Path(scene_config["ROBOT_SCENE"])
    if not scene_path.is_absolute():
        scene_config["ROBOT_SCENE"] = str((upstream_root / scene_path).resolve())
    scene_config["SIMULATE_DT"] = float(args.sim_dt)
    scene_config["ENABLE_ELASTIC_BAND"] = not bool(args.disable_elastic_band)
    simulation = HDMI(robot_config, scene_config)

    if args.initialize_motion_frame:
        motion = np.load(args.motion, allow_pickle=False)
        meta = json.loads(args.motion_meta.read_text())
        for body_name in scene_config.get("publish_object_names", []):
            if body_name not in meta["body_names"]:
                continue
            joint_name = "pelvis_root" if body_name == "pelvis" else f"{body_name}_root"
            joint = simulation.mj_model.joint(joint_name)
            address = int(simulation.mj_model.jnt_qposadr[joint.id])
            body_index = meta["body_names"].index(body_name)
            simulation.mj_data.qpos[address : address + 3] = motion["body_pos_w"][0, body_index]
            simulation.mj_data.qpos[address + 3 : address + 7] = motion["body_quat_w"][0, body_index]
        for joint_name, value in zip(
            meta["joint_names"], motion["joint_pos"][0], strict=True
        ):
            joint = simulation.mj_model.joint(joint_name)
            address = int(simulation.mj_model.jnt_qposadr[joint.id])
            simulation.mj_data.qpos[address] = value
        mujoco.mj_forward(simulation.mj_model, simulation.mj_data)

    qpos_history: list[np.ndarray] = []
    original_step = simulation.sim_step

    def recorded_step() -> None:
        simulation.sim_bridge._poll_low_cmd()
        if not simulation.sim_bridge.has_received_command:
            simulation.sim_bridge.publish_low_state()
            time.sleep(simulation.sim_dt)
            return
        if (
            args.elastic_band_release_after is not None
            and simulation.scene_config["ENABLE_ELASTIC_BAND"]
            and simulation.elastic_band.enable
            and simulation.mj_data.time >= float(args.elastic_band_release_after)
        ):
            simulation.elastic_band.enable = False
            simulation.mj_data.xfrc_applied[simulation.band_attached_link, :3] = 0.0
            print(f"released elastic band at sim_time={simulation.mj_data.time:.3f}")
        original_step()
        qpos_history.append(simulation.mj_data.qpos.copy())

    simulation.sim_step = recorded_step
    simulation.sim_thread.start()
    simulation.sim_thread.join()
    args.trajectory.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.trajectory,
        qpos=np.asarray(qpos_history),
        metadata=np.asarray(
            json.dumps(
                {
                    "runtime": "external_sim2real_hdmi_tag",
                    "task": "move_suitcase",
                    "simulate_dt": float(args.sim_dt),
                    "elastic_band": not bool(args.disable_elastic_band),
                    "elastic_band_release_after": args.elastic_band_release_after,
                    "initialize_motion_frame": bool(args.initialize_motion_frame),
                    "root_anchor": False,
                },
                sort_keys=True,
            )
        ),
    )
    print(f"saved HDMI-tag trajectory: {args.trajectory} frames={len(qpos_history)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
