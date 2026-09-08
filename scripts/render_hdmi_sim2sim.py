#!/usr/bin/env python3
"""Render an HDMI task trajectory using its materialized task-specific scene."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import imageio.v2 as imageio
import mujoco
import numpy as np

from somaforce_deploy.hdmi_sim2sim import TASKS, get_task, materialize_scene


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, default=REPO_ROOT.parent / "sim2real-hdmi-upstream")
    parser.add_argument("--hdmi-root", type=Path, default=REPO_ROOT.parent / "HDMI")
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-azimuth", type=float, default=135.0)
    parser.add_argument("--camera-elevation", type=float, default=-18.0)
    parser.add_argument("--camera-distance", type=float, default=None)
    parser.add_argument("--follow-robot", action="store_true")
    args = parser.parse_args()

    record = np.load(args.record, allow_pickle=False)
    qpos = np.asarray(record["qpos"])
    if qpos.ndim != 2:
        raise ValueError(f"expected [T,nq] qpos, got {qpos.shape}")
    qvel = np.asarray(record["qvel"]) if "qvel" in record else None
    task = get_task(args.task)

    with TemporaryDirectory(prefix=f"render_{args.task}_") as temp_dir:
        scene_path = materialize_scene(
            task,
            upstream_root=args.upstream_root.resolve(),
            hdmi_root=args.hdmi_root.resolve(),
            output_dir=Path(temp_dir),
        )
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        if qpos.shape[1] != model.nq:
            raise ValueError(f"qpos shape {qpos.shape} does not match scene nq={model.nq}")
        data = mujoco.MjData(model)
        if task.fixed_object_body is not None:
            motion_dir = REPO_ROOT / task.motion_dir
            motion = np.load(motion_dir / "motion.npz", allow_pickle=False)
            meta = json.loads((motion_dir / "meta.json").read_text())
            body_index = meta["body_names"].index(task.fixed_object_body)
            body = model.body(task.fixed_object_body)
            model.body_pos[body.id] = motion["body_pos_w"][0, body_index]
            model.body_quat[body.id] = motion["body_quat_w"][0, body_index]
        pelvis_id = model.body("pelvis").id
        object_body_name = task.primary_object_body
        try:
            object_id = model.body(object_body_name).id
        except KeyError:
            object_body_name = f"{object_body_name}_body"
            object_id = model.body(object_body_name).id
        robot_points = []
        object_points = []
        for index, qpos_i in enumerate(qpos):
            data.qpos[:] = qpos_i
            if qvel is not None:
                data.qvel[:] = qvel[index]
            mujoco.mj_forward(model, data)
            robot_points.append(data.xpos[pelvis_id].copy())
            object_points.append(data.xpos[object_id].copy())
        robot_points = np.asarray(robot_points)
        object_points = np.asarray(object_points)
        all_points = np.concatenate((robot_points, object_points), axis=0)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.azimuth = args.camera_azimuth
        camera.elevation = args.camera_elevation
        center = 0.5 * (all_points.min(axis=0) + all_points.max(axis=0))
        center[2] = max(0.55, float(center[2]))
        camera.lookat[:] = center
        camera.distance = float(args.camera_distance or max(2.8, np.ptp(all_points[:, :2], axis=0).max() * 1.8))
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(args.output, fps=args.fps, macro_block_size=1, codec="libx264") as writer:
            for index, qpos_i in enumerate(qpos):
                data.qpos[:] = qpos_i
                if qvel is not None:
                    data.qvel[:] = qvel[index]
                mujoco.mj_forward(model, data)
                if args.follow_robot:
                    target = 0.65 * data.xpos[pelvis_id] + 0.35 * data.xpos[object_id]
                    target[2] = max(0.65, float(target[2]))
                    camera.lookat[:] = target
                renderer.update_scene(data, camera)
                writer.append_data(renderer.render())
    print(f"saved HDMI sim2sim video: {args.output} frames={len(qpos)} fps={args.fps} task={args.task}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
