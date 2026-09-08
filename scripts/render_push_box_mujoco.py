#!/usr/bin/env python3
"""Render a recorded MuJoCo trajectory NPZ to MP4."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scene",
        choices=("push-box", "suitcase"),
        default="push-box",
        help="MuJoCo scene matching the recorded qpos layout",
    )
    parser.add_argument(
        "--scene-path",
        type=Path,
        default=None,
        help="explicit MJCF path; overrides --scene's bundled default",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--camera-azimuth", type=float, default=None)
    parser.add_argument("--camera-elevation", type=float, default=None)
    parser.add_argument("--camera-distance", type=float, default=None)
    parser.add_argument(
        "--camera-lookat",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="camera target; defaults to the midpoint of robot/object trajectory",
    )
    args = parser.parse_args()
    record = np.load(args.record, allow_pickle=False)
    if "qpos" not in record:
        raise ValueError("record does not contain qpos; rerun evaluator with current version")
    root = Path(__file__).resolve().parents[1]
    scenes = {
        "push-box": root / "assets/mujoco/g1_29dof_nohand/g1_29dof_nohand-box.xml",
        "suitcase": root / "assets/mujoco/hdmi_tag/upstream/data/robots/g1/g1_29dof_rubberhand-suitcase.xml",
    }
    scene = args.scene_path if args.scene_path is not None else scenes[args.scene]
    if not scene.is_file():
        raise FileNotFoundError(f"scene not found for --scene {args.scene}: {scene}")
    model = mujoco.MjModel.from_xml_path(str(scene))
    qpos = np.asarray(record["qpos"])
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(
            f"record qpos shape {qpos.shape} does not match {args.scene} model nq={model.nq}"
        )
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.azimuth = float(135.0 if args.camera_azimuth is None else args.camera_azimuth)
    camera.elevation = float(-18.0 if args.camera_elevation is None else args.camera_elevation)
    if args.camera_distance is None:
        camera.distance = 3.5
    else:
        camera.distance = float(args.camera_distance)
    if args.camera_lookat is not None:
        camera.lookat[:] = np.asarray(args.camera_lookat, dtype=np.float64)
    else:
        # Both bundled scenes store the object free joint before the robot root.
        # Center the view on the recorded robot/object positions so suitcase runs
        # (whose object starts at x=0.4) are framed without hand-tuned cameras.
        object_xyz = qpos[:, :3]
        robot_xyz = qpos[:, 7:10]
        object_center = 0.5 * (object_xyz.min(0) + object_xyz.max(0))
        robot_center = 0.5 * (robot_xyz.min(0) + robot_xyz.max(0))
        center = 0.5 * (object_center + robot_center)
        center[2] = max(0.55, float(center[2]))
        camera.lookat[:] = center
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError("video rendering requires imageio") from exc
    qvel = record["qvel"] if "qvel" in record else None
    try:
        writer = imageio.get_writer(
            args.output, fps=args.fps, macro_block_size=1, codec="libx264"
        )
    except Exception as exc:
        raise RuntimeError("video rendering requires an imageio ffmpeg writer") from exc
    try:
        for index, qpos_i in enumerate(qpos):
            data.qpos[:] = qpos_i
            if qvel is not None:
                data.qvel[:] = qvel[index]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera)
            writer.append_data(renderer.render())
    finally:
        writer.close()
    print(f"saved video: {args.output} frames={len(qpos)} fps={args.fps} scene={args.scene}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
