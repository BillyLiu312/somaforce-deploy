#!/usr/bin/env python3
"""Convert a native HDMI motion NPZ into the any4hdmi tree used by SONIC.

HDMI exports already contain the full body/joint trajectory, but SONIC's NPZ
backend intentionally accepts only an any4hdmi manifest plus qpos motions. The
converter preserves the native frame order and timing and drops task-object
joints that are not part of the canonical G1 qpos contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from any4hdmi.core.format import MOTION_DTYPE, MOTIONS_SUBDIR, save_motion, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--motion-meta", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--name", default=None, help="output motion filename (default: source stem)")
    args = parser.parse_args()

    meta = json.loads(args.motion_meta.read_text())
    manifest = json.loads(args.reference_manifest.read_text())
    qpos_names = [str(name) for name in manifest.get("qpos_names", [])]
    if len(qpos_names) < 8 or "mjcf" not in manifest:
        raise ValueError("reference manifest must contain qpos_names and mjcf")
    source_joint_names = [str(name) for name in meta.get("joint_names", [])]
    missing = [name for name in qpos_names[7:] if name not in source_joint_names]
    if missing:
        raise ValueError(f"HDMI motion is missing canonical G1 joints: {missing}")

    with np.load(args.motion, allow_pickle=False) as source:
        if "joint_pos" not in source:
            raise ValueError("HDMI motion must contain joint_pos")
        joint_pos = np.asarray(source["joint_pos"], dtype=MOTION_DTYPE)
    if joint_pos.ndim != 2 or joint_pos.shape[1] != len(source_joint_names):
        raise ValueError(
            f"joint_pos shape {joint_pos.shape} does not match metadata joint_names "
            f"({len(source_joint_names)})"
        )
    indices = [source_joint_names.index(name) for name in qpos_names[7:]]
    qpos = np.zeros((joint_pos.shape[0], len(qpos_names)), dtype=MOTION_DTYPE)
    with np.load(args.motion, allow_pickle=False) as source:
        qpos[:, :3] = np.asarray(source["body_pos_w"][:, meta["body_names"].index("pelvis")], dtype=MOTION_DTYPE)
        qpos[:, 3:7] = np.asarray(source["body_quat_w"][:, meta["body_names"].index("pelvis")], dtype=MOTION_DTYPE)
    qpos[:, 7:] = joint_pos[:, indices]
    if not np.isfinite(qpos).all():
        raise ValueError("converted qpos contains non-finite values")

    out_dir = args.out_dir.resolve()
    output_name = args.name or f"{args.motion.stem}.npz"
    output_path = out_dir / MOTIONS_SUBDIR / output_name
    save_motion(output_path, qpos)
    fps = float(meta.get("fps", 50.0))
    write_manifest(
        out_dir,
        dataset_name="hdmi_native_reference",
        mjcf=manifest["mjcf"],
        timestep=1.0 / fps,
        qpos_names=qpos_names,
        num_motions=1,
        total_hours=qpos.shape[0] / fps / 3600.0,
        source={
            "source_motion": str(args.motion.resolve()),
            "source_motion_meta": str(args.motion_meta.resolve()),
            "reference_manifest": str(args.reference_manifest.resolve()),
            "dropped_source_joints": [name for name in source_joint_names if name not in qpos_names[7:]],
        },
    )
    print(f"converted {qpos.shape[0]} HDMI frames -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
