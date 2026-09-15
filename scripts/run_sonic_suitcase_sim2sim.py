#!/usr/bin/env python3
"""Run the official split Sonic G1 policy on the suitcase motion in MuJoCo."""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import yaml

from somaforce_deploy.sonic import validate_sonic_bundle


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs/sonic/suitcase_g1.yaml"
DEFAULT_MOTION = REPO_ROOT / "assets/mujoco/reference/hdmi_suitcase/motion.npz"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=Path, default=Path("/absolute/path/sonic_hf/model_encoder.onnx"))
    parser.add_argument("--decoder", type=Path, default=Path("/absolute/path/sonic_hf/model_decoder.onnx"))
    parser.add_argument("--policy-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument(
        "--mjcf-path",
        type=Path,
        default=REPO_ROOT / "assets/mujoco/hdmi_tag/upstream/data/robots/g1/g1_29dof_rubberhand-suitcase.xml",
        help="MuJoCo scene containing both the G1 and the suitcase free joint.",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/sonic_suitcase_sim2sim")
    parser.add_argument("--sim-dt", type=float, default=0.002)
    parser.add_argument("--env-dt", type=float, default=0.02)
    parser.add_argument("--initial-pause-s", type=float, default=0.0)
    parser.add_argument("--inference-backend", choices=("onnx-cpu", "onnx-gpu", "tensorrt"), default="onnx-cpu")
    parser.add_argument("--run-once", action="store_true", default=True)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--stop-on-tracking-failure", action="store_true")
    args = parser.parse_args()

    from sim2real.sim_env.integrated_sim2sim import IntegratedSim2Sim, IntegratedSim2SimArgs

    encoder = args.encoder.expanduser().resolve()
    decoder = args.decoder.expanduser().resolve()
    bundle = validate_sonic_bundle(encoder, decoder)
    config = yaml.safe_load(args.policy_config.expanduser().read_text())
    config["sonic_encoder_model"] = str(encoder)
    config["sonic_decoder_model"] = str(decoder)
    config["model_path"] = str(decoder)
    config["object_joint_name"] = "suitcase_root"
    config["object_motion_body_name"] = "suitcase"
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="sonic_suitcase_config_") as temp_dir:
        config_path = Path(temp_dir) / "suitcase_g1.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        runtime_args = IntegratedSim2SimArgs(
            policy_config=str(config_path),
            motion_path=str(args.motion.expanduser().resolve()),
            robot="g1",
            env_dt=float(args.env_dt),
            sim_dt=float(args.sim_dt),
            mjcf_path=str(args.mjcf_path.expanduser().resolve()),
            initial_pause_s=float(args.initial_pause_s),
            inference_backend=args.inference_backend,
            headless=True,
            run_once=True,
            record=bool(args.record),
            record_output=str(output_dir / "policy_record.npz") if args.record else None,
            root_trajectory_output=str(output_dir / "root_trajectory.npz"),
            trajectory_output=str(output_dir / "trajectory.npz"),
            stop_on_tracking_failure=bool(args.stop_on_tracking_failure),
        )
        print({"sonic_bundle": bundle, "motion": str(args.motion.resolve()), "output_dir": str(output_dir)})
        IntegratedSim2Sim(runtime_args).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
