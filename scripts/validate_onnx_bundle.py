#!/usr/bin/env python3
"""Run a deterministic ONNX bundle smoke through the deploy composition layer."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from sim2real.rl_policy.inference import build_inference_module
from somaforce_deploy.nominal import HDMIStudentNominal
from somaforce_deploy.residual import CrossResidual
from somaforce_deploy.runtime import DeploymentStack


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--residual", type=Path, required=True)
    parser.add_argument("--authority", type=float, default=0.0)
    parser.add_argument("--contact-gain", type=float, default=0.0)
    parser.add_argument("--shadow", action="store_true")
    args = parser.parse_args()

    student = HDMIStudentNominal(
        build_inference_module(str(args.student), "onnx-cpu")
    )
    residual = CrossResidual(
        build_inference_module(str(args.residual), "onnx-cpu")
    )
    stack = DeploymentStack(
        nominal=student,
        residual=residual,
        mode="hdmi_student_residual_shadow" if args.shadow else "hdmi_student_residual",
        authority=args.authority,
        contact_gain=args.contact_gain,
    )
    result = stack.step(
        nominal_kwargs={
            "command": np.zeros((1, 356), np.float32),
            "policy": np.zeros((1, 249), np.float32),
            "object_obs": np.zeros((1, 10), np.float32),
        },
        wrist_tokens=np.zeros((1, 2, 16, 14), np.float32),
        proprio=np.zeros((1, 64), np.float32),
    )
    assert result.nominal.shape == (1, 23)
    assert result.residual.shape == (1, 23)
    assert result.composed.shape == (1, 23)
    assert result.applied.shape == (1, 23)
    print(
        "bundle smoke: PASS "
        f"nominal_max={np.abs(result.nominal).max():.6g} "
        f"residual_max={np.abs(result.residual).max():.6g} "
        f"applied_max={np.abs(result.applied).max():.6g} "
        f"shadow={result.shadow}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
