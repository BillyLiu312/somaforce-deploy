#!/usr/bin/env python3
"""CPU-only validation of the deploy contract; no robot or simulator is started."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from somaforce_deploy.contracts import ACTION_JOINT_NAMES, ActionContract
from somaforce_deploy.nominal import HDMIStudentTwoStageNominal
from somaforce_deploy.residual import CrossResidual, compose_action
from somaforce_deploy.runtime import DeploymentStack


class _Adapt:
    def __call__(self, inputs):
        assert set(inputs) == {"policy", "command", "object"}
        return {"priv_pred": np.zeros((1, 256), dtype=np.float32)}


class _Actor:
    def __call__(self, inputs):
        assert {"command", "policy", "priv_pred"} <= set(inputs)
        return {"action": np.zeros((1, 23), dtype=np.float32)}


class _Residual:
    def __call__(self, inputs):
        assert inputs["wrist_tokens"].shape == (1, 2, 16, 14)
        return {"residual": np.ones((1, 23), dtype=np.float32)}


if __name__ == "__main__":
    ActionContract(ACTION_JOINT_NAMES, tuple([.44] * 23), {})
    nominal = HDMIStudentTwoStageNominal(_Adapt(), _Actor())
    stack = DeploymentStack(
        nominal=nominal,
        residual=CrossResidual(_Residual()),
        mode="hdmi_student_residual",
        authority=.1,
        contact_gain=.2,
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
    assert np.allclose(result.applied, .02)
    assert np.allclose(
        compose_action(
            np.zeros((1, 23), np.float32),
            np.ones((1, 23), np.float32),
            authority=.1,
            contact_gain=.2,
        ),
        .02,
    )
    print("SomaForce deployment contracts: PASS")
