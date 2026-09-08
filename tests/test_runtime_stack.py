import numpy as np

from somaforce_deploy.residual import CrossResidual
from somaforce_deploy.runtime import DeploymentStack


class Nominal:
    def step(self, **kwargs):
        return np.zeros((1, 23), np.float32)


class Residual:
    def __call__(self, inputs):
        return {"residual": np.ones((1, 23), np.float32)}


def test_shadow_computes_but_does_not_apply_residual():
    stack = DeploymentStack(
        nominal=Nominal(),
        residual=CrossResidual(Residual()),
        mode="hdmi_student_residual_shadow",
        authority=1.0,
        contact_gain=1.0,
    )
    result = stack.step(
        nominal_kwargs={},
        wrist_tokens=np.zeros((1, 2, 16, 14), np.float32),
        proprio=np.zeros((1, 64), np.float32),
    )
    np.testing.assert_allclose(result.residual, 1.0)
    np.testing.assert_allclose(result.composed, np.tanh(1.0))
    np.testing.assert_allclose(result.applied, 0.0)
    assert result.shadow
    assert stack.history.values.shape == (1, 23, 3)


def test_residual_receives_current_nominal_and_previous_executed_separately():
    seen = []

    class IncreasingNominal:
        def __init__(self):
            self.value = 0.0

        def step(self, **kwargs):
            self.value += 1.0
            return np.full((1, 23), self.value, np.float32)

    class CaptureResidual:
        def __call__(self, inputs):
            seen.append(
                (
                    inputs["a_nom_history"].copy(),
                    inputs["previous_a_total"].copy(),
                )
            )
            return {"residual": np.zeros((1, 23), np.float32)}

    stack = DeploymentStack(
        nominal=IncreasingNominal(),
        residual=CrossResidual(CaptureResidual()),
        mode="hdmi_student_residual",
        authority=1.0,
        contact_gain=1.0,
        action_limit=10.0,
    )
    inputs = {
        "nominal_kwargs": {},
        "wrist_tokens": np.zeros((1, 2, 16, 14), np.float32),
        "proprio": np.zeros((1, 64), np.float32),
    }
    stack.step(**inputs)
    stack.step(**inputs)
    np.testing.assert_allclose(seen[0][0][:, :, 0], 1.0)
    np.testing.assert_allclose(seen[0][1], 0.0)
    np.testing.assert_allclose(seen[1][0][:, :, 0], 2.0)
    np.testing.assert_allclose(seen[1][0][:, :, 1], 1.0)
    np.testing.assert_allclose(seen[1][1], 1.0)
