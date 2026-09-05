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
    np.testing.assert_allclose(result.composed, 1.0)
    np.testing.assert_allclose(result.applied, 0.0)
    assert result.shadow
    assert stack.history.values.shape == (1, 3, 23)
