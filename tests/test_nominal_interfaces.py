import numpy as np
import pytest
from somaforce_deploy.nominal import HDMIStudentNominal, HDMIStudentTwoStageNominal

class FakeStudent:
    def __call__(self, inputs):
        assert set(inputs) == {"command", "policy", "object"}
        return {"action": np.zeros((1, 23), np.float32)}

def test_hdmi_student_requires_object_observation():
    policy = HDMIStudentNominal(FakeStudent())
    with pytest.raises(ValueError, match="object observation"):
        policy.step(command=np.zeros((1, 356), np.float32), policy=np.zeros((1, 249), np.float32))

def test_hdmi_student_output_boundary():
    action = HDMIStudentNominal(FakeStudent()).step(command=np.zeros((1, 356), np.float32), policy=np.zeros((1, 249), np.float32), object_obs=np.zeros((1, 10), np.float32))
    assert action.shape == (1, 23)

def test_hdmi_two_stage_binding():
    class Adapt:
        def __call__(self, inputs):
            assert set(inputs) == {"policy", "command", "object"}
            return {"priv_pred": np.zeros((1, 256), np.float32)}
    class Actor:
        def __call__(self, inputs):
            return {"action": np.zeros((1, 23), np.float32)}
    action = HDMIStudentTwoStageNominal(Adapt(), Actor()).step(command=np.zeros((1, 356), np.float32), policy=np.zeros((1, 249), np.float32), object_obs=np.zeros((1, 10), np.float32))
    assert action.shape == (1, 23)
