import numpy as np
import pytest
from somaforce_deploy.nominal import HDMIStudentNominal, HDMIStudentTwoStageNominal, SonicNominal

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

def test_hdmi_student_accepts_native_linear_output_name():
    class NativeExport:
        def __call__(self, _inputs):
            return {
                "linear_6": np.zeros((23,), np.float32),
                "mul": np.zeros((23,), np.float32),
            }

    action = HDMIStudentNominal(NativeExport()).step(
        command=np.zeros((1, 356), np.float32),
        policy=np.zeros((1, 249), np.float32),
        object_obs=np.zeros((1, 10), np.float32),
    )
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


def test_sonic_29d_action_is_selected_in_cross_order():
    class Sonic:
        def __call__(self, inputs):
            assert inputs == {"g1_input": "motion", "proprioception": "state"}
            return {"action": np.arange(29, dtype=np.float32)}

    policy = SonicNominal(
        Sonic(),
        lambda **_: {"g1_input": "motion", "proprioception": "state"},
    )
    action = policy.step(command="motion", policy="state")
    expected = np.arange(23, dtype=np.float32)[None, :]
    np.testing.assert_allclose(action, expected)


def test_sonic_rejects_non_29d_action():
    class Sonic:
        def __call__(self, _inputs):
            return {"action": np.zeros((1, 23), dtype=np.float32)}

    policy = SonicNominal(Sonic(), lambda **_: {})
    with pytest.raises(ValueError, match="Sonic action"):
        policy.step(command=None, policy=None)
