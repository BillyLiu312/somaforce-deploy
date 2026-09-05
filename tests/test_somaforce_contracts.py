import numpy as np
import pytest
from somaforce_deploy.contracts import ACTION_JOINT_NAMES, ActionContract
from somaforce_deploy.residual import compose_action

def test_action_contract_and_single_scaling_boundary():
    contract = ActionContract(joint_names=ACTION_JOINT_NAMES, action_scale=tuple([0.44] * 23), default_joint_pos={})
    assert contract.control_hz == 50.0
    total = compose_action(np.zeros((1, 23), np.float32), np.ones((1, 23), np.float32), authority=.25, contact_gain=.5)
    np.testing.assert_allclose(total, .125)

def test_action_contract_rejects_wrong_order():
    with pytest.raises(ValueError):
        ActionContract(joint_names=tuple(reversed(ACTION_JOINT_NAMES)), action_scale=tuple([.44] * 23), default_joint_pos={})
