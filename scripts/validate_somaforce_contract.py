#!/usr/bin/env python3
"""CPU-only validation of the deploy contract; no robot or simulator is started."""
import numpy as np
from somaforce_deploy.contracts import ACTION_JOINT_NAMES, ActionContract
from somaforce_deploy.residual import compose_action

if __name__ == "__main__":
    ActionContract(ACTION_JOINT_NAMES, tuple([.44]*23), {})
    out=compose_action(np.zeros((1,23),np.float32), np.ones((1,23),np.float32), authority=.1, contact_gain=.2)
    assert np.allclose(out,.02)
    print("SomaForce deployment contracts: PASS")
