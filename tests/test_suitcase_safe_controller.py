import numpy as np

from scripts.g1.suitcase_safe_controller import GuardedNominalPilot, SafePoseStateMachine
from somaforce_deploy.nominal_proposal import NominalProposal


def _controller():
    joints = np.zeros(29, dtype=np.float32)
    pilot = GuardedNominalPilot(
        joints,
        np.full(29, -2.0, dtype=np.float32),
        np.full(29, 2.0, dtype=np.float32),
        authority=1.0,
        joint_limit_margin=0.1,
        max_target_step=0.2,
        proposal_timeout_s=0.1,
        proposal_start_timeout_s=0.2,
        max_tilt_rad=np.pi,
        max_joint_speed=10.0,
    )
    machine = SafePoseStateMachine(joints, init_duration_s=1.0, pilot=pilot)
    machine.initialize(joints)
    machine.set_mode("pilot", joints, now=1.0)
    return machine, joints


def test_stale_proposal_aborts_to_current_position_hold():
    machine, joints = _controller()
    current = joints + 0.15
    proposal = NominalProposal(
        source_time_ns=1_000_000_000,
        sequence=1,
        reference_step=0,
        action=np.zeros(23, dtype=np.float32),
        q_target=np.ones(29, dtype=np.float32),
    )

    target = machine.target(
        current,
        now=1.5,
        monotonic_ns=1_500_000_000,
        proposal=proposal,
    )

    assert machine.mode == "hold"
    assert machine.last_abort_reason == "proposal_stale"
    np.testing.assert_array_equal(target, current)
    np.testing.assert_array_equal(machine.target(joints, now=2.0), current)


def test_missing_proposal_after_start_timeout_aborts_to_hold():
    machine, joints = _controller()
    current = joints - 0.2

    target = machine.target(current, now=1.21, proposal=None)

    assert machine.mode == "hold"
    assert machine.last_abort_reason == "proposal_start_timeout"
    np.testing.assert_array_equal(target, current)
