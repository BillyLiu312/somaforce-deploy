from pathlib import Path

import numpy as np
import pytest

from somaforce_deploy.attached_object_pose import (
    ReferenceAttachedObjectEstimator,
    _pose_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MOTION = REPO_ROOT / "assets/mujoco/reference/hdmi_suitcase/motion.npz"
MOTION_META = REPO_ROOT / "assets/mujoco/reference/hdmi_suitcase/meta.json"


def _pose(position, quaternion=(1.0, 0.0, 0.0, 0.0)):
    return np.asarray([*position, *quaternion], dtype=np.float64)


def test_failed_grasp_does_not_arm_attachment_fallback():
    pelvis = np.stack([_pose((step * 0.01, 0.0, 0.8)) for step in range(4)])
    suitcase = np.stack([_pose((0.5, 0.0, 0.0)) for _ in range(4)])
    estimator = ReferenceAttachedObjectEstimator(
        pelvis,
        suitcase,
        np.ones(4, dtype=bool),
        lift_threshold_m=0.05,
    )
    estimator.begin_motion(suitcase[0])
    for step in range(3):
        estimator.observe_live(pelvis[step], suitcase[step], step)

    assert not estimator.confirmed
    assert estimator.estimate(pelvis[3], 3) is None


def test_reference_guided_fallback_reconstructs_suitcase_lift_motion():
    import json

    metadata = json.loads(MOTION_META.read_text())
    with np.load(MOTION, allow_pickle=False) as motion:
        pelvis_index = metadata["body_names"].index("pelvis")
        suitcase_index = metadata["body_names"].index("suitcase")
        pelvis = np.concatenate(
            (
                motion["body_pos_w"][:, pelvis_index],
                motion["body_quat_w"][:, pelvis_index],
            ),
            axis=1,
        )
        suitcase = np.concatenate(
            (
                motion["body_pos_w"][:, suitcase_index],
                motion["body_quat_w"][:, suitcase_index],
            ),
            axis=1,
        )

    estimator = ReferenceAttachedObjectEstimator.from_motion(MOTION, MOTION_META)
    estimator.begin_motion(suitcase[0])
    for step in range(228):
        estimator.observe_live(pelvis[step], suitcase[step], step)

    assert estimator.confirmed
    assert estimator.confirmed_step == 179
    for step in range(228, 340):
        predicted = estimator.estimate(pelvis[step], step)
        assert predicted is not None
        predicted_matrix = _pose_matrix(predicted)
        actual_matrix = _pose_matrix(suitcase[step])
        assert predicted_matrix[:3, 3] == pytest.approx(
            actual_matrix[:3, 3], abs=1e-6
        )
        assert predicted_matrix[:3, :3] == pytest.approx(
            actual_matrix[:3, :3], abs=1e-6
        )

    reacquisition = estimator.observe_live(pelvis[340], suitcase[340], 340)
    assert reacquisition is not None
    assert reacquisition.position_m == pytest.approx(0.0, abs=1e-6)
    assert reacquisition.orientation_deg == pytest.approx(0.0, abs=1e-5)
    assert not estimator.fallback_active


def test_attachment_continues_past_reference_contact_end_until_reacquisition():
    pelvis = np.stack([_pose((step * 0.1, 0.0, 0.8)) for step in range(4)])
    suitcase = np.stack(
        [
            _pose((0.5, 0.0, 0.0)),
            _pose((0.6, 0.0, 0.1)),
            _pose((0.7, 0.0, 0.1)),
            _pose((0.8, 0.0, 0.1)),
        ]
    )
    estimator = ReferenceAttachedObjectEstimator(
        pelvis,
        suitcase,
        np.asarray([False, True, False, False]),
        lift_threshold_m=0.05,
    )
    estimator.begin_motion(suitcase[0])
    estimator.observe_live(pelvis[1], suitcase[1], 1)

    assert estimator.estimate(pelvis[1], 1) is not None
    assert estimator.estimate(pelvis[2], 2) is not None
    assert estimator.estimate(pelvis[3], 3) is not None
