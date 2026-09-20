import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.quick_calibrate_robot_markers import (
    load_frame_zero_suitcase_from_torso,
    update_calibration_payload,
    validate_robot_marker_names,
)


def _transform(position=(0.0, 0.0, 0.0), yaw_deg=0.0):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    result[:3, 3] = position
    return result


def test_motion_frame_zero_reference_uses_full_suitcase_to_torso_transform(tmp_path):
    metadata = tmp_path / "meta.json"
    motion = tmp_path / "motion.npz"
    metadata.write_text(json.dumps({"body_names": ["torso_link", "suitcase"]}))
    positions = np.asarray([[[1.0, 2.0, 1.0], [0.5, 1.5, 0.1]]])
    torso_quat = Rotation.from_euler("z", 35.0, degrees=True).as_quat()
    suitcase_quat = Rotation.from_euler("z", -10.0, degrees=True).as_quat()
    quaternions = np.asarray(
        [
            [
                torso_quat[[3, 0, 1, 2]],
                suitcase_quat[[3, 0, 1, 2]],
            ]
        ]
    )
    np.savez(motion, body_pos_w=positions, body_quat_w=quaternions)

    result = load_frame_zero_suitcase_from_torso(motion, metadata)

    world_from_torso = _transform((1.0, 2.0, 1.0), 35.0)
    world_from_suitcase = _transform((0.5, 1.5, 0.1), -10.0)
    assert result == pytest.approx(
        np.linalg.inv(world_from_suitcase) @ world_from_torso, abs=1e-8
    )


def test_quick_calibration_requires_exact_robot123_marker_set():
    validate_robot_marker_names(["robot3", "robot1", "robot2"])

    with pytest.raises(ValueError, match=r"missing=\['robot3'\]"):
        validate_robot_marker_names(["robot1", "robot2"])
    with pytest.raises(ValueError, match=r"unexpected=\['robot4'\]"):
        validate_robot_marker_names(["robot1", "robot2", "robot3", "robot4"])


def test_payload_update_preserves_suitcase_and_updates_legacy_torso():
    identity = np.eye(4).tolist()
    payload = {
        "valid": True,
        "marker_from_torso": identity,
        "marker_sources": {
            "torso": [
                {
                    "name": "robot1",
                    "topic": "/robot1/pose",
                    "marker_from_target": identity,
                },
                {
                    "name": "robot2",
                    "topic": "/robot2/pose",
                    "marker_from_target": identity,
                },
            ],
            "suitcase": [
                {
                    "name": "case1",
                    "topic": "/case1/pose",
                    "marker_from_target": identity,
                }
            ],
        },
        "calibration": {"keep": True},
    }
    transforms = {
        "robot1": _transform((0.1, 0.0, 0.0)),
        "robot2": _transform((0.2, 0.0, 0.0)),
    }

    result = update_calibration_payload(payload, transforms, metadata={"run": 1})

    assert result["marker_sources"]["torso"][0]["marker_from_target"] == pytest.approx(
        transforms["robot1"]
    )
    assert result["marker_from_torso"] == pytest.approx(transforms["robot1"])
    assert result["marker_sources"]["suitcase"] == payload["marker_sources"]["suitcase"]
    assert result["calibration"] == {
        "keep": True,
        "robot_marker_quick_calibration": {"run": 1},
    }
    assert payload["marker_from_torso"] == identity
