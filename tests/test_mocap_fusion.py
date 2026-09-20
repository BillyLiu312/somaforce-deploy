import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from somaforce_deploy.mocap_fusion import (
    MarkerSource,
    MultiMarkerPoseFusion,
    load_marker_sources,
)
from scripts.calibrate_redundant_markers import _calibrate_role
from scripts.ros2_pose_to_zmq import stale_watchdog_expired


def _transform(position=(0.0, 0.0, 0.0), yaw_deg=0.0):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
    result[:3, 3] = position
    return result


def _sources():
    target_from_markers = (
        _transform((0.10, 0.00, 0.00)),
        _transform((0.00, 0.15, 0.00), 20.0),
        _transform((-0.08, 0.00, 0.12), -15.0),
    )
    return tuple(
        MarkerSource(
            name=f"marker{index + 1}",
            topic=f"/marker{index + 1}/pose",
            marker_from_target=np.linalg.inv(target_from_marker),
        )
        for index, target_from_marker in enumerate(target_from_markers)
    ), target_from_markers


def test_any_single_calibrated_marker_recovers_target_pose():
    sources, target_from_markers = _sources()
    world_from_target = _transform((1.2, -0.3, 0.8), 35.0)

    for source, target_from_marker in zip(sources, target_from_markers, strict=True):
        fusion = MultiMarkerPoseFusion(sources, stale_timeout_s=0.25)
        fusion.update(
            source.name,
            world_from_target @ target_from_marker,
            received_at=1.0,
        )
        result = fusion.resolve(now=1.01)
        assert result is not None
        assert result.used_sources == (source.name,)
        assert result.world_from_target == pytest.approx(world_from_target, abs=1e-8)


def test_three_sources_fuse_two_consistent_candidates_and_reject_outlier():
    sources, target_from_markers = _sources()
    target = _transform((0.5, 0.2, 0.9), 10.0)
    fusion = MultiMarkerPoseFusion(
        sources,
        stale_timeout_s=0.25,
        position_consensus_m=0.05,
        orientation_consensus_deg=5.0,
    )
    fusion.update("marker1", target @ target_from_markers[0], received_at=1.0)
    fusion.update(
        "marker2",
        _transform((0.01, -0.01, 0.0), 1.0) @ target @ target_from_markers[1],
        received_at=1.0,
    )
    fusion.update(
        "marker3",
        _transform((0.5, 0.0, 0.0), 45.0) @ target @ target_from_markers[2],
        received_at=1.0,
    )

    result = fusion.resolve(now=1.01)
    assert result is not None
    assert set(result.used_sources) == {"marker1", "marker2"}
    assert result.rejected_sources == ("marker3",)
    assert np.linalg.norm(result.world_from_target[:3, 3] - target[:3, 3]) < 0.02


def test_three_consistent_sources_are_averaged():
    sources, target_from_markers = _sources()
    target = _transform((0.5, 0.2, 0.9), 10.0)
    offsets = (
        _transform((-0.01, 0.0, 0.0), -1.0),
        _transform((0.0, 0.0, 0.0), 0.0),
        _transform((0.01, 0.0, 0.0), 1.0),
    )
    fusion = MultiMarkerPoseFusion(sources, stale_timeout_s=0.25)
    for source, target_from_marker, offset in zip(
        sources, target_from_markers, offsets, strict=True
    ):
        fusion.update(
            source.name,
            offset @ target @ target_from_marker,
            received_at=1.0,
        )

    result = fusion.resolve(now=1.01)

    assert result is not None
    assert result.used_sources == ("marker1", "marker2", "marker3")
    assert result.rejected_sources == ()
    candidates = [offset @ target for offset in offsets]
    expected_position = np.mean([candidate[:3, 3] for candidate in candidates], axis=0)
    expected_rotation = (
        Rotation.from_matrix(np.stack([candidate[:3, :3] for candidate in candidates]))
        .mean()
        .as_matrix()
    )
    assert result.world_from_target[:3, 3] == pytest.approx(expected_position, abs=1e-8)
    assert result.world_from_target[:3, :3] == pytest.approx(
        expected_rotation, abs=1e-8
    )


def test_preferred_source_is_used_directly_when_synchronized():
    sources, target_from_markers = _sources()
    target = _transform((0.5, 0.2, 0.9), 10.0)
    preferred_target = _transform((0.7, -0.1, 0.8), 25.0)
    fusion = MultiMarkerPoseFusion(
        sources,
        stale_timeout_s=0.25,
        synchronization_window_s=0.05,
        position_consensus_m=0.12,
        orientation_consensus_deg=12.0,
        preferred_source_name="marker3",
    )
    fusion.update("marker1", target @ target_from_markers[0], received_at=1.0)
    fusion.update("marker2", target @ target_from_markers[1], received_at=1.0)
    fusion.update(
        "marker3",
        preferred_target @ target_from_markers[2],
        received_at=1.0,
    )

    result = fusion.resolve(now=1.01)

    assert result is not None
    assert result.used_sources == ("marker3",)
    assert result.rejected_sources == ("marker1", "marker2")
    assert result.world_from_target == pytest.approx(preferred_target, abs=1e-8)


def test_missing_preferred_source_falls_back_to_consensus_mean():
    sources, target_from_markers = _sources()
    target = _transform((0.5, 0.2, 0.9), 10.0)
    offset = _transform((0.02, 0.0, 0.0), 2.0)
    fusion = MultiMarkerPoseFusion(
        sources,
        stale_timeout_s=0.25,
        synchronization_window_s=0.05,
        position_consensus_m=0.12,
        orientation_consensus_deg=12.0,
        preferred_source_name="marker3",
    )
    fusion.update("marker1", target @ target_from_markers[0], received_at=1.0)
    fusion.update(
        "marker2",
        offset @ target @ target_from_markers[1],
        received_at=1.0,
    )

    result = fusion.resolve(now=1.01)

    assert result is not None
    assert result.used_sources == ("marker1", "marker2")
    expected = np.mean([target[:3, 3], (offset @ target)[:3, 3]], axis=0)
    assert result.world_from_target[:3, 3] == pytest.approx(expected, abs=1e-8)


def test_unknown_preferred_source_is_rejected():
    sources, _ = _sources()

    with pytest.raises(ValueError, match="preferred marker source"):
        MultiMarkerPoseFusion(
            sources,
            stale_timeout_s=0.25,
            preferred_source_name="missing",
        )


def test_source_set_switch_blends_without_a_pose_step():
    sources, target_from_markers = _sources()
    original = _transform((0.2, 0.1, 0.7), -5.0)
    moved = _transform((0.5, 0.1, 0.7), -5.0)
    fusion = MultiMarkerPoseFusion(
        sources,
        stale_timeout_s=1.0,
        synchronization_window_s=0.04,
        source_switch_blend_s=0.2,
    )
    fusion.update("marker1", original @ target_from_markers[0], received_at=1.0)
    first = fusion.resolve(now=1.0)
    assert first is not None

    fusion.update("marker3", moved @ target_from_markers[2], received_at=1.2)
    switched = fusion.resolve(now=1.2)
    halfway = fusion.resolve(now=1.3)
    settled = fusion.resolve(now=1.4)

    assert switched is not None
    assert halfway is not None
    assert settled is not None
    assert switched.used_sources == ("marker3",)
    assert switched.world_from_target == pytest.approx(
        first.world_from_target, abs=1e-8
    )
    assert halfway.world_from_target[:3, 3] == pytest.approx([0.35, 0.1, 0.7], abs=1e-8)
    assert settled.world_from_target == pytest.approx(moved, abs=1e-8)


def test_stale_sources_drop_out_and_remaining_source_continues():
    sources, target_from_markers = _sources()
    target = _transform((0.2, 0.1, 0.7), -5.0)
    fusion = MultiMarkerPoseFusion(sources, stale_timeout_s=0.1)
    fusion.update("marker1", target @ target_from_markers[0], received_at=1.0)
    fusion.update("marker2", target @ target_from_markers[1], received_at=1.0)
    assert fusion.resolve(now=1.01) is not None

    moved = _transform((0.3, 0.1, 0.7), -5.0)
    fusion.update("marker3", moved @ target_from_markers[2], received_at=1.2)
    result = fusion.resolve(now=1.21)
    assert result is not None
    assert result.used_sources == ("marker3",)
    assert result.world_from_target == pytest.approx(moved, abs=1e-8)


def test_occluded_sources_leave_synchronization_window_before_stale_timeout():
    sources, target_from_markers = _sources()
    original = _transform((0.2, 0.1, 0.7), -5.0)
    moved = _transform((0.5, 0.1, 0.7), -5.0)
    fusion = MultiMarkerPoseFusion(
        sources,
        stale_timeout_s=0.25,
        synchronization_window_s=0.04,
    )
    for source, target_from_marker in zip(sources, target_from_markers, strict=True):
        fusion.update(
            source.name,
            original @ target_from_marker,
            received_at=1.0,
        )
    assert fusion.resolve(now=1.01) is not None

    fusion.update("marker3", moved @ target_from_markers[2], received_at=1.06)
    result = fusion.resolve(now=1.06)
    assert result is not None
    assert result.used_sources == ("marker3",)
    assert result.world_from_target == pytest.approx(moved, abs=1e-8)


def test_two_disagreeing_sources_without_history_fail_closed():
    sources, target_from_markers = _sources()
    fusion = MultiMarkerPoseFusion(
        sources,
        stale_timeout_s=0.25,
        position_consensus_m=0.05,
        orientation_consensus_deg=5.0,
    )
    fusion.update("marker1", target_from_markers[0], received_at=1.0)
    fusion.update(
        "marker2",
        _transform((1.0, 0.0, 0.0), 90.0) @ target_from_markers[1],
        received_at=1.0,
    )
    assert fusion.resolve(now=1.01) is None


def test_two_disagreeing_sources_fail_closed_even_after_valid_history():
    sources, target_from_markers = _sources()
    fusion = MultiMarkerPoseFusion(
        sources,
        stale_timeout_s=0.25,
        position_consensus_m=0.05,
        orientation_consensus_deg=5.0,
    )
    target = _transform((0.1, 0.0, 0.8))
    fusion.update("marker1", target @ target_from_markers[0], received_at=1.0)
    assert fusion.resolve(now=1.0) is not None

    fusion.update("marker1", target @ target_from_markers[0], received_at=1.1)
    fusion.update(
        "marker2",
        _transform((0.5, 0.0, 0.8), 45.0) @ target_from_markers[1],
        received_at=1.1,
    )
    assert fusion.resolve(now=1.11) is None


def test_v2_loader_and_legacy_loader():
    matrix = _transform((0.1, 0.2, 0.3)).tolist()
    v2 = {
        "marker_sources": {
            "suitcase": [
                {
                    "name": "case1",
                    "topic": "/suitcase1/pose",
                    "marker_from_target": matrix,
                },
                {
                    "name": "case2",
                    "topic": "/suitcase2/pose",
                    "marker_from_target": matrix,
                },
            ]
        }
    }
    sources = load_marker_sources(v2, role="suitcase", legacy_topic="/unused")
    assert [source.topic for source in sources] == [
        "/suitcase1/pose",
        "/suitcase2/pose",
    ]

    legacy = {"marker_from_suitcase": matrix}
    sources = load_marker_sources(
        legacy, role="suitcase", legacy_topic="/suitcase/pose"
    )
    assert len(sources) == 1
    assert sources[0].marker_from_target == pytest.approx(np.asarray(matrix))


def test_loader_rejects_duplicate_topics():
    source = {
        "topic": "/same/pose",
        "marker_from_target": np.eye(4).tolist(),
    }
    with pytest.raises(ValueError, match="duplicate topics"):
        load_marker_sources(
            {
                "marker_sources": {
                    "torso": [
                        {"name": "one", **source},
                        {"name": "two", **source},
                    ]
                }
            },
            role="torso",
            legacy_topic="/unused",
        )


def test_loader_applies_persistent_target_frame_correction_on_the_right():
    base = _transform((0.1, 0.2, 0.3), 15.0)
    correction = _transform((0.0, 0.0, 0.1), 180.0)
    payload = {
        "marker_sources": {
            "torso": [
                {
                    "name": "robot1",
                    "topic": "/robot1/pose",
                    "marker_from_target": base.tolist(),
                }
            ]
        }
    }
    corrections = {
        "sources": {
            "torso": {
                "robot1": {
                    "target_frame_correction": correction.tolist(),
                }
            }
        }
    }
    source = load_marker_sources(
        payload,
        role="torso",
        legacy_topic="/unused",
        corrections=corrections,
    )[0]
    assert source.marker_from_target == pytest.approx(base @ correction, abs=1e-8)


def test_relative_calibration_recovers_each_marker_transform():
    sources, target_from_markers = _sources()
    marker_from_targets = [np.linalg.inv(value) for value in target_from_markers]
    samples = []
    for step in range(20):
        world_from_target = _transform((0.01 * step, -0.02, 0.8), step * 2.0)
        samples.append(
            {
                source.name: world_from_target @ target_from_marker
                for source, target_from_marker in zip(
                    sources, target_from_markers, strict=True
                )
            }
        )

    calibrated, diagnostics = _calibrate_role(
        samples,
        [(source.name, source.topic) for source in sources],
        anchor_name="marker1",
        anchor_marker_from_target=marker_from_targets[0],
    )
    for item, expected in zip(calibrated, marker_from_targets, strict=True):
        assert np.asarray(item["marker_from_target"]) == pytest.approx(
            expected, abs=1e-8
        )
        assert diagnostics[item["name"]]["position_error_m_max"] < 1e-8
        assert diagnostics[item["name"]]["orientation_error_deg_max"] < 1e-5


def test_stale_watchdog_ends_startup_grace_after_first_valid_stream():
    assert not stale_watchdog_expired(
        (float("inf"),),
        elapsed_s=1.0,
        startup_timeout_s=10.0,
        stale_timeout_s=0.25,
        stream_started=False,
    )
    assert stale_watchdog_expired(
        (0.251,),
        elapsed_s=1.0,
        startup_timeout_s=10.0,
        stale_timeout_s=0.25,
        stream_started=True,
    )
