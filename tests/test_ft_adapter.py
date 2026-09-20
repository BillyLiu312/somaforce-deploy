import json
from pathlib import Path
import struct
import time

import numpy as np
import pytest
import yaml

from somaforce_deploy.ft_adapter import (
    CALIBRATION_SCHEMA,
    RAW_SENSOR_SCHEMA,
    FTAdapterConfig,
    FTTokenAdapter,
    RawWrenchSample,
    WristCalibration,
    WristKinematics,
    estimate_stationary_tare,
    gravity_wrench_sensor,
    transform_sensor_wrench,
)
from somaforce_deploy.hdmi_residual_runtime import ResidualFTFrame, ResidualFTReceiver
from scripts.run_hps_ft_adapter import _snapshot_wrench_samples


REPO_ROOT = Path(__file__).resolve().parents[1]


def _raw(
    wrench=(0, 0, 0, 0, 0, 0),
    *,
    device_id=10,
    status=0,
    sequence=1,
    received_ns=1_000_000_000,
):
    return RawWrenchSample(
        device_id=device_id,
        status=status,
        source_time_ns=123,
        sequence=sequence,
        wrench_sensor=np.asarray(wrench, dtype=np.float64),
        received_monotonic_ns=received_ns,
    )


def _calibration(device_id, *, transform=None, sign=1.0):
    return WristCalibration(
        device_id=device_id,
        measurement_sign=sign,
        bias_sensor=np.zeros(6),
        wrist_from_sensor=np.eye(4) if transform is None else transform,
        force_limit_N=500.0,
        moment_limit_Nm=50.0,
    )


def _config():
    return FTAdapterConfig(
        wrists=(_calibration(10), _calibration(20)),
        smoothing_alpha=0.0,
    )


def _kinematics(received_ns=1_000_000_000):
    return WristKinematics(
        base_quaternion_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        world_from_wrist_rotation=np.stack((np.eye(3), np.eye(3))),
        twist_base_yaw=np.asarray(
            [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=np.float64
        ),
        received_monotonic_ns=received_ns,
    )


def test_parse_versioned_raw_sensor_message():
    payload = {
        "schema": RAW_SENSOR_SCHEMA,
        "monotonic_ns": 123456,
        "sequence": 7,
        "device_id": 18174,
        "status": 0,
        "fx": 1,
        "fy": 2,
        "fz": 3,
        "mx": 4,
        "my": 5,
        "mz": 6,
    }
    sample = RawWrenchSample.from_json_line(
        json.dumps(payload), received_monotonic_ns=999
    )
    assert sample.device_id == 18174
    assert sample.sequence == 7
    assert sample.source_time_ns == 123456
    assert sample.received_monotonic_ns == 999
    np.testing.assert_array_equal(sample.wrench_sensor, [1, 2, 3, 4, 5, 6])


def test_sample_snapshot_timestamp_is_after_both_client_reads():
    class Client:
        def __init__(self, device_id):
            self.device_id = device_id

        def latest(self):
            return _raw(
                device_id=self.device_id,
                received_ns=time.monotonic_ns(),
            )

    samples, snapshot_ns = _snapshot_wrench_samples((Client(10), Client(20)))

    assert all(sample.received_monotonic_ns <= snapshot_ns for sample in samples)


def test_legacy_raw_sensor_message_is_opt_in():
    payload = {
        "timestamp_ms": 100,
        "device_id": 1,
        "status": 0,
        "fx": 0,
        "fy": 0,
        "fz": 0,
        "mx": 0,
        "my": 0,
        "mz": 0,
    }
    with pytest.raises(ValueError, match="schema"):
        RawWrenchSample.from_json_line(json.dumps(payload))
    sample = RawWrenchSample.from_json_line(
        json.dumps(payload), allow_legacy=True, legacy_sequence=9
    )
    assert sample.clock == "realtime"
    assert sample.sequence == 9


def test_versioned_raw_sensor_message_rejects_missing_sequence():
    payload = {
        "schema": RAW_SENSOR_SCHEMA,
        "monotonic_ns": 100,
        "device_id": 1,
        "status": 0,
        "fx": 0,
        "fy": 0,
        "fz": 0,
        "mx": 0,
        "my": 0,
        "mz": 0,
    }
    with pytest.raises(ValueError, match="timestamp/sequence"):
        RawWrenchSample.from_json_line(json.dumps(payload))


def test_wrench_transform_rotates_and_shifts_moment_to_wrist_origin():
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(((0, -1, 0), (1, 0, 0), (0, 0, 1)))
    transform[:3, 3] = [0.0, 0.1, 0.0]
    calibration = _calibration(10, transform=transform)
    result = transform_sensor_wrench(
        _raw((10, 0, 0, 0, 0, 0)),
        calibration,
        world_from_wrist_rotation=np.eye(3),
        world_from_base_yaw_rotation=np.eye(3),
    )
    # Sensor +X maps to wrist +Y. r=[0,.1,0] is parallel to force, so no moment.
    np.testing.assert_allclose(result, [0, 10, 0, 0, 0, 0], atol=1e-6)

    transform[:3, 3] = [0.1, 0.0, 0.0]
    calibration = _calibration(10, transform=transform)
    result = transform_sensor_wrench(
        _raw((10, 0, 0, 0, 0, 0)),
        calibration,
        world_from_wrist_rotation=np.eye(3),
        world_from_base_yaw_rotation=np.eye(3),
    )
    # r x F = [0.1,0,0] x [0,10,0] = [0,0,1] Nm.
    np.testing.assert_allclose(result, [0, 10, 0, 0, 0, 1], atol=1e-6)


def test_adapter_builds_training_order_and_handles_one_stale_side():
    adapter = FTTokenAdapter(_config())
    frame = adapter.build_frame(
        (
            _raw((10, 0, 0, 0, 0, 0), device_id=10),
            _raw((0, 20, 0, 0, 0, 0), device_id=20, received_ns=0),
        ),
        _kinematics(),
        now_monotonic_ns=1_000_000_000,
        publish_time_ns=2_000_000_000,
    )
    assert frame.token.shape == (2, 14)
    np.testing.assert_allclose(frame.token[0, :6], [0.1, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(frame.token[0, 6:12], [1, 2, 3, 4, 5, 6])
    assert frame.token[0, 12] > 0.5
    assert frame.token[0, 13] == 1.0
    np.testing.assert_allclose(frame.token[1, :6], 0.0)
    np.testing.assert_allclose(frame.token[1, 6:12], [7, 8, 9, 10, 11, 12])
    assert frame.token[1, 12] == 0.0
    assert frame.token[1, 13] == 0.0
    assert frame.contact_count == 1


def test_adapter_fails_open_when_robot_state_is_stale():
    adapter = FTTokenAdapter(_config())
    frame = adapter.build_frame(
        (_raw((100, 0, 0, 0, 0, 0)), _raw(device_id=20)),
        _kinematics(received_ns=0),
        now_monotonic_ns=1_000_000_000,
    )
    np.testing.assert_allclose(frame.token[:, :12], 0.0)
    np.testing.assert_allclose(frame.token[:, 13], 0.0)
    assert frame.contact_count == 0


def test_adapter_frame_roundtrips_existing_residual_transport():
    adapter = FTTokenAdapter(_config())
    frame = adapter.build_frame(
        (_raw((10, 0, 0, 0, 0, 0)), _raw(device_id=20)),
        _kinematics(),
        now_monotonic_ns=1_000_000_000,
        publish_time_ns=2_000_000_000,
    )
    decoded = type(frame).from_bytes(frame.to_bytes())
    assert decoded.timestamp_ns == frame.timestamp_ns
    assert decoded.sequence == frame.sequence
    assert decoded.publish_monotonic_ns == 1_000_000_000
    assert decoded.kinematics_monotonic_ns == 1_000_000_000
    np.testing.assert_array_equal(
        decoded.sample_received_monotonic_ns, [1_000_000_000, 1_000_000_000]
    )
    np.testing.assert_array_equal(decoded.sample_source_time_ns, [123, 123])
    np.testing.assert_array_equal(decoded.sample_sequence, [1, 1])
    np.testing.assert_array_equal(decoded.sample_device_id, [10, 20])
    np.testing.assert_array_equal(decoded.sample_status, [0, 0])
    np.testing.assert_array_equal(
        decoded.wrench_sensor,
        [[10, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]],
    )
    np.testing.assert_array_equal(decoded.token, frame.token)
    np.testing.assert_array_equal(decoded.wrench_base_yaw, frame.wrench_base_yaw)


def test_residual_transport_decodes_legacy_frame_without_timing_metadata():
    token = np.arange(28, dtype=np.float32).reshape(2, 14)
    wrench = np.arange(12, dtype=np.float32).reshape(2, 6)
    payload = struct.pack("<Qqif", 123, 7, 2, 3.5) + np.concatenate(
        (token.reshape(-1), wrench.reshape(-1))
    ).astype("<f4").tobytes()

    decoded = ResidualFTFrame.from_bytes(payload)

    assert decoded.timestamp_ns == 123
    assert decoded.publish_monotonic_ns == 0
    assert decoded.kinematics_monotonic_ns == -1
    np.testing.assert_array_equal(decoded.sample_sequence, [-1, -1])
    np.testing.assert_array_equal(decoded.sample_device_id, [-1, -1])
    np.testing.assert_array_equal(decoded.sample_status, [-1, -1])
    np.testing.assert_array_equal(decoded.wrench_sensor, np.zeros((2, 6)))
    np.testing.assert_array_equal(decoded.token, token)
    np.testing.assert_array_equal(decoded.wrench_base_yaw, wrench)


def test_unavailable_residual_frame_disables_both_sides():
    frame = ResidualFTFrame.unavailable(timestamp_ns=123, sequence=9)
    assert frame.timestamp_ns == 123
    assert frame.sequence == 9
    np.testing.assert_array_equal(frame.token, np.zeros((2, 14), np.float32))
    np.testing.assert_array_equal(frame.wrench_base_yaw, np.zeros((2, 6), np.float32))


def test_receiver_reuses_fresh_cache_then_fails_open_without_blocking():
    class EmptySocket:
        def poll(self, *, timeout, flags):
            assert timeout == 0
            return False

    cached = ResidualFTFrame(
        timestamp_ns=99_000_000_000,
        sequence=9,
        contact_count=0,
        total_force_norm=0.0,
        token=np.zeros((2, 14), dtype=np.float32),
        wrench_base_yaw=np.zeros((2, 6), dtype=np.float32),
        publish_monotonic_ns=1_000_000_000,
    )
    cached.token[:, 13] = 1.0
    receiver = ResidualFTReceiver.__new__(ResidualFTReceiver)
    receiver.socket = EmptySocket()
    receiver._latest = cached

    fresh = receiver.receive_latest_or_unavailable(
        now_ns=500, now_monotonic_ns=1_050_000_000, max_age_ms=100
    )
    assert fresh is cached
    stale = receiver.receive_latest_or_unavailable(
        now_ns=500, now_monotonic_ns=1_101_000_000, max_age_ms=100
    )
    assert stale is not cached
    np.testing.assert_array_equal(stale.token[:, 13], 0.0)


def test_calibration_requires_explicit_valid_marker():
    payload = {
        "schema": CALIBRATION_SCHEMA,
        "valid": False,
        "sides": {},
    }
    with pytest.raises(ValueError, match="not marked valid"):
        FTAdapterConfig.from_mapping(payload)


def test_calibration_mapping_preserves_left_right_device_order():
    side = {
        "measurement_sign": 1,
        "bias_sensor": [0, 0, 0, 0, 0, 0],
        "wrist_from_sensor": np.eye(4).tolist(),
    }
    config = FTAdapterConfig.from_mapping(
        {
            "schema": CALIBRATION_SCHEMA,
            "valid": True,
            "sides": {
                "left": {**side, "device_id": 10},
                "right": {**side, "device_id": 20},
            },
            "token": {},
        }
    )
    assert [wrist.device_id for wrist in config.wrists] == [10, 20]


def test_residual_authority_requires_explicit_scope_and_gravity_handling():
    config = _config()
    with pytest.raises(ValueError, match="not validated"):
        config.require_residual_authority()

    payload = {
        "schema": CALIBRATION_SCHEMA,
        "valid": True,
        "validation_scope": "residual_authority",
        "upstream_distal_load_compensated": True,
        "sides": {
            "left": {
                "device_id": 10,
                "measurement_sign": 1,
                "bias_sensor": [0] * 6,
                "wrist_from_sensor": np.eye(4).tolist(),
            },
            "right": {
                "device_id": 20,
                "measurement_sign": 1,
                "bias_sensor": [0] * 6,
                "wrist_from_sensor": np.eye(4).tolist(),
            },
        },
        "token": {},
    }
    authority_config = FTAdapterConfig.from_mapping(payload)
    authority_config.require_residual_authority()


def test_stationary_runtime_tare_estimates_bias_and_zeroes_wrench():
    left = [
        _raw((10 + delta, 2, 3, 0.1, 0.2, 0.3), sequence=index)
        for index, delta in enumerate((-0.1, 0.0, 0.1))
    ]
    right = [
        _raw((4, 5 + delta, 6, 0.4, 0.5, 0.6), device_id=20, sequence=index)
        for index, delta in enumerate((-0.1, 0.0, 0.1))
    ]
    config = _config()
    tare = estimate_stationary_tare(
        (left, right),
        config,
        minimum_samples=3,
        max_force_std_N=0.2,
        max_moment_std_Nm=0.01,
    )
    np.testing.assert_allclose(tare.bias_sensor[0], [10, 2, 3, 0.1, 0.2, 0.3])
    np.testing.assert_allclose(tare.bias_sensor[1], [4, 5, 6, 0.4, 0.5, 0.6])

    adapter = FTTokenAdapter(config)
    adapter.apply_runtime_tare(tare.bias_sensor)
    frame = adapter.build_frame(
        (left[1], right[1]),
        _kinematics(),
        now_monotonic_ns=1_000_000_000,
    )
    np.testing.assert_allclose(frame.wrench_base_yaw, 0.0, atol=1e-6)
    np.testing.assert_array_equal(frame.token[:, 13], 1.0)


def test_stationary_runtime_tare_rejects_motion():
    left = [_raw((value, 0, 0, 0, 0, 0), sequence=index) for index, value in enumerate((0, 5, 10))]
    right = [_raw(device_id=20, sequence=index) for index in range(3)]
    with pytest.raises(ValueError, match="force standard deviation"):
        estimate_stationary_tare(
            (left, right),
            _config(),
            minimum_samples=3,
            max_force_std_N=1.0,
            max_moment_std_Nm=0.1,
        )


def test_gravity_aware_stationary_bias_keeps_modeled_hand_weight_out_of_bias():
    mass_kg = 0.115
    com = np.asarray([0.0, 0.0, 0.076])
    wrists = tuple(
        WristCalibration(
            device_id=device_id,
            measurement_sign=1.0,
            bias_sensor=np.zeros(6),
            wrist_from_sensor=np.eye(4),
            force_limit_N=500.0,
            moment_limit_Nm=50.0,
            downstream_mass_kg=mass_kg,
            downstream_com_sensor_m=com,
            gravity_compensation=True,
        )
        for device_id in (10, 20)
    )
    config = FTAdapterConfig(wrists=wrists, smoothing_alpha=0.0)
    rotation_world_wrist = np.asarray(
        ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (-1.0, 0.0, 0.0))
    )
    kinematics = WristKinematics(
        base_quaternion_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        world_from_wrist_rotation=np.stack(
            (rotation_world_wrist, rotation_world_wrist)
        ),
        twist_base_yaw=np.zeros((2, 6)),
        received_monotonic_ns=1_000_000_000,
    )
    gravity = gravity_wrench_sensor(
        wrists[0], world_from_wrist_rotation=rotation_world_wrist
    )
    expected_force = mass_kg * 9.80665
    np.testing.assert_allclose(
        gravity,
        [expected_force, 0.0, 0.0, 0.0, com[2] * expected_force, 0.0],
        atol=1e-8,
    )
    biases = np.asarray(
        [[0.2, -0.1, 0.3, 0.01, -0.02, 0.03], [-0.3, 0.2, 0.1, -0.01, 0.04, -0.02]]
    )
    samples = tuple(
        [
            _raw(
                biases[side] + gravity,
                device_id=(10, 20)[side],
                sequence=index,
            )
            for index in range(3)
        ]
        for side in range(2)
    )
    result = estimate_stationary_tare(
        samples,
        config,
        minimum_samples=3,
        max_force_std_N=0.2,
        max_moment_std_Nm=0.01,
        kinematics=kinematics,
    )
    np.testing.assert_allclose(result.bias_sensor, biases, atol=1e-8)
    np.testing.assert_allclose(
        result.gravity_wrench_sensor, np.stack((gravity, gravity)), atol=1e-8
    )

    adapter = FTTokenAdapter(config)
    adapter.apply_runtime_bias(result.bias_sensor)
    frame = adapter.build_frame(
        (samples[0][0], samples[1][0]),
        kinematics,
        now_monotonic_ns=1_000_000_000,
    )
    np.testing.assert_allclose(frame.wrench_base_yaw, 0.0, atol=1e-6)


def test_measured_g1_wrist_sensor_extrinsics_follow_axis_convention():
    path = REPO_ROOT / "calibration/hps_g1.yaml"
    payload = yaml.safe_load(path.read_text())
    config = FTAdapterConfig.from_mapping(payload)

    assert [item.device_id for item in config.wrists] == [18174, 18174]
    assert config.require_runtime_tare
    assert config.validation_scope == "runtime_tare_and_residual_shadow"
    assert not config.upstream_distal_load_compensated
    assert all(item.gravity_compensation for item in config.wrists)
    np.testing.assert_allclose(
        [item.downstream_mass_kg for item in config.wrists], [0.115, 0.115]
    )
    np.testing.assert_allclose(
        [item.downstream_com_sensor_m for item in config.wrists],
        [[0.0, 0.0, 0.076], [0.0, 0.0, 0.076]],
    )
    expected_weight = 0.115 * 9.80665
    np.testing.assert_allclose(
        gravity_wrench_sensor(
            config.wrists[0], world_from_wrist_rotation=np.eye(3)
        ),
        [expected_weight, 0.0, 0.0, 0.0, 0.076 * expected_weight, 0.0],
    )
    np.testing.assert_allclose(
        gravity_wrench_sensor(
            config.wrists[1], world_from_wrist_rotation=np.eye(3)
        ),
        [-expected_weight, 0.0, 0.0, 0.0, -0.076 * expected_weight, 0.0],
    )
    left = config.wrists[0].wrist_from_sensor
    right = config.wrists[1].wrist_from_sensor
    np.testing.assert_allclose(left[:3, 3], [0.0725, 0, 0])
    np.testing.assert_allclose(right[:3, 3], [0.0725, 0, 0])
    np.testing.assert_array_equal(left[:3, 0], [0, 0, -1])
    np.testing.assert_array_equal(left[:3, 1], [0, 1, 0])
    np.testing.assert_array_equal(left[:3, 2], [1, 0, 0])
    np.testing.assert_array_equal(right[:3, 0], [0, 0, 1])
    np.testing.assert_array_equal(right[:3, 1], [0, -1, 0])
    np.testing.assert_array_equal(right[:3, 2], [1, 0, 0])
    assert np.linalg.det(left[:3, :3]) == pytest.approx(1.0)
    assert np.linalg.det(right[:3, :3]) == pytest.approx(1.0)
