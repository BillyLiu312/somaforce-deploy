from types import SimpleNamespace
from pathlib import Path
import json

import numpy as np
import pytest

from scripts.run_hdmi_suitcase_nominal_shadow import (
    ShadowCommandSender,
    _build_parser,
    _freeze_reference_at_frame_zero,
    _load_reference_states,
    _live_placement,
    _percentiles,
    _save_partial_record,
    _startup_stream_timeout_message,
)
from somaforce_deploy.chunked_recording import (
    ChunkedNpzRecorder,
    finalize_chunked_recording,
)


def test_shadow_command_sender_has_no_transport_and_fails_closed():
    sender = ShadowCommandSender({}, {})

    assert sender.send_calls == 0
    with pytest.raises(RuntimeError, match="attempted to send"):
        sender.send_command(np.zeros(1))
    assert sender.send_calls == 1


def test_shadow_percentiles_are_serializable_scalars():
    result = _percentiles(np.asarray([1.0, 2.0, 3.0, 4.0]))

    assert result == pytest.approx({"p50": 2.5, "p95": 3.85, "max": 4.0})
    assert all(isinstance(value, float) for value in result.values())


def test_live_placement_preserves_direction_in_pelvis_yaw_frame():
    half = np.sqrt(0.5)
    pelvis = np.asarray([0.0, 0.0, 0.8, half, 0.0, 0.0, half])
    suitcase = np.asarray([0.0, 0.532, 0.0, half, 0.0, 0.0, half])

    placement = _live_placement(pelvis, suitcase)

    assert placement["relative_xy_m"] == pytest.approx([0.532, 0.0])
    assert placement["relative_z_m"] == pytest.approx(-0.8)
    assert placement["relative_yaw_rad"] == pytest.approx(0.0)


def test_freeze_reference_initializes_frame_zero_and_keeps_policy_live():
    class FakeMotionObservation:
        def __init__(self):
            self.t = np.array([5])
            self.motion_dataset = object()
            self.updates = []

        def update(self, state):
            self.updates.append(dict(state))
            if not state.get("paused", False):
                self.t += 1

    motion = FakeMotionObservation()
    policy = SimpleNamespace(
        observations={"command": SimpleNamespace(funcs={"motion": motion})},
        state_dict={"paused": False},
    )

    _freeze_reference_at_frame_zero(policy)

    assert motion.t.tolist() == [0]
    assert motion.updates == [{"paused": False}]
    assert policy.state_dict["paused"] is True


def test_runner_has_no_pose_substitution_or_grace_options():
    option_strings = {
        option
        for action in _build_parser()._actions
        for option in action.option_strings
    }
    removed_options = {
        "--stabilization-pose-grace",
        "--motion-pelvis-pose-grace",
        "--suitcase-attachment-fallback",
        "--suitcase-lift-threshold",
        "--suitcase-fallback-delay",
    }
    assert option_strings.isdisjoint(removed_options)
    assert {
        "--ft-port",
        "--ft-max-age-ms",
        "--ft-calibration",
        "--require-both-ft-valid",
        "--residual-mode",
        "--residual",
        "--completion-file",
        "--completion-ack-file",
    }.issubset(option_strings)


def test_suitcase_reference_record_has_full_target_state():
    states = _load_reference_states(
        Path(__file__).resolve().parents[1]
        / "assets/mujoco/reference/hdmi_suitcase/motion.npz"
    )

    assert states["joint_pos"].shape == (472, 29)
    assert states["joint_vel"].shape == (472, 29)
    assert states["body_pos_w"].shape == (472, 29, 3)
    assert states["body_quat_w"].shape == (472, 29, 4)
    assert states["object_contact"].shape == (472, 1)


def test_partial_record_keeps_completed_policy_rows(tmp_path):
    output = tmp_path / "aborted.npz"
    _save_partial_record(
        output,
        {
            "time_ns": [1, 2],
            "joint_pos": [np.zeros(29), np.ones(29)],
            "unfinished_field": [np.zeros(1)],
        },
        reason="SIGTERM",
        residual_mode="shadow",
    )

    with np.load(output, allow_pickle=False) as record:
        assert record["time_ns"].tolist() == [1, 2]
        assert record["joint_pos"].shape == (2, 29)
        assert "unfinished_field" not in record
        metadata = json.loads(str(record["metadata"]))
    assert metadata["complete"] is False
    assert metadata["termination_reason"] == "SIGTERM"


def test_chunked_recording_recovers_every_completed_field(tmp_path):
    output = tmp_path / "trial.npz"
    records = {
        "time_ns": [],
        "joint_pos": [],
        "ft_wrench_sensor": [],
    }
    recorder = ChunkedNpzRecorder(
        output,
        tuple(records),
        base_metadata={"mode": "off"},
        chunk_size=2,
    )
    for index in range(5):
        records["time_ns"].append(index)
        records["joint_pos"].append(np.full(29, index, dtype=np.float32))
        records["ft_wrench_sensor"].append(
            np.full((2, 6), index, dtype=np.float32)
        )
        recorder.capture(records)
    assert recorder.close(records, status="partial", reason="fall") == 5

    frames = finalize_chunked_recording(
        output,
        metadata={"schema": "test"},
        complete=False,
        reason="fall",
    )

    assert frames == 5
    with np.load(output, allow_pickle=False) as record:
        assert record["time_ns"].tolist() == [0, 1, 2, 3, 4]
        assert record["joint_pos"].shape == (5, 29)
        assert record["ft_wrench_sensor"].shape == (5, 2, 6)
        metadata = json.loads(str(record["metadata"]))
    assert metadata["complete"] is False
    assert metadata["termination_reason"] == "fall"
    assert metadata["mode"] == "off"

    # The independently readable chunks remain sufficient to rebuild a lost NPZ.
    output.unlink()
    finalize_chunked_recording(
        output,
        metadata={"schema": "test"},
        complete=False,
        reason="recovered",
    )
    with np.load(output, allow_pickle=False) as recovered:
        np.testing.assert_array_equal(recovered["joint_pos"][:, 0], np.arange(5))


def test_startup_timeout_identifies_missing_and_stale_streams():
    message = _startup_stream_timeout_message(
        low_state_ready=True,
        policy_pose_missing=["pelvis_pos", "pelvis_quat"],
        latest_pose_time={"pelvis": 0.0, "suitcase": 9.5},
        now=10.0,
        stale_timeout=0.25,
    )

    assert "missing_or_stale=['pelvis', 'suitcase']" in message
    assert "low_state=ready" in message
    assert "pelvis=never_received" in message
    assert "suitcase=age_0.500s" in message
    assert "policy_pose_missing=['pelvis_pos', 'pelvis_quat']" in message
