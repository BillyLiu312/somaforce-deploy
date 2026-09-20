from scripts.ros2_pose_to_zmq import stale_watchdog_expired


def test_watchdog_keeps_startup_grace_until_streams_are_healthy():
    assert not stale_watchdog_expired(
        (0.9,),
        elapsed_s=3.0,
        startup_timeout_s=10.0,
        stale_timeout_s=0.25,
        stream_started=False,
    )

    assert stale_watchdog_expired(
        (0.9,),
        elapsed_s=3.0,
        startup_timeout_s=10.0,
        stale_timeout_s=0.25,
        stream_started=True,
    )
