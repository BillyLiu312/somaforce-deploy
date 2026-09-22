from __future__ import annotations

from typing import Literal

from sim2real.config.robots.base import RobotCfg, normalize_robot_name
from sim2real.rl_policy.robot_io.base import RobotIO
from sim2real.rl_policy.robot_io.zmq import ZMQRobotIO


def create_robot_io(
    *,
    mode: Literal["zmq", "inline"],
    robot_name: str,
    robot_cfg: RobotCfg,
    interface: str,
    command_output: bool = True,
) -> RobotIO:
    if mode == "zmq":
        if command_output:
            return ZMQRobotIO(robot_cfg)
        return ZMQRobotIO(robot_cfg, command_output=False)

    if mode != "inline":
        raise ValueError(f"Unsupported robot_io: {mode}")

    normalized_name = normalize_robot_name(robot_name)
    if normalized_name == "g1":
        from sim2real.rl_policy.robot_io.g1 import G1RobotIO

        return G1RobotIO(robot_cfg, interface=interface)

    raise NotImplementedError(
        f"robot_io='inline' is not implemented for robot={robot_name!r}"
    )
