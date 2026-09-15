---
title: ROS2 Pose Adapter
sidebar_position: 3
---

# ROS2 Pose Adapter

The HDMI suitcase policy does not subscribe to ROS2 directly. It expects the
legacy ZMQ pose ABI:

```text
pelvis   -> tcp://127.0.0.1:5555
suitcase -> tcp://127.0.0.1:5561
payload  -> float32 [x, y, z, qw, qx, qy, qz]
```

Run the adapter on the same computer as the policy:

```bash
source /opt/ros/humble/setup.bash
source /home/irmv/catkin_vr/install/setup.bash
export PYTHONPATH=/opt/ros/humble/local/lib/python3.10/dist-packages:${PYTHONPATH:-}
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 \
.venv/bin/python scripts/ros2_pose_to_zmq.py
```

After calibration, pass the generated common world transform if you need the
ZMQ poses expressed in the calibration frame:

```bash
.venv/bin/python scripts/ros2_pose_to_zmq.py \
  --transform-json calibration/suitcase_pose.json
```

VRPN/NOKOV commonly publishes positions in millimetres. The adapter defaults
to `--position-scale 0.001`; use `--position-scale 1.0` only after confirming
the producer already publishes metres. ROS stores quaternions as `x,y,z,w`,
while this repository uses `w,x,y,z`; the adapter performs that conversion and
normalizes the quaternion.

Before connecting a policy, verify the source topics:

```bash
ros2 topic type /suitcase/pose
ros2 topic type /robot_g1/pose
ros2 topic hz /suitcase/pose
ros2 topic hz /robot_g1/pose
ros2 topic echo --once /suitcase/pose
ros2 topic echo --once /robot_g1/pose
```

Verify the conversion without ROS or hardware:

```bash
.venv/bin/python scripts/ros2_pose_to_zmq.py --self-test
```

For a bring-up watchdog, stop the adapter when either source is stale:

```bash
.venv/bin/python scripts/ros2_pose_to_zmq.py \
  --stale-timeout 0.25 --exit-on-stale
```

## Calibration

1. Keep both rigid bodies visible and stationary. Record 10--20 seconds of
   `/suitcase/pose` and `/robot_g1/pose`.
2. Confirm `frame_id`, units, quaternion convention, and update rate. A value
   around `200` for a room-scale coordinate is millimetres; it must become
   `0.200` metres before entering the policy.
3. Measure the G1 pelvis rigid-body marker offset. The policy's `pelvis` pose
   must represent the pelvis reference point, not the marker cluster origin.
   Apply a fixed rigid-body offset in the mocap model or adapter.
4. Place the suitcase in the training start relationship: its center is about
   `0.532 m` in front of the pelvis, with nearly zero lateral offset and about
   `2 degrees` yaw difference. The policy contact targets are at local offsets
   `(-0.10, +0.18, 0.25)` and `(-0.10, -0.18, 0.25)` metres.
5. Check the relative pose computed from ROS data, not just each absolute pose.
   Move the suitcase by 10 cm and confirm the reported relative position
   changes by 10 cm in the expected axis. Rotate it by 90 degrees and confirm
   the heading vector rotates accordingly.
6. Do not silently remap axes in the policy. If the mocap world axes differ,
   apply one documented rigid transform to both suitcase and pelvis, then
   repeat the relative-pose check.
7. Before motor commands, run the adapter and the ZMQ pose visualizer for at
   least one minute. The two streams should remain finite and below the stale
   timeout with no rejected messages.

The live calibration helper collects paired ROS samples while the robot and
suitcase are stationary:

```bash
.venv/bin/python scripts/calibrate_ros2_pose.py \
  --samples 120 \
  --relative-position 0.532 0.0 -0.793 \
  --relative-yaw-deg 0 \
  --output calibration/suitcase_pose.json
```

The default target means: pelvis is the calibration origin and the suitcase
root is 0.532 m forward, at zero lateral offset, with nearly parallel yaw.
Replace these values with measured target geometry when your setup differs.
The solver returns a common `world_from_mocap` matrix and residuals. A large
two-body relative residual is a calibration failure, not a matrix to accept.

For arbitrary marker placement, provide the fixed marker-frame-to-policy-frame
matrices in a JSON file:

```json
{
  "pelvis": [[1,0,0,0.02], [0,1,0,0], [0,0,1,0.10], [0,0,0,1]],
  "suitcase": [[1,0,0,-0.10], [0,1,0,0], [0,0,1,0.20], [0,0,0,1]]
}
```

These matrices use `T_marker_policy`: the policy-frame origin and axes expressed
in the marker frame. If you measure the marker origin in policy/body
coordinates (`T_policy_marker`), invert that matrix before putting it in the
file. The helper performs this inversion from measured metres and XYZ Euler
angles:

```bash
.venv/bin/python scripts/make_marker_to_policy.py \
  --pelvis-marker-position <px> <py> <pz> \
  --pelvis-marker-rpy-deg <roll> <pitch> <yaw> \
  --suitcase-marker-position <sx> <sy> <sz> \
  --suitcase-marker-rpy-deg <sroll> <spitch> <syaw> \
  --output calibration/marker_to_policy.json
```

Each position is the marker origin in the true body/policy frame, in metres.
Each RPY describes marker axes in that same body frame. If the suitcase policy
axes are aligned with the mocap world axes, use the current ROS marker
orientation as `T_policy_marker` without negating it. The helper inverts the
complete rotation when producing `T_marker_policy`; do not negate RPY here and
then invert again. Then run:

For a direct matrix entry (without the helper), use the inverse rotation:
`R_marker_policy = R_policy_marker.T` or the quaternion conjugate. Negating
Euler components individually is not generally equivalent because XYZ Euler
rotations reverse their multiplication order when inverted.

```bash
.venv/bin/python scripts/calibrate_ros2_pose.py \
  --marker-to-policy-json calibration/marker_to_policy.json \
  --output calibration/suitcase_pose.json
```

The generated file records both matrices and the solved common world transform.
The adapter applies `T_calibration_mocap * T_mocap_marker * T_marker_policy`.

The adapter only transports poses. G1 joint state still comes from
`scripts/g1/real_bridge.py` through Unitree DDS, and no pose stream can replace
that low-state input.

## Marker-only MuJoCo viewer

To validate marker transforms without Unitree state, use the real-time,
no-dynamics viewer:

```json
{
  "marker_from_torso": [[...], [...], [...], [0, 0, 0, 1]],
  "marker_from_suitcase": [[...], [...], [...], [0, 0, 0, 1]]
}
```

The matrices are `T_marker_torso` and `T_marker_suitcase`; each maps the policy
frame into its marker frame. Run:

```bash
MUJOCO_GL=egl .venv/bin/python scripts/render_live_marker_policy_mujoco.py \
  --torso-topic /robot_g1/pose \
  --suitcase-topic /suitcase/pose \
  --calibration calibration/marker_policy.json
```

The viewer freezes all actuated joints, applies the two marker transforms, and
calls `mj_forward` only. It is independent of `rt/lowstate`, waist FK, DDS, and
motor commands.

## Torso-derived pelvis

When a low pelvis rigid body is hard to track, use the torso marker and the
real-time waist angles from ZMQ low-state:

```bash
.venv/bin/python scripts/ros2_pose_to_zmq.py --suitcase-only
.venv/bin/python scripts/ros2_torso_to_pelvis.py \
  --torso-topic /robot_torso/pose \
  --torso-from-marker-json calibration/torso_from_marker.json \
  --stale-timeout 0.25 --exit-on-stale
```

The JSON normally stores `T_torso_marker`, the marker pose expressed in the
torso frame. The runtime also accepts `marker_from_torso`, which is often easier
to measure directly: the torso-frame origin expressed in marker coordinates.
If both are supplied, `marker_from_torso` is inverted internally.
The solver inverts it internally, reads `waist_yaw`, `waist_roll`, and
`waist_pitch` from low-state, computes the exact MJCF FK, and publishes pelvis
to port `5555`. The same `TorsoToPelvisFK` core is used for MuJoCo and real
inputs; do not replace the live waist angles with fixed defaults.

The torso frame origin is the center of the G1 waist-pitch joint axis, not the
visual chest center. In the MJCF at zero waist angles it is
`[-0.0039635, 0, 0.044] m` relative to pelvis. Locate the horizontal waist-pitch
axis on the real robot and use its axis center as the torso origin. The torso
mesh extends upward from this point, so its visual/geometric center is not a
valid policy frame.

## Suitcase hardware bring-up

Use the ordered launcher after the G1 interface, VRPN server, and calibration
have been verified:

```bash
# Bounded read-only check; never creates rt/lowcmd or MotionSwitcher.
bash scripts/run_suitcase_hardware.sh --check-only

# Interactive zero/hold/init controller. Requires typing ARM before mode release.
bash scripts/run_suitcase_hardware.sh --armed
```

Override the G1 interface when needed with `--interface <name>`. The launcher
starts VRPN first, waits for both tracker topics, starts the G1 bridge and pose
relays, then checks ports `5555`, `5561`, and `5590`. Armed startup keeps the
bridge disarmed until the local controller has produced a fresh timestamped
hold command.

The armed prompt exposes only:

- `z`: zero-policy/current-follow target with the configured HDMI PD gains;
- `h`: latch and hold the current joint positions;
- `i`: smoothly interpolate to suitcase motion frame 0 over 10 seconds;
- `q`: return to zero mode and stop the stack.

The base `--armed` launcher does not load the suitcase ONNX and has no policy
mode.

For the next validation stage, enable nominal shadow explicitly:

```bash
bash scripts/run_suitcase_hardware.sh --armed --nominal-shadow
```

Run `i`; the launcher waits for the 10-second initialization and reports when it
is complete. Then enter `s`.
Before `s`, place the suitcase at motion frame 0: the script checks the full
pelvis-yaw-frame XY offset (not only distance), approximately `-0.793 m`
relative height, and `1.95 deg` relative yaw. It fails closed when placement
exceeds its configured tolerances; the default pelvis-yaw-frame XY tolerance is
`0.20 m`.
The safe controller continues holding the initialized pose while the official
HDMI observation/history implementation runs the student ONNX for 472 steps at
50 Hz. The shadow process itself never binds or writes the low-command port. It records the
three ONNX inputs, nominal action and target, robot state, corrected poses,
inference timing, OOD ratios, joint-limit margins, and stream age under the
current `outputs/suitcase_hardware/<timestamp>/` directory. Review the generated
`nominal_shadow_*.summary.json` before adding a policy control mode.

## Guarded full-authority nominal run

The full-policy entry remains guarded even when nominal authority is 100%:

```bash
bash scripts/run_suitcase_hardware.sh --armed --nominal-apply
```

The HDMI process is proposal-only on local port `5594`; it cannot own or write
the low-command port `5591`. The command owner runs in policy-faithful direct
mode: each nominal `q_target` is copied unchanged to the G1 low command with the
upstream HDMI gains, zero velocity target, and zero feed-forward torque. There
is no authority blend, joint-position clipping, or target slew limit. This is
intentional because HDMI outputs virtual PD setpoints, including setpoints
beyond a mechanical joint range when it needs boundary torque. Non-finite or
stale proposals, stale low-state, invalid sequence order, or observed joint
speed above `12 rad/s` still switch control to a current-position hold. The
`180 deg` tilt setting disables the absolute-tilt abort.

Use this sequence:

1. Put the robot in debug mode, provide physical fall protection, keep the
   Unitree remote in the operator's hand, and clear people and obstacles from
   the motion envelope. Do not attach or lift the suitcase in this first pilot.
2. Start the command above and type `ARM` at the first confirmation.
3. Enter `i`. Watch the full 10-second move to motion frame 0. Enter `h` or use
   the remote emergency stop immediately if balance, feet, or joints look
   wrong. This pose differs from the earlier YAML-default init by as much as
   about `0.48 rad` on some joints.
4. Place the suitcase at the already validated frame-0 relationship. Enter
   `p`; the runner checks low-state, both corrected poses, the init error
   (`<=0.50 rad`), and object placement before offering the second gate.
5. Type `FULL` to enable full-authority closed-loop policy stabilization while
   the motion reference remains frozen at frame 0. Confirm that the robot is
   supporting itself, then fully release the safety ropes without resetting the
   policy. A marker occlusion during this stage does not immediately terminate
   the run: for up to 30 seconds the runner keeps the last corrected pose,
   continues policy inference and history updates against frozen motion frame 0,
   and prints `MARKER OCCLUDED`; `GO` is rejected while the corrected pelvis or
   suitcase pose is stale. When fresh poses return it prints `MARKER RESTORED`
   and continues with live pose input without resetting the policy.
6. Type `GO` only after the ropes are clear. This starts the advancing 472-step
   sequence, which lasts about 9.5 seconds. During either stabilization or
   motion, type `h` followed by Enter to stop policy control and hold. Use the
   physical remote rather than waiting for terminal input if motion is unstable.
   During motion, a pelvis-only marker dropout is tolerated for up to 2 seconds:
   inference and the reference continue using the last pelvis pose so the deep
   bend can pass through the occluded region. Fresh pelvis data is used
   automatically when it returns. Suitcase dropout is handled only after the
   reference contact flag is active and Vicon has measured at least 5 cm of
   real suitcase lift. At that point the runner anchors the last measured
   pelvis-to-suitcase transform and advances its relative motion from the
   reference while following the live pelvis globally. Prediction starts after
   60 ms without a new suitcase sample, before the general 250 ms stale gate,
   so the observation does not freeze for many policy frames and then jump. It reports
   `SUITCASE MARKER OCCLUDED AFTER CONFIRMED LIFT` and keeps inference running
   until Vicon returns or the motion ends. Reacquisition switches directly back
   to the corrected live pose and logs the position/orientation discrepancy.
   A suitcase dropout before measured lift confirmation still terminates the
   motion; this prevents a failed grasp from making a suitcase left on the
   floor follow the robot synthetically.
7. After normal completion the launcher sends `h` automatically. Enter `q`
   only after confirming the robot is stationary and supported.

The launcher fails closed if the policy exits or the safety controller reports
an abort. Raw policy records are written as `nominal_apply_*.npz` plus
`nominal_apply_*.summary.json`; the command owner's raw and actually applied
targets are in `nominal_pilot_applied.jsonl`, and process logs are in the same
`outputs/suitcase_hardware/<timestamp>/` directory. A successful process exit
only proves that the transport and anomaly gates ran; assess balance, contact,
target tracking, and the applied log before another run.
The inference NPZ includes a `phase` array (`stabilize`,
`stabilize_pose_grace`, `motion_pose_grace`, `motion_object_fallback`, or
`motion`), plus counters for each grace mode. `suitcase_pose_source` identifies
`live`, `held_last`, and `reference_attached` inputs; `suitcase_pose_measured`
retains the last raw corrected measurement, and reacquisition error fields make
the substitution auditable.
