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

## Redundant rigid-body markers

The runtime can derive one torso or suitcase pose from multiple independently
tracked rigid bodies. Each source needs its own fixed `T_marker_target` because
NOKOV/VRPN assigns a different origin and orientation to every rigid body. Do
not reuse one matrix for all three markers unless their frames are physically
identical.

Measure each marker pose in its target frame as `T_target_marker`: position in
metres followed by target-frame XYZ RPY in degrees. Generate a v2 calibration
by repeating each source argument:

```bash
.venv/bin/python scripts/make_marker_to_policy.py \
  --torso-source robot1 /robot1/pose <x> <y> <z> <roll> <pitch> <yaw> \
  --torso-source robot2 /robot2/pose <x> <y> <z> <roll> <pitch> <yaw> \
  --torso-source robot3 /robot3/pose <x> <y> <z> <roll> <pitch> <yaw> \
  --suitcase-source suitcase1 /suitcase1/pose <x> <y> <z> <roll> <pitch> <yaw> \
  --suitcase-source suitcase2 /suitcase2/pose <x> <y> <z> <roll> <pitch> <yaw> \
  --suitcase-source suitcase3 /suitcase3/pose <x> <y> <z> <roll> <pitch> <yaw> \
  --output calibration/marker_policy_v2.json
```

The file stores each topic and its `marker_from_target` matrix under
`marker_sources.torso` or `marker_sources.suitcase`. At runtime every fresh
marker is independently converted with
`T_world_target = T_world_mocap * T_mocap_marker * T_marker_target`.

If one marker on each object already has a trusted transform in the old
calibration, the remaining matrices can be solved from simultaneous stationary
samples instead of being measured manually:

```bash
.venv/bin/python scripts/calibrate_redundant_markers.py \
  --anchor-calibration calibration/marker_policy.json \
  --torso-anchor robot1 --suitcase-anchor suitcase1 \
  --torso-source robot1 /robot1/pose \
  --torso-source robot2 /robot2/pose \
  --torso-source robot3 /robot3/pose \
  --suitcase-source suitcase1 /suitcase1/pose \
  --suitcase-source suitcase2 /suitcase2/pose \
  --suitcase-source suitcase3 /suitcase3/pose \
  --samples 240 --output calibration/marker_policy_v2.json
```

All three rigid bodies on each object must be visible during calibration. The
tool records residuals and marks the output invalid if the fixed relative-pose
assumption is not supported. The anchor transform still determines the target
frame, so choose an anchor whose old transform has already been validated.

One visible source is sufficient. All sources in the selected consistent set
are fused using a position mean and an SO(3) rotation mean; when all three are
valid, the output is the three-marker mean. With three fresh sources, a single
positional or angular outlier is excluded. If fresh sources conflict and no
consensus can be established, no new pose is published and the existing stale
watchdog fails closed. Sources also have to fall within the default 50 ms
synchronization window to participate in the same fusion update, so an occluded
marker stops influencing output before its full stale timeout.

When the selected source set changes, the runtime preserves the previous output
on the switch frame and decays the old-to-new local SE(3) alignment over 0.25 s.
The new mean continues following live motion during that interval, avoiding a
position or orientation step without applying a permanent low-pass lag. Set
`--marker-source-switch-blend-s` to change the transition duration. The default
consensus limits are 8 cm and 12 degrees; change them only
from measured residuals using `--marker-position-consensus-m` and
`--marker-orientation-consensus-deg`.

The suitcase hardware launcher uses `suitcase4` as its preferred source. While
that source remains inside the 50 ms synchronization window, its calibrated
pose is used directly. When it is unavailable, the remaining suitcase sources
fall back to the same consensus-and-mean algorithm, with a hardware-specific
position limit of 12 cm and the unchanged 12 degree orientation limit. Pelvis
fusion retains the default 8 cm position limit and has no preferred source.

For a remounted suitcase marker set, keep a trusted `suitcase1` transform as the
anchor and solve `suitcase2`/`suitcase3` from synchronized stationary samples:

```bash
ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=1 \
.venv/bin/python scripts/calibrate_redundant_markers.py \
  --role suitcase \
  --anchor-calibration calibration/marker_policy.json \
  --suitcase-anchor suitcase1 \
  --suitcase-source suitcase1 /suitcase1/pose \
  --suitcase-source suitcase2 /suitcase2/pose \
  --suitcase-source suitcase3 /suitcase3/pose \
  --samples 240 --output /tmp/suitcase_marker_calibration.json
```

An added extension marker should be solved independently against each existing
trusted anchor. Accept it only when every fit passes and the independently
anchored `marker_from_target` estimates agree. The deployed `suitcase4` entry
uses the position mean and SO(3) mean of solutions anchored by suitcase1, 2,
and 3; its recorded cross-validation limits are 30 mm and 5 degrees. Because
the fusion source list is data-driven, adding `/suitcase4/pose` to
`marker_sources.suitcase` automatically includes it in averaging, consensus,
outlier rejection, and source-switch blending after the relay restarts.

Pass the same v2 file to both relays:

```bash
.venv/bin/python scripts/ros2_pose_to_zmq.py \
  --suitcase-only --transform-json calibration/marker_policy_v2.json
.venv/bin/python scripts/ros2_torso_to_pelvis.py \
  --torso-from-marker-json calibration/marker_policy_v2.json
```

The hardware launcher reads the source topics from this file and waits for any
one torso source and any one suitcase source. Existing v1 single-marker files
remain supported. Select the v2 file during bring-up with
`bash scripts/run_suitcase_hardware.sh --calibration calibration/marker_policy_v2.json --check-only`.

### Fast robot marker-set calibration

The hardware launcher currently does not load the separate
`marker_frame_corrections.json` layer. Robot marker changes are calibrated
directly into `marker_sources.torso` in the main calibration file, avoiding a
second runtime transform whose reference may be unstable.

The normal workflow is to let the controller establish the robot side of the
frame-0 relationship, then use the measured suitcase pose as the field anchor:

```bash
bash scripts/run_suitcase_hardware.sh --armed
# type ARM, then i; after init completes, place the suitcase at frame 0 and type c
```

While G1 remains in the frame-0 init hold, `c` reads the actual fused suitcase
pose and all three raw robot marker poses. The calibrator obtains the full
suitcase-to-`torso_link` SE(3) reference from frame 0 of
`assets/mujoco/reference/hdmi_suitcase/motion.npz` and computes:

```text
T_marker_torso = inverse(T_world_marker_measured)
    * T_world_suitcase_measured
    * T_suitcase_torso_frame0
```

This uses the actual suitcase location in the mocap world; it does not align the
robot markers to a hard-coded world pose. `robot1`, `robot2`, and `robot3` must
all remain visible and each must provide the full sample count; one valid
suitcase marker is sufficient for the suitcase anchor. After a successful update the launcher
exits so the next run reloads the new matrices. The standalone
`--quick-calibrate-robot-markers` mode remains available when G1 is already
held externally in the exact frame-0 pose; that mode is VRPN-only and never
creates a G1 bridge or low-command publisher.

Each robot marker-to-torso transform is solved independently from that marker's
measured pose using the equation above. No old inter-marker relationship or
missing-marker inference is used. If any one of the three robot markers is
missing or unstable, the complete calibration is rejected without modifying
the file. A successful run updates `calibration/marker_policy.json` atomically
and first creates a timestamped
`marker_policy.json.before-robot-marker-<UTC>` backup. Run `--check-only` after
calibration and inspect multi-marker agreement before arming.

During an armed interactive startup, a missing fused pelvis does not terminate
the launcher because a bad robot-marker calibration must remain recoverable.
The launcher keeps the robot in safe hold and exposes `i` followed by `c` for
frame-0 recalibration. This exception applies only to the initial recovery
check: `--check-only`, the `p` preflight, and the runtime watchdog still require
a live pelvis stream before policy control can run.

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

# VRPN-only frame-0 robot marker calibration; updates the selected calibration.
bash scripts/run_suitcase_hardware.sh --quick-calibrate-robot-markers

# Interactive zero/hold/init controller. Requires typing ARM before mode release.
bash scripts/run_suitcase_hardware.sh --armed
```

Override the G1 interface when needed with `--interface <name>`. The launcher
starts VRPN first, waits for both tracker topics, starts the G1 bridge and pose
relays, then checks ports `5555`, `5561`, and `5590`. Armed startup keeps the
bridge disarmed until the local controller has produced a fresh timestamped
hold command. The normal read-only and armed paths use only the transforms in
the selected calibration file; the separate marker correction file is muted.
In an armed interactive session, pose relays remain alive across a complete
marker dropout but publish no stale pose. They resume only after a valid live
consensus returns. Every `s` and `p` command runs a fresh
pelvis/suitcase/low-state preflight, and each policy runner independently
enforces the 250 ms age limit, so keeping the relay process alive does not
weaken fail-closed policy behavior.

The armed prompt exposes only:

- `z`: zero-policy/current-follow target with the configured HDMI PD gains;
- `h`: latch and hold the current joint positions;
- `i`: smoothly interpolate to suitcase motion frame 0 over 10 seconds;
- `c`: use the measured suitcase pose to calibrate robot markers, then exit;
- `q`: return to zero mode and stop the stack.

The base `--armed` launcher does not load the suitcase ONNX and has no policy
mode.

For the next validation stage, enable nominal shadow explicitly:

```bash
bash scripts/run_suitcase_hardware.sh --armed --nominal-shadow
```

Run `i`; the launcher waits for the 10-second initialization and reports when it
is complete. Then enter `s`. Before loading ONNX, `s` runs the same one-second
pelvis/suitcase/low-state preflight used by the apply path.
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
speed above `18 rad/s` still switch control to a current-position hold. The
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
   `p`; the launcher first runs a one-second `pelvis`/`suitcase`/`low_state`
   stream preflight, then the runner checks freshness, init error
   (`<=0.50 rad`), and object placement before offering the second gate. A
   rejection prints the specific missing or stale stream, pose age, or measured
   error and limit, plus the full log path.
5. Type `FULL` to enable full-authority closed-loop policy stabilization while
   the motion reference remains frozen at frame 0. Confirm that the robot is
   supporting itself, then fully release the safety ropes without resetting the
   policy. Redundant markers are the only occlusion mechanism: losing one source
   is acceptable while another calibrated source maintains a valid fused pose.
   If all sources for either object are stale, or fresh sources have no valid
   consensus, the relay stops publishing. After the configured stale timeout the
   proposal process exits and the command controller switches to hold. The runner
   never continues inference with a held, pelvis-attached, or reference-predicted
   object pose.
6. Type `GO` only after the ropes are clear. This starts the advancing 472-step
   sequence, which lasts about 9.5 seconds. During either stabilization or
   motion, type `h` followed by Enter to stop policy control and hold. Use the
   physical remote rather than waiting for terminal input if motion is unstable.
   The same live-only multi-marker rule applies during motion; there is no pelvis
   grace period and no suitcase attachment fallback.
7. After normal completion the launcher sends `h` automatically. Enter `q`
   only after confirming the robot is stationary and supported.

The launcher fails closed if the policy exits or the safety controller reports
an abort. Raw policy records are written as `nominal_apply_*.npz` plus
`nominal_apply_*.summary.json`; the command owner's raw and actually applied
targets are in `nominal_pilot_applied.jsonl`, and process logs are in the same
`outputs/suitcase_hardware/<timestamp>/` directory. A successful process exit
only proves that the transport and anomaly gates ran; assess balance, contact,
target tracking, and the applied log before another run.
Apply rows are committed incrementally under the adjacent `*.recording/`
directory. If the policy process is interrupted, the launcher rebuilds a
readable partial NPZ from committed chunks instead of exposing a half-written
ZIP archive.
The inference NPZ records only live fused `pelvis_pose` and `suitcase_pose`.
Its summary declares `pose_source=multi_marker_live_only`; the phase is limited
to `stabilize`, `motion`, or `frozen`.

For synchronized wrist F/T input and a paper-oriented record, add `--ft` and
`--record-dir <local-directory>`. `--ft` starts the HPS adapter and defaults to
Cross residual shadow. After `i`, keep the configured hands installed, remove
other wrist loads, hold still, and enter `t`, then confirm `TARE`; F/T preflight
must pass before `s` or `p`. Each attempt stores a
policy-tick-aligned NPZ under `<local-directory>/<run-id>/`. Use
`--residual-mode off` for logging only, or explicitly select `c1`/`c2` with
`--nominal-apply` only after shadow review. Full calibration and field details
are documented in [HPS wrist F/T adapter](./hps-ft-adapter.md).
