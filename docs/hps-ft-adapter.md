# HPS wrist F/T adapter

The deployment adapter owns calibration, robot kinematics, frame transforms,
contact estimation, and Cross token construction. The sensor SDK owns only
lossless acquisition and transport of raw SI-unit measurements.

## Sensor transport contract

Each sensor exposes one reconnectable TCP stream. Every measurement is one
newline-terminated UTF-8 JSON object:

```json
{"schema":"hps6axis.wrench.v1","monotonic_ns":123456789,"sequence":42,"device_id":18174,"status":0,"fx":1.0,"fy":2.0,"fz":3.0,"mx":0.1,"my":0.2,"mz":0.3}
```

- `monotonic_ns` is sampled immediately after the complete sensor frame is
  decoded, using `CLOCK_MONOTONIC` on the SDK host.
- `sequence` starts at zero for a server process and increments exactly once
  for every decoded sensor frame.
- force is in N and moment is in Nm, in the native sensor frame.
- `device_id` is the ID returned by the sensor, not a USB device path. HPS
  sensors may report the same ID, so endpoint 9000/9001 remains the authority
  for left/right identity.
- `status=0` means valid. Exception, CRC-failed, partial, or timed-out frames are
  never published as valid measurements.
- One slow TCP client must not block acquisition or other clients. Dropping an
  old outbound sample is preferable to delaying acquisition.

The deploy client reconnects automatically. A sequence reset is accepted only
after a TCP reconnect. The old `timestamp_ms` message can be enabled with
`--allow-legacy-sensor-protocol` for bench tests, but it is not accepted for a
hardware residual pilot.

## Calibration contract

Copy `configs/ft/hps_g1.example.yaml`, fill both sensor IDs and measured values,
then set `valid: true`. `wrist_from_sensor` is `T_wrist_sensor`: it maps a point
in sensor coordinates into `left_wrist_yaw_link` or
`right_wrist_yaw_link`. Its translation is the sensor origin measured from the
wrist body origin, expressed in the wrist frame.

The adapter subtracts bias, applies `measurement_sign`, translates moment to
the wrist origin with `r x F`, rotates the result into the pelvis-yaw frame,
and normalizes force by 100 N and moment by 10 Nm. The measured G1 setup now
uses `0.115 kg` and `[0, 0, 0.076] m` in each sensor frame for the installed
hand mass and CoM. A residual hardware pilot requires validated gravity
compensation or evidence that mounted distal-load wrench has already been
removed upstream; a single-pose tare is not sufficient across arm orientations.

The emitted 14-D token order is:

```text
[Fx, Fy, Fz, Mx, My, Mz] in base-yaw, normalized
[wrist linear velocity xyz, wrist angular velocity xyz] in base-yaw
[contact_probability, quality]
```

`quality` is independently zeroed for a side when its device ID or status is
wrong, its sample is stale, it exceeds configured limits, or G1 state is stale.
The adapter continues publishing at 50 Hz in all these cases. This makes sensor
failure disable the residual through contact gating without pausing nominal
inference. The residual receiver also converts an adapter-process outage or a
frame older than 100 ms into the same zero-quality token instead of blocking.

## Per-run gravity-aware bias calibration

Keep SDK output raw and avoid writing a persistent hardware zero during normal
startup. After the safe controller has moved to a fixed calibration pose, keep
the configured hands installed, remove every other external wrist load, and
start the adapter with:

```bash
python scripts/run_hps_ft_adapter.py \
  --calibration calibration/hps_g1.yaml \
  --tare-on-start --tare-samples 200 \
  --tare-output outputs/hps_ft_runtime_bias.json
```

Without a trigger file, the adapter waits for the operator to press Enter. For
an orchestrated launch, pass `--tare-trigger-file <path>` and create that file
only after the robot is stationary in the calibration pose. The adapter accepts
only distinct sensor frames, checks both device IDs and status bytes, and
rejects the calibration when force standard deviation exceeds 1 N or moment
standard deviation exceeds 0.1 Nm by default. With gravity compensation enabled,
the adapter reads the current G1 wrist orientation, predicts the configured hand
gravity wrench in each sensor frame, and subtracts it from the stationary raw
mean. The result is an electronic bias rather than a pose-specific zero. The
process-local JSON uses schema `somaforce_ft_runtime_bias_v2` and records the
raw standard deviation, modeled gravity wrench, mass, CoM, and resulting bias.
It also stores the low-state tick, base quaternion, joint pose, and wrist
rotations used for the calculation.

One pose can estimate bias after mass and CoM are known, but cannot validate
those mass/CoM values. Validation still requires multiple substantially
different wrist orientations; the compensated no-contact wrench should remain
near zero in held-out poses.

The measured G1 calibration sets `require_runtime_tare: true`; the adapter
refuses to run from that file unless `--tare-on-start` is present.

## Run

Start the two raw SDK servers and the existing G1 low-state bridge. The SDK
launcher arguments are the left ttyUSB number, right ttyUSB number, and optional
bind address, in that order. It maps the left sensor to port 9000 and the right
sensor to port 9001:

```bash
cd /home/irmv/Workspace/Somaforce/HPS_6axis_SDK
./scripts/start_dual_servers.sh 1 0 0.0.0.0

cd /home/irmv/Workspace/Somaforce/somaforce-deploy
python scripts/run_hps_ft_adapter.py \
  --calibration calibration/hps_g1.yaml \
  --validate-only

python scripts/run_hps_ft_adapter.py \
  --calibration calibration/hps_g1.yaml \
  --left-port 9000 --right-port 9001 \
  --tare-on-start --tare-output outputs/hps_ft_runtime_bias.json

python scripts/check_hps_ft_stream.py \
  --duration 60 --require-both-valid \
  --output outputs/hps_ft_acceptance.npz
```

The general SDK launcher form is:

```text
start_dual_servers.sh <left-ttyUSB-number> <right-ttyUSB-number> [bind-address]
```

The adapter consumes low-state on port 5590, optionally consumes corrected
pelvis pose on port 5555 for base translation velocity, and publishes
`ResidualFTFrame` on port 5580. Use `--no-pelvis` when validating F/T without a
mocap stream; wrist twist then excludes base translation.

`ResidualFTFrame` v2 carries the adapter's realtime and monotonic publish times
plus, for each wrist, the raw sensor-frame wrench, device/status, SDK source
time, raw sequence, and the deploy host's monotonic TCP receive time. It also
records the G1 kinematics time used for the frame transform. Old frames remain
decodable. Freshness and policy
alignment use `CLOCK_MONOTONIC` on the deploy host; realtime is retained only to
correlate independent logs. An SDK source timestamp is directly comparable only
when the SDK and deploy adapter run in the same clock domain.
At each 50 Hz policy tick the receiver selects the newest frame no older than
100 ms; otherwise it supplies a zero-quality token. It does not silently
interpolate wrench samples. The record includes sample age and signed
sample-to-kinematics skew so trials can impose a tighter offline synchronization
cut without guessing from row indices.

## Suitcase policy integration and recording

Start both SDK servers first, then launch the complete hardware stack. `--ft`
defaults the Cross residual to `shadow`, so the residual is evaluated and
recorded but the robot still receives the nominal action:

```bash
bash scripts/run_suitcase_hardware.sh \
  --armed --nominal-shadow --ft \
  --record-dir /data/somaforce/suitcase_trials
```

After entering `ARM`, use `i` to reach motion frame 0, keep both configured hands
installed, remove every other external wrist load, and use `t`; confirm with
`TARE`. The launcher waits for both sensors
to pass the rate, age, and quality check before `s` or `p` may start. Use
`--residual-mode off` to record F/T without evaluating Cross. `c1` and `c2`
apply their bounded residual and therefore require `--nominal-apply`; enable
them only after reviewing shadow records. Keep the per-run gravity-aware bias
calibration enabled unless a separately validated persistent electronic bias is
written into the calibration file.

Each policy attempt writes one compressed NPZ under
`<record-dir>/<run-id>/`. Every row is one policy tick and includes both clocks,
low-state tick and stream ages, joint position/velocity/torque, IMU attitude and
angular velocity, live pelvis and suitcase poses, full reference joint/body
state, nominal and applied targets, all nominal observations and OOD ratios,
F/T token/wrench/timestamps/sequences, Cross inputs and histories, residual
safety stages, and loop/inference timing. Metadata records joint/body order,
policy hashes, the full F/T calibration plus its hash, normalization, history
order, and the clock contract.
If a watchdog or operator terminates an attempt, the completed rows are still
written. During execution the runner first commits atomic chunks under
`<record>.recording/manifest.json`; a sibling `.partial.json` marks an
incomplete final record. The launcher can rebuild the NPZ from those chunks even
when the process is stopped during final compression. Normal completion first
confirms controller hold, then finalizes the NPZ, preventing proposal watchdog
timeouts from corrupting the archive.

## Calibration procedure before residual authority

1. Warm both sensors to thermal steady state and make their left/right device
   mapping persistent. The installed pair currently shares one `device_id`, so
   udev paths and the 9000/9001 mapping are part of the calibration.
2. With no external load, exercise all six positive and negative axes using a
   known force/lever arm. Confirm N/Nm units, sign, cross-axis coupling, status,
   saturation, and the configured `measurement_sign`.
3. Measure `T_wrist_sensor` for each side: a proper rotation plus the sensor
   origin relative to `wrist_yaw_link`. Verify it by applying a known force at a
   known point and checking both transformed force and the `r x F` moment.
4. Estimate electronic bias after warm-up. For shadow trials, repeat the
   stationary frame-0 gravity-aware bias calibration every run and retain its
   JSON record.
5. Before `c1/c2`, collect no-external-contact data in at least six
   well-separated arm orientations. Verify, and refine if necessary, the
   measured `0.115 kg` mass and `[0, 0, 0.076] m` sensor-frame CoM. Held-out
   orientations must have low residual force and moment; a one-pose result
   cannot pass this requirement.
6. Determine noise, drift, overload, contact thresholds, and the 100 ms freshness
   limit from repeated static/no-contact and known-contact trials. Keep the
   100 N/10 Nm normalization fixed unless the Cross training contract changes.
7. Run at least one minute of dual-stream acceptance, then nominal + Cross
   shadow trials. Check axis/sign plots, F/T age and quality coverage, contact
   timing, target tracking, measured torque, loop overruns, and residual safety
   clipping before granting any residual authority.

The launcher enforces this boundary. The current measured file declares
`validation_scope: runtime_tare_and_residual_shadow`, so it rejects `c1/c2`.
After completing the multi-pose validation, set the scope to
`residual_authority`. Both sides already have deploy-side gravity compensation
enabled; keep `upstream_distal_load_compensated: false`. The per-run
gravity-aware bias calibration remains compatible with this compensation and
should stay enabled to track thermal/electronic bias drift.

## Acceptance boundary

The SDK delivery is accepted when:

1. Both ports run for 1 minute with no malformed JSON, duplicate/inverted
   sequence, non-finite values, or acquisition stall.
2. Unplugging either USB adapter does not stop the other stream; reconnecting it
   resumes with a new TCP connection without restarting the deploy adapter.
3. At the intended sensor rate, receive interval p99 is below two sensor
   periods and no valid frame is older than 100 ms at the deploy input.
4. Six known signed axis loads produce the correct raw axis and sign. Force and
   moment units agree with N and Nm before deploy-side transforms.
5. `status != 0`, CRC failure, timeout, and overload are observable and never
   silently emitted as `status=0`.

Because both installed sensors currently report `device_id=18174`, swapping the
two ttyUSB arguments cannot be detected from payload data. Preserve the launcher
mapping (`ttyUSB1 -> left/9000`, `ttyUSB0 -> right/9001`) and add persistent udev
symlinks before residual authority is enabled.

The deploy adapter is accepted when unit tests pass, a stale or disconnected
side reaches `quality=0` within 100 ms while port 5580 remains at 50 Hz, known
loads reproduce expected base-yaw force and `r x F` moment, and an F/T outage
does not create a nominal proposal gap.
