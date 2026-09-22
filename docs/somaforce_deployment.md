# SomaForce deployment

This repository is based on HDMI `sim2real` commit
`d9e1f700667bc75d8d2eeb5ef74bb2a066600612`. It retains the official G1 I/O,
ONNX/TensorRT inference boundary, motion backends, and MuJoCo sim2sim loop.

## Runtime topology

```text
HDMI student -> normalized a_nom[23]; Sonic G1 nominal -> q_target[29]
    -> Cross F/T residual -> contact gain and authority
    -> safety/watchdog -> Unitree G1 or MuJoCo
```

The same `DeploymentStack` and action history are used by offline replay, MuJoCo,
and hardware. Only the RobotIO implementation changes.

## Modes

- `hdmi_student_baseline`: HDMI student only.
- `hdmi_student_residual`: main zero-shot force-adaptation path.
- `sonic_baseline`: Sonic without HDMI object-state input.
- `sonic_residual`: reserved contract; the current Sonic G1 export emits 29
  joint targets and is therefore nominal-only until a 23-action adapter is
  explicitly validated.
- `hdmi_student_residual_shadow` and `sonic_residual_shadow`: compute residual,
  send nominal action only.

## Model bindings

The native HDMI student can be exported as one deterministic action graph or as
two graphs:

```text
adapt_ema(policy[249], command[356], object[10]) -> priv_pred[256]
actor_adapt(command[356], policy[249], priv_pred[256]) -> action[23]
```

`HDMIStudentTwoStageNominal` enforces this shape contract. The residual export
takes `wrist_tokens[1,2,16,14]`, `proprio[1,64]`,
`a_nom_history[1,23,3]`, and `previous_a_total[1,23]`, returning normalized
`delta_a[1,23]`. F/T calibration, frame transforms, contact gating, authority
ramping, and physical action scaling remain outside ONNX.

Sonic reference conversion must pin 50 Hz timing, root-yaw alignment, and
future-step semantics. The push-door hardware wrapper is SONIC-only: it
converts the native HDMI body/joint motion into an any4hdmi qpos tree and
publishes guarded 29-joint G1 proposals. Both wrist F/T streams are required
for every hardware mode and recorded for synchronized audit, but do not enter
the SONIC ONNX. Shape
equality alone is not evidence of Sonic residual compatibility.

## Push-door-hand hardware entry point

The physical deployment script is `scripts/run_push_door_hand_hardware.sh`.
It follows the suitcase script's F/T, G1 bridge, hold/init, shadow, and
guarded-pilot protocol, but starts no ROS/VRPN service or marker relay. SONIC
uses G1 proprioception and the 573-frame HDMI reference; the F/T adapter uses
low-state kinematics with `--no-pelvis`. Policy/F-T records use atomic 25-frame
chunks. Normal exceptions and termination signals produce a partial NPZ
automatically; after `SIGKILL` or power loss, run
`scripts/finalize_chunked_record.py --output <record>.npz`. G1/F-T read-only
preflight is hardware-verified; shadow/apply remain separate acceptance stages.

The default pilot keeps the safe controller's joint-limit clipping and
`0.08 rad/tick` target slew limit enabled. `--direct-policy-targets` remains as
a commented launcher option for later restoration. Before publication, raw
SONIC targets fail closed on joint-limit overshoot greater than `0.05 rad` or a
single-tick jump greater than `0.50 rad`. A controller `pilot_abort:*` stops the
runner immediately. Runtime F/T invalidity is warning-only and is preserved as
quality-zero data. The original HDMI NPZ is never modified: the generated
deployment cache defaults to 2x time interpolation and a nine-frame
Savitzky-Golay smoothing window.

## Artifact verification

Keep checkpoints, private F/T calibration, and sensor SDKs outside Git. Start from
`configs/tasks/manifest.example.json`, replace every placeholder SHA-256, then run:

```bash
python scripts/verify_artifacts.py artifacts/<task>/manifest.json
```

Missing required files, path traversal, and hash mismatches fail closed.

## Progression

Use this order:

```text
offline replay -> MuJoCo sim2sim -> nominal hardware
    -> residual shadow -> C0 parity -> low-authority pilot
```

MuJoCo validates export parity, history/reset behavior, joint mapping, timing,
synthetic F/T token handling, and safety behavior. It does not replace Isaac
C1/C2/C3 interaction evaluation.
