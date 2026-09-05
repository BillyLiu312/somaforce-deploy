# SomaForce deployment

This repository is based on HDMI `sim2real` commit
`d9e1f700667bc75d8d2eeb5ef74bb2a066600612`. It retains the official G1 I/O,
ONNX/TensorRT inference boundary, motion backends, and MuJoCo sim2sim loop.

## Runtime topology

```text
HDMI student or Sonic nominal -> normalized a_nom[23]
    -> Cross F/T residual -> contact gain and authority
    -> safety/watchdog -> Unitree G1 or MuJoCo
```

The same `DeploymentStack` and action history are used by offline replay, MuJoCo,
and hardware. Only the RobotIO implementation changes.

## Modes

- `hdmi_student_baseline`: HDMI student only.
- `hdmi_student_residual`: main zero-shot force-adaptation path.
- `sonic_baseline`: Sonic without HDMI object-state input.
- `sonic_residual`: Sonic scaffold plus the same Cross residual.
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

Sonic reference conversion must pin 50 Hz timing, root-yaw alignment, future-step
semantics, and the 23-joint mapping. Shape equality alone is not evidence of
Sonic residual compatibility.

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
