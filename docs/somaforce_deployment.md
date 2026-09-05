# SomaForce deployment

This repository is based on HDMI `sim2real` commit `d9e1f700667bc75d8d2eeb5ef74bb2a066600612`. It retains the official G1 I/O, ONNX/TensorRT, motion backends, and MuJoCo sim2sim loop, and adds two nominal backends:

```text
HDMI student -> a_nom[23]
Sonic         -> a_nom[23]
```

Both share the Cross residual, F/T token pipeline, safety composition, replay, and Unitree G1 I/O.

## Modes

- `hdmi_student_baseline`: student only.
- `hdmi_student_residual`: main SomaForce path.
- `sonic_baseline`: Sonic without object observation.
- `sonic_residual`: cross-scaffold transfer path.
- `*_shadow`: compute residual, do not apply it.

The HDMI student export must implement `adapt_ema(policy[249], object[10]) -> priv_pred[256]` followed by `actor_adapt(command[356], policy[249], priv_pred[256]) -> action[23]`. It should be exported as a plain-tensor deterministic-mean ONNX graph with no HDMI/Isaac/TorchRL runtime dependency.

The residual export accepts `wrist_tokens[1,2,16,14]`, `proprio[1,64]`, `a_nom_history[1,3,23]`, and `previous_a_total[1,23]`, returning normalized `delta_a[1,23]`. F/T calibration, frame transforms, contact gating, authority ramping, and one-time physical action scaling stay outside the graph.

Sonic uses the canonical HDMI/Cross reference after conversion. The converter must pin 50 Hz timing, root-yaw alignment, future-step semantics, joint mapping, and whether Sonic emits normalized deltas or already-scaled targets.

## Validation

Use the same policy wrapper and action composer for MuJoCo and hardware; only `RobotIO` changes. The order is `offline replay -> MuJoCo sim2sim -> nominal hardware -> residual shadow -> C0 -> low-authority pilot`. MuJoCo validates export parity, history/reset, joint mapping, timing, F/T tokens, and safety behavior; it does not replace Isaac C1/C2/C3 interaction evaluation.
