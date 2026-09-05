# somaforce-deploy

Deployment runtime for HDMI student and Sonic nominal policies with a shared
SomaForce Cross force-adaptation residual. The repository is based on the pinned
HDMI/sim2real runtime and keeps the same policy wrapper, action history, MuJoCo
sim2sim backend, and Unitree G1 I/O boundary.

## Runtime modes

- HDMI student baseline: HDMI student only.
- HDMI student residual: the main zero-shot force-adaptation path.
- Sonic baseline: Sonic nominal policy without HDMI object-state input.
- Sonic residual: Sonic scaffold plus the same Cross residual.
- Shadow suffix: compute the residual but send nominal action only.

The shared DeploymentStack is used for offline replay, MuJoCo, and hardware. Only
the RobotIO backend changes. Start with offline replay, then MuJoCo sim2sim,
nominal hardware, residual shadow, C0 parity, and finally a low-authority pilot.

## Install

    uv sync --extra inference-cpu
    # G1 hardware:
    uv sync --extra inference-cpu --extra robot-g1

Set HF_HUB_OFFLINE=1 and HF_HUB_DISABLE_TELEMETRY=1 on the robot.

## Artifact contract

Keep checkpoints, private F/T calibration, and sensor SDKs outside Git. Each task
bundle must contain manifest.json, nominal and residual ONNX exports, frozen
normalization, and reference data. Verify a bundle before starting:

    python -c "from somaforce_deploy.artifacts import ArtifactManifest; ArtifactManifest.load('artifacts/<task>/manifest.json').verify()"

The HDMI student export may be one deterministic action graph or the two-stage
adapt_ema(policy, command, object) -> priv_pred and actor_adapt(command, policy, priv_pred)
binding exposed by HDMIStudentTwoStageNominal. Sonic reference conversion must pin
50 Hz timing, root-yaw alignment, future-step semantics, and joint mapping.

## Validation

    python scripts/validate_somaforce_contract.py
    pytest -q tests/test_somaforce_contracts.py tests/test_nominal_interfaces.py
    python -m compileall -q somaforce_deploy

Models and private calibration are intentionally not included in this source
repository. See docs/somaforce_deployment.md.
