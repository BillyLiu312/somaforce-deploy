# somaforce-deploy

Deployment runtime for the four HDMI student policies and the Sonic nominal
backend with a shared SomaForce Cross force-adaptation residual. The runtime is
based on the pinned HDMI sim2real stack and keeps G1 Unitree I/O, ONNX/TensorRT,
motion streams, and MuJoCo sim2sim.

## Install

```bash
uv sync --extra inference-cpu
# G1 hardware: uv sync --extra inference-cpu --extra robot-g1
```

## Modes

Use `configs/profiles/mujoco.yaml` for sim2sim, `shadow.yaml` for HDMI student
residual shadowing, `sonic_shadow.yaml` for Sonic shadowing, and `pilot.yaml`
only after the hardware safety gates are complete.

Models, references, F/T calibration, and sensor SDKs are external artifacts;
do not commit them. See [deployment contract](docs/somaforce_deployment.md).

The upstream HDMI sim2real documentation remains under `docs/` and the
original `sim2real/` package.
