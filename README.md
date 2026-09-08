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

## HDMI + Cross MuJoCo smoke

The local `artifacts/hdmi_push_box/` directory contains the two ONNX graphs;
model binaries are ignored by Git. Run the deploy-side composition smoke, then the headless MuJoCo physics
smoke (the latter uses explicit zero synthetic F/T tokens until task sensors are
bound):

    python scripts/validate_onnx_bundle.py --student artifacts/hdmi_push_box/student.onnx --residual artifacts/hdmi_push_box/cross_residual.onnx --authority 0.0 --contact-gain 0.0 --shadow
    python scripts/mujoco_bundle_smoke.py --student artifacts/hdmi_push_box/student.onnx --residual artifacts/hdmi_push_box/cross_residual.onnx --control-steps 25 --authority 0.1 --contact-gain 0.2

For the real local G1+box contact evaluation, use the copied HDMI MJCF scene and
reference motion:

    python scripts/evaluate_push_box_mujoco.py --student artifacts/hdmi_push_box/student.onnx --residual artifacts/hdmi_push_box/cross_residual.onnx --output outputs/mujoco_eval/push_box_movable_full.npz --authority 0.05 --contact-gain 1.0

The evaluator records `[T,2,6]` wrist-local wrenches and `[T,2,16,14]` Cross
tokens. It keeps the root free and restricts collision geometry to wrist-to-box
pairs; this is an explicit sensor-mapping evaluation configuration, not a
full-body locomotion acceptance run.

Models and private calibration are intentionally not included in this source
repository. See docs/somaforce_deployment.md.
## Free-root MuJoCo evaluation and rendering

Run the combined HDMI student + Cross residual with an unconstrained G1 floating base. The evaluator stops on a non-finite state or pelvis height below the configured failure threshold and records full `qpos/qvel` alongside contact wrench tokens:

    python scripts/evaluate_push_box_mujoco.py --student artifacts/hdmi_push_box/student.onnx --residual artifacts/hdmi_push_box/cross_residual.onnx --output outputs/mujoco_eval/push_box_free_root_terminated.npz --authority 0.05 --contact-gain 1.0 --stop-on-instability --root-height-failure 0.45

Render the recorded trajectory headlessly. The renderer selects EGL automatically when `DISPLAY` is absent:

    python scripts/render_push_box_mujoco.py --record outputs/mujoco_eval/push_box_free_root_terminated.npz --output outputs/mujoco_eval/push_box_free_root_terminated.mp4

For a full-model contact check (without wrist-to-box geometry filtering), add `--all-contact-geometry`.
### Free-root comparison

A Cross `scaffold-only` MuJoCo adapter is available as a simulation-only privileged-teacher baseline. Root anchoring is intentionally unsupported because it bypasses the stability question. Current free-root evidence shows scaffold, student nominal, and student plus Cross residual falling within roughly 40-55 control steps, so physical calibration and asset/runtime alignment remain unresolved.

The default residual artifact is the task-onehot checkpoint export
`cross_residual.onnx` from `segment_0023` (`iteration=733`, `6,004,736`
transitions, stage `C2`). The former segment-0079 graph is retained as
`cross_residual_segment0079.onnx` for provenance only.

## HDMI nominal baseline status

The current baseline target is the HDMI student alone; Cross residual is not
part of this claim. HDMI-native Isaac headless rollout completed a 792-step
push-box episode with `success=1.0` using the student finetune resume
checkpoint. The corresponding MuJoCo free-root run still terminates on the
pelvis-height guard after 57 control steps using HDMI's `mujoco_physics_dt`
reference of 0.002 s and decimation 10. This is an asset/dynamics alignment
failure, not evidence that the distilled student is invalid.

## Official HDMI tag runtime

The headless harnesses load the policy and MuJoCo modules from a separate
checkout of the official HDMI tag. Upstream source is not vendored here. The
harnesses also override the tag's default low-command port mismatch:
`CommandSender` used `55901`, while the MuJoCo bridge listens on `5591`; both
processes use `5591` locally. These runs remain sim2sim checks and are not
hardware acceptance claims.

The default HDMI-tag harness task is the upstream suitcase setup. The upstream
scene uses `SIMULATE_DT=0.005`, but the locally trained suitcase checkpoint
records `mujoco_physics_dt=0.002`. The checkpoint-aligned, one-motion-cycle
headless reproduction is:

Large model, motion, and mesh files are intentionally ignored. Prepare them
from the local HDMI export and a checkout of the upstream `hdmi` tag:

```bash
git clone --depth 1 --branch hdmi https://github.com/EGalahad/sim2real.git ../sim2real-hdmi-upstream
mkdir -p artifacts/hdmi_move_suitcase/hdmi_tag assets/mujoco/reference/hdmi_suitcase
cp ../HDMI/scripts/exports/G1TrackSuitcase/policy-cbvbj5hd-final.onnx artifacts/hdmi_move_suitcase/hdmi_tag/student.onnx
cp ../HDMI/scripts/exports/G1TrackSuitcase/policy-cbvbj5hd-final.yaml artifacts/hdmi_move_suitcase/hdmi_tag/policy.yaml
cp ../HDMI/scripts/exports/G1TrackSuitcase/policy-cbvbj5hd-final.json artifacts/hdmi_move_suitcase/hdmi_tag/policy.json
cp ../HDMI/data/motion/g1/omomo/sub1_suitcase_011/motion.npz assets/mujoco/reference/hdmi_suitcase/motion.npz
cp ../HDMI/data/motion/g1/omomo/sub1_suitcase_011/meta.json assets/mujoco/reference/hdmi_suitcase/meta.json
```

The expected student ONNX SHA256 is
`1f847c8b648f09d1046518c05264b8a5c591aca1ba020554404b71bf919787ad`.

```bash
# terminal 1
python scripts/run_hdmi_tag_headless_sim.py \
  --seconds 12 --sim-dt 0.002 --initialize-motion-frame \
  --elastic-band-release-after 0 \
  --trajectory outputs/hdmi_tag_suitcase/trajectory.npz

# terminal 2, started within two seconds of terminal 1
python scripts/run_hdmi_tag_headless_policy.py --steps 472

MUJOCO_GL=egl python scripts/render_push_box_mujoco.py --scene suitcase \
  --scene-path ../sim2real-hdmi-upstream/data/robots/g1/g1_29dof_rubberhand-suitcase.xml \
  --record outputs/hdmi_tag_suitcase/trajectory.npz \
  --output outputs/hdmi_tag_suitcase/trajectory.mp4 --fps 500
```

`--initialize-motion-frame` initializes both the robot and suitcase from the
reference first frame. `--elastic-band-release-after 0` models pressing `9`
immediately after policy start and clears the last applied gantry wrench. The
unmodified tag `0.005` timing falls in the current local comparison, while the
checkpoint-aligned `0.002` run completes the first suitcase move cycle. Neither
result is hardware acceptance, and a band-enabled run must not be reported as
unassisted locomotion stability.

### Multi-task HDMI student sim2sim

The same harness supports the locally trained `push_door_hand`, `push_box`, and
`move_largebox` students. Prepare their exports and motion files from `HDMI`:

```bash
mkdir -p artifacts/hdmi_push_door_hand/hdmi_tag assets/mujoco/reference/hdmi_push_door_hand
cp ../HDMI/scripts/exports/G1PushDoorHand/policy-4ta6gpm0-final.{onnx,yaml,json} artifacts/hdmi_push_door_hand/hdmi_tag/
cp ../HDMI/data/motion/data_for_sim/push_door-hand-0828/{motion.npz,meta.json} assets/mujoco/reference/hdmi_push_door_hand/
mkdir -p artifacts/hdmi_push_box/hdmi_tag assets/mujoco/reference/push_box
cp ../HDMI/scripts/exports/G1PushBox/policy-3i8rdxsd-final.{onnx,yaml,json} artifacts/hdmi_push_box/hdmi_tag/
cp ../HDMI/data/motion/g1/push_box/push_box-VID_20250423_220958-light-high-adjust_root_height/{motion.npz,meta.json} assets/mujoco/reference/push_box/
mkdir -p artifacts/hdmi_move_largebox/hdmi_tag assets/mujoco/reference/hdmi_move_largebox
cp ../HDMI/scripts/exports/G1MoveLargeboxOmni/policy-cnrls2ul-final.{onnx,yaml,json} artifacts/hdmi_move_largebox/hdmi_tag/
cp ../HDMI/data/motion/g1/omomo/sub10_largebox_014/{motion.npz,meta.json} assets/mujoco/reference/hdmi_move_largebox/
```

Run one cycle by selecting `--task push_door_hand`, `--task push_box`, or
`--task move_largebox` on both headless commands. On 2026-09-08, door completed
its task motion and one push-box run moved the box about `2.00 m`; push-box has
not yet been shown repeatable across launches. Large-box exported and ran end to
end but still lost pelvis height after about three seconds, so it is not accepted.

Render the two accepted demo trajectories with the task-aware renderer:

```bash
MUJOCO_GL=egl python scripts/render_hdmi_sim2sim.py --task push_door_hand \
  --record outputs/hdmi_multitask_sim2sim_20260908/push_door_hand_v3_rubberhand_v2/trajectory.npz \
  --output outputs/hdmi_multitask_sim2sim_20260908/push_door_hand_v3_rubberhand_v2/trajectory_behind_robot.mp4 \
  --fps 500 --camera-azimuth 270 --camera-elevation -12 --camera-distance 3.2 --follow-robot

MUJOCO_GL=egl python scripts/render_hdmi_sim2sim.py --task move_suitcase \
  --record outputs/hdmi_tag_sim2sim_suitcase_repro_20260908_v8_checkpoint_aligned/trajectory.npz \
  --output outputs/hdmi_tag_sim2sim_suitcase_repro_20260908_v8_checkpoint_aligned/trajectory_hdmi_render.mp4 \
  --fps 500
```

### Cross residual sim2sim

The residual loop has a separate lockstep diagnostic. Start the simulator with
`--publish-residual-ft --lockstep-port 5581`, then start policy with
`--residual-mode shadow` or `--residual-mode c1`, `--lockstep-port 5581`, and
`--residual-record <path>`. Shadow computes the residual but applies nominal;
C1 applies the frozen per-joint authority, contact ramp, and safety limiters.
The paired evidence is recorded in
`outputs/hdmi_residual_sim2sim_20260908/summary.json`. Both tasks retain their
motion in the C1 runs; this verifies the inference/composition/F-T loop but does
not establish statistical benefit or hardware readiness.
The suitcase sample had slightly lower force metrics with C1; the door sample
retained the task but had higher force metrics.
