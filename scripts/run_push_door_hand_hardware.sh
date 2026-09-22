#!/usr/bin/env bash
# Run the SONIC policy with the native HDMI push_door_hand reference on a G1.
#
# This intentionally mirrors run_suitcase_hardware.sh's G1/F-T operator
# protocol, but SONIC uses proprioception only and starts no ROS/VRPN services.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INVOCATION_DIR="$PWD"
ROBOT_INTERFACE="enx9c69d373f0ff"
G1_ADDRESS="${G1_ADDRESS:-192.168.123.161}"
ARMED=0
CHECK_ONLY=0
NOMINAL_SHADOW=0
NOMINAL_APPLY=0
FT_ENABLED=0
FT_CALIBRATION="$REPO_ROOT/calibration/hps_g1.yaml"
FT_LEFT_HOST="127.0.0.1"
FT_LEFT_PORT=9000
FT_RIGHT_HOST="127.0.0.1"
FT_RIGHT_PORT=9001
FT_SKIP_TARE=0
RECORD_DIR=""
NOMINAL_BACKEND=sonic
SONIC_POLICY_CONFIG="$REPO_ROOT/checkpoints/sonic/release/g1/policy.yaml"
SONIC_MODEL="$REPO_ROOT/checkpoints/sonic/release/g1/policy.onnx"
SONIC_REFERENCE_MANIFEST="$REPO_ROOT/checkpoints/sonic/release/g1/manifest.json"
SONIC_MOTION_ROOT="$REPO_ROOT/outputs/sonic_reference/hdmi_push_door_hand"
REFERENCE_TIME_SCALE="${REFERENCE_TIME_SCALE:-2.0}"
REFERENCE_SMOOTH_WINDOW="${REFERENCE_SMOOTH_WINDOW:-9}"
MAX_PROPOSAL_STEP="${MAX_PROPOSAL_STEP:-0.50}"
PROPOSAL_LIMIT_TOLERANCE="${PROPOSAL_LIMIT_TOLERANCE:-0.05}"

MOTION="$REPO_ROOT/assets/mujoco/reference/hdmi_push_door_hand/motion.npz"
MOTION_META="$REPO_ROOT/assets/mujoco/reference/hdmi_push_door_hand/meta.json"
DOOR_STEPS=0

usage() {
  cat <<EOF
Usage: $0 [--interface NAME] [--armed]
  [--param sonic] [--nominal-backend sonic]
  [--nominal-shadow|--nominal-apply]
  [--sonic-policy-config FILE] [--sonic-model FILE] [--sonic-motion-root DIR]
  [--sonic-reference-manifest FILE] [--ft] [--ft-calibration FILE]
  [--reference-time-scale SCALE] [--reference-smooth-window ODD]
  [--max-proposal-step RAD]
  [--proposal-limit-tolerance RAD]
  [--ft-left-host HOST] [--ft-left-port PORT] [--ft-right-host HOST]
  [--ft-right-port PORT] [--ft-skip-tare]
  [--record-dir DIR] [--check-only]

All hardware modes require --ft. Check-only validates raw SDK reachability and
the calibration contract; armed shadow/apply uses the interactive init -> tare flow.
--nominal-shadow audits the time-scaled policy while the robot remains in hold;
--nominal-apply enables the guarded proposal pilot.
EOF
}

while (($#)); do
  case "$1" in
    --interface) ROBOT_INTERFACE="$2"; shift 2;;
    --armed) ARMED=1; shift;;
    --check-only) CHECK_ONLY=1; shift;;
    --nominal-shadow) NOMINAL_SHADOW=1; shift;;
    --nominal-apply) NOMINAL_APPLY=1; shift;;
    --nominal-backend) NOMINAL_BACKEND="$2"; shift 2;;
    --param) NOMINAL_BACKEND="$2"; shift 2;;
    --sonic-policy-config) SONIC_POLICY_CONFIG="$2"; shift 2;;
    --sonic-model) SONIC_MODEL="$2"; shift 2;;
    --sonic-motion-root) SONIC_MOTION_ROOT="$2"; shift 2;;
    --sonic-reference-manifest) SONIC_REFERENCE_MANIFEST="$2"; shift 2;;
    --reference-time-scale) REFERENCE_TIME_SCALE="$2"; shift 2;;
    --reference-smooth-window) REFERENCE_SMOOTH_WINDOW="$2"; shift 2;;
    --max-proposal-step) MAX_PROPOSAL_STEP="$2"; shift 2;;
    --proposal-limit-tolerance) PROPOSAL_LIMIT_TOLERANCE="$2"; shift 2;;
    --ft) FT_ENABLED=1; shift;;
    --ft-calibration) FT_CALIBRATION="$2"; shift 2;;
    --ft-left-host) FT_LEFT_HOST="$2"; shift 2;;
    --ft-left-port) FT_LEFT_PORT="$2"; shift 2;;
    --ft-right-host) FT_RIGHT_HOST="$2"; shift 2;;
    --ft-right-port) FT_RIGHT_PORT="$2"; shift 2;;
    --ft-skip-tare) FT_SKIP_TARE=1; shift;;
    --record-dir) RECORD_DIR="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done

[[ "$NOMINAL_BACKEND" == sonic ]] || { echo "this launcher is SONIC-only; use --param sonic" >&2; exit 2; }
if ((NOMINAL_SHADOW && NOMINAL_APPLY)); then echo "choose one nominal mode" >&2; exit 2; fi
if ((NOMINAL_SHADOW || NOMINAL_APPLY)) && ((!ARMED)); then
  echo "nominal mode requires --armed" >&2; exit 2
fi
if [[ "$NOMINAL_BACKEND" == sonic ]] && ((!FT_ENABLED)); then
  echo "SONIC hardware runtime requires --ft for synchronized wrist F/T recording" >&2; exit 2
fi
if ((FT_SKIP_TARE && !FT_ENABLED)); then echo "--ft-skip-tare requires --ft" >&2; exit 2; fi
if ((!ARMED && !CHECK_ONLY)); then
  echo "SONIC runtime requires --check-only or --armed; live calibrated F/T needs armed init/tare" >&2; exit 2
fi

cd "$REPO_ROOT"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

REQUIRED_FILES=("$MOTION" "$MOTION_META")
REQUIRED_FILES+=("$SONIC_POLICY_CONFIG" "$SONIC_MODEL" "$SONIC_REFERENCE_MANIFEST")
if ((FT_ENABLED)); then REQUIRED_FILES+=("$FT_CALIBRATION"); fi
for required in "${REQUIRED_FILES[@]}"; do
  [[ -f "$required" ]] || { echo "Missing required file: $required" >&2; exit 1; }
done
if [[ "$NOMINAL_BACKEND" == sonic ]]; then
  # SONIC consumes an any4hdmi qpos tree. This is the deliberate reference
  # conversion from the native HDMI motion; it does not alter the HDMI path.
  .venv/bin/python scripts/convert_hdmi_motion_to_any4hdmi.py \
    --motion "$MOTION" --motion-meta "$MOTION_META" \
    --reference-manifest "$SONIC_REFERENCE_MANIFEST" --out-dir "$SONIC_MOTION_ROOT" \
    --name push_door_hand.npz --time-scale "$REFERENCE_TIME_SCALE" \
    --smooth-window "$REFERENCE_SMOOTH_WINDOW"
  DOOR_STEPS="$(.venv/bin/python - "$SONIC_MOTION_ROOT/motions/push_door_hand.npz" <<'PY'
import sys
import numpy as np
with np.load(sys.argv[1], allow_pickle=False) as motion:
    print(int(motion["qpos"].shape[0]))
PY
)"
fi

ip link show "$ROBOT_INTERFACE" >/dev/null 2>&1 || { echo "Robot interface missing: $ROBOT_INTERFACE" >&2; exit 1; }
[[ "$(cat "/sys/class/net/$ROBOT_INTERFACE/carrier" 2>/dev/null || true)" == 1 ]] || { echo "Robot interface has no carrier" >&2; exit 1; }
ping -I "$ROBOT_INTERFACE" -c 1 -W 1 "$G1_ADDRESS" >/dev/null || { echo "G1 unreachable: $G1_ADDRESS" >&2; exit 1; }
nc -z -w 2 "$FT_LEFT_HOST" "$FT_LEFT_PORT" || { echo "left F/T stream unreachable" >&2; exit 1; }
nc -z -w 2 "$FT_RIGHT_HOST" "$FT_RIGHT_PORT" || { echo "right F/T stream unreachable" >&2; exit 1; }

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/outputs/push_door_hand_hardware/$RUN_ID"
if [[ -n "$RECORD_DIR" ]]; then
  [[ "$RECORD_DIR" = /* ]] || RECORD_DIR="$INVOCATION_DIR/$RECORD_DIR"
  RECORD_OUTPUT_DIR="$(realpath -m "$RECORD_DIR")/$RUN_ID"
else RECORD_OUTPUT_DIR="$LOG_DIR"; fi
RUNTIME_DIR="$(mktemp -d /tmp/push_door_hand_hardware.XXXXXX)"
mkdir -p "$LOG_DIR" "$RECORD_OUTPUT_DIR"
PIDS=()
signal_group() { kill -TERM -- "-$1" 2>/dev/null || kill -TERM "$1" 2>/dev/null || true; }
report_log_failure() {
  local label="$1" log_file="$2"
  echo "$label; tail of $log_file:" >&2
  tail -n 40 "$log_file" >&2
}
require_ft_preflight() {
  local label="$1" log_file="$2"
  if ! .venv/bin/python scripts/check_hps_ft_stream.py --duration 1 --require-both-valid \
    >"$log_file" 2>&1; then
    report_log_failure "$label" "$log_file"
    return 1
  fi
  cat "$log_file"
}
cleanup() {
  trap - EXIT INT TERM
  if ((ARMED)) && [[ -p "$RUNTIME_DIR/commands" ]]; then
    echo h >&3 2>/dev/null || true
    sleep 0.1
  fi
  for pid in "${PIDS[@]}"; do signal_group "$pid"; done
  for _ in $(seq 1 100); do
    local_alive=0
    for pid in "${PIDS[@]}"; do kill -0 "$pid" 2>/dev/null && local_alive=1; done
    ((local_alive == 0)) && break
    sleep 0.1
  done
  for pid in "${PIDS[@]}"; do kill -KILL "$pid" 2>/dev/null || true; done
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  rm -rf "$RUNTIME_DIR"
}
trap cleanup EXIT INT TERM

if ((ARMED)); then
  read -r -p "Type ARM to permit MotionSwitcher release and G1 commands: " confirmation
  [[ "$confirmation" == ARM ]] || { echo "Arm confirmation rejected"; exit 1; }
  mkfifo "$RUNTIME_DIR/commands"; exec 3<>"$RUNTIME_DIR/commands"
  # HARDWARE-ONLY DIFFERENCE: real_bridge is the physical G1 RobotIO backend.
  setsid .venv/bin/python scripts/g1/real_bridge.py --robot g1 --interface "$ROBOT_INTERFACE" \
    --wait-for-command --ready-file "$RUNTIME_DIR/bridge.ready" >"$LOG_DIR/g1_bridge.log" 2>&1 & PIDS+=("$!")
  CONTROLLER_ARGS=(--policy-config "$SONIC_POLICY_CONFIG" --motion "$MOTION" --motion-meta "$MOTION_META" \
    --pilot-authority 1.0 --max-target-step 0.08 --pilot-init-tolerance 0.50 --max-tilt-deg 180 \
    --max-joint-speed 18 --command-fifo "$RUNTIME_DIR/commands" --ready-file "$RUNTIME_DIR/controller.ready" \
    --status-file "$RUNTIME_DIR/controller.status")
  # Restore this only after filtered hardware acceptance:
  # CONTROLLER_ARGS+=(--direct-policy-targets)
  if ((NOMINAL_APPLY)); then CONTROLLER_ARGS+=(--proposal-port 5594 --pilot-log "$RECORD_OUTPUT_DIR/controller_applied.jsonl"); fi
  setsid .venv/bin/python scripts/g1/suitcase_safe_controller.py "${CONTROLLER_ARGS[@]}" \
    >"$LOG_DIR/safe_controller.log" 2>&1 & CONTROLLER_PID="$!"; PIDS+=("$CONTROLLER_PID")
else
  setsid .venv/bin/python scripts/g1/real_bridge.py --robot g1 --interface "$ROBOT_INTERFACE" --read-only \
    >"$LOG_DIR/g1_bridge.log" 2>&1 & PIDS+=("$!")
fi

if ((FT_ENABLED)); then
  FT_ARGS=(--calibration "$FT_CALIBRATION" --left-host "$FT_LEFT_HOST" --left-port "$FT_LEFT_PORT" \
    --right-host "$FT_RIGHT_HOST" --right-port "$FT_RIGHT_PORT" --output-port 5580 --no-pelvis)
  if ((!FT_SKIP_TARE)); then FT_ARGS+=(--tare-on-start --tare-trigger-file "$RUNTIME_DIR/ft_tare.trigger" \
    --tare-timeout 3600 --tare-output "$RECORD_OUTPUT_DIR/ft_runtime_bias.json"); fi
  # HARDWARE-ONLY DIFFERENCE: SDK F/T sockets are physical wrist sensors.
  setsid .venv/bin/python scripts/run_hps_ft_adapter.py "${FT_ARGS[@]}" >"$LOG_DIR/ft_adapter.log" 2>&1 &
  FT_ADAPTER_PID="$!"; PIDS+=("$FT_ADAPTER_PID")
fi

if ((ARMED)); then
  for _ in $(seq 1 150); do [[ -f "$RUNTIME_DIR/controller.ready" ]] && break; sleep 0.1; done
  [[ -f "$RUNTIME_DIR/controller.ready" ]] || { echo "safe controller did not become ready" >&2; exit 1; }
  for _ in $(seq 1 300); do [[ -f "$RUNTIME_DIR/bridge.ready" ]] && break; sleep 0.1; done
  [[ -f "$RUNTIME_DIR/bridge.ready" ]] || { echo "G1 bridge did not become ready" >&2; exit 1; }
fi
sleep 2
.venv/bin/python scripts/check_suitcase_streams.py --duration 3 --low-state-only \
  | tee "$LOG_DIR/stream_check.log"
if ((CHECK_ONLY)); then
  .venv/bin/python scripts/run_hps_ft_adapter.py --calibration "$FT_CALIBRATION" --validate-only
  exit 0
fi

SONIC_FT_ARGS=()
if ((FT_ENABLED)); then SONIC_FT_ARGS+=(--ft-port 5580 --ft-max-age-ms 100 --ft-calibration "$FT_CALIBRATION" --require-both-ft-valid); fi

run_nominal_shadow() {
  local stem="$1"
  touch "$RUNTIME_DIR/proposal.start" "$RUNTIME_DIR/motion.start" "$RUNTIME_DIR/proposal.complete.ack"
  .venv/bin/python scripts/run_sonic_nominal_proposal.py \
    --policy-config "$SONIC_POLICY_CONFIG" --model "$SONIC_MODEL" \
    --motion-root "$SONIC_MOTION_ROOT" --source-motion "$MOTION" --source-motion-meta "$MOTION_META" \
    --steps "$DOOR_STEPS" --rate 50 --ort-num-threads 2 --chunk-size 25 \
    --reference-time-scale "$REFERENCE_TIME_SCALE" --max-proposal-step "$MAX_PROPOSAL_STEP" \
    --proposal-limit-tolerance "$PROPOSAL_LIMIT_TOLERANCE" \
    --proposal-port 5594 --ready-file "$RUNTIME_DIR/proposal.ready" \
    --start-file "$RUNTIME_DIR/proposal.start" --motion-start-file "$RUNTIME_DIR/motion.start" \
    --completion-file "$RUNTIME_DIR/proposal.complete" \
    --completion-ack-file "$RUNTIME_DIR/proposal.complete.ack" --output "$stem.npz" "${SONIC_FT_ARGS[@]}"
}

run_nominal_apply() {
  local stem="$1"
  .venv/bin/python scripts/run_sonic_nominal_proposal.py \
    --policy-config "$SONIC_POLICY_CONFIG" --model "$SONIC_MODEL" \
    --motion-root "$SONIC_MOTION_ROOT" --source-motion "$MOTION" --source-motion-meta "$MOTION_META" \
    --steps "$DOOR_STEPS" --rate 50 --ort-num-threads 2 --chunk-size 25 \
    --reference-time-scale "$REFERENCE_TIME_SCALE" --max-proposal-step "$MAX_PROPOSAL_STEP" \
    --proposal-limit-tolerance "$PROPOSAL_LIMIT_TOLERANCE" \
    --proposal-port 5594 --ready-file "$RUNTIME_DIR/proposal.ready" \
    --start-file "$RUNTIME_DIR/proposal.start" --motion-start-file "$RUNTIME_DIR/motion.start" \
    --completion-file "$RUNTIME_DIR/proposal.complete" \
    --completion-ack-file "$RUNTIME_DIR/proposal.complete.ack" --output "$stem.npz" "${SONIC_FT_ARGS[@]}"
}
echo "Door safe controller active. Commands: z=zero, h=hold, i=init, t=tare, s=shadow, p=pilot, q=quit"
while kill -0 "$CONTROLLER_PID" 2>/dev/null; do
  read -r -p "push-door-safe> " command || command=q
  case "$command" in
    z|zero|h|hold|i|init|q|quit|exit) echo "$command" >&3;;
    t|tare)
      ((FT_ENABLED)) || { echo "restart with --ft to tare"; continue; }
      ((FT_SKIP_TARE)) && { echo "--ft-skip-tare is active"; continue; }
      [[ "$(cat "$RUNTIME_DIR/controller.status" 2>/dev/null || true)" == init_complete ]] || { echo "F/T tare requires completed init; press i first"; continue; }
      echo "Remove all external wrist loads, keep both arms still, and type TARE."
      read -r -p "F/T tare confirmation: " tare_confirmation
      [[ "$tare_confirmation" == TARE ]] || { echo "F/T tare cancelled"; continue; }
      touch "$RUNTIME_DIR/ft_tare.trigger"
      for _ in $(seq 1 400); do
        [[ -f "$RECORD_OUTPUT_DIR/ft_runtime_bias.json" ]] && break
        kill -0 "$FT_ADAPTER_PID" 2>/dev/null || break
        sleep 0.05
      done
      if require_ft_preflight "F/T tare or stream validation failed" "$LOG_DIR/ft_bias_check.log"; then
        echo "F/T gravity-aware bias calibration accepted"
      fi;;
    s|shadow)
      ((NOMINAL_SHADOW)) || { echo "restart with --armed --nominal-shadow"; continue; }
      [[ "$(cat "$RUNTIME_DIR/controller.status" 2>/dev/null || true)" == init_complete ]] || { echo "SONIC shadow requires completed init; press i first"; continue; }
      require_ft_preflight "SONIC shadow rejected: F/T stream check failed" "$LOG_DIR/s_ft_preflight_$(date +%H%M%S).log" || continue
      STEM="$RECORD_OUTPUT_DIR/policy_shadow_$(date +%H%M%S)"
      if ! run_nominal_shadow "$STEM" >"$STEM.log" 2>&1; then
        tail -n 40 "$STEM.log" >&2
      fi
      if [[ ! -f "$STEM.npz" && -f "$STEM.recording/manifest.json" ]]; then
        .venv/bin/python scripts/finalize_chunked_record.py --output "$STEM.npz" \
          --reason runner_exit >"$STEM.finalize.log" 2>&1 || \
          report_log_failure "SONIC shadow chunk finalization failed" "$STEM.finalize.log"
      fi;;
    p|pilot)
      ((NOMINAL_APPLY)) || { echo "restart with --armed --nominal-apply"; continue; }
      [[ "$(cat "$RUNTIME_DIR/controller.status" 2>/dev/null || true)" == init_complete ]] || { echo "press i first"; continue; }
      require_ft_preflight "SONIC pilot rejected: F/T stream check failed" "$LOG_DIR/p_ft_preflight_$(date +%H%M%S).log" || continue
      STEM="$RECORD_OUTPUT_DIR/policy_apply_$(date +%H%M%S)"; rm -f "$RUNTIME_DIR"/{proposal.ready,proposal.start,motion.start,pose.status,proposal.complete,proposal.complete.ack}
      (run_nominal_apply "$STEM") >"$STEM.log" 2>&1 & APPLY_PID="$!"; PIDS+=("$APPLY_PID")
      for _ in $(seq 1 300); do [[ -f "$RUNTIME_DIR/proposal.ready" ]] && break; sleep 0.1; done
      [[ -f "$RUNTIME_DIR/proposal.ready" ]] || { echo "door proposal failed" >&2; tail -n 40 "$STEM.log" >&2; continue; }
      echo p >&3; touch "$RUNTIME_DIR/proposal.start"
      for _ in $(seq 1 40); do
        [[ "$(cat "$RUNTIME_DIR/controller.status" 2>/dev/null || true)" == pilot_active ]] && break
        sleep 0.05
      done
      if [[ "$(cat "$RUNTIME_DIR/controller.status" 2>/dev/null || true)" != pilot_active ]]; then
        echo "Safe controller did not enter SONIC stabilization; returning to hold" >&2
        signal_group "$APPLY_PID"; wait "$APPLY_PID" 2>/dev/null || true; echo h >&3
        continue
      fi
      echo "SONIC frame-0 stabilization active. Type GO only after the robot is steady."
      echo "Type GO to start the $DOOR_STEPS-step door motion, or h/q to stop."
      go=""
      PILOT_ABORTED=0
      while kill -0 "$APPLY_PID" 2>/dev/null; do
        PILOT_STATUS="$(cat "$RUNTIME_DIR/controller.status" 2>/dev/null || true)"
        if [[ "$PILOT_STATUS" == pilot_abort:* ]]; then
          echo "Safe controller aborted SONIC stabilization: $PILOT_STATUS" >&2
          signal_group "$APPLY_PID"
          PILOT_ABORTED=1
          break
        fi
        if read -r -t 0.1 go; then break; fi
      done
      if ((PILOT_ABORTED)); then
        wait "$APPLY_PID" 2>/dev/null || true
        echo h >&3
        continue
      elif [[ "$go" == GO ]]; then
        touch "$RUNTIME_DIR/motion.start"
      else
        echo h >&3
        signal_group "$APPLY_PID"
        wait "$APPLY_PID" 2>/dev/null || true
        echo "GO cancelled; controller returned to hold"
        continue
      fi
      stop=""
      PILOT_QUIT=0
      while kill -0 "$APPLY_PID" 2>/dev/null; do
        [[ -f "$RUNTIME_DIR/proposal.complete" ]] && { echo h >&3; touch "$RUNTIME_DIR/proposal.complete.ack"; break; }
        PILOT_STATUS="$(cat "$RUNTIME_DIR/controller.status" 2>/dev/null || true)"
        if [[ "$PILOT_STATUS" == pilot_abort:* ]]; then
          echo "Safe controller aborted SONIC motion: $PILOT_STATUS" >&2
          signal_group "$APPLY_PID"
          stop="$PILOT_STATUS"
          break
        fi
        if read -r -t 0.1 stop; then
          case "$stop" in
            h|hold) echo h >&3; signal_group "$APPLY_PID"; break;;
            q|quit|exit) echo h >&3; signal_group "$APPLY_PID"; PILOT_QUIT=1; break;;
            *) echo "During SONIC motion, allowed commands: h q";;
          esac
        fi
      done
      wait "$APPLY_PID" 2>/dev/null || true; echo h >&3
      if [[ ! -f "$STEM.npz" && -f "$STEM.recording/manifest.json" ]]; then
        .venv/bin/python scripts/finalize_chunked_record.py --output "$STEM.npz" \
          --reason "${stop:-runner_exit}" >"$STEM.finalize.log" 2>&1 || \
          report_log_failure "SONIC chunk finalization failed" "$STEM.finalize.log"
      fi
      if ((PILOT_QUIT)); then
        echo q >&3
        wait "$CONTROLLER_PID" 2>/dev/null || true
        break
      fi;;
    *) echo "Allowed commands: z h i t s p q";;
  esac
  [[ "$command" == i || "$command" == init ]] && { for _ in $(seq 1 150); do [[ "$(cat "$RUNTIME_DIR/controller.status" 2>/dev/null || true)" == init_complete ]] && break; sleep 0.1; done; }
  [[ "$command" == q || "$command" == quit || "$command" == exit ]] && { wait "$CONTROLLER_PID" 2>/dev/null || true; break; }
done
