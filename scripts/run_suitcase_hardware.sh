#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INVOCATION_DIR="$PWD"
CATKIN_VR_ROOT="${CATKIN_VR_ROOT:-/home/irmv/catkin_vr}"
ROBOT_INTERFACE="enx9c69d373f0ff"
G1_ADDRESS="${G1_ADDRESS:-192.168.123.161}"
VRPN_ADDRESS="${VRPN_ADDRESS:-192.168.5.58}"
VRPN_PORT="${VRPN_PORT:-3883}"
ARMED=0
CHECK_ONLY=0
NOMINAL_SHADOW=0
NOMINAL_APPLY=0
CALIBRATION="$REPO_ROOT/calibration/marker_policy.json"
QUICK_CALIBRATE_ROBOT_MARKERS=0
FT_ENABLED=0
FT_CALIBRATION="$REPO_ROOT/calibration/hps_g1.yaml"
FT_LEFT_HOST="127.0.0.1"
FT_LEFT_PORT=9000
FT_RIGHT_HOST="127.0.0.1"
FT_RIGHT_PORT=9001
FT_SKIP_TARE=0
RESIDUAL_MODE="off"
RESIDUAL_MODE_SET=0
RESIDUAL_MODEL="$REPO_ROOT/artifacts/hdmi_push_box/cross_residual.onnx"
RECORD_DIR=""

usage() {
  echo "Usage: $0 [--interface NAME] [--calibration FILE] [--quick-calibrate-robot-markers] [--armed] [--nominal-shadow|--nominal-apply] [--ft] [--ft-calibration FILE] [--ft-left-host HOST] [--ft-left-port PORT] [--ft-right-host HOST] [--ft-right-port PORT] [--ft-skip-tare] [--residual-mode off|shadow|c1|c2] [--residual-model FILE] [--record-dir DIR] [--check-only]"
}

wait_for_any_topic() {
  local role="$1"
  shift
  local deadline=$((SECONDS + 20))
  while ((SECONDS < deadline)); do
    local available
    available="$(ros2 topic list 2>/dev/null || true)"
    local topic
    for topic in "$@"; do
      if grep -Fxq "$topic" <<<"$available" && timeout 5 ros2 topic echo \
        --qos-reliability best_effort --once "$topic" >/dev/null; then
        echo "$role marker ready: $topic"
        return 0
      fi
    done
    sleep 0.2
  done
  echo "Timed out waiting for any $role marker topic: $*" >&2
  return 1
}

while (($#)); do
  case "$1" in
    --interface)
      ROBOT_INTERFACE="$2"
      shift 2
      ;;
    --calibration)
      CALIBRATION="$2"
      shift 2
      ;;
    --quick-calibrate-robot-markers)
      QUICK_CALIBRATE_ROBOT_MARKERS=1
      shift
      ;;
    --ft)
      FT_ENABLED=1
      shift
      ;;
    --ft-calibration)
      FT_CALIBRATION="$2"
      shift 2
      ;;
    --ft-left-host)
      FT_LEFT_HOST="$2"
      shift 2
      ;;
    --ft-left-port)
      FT_LEFT_PORT="$2"
      shift 2
      ;;
    --ft-right-host)
      FT_RIGHT_HOST="$2"
      shift 2
      ;;
    --ft-right-port)
      FT_RIGHT_PORT="$2"
      shift 2
      ;;
    --ft-skip-tare)
      FT_SKIP_TARE=1
      shift
      ;;
    --residual-mode)
      RESIDUAL_MODE="$2"
      RESIDUAL_MODE_SET=1
      shift 2
      ;;
    --residual-model)
      RESIDUAL_MODEL="$2"
      shift 2
      ;;
    --record-dir)
      RECORD_DIR="$2"
      shift 2
      ;;
    --armed)
      ARMED=1
      shift
      ;;
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    --nominal-shadow)
      NOMINAL_SHADOW=1
      shift
      ;;
    --nominal-apply)
      NOMINAL_APPLY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done
if ((NOMINAL_SHADOW && !ARMED)); then
  echo "--nominal-shadow requires --armed" >&2
  exit 2
fi
if ((NOMINAL_APPLY && !ARMED)); then
  echo "--nominal-apply requires --armed" >&2
  exit 2
fi
if ((NOMINAL_SHADOW && NOMINAL_APPLY)); then
  echo "Choose only one of --nominal-shadow or --nominal-apply" >&2
  exit 2
fi
if ((QUICK_CALIBRATE_ROBOT_MARKERS && (ARMED || NOMINAL_SHADOW || NOMINAL_APPLY || CHECK_ONLY))); then
  echo "--quick-calibrate-robot-markers must be used by itself" >&2
  exit 2
fi
case "$RESIDUAL_MODE" in
  off|shadow|c1|c2) ;;
  *)
    echo "--residual-mode must be off, shadow, c1, or c2" >&2
    exit 2
    ;;
esac
if ((FT_ENABLED && !RESIDUAL_MODE_SET && (NOMINAL_SHADOW || NOMINAL_APPLY))); then
  RESIDUAL_MODE="shadow"
fi
if [[ "$RESIDUAL_MODE" != "off" ]] && ((!FT_ENABLED)); then
  echo "--residual-mode $RESIDUAL_MODE requires --ft" >&2
  exit 2
fi
if [[ "$RESIDUAL_MODE" == "c1" || "$RESIDUAL_MODE" == "c2" ]] && ((!NOMINAL_APPLY)); then
  echo "--residual-mode $RESIDUAL_MODE requires --armed --nominal-apply" >&2
  exit 2
fi
if ((FT_SKIP_TARE && !FT_ENABLED)); then
  echo "--ft-skip-tare requires --ft" >&2
  exit 2
fi
if ((FT_ENABLED && !ARMED && !FT_SKIP_TARE && !CHECK_ONLY)); then
  echo "--ft without --armed requires --ft-skip-tare; interactive tare is only available in armed mode" >&2
  exit 2
fi

cd "$REPO_ROOT"
source /opt/ros/humble/setup.bash
source "$CATKIN_VR_ROOT/install/setup.bash"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

POLICY_CONFIG="$REPO_ROOT/artifacts/hdmi_move_suitcase/hdmi_tag/policy.yaml"
MOTION="$REPO_ROOT/assets/mujoco/reference/hdmi_suitcase/motion.npz"
MOTION_META="$REPO_ROOT/assets/mujoco/reference/hdmi_suitcase/meta.json"
MODEL="$REPO_ROOT/artifacts/hdmi_move_suitcase/hdmi_tag/student.onnx"
REQUIRED_FILES=("$CALIBRATION" "$MOTION" "$MOTION_META")
if ((!QUICK_CALIBRATE_ROBOT_MARKERS)); then
  REQUIRED_FILES+=("$POLICY_CONFIG")
fi
if ((FT_ENABLED)); then
  REQUIRED_FILES+=("$FT_CALIBRATION")
fi
if [[ "$RESIDUAL_MODE" != "off" ]]; then
  REQUIRED_FILES+=("$RESIDUAL_MODEL")
fi
for required in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done
if [[ "$RESIDUAL_MODE" == "c1" || "$RESIDUAL_MODE" == "c2" ]]; then
  .venv/bin/python scripts/run_hps_ft_adapter.py \
    --calibration "$FT_CALIBRATION" \
    --require-residual-authority \
    --validate-only
fi
if ((NOMINAL_SHADOW || NOMINAL_APPLY)) && [[ ! -f "$MODEL" ]]; then
  echo "Missing required file: $MODEL" >&2
  exit 1
fi
if ! nc -z -w 2 "$VRPN_ADDRESS" "$VRPN_PORT"; then
  echo "VRPN server is unreachable at $VRPN_ADDRESS:$VRPN_PORT" >&2
  exit 1
fi
if ((!QUICK_CALIBRATE_ROBOT_MARKERS)); then
  if ! ip link show "$ROBOT_INTERFACE" >/dev/null 2>&1; then
    echo "Robot interface does not exist: $ROBOT_INTERFACE" >&2
    exit 1
  fi
  if [[ "$(cat "/sys/class/net/$ROBOT_INTERFACE/carrier" 2>/dev/null || true)" != "1" ]]; then
    echo "Robot interface has no carrier: $ROBOT_INTERFACE" >&2
    exit 1
  fi
  if ! ping -I "$ROBOT_INTERFACE" -c 1 -W 1 "$G1_ADDRESS" >/dev/null; then
    echo "G1 is unreachable at $G1_ADDRESS through $ROBOT_INTERFACE" >&2
    exit 1
  fi
  if ((FT_ENABLED)); then
    if ! nc -z -w 2 "$FT_LEFT_HOST" "$FT_LEFT_PORT"; then
      echo "Left F/T SDK stream is unreachable at $FT_LEFT_HOST:$FT_LEFT_PORT" >&2
      exit 1
    fi
    if ! nc -z -w 2 "$FT_RIGHT_HOST" "$FT_RIGHT_PORT"; then
      echo "Right F/T SDK stream is unreachable at $FT_RIGHT_HOST:$FT_RIGHT_PORT" >&2
      exit 1
    fi
  fi
  PORT_PATTERN=':(5555|5561|5590|5591)'
  PORT_LABEL="5555/5561/5590/5591"
  if ((NOMINAL_APPLY)); then
    PORT_PATTERN=':(5555|5561|5590|5591|5594)'
    PORT_LABEL="$PORT_LABEL/5594"
  fi
  if ((FT_ENABLED)); then
    PORT_PATTERN="${PORT_PATTERN%)}|5580)"
    PORT_LABEL="$PORT_LABEL/5580"
  fi
  if ss -ltn | grep -Eq "$PORT_PATTERN[[:space:]]"; then
    echo "A suitcase runtime port is already in use ($PORT_LABEL)" >&2
    exit 1
  fi
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/outputs/suitcase_hardware/$RUN_ID"
if [[ -n "$RECORD_DIR" ]]; then
  if [[ "$RECORD_DIR" != /* ]]; then
    RECORD_DIR="$INVOCATION_DIR/$RECORD_DIR"
  fi
  RECORD_OUTPUT_DIR="$(realpath -m "$RECORD_DIR")/$RUN_ID"
else
  RECORD_OUTPUT_DIR="$LOG_DIR"
fi
RUNTIME_DIR="$(mktemp -d /tmp/suitcase_hardware.XXXXXX)"
mkdir -p "$LOG_DIR" "$RECORD_OUTPUT_DIR"
PIDS=()

signal_process_group() {
  local signal="$1"
  local pid="$2"
  kill -s "$signal" -- "-$pid" 2>/dev/null || kill -s "$signal" "$pid" 2>/dev/null || true
}

report_log_failure() {
  local label="$1"
  local log_file="$2"
  local reason
  reason="$(grep -E '^(RuntimeError|ValueError|FileNotFoundError):' "$log_file" 2>/dev/null | tail -n 1 || true)"
  if [[ -n "$reason" ]]; then
    echo "$label: ${reason#*: }" >&2
    echo "Full log: $log_file" >&2
  else
    echo "$label; tail of $log_file:" >&2
    tail -n 40 "$log_file" >&2
  fi
}

cleanup() {
  trap - EXIT
  trap '' INT TERM
  if ((ARMED)) && [[ -p "$RUNTIME_DIR/commands" ]]; then
    echo h >&3 2>/dev/null || true
    sleep 0.1
  fi
  for pid in "${PIDS[@]}"; do
    signal_process_group TERM "$pid"
  done
  # Give recorders time to flush their final atomic chunk and manifest.
  for _ in $(seq 1 100); do
    any_alive=0
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null || kill -0 -- "-$pid" 2>/dev/null; then
        any_alive=1
      fi
    done
    ((any_alive == 0)) && break
    sleep 0.1
  done
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null || kill -0 -- "-$pid" 2>/dev/null; then
      signal_process_group KILL "$pid"
    fi
  done
  for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  rm -f "$RUNTIME_DIR/commands" "$RUNTIME_DIR/controller.ready"
  rm -f "$RUNTIME_DIR/bridge.ready"
  rm -f "$RUNTIME_DIR/controller.status"
  rm -f "$RUNTIME_DIR/proposal.ready" "$RUNTIME_DIR/proposal.start"
  rm -f "$RUNTIME_DIR/motion.start" "$RUNTIME_DIR/pose.status"
  rm -f "$RUNTIME_DIR/proposal.complete" "$RUNTIME_DIR/proposal.complete.ack"
  rm -f "$RUNTIME_DIR/ft_tare.trigger"
  rmdir "$RUNTIME_DIR" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

echo "Starting VRPN client; logs: $LOG_DIR"
echo "Policy NPZ records: $RECORD_OUTPUT_DIR"
setsid ros2 launch vrpn_client_ros sample.launch.py >"$LOG_DIR/vrpn.log" 2>&1 &
PIDS+=("$!")
mapfile -t TORSO_TOPICS < <(.venv/bin/python scripts/ros2_torso_to_pelvis.py \
  --torso-topic /robot_g1/pose \
  --torso-from-marker-json "$CALIBRATION" \
  --print-source-topics)
mapfile -t SUITCASE_TOPICS < <(.venv/bin/python scripts/ros2_pose_to_zmq.py \
  --suitcase-only \
  --transform-json "$CALIBRATION" \
  --print-source-topics)
if ((${#TORSO_TOPICS[@]} == 0 || ${#SUITCASE_TOPICS[@]} == 0)); then
  echo "Calibration did not provide torso and suitcase marker topics: $CALIBRATION" >&2
  exit 1
fi
wait_for_any_topic torso "${TORSO_TOPICS[@]}"
wait_for_any_topic suitcase "${SUITCASE_TOPICS[@]}"
echo "VRPN trackers ready"

if ((QUICK_CALIBRATE_ROBOT_MARKERS)); then
  echo "Running frame-0 robot marker quick calibration"
  .venv/bin/python scripts/quick_calibrate_robot_markers.py \
    --calibration "$CALIBRATION" \
    --motion "$MOTION" \
    --motion-meta "$MOTION_META"
  exit 0
fi

if ((ARMED)); then
  read -r -p "Type ARM to permit MotionSwitcher release and zero/hold/init commands: " confirmation
  if [[ "$confirmation" != "ARM" ]]; then
    echo "Arm confirmation rejected"
    exit 1
  fi
  mkfifo "$RUNTIME_DIR/commands"
  exec 3<>"$RUNTIME_DIR/commands"
  setsid .venv/bin/python scripts/g1/real_bridge.py \
    --robot g1 \
    --interface "$ROBOT_INTERFACE" \
    --wait-for-command \
    --ready-file "$RUNTIME_DIR/bridge.ready" \
    >"$LOG_DIR/g1_bridge.log" 2>&1 &
  PIDS+=("$!")
  CONTROLLER_ARGS=(
    --policy-config "$POLICY_CONFIG" \
    --motion "$MOTION" \
    --motion-meta "$MOTION_META" \
    --pilot-authority 1.0 \
    --direct-policy-targets \
    --pilot-init-tolerance 0.50 \
    --max-tilt-deg 180 \
    --max-joint-speed 18 \
    --command-fifo "$RUNTIME_DIR/commands" \
    --ready-file "$RUNTIME_DIR/controller.ready" \
    --status-file "$RUNTIME_DIR/controller.status"
  )
  if ((NOMINAL_APPLY)); then
    CONTROLLER_ARGS+=(
      --proposal-port 5594
      --pilot-log "$RECORD_OUTPUT_DIR/controller_applied.jsonl"
    )
  fi
  setsid .venv/bin/python scripts/g1/suitcase_safe_controller.py \
    "${CONTROLLER_ARGS[@]}" \
    >"$LOG_DIR/safe_controller.log" 2>&1 &
  CONTROLLER_PID="$!"
  PIDS+=("$CONTROLLER_PID")
else
  setsid .venv/bin/python scripts/g1/real_bridge.py \
    --robot g1 \
    --interface "$ROBOT_INTERFACE" \
    --read-only \
    >"$LOG_DIR/g1_bridge.log" 2>&1 &
  PIDS+=("$!")
fi

POSE_RELAY_WATCHDOG_ARGS=(--exit-on-stale)
if ((ARMED)); then
  # Keep interactive recovery available after a transient full occlusion. The
  # relays publish no stale pose; policy runners still fail closed on stream age.
  POSE_RELAY_WATCHDOG_ARGS=()
fi
setsid .venv/bin/python scripts/ros2_pose_to_zmq.py \
    --suitcase-only \
    --suitcase-topic /suitcase/pose \
    --transform-json "$CALIBRATION" \
    --suitcase-preferred-marker-source suitcase4 \
    --marker-position-consensus-m 0.12 \
  --startup-timeout 10 \
  "${POSE_RELAY_WATCHDOG_ARGS[@]}" \
  >"$LOG_DIR/suitcase_relay.log" 2>&1 &
PIDS+=("$!")
setsid .venv/bin/python scripts/ros2_torso_to_pelvis.py \
  --torso-topic /robot_g1/pose \
  --torso-from-marker-json "$CALIBRATION" \
  --startup-timeout 10 \
  "${POSE_RELAY_WATCHDOG_ARGS[@]}" \
  >"$LOG_DIR/pelvis_fk.log" 2>&1 &
PIDS+=("$!")

if ((FT_ENABLED)); then
  FT_ADAPTER_ARGS=(
    --calibration "$FT_CALIBRATION"
    --left-host "$FT_LEFT_HOST"
    --left-port "$FT_LEFT_PORT"
    --right-host "$FT_RIGHT_HOST"
    --right-port "$FT_RIGHT_PORT"
    --output-port 5580
  )
  if ((!FT_SKIP_TARE)); then
    FT_ADAPTER_ARGS+=(
      --tare-on-start
      --tare-trigger-file "$RUNTIME_DIR/ft_tare.trigger"
      --tare-timeout 3600
      --tare-output "$RECORD_OUTPUT_DIR/ft_runtime_bias.json"
    )
  fi
  setsid .venv/bin/python scripts/run_hps_ft_adapter.py \
    "${FT_ADAPTER_ARGS[@]}" \
    >"$LOG_DIR/ft_adapter.log" 2>&1 &
  FT_ADAPTER_PID="$!"
  PIDS+=("$FT_ADAPTER_PID")
fi

if ((ARMED)); then
  for _ in $(seq 1 150); do
    [[ -f "$RUNTIME_DIR/controller.ready" ]] && break
    kill -0 "$CONTROLLER_PID" 2>/dev/null || {
      echo "Safe controller exited before becoming ready" >&2
      exit 1
    }
    sleep 0.1
  done
  if [[ ! -f "$RUNTIME_DIR/controller.ready" ]]; then
    echo "Timed out waiting for safe controller" >&2
    exit 1
  fi
  for _ in $(seq 1 300); do
    [[ -f "$RUNTIME_DIR/bridge.ready" ]] && break
    sleep 0.1
  done
  if [[ ! -f "$RUNTIME_DIR/bridge.ready" ]]; then
    echo "Timed out waiting for armed G1 bridge" >&2
    exit 1
  fi
fi

sleep 2
if ((ARMED && !CHECK_ONLY)); then
  # A bad robot-marker calibration can suppress the fused pelvis stream. Keep
  # hold/init/calibration reachable so the operator can repair that calibration;
  # the policy preflight below still requires pelvis, suitcase, and low-state.
  .venv/bin/python scripts/check_suitcase_streams.py --duration 3 \
    --allow-missing pelvis \
    | tee "$LOG_DIR/stream_check.log"
  if grep -Eq '^pelvis: frames=0([[:space:]]|$)' "$LOG_DIR/stream_check.log"; then
    echo "WARNING: fused pelvis is unavailable; policy control remains blocked."
    echo "Calibration recovery mode: press i, wait for init completion, then press c."
  fi
else
  .venv/bin/python scripts/check_suitcase_streams.py --duration 3 \
    | tee "$LOG_DIR/stream_check.log"
fi

if ((CHECK_ONLY)); then
  if ((FT_ENABLED)); then
    .venv/bin/python scripts/run_hps_ft_adapter.py \
      --calibration "$FT_CALIBRATION" --validate-only
  fi
  exit 0
fi

if ((!ARMED)); then
  echo "Read-only suitcase stack is running. Press Ctrl-C to stop."
  while true; do sleep 1; done
fi

RUNNER_FT_ARGS=()
if ((FT_ENABLED)); then
  RUNNER_FT_ARGS+=(
    --ft-port 5580
    --ft-max-age-ms 100
    --ft-calibration "$FT_CALIBRATION"
    --require-both-ft-valid
    --residual-mode "$RESIDUAL_MODE"
  )
  if [[ "$RESIDUAL_MODE" != "off" ]]; then
    RUNNER_FT_ARGS+=(--residual "$RESIDUAL_MODEL")
  fi
fi

echo "Safe controller active. No ONNX policy is loaded."
if ((NOMINAL_APPLY)); then
  echo "Commands: z=zero-policy/current-follow, h=hold current, i=motion-frame-0 init, t=tare F/T, c=calibrate robot markers, p=policy pilot, q=quit"
elif ((NOMINAL_SHADOW)); then
  echo "Commands: z=zero-policy/current-follow, h=hold current, i=init pose, t=tare F/T, c=calibrate robot markers, s=policy shadow, q=quit"
else
  echo "Commands: z=zero-policy/current-follow, h=hold current, i=init pose, t=tare F/T, c=calibrate robot markers, q=quit"
fi
while kill -0 "$CONTROLLER_PID" 2>/dev/null; do
  read -r -p "suitcase-safe> " command || command=q
  case "$command" in
    z|zero|h|hold|i|init|q|quit|exit)
      echo "$command" >&3
      ;;
    t|tare)
      if ((!FT_ENABLED)); then
        echo "F/T integration is disabled; restart with --ft"
        continue
      fi
      if ((FT_SKIP_TARE)); then
        echo "This run uses --ft-skip-tare; no runtime bias calibration is pending"
        continue
      fi
      if [[ "$(cat "$RUNTIME_DIR/controller.status" 2>/dev/null || true)" != "init_complete" ]]; then
        echo "F/T tare requires motion-frame-0 init; press i and wait for completion"
        continue
      fi
      echo "Keep both configured hands installed, remove all other wrist loads, keep both arms still, and type TARE to continue."
      read -r -p "F/T tare confirmation: " tare_confirmation
      if [[ "$tare_confirmation" != "TARE" ]]; then
        echo "F/T tare cancelled"
        continue
      fi
      touch "$RUNTIME_DIR/ft_tare.trigger"
      for _ in $(seq 1 400); do
        [[ -f "$RECORD_OUTPUT_DIR/ft_runtime_bias.json" ]] && break
        kill -0 "$FT_ADAPTER_PID" 2>/dev/null || break
        sleep 0.05
      done
      FT_TARE_CHECK_LOG="$LOG_DIR/ft_bias_check.log"
      if .venv/bin/python scripts/check_hps_ft_stream.py \
        --duration 3 --require-both-valid \
        --output "$RECORD_OUTPUT_DIR/ft_bias_acceptance.npz" \
        >"$FT_TARE_CHECK_LOG" 2>&1; then
        cat "$FT_TARE_CHECK_LOG"
        echo "F/T gravity-aware bias calibration accepted; synchronized policy input is ready"
      else
        report_log_failure "F/T bias calibration or stream validation failed" "$FT_TARE_CHECK_LOG"
      fi
      ;;
    c|calibrate)
      if [[ "$(cat "$RUNTIME_DIR/controller.status" 2>/dev/null || true)" != "init_complete" ]]; then
        echo "Robot marker calibration requires motion-frame-0 init; press i and wait about 10 seconds"
        continue
      fi
      echo "Keep G1 still in init hold and place the suitcase at its motion frame-0 pose."
      echo "Calibrating robot1/robot2/robot3 T_marker_torso directly from measured poses..."
      if .venv/bin/python scripts/quick_calibrate_robot_markers.py \
        --calibration "$CALIBRATION" \
        --motion "$MOTION" \
        --motion-meta "$MOTION_META" \
        2>&1 | tee "$LOG_DIR/robot_marker_calibration.log"; then
        echo "Calibration updated. Exiting so all pose relays reload the new matrices on the next run."
        echo q >&3
        wait "$CONTROLLER_PID" 2>/dev/null || true
        break
      fi
      echo "Robot marker calibration failed; controller remains in hold" >&2
      ;;
    p|pilot)
      if ((!NOMINAL_APPLY)); then
        echo "Nominal apply was not enabled; restart with --armed --nominal-apply"
        continue
      fi
      if [[ "$(cat "$RUNTIME_DIR/controller.status" 2>/dev/null || true)" != "init_complete" ]]; then
        echo "Nominal pilot requires a completed motion-frame-0 init; press i and wait about 10 seconds"
        continue
      fi
      PREFLIGHT_LOG="$LOG_DIR/p_stream_preflight_$(date +%H%M%S).log"
      if ! .venv/bin/python scripts/check_suitcase_streams.py --duration 1 \
        >"$PREFLIGHT_LOG" 2>&1; then
        report_log_failure "p rejected before ONNX startup: required live stream check failed" "$PREFLIGHT_LOG"
        echo h >&3
        continue
      fi
      cat "$PREFLIGHT_LOG"
      if ((FT_ENABLED)); then
        FT_PREFLIGHT_LOG="$LOG_DIR/p_ft_preflight_$(date +%H%M%S).log"
        if ! .venv/bin/python scripts/check_hps_ft_stream.py \
          --duration 1 --require-both-valid >"$FT_PREFLIGHT_LOG" 2>&1; then
          report_log_failure "p rejected before ONNX startup: F/T stream check failed" "$FT_PREFLIGHT_LOG"
          echo h >&3
          continue
        fi
        cat "$FT_PREFLIGHT_LOG"
      fi
      APPLY_STEM="$RECORD_OUTPUT_DIR/policy_apply_$(date +%H%M%S)"
      rm -f "$RUNTIME_DIR/proposal.ready" "$RUNTIME_DIR/proposal.start"
      rm -f "$RUNTIME_DIR/motion.start" "$RUNTIME_DIR/pose.status"
      rm -f "$RUNTIME_DIR/proposal.complete" "$RUNTIME_DIR/proposal.complete.ack"
      setsid .venv/bin/python scripts/run_hdmi_suitcase_nominal_shadow.py \
        --steps 472 \
        --rate 50 \
        --reference-mode advance \
        --max-initial-error 0.50 \
        --max-object-xy-error 0.20 \
        --proposal-port 5594 \
        --ready-file "$RUNTIME_DIR/proposal.ready" \
        --start-file "$RUNTIME_DIR/proposal.start" \
        --motion-start-file "$RUNTIME_DIR/motion.start" \
        --pose-status-file "$RUNTIME_DIR/pose.status" \
        --completion-file "$RUNTIME_DIR/proposal.complete" \
        --completion-ack-file "$RUNTIME_DIR/proposal.complete.ack" \
        "${RUNNER_FT_ARGS[@]}" \
        --output "$APPLY_STEM.npz" \
        >"$APPLY_STEM.log" 2>&1 &
      APPLY_PID="$!"
      PIDS+=("$APPLY_PID")
      for _ in $(seq 1 300); do
        [[ -f "$RUNTIME_DIR/proposal.ready" ]] && break
        kill -0 "$APPLY_PID" 2>/dev/null || break
        sleep 0.1
      done
      if [[ ! -f "$RUNTIME_DIR/proposal.ready" ]]; then
        wait "$APPLY_PID" 2>/dev/null || true
        report_log_failure "p rejected before policy activation" "$APPLY_STEM.log"
        echo h >&3
        continue
      fi
      echo "Proposal process is ready. The run lasts about 9.5 seconds at 100% authority."
      read -r -p "Type FULL to start direct policy-faithful nominal control: " pilot_confirmation
      if [[ "$pilot_confirmation" != "FULL" ]]; then
        echo "Pilot confirmation rejected; robot remains in hold"
        signal_process_group TERM "$APPLY_PID"
        wait "$APPLY_PID" 2>/dev/null || true
        continue
      fi
      echo p >&3
      for _ in $(seq 1 20); do
        [[ "$(cat "$RUNTIME_DIR/controller.status" 2>/dev/null || true)" == "pilot_waiting" ]] && break
        sleep 0.05
      done
      if [[ "$(cat "$RUNTIME_DIR/controller.status" 2>/dev/null || true)" != "pilot_waiting" ]]; then
        echo "Safe controller rejected pilot entry; robot remains in hold" >&2
        signal_process_group TERM "$APPLY_PID"
        wait "$APPLY_PID" 2>/dev/null || true
        echo h >&3
        continue
      fi
      touch "$RUNTIME_DIR/proposal.start"
      PILOT_STOPPED=0
      PILOT_QUIT=0
      MOTION_STARTED=0
      MOTION_COMPLETED=0
      LAST_POSE_STATUS=""
      for _ in $(seq 1 40); do
        PILOT_STATUS="$(cat "$RUNTIME_DIR/controller.status" 2>/dev/null || true)"
        [[ "$PILOT_STATUS" == "pilot_active" ]] && break
        [[ "$PILOT_STATUS" == pilot_abort:* ]] && break
        sleep 0.05
      done
      if [[ "$(cat "$RUNTIME_DIR/controller.status" 2>/dev/null || true)" != "pilot_active" ]]; then
        echo "Policy did not enter stabilization; robot remains in hold" >&2
        signal_process_group TERM "$APPLY_PID"
        wait "$APPLY_PID" 2>/dev/null || true
        echo h >&3
        continue
      fi
      echo "DIRECT POLICY STABILIZATION ACTIVE at motion frame 0. Fully release the ropes, then type GO."
      echo "At any time, type h then Enter to stop policy control; use the remote emergency stop for instability."
      while kill -0 "$APPLY_PID" 2>/dev/null; do
        if [[ -f "$RUNTIME_DIR/proposal.complete" ]]; then
          echo "Policy motion complete; switching controller to hold before finalizing the record."
          echo h >&3
          for _ in $(seq 1 40); do
            [[ "$(cat "$RUNTIME_DIR/controller.status" 2>/dev/null || true)" == "hold" ]] && break
            sleep 0.05
          done
          if [[ "$(cat "$RUNTIME_DIR/controller.status" 2>/dev/null || true)" != "hold" ]]; then
            echo "Controller did not confirm hold after policy completion" >&2
            signal_process_group TERM "$APPLY_PID"
            PILOT_STOPPED=1
          else
            touch "$RUNTIME_DIR/proposal.complete.ack"
            MOTION_COMPLETED=1
          fi
          break
        fi
        POSE_STATUS="$(cat "$RUNTIME_DIR/pose.status" 2>/dev/null || true)"
        if [[ "$POSE_STATUS" != "$LAST_POSE_STATUS" ]]; then
          if [[ "$POSE_STATUS" == blocked:* ]]; then
            echo "MULTI-MARKER POSE LOST ($POSE_STATUS): proposal is aborting and the controller will hold."
          fi
          LAST_POSE_STATUS="$POSE_STATUS"
        fi
        if read -r -t 0.1 pilot_command; then
          case "$pilot_command" in
            g|go|G|GO)
              if ((!MOTION_STARTED)); then
                if [[ "$POSE_STATUS" != "ready" ]]; then
                  echo "GO refused: corrected robot/suitcase pose is not currently visible"
                else
                  touch "$RUNTIME_DIR/motion.start"
                  MOTION_STARTED=1
                  echo "MOTION ACTIVE: advancing 472 reference steps (about 9.5 seconds)."
                fi
              else
                echo "Motion is already active"
              fi
              ;;
            h|hold)
              echo h >&3
              signal_process_group TERM "$APPLY_PID"
              PILOT_STOPPED=1
              ;;
            q|quit|exit)
              echo h >&3
              signal_process_group TERM "$APPLY_PID"
              PILOT_STOPPED=1
              PILOT_QUIT=1
              ;;
            *)
              if ((MOTION_STARTED)); then
                echo "During motion, allowed commands: h q"
              else
                echo "During stabilization, allowed commands: GO h q"
              fi
              ;;
          esac
        fi
        PILOT_STATUS="$(cat "$RUNTIME_DIR/controller.status" 2>/dev/null || true)"
        if [[ "$PILOT_STATUS" == pilot_abort:* ]]; then
          echo "Safe controller aborted nominal pilot: $PILOT_STATUS" >&2
          signal_process_group TERM "$APPLY_PID"
          PILOT_STOPPED=1
        fi
        ((PILOT_STOPPED)) && break
      done
      if wait "$APPLY_PID"; then
        APPLY_OK=1
      else
        APPLY_OK=0
      fi
      echo h >&3
      if [[ ! -f "$APPLY_STEM.summary.json" && -f "$APPLY_STEM.recording/manifest.json" ]]; then
        echo "Finalizing recoverable policy chunks after runner exit"
        .venv/bin/python scripts/finalize_suitcase_record.py \
          --output "$APPLY_STEM.npz" \
          --reason "${PILOT_STATUS:-runner_exit}" \
          >"$APPLY_STEM.finalize.log" 2>&1 || \
          report_log_failure "Policy chunk finalization failed" "$APPLY_STEM.finalize.log"
      fi
      if ((APPLY_OK)); then
        echo "Nominal proposal sequence completed; controller returned to hold"
        cat "$APPLY_STEM.summary.json"
      elif ((!PILOT_STOPPED)); then
        report_log_failure "Nominal proposal process failed" "$APPLY_STEM.log"
      fi
      if ((PILOT_QUIT)); then
        echo q >&3
        wait "$CONTROLLER_PID" 2>/dev/null || true
        break
      fi
      ;;
    s|shadow)
      if ((!NOMINAL_SHADOW)); then
        echo "Nominal shadow was not enabled; restart with --armed --nominal-shadow"
        continue
      fi
      if [[ "$(cat "$RUNTIME_DIR/controller.status" 2>/dev/null || true)" != "init_complete" ]]; then
        echo "Nominal shadow requires a completed init; press i and wait about 10 seconds"
        continue
      fi
      PREFLIGHT_LOG="$LOG_DIR/s_stream_preflight_$(date +%H%M%S).log"
      if ! .venv/bin/python scripts/check_suitcase_streams.py --duration 1 \
        >"$PREFLIGHT_LOG" 2>&1; then
        report_log_failure "s rejected before ONNX startup: required live stream check failed" "$PREFLIGHT_LOG"
        continue
      fi
      cat "$PREFLIGHT_LOG"
      if ((FT_ENABLED)); then
        FT_PREFLIGHT_LOG="$LOG_DIR/s_ft_preflight_$(date +%H%M%S).log"
        if ! .venv/bin/python scripts/check_hps_ft_stream.py \
          --duration 1 --require-both-valid >"$FT_PREFLIGHT_LOG" 2>&1; then
          report_log_failure "s rejected before ONNX startup: F/T stream check failed" "$FT_PREFLIGHT_LOG"
          continue
        fi
        cat "$FT_PREFLIGHT_LOG"
      fi
      SHADOW_STEM="$RECORD_OUTPUT_DIR/policy_shadow_$(date +%H%M%S)"
      echo "Running 472-step policy shadow; robot remains in hold"
      if .venv/bin/python scripts/run_hdmi_suitcase_nominal_shadow.py \
        --steps 472 \
        --rate 50 \
        "${RUNNER_FT_ARGS[@]}" \
        --output "$SHADOW_STEM.npz" \
        >"$SHADOW_STEM.log" 2>&1; then
        cat "$SHADOW_STEM.summary.json"
      else
        echo "Nominal shadow failed; tail of $SHADOW_STEM.log:" >&2
        tail -n 40 "$SHADOW_STEM.log" >&2
      fi
      ;;
    *)
      if ((NOMINAL_APPLY)); then
        echo "Allowed commands: z h i t c p q"
      elif ((NOMINAL_SHADOW)); then
        echo "Allowed commands: z h i t c s q"
      else
        echo "Allowed commands: z h i t c q"
      fi
      continue
      ;;
  esac
  case "$command" in
    i|init)
      echo "Waiting for the 10-second init interpolation to complete..."
      for _ in $(seq 1 150); do
        [[ "$(cat "$RUNTIME_DIR/controller.status" 2>/dev/null || true)" == "init_complete" ]] && break
        kill -0 "$CONTROLLER_PID" 2>/dev/null || break
        sleep 0.1
      done
      if [[ "$(cat "$RUNTIME_DIR/controller.status" 2>/dev/null || true)" == "init_complete" ]]; then
        echo "Init complete; robot is holding the suitcase initial pose"
      else
        echo "Init did not complete; inspect $LOG_DIR/safe_controller.log" >&2
      fi
      ;;
  esac
  case "$command" in
    q|quit|exit)
      wait "$CONTROLLER_PID" 2>/dev/null || true
      break
      ;;
  esac
done
