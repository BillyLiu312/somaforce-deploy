#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CATKIN_VR_ROOT="${CATKIN_VR_ROOT:-/home/irmv/catkin_vr}"
ROBOT_INTERFACE="enx9c69d373f0ff"
G1_ADDRESS="${G1_ADDRESS:-192.168.123.161}"
VRPN_ADDRESS="${VRPN_ADDRESS:-192.168.5.58}"
VRPN_PORT="${VRPN_PORT:-3883}"
ARMED=0
CHECK_ONLY=0
NOMINAL_SHADOW=0
NOMINAL_APPLY=0

usage() {
  echo "Usage: $0 [--interface NAME] [--armed] [--nominal-shadow|--nominal-apply] [--check-only]"
}

wait_for_topic() {
  local topic="$1"
  local deadline=$((SECONDS + 20))
  while ((SECONDS < deadline)); do
    if ros2 topic list 2>/dev/null | grep -Fxq "$topic"; then
      timeout 5 ros2 topic echo \
        --qos-reliability best_effort \
        --once \
        "$topic" \
        >/dev/null
      return 0
    fi
    sleep 0.2
  done
  echo "Timed out waiting for ROS2 topic: $topic" >&2
  return 1
}

while (($#)); do
  case "$1" in
    --interface)
      ROBOT_INTERFACE="$2"
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

cd "$REPO_ROOT"
source /opt/ros/humble/setup.bash
source "$CATKIN_VR_ROOT/install/setup.bash"
export HF_HUB_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

POLICY_CONFIG="$REPO_ROOT/artifacts/hdmi_move_suitcase/hdmi_tag/policy.yaml"
CALIBRATION="$REPO_ROOT/calibration/marker_policy.json"
MOTION="$REPO_ROOT/assets/mujoco/reference/hdmi_suitcase/motion.npz"
MOTION_META="$REPO_ROOT/assets/mujoco/reference/hdmi_suitcase/meta.json"
MODEL="$REPO_ROOT/artifacts/hdmi_move_suitcase/hdmi_tag/student.onnx"
for required in "$POLICY_CONFIG" "$CALIBRATION" "$MOTION" "$MOTION_META"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing required file: $required" >&2
    exit 1
  fi
done
if ((NOMINAL_SHADOW || NOMINAL_APPLY)) && [[ ! -f "$MODEL" ]]; then
  echo "Missing required file: $MODEL" >&2
  exit 1
fi
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
if ! nc -z -w 2 "$VRPN_ADDRESS" "$VRPN_PORT"; then
  echo "VRPN server is unreachable at $VRPN_ADDRESS:$VRPN_PORT" >&2
  exit 1
fi
PORT_PATTERN=':(5555|5561|5590|5591)'
PORT_LABEL="5555/5561/5590/5591"
if ((NOMINAL_APPLY)); then
  PORT_PATTERN=':(5555|5561|5590|5591|5594)'
  PORT_LABEL="$PORT_LABEL/5594"
fi
if ss -ltn | grep -Eq "$PORT_PATTERN[[:space:]]"; then
  echo "A suitcase runtime port is already in use ($PORT_LABEL)" >&2
  exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/outputs/suitcase_hardware/$RUN_ID"
RUNTIME_DIR="$(mktemp -d /tmp/suitcase_hardware.XXXXXX)"
mkdir -p "$LOG_DIR"
PIDS=()

signal_process_group() {
  local signal="$1"
  local pid="$2"
  kill -s "$signal" -- "-$pid" 2>/dev/null || kill -s "$signal" "$pid" 2>/dev/null || true
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
  for _ in $(seq 1 20); do
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
  rmdir "$RUNTIME_DIR" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

echo "Starting VRPN client; logs: $LOG_DIR"
setsid ros2 launch vrpn_client_ros sample.launch.py >"$LOG_DIR/vrpn.log" 2>&1 &
PIDS+=("$!")
wait_for_topic /robot_g1/pose
wait_for_topic /suitcase/pose
echo "VRPN trackers ready"

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
    --max-joint-speed 12 \
    --command-fifo "$RUNTIME_DIR/commands" \
    --ready-file "$RUNTIME_DIR/controller.ready" \
    --status-file "$RUNTIME_DIR/controller.status"
  )
  if ((NOMINAL_APPLY)); then
    CONTROLLER_ARGS+=(
      --proposal-port 5594
      --pilot-log "$LOG_DIR/nominal_pilot_applied.jsonl"
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
if ((NOMINAL_APPLY)); then
  POSE_RELAY_WATCHDOG_ARGS=()
fi
setsid .venv/bin/python scripts/ros2_pose_to_zmq.py \
  --suitcase-only \
  --transform-json "$CALIBRATION" \
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
.venv/bin/python scripts/check_suitcase_streams.py --duration 3 \
  | tee "$LOG_DIR/stream_check.log"

if ((CHECK_ONLY)); then
  exit 0
fi

if ((!ARMED)); then
  echo "Read-only suitcase stack is running. Press Ctrl-C to stop."
  while true; do sleep 1; done
fi

echo "Safe controller active. No ONNX policy is loaded."
if ((NOMINAL_APPLY)); then
  echo "Commands: z=zero-policy/current-follow, h=hold current, i=motion-frame-0 init, p=100% policy-faithful nominal, q=quit"
elif ((NOMINAL_SHADOW)); then
  echo "Commands: z=zero-policy/current-follow, h=hold current, i=init pose, s=nominal shadow, q=quit"
else
  echo "Commands: z=zero-policy/current-follow, h=hold current, i=init pose, q=quit"
fi
while kill -0 "$CONTROLLER_PID" 2>/dev/null; do
  read -r -p "suitcase-safe> " command || command=q
  case "$command" in
    z|zero|h|hold|i|init|q|quit|exit)
      echo "$command" >&3
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
      APPLY_STEM="$LOG_DIR/nominal_apply_$(date +%H%M%S)"
      rm -f "$RUNTIME_DIR/proposal.ready" "$RUNTIME_DIR/proposal.start"
      rm -f "$RUNTIME_DIR/motion.start" "$RUNTIME_DIR/pose.status"
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
        --motion-pelvis-pose-grace 2.0 \
        --suitcase-attachment-fallback \
        --suitcase-lift-threshold 0.05 \
        --suitcase-fallback-delay 0.06 \
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
        echo "Nominal proposal process failed before readiness; tail of $APPLY_STEM.log:" >&2
        wait "$APPLY_PID" 2>/dev/null || true
        tail -n 40 "$APPLY_STEM.log" >&2
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
        POSE_STATUS="$(cat "$RUNTIME_DIR/pose.status" 2>/dev/null || true)"
        if [[ "$POSE_STATUS" != "$LAST_POSE_STATUS" ]]; then
          if [[ "$POSE_STATUS" == blocked:* ]]; then
            echo "MARKER OCCLUDED ($POSE_STATUS): policy continues with the last pose; GO is disabled."
          elif [[ "$POSE_STATUS" == "degraded:suitcase_attached" ]]; then
            echo "SUITCASE MARKER OCCLUDED AFTER CONFIRMED LIFT: using reference-guided attached pose."
          elif [[ "$POSE_STATUS" == "degraded:pelvis,suitcase_attached" ]]; then
            echo "BOTH MARKERS OCCLUDED: using attached suitcase estimate and the brief pelvis grace."
          elif [[ "$POSE_STATUS" == degraded:* ]]; then
            echo "ROBOT MARKER OCCLUDED DURING MOTION: continuing briefly with the last pelvis pose."
          elif [[ "$POSE_STATUS" == "ready" && ( "$LAST_POSE_STATUS" == blocked:* || "$LAST_POSE_STATUS" == degraded:* ) ]]; then
            if [[ "$LAST_POSE_STATUS" == *suitcase_attached* ]]; then
              echo "SUITCASE MARKER RESTORED: returning to live corrected suitcase pose."
            elif ((MOTION_STARTED)); then
              echo "ROBOT MARKER RESTORED: continuing motion with live pelvis pose."
            else
              echo "MARKER RESTORED: frame-0 policy stabilization resumed; GO is enabled."
            fi
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
      if ((APPLY_OK)); then
        echo "Nominal proposal sequence completed; controller returned to hold"
        cat "$APPLY_STEM.summary.json"
      elif ((!PILOT_STOPPED)); then
        echo "Nominal proposal process failed; tail of $APPLY_STEM.log:" >&2
        tail -n 40 "$APPLY_STEM.log" >&2
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
      SHADOW_STEM="$LOG_DIR/nominal_shadow_$(date +%H%M%S)"
      echo "Running 472-step nominal shadow; robot remains in hold"
      if .venv/bin/python scripts/run_hdmi_suitcase_nominal_shadow.py \
        --steps 472 \
        --rate 50 \
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
        echo "Allowed commands: z h i p q"
      elif ((NOMINAL_SHADOW)); then
        echo "Allowed commands: z h i s q"
      else
        echo "Allowed commands: z h i q"
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
