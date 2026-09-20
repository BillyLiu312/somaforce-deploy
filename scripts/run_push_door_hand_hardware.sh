#!/usr/bin/env bash
# Run the HDMI push_door_hand student on a physical G1.
#
# This intentionally mirrors run_suitcase_hardware.sh's operator protocol.  The
# HARDWARE-ONLY DIFFERENCE blocks are the parts that do not exist in the
# push_door_hand sim2sim harness: ROS/VRPN discovery, live G1 low-state and
# MotionSwitcher I/O, the two calibrated door pose relays, and optional wrist
# F/T ingestion.  No hardware test is performed by this repository change.
set -euo pipefail

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
DOOR_CALIBRATION=""
DOOR_MARKER_ROLE="suitcase"
DOOR_PANEL_MARKER_ROLE="suitcase"
QUICK_CALIBRATE=0
FT_ENABLED=0
FT_CALIBRATION="$REPO_ROOT/calibration/hps_g1.yaml"
FT_LEFT_HOST="127.0.0.1"
FT_LEFT_PORT=9000
FT_RIGHT_HOST="127.0.0.1"
FT_RIGHT_PORT=9001
FT_SKIP_TARE=0
RESIDUAL_MODE=off
RESIDUAL_MODE_SET=0
RESIDUAL_MODEL="$REPO_ROOT/artifacts/hdmi_push_box/cross_residual.onnx"
RECORD_DIR=""

POLICY_CONFIG="$REPO_ROOT/artifacts/hdmi_push_door_hand/hdmi_tag/policy.yaml"
MODEL="$REPO_ROOT/artifacts/hdmi_push_door_hand/hdmi_tag/student.onnx"
MOTION="$REPO_ROOT/assets/mujoco/reference/hdmi_push_door_hand/motion.npz"
MOTION_META="$REPO_ROOT/assets/mujoco/reference/hdmi_push_door_hand/meta.json"
DOOR_STEPS=573

usage() {
  cat <<EOF
Usage: $0 [--interface NAME] [--calibration FILE]
  [--door-calibration FILE] [--door-marker-role ROLE] [--door-panel-marker-role ROLE]
  [--quick-calibrate-robot-markers] [--armed]
  [--nominal-shadow|--nominal-apply] [--ft] [--ft-calibration FILE]
  [--ft-left-host HOST] [--ft-left-port PORT] [--ft-right-host HOST]
  [--ft-right-port PORT] [--ft-skip-tare]
  [--residual-mode off|shadow|c1|c2] [--residual-model FILE]
  [--record-dir DIR] [--check-only]

Default mode is read-only.  --nominal-shadow audits the 573-step policy while
the robot remains in hold; --nominal-apply enables the guarded proposal pilot.
EOF
}

wait_for_any_topic() {
  local role="$1"; shift
  local deadline=$((SECONDS + 20))
  while ((SECONDS < deadline)); do
    local available; available="$(ros2 topic list 2>/dev/null || true)"
    local topic
    for topic in "$@"; do
      if grep -Fxq "$topic" <<<"$available" && timeout 5 ros2 topic echo \
        --qos-reliability best_effort --once "$topic" >/dev/null; then
        echo "$role marker ready: $topic"; return 0
      fi
    done
    sleep 0.2
  done
  echo "Timed out waiting for $role marker topic: $*" >&2
  return 1
}

while (($#)); do
  case "$1" in
    --interface) ROBOT_INTERFACE="$2"; shift 2;;
    --calibration) CALIBRATION="$2"; shift 2;;
    --door-calibration) DOOR_CALIBRATION="$2"; shift 2;;
    --door-marker-role) DOOR_MARKER_ROLE="$2"; shift 2;;
    --door-panel-marker-role) DOOR_PANEL_MARKER_ROLE="$2"; shift 2;;
    --quick-calibrate-robot-markers) QUICK_CALIBRATE=1; shift;;
    --armed) ARMED=1; shift;;
    --check-only) CHECK_ONLY=1; shift;;
    --nominal-shadow) NOMINAL_SHADOW=1; shift;;
    --nominal-apply) NOMINAL_APPLY=1; shift;;
    --ft) FT_ENABLED=1; shift;;
    --ft-calibration) FT_CALIBRATION="$2"; shift 2;;
    --ft-left-host) FT_LEFT_HOST="$2"; shift 2;;
    --ft-left-port) FT_LEFT_PORT="$2"; shift 2;;
    --ft-right-host) FT_RIGHT_HOST="$2"; shift 2;;
    --ft-right-port) FT_RIGHT_PORT="$2"; shift 2;;
    --ft-skip-tare) FT_SKIP_TARE=1; shift;;
    --residual-mode) RESIDUAL_MODE="$2"; RESIDUAL_MODE_SET=1; shift 2;;
    --residual-model) RESIDUAL_MODEL="$2"; shift 2;;
    --record-dir) RECORD_DIR="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2;;
  esac
done

if [[ -z "$DOOR_CALIBRATION" ]]; then DOOR_CALIBRATION="$CALIBRATION"; fi
if ((NOMINAL_SHADOW && NOMINAL_APPLY)); then echo "choose one nominal mode" >&2; exit 2; fi
if ((NOMINAL_SHADOW || NOMINAL_APPLY)) && ((!ARMED)); then
  echo "nominal mode requires --armed" >&2; exit 2
fi
if ((QUICK_CALIBRATE && (ARMED || NOMINAL_SHADOW || NOMINAL_APPLY || CHECK_ONLY))); then
  echo "--quick-calibrate-robot-markers must be used by itself" >&2; exit 2
fi
case "$RESIDUAL_MODE" in off|shadow|c1|c2) ;; *) echo "invalid residual mode" >&2; exit 2;; esac
if ((FT_ENABLED && !RESIDUAL_MODE_SET && (NOMINAL_SHADOW || NOMINAL_APPLY))); then RESIDUAL_MODE=shadow; fi
if [[ "$RESIDUAL_MODE" != off ]] && ((!FT_ENABLED)); then echo "residual mode requires --ft" >&2; exit 2; fi
if [[ "$RESIDUAL_MODE" == c1 || "$RESIDUAL_MODE" == c2 ]] && ((!NOMINAL_APPLY)); then
  echo "c1/c2 requires --armed --nominal-apply" >&2; exit 2
fi
if ((FT_SKIP_TARE && !FT_ENABLED)); then echo "--ft-skip-tare requires --ft" >&2; exit 2; fi
if ((FT_ENABLED && !ARMED && !FT_SKIP_TARE && !CHECK_ONLY)); then
  echo "unarmed F/T requires --ft-skip-tare" >&2; exit 2
fi

cd "$REPO_ROOT"
source /opt/ros/humble/setup.bash
source "$CATKIN_VR_ROOT/install/setup.bash"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1

REQUIRED_FILES=("$CALIBRATION" "$DOOR_CALIBRATION" "$POLICY_CONFIG" "$MOTION" "$MOTION_META")
if ((FT_ENABLED)); then REQUIRED_FILES+=("$FT_CALIBRATION"); fi
if [[ "$RESIDUAL_MODE" != off ]]; then REQUIRED_FILES+=("$RESIDUAL_MODEL"); fi
for required in "${REQUIRED_FILES[@]}"; do
  [[ -f "$required" ]] || { echo "Missing required file: $required" >&2; exit 1; }
done
if [[ "$RESIDUAL_MODE" == c1 || "$RESIDUAL_MODE" == c2 ]]; then
  .venv/bin/python scripts/run_hps_ft_adapter.py --calibration "$FT_CALIBRATION" \
    --require-residual-authority --validate-only
fi

# HARDWARE-ONLY DIFFERENCE: these checks are intentionally skipped only for
# --quick-calibrate-robot-markers, exactly as in the suitcase deployment.
if ((!QUICK_CALIBRATE)); then
  ip link show "$ROBOT_INTERFACE" >/dev/null 2>&1 || { echo "Robot interface missing: $ROBOT_INTERFACE" >&2; exit 1; }
  [[ "$(cat "/sys/class/net/$ROBOT_INTERFACE/carrier" 2>/dev/null || true)" == 1 ]] || { echo "Robot interface has no carrier" >&2; exit 1; }
  ping -I "$ROBOT_INTERFACE" -c 1 -W 1 "$G1_ADDRESS" >/dev/null || { echo "G1 unreachable: $G1_ADDRESS" >&2; exit 1; }
  nc -z -w 2 "$VRPN_ADDRESS" "$VRPN_PORT" || { echo "VRPN unreachable: $VRPN_ADDRESS:$VRPN_PORT" >&2; exit 1; }
  if ((FT_ENABLED)); then
    nc -z -w 2 "$FT_LEFT_HOST" "$FT_LEFT_PORT" || { echo "left F/T stream unreachable" >&2; exit 1; }
    nc -z -w 2 "$FT_RIGHT_HOST" "$FT_RIGHT_PORT" || { echo "right F/T stream unreachable" >&2; exit 1; }
  fi
fi

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
cleanup() {
  trap - EXIT INT TERM
  if ((ARMED)) && [[ -p "$RUNTIME_DIR/commands" ]]; then echo h >&3 2>/dev/null || true; fi
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

echo "Starting VRPN client; logs: $LOG_DIR"
setsid ros2 launch vrpn_client_ros sample.launch.py >"$LOG_DIR/vrpn.log" 2>&1 & PIDS+=("$!")
mapfile -t TORSO_TOPICS < <(.venv/bin/python scripts/ros2_torso_to_pelvis.py --torso-topic /robot_g1/pose \
  --torso-from-marker-json "$CALIBRATION" --print-source-topics)
mapfile -t DOOR_TOPICS < <(.venv/bin/python scripts/ros2_pose_to_zmq.py --suitcase-only \
  --transform-json "$DOOR_CALIBRATION" --marker-role "$DOOR_MARKER_ROLE" --print-source-topics)
mapfile -t PANEL_TOPICS < <(.venv/bin/python scripts/ros2_pose_to_zmq.py --suitcase-only \
  --transform-json "$DOOR_CALIBRATION" --marker-role "$DOOR_PANEL_MARKER_ROLE" --print-source-topics)
if ((${#TORSO_TOPICS[@]} == 0 || ${#DOOR_TOPICS[@]} == 0 || ${#PANEL_TOPICS[@]} == 0)); then
  echo "Calibration did not provide torso, door, and door_panel topics" >&2; exit 1
fi
wait_for_any_topic torso "${TORSO_TOPICS[@]}"
wait_for_any_topic door "${DOOR_TOPICS[@]}"
wait_for_any_topic door_panel "${PANEL_TOPICS[@]}"
echo "VRPN trackers ready (torso, door, door_panel)"

if ((QUICK_CALIBRATE)); then
  echo "Door task uses the same frame-0 robot marker calibration utility as suitcase."
  .venv/bin/python scripts/quick_calibrate_robot_markers.py --calibration "$CALIBRATION" \
    --motion "$MOTION" --motion-meta "$MOTION_META"
  exit 0
fi

if ((ARMED)); then
  read -r -p "Type ARM to permit MotionSwitcher release and G1 commands: " confirmation
  [[ "$confirmation" == ARM ]] || { echo "Arm confirmation rejected"; exit 1; }
  mkfifo "$RUNTIME_DIR/commands"; exec 3<>"$RUNTIME_DIR/commands"
  # HARDWARE-ONLY DIFFERENCE: real_bridge is the physical G1 RobotIO backend.
  setsid .venv/bin/python scripts/g1/real_bridge.py --robot g1 --interface "$ROBOT_INTERFACE" \
    --wait-for-command --ready-file "$RUNTIME_DIR/bridge.ready" >"$LOG_DIR/g1_bridge.log" 2>&1 & PIDS+=("$!")
  CONTROLLER_ARGS=(--policy-config "$POLICY_CONFIG" --motion "$MOTION" --motion-meta "$MOTION_META" \
    --pilot-authority 1.0 --direct-policy-targets --pilot-init-tolerance 0.50 --max-tilt-deg 180 \
    --max-joint-speed 18 --command-fifo "$RUNTIME_DIR/commands" --ready-file "$RUNTIME_DIR/controller.ready" \
    --status-file "$RUNTIME_DIR/controller.status")
  if ((NOMINAL_APPLY)); then CONTROLLER_ARGS+=(--proposal-port 5594 --pilot-log "$RECORD_OUTPUT_DIR/controller_applied.jsonl"); fi
  setsid .venv/bin/python scripts/g1/suitcase_safe_controller.py "${CONTROLLER_ARGS[@]}" \
    >"$LOG_DIR/safe_controller.log" 2>&1 & CONTROLLER_PID="$!"; PIDS+=("$CONTROLLER_PID")
else
  setsid .venv/bin/python scripts/g1/real_bridge.py --robot g1 --interface "$ROBOT_INTERFACE" --read-only \
    >"$LOG_DIR/g1_bridge.log" 2>&1 & PIDS+=("$!")
fi

# HARDWARE-ONLY DIFFERENCE: live relays provide the policy's door and panel
# mocap bodies; sim2sim gets these from MuJoCo's in-process publisher.
setsid .venv/bin/python scripts/ros2_pose_to_zmq.py --suitcase-only --suitcase-topic /door/pose \
  --suitcase-port 5561 --transform-json "$DOOR_CALIBRATION" --marker-role "$DOOR_MARKER_ROLE" \
  --suitcase-preferred-marker-source suitcase4 --startup-timeout 10 >"$LOG_DIR/door_relay.log" 2>&1 & PIDS+=("$!")
setsid .venv/bin/python scripts/ros2_pose_to_zmq.py --suitcase-only --suitcase-topic /door_panel/pose \
  --suitcase-port 5562 --transform-json "$DOOR_CALIBRATION" --marker-role "$DOOR_PANEL_MARKER_ROLE" \
  --suitcase-preferred-marker-source suitcase4 --startup-timeout 10 >"$LOG_DIR/door_panel_relay.log" 2>&1 & PIDS+=("$!")
setsid .venv/bin/python scripts/ros2_torso_to_pelvis.py --torso-topic /robot_g1/pose \
  --torso-from-marker-json "$CALIBRATION" --startup-timeout 10 >"$LOG_DIR/pelvis_fk.log" 2>&1 & PIDS+=("$!")

if ((FT_ENABLED)); then
  FT_ARGS=(--calibration "$FT_CALIBRATION" --left-host "$FT_LEFT_HOST" --left-port "$FT_LEFT_PORT" \
    --right-host "$FT_RIGHT_HOST" --right-port "$FT_RIGHT_PORT" --output-port 5580)
  if ((!FT_SKIP_TARE)); then FT_ARGS+=(--tare-on-start --tare-trigger-file "$RUNTIME_DIR/ft_tare.trigger" \
    --tare-timeout 3600 --tare-output "$RECORD_OUTPUT_DIR/ft_runtime_bias.json"); fi
  # HARDWARE-ONLY DIFFERENCE: SDK F/T sockets are physical wrist sensors.
  setsid .venv/bin/python scripts/run_hps_ft_adapter.py "${FT_ARGS[@]}" >"$LOG_DIR/ft_adapter.log" 2>&1 & PIDS+=("$!")
fi

if ((ARMED)); then
  for _ in $(seq 1 150); do [[ -f "$RUNTIME_DIR/controller.ready" ]] && break; sleep 0.1; done
  [[ -f "$RUNTIME_DIR/controller.ready" ]] || { echo "safe controller did not become ready" >&2; exit 1; }
  for _ in $(seq 1 300); do [[ -f "$RUNTIME_DIR/bridge.ready" ]] && break; sleep 0.1; done
  [[ -f "$RUNTIME_DIR/bridge.ready" ]] || { echo "G1 bridge did not become ready" >&2; exit 1; }
fi
sleep 2
.venv/bin/python scripts/check_suitcase_streams.py --duration 3 | tee "$LOG_DIR/stream_check.log"
if ((CHECK_ONLY)); then
  ((FT_ENABLED)) && .venv/bin/python scripts/run_hps_ft_adapter.py --calibration "$FT_CALIBRATION" --validate-only
  exit 0
fi
if ((!ARMED)); then echo "Read-only push_door_hand stack is running. Press Ctrl-C to stop."; while true; do sleep 1; done; fi

RUNNER_FT_ARGS=()
if ((FT_ENABLED)); then RUNNER_FT_ARGS+=(--ft-port 5580 --ft-max-age-ms 100 --ft-calibration "$FT_CALIBRATION" \
  --require-both-ft-valid --residual-mode "$RESIDUAL_MODE"); [[ "$RESIDUAL_MODE" == off ]] || RUNNER_FT_ARGS+=(--residual "$RESIDUAL_MODEL"); fi
echo "Door safe controller active. Commands: z=zero, h=hold, i=init, t=tare, s=shadow, p=pilot, q=quit"
while kill -0 "$CONTROLLER_PID" 2>/dev/null; do
  read -r -p "push-door-safe> " command || command=q
  case "$command" in
    z|zero|h|hold|i|init|q|quit|exit) echo "$command" >&3;;
    t|tare)
      ((FT_ENABLED)) || { echo "restart with --ft to tare"; continue; }
      ((FT_SKIP_TARE)) && { echo "--ft-skip-tare is active"; continue; }
      touch "$RUNTIME_DIR/ft_tare.trigger"; echo "F/T tare requested";;
    s|shadow)
      ((NOMINAL_SHADOW)) || { echo "restart with --armed --nominal-shadow"; continue; }
      STEM="$RECORD_OUTPUT_DIR/policy_shadow_$(date +%H%M%S)"
      .venv/bin/python scripts/run_hdmi_suitcase_nominal_shadow.py --upstream-root ../sim2real-hdmi-upstream \
        --policy-config "$POLICY_CONFIG" --model "$MODEL" --motion "$MOTION" --motion-meta "$MOTION_META" \
        --object-name door --object-port 5561 --aux-object-name door_panel --aux-object-port 5562 \
        --steps "$DOOR_STEPS" --rate 50 --output "$STEM.npz" "${RUNNER_FT_ARGS[@]}" \
        >"$STEM.log" 2>&1 || tail -n 40 "$STEM.log" >&2;;
    p|pilot)
      ((NOMINAL_APPLY)) || { echo "restart with --armed --nominal-apply"; continue; }
      [[ "$(cat "$RUNTIME_DIR/controller.status" 2>/dev/null || true)" == init_complete ]] || { echo "press i first"; continue; }
      STEM="$RECORD_OUTPUT_DIR/policy_apply_$(date +%H%M%S)"; rm -f "$RUNTIME_DIR"/{proposal.ready,proposal.start,motion.start,pose.status,proposal.complete,proposal.complete.ack}
      setsid .venv/bin/python scripts/run_hdmi_suitcase_nominal_shadow.py --upstream-root ../sim2real-hdmi-upstream \
        --policy-config "$POLICY_CONFIG" --model "$MODEL" --motion "$MOTION" --motion-meta "$MOTION_META" \
        --object-name door --object-port 5561 --aux-object-name door_panel --aux-object-port 5562 \
        --steps "$DOOR_STEPS" --rate 50 --reference-mode advance --max-initial-error 0.50 \
        --proposal-port 5594 --ready-file "$RUNTIME_DIR/proposal.ready" --start-file "$RUNTIME_DIR/proposal.start" \
        --motion-start-file "$RUNTIME_DIR/motion.start" --pose-status-file "$RUNTIME_DIR/pose.status" \
        --completion-file "$RUNTIME_DIR/proposal.complete" --completion-ack-file "$RUNTIME_DIR/proposal.complete.ack" \
        --output "$STEM.npz" "${RUNNER_FT_ARGS[@]}" >"$STEM.log" 2>&1 & APPLY_PID="$!"; PIDS+=("$APPLY_PID")
      for _ in $(seq 1 300); do [[ -f "$RUNTIME_DIR/proposal.ready" ]] && break; sleep 0.1; done
      [[ -f "$RUNTIME_DIR/proposal.ready" ]] || { echo "door proposal failed" >&2; tail -n 40 "$STEM.log" >&2; continue; }
      echo p >&3; touch "$RUNTIME_DIR/proposal.start"; read -r -p "Type GO to start 573-step door motion: " go
      [[ "$go" == GO ]] && touch "$RUNTIME_DIR/motion.start" || { echo h >&3; continue; }
      stop=""
      while kill -0 "$APPLY_PID" 2>/dev/null; do
        [[ -f "$RUNTIME_DIR/proposal.complete" ]] && { echo h >&3; touch "$RUNTIME_DIR/proposal.complete.ack"; break; }
        read -r -t 0.1 stop || true; [[ "$stop" == h || "$stop" == q ]] && { echo h >&3; signal_group "$APPLY_PID"; break; }
      done
      wait "$APPLY_PID" 2>/dev/null || true; echo h >&3
      [[ "$command" == q ]] && { echo q >&3; break; };;
    *) echo "Allowed commands: z h i t s p q";;
  esac
  [[ "$command" == i || "$command" == init ]] && { for _ in $(seq 1 150); do [[ "$(cat "$RUNTIME_DIR/controller.status" 2>/dev/null || true)" == init_complete ]] && break; sleep 0.1; done; }
  [[ "$command" == q || "$command" == quit || "$command" == exit ]] && { wait "$CONTROLLER_PID" 2>/dev/null || true; break; }
done
