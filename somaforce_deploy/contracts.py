"""Stable tensor and joint contracts shared by deployment backends."""
from dataclasses import dataclass
from typing import Mapping, Sequence
import numpy as np
ACTION_DIM=23
HDMI_STUDENT_COMMAND_DIM=356
HDMI_STUDENT_POLICY_DIM=249
HDMI_STUDENT_OBJECT_DIM=10
HDMI_STUDENT_LATENT_DIM=256
RESIDUAL_ACTOR_INPUT_DIM=220
PROPRIO_DIM=64
WRIST_TOKEN_SHAPE=(2,16,14)
G1_JOINT_NAMES=(
"left_hip_pitch_joint","left_hip_roll_joint","left_hip_yaw_joint","left_knee_joint","left_ankle_pitch_joint","left_ankle_roll_joint",
"right_hip_pitch_joint","right_hip_roll_joint","right_hip_yaw_joint","right_knee_joint","right_ankle_pitch_joint","right_ankle_roll_joint",
"waist_yaw_joint","waist_roll_joint","waist_pitch_joint","left_shoulder_pitch_joint","left_shoulder_roll_joint","left_shoulder_yaw_joint","left_elbow_joint",
"left_wrist_roll_joint","left_wrist_pitch_joint","left_wrist_yaw_joint","right_shoulder_pitch_joint","right_shoulder_roll_joint","right_shoulder_yaw_joint","right_elbow_joint",
"right_wrist_roll_joint","right_wrist_pitch_joint","right_wrist_yaw_joint")
ACTION_JOINT_NAMES=(
"left_hip_pitch_joint","right_hip_pitch_joint","waist_yaw_joint","left_hip_roll_joint","right_hip_roll_joint","waist_roll_joint",
"left_hip_yaw_joint","right_hip_yaw_joint","waist_pitch_joint","left_knee_joint","right_knee_joint","left_shoulder_pitch_joint",
"right_shoulder_pitch_joint","left_ankle_pitch_joint","right_ankle_pitch_joint","left_shoulder_roll_joint","right_shoulder_roll_joint","left_ankle_roll_joint",
"right_ankle_roll_joint","left_shoulder_yaw_joint","right_shoulder_yaw_joint","left_elbow_joint","right_elbow_joint")
def require_array(value, shape: Sequence[int], name: str):
    a=np.asarray(value,dtype=np.float32); expected=tuple(int(x) for x in shape)
    if a.shape!=expected: raise ValueError(f"{name} must have shape {expected}, got {a.shape}")
    if not np.isfinite(a).all(): raise ValueError(f"{name} contains non-finite values")
    return a
@dataclass(frozen=True)
class ActionContract:
    joint_names: tuple[str,...]
    action_scale: tuple[float,...]
    default_joint_pos: Mapping[str,float]
    control_hz: float=50.0
    def __post_init__(self):
        if tuple(self.joint_names)!=ACTION_JOINT_NAMES: raise ValueError("action joint order mismatch")
        if len(self.action_scale)!=ACTION_DIM or any(float(x)<=0 for x in self.action_scale): raise ValueError("action_scale must contain 23 positive values")
        if self.control_hz<=0: raise ValueError("control_hz must be positive")
def validate_joint_mapping(names):
    if tuple(names)!=ACTION_JOINT_NAMES: raise ValueError("policy joint order mismatch")
