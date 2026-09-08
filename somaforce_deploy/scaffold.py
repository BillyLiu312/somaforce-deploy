"""Simulation-only Cross scaffold baseline adapter for MuJoCo."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch

BODY_NAMES = ("pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "left_hip_yaw_link", "right_hip_yaw_link", "torso_link", "left_knee_link", "right_knee_link", "left_shoulder_pitch_link", "right_shoulder_pitch_link", "left_ankle_roll_link", "right_ankle_roll_link", "left_elbow_link", "right_elbow_link", "left_wrist_yaw_link", "right_wrist_yaw_link")

class ScaffoldNominal:
    def __init__(self, evaluator, artifact_dir: str | Path):
        from somaforce_cross.scaffold.pretrained_hdmi import (
            FrozenHDMITeacherPolicy, HDMIObservationBatch, get_hdmi_task_spec,
        )
        self.evaluator = evaluator
        self.artifact_dir = Path(artifact_dir)
        manifest = json.loads((self.artifact_dir / "manifest.json").read_text())
        self.task_spec = get_hdmi_task_spec("push_box")
        self.policy = FrozenHDMITeacherPolicy(self.task_spec.observation_dims, self.task_spec.network)
        payload = torch.load(self.artifact_dir / manifest["files"]["policy"]["path"], map_location="cpu", weights_only=True)
        self.policy.load_state_dict(payload["state_dict"], strict=True)
        norm = np.load(self.artifact_dir / "normalization.npz", allow_pickle=False)
        for name in ("command", "policy", "object", "privileged"):
            getattr(self.policy, f"{name}_mean").copy_(torch.from_numpy(norm[f"{name}_mean"]))
            getattr(self.policy, f"{name}_scale").copy_(torch.from_numpy(norm[f"{name}_scale"]))
        self.policy.eval()
        self._batch_cls = HDMIObservationBatch

    def step(self, *, command, policy, object_obs=None, scaffold_observation=None):
        if scaffold_observation is None:
            raise ValueError("scaffold baseline requires scaffold_observation")
        with torch.inference_mode():
            observation = self._batch_cls(**{
                key: torch.from_numpy(value.astype(np.float32))
                for key, value in scaffold_observation.items()
            })
            action = self.policy(observation).numpy().astype(np.float32)
        return action


def _quat_conj(q):
    q = np.asarray(q, dtype=np.float64).copy(); q[..., 1:] *= -1; return q

def _quat_mul(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0); bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), axis=-1)

def _quat_apply_inverse(q, v):
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    matrix = _quat_matrix(q)
    matrix = np.swapaxes(matrix, -1, -2)
    while matrix.ndim < v.ndim + 1:
        matrix = np.expand_dims(matrix, axis=-3)
    matrix = np.broadcast_to(matrix, v.shape[:-1] + (3, 3))
    return np.einsum("...ij,...j->...i", matrix, v)

def _quat_matrix(q):
    w,x,y,z = np.moveaxis(np.asarray(q, dtype=np.float64), -1, 0)
    return np.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                     2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                     2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),axis=-1).reshape(np.asarray(q).shape[:-1]+(3,3))

def _yaw_quat(q):
    q=np.asarray(q,dtype=np.float64); w,x,y,z=np.moveaxis(q,-1,0); yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)); h=yaw*.5
    return np.stack((np.cos(h),np.zeros_like(h),np.zeros_like(h),np.sin(h)),axis=-1)

def build_scaffold_observation(evaluator, step: int) -> dict[str, np.ndarray]:
    motion = evaluator.motion; names=evaluator.motion_body_names; robot=evaluator.model; data=evaluator.data
    root_idx=names.index("pelvis"); box_idx=names.index("box")
    root_pos=np.asarray(data.qpos[evaluator.root_qpos_adr:evaluator.root_qpos_adr+3],dtype=np.float64)
    root_quat=np.asarray(data.qpos[evaluator.root_qpos_adr+3:evaluator.root_qpos_adr+7],dtype=np.float64)
    root_yaw=_yaw_quat(root_quat)
    robot_body_ids=[evaluator._body_id(name) for name in BODY_NAMES]
    robot_pos=np.asarray(data.xpos[robot_body_ids]); robot_quat=np.asarray(data.xquat[robot_body_ids])
    robot_local=_quat_apply_inverse(root_yaw, robot_pos-np.array([root_pos[0],root_pos[1],0.0]))
    robot_quat_local=_quat_mul(_quat_conj(root_yaw),robot_quat)
    robot_lin=np.asarray(data.cvel[robot_body_ids,3:6]); robot_ang=np.asarray(data.cvel[robot_body_ids,:3])
    robot_lin_local=_quat_apply_inverse(root_quat,robot_lin); robot_ang_local=_quat_apply_inverse(root_quat,robot_ang)
    frame_ids=[min(step+o,evaluator.motion_length-1) for o in (1,2,8,16,32)]
    ref_pos=np.stack([motion["body_pos_w"][i,[names.index(n) for n in BODY_NAMES]] for i in frame_ids])
    ref_quat=np.stack([motion["body_quat_w"][i,[names.index(n) for n in BODY_NAMES]] for i in frame_ids])
    ref_root_pos=np.stack([motion["body_pos_w"][i,root_idx] for i in frame_ids])
    ref_root_quat=np.stack([motion["body_quat_w"][i,root_idx] for i in frame_ids])
    ref_root_yaw=_yaw_quat(ref_root_quat)
    ref_local=_quat_apply_inverse(ref_root_yaw,ref_pos-np.concatenate((ref_root_pos[:,:2],np.zeros((len(frame_ids),1))),axis=-1)[:,None])
    ref_quat_local=_quat_mul(_quat_conj(ref_root_yaw)[:,None],ref_quat)
    ref_lin=motion["body_lin_vel_w"][frame_ids][:, [names.index(n) for n in BODY_NAMES]]
    ref_ang=motion["body_ang_vel_w"][frame_ids][:, [names.index(n) for n in BODY_NAMES]]
    ref_lin_local=_quat_apply_inverse(ref_root_yaw[:,None],ref_lin)
    ref_ang_local=_quat_apply_inverse(ref_root_yaw[:,None],ref_ang)
    diff_ori=_quat_mul(_quat_conj(robot_quat_local)[None],ref_quat_local)
    ref_root_pos_b=_quat_apply_inverse(root_quat,ref_root_pos-root_pos)
    ref_root_ori_b=_quat_matrix(_quat_mul(_quat_conj(root_quat),ref_root_quat))[:,:2]
    ankle_ids=[evaluator._body_id(n) for n in ("left_ankle_roll_link","right_ankle_roll_link")]
    ankle_pos=_quat_apply_inverse(root_yaw,np.asarray(data.xpos[ankle_ids])-np.array([root_pos[0],root_pos[1],0.0]))
    ankle_vel=_quat_apply_inverse(root_quat,np.asarray(data.cvel[ankle_ids,3:6]))
    heights=np.asarray([data.xpos[evaluator._body_id(n),2] for n in ("left_ankle_roll_link","right_ankle_roll_link","pelvis","torso_link")])
    robot_root_lin=_quat_apply_inverse(root_quat,np.asarray(data.qvel[evaluator.root_qvel_adr:evaluator.root_qvel_adr+3]))
    object_body=evaluator._body_id("box"); object_pos=np.asarray(data.xpos[object_body]); object_quat=np.asarray(data.xquat[object_body])
    object_pos_b=_quat_apply_inverse(root_quat,object_pos-root_pos)
    object_ori=_quat_matrix(_quat_mul(_quat_conj(root_quat),object_quat))
    ref_box_pos=motion["body_pos_w"][frame_ids,box_idx]; ref_box_quat=motion["body_quat_w"][frame_ids,box_idx]
    diff_obj_pos=_quat_apply_inverse(object_quat,ref_box_pos-object_pos)
    diff_obj_ori=_quat_matrix(_quat_mul(_quat_conj(object_quat),ref_box_quat))
    contact_targets=object_pos[None]+np.asarray([[0,-.2,.8],[0,.2,.8]])
    wrist_ids=[evaluator._body_id("left_wrist_yaw_link"),evaluator._body_id("right_wrist_yaw_link")]
    eef_pos=np.asarray(data.xpos[wrist_ids])+np.asarray([[.1,0,0],[.1,0,0]])
    diff_contact=_quat_apply_inverse(root_quat,contact_targets-eef_pos)
    action=np.asarray(evaluator.action_history[0],dtype=np.float32)
    torque=np.asarray([data.actuator_force[evaluator.act_adrs[name]] for name in evaluator.robot_joint_names],dtype=np.float32)
    joint_hist=np.stack(tuple(evaluator.joint_history),axis=0)
    root_hist=np.stack(tuple(evaluator.root_ang_history),axis=0)
    grav_hist=np.stack(tuple(evaluator.gravity_history),axis=0)
    action_indices=[evaluator.motion_joint_names.index(n) for n in evaluator.task_spec_action_names]
    ref_joint=motion["joint_pos"][frame_ids][:,action_indices]
    default_action=np.asarray([evaluator.default_by_joint[n] for n in evaluator.task_spec_action_names],dtype=np.float32)
    scale=np.asarray([next(value for key,value in {"hip_yaw":0.55,"hip_roll":0.35,"hip_pitch":0.55,"knee":0.35,"ankle_pitch":0.44,"ankle_roll":0.44,"waist_roll":0.44,"waist_pitch":0.44,"waist_yaw":0.55,"shoulder_pitch":0.44,"shoulder_roll":0.44,"shoulder_yaw":0.44,"elbow":0.44}.items() if key in n) for n in evaluator.task_spec_action_names],dtype=np.float32)
    ref_action=(motion["joint_pos"][min(step,evaluator.motion_length-1),action_indices]-default_action)/scale
    privileged=np.concatenate((root_hist.reshape(-1),grav_hist.reshape(-1),joint_hist.reshape(-1),
      ref_root_pos_b.reshape(-1),ref_root_ori_b.reshape(-1),(ref_local-robot_local[None]).reshape(-1),
      _quat_matrix(diff_ori)[:,:,:,:2].reshape(-1),(ref_lin_local-robot_lin_local[None]).reshape(-1),
      (ref_ang_local-robot_ang_local[None]).reshape(-1),robot_root_lin.reshape(-1),ankle_pos.reshape(-1),ankle_vel.reshape(-1),heights.reshape(-1),action,torque,
      object_pos_b,object_ori.reshape(-1),diff_obj_pos.reshape(-1),diff_obj_ori.reshape(-1),motion["object_contact"][frame_ids].reshape(-1),diff_contact.reshape(-1))).astype(np.float32)
    if privileged.size != 1714: raise ValueError(f"scaffold privileged width mismatch: {privileged.size}")
    return {"command":evaluator._reference_command(step),"policy":evaluator._policy_observation(),"object":evaluator._object_observation(),"privileged":privileged[None],"reference_action":ref_action[None].astype(np.float32)}
