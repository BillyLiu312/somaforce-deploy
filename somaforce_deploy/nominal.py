"""Nominal-policy interfaces for HDMI student and Sonic backends."""
from abc import ABC, abstractmethod
from typing import Any, Mapping, Protocol
import numpy as np
from .contracts import *

class InferenceModule(Protocol):
    def __call__(self, inputs: Mapping[str,np.ndarray]) -> Mapping[str,np.ndarray]: ...

class NominalPolicy(ABC):
    @abstractmethod
    def step(self, *, command, policy, object_obs=None): raise NotImplementedError

def _action_from_inference(output: Mapping[str, Any]) -> np.ndarray:
    """Resolve native HDMI ONNX output without requiring its sidecar metadata."""
    if "action" in output:
        value = output["action"]
    elif "linear_6" in output:
        # Native HDMI's one-graph exporter names the actor mean ``linear_6``
        # after TensorDict side outputs are stripped from the ONNX graph.
        value = output["linear_6"]
    else:
        candidates = [
            value
            for value in output.values()
            if np.asarray(value).shape in {(1, ACTION_DIM), (ACTION_DIM,)}
        ]
        if len(candidates) != 1:
            raise ValueError(
                "native HDMI output must expose exactly one 23-D action tensor; "
                f"found {len(candidates)} candidates"
            )
        value = candidates[0]
    action = np.asarray(value, dtype=np.float32)
    if action.shape == (ACTION_DIM,):
        action = action[None, :]
    return require_array(action, (1, ACTION_DIM), "student action")

class HDMIStudentNominal(NominalPolicy):
    name="hdmi_student"
    def __init__(self,inference: InferenceModule): self.inference=inference
    def step(self, *, command, policy, object_obs=None):
        command=require_array(command,(1,HDMI_STUDENT_COMMAND_DIM),"command")
        policy=require_array(policy,(1,HDMI_STUDENT_POLICY_DIM),"policy")
        if object_obs is None: raise ValueError("HDMI student requires object observation [1,10]")
        object_obs=require_array(object_obs,(1,HDMI_STUDENT_OBJECT_DIM),"object_obs")
        out=self.inference({"command":command,"policy":policy,"object":object_obs})
        return _action_from_inference(out)

class HDMIStudentTwoStageNominal(NominalPolicy):
    name="hdmi_student_two_stage"
    def __init__(self,adapt_ema: InferenceModule,actor_adapt: InferenceModule):
        self.adapt_ema=adapt_ema; self.actor_adapt=actor_adapt
    def step(self, *, command, policy, object_obs=None):
        command=require_array(command,(1,HDMI_STUDENT_COMMAND_DIM),"command")
        policy=require_array(policy,(1,HDMI_STUDENT_POLICY_DIM),"policy")
        if object_obs is None: raise ValueError("HDMI student requires object observation [1,10]")
        object_obs=require_array(object_obs,(1,HDMI_STUDENT_OBJECT_DIM),"object_obs")
        # ppo_roa uses CatTensors([policy, command, object]) when
        # adapt_module_input_cmd=true, which is the current student config.
        latent_out=self.adapt_ema({"policy":policy,"command":command,"object":object_obs})
        latent=require_array(latent_out.get("priv_pred",latent_out.get("latent")),(1,HDMI_STUDENT_LATENT_DIM),"student latent")
        action_out=self.actor_adapt({"command":command,"policy":policy,"priv_pred":latent,"latent":latent})
        return require_array(action_out["action"],(1,ACTION_DIM),"student action")

class SonicNominal(NominalPolicy):
    name="sonic"
    def __init__(self,inference: InferenceModule,input_builder): self.inference=inference; self.input_builder=input_builder
    def step(self, *, command, policy, object_obs=None):
        out=self.inference(self.input_builder(command=command,policy=policy))
        return require_array(out["action"],(1,ACTION_DIM),"Sonic action")
