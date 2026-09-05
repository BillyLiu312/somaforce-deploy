"""Nominal-policy interfaces for HDMI student and Sonic backends."""
from abc import ABC, abstractmethod
from typing import Mapping, Protocol
import numpy as np
from .contracts import *
class InferenceModule(Protocol):
    def __call__(self, inputs: Mapping[str,np.ndarray])->Mapping[str,np.ndarray]: ...
class NominalPolicy(ABC):
    @abstractmethod
    def step(self, *, command, policy, object_obs=None): raise NotImplementedError
class HDMIStudentNominal(NominalPolicy):
    name="hdmi_student"
    def __init__(self,inference): self.inference=inference
    def step(self, *, command, policy, object_obs=None):
        command=require_array(command,(1,HDMI_STUDENT_COMMAND_DIM),"command")
        policy=require_array(policy,(1,HDMI_STUDENT_POLICY_DIM),"policy")
        if object_obs is None: raise ValueError("HDMI student requires object observation [1,10]")
        object_obs=require_array(object_obs,(1,HDMI_STUDENT_OBJECT_DIM),"object_obs")
        out=self.inference({"command":command,"policy":policy,"object":object_obs})
        return require_array(out["action"],(1,ACTION_DIM),"student action")
class SonicNominal(NominalPolicy):
    name="sonic"
    def __init__(self,inference,input_builder): self.inference=inference; self.input_builder=input_builder
    def step(self, *, command, policy, object_obs=None):
        out=self.inference(self.input_builder(command=command,policy=policy))
        return require_array(out["action"],(1,ACTION_DIM),"Sonic action")
