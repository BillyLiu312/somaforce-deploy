"""Cross residual inference and guarded normalized-action composition."""
from typing import Mapping, Protocol
import numpy as np
from .contracts import *
class InferenceModule(Protocol):
    def __call__(self, inputs: Mapping[str,np.ndarray])->Mapping[str,np.ndarray]: ...
class CrossResidual:
    def __init__(self,inference): self.inference=inference
    def step(self, *, wrist_tokens, proprio, a_nom_history, previous_a_total):
        inputs={"wrist_tokens":require_array(wrist_tokens,(1,2,16,14),"wrist_tokens"),"proprio":require_array(proprio,(1,64),"proprio"),"a_nom_history":require_array(a_nom_history,(1,3,23),"a_nom_history"),"previous_a_total":require_array(previous_a_total,(1,23),"previous_a_total")}
        out=self.inference(inputs); key="residual" if "residual" in out else "action"
        return require_array(out[key],(1,23),"residual action")
def compose_action(a_nom,delta_a,*,authority=1.0,contact_gain=1.0,action_limit=1.0):
    nominal=require_array(a_nom,(1,23),"a_nom"); residual=require_array(delta_a,(1,23),"delta_a")
    return np.clip(nominal+residual*np.asarray(authority,dtype=np.float32)*np.asarray(contact_gain,dtype=np.float32),-float(action_limit),float(action_limit)).astype(np.float32)
