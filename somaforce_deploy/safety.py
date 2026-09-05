"""Deployment-side watchdog checks."""
import time
import numpy as np
class Watchdog:
    def __init__(self,state_timeout_s=.10,ft_timeout_s=.10): self.state_timeout_s=float(state_timeout_s); self.ft_timeout_s=float(ft_timeout_s)
    def require_fresh(self,*,state_timestamp_ns,ft_timestamp_ns,now_ns=None):
        now=time.time_ns() if now_ns is None else int(now_ns)
        if (now-int(state_timestamp_ns))/1e9>self.state_timeout_s: raise RuntimeError("stale G1 state")
        if (now-int(ft_timestamp_ns))/1e9>self.ft_timeout_s: raise RuntimeError("stale wrist F/T state")
    @staticmethod
    def require_finite_action(action):
        if not np.isfinite(np.asarray(action)).all(): raise RuntimeError("policy action is non-finite")
