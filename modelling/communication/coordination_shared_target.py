# coordination/shared_target.py
import time
import numpy as np

class SharedTargetFusion:
    def __init__(self):
        # keep map of last observations: robot -> (pos, vel, conf, t)
        self.obs = {}

    def add_observation(self, robot_id, pos, vel, conf, timestamp=None):
        self.obs[robot_id] = {"pos":pos, "vel":vel, "conf":max(0.0,min(1.0,conf)), "t": timestamp or time.time()}

    def get_fused(self):
        # weighted average by confidence and recency
        now = time.time()
        num = np.array([0.0,0.0])
        den = 0.0
        for r,v in self.obs.items():
            age = now - v['t']
            age_w = np.exp(-age/1.0)  # older obs decay
            w = v['conf'] * age_w
            num += w * np.array(v['pos'])
            den += w
        if den <= 1e-6:
            return None
        fused = tuple((num/den).tolist())
        return fused

    def cleanup(self, max_age=2.0):
        now = time.time()
        to_delete = [r for r,v in self.obs.items() if now - v['t'] > max_age]
        for k in to_delete:
            del self.obs[k]
