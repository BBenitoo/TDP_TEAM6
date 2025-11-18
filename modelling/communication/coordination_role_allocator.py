# coordination/role_allocator.py
import time
from collections import defaultdict

HYSTERESIS_SEC = 0.8  # 防止频繁切换

class RoleAllocator:
    def __init__(self, my_name, num_chasers=1, num_support=1):
        self.my_name = my_name
        self.num_chasers = num_chasers
        self.num_support = num_support
        self.last_assignments = {}
        self.last_change_time = 0

    def decide(self, heartbeats):
        """
        heartbeats: dict name -> {'ball_dist':..., 'timestamp':...}
        returns assignments dict
        """
        entries = [(v['ball_dist'], name) for name,v in heartbeats.items()]
        # ensure local present
        if self.my_name not in [n for _,n in entries]:
            entries.append((heartbeats.get(self.my_name, {}).get('ball_dist', float('inf')), self.my_name))
        entries.sort(key=lambda x: (x[0], x[1]))
        assignments = {}
        idx = 0
        for role, count in [('CHASER', self.num_chasers), ('SUPPORT', self.num_support)]:
            for i in range(count):
                if idx < len(entries):
                    assignments[entries[idx][1]] = role
                    idx += 1
        for j in range(idx, len(entries)):
            assignments[entries[j][1]] = 'DEFENDER'
        # hysteresis: if assignments differ from last and last change was recent, keep old
        now = time.time()
        if self.last_assignments and (now - self.last_change_time) < HYSTERESIS_SEC:
            # respect old assignment for robots in conflict
            for name in assignments:
                if self.last_assignments.get(name) and self.last_assignments[name] != assignments[name]:
                    assignments[name] = self.last_assignments[name]
        else:
            self.last_change_time = now
            self.last_assignments = assignments.copy()
        return assignments
