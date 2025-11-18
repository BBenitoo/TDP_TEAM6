# zmp_multi_robot_supervisor.py
# Supervisor controller: multi-robot ZMP + convex-hull support polygon + visualization markers
from controller import Supervisor
import math, numpy as np
import collections
import traceback

# ----------------------------- CONFIG: 请根据你的 world 修改 --------------------------------
CONFIG = {
    # 每台要监控的机器人配置（填入你场景中显示的 DEF 名）
    # 每个条目键为任意 id（仅本脚本内部使用），value 为 dict:
    #   'robot_def'    : robot 根节点 DEF 名
    #   'left_foot'    : 左脚的 DEF 名（必须在 world 中存在）
    #   'right_foot'   : 右脚的 DEF 名（必须在 world 中存在）
    #   'left_force'   : 左脚 ForceSensor device 名（可选）
    #   'right_force'  : 右脚 ForceSensor device 名（可选）
    'robot_configs': {
        'blue0': {
            'robot_def': 'PLAYER_BLUE_0',
            'left_foot': 'PLAYER_BLUE_0_L_FOOT',   # <-- 请替换为真实 DEF 名
            'right_foot': 'PLAYER_BLUE_0_R_FOOT',  # <-- 请替换为真实 DEF 名
            'left_force': None,   # e.g. 'LFootForceSensor' if present
            'right_force': None
        },
        'red0': {
            'robot_def': 'PLAYER_RED_0',
            'left_foot': 'PLAYER_RED_0_L_FOOT',    # <-- 替换
            'right_foot': 'PLAYER_RED_0_R_FOOT',   # <-- 替换
            'left_force': None,
            'right_force': None
        }
    },

    # 物理/数值参数（如需请调整）
    'foot_length': 0.12,   # m, 脚本假定脚掌矩形长度
    'foot_width' : 0.06,   # m, 脚掌宽度
    'force_threshold': 5.0,     # N, 接触判定阈值（若有 force sensor）
    'com_history_seconds': 0.25, # 用于数值微分的 COM 历史窗口（秒）
    'print_every_n_steps': 10,   # 控制台打印频率（步）
    'zmp_marker_radius': 0.03,   # 可视化小球半径（米）
}
# -----------------------------------------------------------------------------------------

# ------------------------------ 几何与算法工具函数 -----------------------------------------
def axis_angle_to_rotmat(axis, angle):
    ux, uy, uz = axis
    c = math.cos(angle); s = math.sin(angle); C = 1 - c
    return np.array([
        [c + ux*ux*C, ux*uy*C - uz*s, ux*uz*C + uy*s],
        [uy*ux*C + uz*s, c + uy*uy*C, uy*uz*C - ux*s],
        [uz*ux*C - uy*s, uz*uy*C + ux*s, c + uz*uz*C]
    ])

def foot_corners_world(supervisor, def_name, foot_length, foot_width):
    """返回脚掌四角在世界坐标系下的 2D 点列表 [(x,y),...]；若节点不存在返回 []"""
    try:
        node = supervisor.getFromDef(def_name)
        if node is None:
            return []
        t = np.array(node.getField('translation').getSFVec3f())  # (x,y,z)
        rot = node.getField('rotation').getSFRotation()         # [ax,ay,az,angle]
        axis = rot[:3]; angle = rot[3]
        R = axis_angle_to_rotmat(axis, angle)
        halfL = foot_length / 2.0; halfW = foot_width / 2.0
        local_corners = [
            np.array([ halfL,  halfW, 0.0]),
            np.array([ halfL, -halfW, 0.0]),
            np.array([-halfL, -halfW, 0.0]),
            np.array([-halfL,  halfW, 0.0])
        ]
        world_xy = []
        for p in local_corners:
            wp = R.dot(p) + t
            world_xy.append((float(wp[0]), float(wp[1])))
        return world_xy
    except Exception:
        return []

def convex_hull(points):
    """Monotone chain convex hull. 输入点列表 [(x,y),...]，返回逆时针 hull 列表"""
    pts = sorted(set(points))
    if len(pts) <= 1: return pts
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return hull

def point_in_polygon(pt, poly):
    """ray casting 判点在多边形内；poly 为 [(x,y),...]"""
    if not poly:
        return False
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % n]
        # avoid division by zero
        if (y1 > y) != (y2 > y):
            xinters = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1
            if x < xinters:
                inside = not inside
    return inside
# -----------------------------------------------------------------------------------------

class MultiRobotZMPSupervisor:
    def __init__(self, sup: Supervisor, cfg: dict):
        self.sup = sup
        self.cfg = cfg
        self.timestep = int(self.sup.getBasicTimeStep())
        self.dt = self.timestep / 1000.0
        self.g = 9.81
        self.foot_length = cfg['foot_length']
        self.foot_width = cfg['foot_width']
        self.force_threshold = cfg['force_threshold']
        self.print_every = cfg['print_every_n_steps']
        # prepare per-robot data containers
        self.robots = {}  # key -> dict with state and history

        # create structure for each robot in CONFIG
        for key, rc in cfg['robot_configs'].items():
            r = {
                'robot_def': rc.get('robot_def'),
                'left_foot_def': rc.get('left_foot'),
                'right_foot_def': rc.get('right_foot'),
                'left_force_name': rc.get('left_force'),
                'right_force_name': rc.get('right_force'),
                'com_history': collections.deque(maxlen=max(3, int(round(cfg['com_history_seconds'] / self.dt)))),
                'zmp_marker_def': f"ZMP_MARKER_{rc.get('robot_def')}",
                'hull': [],
                'left_contact': False,
                'right_contact': False,
            }
            # try to bind force sensors if names provided
            if r['left_force_name']:
                try:
                    sensor = self.sup.getDevice(r['left_force_name'])
                    sensor.enable(self.timestep)
                    r['left_force_device'] = sensor
                except Exception:
                    r['left_force_device'] = None
            else:
                r['left_force_device'] = None
            if r['right_force_name']:
                try:
                    sensor = self.sup.getDevice(r['right_force_name'])
                    sensor.enable(self.timestep)
                    r['right_force_device'] = sensor
                except Exception:
                    r['right_force_device'] = None
            else:
                r['right_force_device'] = None

            # create marker in scene (if not exists)
            self._create_zmp_marker_if_needed(r['zmp_marker_def'], cfg['zmp_marker_radius'])
            self.robots[key] = r

        self._step_counter = 0
        print("[ZMP Supervisor] Initialized. timestep={} ms, dt={:.4f} s".format(self.timestep, self.dt))

    # ---------------- utility ----------------
    def _create_zmp_marker_if_needed(self, marker_def, radius):
        try:
            if self.sup.getFromDef(marker_def) is None:
                root_children = self.sup.getRoot().getField('children')
                node_str = (
                    f'Transform {{ DEF {marker_def} translation 0 0 0 '
                    f' children [ Shape {{ geometry Sphere {{ radius {radius} }} '
                    f'appearance Appearance {{ material Material {{ diffuseColor 1 0 0 }} }} }} ] }}'
                )
                root_children.importMFNodeFromString(-1, node_str)
        except Exception:
            pass

    def _update_marker_position(self, marker_def, x, y, z=0.02):
        try:
            node = self.sup.getFromDef(marker_def)
            if node:
                node.getField('translation').setSFVec3f([float(x), float(y), float(z)])
        except Exception:
            pass

    def _read_force_sensor(self, dev):
        if dev is None:
            return 0.0
        try:
            vals = dev.getValues()
            if isinstance(vals, (list, tuple)) and len(vals) >= 3:
                return abs(vals[2])
        except Exception:
            try:
                return float(dev.getValue())
            except Exception:
                return 0.0
        return 0.0

    def _robot_root_translation(self, robot_def):
        node = self.sup.getFromDef(robot_def)
        if node:
            try:
                return tuple(node.getField('translation').getSFVec3f())
            except Exception:
                pass
        try:
            t = self.sup.getSelf().getField('translation').getSFVec3f()
            return tuple(t)
        except Exception:
            return (0.0, 0.0, 0.0)

    # ---------------- COM & ZMP ----------------
    def _append_com_sample(self, robot_key):
        r = self.robots[robot_key]
        t = self.sup.getTime()
        # 简单 COM 估计：使用 robot 根节点 translation (如有更精确的 link mass info, 可在此替换)
        pos = self._robot_root_translation(r['robot_def'])
        r['com_history'].append((t, pos[0], pos[1], pos[2]))

    def _compute_com_acc(self, robot_key):
        # require 3 samples
        r = self.robots[robot_key]
        h = r['com_history']
        if len(h) < 3:
            return 0.0, 0.0, 0.0
        t0, x0, y0, z0 = h[-3]
        t1, x1, y1, z1 = h[-2]
        t2, x2, y2, z2 = h[-1]
        dt1 = t1 - t0; dt2 = t2 - t1
        if abs(dt1 - dt2) < 1e-6 and dt1 > 1e-9:
            dt = (dt1 + dt2) / 2.0
            ax = (x2 - 2.0 * x1 + x0) / (dt * dt)
            ay = (y2 - 2.0 * y1 + y0) / (dt * dt)
            az = (z2 - 2.0 * z1 + z0) / (dt * dt)
            return ax, ay, az
        # else fit quadratic
        try:
            ts = np.array([t0, t1, t2])
            xs = np.array([x0, x1, x2])
            ys = np.array([y0, y1, y2])
            zs = np.array([z0, z1, z2])
            ax = 2.0 * np.polyfit(ts, xs, 2)[0]
            ay = 2.0 * np.polyfit(ts, ys, 2)[0]
            az = 2.0 * np.polyfit(ts, zs, 2)[0]
            return float(ax), float(ay), float(az)
        except Exception:
            return 0.0, 0.0, 0.0

    def calculate_zmp_for_robot(self, robot_key):
        self._append_com_sample(robot_key)
        r = self.robots[robot_key]
        if not r['com_history']:
            return (0.0, 0.0), (0.0, 0.0, 0.0)
        _, cx, cy, cz = r['com_history'][-1]
        ax, ay, az = self._compute_com_acc(robot_key)
        z_com = cz if abs(cz) >= 0.01 else 0.01
        x_zmp = cx - (z_com / self.g) * ax
        y_zmp = cy - (z_com / self.g) * ay
        return (float(x_zmp), float(y_zmp)), (float(cx), float(cy), float(cz))

    # ---------------- support polygon from foot corners + convex hull ----------------
    def update_support_polygon_for_robot(self, robot_key):
        r = self.robots[robot_key]
        all_points = []

        # left foot
        lf = r['left_foot_def']
        if lf:
            corners = foot_corners_world(self.sup, lf, self.foot_length, self.foot_width)
            # 如果存在 force sensor，使用其信号判定是否接触（否则默认接触）
            left_contact = True
            if r['left_force_device'] is not None:
                val = self._read_force_sensor(r['left_force_device'])
                left_contact = val > self.force_threshold
            r['left_contact'] = left_contact
            if left_contact:
                all_points.extend(corners)

        # right foot
        rf = r['right_foot_def']
        if rf:
            corners = foot_corners_world(self.sup, rf, self.foot_length, self.foot_width)
            right_contact = True
            if r['right_force_device'] is not None:
                val = self._read_force_sensor(r['right_force_device'])
                right_contact = val > self.force_threshold
            r['right_contact'] = right_contact
            if right_contact:
                all_points.extend(corners)

        if not all_points:
            r['hull'] = []
            return []

        hull = convex_hull(all_points)
        r['hull'] = hull
        return hull

    # ---------------- stability test ----------------
    def check_stability_for_robot(self, robot_key, zmp):
        r = self.robots[robot_key]
        hull = r.get('hull', [])
        stable = point_in_polygon(zmp, hull) if hull else False
        return stable

    # ---------------- main loop ----------------
    def run(self):
        print("[ZMP Supervisor] Running main loop...")
        while self.sup.step(self.timestep) != -1:
            try:
                self._step_counter += 1
                # 对每个 robot 做处理
                for key in self.robots.keys():
                    zmp, com = self.calculate_zmp_for_robot(key)
                    hull = self.update_support_polygon_for_robot(key)
                    stable = self.check_stability_for_robot(key, zmp)
                    # update visualization marker
                    self._update_marker_position(self.robots[key]['zmp_marker_def'], zmp[0], zmp[1], 0.02)
                    # periodic print
                    if (self._step_counter % self.print_every) == 0:
                        cx, cy, cz = com
                        print("-" * 70)
                        print(f"[{key}] t={self.sup.getTime():.3f}s robot_def={self.robots[key]['robot_def']}")
                        print(f"  COM = x:{cx:.3f}, y:{cy:.3f}, z:{cz:.3f} m")
                        print(f"  ZMP = x:{zmp[0]:.4f}, y:{zmp[1]:.4f} m -> {'STABLE' if stable else 'UNSTABLE'}")
                        print(f"  Support hull pts: {len(hull)}")
                        print(f"  Contacts L/R: {self.robots[key]['left_contact']}/{self.robots[key]['right_contact']}")
                        print("-" * 70)
            except Exception:
                print("[ZMP Supervisor] Exception in main loop:")
                traceback.print_exc()
                # continue looping even if one step crashed
# -----------------------------------------------------------------------------------------

if __name__ == "__main__":
    sup = Supervisor()
    controller = MultiRobotZMPSupervisor(sup, CONFIG)
    controller.run()
