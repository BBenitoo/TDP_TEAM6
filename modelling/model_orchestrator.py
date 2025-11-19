# controllers/model_orchestrator/model_orchestrator.py
# -*- coding: utf-8 -*-
from controller import Supervisor
import os, sys, math
import numpy as np
from collections import deque

# ---------- Import path so we can import from project root ----------
_THIS_DIR = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------- Import FK (your six-matrix right-multiply implementation) ----------
try:
    from models.fk_nao_dh import fk as fk_right_leg
except Exception as e:
    fk_right_leg = None
    print("[orchestrator] ERROR importing models.fk_nao_dh.fk:", e)
    print("[orchestrator] Expecting file at:",
          os.path.join(_PROJECT_ROOT, "models", "fk_nao_dh.py"))

# ---------- Import COM / ZMP / Energy / Traj error modules ----------
from com import compute_com
from zmp_position import zmp_from_com_traj
from trajectory_error import compute_traj_error
from nao_energy_controller import energy_over_trajectory

# ---------- NAO right leg motor / joint names ----------
RIGHT_LEG_MOTOR_NAMES = [
    "RHipYawPitch",  # Hinge2 axis-1
    "RHipRoll",
    "RHipPitch",
    "RKneePitch",
    "RAnklePitch",
    "RAnkleRoll",
]

# 简化的 COM 参数（两个 link：pelvis / foot）
COM_M_LIST = np.array([3.0, 1.0], dtype=float)
COM_R_LIST = np.array([
    [0.0, 0.0, 0.0],   # pelvis COM in pelvis frame
    [0.0, 0.0, 0.0],   # foot COM in foot frame
], dtype=float)


# ---------- Helpers ----------
def T_to_p_R(T):
    return T[:3, 3].copy(), T[:3, :3].copy()


def rot_diff_angle_deg(R1, R2):
    R_rel = R1.T @ R2
    tr = np.trace(R_rel)
    c = (tr - 1.0) / 2.0
    c = max(-1.0, min(1.0, c))
    return math.degrees(math.acos(c))


def traverse_all(root):
    """遍历整个场景树，只通过真正的 Node-type 字段（children / endPoint / device）。"""
    stack = [root]
    seen = set()
    while stack:
        n = stack.pop()
        if n is None:
            continue
        if n in seen:
            continue
        seen.add(n)
        yield n
        for fname in ("children", "endPoint", "device"):
            fld = n.getField(fname)
            if not fld:
                continue
            t = fld.getTypeName()
            if t == "MFNode":
                for i in range(fld.getCount()):
                    stack.append(fld.getMFNode(i))
            elif t == "SFNode":
                stack.append(fld.getSFNode())
            # 其它类型（SFString 等）全部忽略


def collect_joints_and_motors(sup):
    """Return:
       joints: list of {'node','def','type','dev_names'} for HingeJoint/Hinge2Joint
       motors: list of {'node','name','parent_joint'}
       solids: list of {'node','def','name'}
    """
    joints, motors, solids = [], [], []
    for n in traverse_all(sup.getRoot()):
        t = n.getTypeName()
        if t in ("HingeJoint", "Hinge2Joint"):
            d = n.getDef() or ""
            names = []
            dev = n.getField("device")
            if dev:
                if dev.getTypeName() == "MFNode":
                    for i in range(dev.getCount()):
                        dn = dev.getMFNode(i)
                        if dn and dn.getTypeName() == "RotationalMotor":
                            names.append(dn.getField("name").getSFString())
                elif dev.getTypeName() == "SFNode":
                    dn = dev.getSFNode()
                    if dn and dn.getTypeName() == "RotationalMotor":
                        names.append(dn.getField("name").getSFString())
            joints.append({
                "node": n,
                "def": d,
                "type": t,
                "dev_names": names
            })
        elif t == "RotationalMotor":
            name = n.getField("name").getSFString()
            motors.append({"node": n, "name": name, "parent_joint": None})
        elif t == "Solid":
            solids.append({
                "node": n,
                "def": n.getDef() or "",
                "name": n.getField("name").getSFString() if n.getField("name") else ""
            })
    return joints, motors, solids


def robust_find_right_leg(sup):
    joints, motors, solids = collect_joints_and_motors(sup)
    print(f"[orchestrator] fallback discovery: joints={len(joints)}, "
          f"motors={len(motors)}, solids={len(solids)}")

    mapping = {nm: None for nm in RIGHT_LEG_MOTOR_NAMES}
    for j in joints:
        names = j["dev_names"]
        for idx, dn in enumerate(names, start=1):  # idx=1 or 2 for Hinge2
            if dn in mapping and mapping[dn] is None:
                axis = 1 if j["type"] == "HingeJoint" else idx
                mapping[dn] = (j["node"], axis)
    nodes = [mapping[nm][0] if mapping[nm] else None for nm in RIGHT_LEG_MOTOR_NAMES]
    axes = [mapping[nm][1] if mapping[nm] else 1 for nm in RIGHT_LEG_MOTOR_NAMES]
    return nodes, axes


def find_pelvis_from_hip(sup, hip_node):
    """优先用 hip.parent；找不到就全局搜 name/def 里带 torso/body/pelvis 的 Solid。"""
    if hip_node:
        pf = hip_node.getField("parent")
        if pf:
            node = pf.getSFNode()
            if node and node.getTypeName() == "Solid":
                return node
    _, _, solids = collect_joints_and_motors(sup)
    for s in solids:
        nm = (s["name"] or s["def"] or "").lower()
        if any(k in nm for k in ("torso", "pelvis", "body", "hip", "abdomen")):
            return s["node"]
    return None


def _find_first_solid(node):
    if not node:
        return None
    if node.getTypeName() == "Solid":
        return node
    for fname in ("children", "endPoint", "device"):
        fld = node.getField(fname)
        if not fld:
            continue
        if fld.getTypeName() == "MFNode":
            for i in range(fld.getCount()):
                s = _find_first_solid(fld.getMFNode(i))
                if s:
                    return s
        elif fld.getTypeName() == "SFNode":
            s = _find_first_solid(fld.getSFNode())
            if s:
                return s
    return None


def find_right_foot_from_last_joint(sup, last_joint):
    """优先从 last_joint.endPoint 往下找；找不到就全局搜名字里带 rfoot/right foot/ankle 的 Solid。"""
    if last_joint:
        ep = last_joint.getField("endPoint")
        if ep:
            node = ep.getSFNode()
            s = _find_first_solid(node)
            if s:
                return s
    _, _, solids = collect_joints_and_motors(sup)
    for s in solids:
        nm = (s["name"] or s["def"] or "").lower()
        if any(k in nm for k in ("rfoot", "right foot", "ankle", "foot right")):
            return s["node"]
    return None


# ---------- Main loop ----------
def main():
    sup = Supervisor()
    dt_ms = int(sup.getBasicTimeStep())
    dt = dt_ms / 1000.0

    # Robust right-leg discovery
    joints, axes = robust_find_right_leg(sup)
    for nm, nd, ax in zip(RIGHT_LEG_MOTOR_NAMES, joints, axes):
        if nd is None:
            print(f"[orchestrator] joint {nm}: NOT FOUND")
        else:
            print(f"[orchestrator] joint {nm}: FOUND (axis={ax})")

    # Pelvis & Foot
    pelvis_solid = find_pelvis_from_hip(sup, joints[0])
    print("[orchestrator] pelvis Solid:", "FOUND" if pelvis_solid else "NOT FOUND")

    foot_node = find_right_foot_from_last_joint(sup, joints[-1])
    print("[orchestrator] right foot Solid:", "FOUND" if foot_node else "NOT FOUND")

    if fk_right_leg is None:
        print("[orchestrator] FK function import failed. Ensure models/fk_nao_dh.py exists.")

    label_fk = 11          # 右上角：FK 误差
    label_comzmp = 12      # 左下角：COM & ZMP
    label_energy = 13      # 右下角：Energy

    calibrated = False
    T_pelvis_hip = np.eye(4)

    # 数据记录：FK / WB / q / COM / τ / 时间 / 能量
    q_traj = []
    p_fk_traj = []
    p_wb_traj = []
    com_traj = []
    tau_traj = []
    time_traj = []
    energy_traj = []

    # COM / ZMP 历史缓存 & 打印频率控制
    com_hist = deque(maxlen=3)
    t_hist = deque(maxlen=3)
    last_print_time = -1.0

    print("[orchestrator] Started main loop")

    while sup.step(dt_ms) != -1:
        t = sup.getTime()

        # -------- 读取右腿 6 关节角 q --------
        q = []
        ok = True
        for nd, ax in zip(joints, axes):
            if nd is None:
                ok = False
                q.append(0.0)
                continue
            tname = nd.getTypeName()
            if tname == "Hinge2Joint" and ax == 2:
                pf = nd.getField("position2")
            else:
                pf = nd.getField("position")
            if not pf:
                ok = False
                q.append(0.0)
                continue
            angle = pf.getSFFloat()
            q.append(angle)

        if not ok or pelvis_solid is None or foot_node is None or fk_right_leg is None:
            sup.setLabel(label_fk, "FK not ready (missing joints/FK/pelvis/foot).",
                         0.02, 0.92, 0.06, 0xFF0000, 0)
            continue

        q = np.array(q, dtype=float)

        # -------- Webots 中 pelvis & foot 的齐次变换 --------
        pelvis_pose = pelvis_solid.getPosition()
        pelvis_rot = pelvis_solid.getOrientation()  # 3x3 row-major
        T_w_pelvis = np.eye(4)
        T_w_pelvis[:3, :3] = np.array(pelvis_rot).reshape((3, 3))
        T_w_pelvis[:3, 3] = np.array(pelvis_pose)

        foot_pos = foot_node.getPosition()
        foot_rot = foot_node.getOrientation()
        T_w_foot = np.eye(4)
        T_w_foot[:3, :3] = np.array(foot_rot).reshape((3, 3))
        T_w_foot[:3, 3] = np.array(foot_pos)

        # -------- FK：hip->foot 齐次变换（你自己的六矩阵 FK）--------
        T_hip_foot = fk_right_leg(q)

        # -------- 标定：根据当前姿态求 T_pelvis_hip --------
        if not calibrated:
            # T_w_foot = T_w_pelvis * T_pelvis_hip * T_hip_foot
            # => T_pelvis_hip = inv(T_w_pelvis) * T_w_foot * inv(T_hip_foot)
            T_pelvis_hip = np.linalg.inv(T_w_pelvis) @ T_w_foot @ np.linalg.inv(T_hip_foot)
            calibrated = True
            print("[orchestrator] Calibration done.")
            p_fk_cal, _ = T_to_p_R(T_w_pelvis @ T_pelvis_hip @ T_hip_foot)
            p_wb_cal, _ = T_to_p_R(T_w_foot)
            pos_err_cal = np.linalg.norm(p_fk_cal - p_wb_cal)
            ang_err_cal = rot_diff_angle_deg(
                (T_w_pelvis @ T_pelvis_hip @ T_hip_foot)[:3, :3],
                T_w_foot[:3, :3]
            )
            print(f"[orchestrator] |p_err|={pos_err_cal*1000:.1f} mm, "
                  f"ang_err={ang_err_cal:.2f} deg")
            print(f"FK p=({p_fk_cal[0]:+.3f},{p_fk_cal[1]:+.3f},{p_fk_cal[2]:+.3f})  "
                  f"WB p=({p_wb_cal[0]:+.3f},{p_wb_cal[1]:+.3f},{p_wb_cal[2]:+.3f})")

        # -------- 用标定后的 T_pelvis_hip 预测足底位置 --------
        T_w_foot_pred = T_w_pelvis @ T_pelvis_hip @ T_hip_foot

        # -------- 比较 FK vs Webots --------
        p_fk, R_fk = T_to_p_R(T_w_foot_pred)
        p_wb, R_wb = T_to_p_R(T_w_foot)
        pos_err = np.linalg.norm(p_fk - p_wb)
        ang_err = rot_diff_angle_deg(R_fk, R_wb)

        msg_fk = (f"[orchestrator] |p_err|={pos_err*1000:.1f} mm, ang_err={ang_err:.2f} deg\n"
                  f"FK p=({p_fk[0]:+.3f},{p_fk[1]:+.3f},{p_fk[2]:+.3f})  "
                  f"WB p=({p_wb[0]:+.3f},{p_wb[1]:+.3f},{p_wb[2]:+.3f})")
        # 降低终端打印频率：每 0.05 秒打印一次
        if last_print_time < 0 or t - last_print_time >= 0.05:
            print(msg_fk)
            last_print_time = t
        sup.setLabel(label_fk, msg_fk, 0.02, 0.92, 0.06, 0xFFFFFF, 0)

        # -------- COM：用 pelvis + 右脚 两个 link 做一个简单示例 --------
        A_list = [T_w_pelvis, T_w_foot]
        com_w = compute_com(A_list, COM_M_LIST, COM_R_LIST)
        com_traj.append(com_w)

        # -------- 实时 ZMP 估计（简单版）+ 左下角显示 --------
        com_hist.append(com_w)
        t_hist.append(t)

        if len(com_hist) == 3:
            c0, c1, c2 = com_hist
            t0, t1, t2 = t_hist
            dt_total = t2 - t0
            if dt_total <= 0:
                dt_local = dt
            else:
                dt_local = dt_total / 2.0

            # 中心差分估计 COM 加速度
            com_acc = (c2 - 2.0 * c1 + c0) / (dt_local ** 2)
            z_c = c1[2]
            g = 9.81 if z_c > 0.0 else 9.81

            zmp_x = c1[0] - z_c / g * com_acc[0]
            zmp_y = c1[1] - z_c / g * com_acc[1]

            msg_comzmp = (
                f"COM=({c1[0]:+.3f},{c1[1]:+.3f},{c1[2]:+.3f})\n"
                f"ZMP=({zmp_x:+.3f},{zmp_y:+.3f})"
            )
            # 左下角显示
            sup.setLabel(label_comzmp, msg_comzmp, 0.02, 0.05, 0.06, 0x00FF00, 0)

        # -------- 实时能量（伪力矩：与关节速度成正比） --------
        if len(q_traj) >= 1:
            q_dot = (q - q_traj[-1]) / dt
        else:
            q_dot = np.zeros_like(q)

        # 这里没有办法从 Supervisor 读取 NAO 的真实 torque，
        # 所以用一个“伪力矩”模型：tau_i ∝ q_dot_i
        # 这样动作越快，能量消耗越大，用于比较不同动作的相对能量。
        alpha = 1.0  # 比例系数（可以调节）
        tau = alpha * q_dot

        # 功率 & 累积能量（取绝对值更直观）
        P_inst = float(np.dot(tau, q_dot))
        P_inst_abs = abs(P_inst)

        if len(energy_traj) == 0:
            E_cum = P_inst_abs * dt
        else:
            E_cum = energy_traj[-1] + P_inst_abs * dt

        energy_traj.append(E_cum)
        tau_traj.append(tau)

        msg_energy = f"Energy≈{E_cum:.3f} (relative)"
        # 右下角显示能量
        sup.setLabel(label_energy, msg_energy, 0.80, 0.05, 0.06, 0xFFA500, 0)

        # -------- 记录轨迹数据 --------
        q_traj.append(q)
        p_fk_traj.append(p_fk)
        p_wb_traj.append(p_wb)
        time_traj.append(t)

    # ========== 仿真循环结束，做离线分析 ==========
    print("[orchestrator] Simulation finished, post-processing...")

    if len(q_traj) < 2:
        print("[orchestrator] Not enough samples for analysis.")
        return

    q_arr = np.vstack(q_traj)          # (T,6)
    p_fk_arr = np.vstack(p_fk_traj)    # (T,3)
    p_wb_arr = np.vstack(p_wb_traj)    # (T,3)
    com_arr = np.vstack(com_traj)      # (T,3)
    tau_arr = np.vstack(tau_traj)      # (T,6)
    time_arr = np.array(time_traj)
    energy_arr = np.array(energy_traj)

    # 步长 dt：优先用仿真时间差
    if len(time_arr) > 1:
        dt_est = (time_arr[-1] - time_arr[0]) / (len(time_arr) - 1)
    else:
        dt_est = dt
    if dt_est <= 0:
        dt_est = dt
    print(f"[orchestrator] dt ≈ {dt_est:.4f} s, samples = {len(time_arr)}")

    # ---- 1) 轨迹误差（RMSE & max） ----
    err_vec_raw, rmse_raw = compute_traj_error(p_fk_arr, p_wb_arr)

    # 估计一个常量 bias（FK - WB 的平均差），用于区分“整体偏移”和“轨迹形状误差”
    bias = np.mean(p_fk_arr - p_wb_arr, axis=0)
    p_fk_debiased = p_fk_arr - bias
    err_vec, rmse = compute_traj_error(p_fk_debiased, p_wb_arr)

    print(f"[orchestrator] Foot trajectory RMSE (raw) = {rmse_raw*1000:.2f} mm, "
          f"max (raw) = {np.max(err_vec_raw)*1000:.2f} mm")
    print(f"[orchestrator] Foot trajectory RMSE (bias-removed) = {rmse*1000:.2f} mm")
    print(f"[orchestrator] Mean bias (FK - WB) = ("
          f"{bias[0]:+.3f}, {bias[1]:+.3f}, {bias[2]:+.3f}) m")

    # ---- 2) ZMP over time ----
    zmp_arr = zmp_from_com_traj(com_arr, time_arr)
    print(f"[orchestrator] ZMP trajectory length = {len(zmp_arr)}")

    # ---- 3) 能量（基于伪力矩 tau_arr）----
    P_traj, E_total = energy_over_trajectory(tau_arr, q_arr, dt_est)
    print(f"[orchestrator] Energy over trajectory (relative model): {E_total:.3f}")

    # ---- 4) 保存数据 ----
    np.save("traj_time.npy", time_arr)
    np.save("traj_q_right_leg.npy", q_arr)
    np.save("traj_p_fk.npy", p_fk_arr)
    np.save("traj_p_wb.npy", p_wb_arr)
    np.save("traj_com.npy", com_arr)
    np.save("traj_zmp.npy", zmp_arr)
    np.save("traj_power_relative.npy", P_traj)
    np.save("traj_energy_cum.npy", energy_arr)
    np.save("traj_error.npy", err_vec_raw)

    print("[orchestrator] Saved .npy files in controller directory.")
    print("  → traj_time, traj_q_right_leg, traj_p_fk, traj_p_wb, traj_com, "
          "traj_zmp, traj_power_relative, traj_energy_cum, traj_error")


if __name__ == "__main__":
    main()
