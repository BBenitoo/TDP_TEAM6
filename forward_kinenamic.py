
from controller import Supervisor
import math


RIGHT_LEG_NAMES = [
    "RHipYawPitch", "RHipRoll", "RHipPitch",
    "RKneePitch", "RAnklePitch", "RAnkleRoll"
]

def mat_eye():
    return [[1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]]

def mat_mul(A,B):
    C=[[0]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            C[i][j]=sum(A[i][k]*B[k][j] for k in range(4))
    return C

# --- 你给定的六个矩阵 ---
def A1(t):
    c,s = math.cos(t), math.sin(t)
    return [[ c,  s, 0, 0],
            [-s,  c, 0, 0],
            [ 0,  0, 1, 0],
            [ 0,  0, 0, 1]]

def A2(t):
    c,s = math.cos(t), math.sin(t)
    return [[ c, 0,  s, 0],
            [ s, 0, -c, 0],
            [ 0, 1,  0, 0],
            [ 0, 0,  0, 1]]

def A3(t):
    c,s = math.cos(t), math.sin(t)
    return [[ c, 0,  s, 0],
            [ s, 0, -c, 0],
            [ 0, 1,  0, 0],
            [ 0, 0,  0, 1]]

def A4(t):
    c,s = math.cos(t), math.sin(t)
    return [[ c, -s, 0, 0.1000*c],
            [ s,  c, 0, 0.1000*s],
            [ 0,  0, 1, 0.0     ],
            [ 0,  0, 0, 1.0     ]]

def A5(t):
    c,s = math.cos(t), math.sin(t)
    return [[ c, -s, 0, 0.1029*c],
            [ s,  c, 0, 0.1029*s],
            [ 0,  0, 1, 0.0     ],
            [ 0,  0, 0, 1.0     ]]

def A6(t):
    c,s = math.cos(t), math.sin(t)
    return [[ c, 0,  s, 0],
            [ s, 0, -c, 0],
            [ 0, 1,  0, 0],
            [ 0, 0,  0, 1]]

def fk_right_foot_from_q(q):
    t1,t2,t3,t4,t5,t6 = q
    T = mat_eye()
    for Ti in (A1(t1), A2(t2), A3(t3), A4(t4), A5(t5), A6(t6)):
        T = mat_mul(T, Ti)
    return T

# ----------------- 场景树遍历与匹配 -----------------
def find_right_leg_hinges(supervisor, expect_names):

    root = supervisor.getRoot()
    children = root.getField("children")
    found_by_def = {nm: None for nm in expect_names}
    found_by_dev = {nm: None for nm in expect_names}

    def dfs(node):
        if not node: return
        tname = node.getTypeName()
        # 先看 DEF 名（比如 "DEF RHipYawPitch HingeJoint"）
        d = node.getDef()
        if d in found_by_def and found_by_def[d] is None and tname in ("HingeJoint", "Hinge2Joint"):
            found_by_def[d] = node

        # 再看 device.name（RotationalMotor 的 name 字段）
        if tname in ("HingeJoint", "Hinge2Joint"):
            dev = node.getField("device")
            if dev and dev.getCount() > 0:
                dnode = dev.getSFNode()
                if dnode:
                    namef = dnode.getField("name")
                    if namef:
                        nm = namef.getSFString()
                        if nm in found_by_dev and found_by_dev[nm] is None:
                            found_by_dev[nm] = node

        # 递归遍历
        ch = node.getField("children")
        if ch:
            for i in range(ch.getCount()):
                dfs(ch.getMFNode(i))
        ep = node.getField("endPoint")
        if ep:
            dfs(ep.getSFNode())

    for i in range(children.getCount()):
        dfs(children.getMFNode(i))

    # 优先 DEF 匹配，其次 device.name
    result = []
    for nm in expect_names:
        result.append(found_by_def.get(nm) or found_by_dev.get(nm))
    return result

def get_joint_pos(jnode):
    if not jnode: return None
    pf = jnode.getField("position")
    return pf.getSFFloat() if pf else None

def get_last_joint_endpoint_solid(jnode):
    if not jnode: return None
    ep = jnode.getField("endPoint")
    if not ep: return None
    node = ep.getSFNode()
    return _find_first_solid(node)

def _find_first_solid(node):
    if not node: return None
    if node.getTypeName() == "Solid":
        return node
    for fname in ("children","endPoint"):
        f = node.getField(fname)
        if f:
            if f.getTypeName() == "MFNode":
                for i in range(f.getCount()):
                    s = _find_first_solid(f.getMFNode(i))
                    if s: return s
            else:
                s = _find_first_solid(f.getSFNode())
                if s: return s
    return None

def rot_diff_angle_deg(Ra,Rb):
    Rt=[[Rb[j][i] for j in range(3)] for i in range(3)]
    Rerr=[[sum(Ra[i][k]*Rt[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
    tr=Rerr[0][0]+Rerr[1][1]+Rerr[2][2]
    c=max(-1.0,min(1.0,(tr-1.0)/2.0))
    return math.degrees(math.acos(c))

def T_to_p_R(T):
    p=(T[0][3],T[1][3],T[2][3])
    R=[[T[i][j] for j in range(3)] for i in range(3)]
    return p,R

# ----------------- 主循环 -----------------
def main():
    sup = Supervisor()
    dt  = int(sup.getBasicTimeStep())

    joints = find_right_leg_hinges(sup, RIGHT_LEG_NAMES)
    for nm,nd in zip(RIGHT_LEG_NAMES, joints):
        print(f"[fk_compare] joint {nm}: {'FOUND' if nd else 'NOT FOUND'}")

    foot_node = get_last_joint_endpoint_solid(joints[-1]) if joints[-1] else None
    print("[fk_compare] foot Solid:", "FOUND" if foot_node else "NOT FOUND (check RAnkleRoll → endPoint)")

    label_id = 7
    while sup.step(dt) != -1:
        q=[]; ok=True
        for nd in joints:
            v=get_joint_pos(nd)
            if v is None: ok=False; v=0.0
            q.append(v)
        if not ok:
            sup.setLabel(label_id,"Some joints not found. Check DEF/device names.",0.01,0.95,0.06,0xFF4444,0)
            continue

        T_fk = fk_right_foot_from_q(q)
        p_fk, R_fk = T_to_p_R(T_fk)

        if foot_node:
            p_wb = foot_node.getPosition()
            R9   = foot_node.getOrientation()
            R_wb = [[R9[0],R9[1],R9[2]],
                    [R9[3],R9[4],R9[5]],
                    [R9[6],R9[7],R9[8]]]
            dp    = (p_fk[0]-p_wb[0], p_fk[1]-p_wb[1], p_fk[2]-p_wb[2])
            pos_e = math.sqrt(dp[0]**2+dp[1]**2+dp[2]**2)
            ang_e = rot_diff_angle_deg(R_fk, R_wb)
            msg=(f"[fk_compare] |p_err|={pos_e*1000:.1f} mm, ang_err={ang_e:.2f} deg\n"
                 f"FK p=({p_fk[0]:+.3f},{p_fk[1]:+.3f},{p_fk[2]:+.3f})  "
                 f"WB p=({p_wb[0]:+.3f},{p_wb[1]:+.3f},{p_wb[2]:+.3f})")
        else:
            msg="[fk_compare] Can't locate foot Solid under RAnkleRoll.endPoint"

        print(msg)
        sup.setLabel(label_id, msg, 0.01, 0.95, 0.06, 0xFFFFFF, 0)

if __name__ == "__main__":
    main()
