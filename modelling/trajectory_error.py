import numpy as np
import matplotlib.pyplot as plt

# ========= 1) 读取数据（两种方式，二选一） =========
# A. 你已有 numpy 数组：
# t: (T,), p_FK: (T,3), p_sim: (T,3)
# t = ...
# p_FK = ...
# p_sim = ...

# B. 从 CSV 读取（示例：列为 time, FK_x, FK_y, FK_z, sim_x, sim_y, sim_z）
# np.loadtxt 也可，用 genfromtxt 更稳健些
# data = np.genfromtxt("foot_data.csv", delimiter=",", names=True)
# t = data["time"]
# p_FK  = np.stack([data["FK_x"],  data["FK_y"],  data["FK_z"]],  axis=1)
# p_sim = np.stack([data["sim_x"], data["sim_y"], data["sim_z"]], axis=1)

# ========= 2) 若时间轴不同，做插值到共同时间网格（可选） =========
def interp_to(t_src, y_src, t_tar):
    # y_src: (Tsrc, D)
    y_src = np.asarray(y_src)
    out = np.zeros((len(t_tar), y_src.shape[1]))
    for d in range(y_src.shape[1]):
        out[:, d] = np.interp(t_tar, t_src, y_src[:, d])
    return out

# 若 p_FK 与 p_sim 的时间戳分别是 t_FK 与 t_sim：
# t = 选一条公共时间轴（例如较密的那一条或自己构造均匀时间轴）
# p_FK  = interp_to(t_FK,  p_FK,  t)
# p_sim = interp_to(t_sim, p_sim, t)

# ========= 3) 计算误差 =========
def compute_errors(p_FK, p_sim):
    """
    p_FK, p_sim: shape (T,3), 单位一致（m）
    返回：
      e_vec: 逐时刻向量误差 (T,3)
      e_mag: 逐时刻标量误差 ||e_vec||2 (T,)
      e_axis_abs: 逐轴绝对误差 (T,3) 方便调试
      e_rms: 标量 RMS（对 e_mag）
      e_rms_axis: 每轴 RMS（对逐轴误差）
    """
    e_vec = p_sim - p_FK                         # (T,3)
    e_axis_abs = np.abs(e_vec)                   # (T,3)
    e_mag = np.linalg.norm(e_vec, axis=1)        # (T,)
    e_rms = np.sqrt(np.mean(e_mag**2))           # 标量 RMS
    e_rms_axis = np.sqrt(np.mean(e_vec**2, axis=0))  # 每轴 RMS
    return e_vec, e_mag, e_axis_abs, e_rms, e_rms_axis

e_vec, e, e_axis_abs, e_rms, e_rms_axis = compute_errors(p_FK, p_sim)

print(f"RMS position error (scalar) e_rms = {e_rms:.6f} m")
print(f"RMS per-axis [ex, ey, ez] = {e_rms_axis}")

# ========= 4) 画误差随时间曲线 =========
plt.figure()
plt.plot(t, e, label="||p_sim - p_FK||")
plt.xlabel("Time (s)")
plt.ylabel("Position error (m)")
plt.title("Foot position error over time")
plt.legend()
plt.tight_layout()
plt.show()