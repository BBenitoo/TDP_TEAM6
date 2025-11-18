import numpy as np

def compute_COM(A_list, m_list, r_com_list):
    """
    Computes the overall Center of Mass (COM) in the base/world frame.

    Parameters
    ----------
    A_list : list of np.ndarray
        List of 4x4 homogeneous transforms A_i (base → link i) from FK.
    m_list : list or np.ndarray
        Mass of each link [m1, m2, ...].
    r_com_list : list of np.ndarray
        Local COM positions of each link in its own frame 
        [ [x1, y1, z1], [x2, y2, z2], ... ] (meters).

    Returns
    -------
    p_com : np.ndarray (3,)
        Overall COM position in base/world frame [x, y, z].
    """

    total_mass = np.sum(m_list)
    weighted_sum = np.zeros(3)

    for i in range(len(m_list)):
        # 1) Convert local COM to homogeneous vector
        r_local = np.append(r_com_list[i], 1)

        # 2) Map to base/world frame using transform A_i
        p_world = A_list[i] @ r_local

        # 3) Take first 3 elements (x, y, z)
        p_world = p_world[:3]

        # 4) Add mass-weighted position
        weighted_sum += m_list[i] * p_world

    # 5) Divide by total mass
    p_com = weighted_sum / total_mass
    return p_com


# ---------------- Example ----------------
if __name__ == "__main__":
    # Example for 3-link robot (replace with NAO6 data)
    A1 = np.eye(4)
    A2 = np.array([[1, 0, 0, 0.1],
                   [0, 1, 0, 0.0],
                   [0, 0, 1, 0.3],
                   [0, 0, 0, 1]])
    A3 = np.array([[1, 0, 0, 0.2],
                   [0, 1, 0, 0.0],
                   [0, 0, 1, 0.5],
                   [0, 0, 0, 1]])

    A_list = [A1, A2, A3]

    # masses (kg)
    m_list = [2.0, 1.0, 1.5]

    # local COM positions (m)
    r_com_list = [np.array([0.0, 0.0, 0.05]),
                  np.array([0.1, 0.0, 0.0]),
                  np.array([0.05, 0.02, 0.0])]

    p_com = compute_COM(A_list, m_list, r_com_list)
    print("Overall COM position [m]:", p_com)
