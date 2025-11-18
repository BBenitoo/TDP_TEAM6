
import sympy as sp

def dh_matrix(alpha, a, theta, d):
    ca, sa = sp.cos(alpha), sp.sin(alpha)
    ct, st = sp.cos(theta), sp.sin(theta)
    return sp.Matrix([
        [ ct, -st*ca,  st*sa, a*ct],
        [ st,  ct*ca, -ct*sa, a*st],
        [  0,     sa,     ca,    d],
        [  0,      0,      0,    1]
    ])

def build_fk_and_jacobian():
    q1, q2, q3, q4, q5, q6 = sp.symbols('q1 q2 q3 q4 q5 q6')
    qs = [q1, q2, q3, q4, q5, q6]
    alphas = [0, sp.pi/2, sp.pi/2, 0, 0, sp.pi/2]
    a_vals = [0, 0, 0, 0.1, 0.1029, 0]
    d_vals = [0, 0, 0, 0, 0, 0]
    T = sp.eye(4)
    Ts = [T]  
    for i in range(6):
        A_i = dh_matrix(alphas[i], a_vals[i], qs[i], d_vals[i])
        T = sp.simplify(T * A_i)
        Ts.append(T)
    T06 = Ts[-1]
    o_list = [Ti[:3, 3] for Ti in Ts]
    z_list = [Ti[:3, 2] for Ti in Ts]
    o6 = o_list[-1]
    Jv_cols, Jw_cols = [], []
    for i in range(6):
        zi = z_list[i]
        oi = o_list[i]
        Jv_cols.append(sp.simplify(zi.cross(o6 - oi)))
        Jw_cols.append(zi)

    Jv = sp.Matrix.hstack(*Jv_cols)
    Jw = sp.Matrix.hstack(*Jw_cols)
    J  = sp.Matrix.vstack(Jv, Jw)
    J  = sp.simplify(J)

    return (qs, T06, J)

if __name__ == "__main__":
    qs, T06, J = build_fk_and_jacobian()
    print("\n=== Jacobian J(q)  ===")
    sp.pprint(J)
