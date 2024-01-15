import numpy as np

def u_turn(pb, pf, pt, cut_off = 0.0):
    normb = np.linalg.norm(pb)
    normf = np.linalg.norm(pf)
    normt = np.linalg.norm(pt)
    f = np.dot(pf, pt) / (normf * normt)
    b = np.dot(pb, pt) / (normb * normt)
    return f <= cut_off or b <= cut_off

def power_two(x):
    return (x != 0) and (x & (x - 1) == 0)
