import numpy as np

def u_turn(position, position_new, momentum_new, cut_off = 0.0):
    return np.dot(position_new - position, momentum_new) <= cut_off

def power_two(x):
    return (x != 0) and (x & (x - 1) == 0)
