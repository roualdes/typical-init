import numpy as np

def hamiltonian(ld, momentum):
    return -ld + 0.5 * np.dot(momentum, momentum)
