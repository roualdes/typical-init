import numpy as np

from hamiltonian import hamiltonian
from leapfrog import leapfrog

def step_size_initializer(position, dims, step_size, rng, ldg):

    q = position.copy()
    momentum = rng.normal(size = dims)
    gradient = np.empty_like(momentum)

    ld = ldg(q, gradient)
    H0 = hamiltonian(ld, momentum)

    ld = leapfrog(q, momentum, step_size, 1, gradient, ldg)
    H = hamiltonian(ld, momentum)
    if np.isnan(H):
        H = np.finfo(np.float64).max

    dH = H0 - H
    direction = 1 if dH > np.log(0.8) else -1

    while True:
        momentum = rng.normal(size = dims)
        q = position.copy()
        ld = ldg(q, gradient)
        H0 = hamiltonian(ld, momentum)

        ld = leapfrog(q, momentum, step_size, 1, gradient, ldg)
        H = hamiltonian(ld, momentum)
        if np.isnan(H):
            H = np.finfo(np.float64).max

        dH = H0 - H
        if np.isnan(dH):
            dH = np.finfo(np.float64).max

        if direction == 1 and not dH > np.log(0.8):
            break
        elif direction == -1 and not dH < np.log(0.8):
            break
        else:
            step_size = 2 * step_size if direction == 1 else 0.5 * step_size

        if step_size > 1e7:
            raise ValueError("Step size too large.  Posterior is improper.  Please check your model.")

        if step_size <= 0.0:
            raise ValueError("Step size too small.  No acceptable step size could be found.  Perhaps the posterior is not continuous.")

    return step_size
