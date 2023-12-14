from leapfrog import leapfrog
from hamiltonian import hamiltonian
from step_size_adapter import StepSizeAdapter
from initialize_draws import initialize_draws
from step_size_initializer import step_size_initializer

from tools import u_turn, power_two

import numpy as np

def initialize(rng, dim, ldg, steps = 75, max_leapfrogs = 1_000, max_delta_H = 1_000.0, cut_off = 0.0):

    position = initialize_draws(rng, dim, ldg)
    step_size = step_size_initializer(position, dim, 1.0, rng, ldg)
    step_size_adapter = StepSizeAdapter()
    
    number_leapfrogs = np.zeros(steps)
    number_divergences = 0
    number_uturns = 0
    gradient = np.zeros(dim)

    for step in range(steps):
        n_leapfrog = 0
        alpha = 0.0
        u_turned = False
        divergent = False
        momentum = rng.normal(size = dim)
        momentum_new = momentum.copy()
        position_new = position.copy()

        ld = ldg(position_new, gradient)
        H0 = hamiltonian(ld, momentum)
        direction = np.random.choice([-1, 1])

        while n_leapfrog < max_leapfrogs:
            n_leapfrog += 1
            ld = leapfrog(position_new, momentum_new, direction * step_size, 1, gradient, ldg)

            H = hamiltonian(ld, momentum)
            if np.isnan(H):
                H = np.inf

            if H - H0 > max_delta_H:
                divergent = True

            delta_H = H0 - H
            if delta_H > 0.0:
                alpha += 1.0
            else:
                alpha += np.exp(delta_H)

            if divergent:
                number_divergences += 1
                break

            if (n_leapfrog <= 64 and power_two(n_leapfrog)) or n_leapfrog % 50 == 0:
                u_turned = u_turn(position, position_new, momentum_new, cut_off = cut_off)

                if u_turned:
                    number_uturns += 1
                    break

        number_leapfrogs[step] = n_leapfrog

        step_size_adapter.update(alpha / n_leapfrog)
        step_size = step_size_adapter.optimum(smooth = False)
        
        if not divergent:
            position = position_new.copy()

    return {
        "position": position,
        "number_leapfrogs": number_leapfrogs,
        "number_divergences": number_divergences,
        "number_uturns": number_uturns,
        "step_size": step_size,
    }
