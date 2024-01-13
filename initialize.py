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
    position_uturn = 0
    momentum_uturn = 0
    count_momentum_uturn = {"m_mn": 0, "m_ms": 0, "ms_mn": 0, "mb_mn": 0, "mb_ms": 0}
    either_uturn = 0
    gradient = np.zeros(dim)

    for step in range(steps):
        n_leapfrog = 0
        alpha = 0.0
        u_turned = False
        q_uturned = False
        p_uturned = False
        divergent = False
        momentum = rng.normal(size = dim)
        momentum_new = momentum.copy()
        position_new = position.copy()

        ld = ldg(position_new, gradient)
        H0 = hamiltonian(ld, momentum_new)
        direction = np.random.choice([-1, 1])

        # one leapfrog in negative direction
        momentum_backwards = momentum.copy()
        position_backwards = position.copy()
        gradient_backwards = gradient.copy()
        leapfrog(position_backwards, momentum_backwards, -1 * direction * step_size, 1, gradient_backwards, ldg)
        momentum_subtree = momentum.copy()
        max_distance = -np.inf
        momentum_total = momentum_backwards.copy()

        while n_leapfrog < max_leapfrogs:
            n_leapfrog += 1
            ld = leapfrog(position_new, momentum_new, direction * step_size, 1, gradient, ldg)

            H = hamiltonian(ld, momentum_new)
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

            momentum_total += momentum_new
            u_turn_m_mn = u_turn(momentum, momentum_new, momentum_total, cut_off = cut_off)
            count_momentum_uturn["m_mn"] += u_turn_m_mn
            
            u_turn_m_ms = u_turn(momentum, momentum_subtree, momentum_total, cut_off = cut_off)
            count_momentum_uturn["m_ms"] += u_turn_m_ms
            
            u_turn_ms_mn = u_turn(momentum_subtree, momentum_new, momentum_total, cut_off = cut_off)  
            count_momentum_uturn["ms_mn"] += u_turn_ms_mn
            
            u_turn_mb_mn = u_turn(momentum_backwards, momentum_new, momentum_total, cut_off = cut_off)  
            count_momentum_uturn["mb_mn"] += u_turn_mb_mn
            
            u_turn_mb_ms = u_turn(momentum_backwards, momentum_subtree, momentum_total, cut_off = cut_off)  
            count_momentum_uturn["mb_ms"] += u_turn_mb_ms

            u_turned = u_turn_m_mn | u_turn_m_ms | u_turn_ms_mn | u_turn_mb_mn | u_turn_mb_ms
            if u_turned:
                p_uturned = True
                
            new_distance = np.linalg.norm(position_new - position)
            if new_distance <= max_distance:
                q_uturned = True
            else:
                max_distance = new_distance
                
            if power_two(n_leapfrog):
                momentum_subtree = momentum_new.copy()

            position_uturn += q_uturned 
            momentum_uturn += p_uturned 
            either_uturn += p_uturned or q_uturned

            if q_uturned or p_uturned:
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
        "momentum_uturn": momentum_uturn,
        "position_uturn": position_uturn,
        "either_uturn": either_uturn,
        "step_size": step_size,
        "count_momentum_uturn": count_momentum_uturn,
    }
