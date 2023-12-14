import numpy as np

def leapfrog(position, momentum, step_size, steps, gradient, ldg):
    ld = 0.0
    momentum += 0.5 * step_size * gradient
    for step in range(steps):
        position += step_size * momentum
        ld = ldg(position, gradient)
        if step != (steps - 1):
            momentum += step_size * gradient
    momentum += 0.5 * step_size * gradient
    return ld
