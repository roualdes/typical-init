import numpy as np

def initialize_draws(rng, dim, ldg, initial_draw_radius = 2, initial_draw_attempts = 100):

    attempts = initial_draw_attempts
    attempt = 0

    initialized = False
    radius = initial_draw_radius
    gradient = np.empty(dim)

    while attempt < attempts and not initialized:
        initial_draw = rng.uniform(-radius, radius, size = dim)
        ld = ldg(initial_draw, gradient)

        if np.isfinite(ld) and not np.isnan(ld):
            initialized = True

        g = np.sum(gradient)
        if not np.isfinite(g) or np.isnan(g):
            initialized = False
            continue

        attempt += 1

    if attempt > attempts:
        print(f"Failed to find initial values in {attempt} attempts")

    return initial_draw
