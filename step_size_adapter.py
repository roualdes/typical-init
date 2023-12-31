import numpy as np

class DualAverage():
    def __init__(self, mu: float = np.log(10.0), gamma: float = 0.05,
                 kappa: float = 0.75, t0: int = 10.0):
        self._sbar = 0.0
        self._x = 0.0
        self._xbar = 0.0
        self._mu = mu
        self._gamma = gamma
        self._t0 = t0
        self._kappa = kappa
        self._counter = 0.0

    def update(self, gradient: float):
        self._counter += 1.0
        eta = 1.0 / (self._counter + self._t0)
        self._sbar = (1.0 - eta) * self._sbar + eta * gradient
        self._x = self._mu - self._sbar * np.sqrt(self._counter) / self._gamma
        xeta = self._counter ** -self._kappa
        self._xbar = (1.0 - xeta) * self._xbar + xeta * self._x

    def reset(self, mu = np.log(10.0)):
        self._mu = mu
        self._sbar = 0.0
        self._xbar = 0.0
        self._counter = 0.0

    def optimum(self, smooth = True):
        if smooth:
            return np.exp(self._xbar)
        return np.exp(self._x)

class StepSizeAdapter():
    def __init__(self, delta: float = 0.8, **kwargs):
        self._delta = delta
        self._dual_average = DualAverage(**kwargs)

    def update(self, adapt_stat: float):
        adapt_stat = 1.0 if adapt_stat > 1.0 else adapt_stat
        self._dual_average.update(self._delta - adapt_stat)

    def reset(self, mu = np.log(10.0)):
        self._dual_average.reset(mu = mu)

    def optimum(self, smooth = True) -> float:
        return self._dual_average.optimum(smooth = smooth)
