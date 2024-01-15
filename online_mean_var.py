import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

class OnlineMeanVar():
    def __init__(self, dims: int = 1):
        self._N = 0
        self._m = np.zeros(dims)
        self._v = np.zeros(dims)

    def update(self, x: FloatArray) -> None:
        self._N += 1
        w = 1 / self._N
        d = x - self._m
        self._m += d * w
        self._v += -self._v * w + w * (1 - w) * d ** 2

    def reset(self) -> None:
        self._N = 0
        self._m[:] = 0.0
        self._v[:] = 0.0

    def N(self) -> int:
        return self._N

    def location(self) -> FloatArray:
        return self._m

    def scale(self) -> FloatArray:
        return np.sqrt(self._v)

class MetricAdapter():
    def update(self, x: FloatArray) -> None:
        self.estimator.update(x)

    def reset(self) -> None:
        self.estimator.reset()

    def metric(self) -> FloatArray:
        N = self.estimator.N()
        V = self.estimator.scale() ** 2
        if N > 2:
            w = N / (N + 5)
            return w * V + (1 - w) * 1e-3
        else:
            return np.ones_like(V)

    def location(self) -> FloatArray:
        return self.estimator.location()

class MetricOnlineMeanVar(MetricAdapter):
    def __init__(self, dims: int = 1):
        self.estimator = OnlineMeanVar(dims)
