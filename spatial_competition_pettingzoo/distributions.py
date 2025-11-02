from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.stats import multivariate_normal, uniform

# Type aliases for cleaner code
FloatArray = float | NDArray[np.floating[Any]]


class DistributionProtocol(Protocol):
    def rvs(self, random_state: np.random.Generator) -> FloatArray: ...


class MultivariateDistributionProtocol(Protocol):
    def rvs(self, random_state: np.random.Generator) -> NDArray[np.floating[Any]]: ...

    @property
    def dim(self) -> int: ...


class ConstantUnivariateDistribution(DistributionProtocol):
    def __init__(self, value: float) -> None:
        self._value = value

    def rvs(self, random_state: np.random.Generator) -> float:  # noqa: ARG002
        return self._value


class ConstantMultivariateDistribution(MultivariateDistributionProtocol):
    def __init__(self, dim: int, value: NDArray[np.floating[Any]]) -> None:
        self._dim = dim
        self._value = value

    def rvs(self, random_state: np.random.Generator) -> NDArray[np.floating[Any]]:  # noqa: ARG002
        return np.full(self._dim, self._value)

    @property
    def dim(self) -> int:
        return self._dim


class MultivariateUniformDistribution(MultivariateDistributionProtocol):
    def __init__(self, dim: int, loc: float, scale: float) -> None:
        self._dim = dim
        self._loc = loc
        self._scale = scale
        self._multivariate_uniform = uniform(loc, scale)

    def rvs(self, random_state: np.random.Generator) -> NDArray[np.floating[Any]]:
        return np.array([self._multivariate_uniform.rvs(random_state=random_state) for _ in range(self._dim)])

    @property
    def dim(self) -> int:
        return self._dim


class MultivariateNormalDistribution(MultivariateDistributionProtocol):
    def __init__(self, mean: NDArray[np.floating[Any]], cov: NDArray[np.floating[Any]]) -> None:
        self._mean = mean
        self._cov = cov
        self._multivariate_normal = multivariate_normal(mean=mean, cov=cov)

    def rvs(self, random_state: np.random.Generator) -> NDArray[np.floating[Any]]:
        return self._multivariate_normal.rvs(random_state=random_state)

    @property
    def dim(self) -> int:
        return self._multivariate_normal.dim
