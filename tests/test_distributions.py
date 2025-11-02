"""Tests for the distributions module."""

import numpy as np

from spatial_competition_pettingzoo.distributions import (
    ConstantMultivariateDistribution,
    ConstantUnivariateDistribution,
    MultivariateNormalDistribution,
    MultivariateUniformDistribution,
)


class TestDistributions:
    """Test class for distributions functionality."""

    def test_constant_univariate_distribution(self) -> None:
        """Test constant univariate distribution functionality."""
        distribution = ConstantUnivariateDistribution(value=0.5)
        assert distribution.rvs(random_state=np.random.default_rng(42)) == 0.5

    def test_constant_multivariate_distribution(self) -> None:
        """Test constant multivariate distribution functionality."""
        distribution = ConstantMultivariateDistribution(dim=2, value=np.array([0.5, 0.5]))
        assert np.array_equal(distribution.rvs(random_state=np.random.default_rng(42)), np.array([0.5, 0.5]))

    def test_multivariate_uniform_distribution(self) -> None:
        """Test multivariate uniform distribution functionality."""
        distribution = MultivariateUniformDistribution(dim=2, loc=0.0, scale=1.0)
        assert distribution.dim == 2

        sample = distribution.rvs(random_state=np.random.default_rng(42))
        assert sample.shape == (2,)
        assert 0 <= sample[0] < 1
        assert 0 <= sample[1] < 1

    def test_multivariate_normal_distribution(self) -> None:
        """Test multivariate normal distribution functionality."""
        distribution = MultivariateNormalDistribution(mean=np.array([0.5, 0.5]), cov=np.array([[0.1, 0.0], [0.0, 0.1]]))
        assert distribution.dim == 2

        sample = distribution.rvs(random_state=np.random.default_rng(42))
        assert sample.shape == (2,)
        assert 0 <= sample[0] < 1
        assert 0 <= sample[1] < 1
