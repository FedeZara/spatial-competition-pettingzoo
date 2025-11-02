from unittest.mock import MagicMock

import pytest
from pettingzoo.utils.env import np

from spatial_competition_pettingzoo.distributions import MultivariateNormalDistribution
from spatial_competition_pettingzoo.utils import sample_and_clip_univariate_distribution


class TestUtils:
    """Test class for Utils functionality."""

    @pytest.fixture
    def mock_rng(self) -> np.random.Generator:
        """Create mock RNG."""
        return np.random.default_rng(42)

    def test_sample_and_clip_univariate_distribution(self, mock_rng: np.random.Generator) -> None:
        """Test sample_and_clip_univariate_distribution functionality."""
        distribution = MagicMock()
        distribution.rvs.side_effect = [0.5]
        distribution.dim = 1

        sample = sample_and_clip_univariate_distribution("test", distribution, mock_rng)
        assert sample == 0.5

    def test_sample_and_clip_univariate_distribution__array_size_error(self, mock_rng: np.random.Generator) -> None:
        """Test sample_and_clip_univariate_distribution raises error for array size mismatch."""
        distribution = MagicMock()
        distribution.rvs.side_effect = [np.array([0.5, 0.6])]
        distribution.dim = 1

        with pytest.raises(ValueError, match="test sample size must be 1"):
            sample_and_clip_univariate_distribution("test", distribution, mock_rng)

    def test_sample_and_clip_univariate_distribution__clipping(self, mock_rng: np.random.Generator) -> None:
        """Test sample_and_clip_univariate_distribution clips values to the specified range."""
        distribution = MagicMock()
        distribution.rvs.side_effect = [-0.5, 1.5]
        distribution.dim = 1

        sample = sample_and_clip_univariate_distribution("test", distribution, mock_rng, min_value=0.0, max_value=1.0)
        assert sample == 0.0
        sample = sample_and_clip_univariate_distribution("test", distribution, mock_rng, min_value=0.0, max_value=1.0)
        assert sample == 1.0

    def test_sample_and_clip_univariate_distribution__multivariate_normal(self, mock_rng: np.random.Generator) -> None:
        """Test sample_and_clip_univariate_distribution uses default range if not specified."""
        mean = np.array([0.5, 0.5])
        cov = np.array([[0.1, 0.0], [0.0, 0.1]])

        with pytest.raises(ValueError, match="test sample size must be 1"):
            sample_and_clip_univariate_distribution(
                "test", MultivariateNormalDistribution(mean=mean, cov=cov), mock_rng
            )
