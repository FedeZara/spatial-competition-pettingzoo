import numpy as np

from spatial_competition_pettingzoo.distributions import DistributionProtocol


def sample_and_clip_univariate_distribution(
    property_name: str,
    distribution: DistributionProtocol,
    rng: np.random.Generator,
    min_value: float = -np.inf,
    max_value: float = np.inf,
) -> float:
    sample = distribution.rvs(random_state=rng)

    # check sample size
    if isinstance(sample, np.ndarray) and sample.size != 1:
        error_msg = f"{property_name} sample size must be 1"
        raise ValueError(error_msg)

    return float(np.clip(sample, min_value, max_value))
