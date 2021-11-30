import numpy as np
import pymc3 as pm

def gaussian_random(n_periods=100, past_obs=None):
    """Builds a random sampling function for use with a Gaussian attempt model.
    """

    def _random(point=None, size=None, n_periods=n_periods, past_obs=past_obs):
        mu, sigma = point['mu'], point['sigma']
        attempts = mu + sigma*np.random.randn(n_periods)

        if past_obs is not None:
            last_obs = np.atleast_1d(past_obs[-1])
            attempts = np.concatenate([last_obs, attempts])
            sample_path = np.minimum.accumulate(attempts)[1:]
            full_sample_path = np.concatenate([past_obs, sample_path])
            return full_sample_path
        else:
            sample_path = np.minimum.accumulate(attempts)
            return sample_path

    return _random


def gumbel_random_min(n_periods=100, past_obs=None):
    """Builds a random sampling function for use with a Gumbel attempt model.
    """

    def _random(point=None, size=None, n_periods=n_periods, past_obs=past_obs):
        mu, sigma = point['mu'], point['sigma']
        beta = pm.math.sqrt((6/(np.pi**2))*(sigma**2))
        mu = (-mu) - beta*np.euler_gamma
        
        attempts = -1*pm.Gumbel.dist(mu=mu, beta=beta).random(size=n_periods)

        if past_obs is not None:
            last_obs = np.atleast_1d(past_obs[-1])
            attempts = np.concatenate([last_obs, attempts])
            sample_path = np.minimum.accumulate(attempts)[1:]
            full_sample_path = np.concatenate([past_obs, sample_path])
            return full_sample_path
        else:
            sample_path = np.minimum.accumulate(attempts)
            return sample_path

    return _random