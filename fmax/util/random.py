import numpy as np
import pymc3 as pm

def get_random_fn(
    n_periods, 
    past_obs = None,
    attempts = "gaussian",
    kind = 'min',
    ):
    """ Get the random sampling function
        for a cumulative distribution whose underlying attempts 
        follow a certain distribution.
    """
    
    def _random(point=None, size=None, 
        n_periods=n_periods, past_obs=past_obs, attempts = attempts):
      """ Random sampling function
      """
      
      if attempts == 'gaussian':
          mu, sigma = point['mu'], point['sigma']
          attempts = mu + sigma*np.random.randn(n_periods)
      
      elif attempts == 'gumbel':
          mu, sigma = point['mu'], point['sigma']
          beta = pm.math.sqrt((6/(np.pi**2))*(sigma**2))
          mu = (-mu) - beta*np.euler_gamma
          attempts = -1*pm.Gumbel.dist(mu=mu, beta=beta).random(size=n_periods)
      
      elif attempts == 'weibull':
          raise NotImplementedError
      
      else: raise NotImplementedError

      if past_obs is not None:
          attempts = np.concatenate([past_obs, attempts])
      
      if kind == 'max':
        sample_path = np.minimum.accumulate(attempts)
      elif kind == 'min':
        sample_path = np.minimum.accumulate(attempts)
        
      return sample_path
      
    return _random