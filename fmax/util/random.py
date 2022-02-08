import numpy as np
import pymc3 as pm

from scipy.stats import weibull_min, weibull_max, gumbel_l, gumbel_r


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
      """ Random sampling function. n_periods is now only used for forecasting,
      and ignored when not.
      """
      model = pm.modelcontext(None)
      which_random = model['random_switch'].get_value()

      # If posterior predictive, ignore original n_periods
      if which_random == 0:
          n_periods = len(past_obs)

      if attempts == 'gaussian':
          mu, sigma = point['mu'], point['sigma']
          attempts = mu + sigma*np.random.randn(n_periods)
      
      elif attempts == 'gumbel':
        if kind == "min":
            mu, sigma = point['mu'], point['sigma']
            beta = (1/np.pi)*np.sqrt(6)*sigma
            #mu = mu + beta*np.euler_gamma
            #scipy_dist = gumbel_l(mu, beta)
            mu = mu - beta*np.euler_gamma
            scipy_dist = gumbel_r(mu, beta)
            attempts = scipy_dist.rvs(size=n_periods)

        else:
            mu, sigma = point['mu'], point['sigma']
            beta = (1/np.pi)*np.sqrt(6)*sigma
            # mu = mu - beta*np.euler_gamma
            # scipy_dist = gumbel_r(mu, beta)
            mu = mu + beta*np.euler_gamma
            scipy_dist = gumbel_l(mu, beta)
            attempts = scipy_dist.rvs(size=n_periods)

      elif attempts == 'weibull':

          alpha, beta = point['alpha'], point['beta']
          attempts = beta*np.random.weibull(a=alpha, size=n_periods)
      
      else: 
          raise NotImplementedError

      # Switch between prior predictive and forecasting
      if which_random == 0:
          pass
      else:
          attempts = np.concatenate([past_obs, attempts])
      
      if kind == 'max':
        sample_path = np.maximum.accumulate(attempts)
      elif kind == 'min':
        sample_path = np.minimum.accumulate(attempts)
        
      return sample_path
      
    return _random