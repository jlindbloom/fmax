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
        # beta = pm.math.sqrt((6/(np.pi**2))*(sigma**2))
        # mu = (-mu) - beta*np.euler_gamma
        beta = (1/np.pi)*np.sqrt(6)*sigma
        mu = mu - beta*np.euler_gamma
        attempts = -np.random.gumbel(mu, beta, n_periods)

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

      #print(which_random)
      #print(point)

      # If posterior predictive, ignore original n_periods
      if which_random == 0:
          n_periods = len(past_obs)

      if attempts == 'gaussian':
          mu, sigma = point['mu'], point['sigma']
          attempts = mu + sigma*np.random.randn(n_periods)
      
      elif attempts == 'gumbel':
          mu, sigma = point['mu'], point['sigma']
          beta = (1/np.pi)*np.sqrt(6)*sigma
          mu = mu - beta*np.euler_gamma
          attempts = np.random.gumbel(mu, beta, n_periods)
          
          # This accounts for flipping the gumbel draws if using a min model 
          if kind == 'min':
              attempts *= -1

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