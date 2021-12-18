import numpy as np
import pymc3 as pm

import fmax as fm

class ForecastModel:
    """Model class for handling forecasting given 
       observed sequence of attempts.
    """
    def __init__(self, 
        record_data, 
        time_index=None, 
        fcast_len = 0,
        
        prior_parameters = {
          'mu' : {
            'mean' : 11,
            'std' : 3,
          },
          'sigma' : {
            'lam' : 1
          }
        },
        
        kind="max", 
        attempt_distribution="gaussian", 
        train="all"
        ):
        self.record_data = record_data
        self.kind = kind
        self.attempt_distribution = attempt_distribution
        self.n_obs = len(record_data)
        self.fcast_len = fcast_len
        self.train = train

        # Handle time index
        if time_index is None:
            self.master_index = [i for i in range(self.n_obs)]
            self.tot_index_len = self.n_obs
        else:
            raise NotImplementedError
            self.master_index = time_index
            
        # Split into training and testing:
        if self.train == "all":
            self.train_data = self.record_data
            self.train_index = self.master_index
            self.test_data = None
            self.test_index = None
        else:
            idx_train_max = int(train*len(self.master_index))
            self.train_data = self.record_data[:idx_train_max]
            self.train_index = self.master_index[:idx_train_max]
            self.test_data = self.record_data[idx_train_max:]
            self.test_index = self.master_index[idx_train_max:]

        # Get jump/flat data
        self.jump_data, self.flat_data =\
          fm.jump_flat_split(self.train_data, kind=self.kind)
        
        # Init PyMC3 model
        self.init_pymc_model(prior_parameters)
    
    def init_pymc_model(self, prior_parameters):
        """ Create a PyMC3 model
        """
        # Define model
        with pm.Model() as self.pymc_model:
            # Initialize priors for the distribution of each attempt
            attempts_mean_mu = prior_parameters['mu']['mean']
            attempts_mean_sigma = prior_parameters['mu']['std']
            attempts_stdev_lam = prior_parameters['sigma']['lam']
            
            priors = {
              'mu' : pm.Normal('mu', mu=attempts_mean_mu, sigma=attempts_mean_sigma),
              'sigma' : pm.Exponential('sigma', lam=attempts_stdev_lam),
            }
            
            # Get random sampling and likelihood for the kind of attempt
            loglike = fm.get_loglikelihood_fn(
                          attempts = self.attempt_distribution,
                          kind = self.kind,
                          )
                          
            posterior_predictive_sampler = fm.get_random_fn(
                                 n_periods=self.n_obs, 
                                 kind=self.kind
                                 )
            
            likelihood = pm.DensityDist('running_record', 
                                        loglike, random=posterior_predictive_sampler, 
                                        observed = {'jump_data':self.jump_data, 
                                                    'flat_data':self.flat_data, 
                                                    **priors}
                                        )
            
            # We create a second sampling function so we can sample
            # the extrapolated distribution of records later
            forecast_sampler = fm.get_random_fn(
                                 n_periods=self.fcast_len, 
                                 past_obs=self.train_data,
                                 kind=self.kind
                                 )
            
            forecasting_likelihood = pm.DensityDist('forecast', 
                                        loglike, random=forecast_sampler, 
                                        observed = priors
                                        )

    def fit(self, 
        chains=2, 
        draws=20000, 
        tune=5000, 
        ):
        """Fits a PyMC model to the training data.
        """
        
        with self.pymc_model:
            self.trace = pm.sample(draws=draws, 
                                   chains=chains, 
                                   tune=tune, 
                                   cores=1, 
                                   target_accept=0.99,
                                   return_inferencedata=True, 
                                   idata_kwargs={"density_dist_obs": False}
                                   )

    def forecast(self):
        """Samples the posterior predictive (includes past and future).
        """
        with self.pymc_model:
            self.ppc = pm.sample_posterior_predictive(self.trace)
        
        return self.ppc['forecast']
    
    def posterior_predictive(self):
        """ Samples the posterior predictive distributions of the observations.
        """
        with self.pymc_model:
            self.ppc = pm.sample_posterior_predictive(self.trace)
        
        return self.ppc['running_record']
