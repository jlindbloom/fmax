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
        kind="max", 
        attempts="gaussian", 
        train="all"
        ):
        self.record_data = record_data
        self.kind = kind
        self.attempts = attempts
        self.n_obs = len(record_data)
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

    def fit(self, 
    
        attempts_mean_mu=11, 
        attempts_mean_sigma=3,
        attempts_stdev_lam=1, 
        
        fcast_len=0,
        
        chains=2, 
        draws=20000, 
        tune=5000, 
        ):
        """Fits a PyMC model to the training data.
        """

        # WIP: figure out how to choose priors reasonably from data

        with pm.Model() as self.pymc_model:
            # Initialize priors
            mu = pm.Normal('mu', mu=attempts_mean_mu, sigma=attempts_mean_sigma)
            sigma = pm.Exponential('sigma', lam=attempts_stdev_lam)
            
            # Get random sampling and likelihood for the kind of attempt
            loglike = fm.get_loglikelihood_fn(
                          attempts = "gaussian",
                          kind = 'min',
                          )
            
            random_sampler = fm.get_random_fn(
                                 n_periods=fcast_len, 
                                 past_obs=self.train_data,
                                 kind=self.kind
                                 )
            
            likelihood = pm.DensityDist('running_max', 
                                        loglike, random=random_sampler, 
                                        observed = {'jump_data':self.jump_data, 
                                                    'flat_data':self.flat_data, 
                                                    'mu': mu, 
                                                    'sigma': sigma}
                                        )
            
            self.trace = pm.sample(draws=draws, 
                                   chains=chains, 
                                   tune=tune, 
                                   cores=1, 
                                   target_accept=0.99,
                                   return_inferencedata=True, 
                                   idata_kwargs={"density_dist_obs": False}
                                   )            


    def draw_forecasts(self):
        """Samples the posterior predictive (includes past and future).
        """                    
        with self.pymc_model:
            self.ppc = pm.sample_posterior_predictive(self.trace)
