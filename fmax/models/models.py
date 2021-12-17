import numpy as np
import pymc3 as pm

import fmax as fm


class ForecastModel:
    """Model class for handling forecasting given observed sequence of attempts.
    """
    def __init__(self, record_data, time_index=None, kind="max", attempts="gaussian", train="all", fcast_len=30):
        self.record_data = record_data
        self.kind = kind
        self.attempts = attempts
        self.n_obs = len(record_data)
        self.train = train
        self.fcast_len = fcast_len

        # Handle time index
        if time_index is None:
            self.master_index = [i for i in range(self.n_obs)]
            self.fcast_index = [i for i in range(self.n_obs, self.fcast_len)]
            self.master_with_fcast_index = [i for i in range(self.n_obs+self.fcast_len)]
            self.tot_index_len = self.n_obs + self.fcast_len
        else:
            raise NotImplementedError
            self.master_index = time_index
            self.fcast_index = [i for i in range(self.n_obs, self.fcast_len)]
            self.master_with_fcast_index = np.concatenate(self.master_index, self.fcast_index)
            self.tot_index_len = len(self.fcast_index) + len(self.master_with_fcast_index)
            
        # Split into training and testing:
        if self.train == "all":
            self.train_data = self.record_data
            self.train_index = self.master_index
            self.test_data = None
            self.test_index = None
            self.fcast_index = [i for i in range(len(self.train_data), len(self.train_data) + self.fcast_len)]
        else:
            idx_train_max = int(train*len(self.master_index))
            self.train_data = self.record_data[:idx_train_max]
            self.train_index = self.master_index[:idx_train_max]
            self.test_data = self.record_data[idx_train_max:]
            self.test_index = self.master_index[idx_train_max:]
            self.fcast_index = [i for i in range(len(self.train_index) + self.fcast_len)]

        # Get jump/flat data
        self.jump_data, self.flat_data = fm.jump_flat_split(self.train_data, kind=self.kind)


    def fit(self, attempts_mean_mu=11, attempts_mean_sigma=3,
                            attempts_stdev_lam=1, chains=2, draws=20000, tune=5000):
        """Fits a PyMC model to the training data.
        """

        # WIP: figure out how to choose priors reasonably from data

        with pm.Model() as self.pymc_model:
            mu = pm.Normal('mu', mu=attempts_mean_mu, sigma=attempts_mean_sigma)
            sigma = pm.Exponential('sigma', lam=attempts_stdev_lam)

        if self.attempts == "gaussian":
            
            random_sampler = fm.gaussian_random(n_periods=self.fcast_len, past_obs=self.train_data)

            with self.pymc_model:
                loglike = fm.gaussian_attempts
                #global fm.gaussian_random
                likelihood = pm.DensityDist('running_max', loglike, random=random_sampler, 
                                observed = {'jump_data':self.jump_data, 
                                            'flat_data':self.flat_data, 
                                            'mu': mu, 
                                            'sigma': sigma})
        
        elif self.attempts == "gumbel":
            

            if self.kind == "min":

                random_sampler = fm.gumbel_random_min(n_periods=self.fcast_len, past_obs=self.train_data)
                
                with self.pymc_model:
                    loglike = fm.gumbel_attempts_min
                    #global fm.gaussian_random
                    likelihood = pm.DensityDist('running_max', loglike, random=random_sampler, 
                                    observed = {'jump_data':self.jump_data, 
                                                'flat_data':self.flat_data, 
                                                'mu': mu, 
                                                'sigma': sigma})
        
        else:
            
            raise NotImplementedError

        with self.pymc_model:
            
            self.trace = pm.sample(draws=draws, chains=chains, tune=tune, cores=1, target_accept=0.99,
                    return_inferencedata=True, 
                    idata_kwargs={"density_dist_obs": False})            


    def draw_forecasts(self):
        """Samples the posterior predictive (includes past and future).
        """                    
        with self.pymc_model:
            self.ppc = pm.sample_posterior_predictive(self.trace)


    

        
