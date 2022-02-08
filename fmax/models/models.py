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
        train="all",
        fcast_test_data=None
        ):
        self.record_data = record_data
        self.kind = kind
        self.prior_parameters = prior_parameters
        self.attempt_distribution = attempt_distribution
        self.n_obs = len(record_data)
        self.fcast_len = fcast_len
        self.train = train
        self.fcast_test_data = fcast_test_data

        # Make sure fcast_len and test_data agree
        if self.fcast_test_data is not None:
            assert fcast_len == len(self.fcast_test_data), "fcast_len must equal len(fcast_test_data)."

        # Handle time index
        if time_index is None:
            self.master_index = [i for i in range(self.n_obs)]
            self.fcast_index = [i for i in range(self.n_obs, self.fcast_len)]
            self.master_with_fcast_index = [i for i in range(self.n_obs+self.fcast_len)]
            self.tot_index_len = self.n_obs + self.fcast_len
        else:
            print("I haven't done this case yet...")
            self.master_index = time_index

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

        # Init PyMC3 model
        self.init_pymc_model(self.prior_parameters)
    
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
            #loglike = fm.gumbel_attempts_min
                          
            # Create switch variable between posterior predictive and forecasting
            # 0 is posterior predictive, 1 is forecasting
            random_switch = pm.Data('random_switch', 0.0)

            # Random sampler              
            posterior_predictive_sampler = fm.get_random_fn(
                                 attempts=self.attempt_distribution,
                                 kind=self.kind,
                                 n_periods=self.fcast_len, 
                                 past_obs=self.train_data,
                                 )
            
            likelihood = pm.DensityDist('path', 
                                        loglike, random=posterior_predictive_sampler, 
                                        observed = {'jump_data':self.jump_data, 
                                                    'flat_data':self.flat_data, 
                                                    **priors}
                                        )

            # Track the log likelihood on the holdout set if available
            if self.fcast_test_data is not None:
                holdout_jump_data, holdout_flat_data = fm.jump_flat_split(self.fcast_test_data, kind=self.kind)
                log_like_holdout = pm.Deterministic("log_like_holdout", loglike(holdout_jump_data, holdout_flat_data, **priors))



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
            pm.set_data({'random_switch':1})
            self.forecast_ppc = pm.sample_posterior_predictive(self.trace)

        self.forecast_samples = self.forecast_ppc['path']

    
    def posterior_predictive(self):
        """ Samples the posterior predictive distributions of the observations.
        """
        with self.pymc_model:
            pm.set_data({'random_switch':0})
            if self.fcast_test_data is None:
                self.posterior_predictive_ppc = pm.sample_posterior_predictive(self.trace)
            else:
                self.posterior_predictive_ppc = pm.sample_posterior_predictive(self.trace, var_names=['path', 'log_like_holdout'])

        self.posterior_predictive_samples = self.posterior_predictive_ppc['path']
        



class WeibullForecastModel:
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
        train="all",
        fcast_test_data=None
        ):
        self.record_data = record_data
        self.kind = kind
        self.prior_parameters = prior_parameters
        self.attempt_distribution = attempt_distribution
        self.n_obs = len(record_data)
        self.fcast_len = fcast_len
        self.train = train
        self.fcast_test_data = fcast_test_data

        # Make sure fcast_len and test_data agree
        if self.fcast_test_data is not None:
            assert fcast_len == len(self.fcast_test_data), "fcast_len must equal len(fcast_test_data)."

        # Handle time index
        if time_index is None:
            self.master_index = [i for i in range(self.n_obs)]
            self.fcast_index = [i for i in range(self.n_obs, self.fcast_len)]
            self.master_with_fcast_index = [i for i in range(self.n_obs+self.fcast_len)]
            self.tot_index_len = self.n_obs + self.fcast_len
        else:
            print("I haven't done this case yet...")
            self.master_index = time_index

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

        # Init PyMC3 model
        self.init_pymc_model(self.prior_parameters)
    
    def init_pymc_model(self, prior_parameters):
        """ Create a PyMC3 model
        """

        # Define model
        with pm.Model() as self.pymc_model:

            # Initialize priors for the distribution of each attempt
            alpha_lower = prior_parameters["alpha"]["lower"] 
            alpha_upper = prior_parameters["alpha"]["upper"] 
            beta_lower = prior_parameters["beta"]["lower"] 
            beta_upper = prior_parameters["beta"]["upper"] 
            
            priors = {
              #'mu' : pm.Normal('mu', mu=attempts_mean_mu, sigma=attempts_mean_sigma),
              #'sigma' : pm.Exponential('sigma', lam=attempts_stdev_lam),
              "alpha": pm.Uniform("alpha", lower=alpha_lower, upper=alpha_upper),
              "beta": pm.Uniform("beta", lower=beta_lower, upper=beta_upper)
            }
            
            # Get random sampling and likelihood for the kind of attempt
            loglike = fm.get_loglikelihood_fn(
                          attempts = self.attempt_distribution,
                          kind = self.kind,
                          )
            #loglike = fm.gumbel_attempts_min
                          
            # Create switch variable between posterior predictive and forecasting
            # 0 is posterior predictive, 1 is forecasting
            random_switch = pm.Data('random_switch', 0.0)

            # Random sampler              
            posterior_predictive_sampler = fm.get_random_fn(
                                 attempts=self.attempt_distribution,
                                 kind=self.kind,
                                 n_periods=self.fcast_len, 
                                 past_obs=self.train_data,
                                 )
            
            likelihood = pm.DensityDist('path', 
                                        loglike, random=posterior_predictive_sampler, 
                                        observed = {'jump_data':self.jump_data, 
                                                    'flat_data':self.flat_data, 
                                                    **priors}
                                        )

            # Track the log likelihood on the holdout set if available
            if self.fcast_test_data is not None:
                holdout_jump_data, holdout_flat_data = fm.jump_flat_split(self.fcast_test_data, kind=self.kind)
                log_like_holdout = pm.Deterministic("log_like_holdout", loglike(holdout_jump_data, holdout_flat_data, **priors))



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
            pm.set_data({'random_switch':1})
            self.forecast_ppc = pm.sample_posterior_predictive(self.trace)

        self.forecast_samples = self.forecast_ppc['path']

    
    def posterior_predictive(self):
        """ Samples the posterior predictive distributions of the observations.
        """
        with self.pymc_model:
            pm.set_data({'random_switch':0})
            if self.fcast_test_data is None:
                self.posterior_predictive_ppc = pm.sample_posterior_predictive(self.trace)
            else:
                self.posterior_predictive_ppc = pm.sample_posterior_predictive(self.trace, var_names=['path', 'log_like_holdout'])

        self.posterior_predictive_samples = self.posterior_predictive_ppc['path']
        

