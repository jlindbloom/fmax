import pymc3 as pm
import numpy as np
import theano.tensor as tt
from fmax import MinGumbel, MaxWeibull


def get_loglikelihood_fn(
    attempts = "gaussian",
    kind = 'min',
    ):
    """ Build a logp function for a cumulative 
    """

    if attempts == 'gaussian':

        generic_x_dist = pm.Normal.dist

        def _logp(jump_data, flat_data, mu, sigma):
            """ Likelihood function
            """
            ## Instantiate underlying distribution
            x_dist = generic_x_dist(mu=mu, sigma=sigma)
            
            # Add likelihood contribution from the jump data
            log_likelihood = pm.math.sum(x_dist.logp(jump_data))

            # Add likelihood contribution from the flat data
            if kind == 'max':
                log_likelihood += pm.math.sum(x_dist.logcdf(flat_data))
                                
            elif kind == 'min':

                log_likelihood += pm.math.sum(log1mexp(-x_dist.logcdf(-flat_data)))

            else: raise ValueError("`kind` must be 'max' or 'min'")
            
            return log_likelihood


    elif attempts == "gumbel":

        if kind == "min":
            #generic_x_dist = MinGumbel.dist
            generic_x_dist = pm.Gumbel.dist
        elif kind == "max":
            #generic_x_dist = pm.Gumbel.dist
            generic_x_dist = MinGumbel.dist
        else: raise ValueError("`kind` must be 'max' or 'min'")
        
        def _logp(jump_data, flat_data, mu, sigma):
            """ Likelihood function
            """
            # The change to (mu, beta) parameterization depends on whether max/min!

            if kind == "min":

                beta = (1/np.pi)*pm.math.sqrt(6)*sigma
                #mu = mu + beta*np.euler_gamma
                mu = mu - beta*np.euler_gamma

                ## Instantiate underlying distribution
                x_dist = generic_x_dist(mu=mu, beta=beta)
                
                # Add likelihood contribution from the jump data
                log_likelihood = pm.math.sum(x_dist.logp(jump_data))

                # Add contribution from the flat data
                log_likelihood += pm.math.sum(log1mexp(-x_dist.logcdf(-flat_data)))

            else:

                beta = (1/np.pi)*pm.math.sqrt(6)*sigma
                #mu = mu - beta*np.euler_gamma
                mu = mu + beta*np.euler_gamma

                ## Instantiate underlying distribution
                x_dist = generic_x_dist(mu=mu, beta=beta)
                
                # Add likelihood contribution from the jump data
                log_likelihood = pm.math.sum(x_dist.logp(jump_data))

                # Add likelihood contribution from the flat data
                log_likelihood += pm.math.sum(x_dist.logcdf(flat_data))
            
            return log_likelihood
    

    elif attempts == "weibull":

        if kind == "min":
            generic_x_dist = pm.Weibull.dist
            #generic_x_dist = MaxWeibull.dist
        elif kind == "max":
            generic_x_dist = MaxWeibull.dist
            #generic_x_dist = pm.Weibull.dist
        else: raise ValueError("`kind` must be 'max' or 'min'")
        
        def _logp(jump_data, flat_data, alpha, beta):
            """ Likelihood function
            """

            ## Instantiate underlying distribution
            x_dist = generic_x_dist(alpha=alpha, beta=beta)
            
            # Add likelihood contribution from the jump data
            log_likelihood = pm.math.sum(x_dist.logp(jump_data))

            # Add likelihood contribution from the flat data
            if kind == 'max':
                log_likelihood += pm.math.sum(x_dist.logcdf(flat_data))
                                
            else:

                log_likelihood += pm.math.sum(log1mexp(-x_dist.logcdf(-flat_data)))
            
            return log_likelihood
    

    else: raise NotImplementedError
        
    return _logp


def log1mexp(x):
    r"""This is ripped straight from PyMC.
    """
    return tt.switch(tt.lt(x, 0.6931471805599453), tt.log(-tt.expm1(-x)), tt.log1p(-tt.exp(-x)))
