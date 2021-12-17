import pymc3 as pm
import numpy as np

def get_loglikelihood_fn(
    attempts = "gaussian",
    kind = 'min',
    ):
    """ Build a logp function for a cumulative 
    """
    if attempts == 'gaussian':
      generic_x_dist = pm.Normal.dist
    
    elif attempts == 'gumbel':
      generic_x_dist = pm.Gumbel.dist
      
    elif attempts == 'weibull':
      generic_x_dist = pm.Weibull.dist
    
    else: raise NotImplementedError
    
    def _logp(jump_data, flat_data, **kwargs):
        """ Likelihood function
        """
        ## Instantiate underlying distribution
        x_dist = generic_x_dist(**kwargs)
        
        # Add likelihood contribution from the jump data
        log_likelihood = pm.math.sum(x_dist.logp(jump_data))

        # Add likelihood contribution from the flat data
        if kind == 'max':
          log_likelihood += pm.math.sum(x_dist.logcdf(flat_data))
                            
        if kind == 'min':
          log_likelihood += pm.math.sum(
                            pm.math.log1mexp(
                            -x_dist.logcdf(flat_data)
                            ))
        else: raise ValueError("`kind` must be 'max' or 'min'")
        
        return log_likelihood
    
    return _logp
