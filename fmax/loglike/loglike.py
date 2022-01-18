import pymc3 as pm
import numpy as np
import theano.tensor as tt

def gaussian_attempts_min(jump_data, flat_data, mu, sigma):
    """Log likelihood for min with a Gaussian attempt distribution.
    """
    x_dist = pm.Normal.dist(mu=mu, sigma=sigma)
    
    # Add likelihood contribution from the jump data
    log_likelihood = pm.math.sum(x_dist.logp(jump_data))

    # Compute distance of flat data from mean
    mu_distance = mu - flat_data
    
    # Now get the value reflected across the mean. 
    # Taking the logcdf of this value gives us the desired 
    # probability on a log scale
    flat_data_reflected = mu + mu_distance
    log_likelihood += pm.math.sum(x_dist.logcdf(flat_data_reflected))

    # Why does this not work????
    # log_likelihood += pm.math.sum(
    #                         pm.math.log1mexp(
    #                         x_dist.logcdf(flat_data)
    #                         ))

    return log_likelihood

def gaussian_attempts_max(jump_data, flat_data, mu, sigma):
    """Log likelihood for max with a Gaussian attempt distribution.
    """
    x_dist = pm.Normal.dist(mu=mu, sigma=sigma)
    
    # Add likelihood contribution from the jump data
    log_likelihood = pm.math.sum(x_dist.logp(jump_data))

    # Add likelihood contribution from the flat data
    log_likelihood += pm.math.sum(x_dist.logcdf(flat_data))

    return log_likelihood
    

def gumbel_attempts_min(jump_data, flat_data, mu, sigma):
    """Log likelihood for min with a Gumbel attempt distribution.
    """

    beta = (1/np.pi)*pm.math.sqrt(6)*sigma
    #pm.math.sqrt((6/(np.pi**2))*(sigma**2))
    mu = mu - beta*np.euler_gamma

    x_dist = pm.Gumbel.dist(mu=mu, beta=beta)
    
    # Add likelihood contribution from the jump data
    log_likelihood = pm.math.sum(x_dist.logp(jump_data))

    # Contribution from the flat data
    #log_likelihood += pm.math.sum(pm.math.log1mexp(x_dist.logcdf(flat_data))) # why does this have trouble?
    log_likelihood += pm.math.sum(pm.math.log(1 - pm.math.exp(x_dist.logcdf(-flat_data))))

    return log_likelihood


def gumbel_attempts_max(jump_data, flat_data, mu, sigma):
    """Log likelihood for max with a Gumbel attempt distribution.
    """

    beta = (1/np.pi)*pm.math.sqrt(6)*sigma
    #pm.math.sqrt((6/(np.pi**2))*(sigma**2))
    mu = mu - beta*np.euler_gamma

    x_dist = pm.Gumbel.dist(mu=mu, beta=beta)
    
    # Add likelihood contribution from the jump data
    log_likelihood = pm.math.sum(x_dist.logp(jump_data))

    # Add likelihood contribution from the flat data
    log_likelihood += pm.math.sum(x_dist.logcdf(flat_data))

    return log_likelihood


def weibull_attemps_min(jump_data, flat_data, alpha, beta):
    """Log likelihood for min with a Weibull attempt distribution/
    """
    x_dist = pm.Weibull.dist(alpha=alpha, beta=beta)
    
    # Add likelihood contribution from the jump data
    log_likelihood = pm.math.sum(x_dist.logp(jump_data))

    # Add likelihood contribution from the flat data
    # log_likelihood += pm.math.sum(
    #                         pm.math.log1mexp(
    #                         -x_dist.logcdf(flat_data)
    #                         ))
    log_likelihood += pm.math.sum(pm.math.log(1 - pm.math.exp(x_dist.logcdf(-flat_data))))

    return log_likelihood


def weibull_attemps_max(jump_data, flat_data, alpha, beta):
    """Log likelihood for max with a Weibull attempt distribution/
    """
    x_dist = pm.Weibull.dist(alpha=alpha, beta=beta)
    
    # Add likelihood contribution from the jump data
    log_likelihood = pm.math.sum(x_dist.logp(jump_data))

    # Add likelihood contribution from the flat data
    log_likelihood += pm.math.sum(x_dist.logcdf(flat_data))

    return log_likelihood
    

def log1mexp(x):
    r"""Return log(1 - exp(-x)).
    This function is numerically more stable than the naive approach.
    For details, see
    https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    References
        ----------
        .. [Machler2012] Martin Mächler (2012).
            "Accurately computing `\log(1-\exp(- \mid a \mid))` Assessed by the Rmpfr
            package"
    """
    return tt.switch(tt.lt(x, 0.6931471805599453), tt.log(-tt.expm1(-x)), tt.log1p(-tt.exp(-x)))



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
                # I'm not sure why I have to do this but I think it's unstable
                log_likelihood += pm.math.sum(pm.math.log(1 - pm.math.exp(x_dist.logcdf(-flat_data))))

            #   log_likelihood += pm.math.sum(
            #                     pm.math.log1mexp(
            #                     -x_dist.logcdf(flat_data)
            #                     ))
            else: raise ValueError("`kind` must be 'max' or 'min'")
            
            return log_likelihood


    elif attempts == "gumbel":

        generic_x_dist = pm.Gumbel.dist


        def _logp(jump_data, flat_data, mu, sigma):
            """ Likelihood function
            """

            beta = (1/np.pi)*pm.math.sqrt(6)*sigma
            #pm.math.sqrt((6/(np.pi**2))*(sigma**2))
            mu = mu - beta*np.euler_gamma

            ## Instantiate underlying distribution
            x_dist = generic_x_dist(mu=mu, beta=beta)
            
            # Add likelihood contribution from the jump data
            log_likelihood = pm.math.sum(x_dist.logp(jump_data))

            # Add likelihood contribution from the flat data
            if kind == 'max':
                log_likelihood += pm.math.sum(x_dist.logcdf(flat_data))
                                
            elif kind == 'min':
                # I'm not sure why I have to do this but I think it's unstable
                log_likelihood += pm.math.sum(pm.math.log(1 - pm.math.exp(x_dist.logcdf(-flat_data))))

                #log_likelihood += pm.math.sum(log1mexp(-x_dist.logcdf(-flat_data)))

            else: raise ValueError("`kind` must be 'max' or 'min'")
            
            return log_likelihood
    

    else: raise NotImplementedError
        

    return _logp






# def gumbel_attempts_min(jump_data, flat_data, mu, sigma):
#     """Log likelihood for min with a Gumbel attempt distribution.
#     """

#     beta = (1/np.pi)*pm.math.sqrt(6)*sigma
#     #pm.math.sqrt((6/(np.pi**2))*(sigma**2))
#     mu = mu - beta*np.euler_gamma

#     x_dist = pm.Gumbel.dist(mu=mu, beta=beta)
    
#     # Add likelihood contribution from the jump data
#     log_likelihood = pm.math.sum(x_dist.logp(jump_data))

#     # Contribution from the flat data
#     #log_likelihood += pm.math.sum(pm.math.log1mexp(x_dist.logcdf(flat_data))) # why does this have trouble?
#     log_likelihood += pm.math.sum(pm.math.log(1 - pm.math.exp(x_dist.logcdf(-flat_data))))

#     return log_likelihood







    
    # elif attempts == 'gumbel':
    #   generic_x_dist = pm.Gumbel.dist
      
    # elif attempts == 'weibull':
    #   generic_x_dist = pm.Weibull.dist
    
    # else: raise NotImplementedError
    
    # def _logp(jump_data, flat_data, **kwargs):
    #     """ Likelihood function
    #     """
    #     ## Instantiate underlying distribution
    #     x_dist = generic_x_dist(**kwargs)
        
    #     # Add likelihood contribution from the jump data
    #     log_likelihood = pm.math.sum(x_dist.logp(jump_data))

    #     # Add likelihood contribution from the flat data
    #     if kind == 'max':
    #       log_likelihood += pm.math.sum(x_dist.logcdf(flat_data))
                            
    #     if kind == 'min':
    #         # I'm not sure why I have to do this but I think it's unstable
    #         log_likelihood += pm.math.sum(pm.math.log(1 - pm.math.exp(x_dist.logcdf(-flat_data))))

    #     #   log_likelihood += pm.math.sum(
    #     #                     pm.math.log1mexp(
    #     #                     -x_dist.logcdf(flat_data)
    #     #                     ))
    #     else: raise ValueError("`kind` must be 'max' or 'min'")
        
    #     return log_likelihood
    
    # return _logp


