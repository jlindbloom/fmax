import pymc3 as pm
import numpy as np

def gaussian_attempts(jump_data, flat_data, mu, sigma):
    """Log likelihood with a Gaussian attempt distribution.
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

    return log_likelihood


def gumbel_attempts_min(jump_data, flat_data, mu, sigma):
    """Log likelihood with a Gumbel attempt distribution.
    """

    beta = pm.math.sqrt((6/(np.pi**2))*(sigma**2))
    mu = (-mu) - beta*np.euler_gamma

    x_dist = pm.Gumbel.dist(mu=mu, beta=beta)
    
    # Add likelihood contribution from the jump data
    log_likelihood = pm.math.sum(x_dist.logp(-jump_data))

    # Contribution from the flat data
    log_likelihood += pm.math.log1mexp(x_dist.logcdf(-flat_data))

    return log_likelihood




