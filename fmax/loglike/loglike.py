import pymc3 as pm

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






