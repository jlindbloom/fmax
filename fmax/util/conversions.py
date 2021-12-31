from scipy.special import gamma, digamma
from scipy import optimize
import numpy as np

def weibull_mu_and_sigma(alpha, beta):
    """Given the shape (alpha) and scale (beta) parameters of a Weibull distribution,
    outputs the corresponding mean and standard deviation as a 2-tuple.
    """
    mu = beta*gamma(1 + (1/alpha))
    sigma = (beta**2)*gamma(1 + (2/alpha) - (mu**2)/(beta**2))
    return mu, sigma



def weibull_alpha_and_beta(mu, sigma):
    """Given the mean (mu) and standard deviation (sigma) parameters of a Weibull distribution,
    outputs the corresponding shape (alpha) and scale (beta) as a 2-tuple. 
    Note: This isn't working very well right now.
    """

    def fun(x, mu, sigma):
        """Function called for solving the system for alpha and beta given a mu and sigma.
        First entry is in x alpha, second is beta.
        """
        alpha = x[0]
        beta = x[1]

        f_val = [beta*gamma(1 + (1/alpha)) - mu,
                (beta**2)*gamma(1 + (2/alpha) - (mu**2)/(beta**2)) - sigma]

        df1_dalpha = -beta*gamma(1 + (1/alpha))*digamma(1 + (1/alpha))/(alpha**2)
        df1_dbeta = gamma(1 + (1/alpha))
        
        temp = 1 + (2/alpha) - ((mu**2)/(beta**2))
        df2_dalpha = -2*(beta**2)*gamma(temp)*digamma(temp)/(alpha**2)
        df2_dbeta = 2*gamma(temp)*( (mu**2)*digamma(temp) + (beta**2))/beta


        f_jac = np.array([[df1_dalpha, df1_dbeta],
                        [df2_dalpha, df2_dbeta]])

        return f_val, f_jac

    res = optimize.root(fun, [1.0, 1.0], args=(mu,sigma), jac=True)

    if res.success == False:
        raise Exception("Solver for alpha and beta failed.")
    else:
        alpha, beta = res.x
        return alpha, beta