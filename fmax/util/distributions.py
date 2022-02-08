import pymc3 as pm


class MinGumbel(pm.Continuous):
  def __init__(self, mu, beta):
    self.mu = mu
    self.beta = beta
  
  def logp(self, x):
    y = (x - self.mu) / self.beta
    return y - pm.math.exp(y) - pm.math.log(self.beta)
  
  def logcdf(self, x):
    y = (x - self.mu) / self.beta
    return pm.math.log(1-pm.math.exp(-pm.math.exp(y)))


class MaxWeibull(pm.Continuous):
  def __init__(self, alpha, beta):
    self.alpha = alpha
    self.beta = beta
  
  def logp(self, x):
    y = (x) / self.beta
    return pm.math.log(self.alpha) + (self.alpha - 1)*pm.math.log(-y) \
           - (-y)**self.alpha - pm.math.log(beta)

  def logcdf(self, x):
    y = (x) / self.beta
    return -(-y)**self.alpha