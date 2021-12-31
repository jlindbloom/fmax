import numpy as np


def mse(forecast, actual):
    """Calculates the mean square error of a forecast.
    """
    return np.mean((forecast-actual)**2)