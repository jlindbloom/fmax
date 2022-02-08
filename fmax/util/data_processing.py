import numpy as np


def get_max(series):
    """Given a sequence of attempts, outputs the corresponding cumulative max.
    """
    return np.maximum.accumulate(series)


def get_min(series):
    """Given a sequence of attempts, outputs the corresponding cumulative min.
    """
    return np.minimum.accumulate(series)


def jump_flat_split(series, kind="max"):
    """Splits an series of a running max/min into the jump and flat components.
    """

    if kind == "max":
        jump_mask = np.insert(np.diff(series) > 0, 0, True) 
        jump_data = series[jump_mask]
        flat_data = series[~jump_mask]

    elif kind == "min":
        jump_mask = np.insert(np.diff(series) < 0, 0, True) 
        jump_data = series[jump_mask]
        flat_data = series[~jump_mask]

    else:
        raise NotImplementedError

    return jump_data, flat_data






