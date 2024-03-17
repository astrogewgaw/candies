"""
Utility functions for Candies.
"""

import numpy as np

"""
The value of the dispersion constant.
"""
kdm: float = 4.1488064239e3


def normalise(data: np.ndarray):
    """
    Normalise the data using a MAD-based prescription.

    Parameters
    ----------
    data: np.ndarray
        The data to normalise as a Numpy array.

    Returns
    -------
    The normalised data as a Numpy array.
    """
    data = np.array(data, dtype=np.float32)
    data -= np.median(data)
    data /= np.std(data)
    return data


def dm2delay(f: float, fref: float, dm: float) -> float:
    """
    Calculate the delay corresponding to a particular DM.

    Parameters
    ----------
    f: float
        The frequency (in MHz) for which the delay needs to be calculated.
    fref: float
        The frequency (in MHz) w.r.t. which the delay will be calculated.
    dm: float
        The DM (in pc cm^-3) for which the delay will be calculated.

    Returns
    -------
    The dispersive delay (in seconds) corresponding to the above values.
    """
    return kdm * dm * (f**-2 - fref**-2)


def delay2dm(f: float, fref: float, t: float) -> float:
    """
    Calculate the DM corresponding to a particular delay.

    Parameters
    ----------
    f: float
        The frequency (in MHz) for which the delay was calculated.
    fref: float
        The frequency (in MHz) w.r.t. which the delay was calculated.
    t: float
        The dispersive delay (in seconds).

    Returns
    -------
    The dispersion measure corresponding to the above values.
    """
    return t / (kdm * (f**-2 - fref**-2))
