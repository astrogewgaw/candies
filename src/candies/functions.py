import numpy as np
from scipy.signal import detrend

K: float = 4.1488064239e3


def znorm(X):
    X = X.astype(np.float32)
    X = np.nan_to_num(X)
    X = detrend(X)
    X = X - np.median(X)
    X = X / np.std(X)
    X = np.nan_to_num(X)
    return X


def dm2delay(f: float, f0: float, dm: float) -> float:
    return K * dm * (f**-2 - f0**-2)


def delay2dm(f: float, f0: float, t: float) -> float:
    return t / (K * (f**-2 - f0**-2))
