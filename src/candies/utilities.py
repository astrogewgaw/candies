import numpy as np

kdm: float = 4.1488064239e3


def normalise(data: np.ndarray):
    data = np.array(data, dtype=np.float32)
    data -= np.median(data)
    data /= np.std(data)
    return data


def dm2delay(f: float, fref: float, dm: float) -> float:
    return kdm * dm * (f**-2 - fref**-2)


def delay2dm(f: float, fref: float, t: float) -> float:
    return t / kdm * (f**-2 - fref**-2)
