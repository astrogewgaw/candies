"""
Miscellaneous utilities for Candies.
"""

import csv
import numpy as np
from pathlib import Path

kdm: float = 4.1488064239e3


def dispersive_delay(
    f: float,
    f0: float,
    dm: float,
) -> float:
    """
    Calculates the dispersive delay (in s) for a particular frequency,
    given a value of the dispersion measure (DM, in pc per cm^-3) and
    the reference frequency. Both frequencies must be in MHz.
    """
    return kdm * dm * (f**-2 - f0**-2)


def dmt_extent(
    fl: float,
    fh: float,
    dt: float,
    t0: float,
    dm: float,
    wbin: float,
    fudge: float = 16,
):
    """
    Calculate the extent of a dispersed burst in the DM v/s time plane until
    a certain percentage drop in SNR. The SNR drop is determined via a fudge
    factor = (ΔSNR)**-2; so, for instance, a 25% drop in SNR is equivalent to
    a fudge factor of 16 (which is the default value).
    """
    width = wbin * dt
    tbin = np.round(t0 / dt).astype(int)
    ddm = (fudge * width) / (kdm * (fl**-2 - fh**-2))
    dbin = np.round(kdm * ddm * (fl**-2 - fh**-2) / dt).astype(int)
    t_range = (tbin - dbin, tbin + dbin)
    dm_range = (dm - ddm, dm + ddm) if ddm < dm else (0.0, 2.0 * dm)
    return (*t_range, *dm_range)


def read_csv(f: str | Path) -> list[dict[str, int | float]]:
    """
    Read in candy-dates from a CSV file.
    """
    with open(f, "r") as fp:
        return [
            {
                "t0": float(row[2]),
                "dm": float(row[4]),
                "snr": float(row[1]),
                "wbin": 2 ** int(row[3]),
            }
            for row in list(csv.reader(fp))[1:]
        ]


def read_presto(f: str | Path) -> list[dict[str, int | float]]:
    """
    Read in candy-dates from a PRESTO *.singlepulse file.
    """
    return [
        {
            "t0": float(row[2]),
            "dm": float(row[0]),
            "snr": float(row[1]),
            "wbin": int(row[3]),
        }
        for row in np.loadtxt(f, usecols=(0, 1, 2, 4))
    ]


def read_astroaccelerate(f: str | Path) -> list[dict[str, int | float]]:
    """
    Read in candy-dates from an AstroAccelerate *.dat file.
    """
    return [
        {
            "t0": float(row[1]),
            "dm": float(row[0]),
            "snr": float(row[2]),
            "wbin": int(row[3]),
        }
        for row in np.fromfile(f, dtype=np.float32).reshape(-1, 4)
    ]
