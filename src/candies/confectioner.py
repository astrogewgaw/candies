import os
import arrow
import click
import logging
import numpy as np
import pandas as pd

from attrs import define
from pathlib import Path
from multiprocessing import Pool
from rich.logging import RichHandler
from priwo.sigproc.hdr import readhdr

from candies.utensils import delay


os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


@define
class Candy:

    """
    Represents a candidate.
    """

    nf: int
    nt: int
    fl: float
    fh: float
    df: float
    dt: float
    dm: float
    t0: float
    w50: float
    label: int
    snr: float
    device: int
    fn: str | Path
    data: np.ndarray

    @classmethod
    def get(
        cls,
        fn: str,
        dm: float,
        t0: float,
        snr: float,
        w50: float,
        label: int,
        device: int | None,
    ):
        hdr = readhdr(fn)

        fh = hdr["fch1"]
        df = hdr["foff"]
        dt = hdr["tsamp"]
        nf = hdr["nchans"]
        fl = fh - (df * nf)
        ti = t0 - delay(dm, fl, fh) - (w50 * dt)
        tf = t0 + delay(dm, fl, fh) + (w50 * dt)

        nt = int((tf - ti) / dt)
        with open(fn, "rb") as f:
            f.seek(hdr["size"])
            data = np.fromfile(
                f,
                dtype=np.uint8,
                count=(nt * nf),
                offset=(int(ti / dt) - 1) * nf,
            )

        return cls(
            fn=fn,
            dm=dm,
            nt=nt,
            t0=t0,
            fh=fh,
            fl=fl,
            df=df,
            dt=dt,
            nf=nf,
            w50=w50,
            snr=snr,
            data=data,
            label=label,
            device=(0 if device is None else device),
        )
