"""
Generally applicable code for Candies.
"""

import mmap
import h5py as h5
import numpy as np
from attrs import define
from pathlib import Path
from priwo import readhdr
from typing_extensions import Self
from collections.abc import MutableSequence

from candies.utilities import (
    read_csv,
    read_presto,
    dispersive_delay,
    read_astroaccelerate,
)


class CandiesError(Exception):
    """
    All errors are CandiesErrors here.
    """

    pass


@define
class Candy:
    """
    Represents a candy-date.
    """

    # Candy-date parameters.
    t0: float
    dm: float
    wbin: int
    snr: float

    # Candy-date's data parameters.
    nf: int
    nt: int
    df: float
    dt: float
    bw: float
    fh: float
    fl: float
    tobs: float

    # Hidden internal parameters.
    skip: int
    mm: mmap.mmap

    # Optional parameters.
    ndms: int | None = None
    dmlow: float | None = None
    dmhigh: float | None = None
    dyn: np.ndarray | None = None
    dmt: np.ndarray | None = None

    @classmethod
    def wrap(
        cls,
        nf: int,
        nt: int,
        df: float,
        dt: float,
        bw: float,
        fh: float,
        fl: float,
        skip: int,
        t0: float,
        dm: float,
        snr: float,
        tobs: float,
        mm: mmap.mmap,
        wbin: int | float,
    ) -> Self:
        """
        Create and wrap up a candy-date.
        """
        return cls(
            nf=nf,
            nt=nt,
            df=df,
            dt=dt,
            bw=bw,
            fh=fh,
            fl=fl,
            t0=t0,
            dm=dm,
            mm=mm,
            snr=snr,
            tobs=tobs,
            skip=skip,
            wbin=int(wbin),
        )

    @property
    def id(self) -> str:
        """
        The candy-date's ID.
        """
        return f"T{self.t0:.7f}DM{self.dm:.5f}SNR{self.snr:.5f}"

    def chop(self) -> np.ndarray:
        """
        Chop out a time slice for a particular candy-date.

        This is done by taking the maximum delay at the candy-date's DM
        and width into account, and then chopping out a time slice around
        the candy-date's arrival time. In case the burst is close to the
        start or end of the file, and enough data is not available, this
        function will pad the array with median values.
        """
        maxdd = dispersive_delay(self.fl, self.fh, self.dm)
        ti = self.t0 - maxdd - (self.wbin * self.dt)
        tf = self.t0 + maxdd + (self.wbin * self.dt)
        return np.pad(
            np.frombuffer(
                self.mm,
                dtype=np.uint8,
                count=int((tf - (0.0 if ti < 0.0 else ti)) / self.dt) * self.nf,
                offset=int(0.0 if ti < 0.0 else ti / self.dt) * self.nf + self.skip,
            )
            .reshape(-1, self.nf)
            .T,
            (
                (0, 0),
                (
                    int(-ti / self.dt) if ti < 0.0 else 0,
                    int((tf - self.tobs) / self.dt) if tf > self.tobs else 0,
                ),
            ),
            mode="median",
        )

    def save(self):
        """
        Save a candy-date as an HDF5 dataset.
        """
        with h5.File(f"{self.id}.h5", "w") as f:
            f.attrs["nt"] = self.nt
            f.attrs["id"] = self.id

            if self.ndms is not None:
                f.attrs["ndms"] = self.ndms

            if self.dmlow is not None:
                f.attrs["dmlow"] = self.dmlow

            if self.dmhigh is not None:
                f.attrs["dmhigh"] = self.dmhigh

            f.attrs["nf"] = self.nf
            f.attrs["dt"] = self.dt
            f.attrs["df"] = self.df
            f.attrs["fl"] = self.fl
            f.attrs["fh"] = self.fh

            f.attrs["t0"] = self.t0
            f.attrs["dm"] = self.dm
            f.attrs["snr"] = self.snr
            f.attrs["wbin"] = self.wbin

            if self.dyn is not None:
                ftset = f.create_dataset(
                    "data_freq_time",
                    data=self.dyn,
                    compression="gzip",
                    compression_opts=9,
                )
                ftset.dims[0].label = b"time"
                ftset.dims[1].label = b"frequency"

            if self.dmt is not None:
                dmtset = f.create_dataset(
                    "data_dm_time",
                    data=self.dmt,
                    compression="gzip",
                    compression_opts=9,
                )
                dmtset.dims[0].label = b"dm"
                dmtset.dims[1].label = b"time"


@define
class Candies(MutableSequence):
    """
    Represents a sequence of candy-dates.
    """

    items: list[Candy]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def __delitem__(self, i):
        del self.items[i]

    def __setitem__(self, i, value):
        self.items[i] = value

    def insert(self, index, value):
        self.items.insert(index, value)

    @classmethod
    def wrap(cls, candfile: str | Path, datafile: str | Path) -> Self:
        """
        Create and wrap up candy-dates.
        """
        ext = Path(candfile).suffix
        if ext == ".csv":
            values = read_csv(candfile)
        elif ext == ".singlepulse":
            values = read_presto(candfile)
        elif ext == ".dat":
            values = read_astroaccelerate(candfile)
        else:
            raise CandiesError("Unsupported format for list of candy-dates.")

        m = readhdr(datafile)
        ff = open(datafile, "rb")
        mm = mmap.mmap(
            ff.fileno(),
            0,
            mmap.MAP_PRIVATE,
            mmap.PROT_READ,
        )
        fh = m["fch1"]
        dt = m["tsamp"]
        nf = m["nchans"]
        skip = m["size"]
        df = np.abs(m["foff"])

        nt = int(mm.size() - skip)
        tobs = nt * dt
        bw = nf * df
        fl = fh - bw + 0.5 * df

        return cls(
            items=[
                Candy.wrap(
                    nf=nf,
                    nt=nt,
                    df=df,
                    dt=dt,
                    bw=bw,
                    fh=fh,
                    fl=fl,
                    mm=mm,
                    tobs=tobs,
                    skip=skip,
                    **value,
                )
                for value in values
            ]
        )
