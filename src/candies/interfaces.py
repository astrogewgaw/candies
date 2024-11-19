"""
I/O interfaces for candies.
"""

import mmap
from typing import Self
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from priwo import readhdr, writefil

from candies.utilities import dm2delay
from candies.base import CandiesError, Candidate


@dataclass
class SIGPROCFilterbank:
    """
    Context manager to read data from a SIGPROC filterbank file.
    """

    filepath: str | Path

    def __enter__(self) -> Self:
        self.header = readhdr(str(self.filepath))

        try:
            self.fh = self.header["fch1"]
            self.df = self.header["foff"]
            self.dt = self.header["tsamp"]
            self.nf = self.header["nchans"]
            self.nskip = self.header["size"]
            self.nbits = self.header["nbits"]
        except KeyError:
            CandiesError("File may not be a valid SIGPROC filterbank file.")

        self.fobject = open(self.filepath, mode="rb")

        if self.df < 0:
            self.df = abs(self.df)
            self.bw = self.nf * self.df
            self.fl = self.fh - self.bw + (0.5 * self.df)
        else:
            self.fl = self.fh
            self.bw = self.nf * self.df
            self.fh = self.fl + self.bw - (0.5 * self.df)

        self.dtype = {
            8: np.uint8,
            16: np.uint16,
            32: np.float32,
            64: np.float64,
        }[self.nbits]

        self.mapped = mmap.mmap(
            self.fobject.fileno(),
            0,
            mmap.MAP_PRIVATE,
            mmap.PROT_READ,
        )

        self.nt = int(int(self.mapped.size() - self.nskip) / self.nf)

        return self

    def __exit__(self, *args) -> None:
        if self.fobject:
            self.fobject.close()

    def get(self, offset: int, count: int) -> np.ndarray:
        """
        Get data from this file.
        """

        data = np.frombuffer(
            self.mapped,
            dtype=self.dtype,
            count=count * self.nf,
            offset=offset * self.nf + self.nskip,
        )
        data = data.reshape(-1, self.nf)
        data = data.T
        return data

    def chop(self, candidate: Candidate) -> tuple[float, float, np.ndarray]:
        """
        Chops out a candy-date from this file.

        Parameters
        ----------
        candidate: Candidate
            The candy-date to chop.
        """

        maxdelay = dm2delay(self.fl, self.fh, candidate.dm)
        binbeg = int((candidate.t0 - maxdelay) / self.dt) - candidate.wbin
        binend = int((candidate.t0 + maxdelay) / self.dt) + candidate.wbin

        nbegin = noffset = binbeg
        nread = ncount = binend - binbeg
        if (candidate.wbin > 2) and (nread // (candidate.wbin // 2) < 256):
            nread = 256 * candidate.wbin // 2
        elif nread < 256:
            nread = 256
        nbegin = noffset - (nread - ncount) // 2

        if (nbegin >= 0) and (nbegin + nread) <= self.nt:
            data = self.get(offset=nbegin, count=nread)
        elif nbegin < 0:
            if (nbegin + nread) <= self.nt:
                d = self.get(offset=0, count=nread + nbegin)
                dmedian = np.median(d, axis=1)
                data = np.ones((self.nf, nread), dtype=self.dtype) * dmedian[:, None]
                data[:, -nbegin:] = d
            else:
                d = self.get(offset=0, count=self.nt)
                dmedian = np.median(d, axis=1)
                data = np.ones((self.nf, nread), dtype=self.dtype) * dmedian[:, None]
                data[:, -nbegin : -nbegin + self.nt] = d
        else:
            d = self.get(offset=nbegin, count=self.nt - nbegin)
            dmedian = np.median(d, axis=1)
            data = np.ones((self.nf, nread), dtype=self.dtype) * dmedian[:, None]
            data[:, : self.nt - nbegin] = d
        tbeg = nbegin * self.dt
        tend = tbeg + (nread * self.dt)
        return tbeg, tend, data

    def store(self, candidate: Candidate) -> None:
        """
        Store a candy-date from this file.

        Parameters
        ----------
        candidates: Candidate
            The candy-date to store.
        """
        meta = self.header.copy()
        meta["refdm"] = candidate.dm
        tbeg, _, data = self.chop(candidate)
        meta["tstart"] = meta["tstart"] + tbeg / 86400.0
        writefil(meta, data, "".join([str(candidate), ".fil"]))
