"""
I/O interfaces for Candies.
"""

import mmap
import numpy as np
from typing import Self
from pathlib import Path
from priwo import readhdr
from dataclasses import dataclass
from candies.base import CandiesError


@dataclass
class Filterbank:
    """
    Context manager to read data from a SIGPROC filterbank file.
    """

    filepath: str | Path

    def __enter__(self) -> Self:
        self.header = readhdr(self.filepath)

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
        Get data from a SIGPROC filterbank file.
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
