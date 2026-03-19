import mmap
import numpy as np
from pathlib import Path

from priwo import readhdr
from typing_extensions import Self

from candies.base import Candy
from dataclasses import dataclass
from candies.functions import dm2delay
from candies.interfaces.base import FileInterface


@dataclass
class SIGPROCFile(FileInterface):

    nskip: int

    @classmethod
    def load(cls, fn: str | Path) -> Self:
        fn = Path(fn)
        hdr = readhdr(fn)
        fh = hdr["fch1"]
        df = hdr["foff"]
        dt = hdr["tsamp"]
        nf = hdr["nchans"]
        nskip = hdr["size"]
        nbits = hdr["nbits"]
        nt = int(int(fn.stat().st_size - nskip) / nf)

        if df < 0:
            df = abs(df)
            bw = nf * df
            fl = fh - bw + (0.5 * df)
        else:
            fl = fh
            bw = nf * df
            fh = fl + bw - (0.5 * df)

        return cls(
            fn=fn,
            nf=nf,
            nt=nt,
            df=df,
            dt=dt,
            fh=fh,
            extras=hdr,
            nbits=nbits,
            nskip=nskip,
        )

    def slice(self, candy: Candy) -> tuple[float, float, np.ndarray]:
        width = candy.wbin * self.dt
        maxdelay = dm2delay(f=self.fl, f0=self.fh, dm=candy.dm)
        tbeg, tend = candy.t0 - maxdelay - width, candy.t0 + maxdelay + width
        dtype = {8: np.uint8, 16: np.uint16, 32: np.float32, 64: np.float64}[self.nbits]

        Nbeg, Nend = int(tbeg / self.dt), int(tend / self.dt)

        N0 = Noff = Nbeg
        NR = Ncount = Nend - Nbeg
        if (candy.wbin > 2) and (NR // (candy.wbin // 2) < 256):
            NR = 256 * candy.wbin // 2
        if NR < 256:
            NR = 256
        N0 = Noff - (NR - Ncount) // 2

        with open(self.fn, mode="rb") as f:
            mm = mmap.mmap(
                length=0,
                fileno=f.fileno(),
                prot=mmap.PROT_READ,
                flags=mmap.MAP_PRIVATE,
            )
            # CASE 1: If the data we need is firmly within the file.
            # This means no padding, so we just get the data.
            if (N0 >= 0) and (N0 + NR) <= self.nt:
                data = (
                    np.frombuffer(
                        mm,
                        dtype=dtype,
                        count=NR * self.nf,
                        offset=N0 * self.nf + self.nskip,
                    )
                    .reshape(-1, self.nf)
                    .T
                )
            # CASE 2: If there is not enough data in the beginning.
            # In this case, we need to pad (with the median of each
            # channel). However, we still need to check...
            elif N0 < 0:
                # CASE 2A: ...if there is enough data for the end.
                # Note that the number of bins required from the
                # beginning = N0 is negative, so NR + N0 < NR, and
                # we pad by -N0 bins.
                if (N0 + NR) <= self.nt:
                    tempdata = (
                        np.frombuffer(
                            mm,
                            dtype=dtype,
                            offset=self.nskip,
                            count=(NR + N0) * self.nf,
                        )
                        .reshape(-1, self.nf)
                        .T
                    )
                    medians = np.median(tempdata, axis=1)
                    data = (
                        np.ones_like(tempdata, shape=(self.nf, NR)) * medians[:, None]
                    )
                    data[:, -N0:] = tempdata
                # CASE 2B: ...if there is not enough data for the end.
                # Here we need to pad both the ways.
                else:
                    tempdata = (
                        np.frombuffer(
                            mm,
                            dtype=dtype,
                            offset=self.nskip,
                            count=self.nt * self.nf,
                        )
                        .reshape(-1, self.nf)
                        .T
                    )
                    medians = np.median(tempdata, axis=1)
                    data = np.ones_like((self.nf, NR)) * medians[:, None]
                    data[:, -N0 : -N0 + self.nt] = tempdata
            # CASE 3: If there is enough data in the beginning, but
            # not enough in the end. In this case, we just need to pad
            # there.
            else:
                tempdata = (
                    np.frombuffer(
                        mm,
                        dtype=dtype,
                        count=(self.nt - N0) * self.nf,
                        offset=N0 * self.nf + self.nskip,
                    )
                    .reshape(-1, self.nf)
                    .T
                )
                medians = np.median(tempdata, axis=1)
                data = np.ones_like(tempdata, shape=(self.nf, NR)) * medians[:, None]
                data[:, : self.nt - N0] = tempdata
            # Calculate the correct reference beginning and end times.
            tbeg = N0 * self.dt
            tend = tbeg + (NR * self.dt)
        return tbeg, tend, data
