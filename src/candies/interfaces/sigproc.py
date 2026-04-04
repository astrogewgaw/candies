import mmap
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from priwo import readhdr
import astropy.units as uzi
from typing_extensions import Self

from candies.base import Candy, Slice
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

        hdr["datafile"] = str(fn)

        if (src := hdr.get("source_name", None)) is not None:
            hdr["source"] = src

        if (mjd := hdr.get("tstart", None)) is not None:
            hdr["mjd"] = mjd

        if (ra := hdr.get("src_raj", None)) is not None:
            hdr["ra"] = hdr["raj2000"] = (
                f"{(hh := int(ra // 10000)):02d}h"
                f"{(mm := int((ra - hh * 10000) // 100)):02d}m"
                f"{(ra - hh * 10000 - mm * 100):05.2f}s"
            )

        if (dec := hdr.get("src_dej", None)) is not None:
            hdr["dec"] = hdr["decj2000"] = (
                f"{'-' if dec < 0 else ''}"
                f"{(dd := int(abs(dec) // 10000)):02d}d"
                f"{(mm := int((abs(dec) - dd * 10000) // 100)):02d}m"
                f"{(abs(dec) - dd * 10000 - mm * 100):05.2f}s"
            )

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

    def slice(self, candy: Candy) -> Slice:
        width = candy.wbin * self.dt
        maxdelay = 4.1488064239e3 * candy.dm * (self.fl**-2 - self.fh**-2)
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
                    data = np.ones((self.nf, NR)) * medians[:, None]
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
            nf, nt = data.shape
            tbeg = N0 * self.dt
            tend = tbeg + (NR * self.dt)

        hdr = self.extras
        if (mjd := hdr.get("mjd", None)) is not None:
            hdr["begmjd"] = mjd + (tbeg * getattr(uzi, "s")).to("day").value
            hdr["endmjd"] = mjd + (tend * getattr(uzi, "s")).to("day").value
            hdr["mjd"] = mjd + (candy.t0 * getattr(uzi, "s")).to("day").value

        return Slice(
            nf=nf,
            nt=nt,
            tbeg=tbeg,
            tend=tend,
            extras=hdr,
            fh=self.fh,
            fl=self.fl,
            df=self.df,
            dt=self.dt,
            nbits=self.nbits,
            fn=Path(f"{candy.id}.h5"),
            data=np.ascontiguousarray(data),
        )
