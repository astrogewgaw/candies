import re
import mmap
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import astropy.units as uzi
from typing_extensions import Self

from candies.base import Candy, Slice
from candies.interfaces.base import FileInterface


@dataclass
class GMRTFile(FileInterface):

    @classmethod
    def load(cls, fn: str | Path) -> Self:
        hdr = {}
        fn = Path(fn)
        with open(fn.with_suffix(".hdr"), "r") as f:
            for line in (line.strip() for line in f.readlines()):
                if line.startswith("#"):
                    continue
                key, val = re.split(r"\s+:\s+", line)
                key = key.strip()
                val = val.strip()
                try:
                    name, conv = {
                        "Site": ("site", str),
                        "Observer": ("observer", str),
                        "Proposal": ("proposal", str),
                        "Array Mode": ("beammode", str),
                        "Observing Mode": ("obsmode", str),
                        "Date": ("istdate", str),
                        "Num Antennas": ("nant", int),
                        "Antenna List": ("antlist", str),
                        "Num Channels": ("nf", int),
                        "Channel width": ("df", float),
                        "Frequency Ch.1": ("f0", float),
                        "Num bits/sample": ("nbits", int),
                        "MJD": ("mjdi", float),
                        "UTC": ("isttime", str),
                        "Source": ("source", str),
                        "Coordinates": ("coords", str),
                        "Obs. Length": ("obslen", int),
                        "Bad Channels": ("badchans", str),
                        "Bit shift value": ("bitshift", int),
                        "Sampling Time": ("dt", lambda x: float(x) * 1e-6),
                        "Polarizations": ("npol", lambda x: 1 if x == "Total I" else 4),
                        "Drift Rate": (
                            "driftrates",
                            lambda x: [float(_.strip()) for _ in x.split(",")],
                        ),
                    }[key]
                    hdr[name] = conv(val)
                except KeyError:
                    pass

        hdr["istdatetime"] = " ".join([hdr["istdate"], hdr["isttime"][:-3]])

        ra, dec = hdr["coords"].split(",")
        rah, ram, ras = ra.strip().split(":")
        decd, decm, decs = dec.strip().split(":")
        ra = hdr["ra"] = hdr["raj2000"] = f"{rah}h{ram}m{ras}s"
        dec = hdr["dec"] = hdr["decj2000"] = f"{decd}d{decm}m{decs}s"
        hdr["coords"] = f"{ra} {dec}"

        if hdr["df"] < 0:
            hdr["fh"] = hdr["f0"]
            hdr["df"] = abs(hdr["df"])
            hdr["bw"] = hdr["nf"] * hdr["df"]
            hdr["fl"] = hdr["fh"] - hdr["bw"] + (0.5 * hdr["df"])
        else:
            hdr["fl"] = hdr["f0"]
            hdr["bw"] = hdr["nf"] * hdr["df"]
            hdr["fh"] = hdr["fl"] + hdr["bw"] - (0.5 * hdr["df"])

        hdr["datafile"] = str(fn)
        return cls(
            fn=fn,
            extras=hdr,
            nf=hdr["nf"],
            nt=hdr["nt"],
            df=hdr["df"],
            dt=hdr["dt"],
            fh=hdr["fh"],
            nbits=hdr["nbits"],
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
                        offset=N0 * self.nf,
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
                            offset=0,
                            dtype=dtype,
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
                            offset=0,
                            dtype=dtype,
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
                        offset=N0 * self.nf,
                        count=(self.nt - N0) * self.nf,
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


__all__ = ["GMRTFile"]
