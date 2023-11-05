import mmap
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
    pass


@define
class Candy:
    t0: float
    dm: float
    wbin: int
    snr: float
    dyn: np.ndarray | None = None
    dmt: np.ndarray | None = None

    @classmethod
    def wrap(
        cls,
        t0: float,
        dm: float,
        snr: float,
        wbin: int | float,
    ) -> Self:
        return cls(t0=t0, dm=dm, snr=snr, wbin=int(wbin))

    @property
    def id(self) -> str:
        return f"candy_t0{self.t0:.7f}_dm{self.dm:.5f}_snr{self.snr:.5f}"


@define
class Candies(MutableSequence):
    items: list[Candy]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]

    def __delitem__(self, i):
        del self.items[i]

    def __setitem__(self, i, value):
        self.items[i] = value

    def insert(self, i, value):
        self.items.insert(i, value)

    @classmethod
    def wrap(cls, f: str | Path) -> Self:
        ext = Path(f).suffix
        if ext == ".csv":
            values = read_csv(f)
        elif ext == ".singlepulse":
            values = read_presto(f)
        elif ext == ".dat":
            values = read_astroaccelerate(f)
        else:
            raise CandiesError("Unsupported format for list of candy-dates.")
        return cls(items=[Candy.wrap(**value) for value in values])


@define
class Filterbank:
    nf: int
    nt: int
    df: float
    dt: float
    bw: float
    fh: float
    fl: float
    tobs: float
    _skip: int
    _mm: mmap.mmap

    @classmethod
    def from_sigproc(cls, f: str | Path) -> Self:
        m = readhdr(f)
        ff = open(f, "rb")
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

        return cls(nf, nt, df, dt, bw, fh, fl, tobs, skip, mm)

    def chop(self, candy: Candy) -> np.ndarray:
        maxdd = dispersive_delay(self.fl, self.fh, candy.dm)
        ti = candy.t0 - maxdd - (candy.wbin * self.dt)
        tf = candy.t0 + maxdd + (candy.wbin * self.dt)
        return np.pad(
            np.frombuffer(
                self._mm,
                dtype=np.uint8,
                count=int((tf - (0.0 if ti < 0.0 else ti)) / self.dt) * self.nf,
                offset=int(0.0 if ti < 0.0 else ti / self.dt) * self.nf + self._skip,
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
