import csv
import h5py as h5
import numpy as np
from typing import Self
from pathlib import Path
from dataclasses import dataclass
from collections.abc import MutableSequence


@dataclass
class Dedispersed:
    nt: int
    nf: int
    df: float
    dt: float
    fl: float
    fh: float
    dm: float
    data: np.ndarray

    @classmethod
    def load(cls, fname: str | Path) -> Self:
        with h5.File(fname, "r") as f:
            return cls(
                nt=f.attrs["nt"],  # type: ignore
                nf=f.attrs["nf"],  # type: ignore
                df=f.attrs["df"],  # type: ignore
                dt=f.attrs["dt"],  # type: ignore
                fl=f.attrs["fl"],  # type: ignore
                fh=f.attrs["fh"],  # type: ignore
                dm=f.attrs["dm"],  # type: ignore
                data=np.asarray(f["dedispersed"]),
            )

    def plot(self, save: bool = True, show: bool = False):
        pass

    def save(self, fname: str | Path) -> None:
        with h5.File(fname, "a") as f:
            f.attrs["nf"] = self.nf
            f.attrs["nt"] = self.nt
            f.attrs["dt"] = self.dt
            f.attrs["df"] = self.df
            f.attrs["fl"] = self.fl
            f.attrs["fh"] = self.fh
            f.attrs["dm"] = self.dm
            dataset = f.create_dataset(
                "dedispersed",
                data=self.data,
                compression="gzip",
                compression_opts=9,
            )
            dataset.dims[1].label = b"time"
            dataset.dims[0].label = b"frequency"


@dataclass
class DMTransform:
    nt: int
    ndms: int
    dm: float
    dt: float
    ddm: float
    dmlow: float
    dmhigh: float
    data: np.ndarray

    @classmethod
    def load(cls, fname: str | Path) -> Self:
        with h5.File(fname, "r") as f:
            return cls(
                nt=f.attrs["nt"],  # type: ignore
                dm=f.attrs["dm"],  # type: ignore
                dt=f.attrs["dt"],  # type: ignore
                ddm=f.attrs["ddm"],  # type: ignore
                ndms=f.attrs["ndms"],  # type: ignore
                dmlow=f.attrs["dmlow"],  # type: ignore
                dmhigh=f.attrs["dmhigh"],  # type: ignore
                data=np.asarray(f["dedispersed"]),
            )

    def plot(self, save: bool = True, show: bool = False):
        pass

    def save(self, fname: str | Path) -> None:
        with h5.File(f"{fname}.h5", "a") as f:
            f.attrs["nt"] = self.nt
            f.attrs["dt"] = self.dt
            f.attrs["dm"] = self.dm
            f.attrs["ddm"] = self.ddm
            f.attrs["ndms"] = self.ndms
            f.attrs["dmlow"] = self.dmlow
            f.attrs["dmhigh"] = self.dmhigh
            dmtset = f.create_dataset(
                "dmtransform",
                data=self.data,
                compression="gzip",
                compression_opts=9,
            )
            dmtset.dims[0].label = b"dm"
            dmtset.dims[1].label = b"time"


@dataclass
class Candidate:
    dm: float
    t0: float
    wbin: int
    snr: float
    dedispersed: Dedispersed | None = None
    dmtransform: DMTransform | None = None

    @classmethod
    def load(cls, fname: str | Path) -> Self:
        with h5.File(fname, "r") as f:
            self = cls(
                dm=f.attrs["dm"],  # type: ignore
                t0=f.attrs["t0"],  # type: ignore
                snr=f.attrs["snr"],  # type: ignore
                wbin=f.attrs["wbin"],  # type: ignore
            )
        self.dmtransform = DMTransform.load(fname)
        self.dedispersed = Dedispersed.load(fname)
        return self

    def plot(self, save: bool = True, show: bool = False):
        pass

    def save(self, fname: str | Path) -> None:
        with h5.File(fname, "w") as f:
            f.attrs["dm"] = self.dm
            f.attrs["t0"] = self.t0
            f.attrs["snr"] = self.snr
            f.attrs["wbin"] = self.wbin
            if self.dedispersed is not None:
                self.dedispersed.save(fname)
            if self.dmtransform is not None:
                self.dmtransform.save(fname)


@dataclass
class CandidateList(MutableSequence):
    candidates: list[Candidate]

    def __len__(self):
        return len(self.candidates)

    def __getitem__(self, i):
        return self.candidates[i]

    def __delitem__(self, i):
        del self.candidates[i]

    def __setitem__(self, i, value):
        self.candidates[i] = value

    def insert(self, index, value):
        self.candidates.insert(index, value)

    @classmethod
    def fromcsv(cls, fname: str | Path) -> Self:
        with open(fname, "r") as f:
            return cls(
                candidates=[
                    Candidate(
                        **{
                            "t0": float(row[2]),
                            "dm": float(row[4]),
                            "wbin": int(row[3]),
                            "snr": float(row[1]),
                        }
                    )
                    for row in list(csv.reader(f))[1:]
                ]
            )
