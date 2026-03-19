from pathlib import Path
from autoregistry import Registry
from typing_extensions import Self
from dataclasses import field, dataclass
from collections.abc import MutableSequence

import h5py as h5
import numpy as np
import pandas as pd
from priwo import readhdr


class CandiesError(Exception):
    pass


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
    def load(cls, fn: str | Path):
        with h5.File(fn, "r") as f:
            attrs = {k: np.asarray(v) for k, v in f.attrs.items()}
            return cls(
                nt=int(attrs["nt"]),
                nf=int(attrs["nf"]),
                df=float(attrs["df"]),
                dt=float(attrs["dt"]),
                fl=float(attrs["fl"]),
                fh=float(attrs["fh"]),
                dm=float(attrs["dm"]),
                data=np.asarray(f["data_freq_time"]).T,
            )

    @property
    def freqs(self) -> np.ndarray:
        return np.linspace(self.fh, self.fl, self.nf)

    @property
    def times(self) -> np.ndarray:
        ntmid = self.nt // 2
        deltat = ntmid * self.dt * 1e3
        return np.linspace(-deltat, +deltat, self.nt)

    @property
    def profile(self) -> np.ndarray:
        return self.data.sum(0)

    def save(self, fn: str | Path) -> None:
        with h5.File(fn, "a") as f:
            f.attrs["nf"] = self.nf
            f.attrs["nt"] = self.nt
            f.attrs["dt"] = self.dt
            f.attrs["df"] = self.df
            f.attrs["fl"] = self.fl
            f.attrs["fh"] = self.fh
            f.attrs["dm"] = self.dm
            dataset = f.create_dataset(
                "data_freq_time",
                data=self.data.T,
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
    lodm: float
    hidm: float
    data: np.ndarray

    @classmethod
    def load(cls, fn: str | Path):
        with h5.File(fn, "r") as f:
            attrs = {k: np.asarray(v) for k, v in f.attrs.items()}
            return cls(
                nt=int(attrs["nt"]),
                dm=float(attrs["dm"]),
                dt=float(attrs["dt"]),
                ddm=float(attrs["ddm"]),
                ndms=int(attrs["ndms"]),
                lodm=float(attrs["lodm"]),
                hidm=float(attrs["hidm"]),
                data=np.asarray(f["data_dm_time"]),
            )

    @property
    def dms(self) -> np.ndarray:
        return np.linspace(self.lodm, self.hidm, self.ndms)

    @property
    def times(self) -> np.ndarray:
        ntmid = self.nt // 2
        deltat = ntmid * self.dt * 1e3
        return np.linspace(-deltat, +deltat, self.nt)

    def save(self, fn: str | Path) -> None:
        with h5.File(fn, "a") as f:
            f.attrs["nt"] = self.nt
            f.attrs["dt"] = self.dt
            f.attrs["dm"] = self.dm
            f.attrs["ddm"] = self.ddm
            f.attrs["ndms"] = self.ndms
            f.attrs["lodm"] = self.lodm
            f.attrs["hidm"] = self.hidm
            dmtset = f.create_dataset(
                "data_dm_time",
                data=self.data,
                compression="gzip",
                compression_opts=9,
            )
            dmtset.dims[0].label = b"dm"
            dmtset.dims[1].label = b"time"


@dataclass
class Candy:

    dm: float
    t0: float
    wbin: int
    snr: float

    label: bool = False
    probability: float = 0.0
    dedispersed: Dedispersed | None = None
    dmtransform: DMTransform | None = None
    extras: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return "".join(
            [
                (
                    f"MJD{mjd:.7f}_"
                    if (
                        mjd := (
                            self.extras.get("tstart", None)
                            if self.extras is not None
                            else None
                        )
                    )
                    is not None
                    else ""
                ),
                f"T{self.t0:.7f}_",
                f"DM{self.dm:.5f}_",
                f"SNR{self.snr:.5f}",
            ]
        )

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return self.__str__()

    @classmethod
    def load(cls, fn: str | Path) -> Self:
        fn = Path(fn)
        with h5.File(fn) as f:
            attrs = dict(f.attrs.items())
        return cls(
            dm=float(attrs["dm"]),
            t0=float(attrs["t0"]),
            snr=float(attrs["snr"]),
            wbin=int(attrs["wbin"]),
            label=bool(attrs["label"]),
            dmtransform=DMTransform.load(fn=fn),
            dedispersed=Dedispersed.load(fn=fn),
            extras=dict(f["extras"].attrs.items()),
            probability=float(attrs["probability"]),
        )

    def save(self, fn: str | Path | None = None) -> None:
        fn = self.id + ".h5" if fn is None else fn
        with h5.File(fn, "w") as f:
            f.attrs["dm"] = self.dm
            f.attrs["t0"] = self.t0
            f.attrs["snr"] = self.snr
            f.attrs["wbin"] = self.wbin
            f.attrs["label"] = self.label
            f.attrs["probability"] = self.probability
            group = f.create_group("extras")
            for key, value in self.extras.items():
                group.attrs[key] = value
            if self.dedispersed is not None:
                self.dedispersed.save(fn=fn)
            if self.dmtransform is not None:
                self.dmtransform.save(fn=fn)


readers = Registry(prefix="read")


@readers
def readyour(fn: str | Path) -> list[Candy]:
    return [
        Candy(
            dm=float(row["dm"]),
            snr=float(row["snr"]),
            t0=float(row["stime"]),
            wbin=int(row["width"]),
        )
        for _, row in pd.read_csv(fn).iterrows()
    ]


@readers
def readpresto(fn: str | Path) -> list[Candy]:
    return [
        Candy(
            dm=float(row[0]),
            t0=float(row[2]),
            wbin=int(row[3]),
            snr=float(row[1]),
        )
        for row in np.loadtxt(fn, usecols=(0, 1, 2, 4))
    ]


@readers
def readastroacc(fn: str | Path) -> list[Candy]:
    return [
        Candy(
            dm=float(row[0]),
            t0=float(row[1]),
            wbin=int(row[3]),
            snr=float(row[2]),
        )
        for row in np.fromfile(fn, dtype=np.float32).reshape(-1, 4)
    ]


@readers
def readtransientx(fn: str | Path) -> list[Candy]:
    return [
        Candy(
            dm=float(row["dm"]),
            snr=float(row["snr"]),
            t0=(float(row["stime"]) - readhdr(row["meta"])["mjd"]),
            wbin=int(float(row["width"]) / readhdr(row["meta"])["dt"]),
        )
        for _, row in pd.read_csv(
            fn,
            sep="\t",
            names=[
                "beam",
                "id",
                "stime",
                "dm",
                "width",
                "snr",
                "fh",
                "fl",
                "png",
                "ddplanid",
                "file",
            ],
        ).iterrows()
    ]


writers = Registry(prefix="write")


@writers
def writeyour(items: list[Candy], fn: str | Path) -> None:
    return pd.DataFrame(
        [
            (
                {
                    "file": str(item.extras.get("datafile", "")),
                    "snr": item.snr,
                    "stime": item.t0,
                    "width": item.wbin,
                    "dm": item.dm,
                    "label": 0,
                    "chan_mask_path": pd.NA,
                    "num_files": 1,
                }
            )
            for item in items
        ]
    ).to_csv(fn)


@writers
def writeh5(items: list[Candy], fn: str | Path | None = None) -> None:
    for ix, candy in enumerate(items):
        candy.save(fn=None if fn is None else f"{Path(fn).with_suffix('')}{ix}.h5")


@dataclass
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

    def insert(self, index, value: Candy):
        self.items.insert(index, value)

    @classmethod
    def load(cls, fn: str | Path) -> Self:
        return cls(
            items=readers[
                (
                    {
                        ".csv": "your",
                        ".dat": "astroacc",
                        ".json": "transientx",
                        ".singlepulse": "presto",
                    }[Path(fn).suffix]
                )
            ](fn)
        )

    def save(self, fn: str | Path | None = None) -> None:
        return writers[
            (
                {
                    ".csv": "your",
                    ".dat": "astroacc",
                    ".json": "transientx",
                    ".singlepulse": "presto",
                }[Path(fn).suffix]
                if fn is not None
                else "h5"
            )
        ](self.items, fn)
