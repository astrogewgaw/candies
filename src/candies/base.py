import csv
import mmap
import random
import h5py as h5
import numpy as np
from numba import cuda
from pathlib import Path
from priwo import readhdr
from typing import Any, Self
from rich.progress import track
from dataclasses import dataclass
from collections.abc import MutableSequence

from candies.kernels import dedisperse, fastdmt
from candies.utilities import delay2dm, dm2delay, normalise


class CandiesError(Exception):
    pass


@dataclass
class Filterbank:
    nf: int
    nt: int
    df: float
    dt: float
    bw: float
    fh: float
    fl: float
    mjd: float
    tobs: float
    hdrsize: int
    mapped: mmap.mmap

    @classmethod
    def load(cls, fname: str | Path) -> Self:
        meta = readhdr(fname)

        fh = meta["fch1"]
        df = meta["foff"]
        dt = meta["tsamp"]
        nf = meta["nchans"]
        nskip = meta["size"]
        mjd = meta["tstart"]

        df = abs(df)
        bw = nf * df
        fl = fh - bw + 0.5 * df

        fptr = open(fname, "rb")
        mapped = mmap.mmap(fptr.fileno(), 0, mmap.MAP_PRIVATE, mmap.PROT_READ)
        nbytes = int(mapped.size - nskip)
        nt = nbytes // nf
        tobs = nt * dt

        return cls(
            nf=nf,
            nt=nt,
            df=df,
            dt=dt,
            bw=bw,
            fh=fh,
            fl=fl,
            mjd=mjd,
            tobs=tobs,
            hdrsize=nskip,
            mapped=mapped,
        )

    def chop(
        self,
        dm: float,
        wbin: int,
        tarrival: float,
    ) -> np.ndarray:
        maxdelay = dm2delay(self.fl, self.fh, dm)
        ntbeg = int((tarrival - maxdelay) / self.dt) - wbin
        ntend = int((tarrival + maxdelay) / self.dt) + wbin
        return np.ascontiguousarray(
            np.pad(
                np.frombuffer(
                    self.mapped,
                    dtype=np.uint8,
                    count=(ntend - (0 if ntbeg < 0 else ntbeg)) * self.nf,
                    offset=(0 if ntbeg < 0 else ntbeg) * self.nf + self.hdrsize,
                )
                .reshape(-1, self.nf)
                .T,
                (
                    (0, 0),
                    (
                        -ntbeg if ntbeg < 0 else 0,
                        ntend - self.nt if ntend > self.nt else 0,
                    ),
                ),
                mode="median",
            )
        )


@dataclass
class Dedispersed:
    nf: int
    nt: int
    df: float
    dt: float
    bw: float
    fh: float
    fl: float
    mjd: float
    tobs: float
    data: np.ndarray | None

    def save(self, fname: str):
        pass


@dataclass
class DMTransform:
    nt: int
    ndms: int
    dmlow: float
    dmhigh: float
    data: np.ndarray

    def save(self, fname: str) -> None:
        with h5.File(f"{fname}.h5", "a") as f:
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
class Candy:
    dm: float
    wbin: int
    snr: float
    tarrival: float
    filterbank: Filterbank | None = None
    dedispersed: Dedispersed | None = None
    dmtransform: DMTransform | None = None

    @classmethod
    def wrap(
        cls,
        dm: float,
        wbin: int,
        snr: float,
        tarrival: float,
    ) -> Self:
        return cls(
            dm=dm,
            snr=snr,
            wbin=wbin,
            tarrival=tarrival,
        )

    @classmethod
    def load(cls, fname: str | Path):
        pass

    @property
    def id(self) -> str:
        idstr = f"T{self.tarrival:.7f}_DM{self.dm:.5f}_SNR{self.snr:.5f}"
        if self.dedispersed is not None:
            idstr = "".join([f"MJD{self.dedispersed.mjd:.7f}", idstr])
        return idstr

    @property
    def isprocessed(self):
        if (self.dedispersed is not None) and (self.dmtransform is not None):
            return True
        else:
            return False

    def process(
        self,
        ndms: int = 256,
        save: bool = True,
        zoom: bool = True,
        zoomby: float = 16,
        stream: Any | None = None,
        gpudevice: int | None = None,
    ):
        if not self.isprocessed:
            if self.filterbank is not None:
                params: dict = {}
                if stream is None:
                    stream = cuda.stream()
                    params = {"stream": stream}
                if gpudevice is not None:
                    cuda.select_device(gpudevice)

                chunk = self.filterbank.chop(self.dm, self.wbin, self.tarrival)
                fh = self.filterbank.fl
                fl = self.filterbank.fh
                df = self.filterbank.df
                dt = self.filterbank.dt
                nf, nt = chunk.shape
                tchunk = nt * dt

                ffactor = np.floor(nf // 256).astype(int)
                tfactor = 1 if self.wbin < 3 else self.wbin // 2
                nfdown = int(nf / ffactor)
                ntdown = int(nt / tfactor)

                filongpu = cuda.to_device(chunk, **params)
                ddongpu = cuda.device_array((nfdown, ntdown), order="C", **params)
                dmtongpu = cuda.device_array((ndms, ntdown), order="C", **params)

                if zoom:
                    dmext = zoomby * delay2dm(fl, fh, nt * dt)
                    dmlow = self.dm - dmext
                    dmhigh = self.dm + dmext
                else:
                    dmlow, dmhigh = 0.0, 2.0 * self.dm
                ddm = (dmhigh - dmlow) / (ndms - 1)

                dedisperse[  # type: ignore
                    (nf // 32, nt // 32),
                    (32, 32),
                    stream,
                ](
                    ddongpu,
                    filongpu,
                    nf,
                    nt,
                    df,
                    dt,
                    fh,
                    self.dm,
                    ffactor,
                    tfactor,
                )

                fastdmt[  # type: ignore
                    nt,
                    ndms,
                    stream,
                ](
                    dmtongpu,
                    filongpu,
                    nf,
                    nt,
                    df,
                    dt,
                    fh,
                    ddm,
                    dmlow,
                    tfactor,
                )

                nt0 = int(ntdown / 2)
                nti = nt0 - 128
                ntf = nt0 + 128
                ntdown = ntf - nti

                dedispersed = ddongpu.copy_to_host(**params)
                dedispersed = dedispersed[:, nti:ntf]
                dedispersed = normalise(dedispersed)
                self.dedispersed = Dedispersed(
                    fh=fh,
                    fl=fl,
                    nf=nfdown,
                    nt=ntdown,
                    bw=fh - fl,
                    tobs=tchunk,
                    data=dedispersed,
                    dt=tchunk / ntdown,
                    df=(fh - fl) / nfdown,
                    mjd=self.filterbank.mjd,
                )

                dmtransform = dmtongpu.copy_to_host(**params)
                dmtransform = dmtransform[:, nti:ntf]
                dmtransform = normalise(dmtransform)
                self.dmtransform = DMTransform(
                    ndms=ndms,
                    nt=ntdown,
                    dmlow=dmlow,
                    dmhigh=dmhigh,
                    data=dmtransform,
                )
                if save:
                    self.save()
                if stream is None:
                    cuda.close()
            else:
                raise CandiesError("Data has not yet been read from file.")

    def save(self):
        with h5.File(f"{self.id}.h5", "w") as f:
            f.attrs["id"] = self.id
            f.attrs["dm"] = self.dm
            f.attrs["snr"] = self.snr
            f.attrs["wbin"] = self.wbin
            f.attrs["tarrival"] = self.tarrival
        if self.dedispersed is not None:
            self.dedispersed.save(self.id)
        if self.dmtransform is not None:
            self.dmtransform.save(self.id)

    def plot(self, save: bool = False):
        pass


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

    def insert(self, index, value):
        self.items.insert(index, value)

    @classmethod
    def wrap(cls, fname: str | Path) -> Self:
        with open(fname, "r") as fp:
            return cls(
                items=[
                    Candy(
                        **{
                            "dm": float(row[4]),
                            "wbin": int(row[3]),
                            "snr": float(row[1]),
                            "tarrival": float(row[2]),
                        }
                    )
                    for row in list(csv.reader(fp))[1:]
                ]
            )

    @classmethod
    def load(cls, fnames: list[str | Path]):
        return cls(items=[Candy.load(fname) for fname in fnames])

    def process(
        self,
        ndms: int = 256,
        save: bool = True,
        zoom: bool = True,
        zoomby: float = 16,
        stream: Any | None = None,
        gpudevice: int | None = None,
    ):
        cuda.select_device(gpudevice)
        random.shuffle(self.items)
        stream = cuda.stream()
        for item in track(self.items, description="Processing candy-dates..."):
            item.process(ndms, save, zoom, zoomby, stream)
        cuda.close()

    def save(self):
        for item in track(self.items, description="Saving candy-dates..."):
            item.save()

    def plot(self, save: bool = False):
        for item in track(self.items, description="Plotting candy-dates..."):
            item.plot(save=save)
