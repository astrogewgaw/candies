import os
import csv
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
    ntcut: int
    tobs: float
    tcut: float
    data: np.ndarray

    @classmethod
    def from_sigproc(
        cls,
        dm: float,
        width: float,
        tarrival: float,
        fname: str | Path,
    ) -> Self:
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

        with open(fname, "rb") as fptr:
            fptr.seek(0, os.SEEK_END)
            size = fptr.tell() - nskip
            nt = int(size / nf)
            tobs = nt * dt
            fptr.seek(0)

            maxdelay = dm2delay(fl, fh, dm)
            ntbeg = int((tarrival - width - maxdelay) / dt)
            ntend = int((tarrival + width + maxdelay) / dt)
            ntcut = ntend - (0 if ntbeg < 0 else 0)
            ntoff = 0 if ntbeg < 0 else ntbeg
            noff = ntoff * nf + nskip
            nchunk = ntcut * nf
            tcut = nchunk * dt

            chunked = np.ascontiguousarray(
                np.pad(
                    np.fromfile(
                        fptr,
                        offset=noff,
                        count=nchunk,
                        dtype=np.uint8,
                    )
                    .reshape(-1, nf)
                    .T,
                    (
                        (0, 0),
                        (
                            -ntbeg if ntbeg < 0 else 0,
                            ntend - nt if ntend > nt else 0,
                        ),
                    ),
                    mode="median",
                )
            )
        return cls(
            nf,
            nt,
            df,
            dt,
            bw,
            fh,
            fl,
            mjd,
            tobs,
            tcut,
            ntcut,
            chunked,
        )

    def save(self, fname: str):
        with h5.File(f"{fname}.h5", "a") as f:
            f.attrs["nf"] = self.nf
            f.attrs["dt"] = self.dt
            f.attrs["df"] = self.df
            f.attrs["fl"] = self.fl
            f.attrs["fh"] = self.fh
            f.attrs["mjd"] = self.mjd
            f.attrs["tobs"] = self.tobs
            f.attrs["tcut"] = self.tcut
            f.attrs["ntcut"] = self.ntcut
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
    snr: float
    width: float
    tarrival: float
    filterbank: Filterbank | None = None
    dedispersed: Filterbank | None = None
    dmtransform: DMTransform | None = None

    @classmethod
    def wrap(
        cls,
        dm: float,
        snr: float,
        width: float,
        tarrival: float,
    ) -> Self:
        return cls(
            dm,
            snr,
            width,
            tarrival,
        )

    @classmethod
    def load(cls, fname: str | Path) -> Self:
        with h5.File(fname, "r") as f:
            meta = {
                key: np.asarray(value)[0]
                for (
                    key,
                    value,
                ) in dict(f.attrs).items()
            }
            return cls(
                dm=meta["dm"],
                snr=meta["snr"],
                width=meta["width"],
                tarrival=meta["tarrival"],
                dmtransform=DMTransform(
                    nt=meta["nt"],
                    ndms=meta["ndms"],
                    dmlow=meta["dmlow"],
                    dmhigh=meta["dmhigh"],
                    data=np.asarray(f["dmtransform"]),
                ),
                dedispersed=Filterbank(
                    nf=meta["nf"],
                    fh=meta["fh"],
                    fl=meta["fl"],
                    df=meta["df"],
                    nt=meta["nt"],
                    dt=meta["dt"],
                    bw=meta["bw"],
                    mjd=meta["mjd"],
                    tobs=meta["tobs"],
                    tcut=meta["tcut"],
                    ntcut=meta["ntcut"],
                    data=np.asarray(f["dedispersed"]),
                ),
            )

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

                nf = self.filterbank.nf
                nt = self.filterbank.nt
                df = self.filterbank.df
                dt = self.filterbank.dt
                fh = self.filterbank.fh
                fl = self.filterbank.fl
                mjd = self.filterbank.mjd
                data = self.filterbank.data
                tcut = self.filterbank.tcut

                ffactor = np.floor(nf // 256).astype(int)
                tfactor = 1 if self.width < 3 else self.width // 2
                nfdown = int(nf / ffactor)
                ntdown = int(nt / tfactor)

                filongpu = cuda.to_device(data, **params)
                ddongpu = cuda.device_array((nfdown, ntdown), order="C", **params)
                dmtongpu = cuda.device_array((ndms, ntdown), order="C", **params)

                if zoom:
                    dmext = zoomby * delay2dm(
                        fl,
                        fh,
                        nt * dt,
                    )
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
                self.dedispersed = Filterbank(
                    nf=nfdown,
                    nt=ntdown,
                    fh=fh,
                    fl=fl,
                    mjd=mjd,
                    bw=fh - fl,
                    tobs=tcut,
                    tcut=tcut,
                    ntcut=ntdown,
                    data=dedispersed,
                    dt=tcut / ntdown,
                    df=(fh - fl) / nfdown,
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
            f.attrs["width"] = self.width
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
                            "width": int(row[3]),
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
