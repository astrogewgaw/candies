"""
Generally applicable code for Candies.
"""

import mmap
import warnings
import h5py as h5
import numpy as np
import proplot as pplt
from numba import cuda
from typing import Any
from attrs import define
from pathlib import Path
from priwo import readhdr
from rich.progress import track
from typing_extensions import Self
from collections.abc import MutableSequence
from candies.kernels import fastdmt, dedisperse
from candies.utilities import dmt_extent, dispersive_delay
from candies.utilities import read_csv, read_presto, read_astroaccelerate


warnings.filterwarnings("ignore")


class CandiesError(Exception):
    """
    All errors are CandiesErrors here.
    """

    pass


@define
class Candy:
    """
    Represents a candy-date.
    """

    # Candy-date parameters.
    t0: float
    dm: float
    wbin: int
    snr: float

    # Candy-date's data parameters.
    nf: int
    nt: int
    df: float
    dt: float
    bw: float
    fh: float
    fl: float
    mjd: float
    tobs: float

    # Hidden internal parameters.
    skip: int | None = None
    mm: mmap.mmap | None = None

    # Optional parameters.
    ndms: int | None = None
    dmlow: float | None = None
    dmhigh: float | None = None
    dyn: np.ndarray | None = None
    dmt: np.ndarray | None = None

    @classmethod
    def wrap(
        cls,
        nf: int,
        nt: int,
        df: float,
        dt: float,
        bw: float,
        fh: float,
        fl: float,
        t0: float,
        dm: float,
        snr: float,
        mjd: float,
        tobs: float,
        wbin: int | float,
        skip: int | None = None,
        mm: mmap.mmap | None = None,
    ) -> Self:
        """
        Create and wrap up a candy-date.
        """
        return cls(
            nf=nf,
            nt=nt,
            df=df,
            dt=dt,
            bw=bw,
            fh=fh,
            fl=fl,
            t0=t0,
            dm=dm,
            mm=mm,
            snr=snr,
            mjd=mjd,
            tobs=tobs,
            skip=skip,
            wbin=int(wbin),
        )

    @classmethod
    def load(cls, fn: str | Path) -> Self:
        """
        Load a candy-date as an HDF5 dataset.
        """
        with h5.File(fn, "r") as f:
            t0 = f.attrs["t0"]
            nf = f.attrs["nf"]
            fh = f.attrs["fh"]
            fl = f.attrs["fl"]
            df = f.attrs["df"]
            nt = f.attrs["nt"]
            dt = f.attrs["dt"]
            dm = f.attrs["dm"]
            mjd = f.attrs["mjd"]
            snr = f.attrs["snr"]
            wbin = f.attrs["wbin"]
            ndms = f.attrs["ndms"]
            dmlow = f.attrs["dmlow"]
            dmhigh = f.attrs["dmhigh"]
            dmt = np.asarray(f["data_dm_time"])
            dyn = np.asarray(f["data_freq_time"])
        return cls(
            fl=fl,
            fh=fh,
            df=df,
            dt=dt,
            nt=nt,
            nf=nf,
            t0=t0,
            dm=dm,
            snr=snr,
            dmt=dmt,
            mjd=mjd,
            dyn=dyn,
            wbin=wbin,
            ndms=ndms,
            dmlow=dmlow,
            dmhigh=dmhigh,
            bw=df * nf,  # type: ignore
            tobs=nt * dt,  # type: ignore
        )

    @property
    def id(self) -> str:
        """
        The candy-date's ID.
        """
        return f"MJD{self.mjd:.7f}_T{self.t0:.7f}_DM{self.dm:.5f}_SNR{self.snr:.5f}"

    def chop(self) -> np.ndarray:
        """
        Chop out a time slice for a particular candy-date.

        This is done by taking the maximum delay at the candy-date's DM
        and width into account, and then chopping out a time slice around
        the candy-date's arrival time. In case the burst is close to the
        start or end of the file, and enough data is not available, this
        function will pad the array with median values.
        """
        if (self.mm is not None) and (self.skip is not None):
            maxdd = dispersive_delay(self.fl, self.fh, self.dm)
            ti = self.t0 - maxdd - (self.wbin * self.dt)
            tf = self.t0 + maxdd + (self.wbin * self.dt)
            return np.pad(
                np.frombuffer(
                    self.mm,
                    dtype=np.uint8,
                    count=int((tf - (0.0 if ti < 0.0 else ti)) / self.dt) * self.nf,
                    offset=int(0.0 if ti < 0.0 else ti / self.dt) * self.nf + self.skip,
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
        else:
            raise CandiesError("Data has not yet been read from file.")

    def process(
        self,
        ndms: int = 256,
        save: bool = True,
        zoom: bool = True,
        stream: Any | None = None,
    ):
        """
        Process a single candy-date.
        """
        cuda_params: dict = {}
        if stream is None:
            stream = cuda.stream()
            cuda_params = {"stream": stream}

        data = self.chop()
        _, self.nt = data.shape

        if zoom:
            _, _, dmlow, dmhigh = dmt_extent(
                self.fl,
                self.fh,
                self.dt,
                self.t0,
                self.dm,
                self.wbin,
            )
        else:
            dmlow, dmhigh = 0.0, 2.0 * self.dm

        self.ndms = ndms
        self.dmlow = dmlow
        self.dmhigh = dmhigh
        ddm = (dmhigh - dmlow) / (ndms - 1)

        tfactor = 1 if self.wbin < 3 else self.wbin // 2
        ffactor = np.floor(self.nf // 256).astype(int)
        nfdown = int(self.nf / ffactor)
        ntdown = int(self.nt / tfactor)

        ft = cuda.to_device(data, **cuda_params)
        dmtx = cuda.device_array((ndms, ntdown), order="C", **cuda_params)
        dynx = cuda.to_device(np.zeros((nfdown, ntdown), order="C"), **cuda_params)

        dedisperse[  # type: ignore
            (self.nf // 32, self.nt // 32),
            (32, 32),
            stream,
        ](
            dynx,
            ft,
            self.nf,
            self.nt,
            self.df,
            self.dt,
            self.fh,
            self.dm,
            ffactor,
            tfactor,
        )

        fastdmt[  # type: ignore
            self.nt,
            self.ndms,
            stream,
        ](
            dmtx,
            ft,
            self.nf,
            self.nt,
            self.df,
            self.dt,
            self.fh,
            ddm,
            dmlow,
            tfactor,
        )

        dyn = dynx.copy_to_host(**cuda_params)
        dmt = dmtx.copy_to_host(**cuda_params)

        ntmid = int(ntdown / 2)
        nticrop = ntmid - 128
        ntfcrop = ntmid + 128
        dmt = dmt[:, nticrop:ntfcrop]
        dyn = dyn[:, nticrop:ntfcrop]

        self.dyn = dyn
        self.dmt = dmt

        if save:
            self.save()

        if stream is None:
            cuda.close()

    def save(self):
        """
        Save a candy-date as an HDF5 dataset.
        """
        with h5.File(f"{self.id}.h5", "w") as f:
            f.attrs["nt"] = self.nt
            f.attrs["id"] = self.id

            if self.ndms is not None:
                f.attrs["ndms"] = self.ndms

            if self.dmlow is not None:
                f.attrs["dmlow"] = self.dmlow

            if self.dmhigh is not None:
                f.attrs["dmhigh"] = self.dmhigh

            f.attrs["nf"] = self.nf
            f.attrs["dt"] = self.dt
            f.attrs["df"] = self.df
            f.attrs["fl"] = self.fl
            f.attrs["fh"] = self.fh
            f.attrs["mjd"] = self.mjd

            f.attrs["t0"] = self.t0
            f.attrs["dm"] = self.dm
            f.attrs["snr"] = self.snr
            f.attrs["wbin"] = self.wbin

            if self.dyn is not None:
                ftset = f.create_dataset(
                    "data_freq_time",
                    data=self.dyn.T,
                    compression="gzip",
                    compression_opts=9,
                )
                ftset.dims[0].label = b"time"
                ftset.dims[1].label = b"frequency"

            if self.dmt is not None:
                dmtset = f.create_dataset(
                    "data_dm_time",
                    data=self.dmt,
                    compression="gzip",
                    compression_opts=9,
                )
                dmtset.dims[0].label = b"dm"
                dmtset.dims[1].label = b"time"

    def plot(
        self,
        save: bool = True,
        show: bool = False,
    ):
        """
        Plot a candy-date.
        """
        if (self.dyn is not None) and (self.dmt is not None):
            pplt.rc.update(
                {
                    "font.size": 20,
                    "suptitle.pad": 20,
                    "subplots.tight": True,
                    "subplots.refwidth": 6.5,
                    "subplots.panelpad": 1.5,
                }
            )

            ntmid = int(self.nt / 2)  # type: ignore
            nti = ntmid - 128
            ntf = ntmid + 128
            ts = self.dyn.sum(axis=0)
            f = np.linspace(self.fh, self.fl, self.dyn.shape[0])  # type: ignore
            dms = np.linspace(self.dmhigh, self.dmlow, self.ndms)  # type: ignore
            t = np.linspace(nti * self.dt, ntf * self.dt, self.dyn.shape[1])  # type: ignore

            fig, axs = pplt.subplots(
                nrows=2,
                ncols=1,
                width=10,
                height=15,
                sharey=False,
                sharex="labels",
            )
            px = axs[0].panel_axes("t", width="5em")

            px.plot(t, ts)
            axs[0].pcolormesh(*np.meshgrid(t, f), self.dyn)
            axs[1].pcolormesh(*np.meshgrid(t, dms), self.dmt)

            px.format(xlabel="Time (in s)", ylabel="Flux (arb. units)")
            axs[0].format(
                xlabel="Time (in s)",
                ylabel="Frequency (in MHz)",
                title="Dedispersed dynamic spectrum",
            )

            axs[1].format(
                xlabel="Time (in s)",
                ylabel="DM (in pc cm$^-3$)",
                title=f"DMT, from {self.dmlow:.2f} to {self.dmhigh:.2f} pc cm$^-3$",
            )

            if save:
                fig.savefig(f"{id}.png", dpi=300)
            if show:
                pplt.show()
        else:
            raise CandiesError("Candy-date has nto yet been processed.")


@define
class Candies(MutableSequence):
    """
    Represents a sequence of candy-dates.
    """

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
    def wrap(cls, candfile: str | Path, datafile: str | Path) -> Self:
        """
        Create and wrap up candy-dates.
        """
        ext = Path(candfile).suffix
        if ext == ".csv":
            values = read_csv(candfile)
        elif ext == ".singlepulse":
            values = read_presto(candfile)
        elif ext == ".dat":
            values = read_astroaccelerate(candfile)
        else:
            raise CandiesError("Unsupported format for list of candy-dates.")

        m = readhdr(datafile)
        ff = open(datafile, "rb")
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
        mjd = m["tstart"]
        df = np.abs(m["foff"])

        nt = int(mm.size() - skip)
        tobs = nt * dt
        bw = nf * df
        fl = fh - bw + 0.5 * df

        return cls(
            items=[
                Candy.wrap(
                    nf=nf,
                    nt=nt,
                    df=df,
                    dt=dt,
                    bw=bw,
                    fh=fh,
                    fl=fl,
                    mm=mm,
                    mjd=mjd,
                    tobs=tobs,
                    skip=skip,
                    **value,
                )
                for value in values
            ]
        )

    @classmethod
    def load(cls, fns: list[str | Path]) -> Self:
        """
        Load candy-dates from HDF5 datasets.
        """
        return cls(items=[Candy.load(fn) for fn in fns])

    def process(
        self,
        ndms: int = 256,
        device: int = 0,
        save: bool = True,
        zoom: bool = True,
    ):
        """
        Process candy-dates.
        """
        cuda.select_device(device)
        stream = cuda.stream()
        for item in track(self.items, description="Processing candy-dates..."):
            item.process(
                ndms,
                save,
                zoom,
                stream,
            )
        cuda.close()

    def save(self):
        """
        Save candy-dates as HDF5 datasets.
        """
        for item in track(self.items, description="Saving candy-dates..."):
            item.save()

    def plot(
        self,
        save: bool = True,
        show: bool = False,
    ):
        """
        Plot candy-dates.
        """
        for item in track(self.items, description="Plotting candy-dates..."):
            item.plot(save=save, show=show)
