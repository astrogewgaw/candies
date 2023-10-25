import h5py as h5
import cupy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numba import cuda
from attrs import define
from pathlib import Path
from functools import cached_property
from priwo.sigproc.hdr import readhdr
from collections.abc import MutableSequence


def delay(dm, fl, fh):
    return 4.1488064239e3 * dm * (fl**-2 - fh**-2)


@cuda.jit
def compute_dmt(dmtx, ft, nt, nf, df, dt, fh, ddm):
    idm = int(cuda.threadIdx.x)
    itime = int(cuda.blockIdx.x)

    temp = 0
    for ifreq in range(nf):
        f = fh - ifreq * df
        shift = int(round(4.1488064239e3 * (idm * ddm) * (f**-2 - fh**-2) / dt))
        if shift > nt:
            shift = 0
        index = shift + itime
        if index >= nt:
            index -= nt
        temp += ft[ifreq, index]
    dmtx[idm, itime] = temp


@define(slots=False)
class Candy:

    """
    Represents a candidate.
    """

    nf: int
    fl: float
    fh: float
    df: float
    dt: float
    dm: float
    t0: float
    ndms: int
    label: int
    hsize: int
    snr: float
    wbin: float
    device: int
    fn: str | Path
    nt: int | None = None
    dmt: np.ndarray | None = None

    @classmethod
    def make(
        cls,
        fn: str,
        dm: float,
        t0: float,
        snr: float,
        wbin: float,
        label: int,
        ndms: int = 256,
        device: int | None = None,
    ):
        """
        Makes a candy-date.
        """

        hdr = readhdr(fn)

        fh = hdr["fch1"]
        df = hdr["foff"]
        dt = hdr["tsamp"]
        nf = hdr["nchans"]
        size = hdr["size"]

        df = np.abs(df)
        fl = fh - (df * nf)

        return cls(
            fn=fn,
            dm=dm,
            t0=t0,
            fh=fh,
            fl=fl,
            dt=dt,
            nf=nf,
            df=df,
            snr=snr,
            wbin=wbin,
            ndms=ndms,
            hsize=size,
            label=label,
            device=(0 if device is None else device),
        )

    @cached_property
    def data(self) -> np.ndarray:
        """
        Get the chunk of data associated with this candidate.
        """

        ti = self.t0 - delay(self.dm, self.fl, self.fh) - (self.wbin * self.dt)
        tf = self.t0 + delay(self.dm, self.fl, self.fh) + (self.wbin * self.dt)

        self.nt = int((tf - ti) / self.dt)
        padding = int(-ti / self.dt) if ti < 0.0 else 0
        nr = int((tf - (0.0 if ti < 0.0 else ti)) / self.dt)
        with open(self.fn, "rb") as f:
            f.seek(self.hsize)
            data = (
                np.fromfile(
                    f,
                    dtype=np.uint8,
                    count=(nr * self.nf),
                    offset=int(0.0 if ti < 0.0 else ti / self.dt) * self.nf,
                )
                .reshape(nr, self.nf)
                .T
            )
            data = data if padding == 0 else np.pad(data, (padding, 0), mode="median")
        return data

    @property
    def id(self) -> str:
        """
        The ID of the candy-date.
        """
        return f"t0{self.t0:.7f}_dm{self.dm:.5f}_snr{self.snr:.5f}"

    @property
    def freqs(self) -> np.ndarray:
        """
        Frequencies for all channels, in MHz.
        """
        return np.linspace(self.fh, self.fl, self.nf)

    def shifts(self, dm: float) -> np.ndarray:
        """
        Bin shifts for all channels, for a specific DM.
        """
        return np.asarray(
            [
                int(dbin)
                if (dbin := np.round(delay(dm, f, self.fh) / self.dt)) < self.nt
                else 0
                for f in self.freqs
            ]
        )

    def allshifts(self) -> np.ndarray:
        """
        Bin shifts for all channels and all DMs from 0.0 to twice this candy-date's DM.
        """
        # TODO: We plan to use a narrower DM range in the future, in order to make the
        # DM v/s time bowties more prominent at lower frequencies (such as those used
        # at the GMRT). We need to discuss how this range will be calculated, based on
        # the frequency band of the observed data.
        return np.vstack(
            [self.shifts(dm) for dm in np.linspace(0.0, 2 * self.dm, self.ndms)]
        )

    def calcdmt(self) -> None:
        """
        Create and store the DM v/s time array.
        """
        # TODO: Probably needs heavy optimisation.
        # TODO: Might need to decimate along the time axis.
        # TODO: Need to crop the array to 256 bins along the time axis.
        with cp.cuda.Device(self.device):
            ft = cp.asarray(self.data)
            dmtx = cp.zeros((self.ndms, self.nt))

            blocks = self.nt
            threads = self.ndms
            compute_dmt[blocks, threads](
                dmtx,
                ft,
                self.nt,
                self.nf,
                (self.fh - self.fl) / (self.nf - 1),
                self.dt,
                self.fh,
                2 * self.dm / (self.ndms - 1),
            )

            self.dmt = dmtx.get()

    def plot(self) -> None:
        """
        Plot the DM v/s time array.
        """
        if self.dmt is not None:
            plt.xlabel("Time (in s)")
            plt.suptitle("DM v/s t Plot")
            plt.ylabel("DM (in pc cm$^-3$)")
            plt.imshow(self.dmt, aspect="auto")
            plt.savefig(f"candy.{self.id}.png", dpi=300)

    def save(self) -> None:
        """
        Save the candidate to an HDF5 file.
        """
        with h5.File(f"candy.{self.id}.h5", "w") as f:
            # Store the metadata first.
            f["id"] = self.id
            f["nt"] = self.nt
            f["nf"] = self.nf
            f["dt"] = self.dt
            f["df"] = self.df
            f["fl"] = self.fl
            f["fh"] = self.fh
            f["t0"] = self.t0
            f["dm"] = self.dm
            f["snr"] = self.snr
            f["wbin"] = self.wbin
            f["ndms"] = self.ndms
            f["label"] = self.label

            if self.dmt is not None:
                # Now, store the DM v/s time array.
                dset = f.create_dataset(
                    "dmt",
                    data=self.dmt,
                    compression="gzip",
                    compression_opts=9,
                )
                dset.dims[0].label = b"dm"
                dset.dims[1].label = b"time"


# TODO: Need to discuss and implement parallelisation across candidates.


@define
class Candies(MutableSequence):

    """
    Represents a list of candidates.
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

    def insert(self, i, value):
        self.items.insert(i, value)

    @classmethod
    def get(
        cls,
        fn: str,
        ndms: int = 256,
        device: int | None = None,
    ):
        df = pd.read_csv(fn)
        return cls(
            items=[
                Candy.make(
                    ndms=ndms,
                    device=device,
                    fn=str(row["file"]),
                    dm=float(row["dm"]),
                    snr=float(row["snr"]),
                    t0=float(row["stime"]),
                    wbin=int(row["width"]),
                    label=int(row["label"]),
                )
                for _, row in df.iterrows()
            ]
        )
