import h5py as h5
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

from attrs import define
from pathlib import Path
from priwo.sigproc.hdr import readhdr


def delay(dm, fl, fh):
    return 4.1488064239e3 * dm * (fl**-2 - fh**-2)


@define
class Candy:

    """
    Represents a candidate.
    """

    nf: int
    nt: int
    fl: float
    fh: float
    df: float
    dt: float
    dm: float
    t0: float
    ndms: int
    label: int
    snr: float
    wbin: float
    device: int
    fn: str | Path
    data: np.ndarray
    dmt: np.ndarray | None = None

    @classmethod
    def get(
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

        df = np.abs(df)
        fl = fh - (df * nf)
        ti = t0 - delay(dm, fl, fh) - (wbin * dt)
        tf = t0 + delay(dm, fl, fh) + (wbin * dt)

        nt = int((tf - ti) / dt)
        with open(fn, "rb") as f:
            f.seek(hdr["size"])
            data = (
                np.fromfile(
                    f,
                    dtype=np.uint8,
                    count=(nt * nf),
                    offset=(int(ti / dt) - 1) * nf,
                )
                .reshape(nt, nf)
                .T
            )

        return cls(
            fn=fn,
            dm=dm,
            nt=nt,
            t0=t0,
            fh=fh,
            fl=fl,
            df=df,
            dt=dt,
            nf=nf,
            snr=snr,
            wbin=wbin,
            ndms=ndms,
            data=data,
            label=label,
            device=(0 if device is None else device),
        )

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
                dbin
                if (dbin := np.round(delay(dm, f, self.fh) / self.dt)) < self.nt
                else 0
                for f in self.freqs
            ]
        )

    def allshifts(self) -> np.ndarray:
        """
        Bin shifts for all channels and all DMs from 0.0 to twice this candy-date's DM.
        """
        # TODO: We plan use a narrower DM range in the future, in order to make the DM
        # v/s time bowties more prominent at lower frequencies, such as those used at
        # the GMRT. We need to discuss how this range will be calculated, based on the
        # frequency band of the observed data.
        return np.vstack(
            [self.shifts(dm) for dm in np.linspace(0.0, 2 * self.dm, self.ndms)]
        )

    def calcdmt(self) -> None:
        """
        Create and store the DM v/s time array.
        """
        # TODO: Might need to decimate along the time axis.
        # TODO: Need to crop the array to 256 bins along the time axis.
        ft = cp.asarray(self.data)
        dmtx = cp.zeros((self.ndms, self.nt))
        allshifts = cp.asarray(self.allshifts())
        for ix in range(self.ndms):
            for jx, shift in enumerate(allshifts[ix, :]):
                dmtx[ix, :] += cp.pad(ft[jx, 0 + shift :], (0, int(shift)), mode="mean")
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


# TODO: Code to read in candidates from a CSV file.
