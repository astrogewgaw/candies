"""
The base code for Candies.
"""

import h5py as h5
import numpy as np
import pandas as pd
import proplot as pplt
from typing import Self
from pathlib import Path
from dataclasses import dataclass
from collections.abc import MutableSequence


class CandiesError(Exception):
    """
    Represents any error in Candies.
    """

    pass


@dataclass
class Dedispersed:
    """
    Represents a dedispersed dynamic spectrum.

    The dynamic spectrum has frequency channels along the y-axis and
    time samples along the x-axis. As is convention, the first channel
    corresponds to the highest frequency in the band. This is one of the
    features used by the classifier.

    Parameters
    ----------
    nt: int
        The number of time samples.
    nf: int
        The number of frequency channels.
    df: float
        The channel width (in MHz).
    dt: float
        The sampling time (in seconds).
    fl: float
        The lowest frequency (in MHz).
    fh: float
        The highest frequency (in MHz).
    dm: float
        The dispersion measure at which the data was dedispersed at.
    data: numpy.ndarray
        The dedispersed dynamic spectrum as a 2D Numpy array.
    """

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
        """
        Load a dedispersed dynamic spectrum from a HDF5 file.

        Parameters
        ----------
        fname: str or Path
            Path to the HDF5 file.
        """
        with h5.File(fname, "r") as f:
            return cls(
                nt=f.attrs["nt"],  # type: ignore
                nf=f.attrs["nf"],  # type: ignore
                df=f.attrs["df"],  # type: ignore
                dt=f.attrs["dt"],  # type: ignore
                fl=f.attrs["fl"],  # type: ignore
                fh=f.attrs["fh"],  # type: ignore
                dm=f.attrs["dm"],  # type: ignore
                data=np.asarray(f["data_freq_time"]).T,
            )

    @property
    def freqs(self) -> np.ndarray:
        """
        The frequency (in MHz) of each channel.
        """
        return np.linspace(self.fh, self.fl, self.nf)

    @property
    def times(self) -> np.ndarray:
        """
        The time (in ms) of each sample, calculated w.r.t. to the arrival time of the burst.
        """
        ntmid = self.nt // 2
        deltat = ntmid * self.dt * 1e3
        return np.linspace(-deltat, +deltat, self.nt)

    @property
    def profile(self) -> np.ndarray:
        """
        The dedispersed profile of the burst.
        """
        return self.data.sum(0)

    def plot(
        self,
        save: bool = True,
        show: bool = False,
        ax: pplt.Axes | None = None,
        saveto: str | Path = "dedispersed.png",
    ):
        """
        Plot a dedispersed dynamic spectrum.

        Parameters
        ----------
        save: bool, optional
            Save the plot to a PNG file. True by default.
        show: bool, optional
            Show the plot in a separate window. False by default.
        ax: pplt.Axes, optional
            The axis, in case we are plotting this as part of another plot. Default is None.
        saveto: str or Path
            Path where to save the plot. Default is "dedispersed.png" in the current working directory.
        """

        def _plot(ax: pplt.Axes) -> None:
            px = ax.panel_axes("top", width="5em", space=0)
            px.plot(self.times, self.profile, lw=0.5, cycle="batlow")
            px.set_yticks([])

            ax.format(xlabel=r"$\Delta t$ (in ms)", ylabel="Frequency (in MHz)")
            ax.pcolormesh(self.times, self.freqs, self.data, cmap="batlow")
            ax.invert_yaxis()

        if ax is None:
            fig = pplt.figure()
            ax = fig.subplot()  # type: ignore
            _plot(ax)  # type: ignore
            if save:
                fig.savefig(saveto)
            if show:
                pplt.show()
            pplt.close(fig)
        else:
            _plot(ax)

    def save(self, fname: str | Path) -> None:
        """
        Save a dedispersed dynamic spectrum to a HDF5 file.

        Parameters
        ----------
        fname: str or Path
            Path to save the HDF5 file to.
        """
        with h5.File(fname, "a") as f:
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
    """
    Represents a DM transform.

    A DM transform is obtained by dedispersing a dynamic spectrum across
    a range of DMs. The value at the center of this range corresponds to
    the "correct" DM value. The time series obtained are then stacked, and
    we obtain a 2D array. This array is the DM transform. This is one of the
    features used by the classifier. We expect to see a bowtie-like structure
    in the DM-time plane for an actual FRB.

    Parameters
    ----------
    nt: int
        The number of time samples.
    ndms: int
        The number of DMs.
    dm: float
        The dispersion measure at the center of the DM range.
    dt: float
        The sampling time (in seconds).
    ddm: float
        The DM step used (in pc cm^-3).
    dmlow: float
        The lowest DM value (in pc cm^-3).
    dmhigh: float
        The highest DM value (in pc cm^-3).
    data: numpy.ndarray
        The DM transform as a 2D Numpy array.
    """

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
        """
        Load a Dm transform from a HDF5 file.

        Parameters
        ----------
        fname: str or Path
            Path to the HDF5 file.
        """
        with h5.File(fname, "r") as f:
            return cls(
                nt=f.attrs["nt"],  # type: ignore
                dm=f.attrs["dm"],  # type: ignore
                dt=f.attrs["dt"],  # type: ignore
                ddm=f.attrs["ddm"],  # type: ignore
                ndms=f.attrs["ndms"],  # type: ignore
                dmlow=f.attrs["dmlow"],  # type: ignore
                dmhigh=f.attrs["dmhigh"],  # type: ignore
                data=np.asarray(f["data_dm_time"]),
            )

    @property
    def dms(self) -> np.ndarray:
        """
        The DMs used to obtain the DM transform.
        """
        return np.linspace(self.dmlow, self.dmhigh, self.ndms)

    @property
    def times(self) -> np.ndarray:
        """
        The time (in ms) of each sample, calculated w.r.t. to the arrival time of the burst.
        """
        ntmid = self.nt // 2
        deltat = ntmid * self.dt * 1e3
        return np.linspace(-deltat, +deltat, self.nt)

    def plot(
        self,
        save: bool = True,
        show: bool = False,
        ax: pplt.Axes | None = None,
        saveto: str = "dmtransform.png",
    ):
        """
        Plot a dedispersed dynamic spectrum.

        Parameters
        ----------
        save: bool, optional
            Save the plot to a PNG file. True by default.
        show: bool, optional
            Show the plot in a separate window. False by default.
        ax: pplt.Axes, optional
            The axis, in case we are plotting this as part of another plot. Default is None.
        saveto: str or Path
            Path where to save the plot. Default is "dmtransform.png" in the current working directory.
        """

        def _plot(ax: pplt.Axes):
            ax.format(xlabel=r"$\Delta t$ (in ms)", ylabel=r"DM (in pc cm$^{-3}$)")
            ax.pcolormesh(self.times, self.dms, self.data, cmap="batlow")
            ax.invert_yaxis()

        if ax is None:
            fig = pplt.figure()
            ax = fig.subplot()  # type: ignore
            _plot(ax)  # type: ignore
            if save:
                fig.savefig(saveto)
            if show:
                pplt.show()
            pplt.close(fig)
        else:
            _plot(ax)

    def save(self, fname: str | Path) -> None:
        """
        Save a DM transform to a HDF5 file.

        Parameters
        ----------
        fname: str or Path
            Path to save the HDF5 file to.
        """
        with h5.File(fname, "a") as f:
            f.attrs["nt"] = self.nt
            f.attrs["dt"] = self.dt
            f.attrs["dm"] = self.dm
            f.attrs["ddm"] = self.ddm
            f.attrs["ndms"] = self.ndms
            f.attrs["dmlow"] = self.dmlow
            f.attrs["dmhigh"] = self.dmhigh
            dmtset = f.create_dataset(
                "data_dm_time",
                data=self.data,
                compression="gzip",
                compression_opts=9,
            )
            dmtset.dims[0].label = b"dm"
            dmtset.dims[1].label = b"time"


@dataclass
class Candidate:
    """
    Represents a FRB candidate.

    Parameters
    ----------
    dm: float
        The dispersion masure (DM) of the candidate (in pc cm^-3).
    t0: float
        The arrival time of the candidate (in seconds).
    wbin: int
        The width of the candidate (in terms of number of bins/samples).
    snr: float
        The signal-to-noise ratio (SNR) of the candidate.
    extras: dict, optional
        Any extra metadata about this candidate. Typically includes metadata
        obtained from the file from which the candidate's data was obtained.
        Including this is optional; by default, it is None.
    dedispersed: np.ndarray, optional
        The dedispersed dynamic spectrum. It is None by default, since it can
        only be obtained after a candidate has been processed. Otherwise it is
        a 2D Numpy array.
    dmtransform: np.ndarray, optional
        The DM transform. It is None by default, since it can only be obtained
        after a candidate has been processed. Otherwise it is a 2D Numpy array.
    """

    dm: float
    t0: float
    wbin: int
    snr: float
    fname: str | None = None
    extras: dict | None = None
    dedispersed: Dedispersed | None = None
    dmtransform: DMTransform | None = None

    @classmethod
    def load(cls, fname: str | Path) -> Self:
        """
        Load a candidate from a HDF5 file.

        Parameters
        ----------
        fname: str or Path
            Path to the HDF5 file.
        """
        with h5.File(fname, "r") as f:
            self = cls(
                dm=f.attrs["dm"],  # type: ignore
                t0=f.attrs["t0"],  # type: ignore
                snr=f.attrs["snr"],  # type: ignore
                wbin=f.attrs["wbin"],  # type: ignore
                extras=dict(f["extras"].attrs),
            )
        self.dmtransform = DMTransform.load(fname)
        self.dedispersed = Dedispersed.load(fname)
        return self

    def plot(
        self,
        save: bool = True,
        show: bool = False,
        saveto: str = "candidate.png",
    ):
        """
        Plot a candidate.

        Parameters
        ----------
        save: bool, optional
            Save the plot to a PNG file. True by default.
        show: bool, optional
            Show the plot in a separate window. False by default.
        saveto: str or Path
            Path where to save the plot. Default is "candidate.png" in the current working directory.
        """
        if (self.dedispersed is not None) and (self.dmtransform is not None):
            fig = pplt.figure(width=7.5, height=5, sharey=False)
            gs = pplt.GridSpec(nrows=2, ncols=3)
            axtab = fig.subplot(gs[:, -1])  # type: ignore
            axtop = fig.subplot(gs[0, :-1])  # type: ignore
            axbtm = fig.subplot(gs[1, :-1])  # type: ignore
            self.dedispersed.plot(ax=axtop)
            self.dmtransform.plot(ax=axbtm)

            labels = [
                r"$t_{cand}$",
                "DM",
                "SNR",
                r"$W_{bin}$",
                r"$N_{t}$",
                r"$N_{\nu}$",
                r"$\delta t$",
                r"$\delta \nu$",
                r"$\nu_{first}$",
                r"$\nu_{last}$",
                r"$N_{DM}$",
                r"$\delta$DM",
                r"$DM_{low}$",
                r"$DM_{high}$",
            ]

            fields = [
                [f"{self.t0:.2f} s"],
                [f"{self.dm:.2f} pc cm$^{{-3}}$"],
                [f"{self.snr:.2f}"],
                [f"{self.wbin:d} bins"],
                [f"{self.dedispersed.nt:d}"],
                [f"{self.dedispersed.nf:d}"],
                [rf"{self.dedispersed.dt * 1e6:.2f} $\mu$s"],
                [f"{self.dedispersed.df * 1e3:.2f} kHz"],
                [f"{self.dedispersed.fh:.2f} MHz"],
                [f"{self.dedispersed.fl:.2f} MHz"],
                [f"{self.dmtransform.ndms:d}"],
                [f"{self.dmtransform.ddm:.2f} pc cm$^{{-3}}$"],
                [f"{self.dmtransform.dmlow:.2f} pc cm$^{{-3}}$"],
                [f"{self.dmtransform.dmhigh:.2f} pc cm$^{{-3}}$"],
            ]

            def sigproc_ra(ra: float):
                """
                Convert SIGPROC-style RA to an actual RA.
                """
                hh = int(ra / 1e4)
                mm = int((ra - hh * 1e4) / 1e2)
                ss = ra - hh * 1e4 - mm * 1e2
                return hh, mm, ss

            def sigproc_dec(dec: float):
                """
                Convert SIGPROC-style DEC to an actual DEC.
                """
                dd = int(dec / 1e4)
                mm = int((dec - dd * 1e4) / 1e2)
                ss = dec - dd * 1e4 - mm * 1e2
                return dd, mm, ss

            if self.extras is not None:
                name = self.extras["source_name"]
                ra = ":".join(map(str, sigproc_ra(self.extras["src_raj"])))
                dec = ":".join(map(str, sigproc_dec(self.extras["src_dej"])))

                labels.insert(0, "Source name")
                labels.insert(1, "Right ascension, RA (J2000)")
                labels.insert(2, "Declination, DEC (J2000)")

                fields.insert(0, [f"{name:s}"])
                fields.insert(1, [f"{ra:s}"])
                fields.insert(2, [f"{dec:s}"])

                title = self.extras.get("rawdatafile", "")
                fig.format(suptitle=f"{title}")  # type: ignore

            axtab.axis("off")
            table = axtab.table(
                loc="center",
                edges="closed",
                cellText=fields,
                rowLabels=labels,
                cellLoc="center",
            )
            table.auto_set_font_size(False)

            if save:
                fig.savefig(saveto)
            if show:
                pplt.show()
            pplt.close(fig)

    def save(self, fname: str | Path) -> None:
        """
        Save a candidate to a HDF5 file.

        Parameters
        ----------
        fname: str or Path
            Path to save the HDF5 file to.
        """
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
    """
    Represents a list of candidates.

    Parameters
    ----------
    candidates: list[Candidate]
        A list of candidates.
    """

    candidates: list[Candidate]

    def __len__(self):
        return len(self.candidates)

    def __getitem__(self, i):
        return self.candidates[i]

    def __delitem__(self, i):
        del self.candidates[i]

    def __setitem__(self, i, value):
        self.candidates[i] = value

    def insert(self, index, value: Candidate):
        """
        Insert a candidate into the list at a particular index.

        Parameters
        ----------
        index: int
            The index at which to insert the candidate.
        value: Candidate
            The candidate to insert.
        """
        self.candidates.insert(index, value)

    @classmethod
    def fromcsv(cls, fname: str | Path) -> Self:
        """
        Get a list of candidates from a CSV file.

        Parameters
        ----------
        fname:str | Path
            Path to the CSV file.
        """
        df = pd.read_csv(fname)
        return cls(
            candidates=[
                Candidate(
                    **(
                        {
                            "dm": float(row["dm"]),
                            "snr": float(row["snr"]),
                            "wbin": int(row["width"]),
                            "t0": float(row["stime"]),
                            "fname": str(row["file"]),
                        }
                    )
                )
                for _, row in df.iterrows()
            ]
        )

    def tocsv(self, fname: str | Path) -> None:
        """
        Save a list of candidates to a CSV file.

        Parameters
        ----------
        fname:str | Path
            Path to the CSV file.
        """
        pd.DataFrame(
            [
                (
                    {
                        "file": pd.NA,
                        "snr": candidate.snr,
                        "stime": candidate.t0,
                        "width": candidate.wbin,
                        "dm": candidate.dm,
                        "label": 0,
                        "chan_mask_path": pd.NA,
                        "num_files": 1,
                    }
                )
                for candidate in self.candidates
            ]
        ).to_csv(fname)
