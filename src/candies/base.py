import re
from pathlib import Path
from autoregistry import Registry
from dataclasses import field, dataclass
from collections.abc import MutableSequence

import h5py as h5
import numpy as np
import pandas as pd
import ultraplot as uplt
from rich.table import Table
from ultraplot.axes import Axes
from rich.console import Console
from typing_extensions import Self
from priwo import readhdr, readfil, writefil


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
            attrs = dict(f.attrs.items())
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

    def plot(
        self,
        ax: Axes | None = None,
        show: bool = True,
        save: str | Path | bool = False,
        **kwargs,
    ):
        def plotter(ax):
            hm = ax.imshow(
                self.data,
                cmap="magma",
                aspect="auto",
                origin="lower",
                interpolation="none",
                vmin=self.data.min(),
                vmax=self.data.max(),
                extent=(self.times[0], self.times[-1], self.freqs[-1], self.freqs[0]),
            )
            ax.colorbar(hm)

        if ax is None:
            fig = getattr(uplt, "figure")(width=3.5, height=3.5)
            plotter(fig.subplot())
            if show:
                getattr(uplt, "show")()
            if save:
                if not isinstance(save, str | Path):
                    save = "dedispersed.png"
                fig.savefig(save, dpi=kwargs.get("dpi", 150))
        else:
            plotter(ax)


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
            attrs = dict(f.attrs.items())
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

    def plot(
        self,
        ax: Axes | None = None,
        show: bool = True,
        save: str | Path | bool = False,
        **kwargs,
    ):
        def plotter(ax):
            hm = ax.imshow(
                self.data,
                cmap="magma",
                aspect="auto",
                origin="lower",
                interpolation="none",
                vmin=self.data.min(),
                vmax=self.data.max(),
                extent=(self.times[0], self.times[-1], self.dms[-1], self.dms[0]),
            )
            ax.colorbar(hm)

        if ax is None:
            fig = getattr(uplt, "figure")(width=3.5, height=3.5)
            plotter(fig.subplot())
            if show:
                getattr(uplt, "show")()
            if save:
                if not isinstance(save, str | Path):
                    save = "dmtransform.png"
                fig.savefig(save, dpi=kwargs.get("dpi", 150))
        else:
            plotter(ax)


@dataclass
class Slice:
    data: np.ndarray
    fn: Path

    nf: int
    nt: int
    df: float
    dt: float
    fh: float
    fl: float
    nbits: int
    tbeg: float
    tend: float
    extras: dict

    @classmethod
    def load(cls, fn: str | Path) -> Self:
        fn = Path(fn)
        match fn.suffix:
            case ".h5":
                with h5.File(fn, "r") as f:
                    extras = dict(f.attrs.items())
                    data = np.asarray(f["data"])
                    return cls(
                        fn=fn,
                        data=data,
                        extras=extras,
                        nf=int(extras["nf"]),
                        nt=int(extras["nt"]),
                        df=float(extras["df"]),
                        dt=float(extras["dt"]),
                        fh=float(extras["fh"]),
                        fl=float(extras["fl"]),
                        nbits=int(extras["nt"]),
                        tbeg=float(extras["tbeg"]),
                        tend=float(extras["tend"]),
                    )
            case ".fil":
                meta, data = readfil(fn)
                nf, nt = data.shape

                fh = meta["fch1"]
                df = meta["foff"]
                dt = meta["tsamp"]
                nbits = meta["nbits"]

                if df < 0:
                    df = abs(df)
                    bw = nf * df
                    fl = fh - bw + (0.5 * df)
                else:
                    fl = fh
                    bw = nf * df
                    fh = fl + bw - (0.5 * df)

                return cls(
                    fn=fn,
                    nf=nf,
                    nt=nt,
                    fh=fh,
                    dt=dt,
                    df=df,
                    tbeg=0.0,
                    data=data,
                    extras=meta,
                    nbits=nbits,
                    tend=nt * dt,
                    fl=fh - bw + (0.5 * df),
                )
            case _:
                raise CandiesError("INVALID DATA FORMAT. ABORT.")

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

    def store(self, fn: str | Path | None = None) -> None:
        fn = Path(fn if fn is not None else self.fn)
        match fn.suffix:
            case ".h5":
                with h5.File(fn, "w") as f:
                    f.attrs["nf"] = self.nf
                    f.attrs["nt"] = self.nt
                    f.attrs["df"] = self.df
                    f.attrs["dt"] = self.dt
                    f.attrs["fh"] = self.fh
                    f.attrs["tbeg"] = self.tbeg
                    f.attrs["tend"] = self.tend
                    f.attrs["nbits"] = self.nbits
                    for key, value in self.extras.items():
                        f.attrs[key] = value
                    dataset = f.create_dataset(
                        "data",
                        data=self.data,
                        compression="gzip",
                        compression_opts=9,
                        dtype=self.data.dtype,
                    )
                    dataset.dims[1].label = b"time"
                    dataset.dims[0].label = b"frequency"
            case ".fil":
                hdr = {
                    "nifs": 1,
                    "data_type": 1,
                    "fch1": self.fh,
                    "foff": -self.df,
                    "tsamp": self.dt,
                    "nchans": self.nf,
                    "nbits": self.nbits,
                    "rawdatafile": str(fn),
                    "tstart": self.extras.get("mjd", 0.0),
                    "src_raj": self.extras.get("src_raj", 0.0),
                    "src_dej": self.extras.get("src_dej", 0.0),
                    "machine_id": self.extras.get("machine_id", 0),
                    "barycentric": self.extras.get("barycentric", 0),
                    "telescope_id": self.extras.get("telescope_id", 0),
                    "pulsarcentric": self.extras.get("pulsarcentric", 0),
                    "source_name": self.extras.get("source_name", "UNKNOWN"),
                }
                if (mjd := self.extras.get("mjd", None)) is not None:
                    hdr["tstart"] = mjd
                if (ra := self.extras.get("ra", None)) is not None:
                    hdr["src_raj"] = float("".join(re.split(r"[hms]", ra)[:-1]))
                if (dec := self.extras.get("ra", None)) is not None:
                    hdr["src_dej"] = float("".join(re.split(r"[dms]", dec)[:-1]))
                writefil(hdr, self.data, fn)
            case _:
                raise CandiesError("INVALID DATA FORMAT. ABORT.")

    def plot(
        self,
        ax: Axes | None = None,
        show: bool = True,
        save: str | Path | bool = False,
        **kwargs,
    ):
        def plotter(ax):
            hm = ax.imshow(
                self.data,
                cmap="magma",
                aspect="auto",
                origin="lower",
                vmin=self.data.min(),
                vmax=self.data.max(),
            )
            ax.colorbar(hm)

        if ax is None:
            fig = getattr(uplt, "figure")(width=3.5, height=3.5)
            plotter(fig.subplot())
            if show:
                getattr(uplt, "show")()
            if save:
                if not isinstance(save, str | Path):
                    save = self.fn.with_suffix(".png")
                fig.savefig(save, dpi=kwargs.get("dpi", 150))
        else:
            plotter(ax)


@dataclass
class Candy:

    dm: float
    t0: float
    wbin: int
    snr: float

    beam: int = 0
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
        with h5.File(fn, "r") as f:
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

    def plot(
        self,
        show: bool = True,
        save: str | Path | bool = False,
        **kwargs,
    ):
        if (self.dedispersed is not None) and (self.dmtransform is not None):
            fig = getattr(uplt, "figure")(width=7.5, height=5, sharey=False)
            gs = uplt.GridSpec(nrows=2, ncols=3)
            axtab = fig.subplot(gs[:, -1])
            axtop = fig.subplot(gs[0, :-1])
            axbtm = fig.subplot(gs[1, :-1])
            self.dedispersed.plot(ax=axtop)
            self.dmtransform.plot(ax=axbtm)

            cells = {}
            if len(hdr := self.extras) > 0:
                cells["Source name"] = [str(hdr.get("source", "NA"))]
                cells["Right ascension, RA (J2000)"] = [
                    str(next((hdr[_] for _ in ["raj2000", "ra"] if _ in hdr), "NA"))
                ]
                cells["Declination, DEC (J2000)"] = [
                    str(next((hdr[_] for _ in ["decj2000", "dec"] if _ in hdr), "NA"))
                ]
            cells[r"$t_{cand}$"] = [f"{self.t0:.2f} s"]
            cells["DM"] = [f"{self.dm:.2f} pc cm$^{{-3}}$"]
            cells["SNR"] = [f"{self.snr:.2f}"]
            cells[r"$W_{bin}$"] = [f"{self.wbin:d} bins"]
            cells[r"$N_{t}$ (original)"] = [
                f"{self.dedispersed.nt * (1 if self.wbin < 3 else int(self.wbin / 2)):d}"
            ]
            cells[r"$N_{t}$ (downsampled)"] = [f"{self.dedispersed.nt:d}"]
            if len(hdr := self.extras) > 0:
                nforig = next((hdr[_] for _ in ["nf", "nchans"] if _ in hdr), "NA")
                cells[r"$N_{\nu}$ (original)"] = [
                    f"{nforig:d}" if isinstance(nforig, int) else nforig
                ]
            cells[r"$N_{\nu}$ (downsampled)"] = [f"{self.dedispersed.nf:d}"]
            if len(hdr := self.extras) > 0:
                dtorig = next((hdr[_] for _ in ["dt", "tsamp"] if _ in hdr), "NA")
                cells[r"$\delta t$ (original)"] = [
                    (
                        rf"{dtorig * 1e6:.2f} $\mu$s"
                        if isinstance(dtorig, float)
                        else dtorig
                    )
                ]
            cells[r"$\delta t$ (downsampled)"] = [
                rf"{self.dedispersed.dt * 1e6:.2f} $\mu$s"
            ]
            if len(hdr := self.extras) > 0:
                dforig = next(
                    (hdr[_] for _ in ["df", "foff", "chanwidth"] if _ in hdr),
                    "NA",
                )
                cells[r"$\delta \nu$ (original)"] = [
                    f"{dforig * 1e3:.2f} kHz" if isinstance(dforig, float) else dforig
                ]
            cells[r"$\delta \nu$ (downsampled)"] = [
                f"{self.dedispersed.df * 1e3:.2f} kHz"
            ]
            cells[r"$\nu_{first}$"] = [f"{self.dedispersed.fh:.2f} MHz"]
            cells[r"$\nu_{last}$"] = [f"{self.dedispersed.fl:.2f} MHz"]
            cells[r"$N_{DM}$"] = [f"{self.dmtransform.ndms:d}"]
            cells[r"$\delta$DM"] = [f"{self.dmtransform.ddm:.2f} pc cm$^{{-3}}$"]
            cells[r"$DM_{low}$"] = [f"{self.dmtransform.lodm:.2f} pc cm$^{{-3}}$"]
            cells[r"$DM_{high}$"] = [f"{self.dmtransform.hidm:.2f} pc cm$^{{-3}}$"]

            axtab.axis("off")
            table = axtab.table(
                loc="center",
                edges="closed",
                cellLoc="center",
                rowLabels=list(cells.keys()),
                cellText=list(cells.values()),
            )
            table.auto_set_font_size(False)

            if show:
                getattr(uplt, "show")()
            if save:
                if not isinstance(save, str | Path):
                    save = f"{self.id}.png"
                fig.savefig(save, dpi=kwargs.get("dpi", 150))
            uplt.close(fig)


readers = Registry(prefix="read")


@readers
def readyour(fn: str | Path) -> list[Candy]:
    return [
        Candy(
            dm=float(row["dm"]),
            snr=float(row["snr"]),
            t0=float(row["stime"]),
            wbin=int(row["width"]),
            extras={"datafile": str(row["file"])},
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
    items = []
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
    ).iterrows():
        t0 = float(row["stime"])
        try:
            hdr = readhdr(row["file"])
            t0 = t0 - hdr["mjd"]
            wbin = int(float(row["width"]) / readhdr(row["file"])["dt"])
            items.append(
                Candy(
                    t0=t0,
                    wbin=wbin,
                    dm=float(row["dm"]),
                    snr=float(row["snr"]),
                    extras={"datafile": str(row["file"])},
                )
            )
        except Exception:
            pass
    if len(items) > 0:
        return items
    else:
        raise CandiesError("DATAFILE NOT FOUND. ABORT.")


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
    ).to_csv(fn, index=False)


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
    def load(cls, fn: str | Path | list) -> Self:
        if isinstance(fn, str | Path):
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
        elif isinstance(fn, list):
            if isinstance(fn[0], str | Path):
                return cls(items=[Candy.load(_) for _ in fn])
        raise CandiesError(f"INVALID FILEPATH: {fn}. ABORT.")

    def save(self, fn: str | Path | None = None) -> None:
        return writers[
            (
                {
                    ".h5": "h5",
                    ".csv": "your",
                    ".dat": "astroacc",
                    ".json": "transientx",
                    ".singlepulse": "presto",
                }[Path(fn).suffix]
                if fn is not None
                else "h5"
            )
        ](self.items, fn)

    def show(self):
        console = Console()
        table = Table(expand=False, padding=(0, 2, 0, 2))
        for i, item in enumerate(self.items):
            datafile = item.extras.get("datafile", None)
            if i == 0:
                if datafile is not None:
                    table.add_column("File")
                table.add_column("DM (in pc cm^-3)")
                table.add_column("Arrival time (in s)")
                table.add_column("SNR")
                table.add_column("Width (in bins)")
            fields = [] if datafile is None else [datafile]
            fields.extend(
                [
                    f"{item.dm:.2f}",
                    f"{item.t0:.2f}",
                    f"{item.snr:.2f}",
                    f"{item.wbin:d}",
                ]
            )
            table.add_row(*fields)
        console.print(table)


__all__ = [
    "Slice",
    "Candy",
    "Candies",
    "Dedispersed",
    "DMTransform",
    "CandiesError",
]
