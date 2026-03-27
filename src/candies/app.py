from pathlib import Path
from typing import Literal
from functools import partial

import cyclopts
import matplotlib
import pandas as pd
from rich.progress import track

from candies.classify import classify
from candies.featurize import featurize
from candies.interfaces import Interface
from candies.base import Candies, CandiesError

app = cyclopts.App()
app["--help"].group = "Admin"
app["--version"].group = "Admin"


@app.command
def list_(
    candidates: list[str | Path],
    show: bool = True,
    save: str | Path | None = None,
):
    candies = Candies.load(candidates)
    if show:
        candies.show()
    if save is not None:
        candies.save(save)


@app.command
def make(
    candidates: str | Path,
    njobs: int = 1,
    gpuid: int = 0,
    zoom: bool = True,
    datafile: str | Path | None = None,
    interface: Literal["splt", "gmrt", "sigproc", "spotlight"] = "sigproc",
):
    candies = Candies.load(candidates)

    df = candies.pandas
    if datafile is not None:
        df = df.replace({pd.NA: str(datafile)})
    groups = df.groupby("fn")

    made = []
    for fn, group in groups:
        minis = Candies.load(group)
        if (x := Interface["file"].get(interface)) is not None:
            maker = partial(featurize, interface=x.load(fn=fn))
        elif (x := Interface["live"].get(interface)) is not None:
            maker = partial(featurize, interface=x.load())
        else:
            raise CandiesError("INVALID INTERFACE. ABORT.")
        minis = maker(zoom=zoom, njobs=njobs, gpuid=gpuid, candies=minis)
        made.extend(minis)

    for candy in track(Candies(items=made), description="Saving...", transient=True):
        candy.save()


@app.command
def label(
    candidates: list[Path],
    gpuid: int = 0,
    batchsize: int = 8,
    model: Literal["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"] = "a",
):
    candies = Candies.load(candidates)
    candies = classify(gpuid=gpuid, modelid=model, batchsize=batchsize, candies=candies)
    for candy in track(candies, description="Saving...", transient=True):
        candy.save()


@app.command
def wrap(
    candidates: str | Path,
    njobs: int = 1,
    gpuid: int = 0,
    zoom: bool = True,
    batchsize: int = 8,
    datafile: str | Path | None = None,
    interface: Literal["splt", "gmrt", "sigproc", "spotlight"] = "sigproc",
    model: Literal["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"] = "a",
):
    candies = Candies.load(candidates)

    df = candies.pandas
    if datafile is not None:
        df = df.replace({pd.NA: str(datafile)})
    groups = df.groupby("fn")

    wrapped = []
    for fn, group in groups:
        minis = Candies.load(group)
        if (x := Interface["file"].get(interface)) is not None:
            maker = partial(featurize, interface=x.load(fn=fn))
        elif (x := Interface["live"].get(interface)) is not None:
            maker = partial(featurize, interface=x.load())
        else:
            raise CandiesError("INVALID INTERFACE. ABORT.")
        minis = maker(zoom=zoom, njobs=njobs, gpuid=gpuid, candies=minis)
        minis = classify(gpuid=gpuid, modelid=model, batchsize=batchsize, candies=minis)
        wrapped.extend(minis)

    for candy in track(Candies(items=wrapped), description="Saving...", transient=True):
        candy.save()


@app.command
def store(
    candidates: str | Path,
    fmt: Literal["fil", "h5"] = "fil",
    datafile: str | Path | None = None,
    interface: Literal["splt", "gmrt", "sigproc", "spotlight"] = "sigproc",
):
    candies = Candies.load(candidates)

    df = candies.pandas
    groups = df.groupby("fn")
    if datafile is not None:
        df = df.replace({pd.NA: str(datafile)})

    for fn, group in groups:
        minis = Candies.load(group)
        if (x := Interface["file"].get(interface)) is not None:
            for mini in track(minis, description="Storing...", transient=True):
                x.load(fn=fn).slice(mini).store(f"{mini.id}.highres.{fmt}")
        elif (x := Interface["live"].get(interface)) is not None:
            for mini in track(minis, description="Storing...", transient=True):
                x.load().slice(mini).store(f"{mini.id}.highres.{fmt}")
        else:
            raise CandiesError("INVALID INTERFACE. ABORT.")


@app.command
def plot(
    candidates: list[Path],
    dpi: int = 96,
    save: bool = True,
    show: bool = False,
    saveto: str | Path = Path.cwd(),
):
    if not show:
        matplotlib.use("agg")
    for candy in track(
        Candies.load(candidates),
        description="Plotting...",
        transient=True,
    ):
        candy.plot(
            dpi=dpi,
            show=show,
            save=Path(saveto) / f"{candy.id}.png" if save else False,
        )


if __name__ == "__main__":
    app()
