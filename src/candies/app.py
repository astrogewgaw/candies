from pathlib import Path
from typing import Literal
from functools import partial

import cyclopts
import matplotlib
import pandas as pd
from rich.progress import track
from rich.console import Console

from candies.logging import log
from candies.classify import classify
from candies.featurize import featurize
from candies.interfaces import Interface
from candies.base import Candies, CandiesError

console = Console()

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
    gpuid: int = -1,
    zoom: bool = True,
    store: bool = False,
    storeas: Literal["fil", "h5"] = "fil",
    datafile: str | Path | None = None,
    interface: Literal["splt", "gmrt", "sigproc", "spotlight"] = "sigproc",
):
    candies = Candies.load(candidates)

    df = candies.pandas
    if datafile is not None:
        df = df.replace({pd.NA: str(datafile)})
    groups = df.groupby("fn")

    made = []
    with console.status("Featurizing..."):
        for fn, group in groups:
            minis = Candies.load(group)
            if (x := Interface["file"].get(interface)) is not None:
                maker = partial(featurize, interface=x.load(fn=fn))
            elif (x := Interface["live"].get(interface)) is not None:
                maker = partial(featurize, interface=x.load())
            else:
                raise CandiesError("INVALID INTERFACE. ABORT.")
            made.extend(
                maker(
                    zoom=zoom,
                    njobs=njobs,
                    gpuid=gpuid,
                    candies=minis,
                    store=store,
                )
            )
    for candy in track(Candies(items=made), description="Saving...", transient=True):
        candy.save()
        if store:
            log.debug(f"Saved {candy.id} to disk.")
            candy.sliced.store(f"{candy.id}.highres.{storeas}")


@app.command
def label(
    candidates: list[Path],
    gpuid: int = -1,
    batchsize: int = 8,
    model: Literal["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"] = "a",
):
    candies = Candies.load(candidates)
    with console.status("Classifying..."):
        candies = classify(
            gpuid=gpuid,
            modelid=model,
            batchsize=batchsize,
            candies=candies,
        )
    for candy in track(candies, description="Saving...", transient=True):
        log.debug(f"Saved {candy.id} to disk.")
        candy.save()


@app.command
def wrap(
    candidates: str | Path,
    njobs: int = 1,
    gpuid: int = -1,
    zoom: bool = True,
    batchsize: int = 8,
    store: bool = False,
    storeas: Literal["fil", "h5"] = "fil",
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
    with console.status("Featurizing and classifying..."):
        for fn, group in groups:
            minis = Candies.load(group)
            if (x := Interface["file"].get(interface)) is not None:
                maker = partial(featurize, interface=x.load(fn=fn))
            elif (x := Interface["live"].get(interface)) is not None:
                maker = partial(featurize, interface=x.load())
            else:
                raise CandiesError("INVALID INTERFACE. ABORT.")
            wrapped.extend(
                classify(
                    gpuid=gpuid,
                    modelid=model,
                    batchsize=batchsize,
                    candies=maker(
                        zoom=zoom,
                        njobs=njobs,
                        gpuid=gpuid,
                        candies=minis,
                        store=store,
                    ),
                )
            )
    for candy in track(Candies(items=wrapped), description="Saving...", transient=True):
        candy.save()
        if store:
            log.debug(f"Saved {candy.id} to disk.")
            candy.sliced.store(f"{candy.id}.highres.{storeas}")


@app.command
def store(
    candidates: str | Path,
    datafile: str | Path | None = None,
    storeas: Literal["fil", "h5"] = "fil",
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
                x.load(fn=fn).slice(mini).store(f"{mini.id}.highres.{storeas}")
        elif (x := Interface["live"].get(interface)) is not None:
            for mini in track(minis, description="Storing...", transient=True):
                x.load().slice(mini).store(f"{mini.id}.highres.{storeas}")
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


__all__ = []

if __name__ == "__main__":
    app()
