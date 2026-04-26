from pathlib import Path
from typing import Literal

import cyclopts
import matplotlib
from rich.progress import track
from rich.console import Console

from candies.classify import classify
from candies.featurize import featurize
from candies.interfaces import FileInterface
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
    datafile: str | Path | None = None,
    storeas: Literal["fil", "h5"] = "fil",
    interface: Literal["splt", "gmrt", "sigproc", "spotlight"] = "sigproc",
):
    candies = Candies.load(candidates)

    df = candies.pandas
    if datafile is not None:
        df["fn"] = df["fn"].fillna(str(datafile))

    made = []
    with console.status("Featurizing..."):
        for fn, group in df.groupby("fn"):
            try:
                made.extend(
                    featurize(
                        zoom=zoom,
                        njobs=njobs,
                        gpuid=gpuid,
                        store=store,
                        candies=Candies.load(group),
                        interface=FileInterface[interface].load(fn=fn),
                    )
                )
            except KeyError:
                raise CandiesError("INVALID INTERFACE. ABORT.")
    for candy in track(Candies(items=made), description="Saving...", transient=True):
        candy.save()
        if store:
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
            candies=candies,
            batchsize=batchsize,
        )
    for candy in track(candies, description="Saving...", transient=True):
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
        df["fn"] = df["fn"].fillna(str(datafile))

    wrapped = []
    with console.status("Featurizing and classifying..."):
        for fn, group in df.groupby("fn"):
            try:
                wrapped.extend(
                    classify(
                        gpuid=gpuid,
                        modelid=model,
                        batchsize=batchsize,
                        candies=featurize(
                            zoom=zoom,
                            njobs=njobs,
                            gpuid=gpuid,
                            store=store,
                            candies=Candies.load(group),
                            interface=FileInterface[interface].load(fn=fn),
                        ),
                    )
                )
            except KeyError:
                raise CandiesError("INVALID INTERFACE. ABORT.")
    for candy in track(Candies(items=wrapped), description="Saving...", transient=True):
        candy.save()
        if store:
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
    if datafile is not None:
        df["fn"] = df["fn"].fillna(str(datafile))

    for fn, group in df.groupby("fn"):
        try:
            for candy in track(
                transient=True,
                description="Storing...",
                sequence=Candies.load(group),
            ):
                candy = FileInterface[interface].load(fn=fn).slice(candy)
                candy.sliced.store(f"{candy.id}.highres.{storeas}")
        except KeyError:
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
        transient=True,
        description="Plotting...",
        sequence=Candies.load(candidates),
    ):
        candy.plot(
            dpi=dpi,
            show=show,
            save=Path(saveto) / f"{candy.id}.png" if save else False,
        )


__all__ = ["app"]

if __name__ == "__main__":
    app()
